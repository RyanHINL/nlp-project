"""Recompute development scores and prepare evidence for offline human review."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from evaluation.contract import (
    DOMAINS, QTYPES, ROOT, digest_json, is_correct, load_gold, load_retrieval,
    parse_json, pilot_uids, read_jsonl, sha256_file,
)
from evaluation.runner import read_stable, validate_result


CATEGORIES = (
    "missing_retrieval_evidence", "speaker_or_group_attribution",
    "time_update_or_reply", "sufficient_evidence_qa_error",
    "suspected_judge_error", "technical_error", "undetermined", "no_issue",
)


def _read_results(path, planned, scope, input_hash, manifest_hash, gold):
    """Quarantine ambiguous UIDs. Never copy foreign records into reports."""
    rows, rejected, errors = {}, {}, []
    seen, conditions = set(), set()
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            try:
                row = parse_json(raw.decode("utf-8"))
            except (ValueError, UnicodeError):
                errors.append({"line": line_no, "issue": "invalid JSONL"})
                continue
            uid = row.get("uid") if isinstance(row, dict) else None
            if not isinstance(uid, str) or uid not in planned:
                errors.append({"line": line_no, "issue": "missing or unplanned UID"})
                continue
            if uid in seen:
                rejected[uid] = "duplicate UID"
                rows.pop(uid, None)
                errors.append({"line": line_no, "uid": uid, "issue": rejected[uid]})
                continue
            seen.add(uid)
            source = planned[uid]
            try:
                if row.get("scope") != scope:
                    raise ValueError("scope differs from plan")
                if any(k not in row or row[k] != v for k, v in source.items() if k != "qa_status"):
                    raise ValueError("retrieval record differs from input")
                if not isinstance(row.get("gold_answer"), str) or not row["gold_answer"].strip():
                    raise ValueError("missing linked reference answer")
                if gold is not None and row["gold_answer"] != gold[uid]:
                    raise ValueError("reference differs from pinned question")
                for stage in ("qa", "judge"):
                    validate_result(row, stage)
                    provenance = row[stage]["provenance"]
                    if (provenance["retrieval_sha256"] != input_hash or
                            provenance["manifest_sha256"] != manifest_hash):
                        raise ValueError("retrieval or manifest digest differs")
                conditions.add(digest_json([row[s]["provenance"] for s in ("qa", "judge")]))
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                rejected[uid] = f"invalid result ({exc})"
                errors.append({"line": line_no, "uid": uid, "issue": rejected[uid]})
                continue
            rows[uid] = row
    if len(conditions) > 1:
        errors.append({"issue": "mixed run provenance"})
        rejected.update(dict.fromkeys(rows, "mixed run provenance"))
        rows.clear()
    return rows, rejected, errors


def _extract_candidates(source):
    excerpts = []
    for hit in source["retrieved"]:
        passage = hit["formatted_passage"]
        _, sep, body = passage.partition("\n")
        excerpts.append({
            "domain": source["domain"], "msg_node": hit["msg_node"], "rank": hit["rank"],
            "speaker": hit["speaker"], "channel": hit["channel"],
            "excerpt": (body if sep and body else passage)[:200],
        })
    return excerpts


def _case(source, result, gold, issue):
    rec = result or {}
    return {
        "uid": source["uid"], "method": source["method"], "domain": source["domain"],
        "qtype": source["qtype"], "split": source["split"], "question": source["question"],
        "asking_user_id": source["asking_user_id"], "agent_answer": rec.get("agent_answer"),
        "gold_answer": gold, "judge_answer": rec.get("judge_answer"),
        "official_correct": is_correct(rec["judge_answer"]) if result is not None else None,
        "judge_online_correct": rec.get("judge_online_correct"),
        "judge_parse_status": rec.get("judge_parse_status"), "qa_status": rec.get("qa_status"),
        "judge_status": rec.get("judge_status"), "integrity_issue": issue,
        "retrieved_excerpts": _extract_candidates(source),
        "case_sha256": digest_json([source, result, gold, issue]),
        "review": {"status": "pending", "category": None, "explanation": "", "evidence": []},
    }


def _statistics(rows):
    total = len(rows)
    correct = sum(case["official_correct"] is True for case in rows)
    return {"correct": correct, "total": total, "correct_over_planned": correct / total if total else None}


def summarize(input_path, judge_path, scope, manifest_path, data_dir=None):
    manifest_hash = sha256_file(manifest_path)
    sources, input_hash = read_stable(input_path, lambda path: load_retrieval(path, manifest_path, scope))
    planned = {r["uid"]: r for r in sources}
    gold = load_gold(list(planned.values()), data_dir, manifest_path) if data_dir is not None else None
    (results, rejected, errors), judge_hash = read_stable(
        judge_path, lambda path: _read_results(path, planned, scope, input_hash, manifest_hash, gold),
    )
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError("manifest changed during report")
    cases = []
    for uid, source in planned.items():
        row = results.get(uid)
        issue = rejected.get(uid)
        if row is None:
            issue = issue or "missing result"
        elif row["qa_status"] != "ok" or row["judge_status"] != "ok":
            issue = "unfinished QA or judge call"
        ref = gold[uid] if gold is not None else (row or {}).get("gold_answer")
        cases.append(_case(source, row, ref, issue))

    unresolved = [c["uid"] for c in cases if c["integrity_issue"]]
    complete = not errors and not unresolved
    by_domain = {d: _statistics([c for c in cases if c["domain"] == d]) for d in DOMAINS}
    by_qtype = {t: _statistics([c for c in cases if c["qtype"] == t]) for t in QTYPES}
    overall = _statistics(cases)
    for stats in [overall, *by_domain.values(), *by_qtype.values()]:
        stats["accuracy"] = stats["correct_over_planned"] if complete else None
    example = next(iter(results.values()), None)
    summary = {
        "scope": scope, "split": "development", "method": next(iter(planned.values()))["method"],
        "status": "complete" if complete else "incomplete",
        "input_sha256": input_hash, "manifest_sha256": manifest_hash, "judge_sha256": judge_hash,
        "provenance": {s: example[s]["provenance"] for s in ("qa", "judge")} if example else None,
        "overall": overall, "by_domain": by_domain, "by_qtype": by_qtype,
        "integrity_errors": errors, "unresolved_uids": unresolved,
        "empty_judge_uids": [c["uid"] for c in cases if c["judge_answer"] is not None and not c["judge_answer"].strip()],
        "qa_status_counts": dict(Counter(c["qa_status"] or "missing" for c in cases)),
        "judge_status_counts": dict(Counter(c["judge_status"] or "missing" for c in cases)),
        "parse_status_counts": dict(Counter(c["judge_parse_status"] or "missing" for c in cases)),
        "online_vs_summary": [
            {"uid": c["uid"], "online": c["judge_online_correct"], "final": c["official_correct"]}
            for c in cases if c["official_correct"] is not None and
            c["judge_online_correct"] != c["official_correct"]
        ],
    }
    return summary, cases, planned


def _conversation_passages(domain, data_dir):
    """Only used for human-supplied post-hoc citations."""
    from retrieval.data_tools import UP

    if data_dir is None:
        raise ValueError("conversation citations require --data-dir")
    name = f"data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json"
    pinned = json.loads((ROOT / "configs/data_sources.json").read_text(encoding="utf-8"))
    path = data_dir / name
    if sha256_file(path) != pinned["files"][name]:
        raise ValueError("conversation source hash mismatch")
    return {m["msg_node"]: UP["format_retrieved_message"](m)
            for m in UP["load_conversation_messages"](str(path))}


def apply_reviews(cases, planned, reviews_path, data_dir=None):
    """Apply reviewed annotations only to the precise case they were written for."""
    by_uid = {c["uid"]: c for c in cases}
    conversations, seen = {}, set()
    for row in read_jsonl(reviews_path):
        uid = row.get("uid")
        if not isinstance(uid, str) or uid not in by_uid or uid in seen:
            raise ValueError("review contains a duplicate or unplanned UID")
        seen.add(uid)
        case = by_uid[uid]
        if row.get("case_sha256") != case["case_sha256"]:
            raise ValueError(f"{uid}: stale case fingerprint")
        review = row["review"]
        if review["status"] == "pending":
            continue
        if review["status"] != "reviewed" or review["category"] not in CATEGORIES:
            raise ValueError(f"{uid}: invalid review status/category")
        if review["category"] == "no_issue" and (not case["official_correct"] or case["integrity_issue"]):
            raise ValueError(f"{uid}: a miss or unfinished case requires a diagnosis or undetermined")
        if not isinstance(review["explanation"], str) or not review["explanation"].strip():
            raise ValueError(f"{uid}: explanation required")
        evidence = review["evidence"]
        if not isinstance(evidence, list):
            raise ValueError(f"{uid}: evidence must be a list")
        checked = []
        for citation in evidence:
            domain, node, quote = citation["domain"], citation["msg_node"], citation["quote"]
            if domain != case["domain"] or not isinstance(quote, str) or not quote.strip():
                raise ValueError(f"{uid}: citation must have the case domain and a verbatim quote")
            kind = citation["source"]
            if kind == "retrieved":
                rank = citation["rank"]
                hits = planned[uid]["retrieved"]
                if type(rank) is not int or not 1 <= rank <= len(hits) or hits[rank - 1]["msg_node"] != node:
                    raise ValueError(f"{uid}: invalid retrieved locator")
                passage = hits[rank - 1]["formatted_passage"]
            elif kind == "conversation":
                if domain not in conversations:
                    conversations[domain] = _conversation_passages(domain, data_dir)
                passage = conversations[domain].get(node, "")
            else:
                raise ValueError(f"{uid}: unknown citation source")
            if quote not in passage:
                raise ValueError(f"{uid}: quote differs from source passage")
            checked.append({"source": kind, "domain": domain, "msg_node": node, "quote": quote})
            if kind == "retrieved":
                checked[-1]["rank"] = rank
        if review["category"] in CATEGORIES[:4] and not checked:
            raise ValueError(f"{uid}: a message-based diagnosis requires a cited passage")
        case["review"] = {
            "status": "reviewed", "category": review["category"],
            "explanation": review["explanation"], "evidence": checked,
        }


def _write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="original retrieval JSONL")
    p.add_argument("--judge", type=Path, required=True)
    p.add_argument("--scope", choices=["sample", "pilot", "development"], required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "configs/manifest.json")
    p.add_argument("--output-dir", type=Path, required=True, help="new directory")
    p.add_argument("--data-dir", type=Path, help="recheck and fill references from pinned data")
    p.add_argument("--reviews", type=Path, help="review.jsonl copy with human annotations")
    args = p.parse_args(argv)
    try:
        summary, cases, planned = summarize(args.input, args.judge, args.scope, args.manifest, args.data_dir)
        if args.reviews:
            apply_reviews(cases, planned, args.reviews, args.data_dir)
        failures = [c for c in cases if c["official_correct"] is not True or c["integrity_issue"]]
        if args.scope == "development":
            pilot = set(pilot_uids(list(planned)))
            review_cases = [c for c in cases if c["uid"] in pilot or c["official_correct"] is not True
                            or c["integrity_issue"] or c["review"]["status"] == "reviewed"]
        else:
            review_cases = cases
        summary["failure_count"] = len(failures)
        summary["human_review"] = {
            "candidates": len(review_cases),
            "reviewed": sum(c["review"]["status"] == "reviewed" for c in review_cases),
            "categories": dict(Counter(c["review"]["category"] for c in review_cases if c["review"]["category"])),
            "suspected_judge_uids": [c["uid"] for c in review_cases if c["review"]["category"] == "suspected_judge_error"],
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_jsonl(args.output_dir / "failures.jsonl", failures)
        _write_jsonl(args.output_dir / "review.jsonl", review_cases)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(2, f"report failed: {exc}\n")
    print(f"{summary['status']} ({args.scope}); {args.output_dir / 'summary.json'}")
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
