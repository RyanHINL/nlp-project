"""Audit the full BM25 development run and its promoted pilot records offline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from evaluation import contract as c, local_model, report, runner
from evaluation.pilot.review_export import frozen, portable_zip


STAGES = ("qa", "judge")
NOT_ATTEMPTED = {"pending", "skipped_qa"}
# Promotion changes only the enclosing QA file reference, cache origin and seal.
PROMOTION_FIELDS = {"provenance", "reused_from_sha256", "audit_sha256"}


def check(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def read_json(path):
    return c.parse_json(path.read_bytes())


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scan(path, allowed, issues):
    """Keep line locators, quarantine both copies of duplicates, omit foreign content."""
    rows, seen = {}, set()
    if not path.is_file():
        issues.append({"file": path.name, "issue": "missing file"})
        return rows, None
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lines = raw.split(b"\n")
    if not lines[-1]:
        lines.pop()
    for line, content in enumerate(lines, 1):
        try:
            record = c.parse_json(content.decode("utf-8"))
            uid = record.get("uid") if isinstance(record, dict) else None
            check(isinstance(uid, str) and uid in allowed, "foreign or missing UID")
            if uid in seen:
                rows.pop(uid, None)
                raise ValueError(f"duplicate UID {uid}")
            seen.add(uid)
            rows[uid] = (line, record)
        except (ValueError, UnicodeError) as exc:
            issues.append({"file": path.name, "line": line, "issue": str(exc)})
    missing = sorted(set(allowed) - rows.keys())
    if missing:
        issues.append({"file": path.name, "issue": "missing or quarantined UIDs", "uids": missing})
    return rows, digest


def validate_rows(stage, scope, path, entries, sources, info, gold, qa_rows, issues):
    valid = {}
    allowed = set(c.RETRIEVAL_FIELDS) | {"scope", "qa", "agent_answer"}
    if stage == "judge":
        allowed.update({"gold_answer", "judge", "judge_status", "judge_answer", "judge_online_correct", "judge_parse_status"})
    for uid, (line, row) in entries.items():
        try:
            check(row.keys() == allowed, "unexpected saved record fields")
            check(row["scope"] == scope and all(row.get(k) == v for k, v in sources[uid].items() if k != "qa_status"),
                  "retrieval source or scope differs")
            runner.validate_result(row, stage)
            audit = row[stage]
            check(audit["provenance"] == info, "source/config/prompt/code fingerprint differs")
            if stage == "judge":
                parent = qa_rows[uid][1]
                check(row["gold_answer"] == gold[uid] and row["qa"] == parent["qa"]
                      and row["qa_status"] == parent["qa_status"] and row["agent_answer"] == parent["agent_answer"],
                      "gold or parent QA differs")
            if audit["status"] == "ok":
                usage = audit["response"]["usage"]
                choice, = audit["response"]["choices"]
                settings = info["settings"]
                check(audit["usage"] == usage and usage["prompt_tokens"] == audit["prompt_tokens"] and
                      usage["completion_tokens"] == usage["total_tokens"] - usage["prompt_tokens"] and
                      0 <= usage["completion_tokens"] <= settings["max_tokens"] and
                      0 < usage["prompt_tokens"] <= settings["context_tokens"] - settings["max_tokens"],
                      "response usage, prompt or completion budget differs")
                check(isinstance(audit["response"].get("id"), str) and audit["response"]["id"] and
                      choice["finish_reason"] == "stop" and audit["response"]["model"] == settings["model"] and
                      audit["server_context_tokens"] >= settings["context_tokens"] and
                      audit["error"] is None and audit["seconds"] >= 0, "inconsistent successful call")
            valid[uid] = (line, row)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            issues.append({"file": path.name, "line": line, "uid": uid, "issue": str(exc)})
    return valid


def compare_cache(uid, stage, audit, pilot, pilot_sha):
    origin = audit.get("reused_from_sha256")
    if uid not in pilot:
        check(origin is None, "unexpected cache origin for an additional development UID")
        return
    check(origin == pilot_sha, "pilot UID missing its exact pilot-file digest")
    prior = pilot[uid][1][stage]
    against = {k: v for k, v in prior.items() if k not in PROMOTION_FIELDS}
    check({k: v for k, v in audit.items() if k not in PROMOTION_FIELDS} == against,
          "promoted call differs from pilot request/response/time/status/cache key")


def cost(rows, stage):
    audits = [row[stage] for row in rows]
    attempted = [a for a in audits if a["status"] not in NOT_ATTEMPTED]
    usage = [a["usage"] for a in attempted if isinstance(a.get("usage"), dict) and
             all(type(a["usage"].get(k)) is int for k in ("prompt_tokens", "completion_tokens"))]
    elapsed = [a["seconds"] for a in attempted if type(a.get("seconds")) in (float, int)]
    return {
        "records": len(audits), "attempts": len(attempted),
        "raw_responses": sum(a.get("response") is not None for a in attempted),
        "unknown_delivery_or_completion": sum(a["status"] in {"timeout", "interrupted", "in_flight"} for a in attempted),
        "missing_usage_on_attempts": len(attempted) - len(usage), "missing_time_on_attempts": len(attempted) - len(elapsed),
        "recorded_prompt_tokens": sum(u["prompt_tokens"] for u in usage),
        "recorded_completion_tokens": sum(u["completion_tokens"] for u in usage),
        "recorded_client_seconds": sum(elapsed),
        "status": dict(Counter(a["status"] for a in audits)),
        "format": dict(Counter(a["format_status"] for a in audits)),
        "finish_reason": dict(Counter(str(a["response"]["choices"][0].get("finish_reason"))
                                      for a in attempted if isinstance(a.get("response"), dict)
                                      and isinstance(a["response"].get("choices"), list)
                                      and len(a["response"]["choices"]) == 1 and isinstance(a["response"]["choices"][0], dict))),
    }


def totals(per_stage):
    keys = ("records", "attempts", "raw_responses", "unknown_delivery_or_completion", "missing_usage_on_attempts",
            "missing_time_on_attempts", "recorded_prompt_tokens", "recorded_completion_tokens", "recorded_client_seconds")
    return {k: sum(per_stage[s][k] for s in STAGES) for k in keys}


def text_after_final_marker(raw):
    """The official parser chooses the last marked line and extracts only that line."""
    if raw is None:
        return False
    later_text = False
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.lower().startswith(("final:", "final answer:")):
            return later_text
        if line:
            later_text = True
    return False


def format_diagnostics(rows, stage):
    by_signal = {key: [] for key in ("missing_raw", "empty_raw", "empty_extracted", "fallback",
                                     "nonempty_lines_after_last_final_marker")}
    for row in rows:
        audit = row[stage]
        raw = audit["raw_output"]
        uid = row["uid"]
        if raw is None:
            by_signal["missing_raw"].append(uid)
            continue
        if not raw.strip():
            by_signal["empty_raw"].append(uid)
        if not row[runner.ANSWER_FIELD[stage]]:
            by_signal["empty_extracted"].append(uid)
        if audit["format_status"] == "fallback":
            by_signal["fallback"].append(uid)
        if text_after_final_marker(raw):
            by_signal["nonempty_lines_after_last_final_marker"].append(uid)
    return {key: {"count": len(uids), "uids": uids} for key, uids in by_signal.items()}


def logged_requests(path):
    raw = path.read_bytes()
    requests = Counter()
    access = re.compile(r'"(GET|POST) ([^ ?"]+)(?:\?[^ "]*)? HTTP/[\d.]+" (\d{3})')
    for line in raw.decode("utf-8").splitlines():
        hit = access.search(line)
        if hit:
            method, route, status = hit.groups()
            requests[(method, route, status)] += 1
    return {"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "chat_posts": sum(n for (m, route, _), n in requests.items() if (m, route) == ("POST", "/v1/chat/completions")),
            "tokenize_posts": sum(n for (m, route, _), n in requests.items() if (m, route) == ("POST", "/tokenize")),
            "non_200_model_posts": sum(n for (m, route, s), n in requests.items()
                                       if m == "POST" and route in ("/v1/chat/completions", "/tokenize") and s != "200"),
            "http_status_counts": {" ".join(key): n for key, n in sorted(requests.items())}}


def stage_view(row, stage, location, original_line):
    if row is None:
        return None
    a = row[stage]
    resp = a.get("response") if isinstance(a.get("response"), dict) else {}
    choices = resp.get("choices") or []
    origin = "pilot_cache" if a.get("reused_from_sha256") else "new_development"
    if a["status"] in NOT_ATTEMPTED:
        origin = "unattempted"
    return {"status": a["status"], "format": a["format_status"], "record_origin": origin, "cache_key": a["cache_key"],
            "nonempty_lines_after_last_final_marker": text_after_final_marker(a["raw_output"]),
            "reused_from_sha256": a.get("reused_from_sha256"), "prompt_tokens": a["prompt_tokens"], "usage": a["usage"],
            "seconds": a["seconds"], "error": a["error"], "response_id": resp.get("id"),
            "finish_reason": choices[0].get("finish_reason") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None,
            "raw_record": location, "source_line_1based": original_line, "audit_sha256": a["audit_sha256"]}


def audit(args):
    run = args.run_dir
    deployment, config = frozen(run)
    sources, input_sha = runner.read_stable(args.input, lambda p: c.load_retrieval(p, c.MANIFEST, "development"))
    ordered = [r["uid"] for r in sources]
    planned = {r["uid"]: r for r in sources}
    pilot = set(c.pilot_uids(ordered))
    gold = c.load_gold(sources, args.data_dir)
    cases_path = args.report_dir / "summary.json"
    stored = read_json(cases_path)
    judged = run / "dev.judge.jsonl"
    computed, cases, _ = report.summarize(args.input, judged, "development", c.MANIFEST, args.data_dir)
    check(all(stored.get(k) == val for k, val in computed.items()), "saved official summary differs from offline recomputation")
    issues = []
    checks = read_json(run / "dev.qa.preflight.json")
    lengths = runner.index_rows(checks["cases"])
    weights = read_json(run / "model.sha256.json")
    check(list(lengths) == ordered and checks["scope"] == checks["split"] == "development" and
          checks["input_sha256"] == input_sha and checks["manifest_sha256"] == c.sha256_file(c.MANIFEST) and
          checks["chat_template_sha256"] == deployment["chat_template_sha256"] and
          all(weights[k]["sha256"] == v for k, v in checks["tokenizer"]["files_sha256"].items()) and
          checks["context_tokens"] == config["context_tokens"] and checks["reserved_completion_tokens"] == config["qa"]["max_tokens"],
          "CPU preflight does not match deployment/manifest/input")
    check(all(type(r["prompt_tokens"]) is int and r["prompt_tokens"] > 0 for r in lengths.values()),
          "invalid preflight token count")
    over_budget = [uid for uid in ordered if lengths[uid]["prompt_tokens"] + config["qa"]["max_tokens"] > config["context_tokens"]]
    check(checks["over_budget_uids"] == over_budget, "inconsistent preflight overflow list")
    artifacts, records = {}, {}
    for prefix, scope, wanted in [("pilot", "pilot", pilot), ("dev", "development", set(ordered))]:
        for stage in STAGES:
            path = run / f"{prefix}.{stage}.jsonl"
            entries, digest = scan(path, wanted, issues)
            artifacts[path.name] = digest
            info = runner.provenance(stage, local_model.settings_for(config, stage), input_sha, c.MANIFEST,
                                     artifacts.get(f"{prefix}.qa.jsonl"))
            rows = validate_rows(stage, scope, path, entries, planned, info, gold, records.get((prefix, "qa")), issues)
            records[prefix, stage] = rows
            if prefix == "dev":
                for uid, (_, row) in rows.items():
                    try:
                        check((uid in pilot) == (uid in records["pilot", stage]), "corrupt/missing pilot parent")
                        compare_cache(uid, stage, row[stage], records["pilot", stage], artifacts[f"pilot.{stage}.jsonl"])
                    except (ValueError, KeyError) as exc:
                        issues.append({"uid": uid, "stage": stage, "issue": str(exc)})
    check(artifacts["dev.judge.jsonl"] == stored["judge_sha256"] and input_sha == stored["input_sha256"],
          "input/judge changed between official report and audit")
    groups = {key: {} for key in ("historical_pilot", "new_development", "all_90")}
    ids = []
    diagnostics = {}
    for stage in STAGES:
        current = [r for _, r in records["dev", stage].values()]
        diagnostics[stage] = format_diagnostics(current, stage)
        groups["historical_pilot"][stage] = cost([r for _, r in records["pilot", stage].values()], stage)
        groups["new_development"][stage] = cost([r for r in current if not r[stage].get("reused_from_sha256")], stage)
        groups["all_90"][stage] = cost(current, stage)
        for row in current:
            audit = row[stage]
            response = audit.get("response")
            if isinstance(response, dict) and response.get("id"):
                ids.append(response["id"])
            if stage == "qa" and audit["prompt_tokens"] is not None and audit["prompt_tokens"] != lengths[row["uid"]]["prompt_tokens"]:
                issues.append({"uid": row["uid"], "stage": stage, "issue": "served token count differs from CPU preflight"})
    duplicate_ids = [key for key, n in Counter(ids).items() if n > 1]
    if duplicate_ids:
        issues.append({"issue": "duplicate response IDs", "response_ids": duplicate_ids})
    for group in groups.values():
        group["combined"] = totals(group)
    service = logged_requests(args.server_log)
    new = groups["new_development"]["combined"]
    service["extra_chat_posts_over_recorded_responses"] = service["chat_posts"] - new["raw_responses"]
    service["extra_tokenize_posts_over_recorded_attempts"] = service["tokenize_posts"] - new["attempts"]
    service["access_log_scope"] = ("Counts cover this dedicated server log. Access lines carry no UID. "
                                   "An aggregate mismatch cannot identify a repeated UID or prove delivery.")
    if service["extra_chat_posts_over_recorded_responses"] or service["extra_tokenize_posts_over_recorded_attempts"] or service["non_200_model_posts"]:
        issues.append({"issue": "server access counts require reconciliation"})
    official = {k: stored[k] for k in ("status", "scope", "method", "overall", "by_domain", "by_qtype", "unresolved_uids",
                                        "empty_judge_uids", "online_vs_summary", "parse_status_counts", "integrity_errors")}
    accounting = {"status": "complete" if not issues and stored["status"] == "complete" else "incomplete",
                  "scope": "development", "planned": 90, "planned_pilot_reuse_per_stage": len(pilot),
                  "planned_new_per_stage": len(ordered)-len(pilot), "official": official,
                  "cost": groups, "service": service, "distinct_response_ids": len(set(ids)), "issues": issues,
                  "response_format_diagnostics": diagnostics,
                  "qa_prompt_preflight": {"cpu_tokenizer": checks["tokenizer"], "context_tokens": config["context_tokens"],
                                          "qa_reserve": config["qa"]["max_tokens"], "judge_reserve": config["judge"]["max_tokens"],
                                          "max_prompt_tokens": max(r["prompt_tokens"] for r in lengths.values()),
                                          "over_budget_uids": over_budget},
                  "cost_units": ("Durations sum observed client calls (tokenization and completion), not run wall time. "
                                 "Historical pilot includes its first case once. Reused rows charge zero new model calls. "
                                 "Totals use only recorded usage; missing usage or time is counted explicitly."),
                  "cache_hits": {s: sum("reused_from_sha256" in r[s] for _, r in records["dev", s].values()) for s in STAGES},
                  "duplicate_response_ids": duplicate_ids,
                  "client_retry_policy": "The frozen client sends at most one completion per non-cached stage attempt, with no answer-driven regeneration.",
                  "human_review": dict(Counter(case["review"]["status"] for case in cases)),
                  "model": deployment["model"], "revision": deployment["revision"],
                  "fingerprints": {**artifacts, args.input.name: input_sha, "manifest.json": c.sha256_file(c.MANIFEST),
                                   "deployment.frozen.json": c.sha256_file(run/"deployment.frozen.json"),
                                   "local.json": c.sha256_file(run/"local.json"),
                                   "dev.qa.preflight.json": c.sha256_file(run/"dev.qa.preflight.json"),
                                   "official_summary.json": c.sha256_file(cases_path)}}
    return accounting, cases, planned, records, lengths


def write(args, accounting, cases, sources, records, lengths):
    output = args.output_dir
    check(not output.exists(), "output directory already exists")
    if args.zip:
        check(not args.zip.exists() and args.zip.resolve() != output.resolve() and
              output.resolve() not in args.zip.resolve().parents, "ZIP must be a new file outside output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        root = Path(td)/"bundle"
        raw = root/"raw"
        raw.mkdir(parents=True)
        locations = {}
        for stage in STAGES:
            path = raw / f"{stage}.jsonl"
            with path.open("x", encoding="utf-8") as out:
                for line, (uid, (_, row)) in enumerate(sorted(records["dev", stage].items(), key=lambda item: item[1][0]), 1):
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                    locations[uid, stage] = {"path": path.relative_to(root).as_posix(), "line_1based": line}
        with (root/"ledger.jsonl").open("x", encoding="utf-8") as out:
            for case in cases:
                uid = case["uid"]
                entry = {k: case[k] for k in ("uid","domain","qtype","split","method","question","asking_user_id",
                         "gold_answer","agent_answer","judge_answer","official_correct","judge_online_correct",
                         "judge_parse_status","integrity_issue","case_sha256","review")}
                entry["audit_issues"] = [finding for finding in accounting["issues"]
                                         if finding.get("uid") == uid or uid in finding.get("uids", ())]
                entry["verified_qa_prompt_tokens_cpu"] = lengths[uid]["prompt_tokens"]
                entry["retrieved_ranked_msg_nodes"] = [d["msg_node"] for d in sources[uid]["retrieved"]]
                for stage in STAGES:
                    number, record = records["dev", stage].get(uid, (None, None))
                    entry[stage] = stage_view(record, stage, locations.get((uid,stage)), number)
                out.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        save_json(root/"accounting.json", accounting)
        (root/"config").mkdir()
        for name in ("local.json","deployment.frozen.json","chat_template.jinja","model.sha256.json"):
            shutil.copyfile(args.run_dir/name,root/"config"/name)
        (root/"README.md").write_text(
            "# BM25 development audit\n\n"
            "Start with [accounting.json](accounting.json) for denominators, group results, integrity findings, cache and cost totals. "
            "[ledger.jsonl](ledger.jsonl) contains one record per planned development UID in manifest order. "
            "Its raw_record path and one-based line select the complete original request, response, passages, error and fingerprints. "
            "Human adjudication is pending. The official score remains the separate judge verdict.\n\n"
            "The response_format_diagnostics entry lists QA and judge UIDs with missing, empty or fallback outputs, "
            "and text after the last final marker. The official parser uses the content on that marked line. "
            "Later nonblank lines can help reviewers locate omitted lists or unrelated trailing remarks. "
            "These flags do not alter the response, status or score.\n\n"
            "Verify the packet from its root with `shasum -a 256 -c SHA256SUMS`. Source file SHA256s are in accounting.json. "
            "The raw JSONL files preserve validated record contents. Source lines and exported lines are both given. "
            "Input data by Yang et al., GroupMemBench, revision 461f00c3e8a1a4a3a2ed5814b729014502c4702f, "
            "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).\n", encoding="utf-8")
        sums = [f"{c.sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*")) if p.is_file()]
        (root/"SHA256SUMS").write_text("".join(sums), encoding="utf-8")
        if args.zip:
            portable_zip(root, args.zip)
        root.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="frozen pilot directory with pilot.* and dev.*")
    parser.add_argument("--server-log", type=Path, required=True, help="dedicated new-78 service log")
    parser.add_argument("--report-dir", type=Path, required=True, help="normal evaluation.report output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--input", type=Path, default=c.ROOT/"runs/bm25_development.jsonl")
    parser.add_argument("--data-dir", type=Path, default=c.DATA)
    args = parser.parse_args()
    try:
        accounting, cases, sources, records, lengths = audit(args)
        write(args, accounting, cases, sources, records, lengths)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(f"{accounting['status']}: {len(cases)} planned records in {args.output_dir}")
    return 0 if accounting["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
