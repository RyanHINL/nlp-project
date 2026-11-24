"""The development-only retrieval-v1 boundary and pinned GroupMemBench QA protocol."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Optional, Tuple

from retrieval.data_tools import DOMAINS, QTYPES
from vendor.GroupMemBench.task_synthesis.summarize_typed_eval import _is_correct as is_correct

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/manifest.json"
DATA = ROOT / "data/GroupMemBench"
VENDOR = ROOT / "vendor/GroupMemBench"
IDENTITIES = ROOT / "evaluation/development_identities.json"
TOP_K = 10
RETRIEVAL_FIELDS = ("uid", "domain", "qtype", "split", "method", "question", "asking_user_id",
                    "retrieved", "retrieval_seconds", "qa_status")
PASSAGE_FIELDS = ("rank", "msg_node", "channel", "speaker", "score", "formatted_passage")
OFFLINE_FIELDS = {"answer", "gold", "reference_answer", "gold_answer", "agent_answer", "judge_answer",
                  "qa", "judge", "judge_parse_status", "scope", "is_noise", "is_decision_point"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"invalid JSON number {value}")


def _finite_float(text):
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite float range")
    return number


def parse_json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant,
                      parse_float=_finite_float)


def _records(lines, path):
    for number, line in enumerate(lines, 1):
        try:
            row = parse_json(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        yield row


def read_jsonl(path):
    """Reject invalid rows, including blanks, duplicate fields and nonfinite numbers."""
    with Path(path).open(encoding="utf-8") as stream:
        return list(_records(stream, path))


def _development(manifest_path):
    pinned = parse_json(MANIFEST.read_text(encoding="utf-8"))
    manifest = parse_json(Path(manifest_path).read_text(encoding="utf-8"))
    for split, size in (("development", 90), ("evaluation", 337)):
        uids = manifest[split]
        if uids != pinned[split] or len(uids) != size or len(set(uids)) != size:
            raise ValueError(f"expected the committed {split} manifest")
    if set(manifest["development"]).intersection(manifest["evaluation"]):
        raise ValueError("development and evaluation manifests overlap")
    return manifest["development"]


def pilot_uids(development_uids):
    """First UID in the fixed manifest for each (domain, qtype), in manifest order."""
    first = {}
    for uid in development_uids:
        domain, qtype, _ = uid.split("/")
        first.setdefault((domain, qtype), uid)
    if set(first) != {(domain, qtype) for domain in DOMAINS for qtype in QTYPES}:
        raise ValueError("pilot requires each domain/question type in development")
    return list(first.values())


def _identity(row):
    return digest_json([row["question"], row.get("asking_user_id") or ""])


def _validate_retrieval(row, development, identities):
    if missing := set(RETRIEVAL_FIELDS) - row.keys():
        raise ValueError(f"missing retrieval fields {sorted(missing)}")
    if OFFLINE_FIELDS.intersection(row):
        raise ValueError("unexpected evaluation fields in retrieval input")
    uid = row["uid"]
    if not isinstance(uid, str) or uid not in development or row["split"] != "development":
        raise ValueError("expected a UID in the fixed development split")
    domain, qtype, _ = uid.split("/")
    if row["domain"] != domain or row["qtype"] != qtype:
        raise ValueError(f"{uid}: UID, domain and qtype disagree")
    for key in ("question", "method"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{uid}: invalid {key}")
    if row["asking_user_id"] is not None and not isinstance(row["asking_user_id"], str):
        raise ValueError(f"{uid}: invalid asking_user_id")
    if _identity(row) != identities[uid]:
        raise ValueError(f"{uid}: question/asking_user_id differs from pinned source")
    seconds = row["retrieval_seconds"]
    if type(seconds) not in (float, int) or seconds < 0:
        raise ValueError(f"{uid}: invalid retrieval_seconds")
    if row["qa_status"] != "not_run":
        raise ValueError(f"{uid}: retrieval input must have qa_status=not_run")
    docs = row["retrieved"]
    if not isinstance(docs, list) or len(docs) > TOP_K:
        raise ValueError(f"{uid}: expected a ranked list of at most {TOP_K} passages")
    seen = set()
    for rank, doc in enumerate(docs, 1):
        if not isinstance(doc, dict) or type(doc.get("rank")) is not int or doc["rank"] != rank:
            raise ValueError(f"{uid}: expected passage rank {rank} in place")
        if OFFLINE_FIELDS.intersection(doc):
            raise ValueError(f"{uid}: unexpected annotation at rank {rank}")
        for key in ("msg_node", "channel", "speaker", "formatted_passage"):
            if not isinstance(doc.get(key), str) or not doc[key].strip():
                raise ValueError(f"{uid}: missing or invalid {key} at rank {rank}")
        if doc["msg_node"] in seen:
            raise ValueError(f"{uid}: repeated msg_node at rank {rank}")
        seen.add(doc["msg_node"])
        score = doc.get("score")
        if type(score) not in (float, int):
            raise ValueError(f"{uid}: invalid score at rank {rank}")


def load_retrieval(path, manifest_path=MANIFEST, scope="development"):
    """Read one condition. Pilot and development both require the full retrieval artifact."""
    if scope not in {"development", "pilot", "sample"}:
        raise ValueError(f"unknown development scope {scope!r}")
    development = _development(manifest_path)
    identities = parse_json(IDENTITIES.read_text(encoding="utf-8"))
    wanted = set(development)
    if identities.keys() != wanted:
        raise ValueError("question/asker identity catalog differs from the development manifest")
    by_uid = {}
    for row in read_jsonl(path):
        _validate_retrieval(row, wanted, identities)
        if row["uid"] in by_uid:
            raise ValueError(f"duplicate retrieval UID {row['uid']}")
        # Only retrieval-v1 fields may flow into downstream QA artifacts.
        projected = {key: row[key] for key in RETRIEVAL_FIELDS}
        projected["retrieved"] = [{key: doc[key] for key in PASSAGE_FIELDS} for doc in row["retrieved"]]
        by_uid[row["uid"]] = projected
    if not by_uid or len({r["method"] for r in by_uid.values()}) != 1:
        raise ValueError("expected a nonempty retrieval input with one method")
    if scope == "sample":
        if len(by_uid) == len(development):
            raise ValueError("use development or pilot scope for a full input")
        return list(by_uid.values())
    missing = wanted - by_uid.keys()
    if missing:
        raise ValueError(f"incomplete development retrieval: {len(missing)} missing, first {min(missing)}")
    selection = development if scope == "development" else pilot_uids(development)
    return [by_uid[uid] for uid in selection]


def _source_rows(wanted, data_dir):
    """Offline projection only. Hash the exact bytes read and yield only requested identities."""
    hashes = parse_json((ROOT / "configs/data_sources.json").read_text(encoding="utf-8"))["files"]
    groups = sorted({tuple(uid.split("/")[:2]) for uid in wanted})
    found = set()
    for domain, qtype in groups:
        name = f"questions/{domain}/{qtype}.jsonl"
        raw = (Path(data_dir) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes[name]:
            raise ValueError(f"{name}: checksum differs from configs/data_sources.json")
        for row in _records(io.StringIO(raw.decode("utf-8")), name):
            uid = f"{domain}/{qtype}/{row['id']}"
            if uid not in wanted:
                continue
            if uid in found:
                raise ValueError(f"{uid}: duplicate in question source")
            found.add(uid)
            yield uid, row
    if found != wanted:
        raise ValueError(f"missing {len(wanted - found)} source questions")


def build_identity_catalog(data_dir=DATA, manifest_path=MANIFEST):
    """Offline, reproducible fingerprint of the only identity fields QA may use."""
    wanted = set(_development(manifest_path))
    return {uid: _identity(q) for uid, q in _source_rows(wanted, data_dir)}


def load_gold(rows, data_dir=DATA, manifest_path=MANIFEST):
    """Join pinned references only after the caller has selected its development scope."""
    wanted = {}
    allowed = set(_development(manifest_path))
    for row in rows:
        uid = row["uid"]
        if uid not in allowed or row["split"] != "development":
            raise ValueError("gold lookup is development-only")
        if uid in wanted or uid.split("/")[:2] != [row["domain"], row["qtype"]]:
            raise ValueError(f"{uid}: duplicate or inconsistent identity in gold lookup")
        wanted[uid] = row
    gold = {}
    for uid, source in _source_rows(set(wanted), data_dir):
        if _identity(source) != _identity(wanted[uid]):
            raise ValueError(f"{uid}: question/asking_user_id differs from pinned source")
        if not isinstance(source.get("answer"), str) or not source["answer"].strip():
            raise ValueError(f"{uid}: missing reference answer")
        gold[uid] = source["answer"]
    return gold


def _system_prompt(role):
    return (VENDOR / f"prompts/hipporag_{role}_system.txt").read_text(encoding="utf-8").strip()


def qa_messages(row):
    """Project only the upstream user prompt inputs, preserving every passage verbatim."""
    passages = "\n\n".join(f"[{i}] {doc['formatted_passage']}" for i, doc in enumerate(row["retrieved"], 1))
    asker = row["asking_user_id"]
    asker_line = f"Asking user: {asker}\n\n" if asker else ""
    user = (f"{asker_line}Question:\n{row['question']}\n\nRetrieved passages:\n{passages}\n\n"
            "Answer the question using the retrieved passages.")
    return [{"role": "system", "content": _system_prompt("agent")}, {"role": "user", "content": user}]


def judge_messages(question, gold, agent_answer):
    user = f"Question:\n{question}\n\nGold Answer:\n{gold}\n\nAgent Answer:\n{agent_answer}\n"
    return [{"role": "system", "content": _system_prompt("judge")}, {"role": "user", "content": user}]


def _official_parsers():
    # Match retrieval.data_tools' approach: execute only pinned pure functions.
    # Importing the entire eval_lib would also import the upstream API client.
    path = VENDOR / "baselines/rag_common/eval_lib.py"
    names = {"extract_final", "split_reasoning_and_final", "parse_judgment"}
    functions = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(Tuple=Tuple, Optional=Optional)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope["split_reasoning_and_final"], scope["parse_judgment"]


split_reasoning_and_final, parse_judgment = _official_parsers()


def answer_format(raw):
    """Describe presence of the upstream final marker without modifying extracted text."""
    if not raw.strip():
        return "empty"
    if any(line.strip().lower().startswith(("final:", "final answer:")) for line in raw.splitlines()):
        return "marked"
    return "fallback"
