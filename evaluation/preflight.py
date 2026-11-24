"""Profile the exact development QA prompts on CPU, without loading weights or gold."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from evaluation.contract import MANIFEST, load_retrieval, qa_messages, sha256_file


def distribution(values):
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[math.ceil(len(ordered) * .50) - 1],
        "p95": ordered[math.ceil(len(ordered) * .95) - 1],
        "max": ordered[-1],
    }


def profile(rows, tokenizer=None, context_tokens=None, reserve=None):
    cases = []
    for row in rows:
        messages = qa_messages(row)
        contents = [message["content"] for message in messages]
        case = {
            "uid": row["uid"], "passages": len(row["retrieved"]),
            "content_characters": sum(map(len, contents)),
            "content_utf8_bytes": sum(len(value.encode("utf-8")) for value in contents),
        }
        if tokenizer is not None:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                return_dict=False)
            case["prompt_tokens"] = len(ids)
        cases.append(case)

    summary = {
        "method": rows[0]["method"], "planned": len(rows),
        "by_domain": dict(Counter(row["domain"] for row in rows)),
        "by_qtype": dict(Counter(row["qtype"] for row in rows)),
        "passage_count_histogram": dict(Counter(case["passages"] for case in cases)),
        "lengths": {key: distribution([case[key] for case in cases])
                    for key in ("content_characters", "content_utf8_bytes")},
        "cases": cases,
    }
    if tokenizer is not None:
        summary["lengths"]["prompt_tokens"] = distribution([case["prompt_tokens"] for case in cases])
        summary["chat_template_sha256"] = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
        if context_tokens is not None:
            summary["context_tokens"] = context_tokens
            summary["reserved_completion_tokens"] = reserve
            summary["over_budget_uids"] = [c["uid"] for c in cases if c["prompt_tokens"] + reserve > context_tokens]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--scope", choices=("sample", "pilot", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New CPU-only profile JSON")
    parser.add_argument("--tokenizer-dir", type=Path, help="Optional local Transformers tokenizer directory")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--reserve", type=int, help="QA completion tokens reserved within the context window")
    args = parser.parse_args()
    if (args.context_tokens is None) != (args.reserve is None):
        parser.error("--context-tokens and --reserve are used together")
    if args.context_tokens is not None and (not args.tokenizer_dir or not 0 < args.reserve < args.context_tokens):
        parser.error("budget checking requires a tokenizer and 0 < reserve < context-tokens")
    rows = load_retrieval(args.input, args.manifest, args.scope)
    tokenizer = None
    if args.tokenizer_dir:
        if not args.tokenizer_dir.is_dir():
            parser.error("--tokenizer-dir must name an existing local directory")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir.resolve()), local_files_only=True,
                                                   trust_remote_code=False)
    summary = profile(rows, tokenizer, args.context_tokens, args.reserve)
    summary.update(scope=args.scope, split="development", stage="cpu_qa_prompt_profile",
                   input_sha256=sha256_file(args.input), manifest_sha256=sha256_file(args.manifest))
    if args.tokenizer_dir:
        from transformers import __version__
        import tokenizers
        summary["tokenizer"] = {
            "class": type(tokenizer).__name__, "transformers": __version__, "tokenizers": tokenizers.__version__,
            "files_sha256": {p.name: sha256_file(p) for p in sorted(args.tokenizer_dir.iterdir())
                             if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model"}},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({k: summary[k] for k in ("scope", "planned", "lengths")}))


if __name__ == "__main__":
    main()
