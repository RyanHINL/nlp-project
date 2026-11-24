# Development QA and evaluation

Contributor B consumes one retrieval-v1 JSONL per method. Run commands from the team repository root. The fixed development manifest contains 90 questions. The question split shares conversation history within a domain. All scopes below select development UIDs only.

## Inputs and CPU checks

Use Python 3.11+ on Linux or macOS. QA, judge, reporting and the default preflight use the standard library. The existing CPU requirements are needed for BM25 and the preparation tests. The optional token profile uses a separate tokenizer environment. A model server is needed only for generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m scripts.fetch_data
mkdir -p runs results
if [ ! -e runs/bm25_development.jsonl ]; then
  python -m retrieval.bm25_retrieve --manifest configs/manifest.json \
    --output runs/bm25_development.jsonl --split development --top-k 10
fi
python -m unittest discover -s tests -p 'test_*.py'
QA_EVAL_VERIFY_FULL_REUSE=1 python -m unittest discover -s evaluation/tests -p 'test_*.py'
python -m evaluation.preflight --scope sample --input examples/bm25_development_sample.jsonl \
  --output results/bm25_sample_profile.json
python -m evaluation.preflight --scope pilot --input runs/bm25_development.jsonl \
  --output results/bm25_pilot_profile.json
```

Prefer an existing full retrieval artifact. For a checked-out upstream dataset, supply `--source-dir /path/to/GroupMemBench` to the fetcher. Restore a missing input once. The profiler writes a new file, so use a fresh output path for another inspection. `sample` labels a development subset such as the two Finance/multi_hop examples. Both `pilot` and `development` validate the full 90-row input before selection. The pilot picks the first UID in committed manifest order for every (domain, qtype) pair, yielding 12.

The reader checks UID, split, domain, type, method, question and asker identity, required v1 fields and contiguous ranks. It accepts explicit empty retrieval. `development_identities.json` contains only development UIDs and SHA256 digests of the question/asker pair. Its regeneration test uses the pinned question files. QA can validate identity offline without opening those files. It sends the question, optional asker and the numbered, unmodified `formatted_passage` strings in source order. The judge joins by full UID, verifies source file hashes and identity, then sends question, reference and parsed agent answer in a separate request.

## Candidate configuration for joint review

Use one fixed local instruction model and tokenizer across methods. A candidate for both roles is Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`. Sharing one checkpoint reduces deployment cost. The pilot should test judge reliability independently, particularly for refusals and paraphrases. We measured this tokenizer on the exact BM25 prompts on CPU. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/README.md)

| Proposed setting | Value |
|---|---|
| Total context, including reserved completion | 4,096 tokens per request |
| Completion reserve | 512 QA, 256 judge, matching the upstream limits |
| Decoding | temperature 0.2, top_p 1.0, seed 0, one completion |
| Overflow | Keep every passage intact, record over_budget before generation |
| Timeout | 120 seconds per HTTP request |
| Planned paid calls | Zero. Use the local server only. |

The longest of the 90 BM25 prompts used 2,231 tokens with the candidate chat template. With the QA reserve it needs 2,743 tokens. Profile each additional retrieval condition before freezing the common budget. Twelve pilot questions require at most 24 completions and 9,216 generated tokens. Completing all 90 with pilot reuse requires at most 180 completions and 69,120 generated tokens for one method. Throughput and wall time await a team deployment.

For exact CPU counts, place the tokenizer files from the specified revision together in an ignored directory. The profile includes their hashes and the exact chat template hash. An existing Python environment with Transformers 5.13.0, tokenizers 0.22.2 and Jinja2 3.1.6 produced the handoff result. The following commands reproduce that measurement in an isolated environment.

```bash
python3 -m venv cache/tokenizer-env
cache/tokenizer-env/bin/python -m pip install 'transformers==5.13.0' 'tokenizers==0.22.2' 'jinja2==3.1.6'
mkdir -p cache/qwen2.5-7b-tokenizer
curl --fail --location --silent --show-error --remote-name-all \
  --output-dir cache/qwen2.5-7b-tokenizer \
  'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/{tokenizer.json,tokenizer_config.json,config.json}'
cache/tokenizer-env/bin/python -m evaluation.preflight --scope development \
  --input runs/bm25_development.jsonl --tokenizer-dir cache/qwen2.5-7b-tokenizer \
  --context-tokens 4096 --reserve 512 --output results/bm25_development_qwen4096.json
```

## Real model stages after agreement

Agree on the served aliases, immutable weight and tokenizer revisions, precision/quantization, chat template hash, vLLM build and effective server generation defaults. Register the job and shared read-only model cache with the team. Keep the concrete configuration and launch settings together in the ignored run directory. The template has required placeholders. The client checks the served alias and the rendered prompt token count. The deployment owner records the loaded revisions and chat template. Use a loopback IP in the job or a loopback tunnel to the team server. The client uses the server's chat-aware `/tokenize` and `/v1/chat/completions` endpoints without external credentials, proxies or redirects.

```bash
mkdir -p runs/qa-eval
cp -n evaluation/config.example.json runs/qa-eval/local.json
# Fill the agreed deployment settings before running the next commands.
INPUT=runs/bm25_development.jsonl
CFG=runs/qa-eval/local.json
OUT=runs/qa-eval
python -m evaluation.runner qa --scope pilot --input "$INPUT" --config "$CFG" --output "$OUT/pilot.qa.jsonl"
python -m evaluation.runner judge --scope pilot --input "$INPUT" --config "$CFG" --qa "$OUT/pilot.qa.jsonl" --output "$OUT/pilot.judge.jsonl"
python -m evaluation.report --scope pilot --input "$INPUT" --judge "$OUT/pilot.judge.jsonl" \
  --data-dir data/GroupMemBench --output-dir results/pilot
```

Read all 12 pilot cases against the original ranked passages, gold, agent final and judge verdict. Record the review before launching the remaining questions. The input-only notes below provide message locations to start from. With identical conditions, promote the pilot results in both stages.

```bash
python -m evaluation.runner qa --scope development --input "$INPUT" --config "$CFG" \
  --reuse "$OUT/pilot.qa.jsonl" --output "$OUT/dev.qa.jsonl"
python -m evaluation.runner judge --scope development --input "$INPUT" --config "$CFG" \
  --qa "$OUT/dev.qa.jsonl" --reuse "$OUT/pilot.judge.jsonl" --output "$OUT/dev.judge.jsonl"
python -m evaluation.report --scope development --input "$INPUT" --judge "$OUT/dev.judge.jsonl" \
  --data-dir data/GroupMemBench --output-dir results/development
```

For an interrupted command, repeat that stage with the same output and `--resume` in place of `--reuse`. All planned UIDs are checkpointed atomically. Only pending calls are made. An in-flight call with unknown delivery becomes `interrupted`. The checkpoint preserves completed calls and technical failures. Reusing pilot results checks source, request, model settings, code and prompt hashes. Judge reuse also checks the originating QA context and parsed answer. For an independent rejudge, use the existing QA file, a fresh judge output and the documented judge configuration. Rerunning the report uses no model.

## Scoring and review artifacts

The pinned parser takes the last marked final line, falling back to the last nonempty line. The online judge parser recognizes substrings. The final score uses the vendored summarizer. It removes all non-ASCII letters, converts to lowercase and requires equality with `correct`. For example, `Correct, indeed!` is online-positive and a final miss. A substantive refusal passes through to the judge. Empty output, an empty final, timeout, truncation, overflow and an interrupted call remain visible. An empty judge final is an official miss. A nonempty unrecognized verdict is also a miss and retains its parse status.

- `summary.json` gives counts and micro-accuracy overall, for both domains and for all six types. It records planned denominators, status counts, empty judges, UID-level online/final disagreements and integrity errors. Missing, duplicate, corrupt, foreign, mixed-condition or unfinished results mark the run incomplete. Official accuracy is then null, with correct-over-planned available for diagnosis.
- `failures.jsonl` includes every official miss and unresolved technical case with its question, predicted/reference answers when available, verdict, locators and verbatim excerpts. `review.jsonl` includes every pilot case and all full-run failures.
- Copy `review.jsonl` and edit its `review` block. Set `status` to `reviewed`, choose `category` from `missing_retrieval_evidence`, `speaker_or_group_attribution`, `time_update_or_reply`, `sufficient_evidence_qa_error`, `suspected_judge_error`, `technical_error`, `undetermined` or `no_issue`. Describe the evidence and remaining uncertainty in `explanation`. Use `evidence` entries with `source="retrieved"`, `domain`, `msg_node`, `rank` and `quote`. For post-hoc original conversation evidence use `source="conversation"` and omit rank. Rerun the same report with `--reviews PATH --data-dir data/GroupMemBench` and a fresh output directory. It checks the case fingerprint and quotes, preserves the official score and lists disputed judge decisions separately.

Use the request/response records under `runs/` to audit token usage, call times, status and exact prompts. Keep individual outputs, reference answers, datasets and logs off Git, apart from the small attributed evidence examples. A speaker/role match alone cannot establish that a passage supports a reference. Human evidence review is case-based because the dataset has no official supporting-message IDs.

[CPU handoff](handoff.md) · [12 pilot input observations](pilot_notes.md)
