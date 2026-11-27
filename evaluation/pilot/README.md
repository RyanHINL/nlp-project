# Reproduce the BM25 development pilot

The completed pilot has a 12-row case inventory, per-type scores and checksums for the model outputs and portable review packet. [Pilot results](results.md)

Run from the repository root on native Apple Silicon macOS with CPython 3.12. The approved deployment uses Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28` for both weights and tokenizer. QA and judge use independent requests to the same local model. The model has four native BF16 safetensors shards totaling 15,231,271,888 bytes. The project used a 32 GiB Apple M5 host.

## Inputs and deployment

Restore the verified data and BM25 input using the [evaluation instructions](../README.md#inputs-and-cpu-checks). Use the same 90-row read-only JSONL for all phases. The original input had SHA256 `897c312cf623125d4ea2ca9ae4d0bc8b3f55734136c7c3c64711cd2cb11da444`. Rebuilding BM25 can change timing fields and the file digest. Reusing existing QA or judge outputs requires the original input bytes.

[deployment.json](deployment.json) lists all nine model filenames, immutable object IDs or SHA256 digests, the two release-wheel hashes, the runtime versions, environment variables and exact server argument vector. The launcher independently checks every shard's dtype, its index, the model configuration, and the chat template extracted from the verified tokenizer. Its UTF-8 byte digest must be `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`. Frozen package versions, model SHA256s, rendered deployment config and template are written under `runs/bm25-real-pilot-001/`.

The runtime uses vLLM 0.34.0+cpu, vllm-metal 0.34.0 and MLX 0.32.4. The plugin runs inference on Metal with native sampling enabled. The CPU wheel supplies the vLLM integration. The plugin supplies its exact torch, torchvision and torchaudio pins. The launch uses vLLM generation defaults, seed 0, a 4,096-token model window, one concurrent sequence, a 4,096-token batch limit, a 536,870,912-byte KV cache, BF16, and HTTP on `127.0.0.1:8000`. Decoding uses temperature 0.2, top_p 1.0 and one completion. QA reserves 512 tokens, judge reserves 256. The checked-in launcher derives the existing runner config with these values and a 120-second HTTP timeout.

```bash
python3.12 -m evaluation.pilot.deployment check
python3.12 -m evaluation.pilot.deployment prepare
python3.12 -m evaluation.pilot.deployment freeze
python3.12 -m evaluation.pilot.deployment serve
```

`check` reads the spec and any existing candidate tokenizer files offline. `prepare` installs into a separate ignored environment and downloads the pinned weights into `cache/`. `freeze` verifies the complete deployment and creates a new run directory. `serve` revalidates it and writes a separate server log. Leave this terminal running. For an existing frozen run use `serve` with the matching environment. Preserve the original frozen files, raw stage artifacts and BM25 input to continue or audit that run. The deployment fingerprint includes platform and installed-package evidence. A fresh environment can produce a different fingerprint even with the same model revision. Give that deployment its own run and review its comparability before combining results.

## First case and remaining eleven

In another terminal at the repository root, use the Python environment that runs the existing evaluation tools.

```bash
PY=.venv/bin/python
INPUT=runs/bm25_development.jsonl
OUT=runs/bm25-real-pilot-001
CFG="$OUT/local.json"
"$PY" -m evaluation.pilot.first_case plan --config "$CFG" --input "$INPUT"
"$PY" -m evaluation.pilot.first_case run --config "$CFG" --input "$INPUT" --output-dir "$OUT/first"
```

The first case is `Finance/multi_hop/multi_hop_4`. Inspect its QA and judge request, raw answer, parsed final, finish reason, token counts and timing in `first.qa.jsonl` and `first.judge.jsonl`. The immutable `first.qa.snapshot.jsonl` also records the other eleven as pending. `inspection.jsonl` records protocol flags and artifact hashes. A valid Correct or Incorrect verdict can pass the protocol check. Resolve any technical or format discrepancy before continuing. An interrupted delivery or terminal error stays in its checkpoint.

```bash
"$PY" -m evaluation.runner qa --scope pilot --input "$INPUT" --config "$CFG" \
  --reuse "$OUT/first/first.qa.jsonl" --output "$OUT/pilot.qa.jsonl"
"$PY" -m evaluation.runner judge --scope pilot --input "$INPUT" --config "$CFG" \
  --qa "$OUT/pilot.qa.jsonl" --reuse "$OUT/first/first.judge.jsonl" --output "$OUT/pilot.judge.jsonl"
"$PY" -m evaluation.report --scope pilot --input "$INPUT" --judge "$OUT/pilot.judge.jsonl" \
  --data-dir data/GroupMemBench --output-dir results/bm25-real-pilot-001
```

Each successful phase reuses one first-case record and generates eleven additional completions. Review all twelve rows in `review.jsonl`, including refusals and strict/online judge discrepancies, using [the preselected evidence notes](../pilot_notes.md). Store per-UID annotations and raw logs with the run. For an interrupted pilot stage, repeat the same command with `--resume` in place of `--reuse`, keeping the existing output path. A distinct retry after a verified technical failure requires a fresh recorded attempt.

## Offline review packet

Run this after both stages and the ordinary offline report. Use a new output directory and ZIP path each time. This cross-checks the exact prompts, per-UID selection, frozen deployment, token counts, references and stage lineage. The ZIP contains the original requests and responses in per-case JSON files, the official score, evidence locators and editable human notes. Its links work after extraction.

```bash
"$PY" -m evaluation.pilot.review_export --scope pilot --run-dir "$OUT" \
  --report-dir results/bm25-real-pilot-001 \
  --output-dir results/bm25-pilot-review-initial --zip results/bm25-pilot-review-initial.zip
```

For the annotated packet, a copy of `notes.jsonl` holds each case fingerprint, the preliminary assessment and literal evidence excerpts. `human_notes` remains empty until an assigned reviewer fills it. Pass the verified data directory to check post-hoc conversation quotations. The original annotated input on the author's machine is `results/bm25-analysis-working/pilot.notes.jsonl`. The delivered ZIP also contains those notes.

```bash
"$PY" -m evaluation.pilot.review_export --scope pilot --run-dir "$OUT" \
  --report-dir results/bm25-real-pilot-001 --data-dir data/GroupMemBench \
  --notes results/bm25-analysis-working/pilot.notes.jsonl \
  --output-dir results/bm25-pilot-review --zip results/bm25-pilot-review.zip
```

## Later promotion to 90 development questions

Once the twelve cases have been reviewed and the team authorizes expansion under the same frozen condition, reuse the pilot artifacts in both stages. Keep the original model, environment, complete config, prompts, retrieval file, execution code and manifest. The runner checks the saved fingerprints and emits a distinct output. With twelve completed pilot records each stage generates at most 78 additional completions.

```bash
"$PY" -m evaluation.runner qa --scope development --input "$INPUT" --config "$CFG" \
  --reuse "$OUT/pilot.qa.jsonl" --output "$OUT/dev.qa.jsonl"
"$PY" -m evaluation.runner judge --scope development --input "$INPUT" --config "$CFG" \
  --qa "$OUT/dev.qa.jsonl" --reuse "$OUT/pilot.judge.jsonl" --output "$OUT/dev.judge.jsonl"
"$PY" -m evaluation.report --scope development --input "$INPUT" --judge "$OUT/dev.judge.jsonl" \
  --data-dir data/GroupMemBench --output-dir results/bm25-real-development-001
```

Each report directory must be new. Development contains 90 questions. The separate 337 evaluation UIDs have no execution path in these commands.
