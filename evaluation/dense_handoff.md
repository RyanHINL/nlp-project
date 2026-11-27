# Dense development handoff

Compare official answers for the same 90 development UIDs (Finance 44, Technology 46). Report four counts overall, by domain and by question type: **both correct**, **BM25 only**, **Dense only**, **both wrong**. The change in accuracy in percentage points is 100 × (Dense only − BM25 only) / planned questions.

**Contributor A supplies**

- The single immutable Dense retrieval-v1 JSONL, its path and SHA256, and the exact `method`.
- The retrieval commit, producing command, embedding and tokenizer revisions, index and query settings, and SHA256 of the source-data manifest.
- The input uses the committed development manifest and the shared top‑10 limit. Contributor B uses the frozen [BM25 deployment](pilot/README.md#inputs-and-deployment).

At the 27 November 2026 handoff check, the published setup base was `523e2bd` and PR #1's `qa-eval` branch was `5da16df`. The BM25 development artifact was available locally. Use A's supplied Dense artifact and producing revision when they are handed over.

## Validate the new input

Follow [retrieval-v1](../docs/INTERFACE.md). Check question/asker identity, UID set, single method, development split and contiguous ranks. Record the configured top‑10 limit, observed length distribution and explicit empty lists. Pass each original ordered `formatted_passage` through unchanged. Resolve an unexpected retrieval count with A. The shared conversation histories and question-level split stay fixed.

```bash
: "${DENSE_INPUT:?Set DENSE_INPUT to the artifact path supplied by Contributor A}"
PY=.venv/bin/python
MODEL_RUN=runs/bm25-real-pilot-001
CFG="$MODEL_RUN/local.json"
OUT=runs/dense-real-development-001
RESULT=results/dense-real-development-001
mkdir -p "$OUT"
shasum -a 256 "$DENSE_INPUT" configs/manifest.json configs/data_sources.json "$CFG"
cache/vllm-metal-0.34.0/bin/python -m evaluation.preflight \
  --scope development --input "$DENSE_INPUT" --tokenizer-dir cache/qwen2.5-7b-a09a354 \
  --context-tokens 4096 --reserve 512 --output "$OUT/preflight.json"
```

Before generation, match the profile’s `input_sha256` and `method` to A’s receipt. The Dense method must differ from `official_bm25_retrieval`. Consume the handed-off JSONL in the same evaluation checkout used for BM25. The preflight uses the frozen [Metal runtime and tokenizer](pilot/README.md#inputs-and-deployment) locally without generation. Check `method`, `planned`, `passage_count_histogram`, `chat_template_sha256` and `over_budget_uids` in the profile. Retain overflows as recorded technical outcomes; changing the common context limit would start a new comparison. The frozen BM25 deployment used [Qwen2.5-7B-Instruct BF16 on Metal](pilot/README.md), revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.2, top_p 1, seed 0, QA/judge reserves 512/256, and 120-second timeout. Deploy or revalidate that environment using the linked instructions. The original frozen deployment SHA256 is `15bd331ce54f3356593daa14ee99ed280e11fb1fe4154e139ab60c56d41b01ef`.

## Generate the Dense condition

After the Dense input and allocation are ready, start in a fresh output directory with empty QA and judge caches. Run the fixed twelve first, inspect two cases of each type, then reuse only those same-condition rows for the remaining 78. This yields 90 distinct Dense QA and judge opportunities each. BM25 final answers and judgments stay in their own artifacts. Each stage uses its own request context. The runner records and skips judge generation when QA has a technical failure.

```bash
"$PY" -m evaluation.runner qa --scope pilot --input "$DENSE_INPUT" --config "$CFG" --output "$OUT/pilot.qa.jsonl"
"$PY" -m evaluation.runner judge --scope pilot --input "$DENSE_INPUT" --config "$CFG" \
  --qa "$OUT/pilot.qa.jsonl" --output "$OUT/pilot.judge.jsonl"
"$PY" -m evaluation.report --scope pilot --input "$DENSE_INPUT" --judge "$OUT/pilot.judge.jsonl" \
  --data-dir data/GroupMemBench --output-dir "${RESULT}-pilot"
# After pilot review and authorization to complete this condition
"$PY" -m evaluation.runner qa --scope development --input "$DENSE_INPUT" --config "$CFG" \
  --reuse "$OUT/pilot.qa.jsonl" --output "$OUT/dev.qa.jsonl"
"$PY" -m evaluation.runner judge --scope development --input "$DENSE_INPUT" --config "$CFG" \
  --qa "$OUT/dev.qa.jsonl" --reuse "$OUT/pilot.judge.jsonl" --output "$OUT/dev.judge.jsonl"
"$PY" -m evaluation.report --scope development --input "$DENSE_INPUT" --judge "$OUT/dev.judge.jsonl" \
  --data-dir data/GroupMemBench --output-dir "$RESULT"
```

Record the input/config/output hashes, model revisions, per-stage attempts, token usage, timing, cache hits, overflows and errors. To continue an interrupted stage use the same arguments and output with `--resume` in place of `--reuse`. Preserve terminal and unknown-delivery failures. Keep raw artifacts in the ignored directories.

## Compare completed conditions offline

Use both full inputs and judge artifacts. On another machine, extract the development review ZIP and verify `SHA256SUMS` at its root. It supplies the original BM25 retrieval JSONL at `input/bm25_development.jsonl` (SHA256 `897c312cf623125d4ea2ca9ae4d0bc8b3f55734136c7c3c64711cd2cb11da444`) and its judge rows at `audit/raw/judge.jsonl`. Rebuilding retrieval changes timing fields and the bytes bound to the original QA. The following command rechecks the 90 planned identities and official scores, requires matching stage configurations, and writes one row per UID with the paired outcomes and both ordered message-ID lists. Each row's `domain` qualifies its message IDs. Raw retrieval scores are method-specific. Review gains and losses against the actual messages and both predicted answers. The official denominator is the total number of paired questions. Evidence recall would require separate human annotation.

<details>
<summary>Run the offline paired comparison</summary>

```bash
# When using an extracted and verified review ZIP, set REVIEW_ROOT to its root:
# BM25_INPUT="$REVIEW_ROOT/input/bm25_development.jsonl"
# BM25_JUDGE="$REVIEW_ROOT/audit/raw/judge.jsonl"
BM25_INPUT="${BM25_INPUT:-runs/bm25_development.jsonl}"
BM25_JUDGE="${BM25_JUDGE:-runs/bm25-real-pilot-001/dev.judge.jsonl}"
"$PY" - "$BM25_INPUT" "$BM25_JUDGE" "$DENSE_INPUT" "$OUT/dev.judge.jsonl" "$RESULT/paired-with-bm25.jsonl" <<'PY'
from collections import Counter
from pathlib import Path
import json, sys
from evaluation import contract as c, report

def load(input_path, judge_path):
    summary, cases, sources = report.summarize(input_path, judge_path, 'development', c.MANIFEST, c.DATA)
    if summary['status'] != 'complete' or summary['overall']['total'] != 90:
        raise ValueError(f'Incomplete condition. Inspect {judge_path} and its integrity report.')
    return summary, {r['uid']: r for r in cases}, sources

bm_input, bm_judge, dense_input, dense_judge, destination = map(Path, sys.argv[1:])
bs, bm, bsrc = load(bm_input, bm_judge)
baseline = {
    'method': 'official_bm25_retrieval',
    'input_sha256': '897c312cf623125d4ea2ca9ae4d0bc8b3f55734136c7c3c64711cd2cb11da444',
    'judge_sha256': 'd7477ab914a7128cd6595e5692a1171fd2ba51d4402618d569792556f25e8bdb',
}
if any(bs[key] != expected for key, expected in baseline.items()):
    raise ValueError('BM25 condition differs from the published baseline')
ds, dense, dsrc = load(dense_input, dense_judge)
if bm.keys() != dense.keys() or bs['method'] == ds['method'] or bs['input_sha256'] == ds['input_sha256']:
    raise ValueError('Expected two distinct retrieval conditions on the same UIDs')
for stage in ('qa', 'judge'):
    for key in ('settings', 'config_sha256', 'code_sha256', 'system_prompt_sha256', 'manifest_sha256'):
        if bs['provenance'][stage][key] != ds['provenance'][stage][key]:
            raise ValueError(f'Condition drift in {stage}.{key}')
if bs['provenance']['judge']['data_sources_sha256'] != ds['provenance']['judge']['data_sources_sha256']:
    raise ValueError('Different reference source manifests')

pairs, groups = [], {}
for uid, before in bm.items():
    after = dense[uid]
    b, d = int(before['official_correct']), int(after['official_correct'])
    domain, qtype, _ = uid.split('/')
    for group in ('overall', domain, qtype):
        groups.setdefault(group, Counter())[b, d] += 1
    pairs.append(dict(uid=uid, domain=domain, qtype=qtype, bm25_correct=b, dense_correct=d,
                      bm25_answer=before['agent_answer'], dense_answer=after['agent_answer'],
                      bm25_messages=[h['msg_node'] for h in bsrc[uid]['retrieved']],
                      dense_messages=[h['msg_node'] for h in dsrc[uid]['retrieved']]))
with destination.open('x', encoding='utf-8') as stream:
    for row in pairs:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
print('group,total,both_correct,bm25_only,dense_only,both_wrong,delta_percentage_points')
for group, n in groups.items():
    total = sum(n.values())
    print(group, total, n[1,1], n[1,0], n[0,1], n[0,0],
          round(100 * (n[0,1] - n[1,0]) / total, 2), sep=',')
PY
```

</details>
