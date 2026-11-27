# BM25 development baseline

The local Qwen judge scored **30/90 (33.3%)** under the pinned GroupMemBench scoring rule. The 90 cases are the complete committed development split. The 337 evaluation questions remain untouched. Question-level splitting gives development and evaluation questions in the same domain access to a shared conversation history.

## Scores by domain and question type

| Group | Correct | Planned | Accuracy |
|---|---:|---:|---:|
| Overall | 30 | 90 | 33.3% |
| Finance | 16 | 44 | 36.4% |
| Technology | 14 | 46 | 30.4% |
| multi_hop | 4 | 19 | 21.1% |
| knowledge_update | 8 | 15 | 53.3% |
| temporal | 3 | 17 | 17.6% |
| user_implicit | 3 | 9 | 33.3% |
| term_ambiguity | 1 | 18 | 5.6% |
| abstention | 11 | 12 | 91.7% |

The overall denominator is the 90 planned questions. The pilot's 5/12 describes a subset selected before model use. Extending the sample changed the denominator and retained the same retrieval and model conditions.

## Integrity and cost

The run started from qa-eval `5da16dfe92f0391ec565fe4cfc12f079228d1adc`. Each raw row records the frozen execution code and system prompt fingerprints. The original 90-row retrieval JSONL was read without rebuilding BM25. Its question order on disk differs from the manifest. The join uses complete UIDs and the output ledger follows manifest order. All 14 source-data files matched the pinned checksums. Each of the 900 ordered passages was supplied as stored. Both roles ran Qwen/Qwen2.5-7B-Instruct with the weights and tokenizer at revision `a09a35458c702b33eeacc393d103063234e8bc28`, native BF16 on the Apple M5 Metal GPU, and one active sequence. The deployment used the pilot chat template, official prompts, seed 0, temperature 0.2, top_p 1 and the 4,096-token intact-input budget. QA reserves 512 output tokens and judge reserves 256. [Deployment and model revision](pilot/README.md)

All 90 QA and 90 judge records succeeded. The audit found 90 distinct expected UIDs, 180 distinct response IDs, and zero missing, duplicate, broken or unfinished records. Each stage reused the exact 12 pilot requests and responses and added 78 completions. The dedicated server access log recorded 156 successful new chat requests and 156 successful tokenizer checks, with zero generation retries, non-200 responses, timeouts or overflows. Every completion finished with `stop`. All 90 actual chat inputs agreed with the deployed tokenizer for each stage. QA inputs used 1,524–2,231 tokens and judge inputs, constructed after QA, used 99–208.

Every response had a final marker. There were zero fallback, empty, length-truncated or malformed final verdicts, and online versus strict verdicts differed on zero UIDs. Two QA responses had nonblank text below the official final line. Technology `user_implicit_21` lost the three operator actions during official extraction. Technology `temporal_14` retained its date but lost a following explanation. Neither response was regenerated. The pinned upstream scorer reproduced the counts for all six types using `--baselines official_bm25_retrieval`. The six preparation and thirty evaluation CPU tests passed separately from inference.

| Charge scope | Stage | Actual generation responses | Input tokens | Output tokens | Sum of client seconds |
|---|---|---:|---:|---:|---:|
| This run | QA | 78 | 144,836 | 8,094 | 1,233.83 |
| This run | JUDGE | 78 | 10,728 | 6,717 | 930.37 |
| Prior pilot | QA | 12 | 22,359 | 1,300 | 210.60 |
| Prior pilot | JUDGE | 12 | 1,744 | 1,069 | 142.94 |
| Full baseline | QA | 90 | 167,195 | 9,394 | 1,444.43 |
| Full baseline | JUDGE | 90 | 12,472 | 7,786 | 1,073.31 |

Both stages together consumed 155,564 input and 14,811 output tokens this run (2,164.20 client seconds). The full baseline consumed 179,667 input and 17,180 output tokens (2,517.73 client seconds). Durations sum the client's tokenization and completion time per request. A cache hit incurs zero new generation. The row retains the original call time and token usage, so the historical 12 are charged once in the 90-question totals. Paid API calls and API fees were zero. Electricity consumption was not metered.

## Evidence review

Codex reviewed all 90 question, answer, reference and judge records. The queue contains all 60 official misses and 5 of the 30 official successes. Its 65 entries include suspected over- and under-awards, wording caveats and conflicting evidence. All 90 human verdict fields are pending. The earlier independent Codex review considered 6 of the 12 pilot answers correct, differing from the local judge on one question. The prior rationale and case identity are recorded separately.

Evidence depth was recorded per question. For 43 questions the review checked the full ten and the original conversation. Another 13 received a full top-ten review, and the remaining 34 were checked against selected ranked evidence. There are 206 verified excerpt entries with domain, message ID, rank or original location, author, channel, phase, timestamp and reply parent. Missing-evidence labels require checking all ten and locating direct evidence in the original conversation outside the ten. All case labels are provisional and may overlap.

| Evidence-review label | Among 60 official misses | Among 30 official successes |
|---|---:|---:|
| Necessary information found outside the supplied ten | 38 | 0 |
| Sufficient retrieved evidence with a QA omission or mismatch | 15 | 3 |
| Speaker, channel, subject or action assignment | 11 | 0 |
| Time, update or reply relation | 17 | 0 |
| Potential semantic judging error | 6 | 1 |
| Judge explanation contradicted or unsupported by the record | 5 | 1 |
| Ambiguous reference, referent or competing instructions | 14 | 0 |
| Accepted answer with a wording or detail caveat | 0 | 2 |
| Technical or final-answer extraction issue | 1 | 0 |
| Insufficient evidence to assign a cause | 0 | 0 |
| Screened with no specific disagreement flagged | 0 | 25 |

The source-confirmed gaps identify candidates for a retrieval comparison, especially when a returned reply lacks the parent or the requested channel is absent. Rows with a direct answer already in the ten also call for examining answer extraction or use of context. Review disagreements and ambiguity before interpreting individual gains or losses. These counts describe this question-level development sample. The dataset supplies no official supporting-message labels, and the same model produced and judged the answers.

| Development case | Evidence and interpretation |
|---|---|
| Finance, `temporal_11` | Rank 3, Msg_9543, links to a parent and mentions owners marking gaps. Its date instruction is outside the ten. The original parent Msg_8800 on 14 July asks owners to mark gaps “by tomorrow”, supporting 15 July. QA declined. A parent link can survive retrieval without its content. |
| Finance, `knowledge_update_30` | Rank 7, Msg_3163, says to merge conditionally, validate against the frozen snapshot and require Finance to approve rerun results before release. QA declined despite this direct evidence. The pilot success Technology, `multi_hop_24`, supplied both parts of another rank-7 rule from Msg_27529. |
| Finance, `temporal_19` and `temporal_12` | Each has a 27 July post with a next-day deadline. In the first, QA declined despite rank 1, Msg_11268. In the second it correctly gave 28 July using an answer consistent with rank 2, Msg_28757. Rank alone does not explain the difference. |
| Technology, `user_implicit_21` | Rank 1, Msg_1077, lists three operator actions. The raw QA response listed all three on lines below a marked heading. The official extractor retained only the heading, which was the entire agent answer presented to the judge. |
| Technology, `multi_hop_38` | QA explicitly says “any mixed response auto-holding the freeze”. The correct trigger appears in the gold and rank 3, Msg_20857. The judge marked the answer Incorrect for additional detail. The message also discusses the details. Review whether they change the substantive answer. |
| Technology, `knowledge_update_7` | The judge marked the answer Correct and attributed Security-scoped field redaction to QA. Rank 2, Msg_4719, and the reference require it. The saved final answer describes when to retain origin and omits the redaction rule. This is a possible over-award. |

Five inherited pilot disputes remain in the queue. Technology `knowledge_update_13` lists the surfaces, changes and explicit signoff despite the judge's objection about widening review. Finance `knowledge_update_10` was accepted despite shifting the freeze from the spec to the tracker. For Finance `user_implicit_8`, rank 6 (Msg_12092) gives the reference blocker and the same asker also raised other blockers. For Technology `user_implicit_25`, the refusal differs from the reference and original messages Msg_24664 and Msg_27584 support “contract debt”. The judge's suggestion that the reference might be irrelevant lacks support. For Technology `term_ambiguity_5`, the predicted Friday matches a different channel's action, while original Msg_14460 ties the requested check to the next day, 27 July. These are provisional interpretations. Their human-decision fields remain pending.

## Results on the run owner's Mac

All paths are relative to the `qa-eval` checkout. The portable ZIP includes the untouched BM25 input, the two preflight receipts, the upstream scorer cross-check and byte-identical copies of both stage outputs. It can be copied to another reviewer. Its `SHA256SUMS` covers the included development artifacts.

| Artifact | SHA256 |
|---|---|
| `runs/bm25_development.jsonl` | `897c312cf623125d4ea2ca9ae4d0bc8b3f55734136c7c3c64711cd2cb11da444` |
| `runs/bm25-real-pilot-001/local.json` | `c6632cba8e71b76786dacef178befb45d878872e86440b508b36c21d3a0280a4` |
| `runs/bm25-real-pilot-001/deployment.frozen.json` | `15bd331ce54f3356593daa14ee99ed280e11fb1fe4154e139ab60c56d41b01ef` |
| `runs/bm25-real-pilot-001/dev.qa.preflight.json` | `48362ce018f75503418e1601f2244fae113bc08ce12ac8a10a6b646430979ce4` |
| `runs/bm25-real-pilot-001/dev.judge.preflight.json` | `f171832ddf633066258f7a25625de29be54ce0b24b95a539504a97e2e26c31e5` |
| `runs/bm25-real-pilot-001/dev.qa.jsonl` | `301d5e3c055e97c2c96d7cd938cbb4b4b8ba6b9c9a8f1152d9389053065d2037` |
| `runs/bm25-real-pilot-001/dev.judge.jsonl` | `d7477ab914a7128cd6595e5692a1171fd2ba51d4402618d569792556f25e8bdb` |
| `runs/bm25-real-pilot-001/server-09gqv1wb.log` | `ed09d08f37a22b2faa384dbd65eb82ba54d24f877eaeef5b96d1aba4d841c2b6` |
| `results/bm25-real-development-001/summary.json` | `bbdfac6b400ab5c08a9c9291c9379555937e6d01638798015d523da5a9ff307a` |
| `results/bm25-real-development-001/audit/accounting.json` | `42aa0438ca8b7bb8c2ae4f9fb105afda07c960bc2746ca1da688cf97071360f9` |
| `results/bm25-real-development-001/analysis/cases.jsonl` | `0ef8ac94da9a1808cb5d1669fdef9e3bc7973e785d49dbe483cdd8d1357322e2` |
| `results/bm25-real-development-001/analysis/review_queue.jsonl` | `587798064a6bcf2e018cc1b20faabef9635db3850c66d58b94f6a6ed710a3a0c` |
| `results/bm25-real-development-001-review.zip` | `3cd17c2104bce0d009a7556bd529e3f1262076721b1c72b1dfc4e7e76a72e9ea` |

The normal report contains the strict scores, complete automatic-failure ledger, and the original pending human review fields. The portable audit joins the 90 UIDs to exact QA/judge requests, raw responses, stage errors or finish reasons, source hashes, model settings, token counts and elapsed time. The separate AI review includes the evidence scope, overlapping tags and verbatim domain-qualified citations. The full QA/judge JSONL and raw logs stay outside Git.

## Commands used

Run from the repository root. The server launcher revalidates the existing frozen deployment, including all four weight shards. Define the same shell variables in the service and client terminals.

```bash
PY=.venv/bin/python
OUT=runs/bm25-real-pilot-001
INPUT=runs/bm25_development.jsonl
CFG="$OUT/local.json"
RESULT=results/bm25-real-development-001

"$PY" -m evaluation.pilot.deployment serve
# In the client terminal
HF_HUB_OFFLINE=1 cache/vllm-metal-0.34.0/bin/python -m evaluation.preflight \
  --scope development --input "$INPUT" --tokenizer-dir cache/qwen2.5-7b-a09a354 \
  --context-tokens 4096 --reserve 512 --output "$OUT/dev.qa.preflight.json"
"$PY" -m evaluation.runner qa --scope development --input "$INPUT" --config "$CFG" \
  --reuse "$OUT/pilot.qa.jsonl" --output "$OUT/dev.qa.jsonl"
"$PY" -m evaluation.runner judge --scope development --input "$INPUT" --config "$CFG" \
  --qa "$OUT/dev.qa.jsonl" --reuse "$OUT/pilot.judge.jsonl" --output "$OUT/dev.judge.jsonl"
"$PY" -m evaluation.report --scope development --input "$INPUT" --data-dir data/GroupMemBench \
  --judge "$OUT/dev.judge.jsonl" --output-dir "$RESULT"
# After shutting down the service in its terminal
"$PY" -m evaluation.development_audit --run-dir "$OUT" --report-dir "$RESULT" \
  --server-log "$OUT/server-09gqv1wb.log" --output-dir "$RESULT/audit"
```

The outputs above already exist on the original Mac. For another offline recount, point `evaluation.report` at the original inputs and a new output directory. Resume only a genuinely interrupted generation stage using the same output path and `--resume` in place of `--reuse`. Preserve technical failures and the earlier responses. Preparation and evaluation tests were run separately on CPU.

Contributor A's Dense artifact is pending delivery. Use the same 90 development UIDs and freeze, a distinct input digest and method ID, and independent Dense QA and judge artifacts. Compare both methods by UID after both conditions pass the completeness checks. [Dense input receipt, run commands and paired comparison](dense_handoff.md)
