# Fixed 12-question BM25 development pilot

On 25 November 2026 the local Qwen QA and judge completed all 12 preselected questions. The official strict score was **5/12 (41.7%)**. Finance contributed 3/6 (50.0%) and Technology 2/6 (33.3%). The unit of selection was the question. The two splits share conversation history within each domain. Human review is pending for every case.

| Question type | Correct / planned | Accuracy |
|---|---:|---:|
| multi_hop | 2/2 | 100% |
| knowledge_update | 1/2 | 50% |
| temporal | 0/2 | 0% |
| user_implicit | 0/2 | 0% |
| term_ambiguity | 0/2 | 0% |
| abstention | 2/2 | 100% |

## Execution and integrity

Both roles used `Qwen/Qwen2.5-7B-Instruct` with native BF16 weights, the same immutable tokenizer and its chat template. The model and tokenizer revision was `a09a35458c702b33eeacc393d103063234e8bc28`. The host was an Apple M5 MacBook Air with 32 GiB unified memory, macOS 26.6.2 and Python 3.12.6. The Metal plugin selected the MLX GPU device. Runtime versions included vLLM 0.34.0+cpu, vllm-metal 0.34.0, MLX 0.32.4, torch 2.16.0.dev20261013, Transformers 5.22.2 and tokenizers 0.23.3. The service listened on `http://127.0.0.1:8000` and was shut down after the pilot.

QA received the original ten ordered BM25 passages per question, along with the question and asker. Judge received the question, checked reference and parsed agent answer in its own request. Both used temperature 0.2, top_p 1, seed 0, one completion and one sequence at a time. The 4,096-token limit included the generation reserve. Overflow handling rejected the whole prompt. All actual requests fit. [Deployment, sampling settings and commands](README.md)

| Stage | Unique attempts / OK | Input tokens (range per UID) | Generated tokens | Reserved per UID | Sum of request times |
|---|---:|---:|---:|---:|---:|
| QA | 12/12 | 22,359 (1,705–2,132) | 1,300 | 512 | 210.60 s |
| Judge | 12/12 | 1,744 (111–197) | 1,069 | 256 | 142.94 s |

The first-case protocol check verified the saved requests, responses, parser outputs and token counts before the remaining eleven calls per stage. Each batch reused its first-case result once. The service log recorded 24 successful chat requests. All 24 returned a marked final answer and finish reason `stop`. There were zero missing or duplicate UIDs, malformed records, technical failures, overflows, truncations or generation retries. The deployed QA tokenizer agreed with the offline profile for all twelve. Online and strict verdicts agreed on all twelve, including the disputed content judgment below. All 31 CPU tests passed separately, including synthetic cache-promotion tests. At pilot close, the remaining 78 development questions and all 337 held-out questions had not run. See the [subsequent 90-question development baseline](../bm25_development.md) for the expanded condition.

## Evidence for human review

One question was selected per domain and type using the first development UID in committed manifest order. The table lists every official miss and success. Locations in each row use the domain of the UID. “Original” denotes a separately checked message outside the supplied top ten. The full packet includes verbatim excerpts, speaker, channel, timestamp, reply information, raw requests and responses, and a separate field for human notes. These diagnoses are provisional. [Input observations recorded before inference](../pilot_notes.md)

| Development UID | Official verdict | Observation to check |
|---|---|---|
| `Finance/multi_hop/multi_hop_4` | Correct | Rank 1, Msg_892, gives intent-only review and its control/disclosure exception. The answer applies both to Legal and Risk. |
| `Finance/knowledge_update/knowledge_update_10` | Correct | Rank 1, Msg_3764, supports the logged wording change, Legal approval and current tracker ownership. |
| `Finance/temporal/temporal_32` | Incorrect | QA declined to give a date. The top ten discuss a different handoff. Original Msg_22075 on Monday 21 July asks Compliance to post the threshold owner/date by Friday, 25 July. |
| `Finance/user_implicit/user_implicit_8` | Incorrect | User_3's own rank-6 message Msg_12092 names portfolio, risk and reporting. The answer instead includes concerns expressed by other speakers, including User_13 at rank 1. |
| `Finance/term_ambiguity/term_ambiguity_41` | Incorrect | QA declined to give a date. Rank 2 mentions another task. Original Msg_21394 and its reply Msg_21702 tie the owner's stamp to “today” on 17 July. |
| `Finance/abstention/abstention_18` | Correct | QA declined to supply an exact matched-pair count. Rank 1, Msg_11959, contains a two-match heuristic. |
| `Technology/multi_hop/multi_hop_24` | Correct | The answer gives both parts of rank 7, Msg_27529, including a named owner in the exception lane and dashboards on the last approved schema hash. |
| `Technology/knowledge_update/knowledge_update_13` | Incorrect | QA names the affected surfaces and explicit signoff. Rank 1, Msg_3148, also removes automatic approval. The judge required the expansion of review to be explicit. Review whether QA expressed the same revised policy. |
| `Technology/temporal/temporal_16` | Incorrect | QA declined to give a date. Rank 1, Msg_19501, assigns SRE the Thursday row from a Sunday 27 July timestamp, supporting 31 July. |
| `Technology/user_implicit/user_implicit_25` | Incorrect | QA declined to supply the classification. Rank 9, Msg_28553, mentions missing signals. Original Msg_24664 and Msg_27584 explicitly classify the gap as contract debt. |
| `Technology/term_ambiguity/term_ambiguity_5` | Incorrect | QA answered Friday. The top ten mix project channels. Rank 5, Msg_7644, gives a general 27 July target. Original CodeReviewAgent Msg_14460 on 26 July ties the QA/Engineering edge-case check to the next day. |
| `Technology/abstention/abstention_46` | Correct | QA declined to estimate implementation time. Rank 4, Msg_11952, describes reconciliation mechanics. |

The date and speaker examples distinguish where retrieval supplied a usable relation and where the original conversation contained a more direct message. Answer alignment with a passage cannot establish which passage caused the prediction. The dataset provides no official supporting-message labels. Treat the seven misses as a case list for joint review before selecting a retrieval change.

## Artifact inventory

Paths below are relative to the author's `qa-eval` checkout. Large artifacts and model weights stay outside Git. The portable ZIP can be copied as one file for the other reviewer. Open `cases.md` after extracting it. Verify its internal files with `shasum -a 256 -c SHA256SUMS`. Keep the original packet when adding human notes. The official summary remains separate from annotations.

| Artifact | SHA256 |
|---|---|
| `runs/bm25_development.jsonl` | `897c312cf623125d4ea2ca9ae4d0bc8b3f55734136c7c3c64711cd2cb11da444` |
| `runs/bm25-real-pilot-001/deployment.frozen.json` | `15bd331ce54f3356593daa14ee99ed280e11fb1fe4154e139ab60c56d41b01ef` |
| `runs/bm25-real-pilot-001/local.json` | `c6632cba8e71b76786dacef178befb45d878872e86440b508b36c21d3a0280a4` |
| `runs/bm25-real-pilot-001/pilot.qa.jsonl` | `9efd68334166a203724449edb5aa4e89e5306ebe980abb55ba5fa8bf9a19b744` |
| `runs/bm25-real-pilot-001/pilot.judge.jsonl` | `acdc030f22aa9184948a780947b5b2c012ad84f5566ab688b62dcd0aa9a1331d` |
| `results/bm25-real-pilot-001/summary.json` | `4910043bee073eddee2cde0328a18b38750d7bb67bfc814146af1bad602b20be` |
| `results/bm25-pilot-review.zip` (169,862 bytes) | `752281c3931562c6503490b66c4c0903b982b431c0eab43fc34d60d709092898` |
| `results/bm25-real-pilot-001/failures.jsonl` | `d15865bd20b0a0e7752f09f82fd820108234c99b8d4a64b6698ed2eb4a5c8864` |

The frozen bundle identifies evaluation code at `ec7f0297b390e24ae0878f80d40e63467eb7b7ea` on branch `qa-eval`. Each row links the exact prompt, input, manifest and stage code digests. The deployment and first-case launchers, the portable exporter and this handoff were added on the same branch. The retrieval interface, manifest, algorithms and vendored prompts retain their pinned versions.
