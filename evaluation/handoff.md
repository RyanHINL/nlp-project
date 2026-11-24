# Contributor B development handoff · 2026-11-24

Branch `qa-eval` is based on the verified team commit `523e2bdfc5dde6df45d2e08755b27ea87f4d5321` on `codex/project12-setup`. At intake this was the remote HEAD and the only published branch. The 39 tracked files matched the supplied snapshot byte-for-byte. Git tree `115a8353a48c740a0a7f76660dfb398683ec2d8e`. The fixed upstream GroupMemBench commit is `461f00c3e8a1a4a3a2ed5814b729014502c4702f`.

## What ran

| Scope | Outcome |
|---|---|
| Two real example inputs | Both Finance/multi_hop examples passed identity, ordering and prompt checks. For their UIDs the rebuilt input matched every example field except retrieval duration. |
| CPU input recovery | Verified 14/14 source SHA256 values and ran the existing BM25 command once for development. All 90 planned UIDs were unique with ten original ranked passages each. Offline references matched question and asker for 90/90. |
| CPU token budget | 90 exact chat inputs measured with the candidate Qwen tokenizer, including the system message and assistant generation prefix. Minimum 1,524, median 1,850, p95 2,088, maximum 2,231 tokens. Total 167,195. Zero would exceed 4,096 with 512 reserved for QA. Quantiles use nearest rank. |
| Pilot evidence inspection | 12 fixed cases, one in each domain/type combination. The cited text, rank and domain-qualified message IDs were checked against the retrieved messages. Human review of predictions and verdicts is pending. |
| Synthetic protocol validation | All 6 preparation tests and 25 evaluation tests passed on CPU. A stub-response test reused the same 12 QA and 12 judge outputs and called each stage for only the remaining 78 UIDs. A separate stub rejudge preserved the QA file hash. Temporary synthetic outputs were discarded. |
| Real model experiment | No QA or judge generation, GPU job or paid call. Pilot predictions, official verdicts, complete-set accuracy and outcome-based failure analysis await the joint configuration. No held-out question was run. |

Development denominators are Finance 44, Technology 46, multi_hop 19, knowledge_update 15, temporal 17, user_implicit 9, term_ambiguity 18 and abstention 12. The official evaluation split retains 337 UIDs. Questions share each domain’s conversation history across splits.

## Reproduction and local artifacts

From the repository root, restore the data with `python -m scripts.fetch_data`. For a missing BM25 input run `python -m retrieval.bm25_retrieve --manifest configs/manifest.json --output runs/bm25_development.jsonl --split development --top-k 10` after creating `runs/`. This checkout used Python 3.12.6, numpy 1.26.4 and rank-bm25 0.2.2 for recovery and tests.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
QA_EVAL_VERIFY_FULL_REUSE=1 .venv/bin/python -m unittest discover -s evaluation/tests -p 'test_*.py' -v
```

| Local path relative to checkout | Contents or SHA256 |
|---|---|
| `configs/manifest.json` | `8bbae5e00e98e6f8972e8ac6b1a1c23d03d9219ca72e0f62c459208e4daf1d72` |
| `configs/data_sources.json` | `c9a38a9e4fe3d5b49aeedbde4fbed75c86e1f571b19bfe4be7da1f9bbd2578c4` |
| `data/GroupMemBench/` | 2 histories and 12 question files. Per-file digests in the source manifest. |
| `runs/bm25_development.jsonl` | `897c312cf623125d4ea2ca9ae4d0bc8b3f55734136c7c3c64711cd2cb11da444` |
| `runs/bm25_development.metrics.json` | CPU indexing and retrieval times from the recovery command |
| `results/bm25_sample_profile.json`, `results/bm25_pilot_profile.json` | Per-UID input lengths for the smoke sample and fixed pilot |
| `results/bm25_development_qwen4096.json` | Full CPU profile. SHA256 `07347321a5ff2c4fe372588f6b540813db20abe3a901141b20d2fe6b1427e1e5` |
| `evaluation/development_identities.json` | Development UID to question/asker digest. File SHA256 `59f4058ae8f38099017b617856fb4c759047c0c17484cf5b4a48c1735e0e5547` |
| `evaluation/pilot_notes.md` | Attributed excerpts and questions for human review |

A fresh retrieval will have different timing fields. For a timing-independent check this run's canonical digest of `[[row["uid"], row["retrieved"]] for row in load_retrieval(path)]` is `37a91fadd2a7b3b9d7009a54fa361a61dfe85858e351c816cbcf65e1cb291056`, using `evaluation.contract.digest_json`. All large files remain in the ignored directories of the local checkout. There is no HPC artifact location yet.

Before real generation, agree on model deployment and the common context budget, inspect all 12 pilot predictions and judge decisions, then use the documented pilot reuse commands. Evaluate alternative retrieval conditions under the same frozen setup. Evidence notes distinguish directly supporting passages, potential distractors and references that accept abstention. Together, the two contributors can use the first reviewed failures to select the next retrieval change.

[Commands and configuration](README.md) · [Pilot evidence](pilot_notes.md)
