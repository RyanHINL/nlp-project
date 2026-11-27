# Project 12: Speaker- and Group-Aware Memory

A two-person course project using GroupMemBench. Current scope: Finance and Technology, BM25 and local dense-retrieval baselines, followed by one focused metadata-aware retrieval change informed by development-set failures.

## Status
- Official source pinned at `461f00c3e8a1a4a3a2ed5814b729014502c4702f`.
- 60,000 messages and 427 questions audited; fixed domain/type-stratified split: 90 development / 337 evaluation.
- All 90 development questions completed BM25 top-10 retrieval in the initial preparation. Two real retrieval records are included in examples/; they contain no QA results.
- The local QA and judge use the same frozen model across retrieval methods. The [BM25 development baseline](evaluation/bm25_development.md) covers the 90 questions, including the [original pilot](evaluation/pilot/results.md). The Dense development input is pending delivery. [Evaluation commands](evaluation/README.md) · [Dense handoff](evaluation/dense_handoff.md)

## Layout and ownership
```text
retrieval/   # Contributor A: BM25, dense retrieval, proposed improvement
evaluation/ # Contributor B: QA, judge, metrics and failure analysis
configs/     # Shared fixed split and upstream data hashes
scripts/     # Dataset setup; future HPC launchers
docs/       # Interface, handoff and experiment notes
tests/       # CPU verification
vendor/      # Small pinned upstream code/prompts/licenses; no datasets
examples/    # Small real BM25 development samples
```

## CPU quick start
Python 3.11 is locally tested. From repository root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python -m scripts.fetch_data
python -m unittest discover -s tests -p 'test_*.py'
mkdir -p runs
python -m retrieval.bm25_retrieve --manifest configs/manifest.json --output runs/bm25_development.jsonl
```
The downloader needs `curl` and verifies the pinned SHA256 of every file. If you already have the official checkout, use `python -m scripts.fetch_data --source-dir /absolute/path/to/GroupMemBench`. Datasets live in ignored data/; they are not checked into this repository.

## Collaboration
Each person clones the repository into their own directory, including on HPC. Use retrieval and qa-eval feature branches and merge via PR. Do not have two agents modify the same checkout. Both agents follow AGENTS.md. See docs/INTERFACE.md and docs/HANDOFF.md for the agreed contract and first tasks.

The question split is at question level; the two splits share each domain's conversation history. It is not a new-conversation generalization test. Do not reshuffle after viewing outcomes.

## Model and evaluation constraints
The upstream defaults use GPT-5 and OpenAI embeddings. This project must adapt the same QA/judge procedure to fixed locally deployed models, with no paid APIs. Record model/revision, prompt, decoding, context budget, token use and failures. Preserve the official prompt/scoring methodology and disclose the model change; do not claim reproduction of original paper scores.

No gold supporting-message IDs exist in the released question files. Recall@k needs additional evidence annotation; do not invent it. Metadata-aware retrieval must actually affect selection/ranking, not merely add metadata after retrieval.

## Licensing and attribution
Vendored upstream code is MIT; see vendor/GroupMemBench/LICENSE. Conversation/question data and the small derived retrieval examples are CC BY 4.0. Attribution: Yang et al., *GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations* (2026), https://github.com/UCSB-NLP-Chang/GroupMemBench . Data hashes and source revision are in configs/data_sources.json. See https://creativecommons.org/licenses/by/4.0/ .
