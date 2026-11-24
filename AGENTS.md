# Project 12 collaboration rules

This project studies GroupMemBench speaker/group-aware retrieval. Read README.md and docs/INTERFACE.md first.

- Retrieval agent owns retrieval/. Evaluation agent owns evaluation/. Work in separate branches and separate checkouts; do not overwrite another agent's work.
- Shared contracts live in configs/ and docs/INTERFACE.md. Keep the committed question split unchanged. If a shared change is necessary, describe it explicitly in the PR and coordinate it with the other contributor.
- Do not change vendor/ files. They are pinned upstream files with attribution and hashes.
- Never pass reference answers, question-type labels, is_noise or is_decision_point annotations into retrieval or QA. Reference answers are for offline evaluation/judging only.
- The official asking_user_id is allowed input, but is not necessarily the author of the supporting evidence.
- Build/test on development questions. Freeze models, prompts and retrieval/context budgets before evaluation. Keep failures and costs visible.
- No paid model APIs. Do not launch GPU jobs merely to test packaging. Models, credentials and large output files stay out of Git.
- Preserve domain/type/id as the unique question key and return the common retrieval JSONL schema.
- State what actually ran, what passed and what remains untested. Retrieval results are not QA accuracy.

Checks from repository root:
python -m scripts.fetch_data
python -m unittest discover -s tests -p 'test_*.py'
