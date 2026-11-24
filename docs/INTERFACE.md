# Retrieval → QA/evaluation interface v1

Transport: one JSON object per line, UTF-8. Stable primary key: `uid=Domain/qtype/original_id`.

Required retrieval record fields:
- uid, domain, qtype, split, method
- question, asking_user_id
- retrieved: ranked list of `{rank, msg_node, channel, speaker, score, formatted_passage}`
- retrieval_seconds, qa_status (initially `not_run`)

`score` is retriever-specific and not comparable across methods. `rank` starts at 1. Fixed baseline top-k is 10. Empty retrieval is valid if explicitly recorded; do not fabricate messages. The shared QA module must format the same fields in the same order for all methods.

Important separation: qtype/split/uid are bookkeeping only. Only the question, legitimate asker identity and retrieved passages go to QA; reference answers and labels do not. See examples/bm25_development_sample.jsonl for actual records.

Evaluation should add agent_answer, judge_answer, judge parsing status, raw request/response references, model/revision, token counts, timings and error status. Obtain reference answers by uid in an offline judge step. Do not expose them to retrieval/QA. Keep invalid judge responses and execution failures visible and report denominators.

Both sides must keep the same manifest, QA/judge models, prompts, top-k and context truncation policy. Agree on an explicit context budget before the first model run. No context policy has yet been chosen or validated.
