# Validation

Initial preparation used numpy 1.26.4 and rank-bm25 0.2.2, Python 3.11, CPU only. Finance and Technology each have 30,000 messages; the question counts are 214 and 213. All 90 development questions completed top-10 BM25 retrieval; 337 evaluation questions remain reserved.

The reorganized repository is tested separately before delivery. No QA, judge, dense embedding or GPU experiment is claimed complete. Repository access/push requires the user's GitHub authentication; a local prepared repository is not evidence of a successful remote push.

Clean-copy validation: copied only Git-staged files into an empty directory, restored the 14 pinned dataset files with checksum verification, passed all 6 tests, and reran all 90 development queries. Retrieved messages, ordering, scores and formatted passages matched the original preparation exactly. No model calls.
