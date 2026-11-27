"""Offline edge cases for the 90-question accounting ledger."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation import development_audit as audit


class DevelopmentAuditTests(unittest.TestCase):
    def test_duplicates_reject_both_and_foreign_data_not_exposed(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "dev.qa.jsonl"
            path.write_bytes(b'{"uid":"a"}\n{"uid":"a"}\n\n{"uid":"only-foreign"}\n{"uid":"b"}\n')
            issues = []
            rows, digest = audit.scan(path, {"a", "b", "c"}, issues)
            self.assertEqual(rows, {"b": (5, {"uid": "b"})})
            self.assertEqual(issues[-1]["uids"], ["a", "c"])
            self.assertNotIn("only-foreign", str(issues))
            self.assertEqual(len(digest), 64)

    def test_promotion_checks_request_response_and_duration(self):
        prior = {"cache_key": "same", "response": {"id": "1"}, "seconds": 7.5,
                 "provenance": {"source_qa_sha256": "pilot-qa"}, "audit_sha256": "original"}
        current = dict(prior, provenance={"source_qa_sha256": "dev-qa"}, reused_from_sha256="pilot-judge")
        records = {"uid1": (1, {"judge": prior})}
        audit.compare_cache("uid1", "judge", current, records, "pilot-judge")
        for field in ("response", "seconds", "cache_key", "reused_from_sha256"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.compare_cache("uid1", "judge", dict(current, **{field: None}), records, "pilot-judge")

    def test_missing_usage_or_time_never_counted_as_observed_zero(self):
        rows = [
            {"qa": {"status": "ok", "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                    "seconds": 2.5, "response": {"id": "x", "choices": [{"finish_reason": "stop"}]}, "format_status": "marked"}},
            {"qa": {"status": "timeout", "usage": None, "seconds": 6, "response": None, "format_status": "empty"}},
            {"qa": {"status": "call_error", "usage": "invalid server usage", "seconds": 1, "response": {},
                    "format_status": "empty"}},
            {"qa": {"status": "pending", "usage": None, "seconds": None, "response": None, "format_status": "empty"}},
        ]
        counts = audit.cost(rows, "qa")
        self.assertEqual((counts["records"], counts["attempts"], counts["missing_usage_on_attempts"]), (4, 3, 2))
        self.assertEqual((counts["recorded_prompt_tokens"], counts["recorded_completion_tokens"]), (3, 4))
        self.assertEqual(counts["recorded_client_seconds"], 9.5)
        self.assertEqual(counts["unknown_delivery_or_completion"], 1)
        self.assertEqual(counts["finish_reason"], {"stop": 1})

    def test_access_log_reconciliation_separates_health_checks_and_failed_calls(self):
        with TemporaryDirectory() as temp:
            log = Path(temp) / "dedicated.log"
            log.write_text('\n'.join([
                '"GET /v1/models HTTP/1.1" 200 OK', '"POST /tokenize HTTP/1.1" 200 OK',
                '"POST /v1/chat/completions HTTP/1.1" 503 Failed', '"POST /v1/chat/completions HTTP/1.1" 200 OK',
            ]))
            summary = audit.logged_requests(log)
            self.assertEqual((summary["chat_posts"], summary["tokenize_posts"], summary["non_200_model_posts"]), (2, 1, 1))
            self.assertEqual(summary["http_status_counts"]["GET /v1/models 200"], 1)

    def test_format_observations_use_only_last_marker_and_distinguish_absent_raw(self):
        samples = [None, "  ", "Answer only", "Final: first\nmore\n   FINAL ANSWER: kept\n  ",
                   "  FiNaL aNsWeR: first item\n\nsecond item", "Final:\nitem",
                   "the final: text in this sentence\nlast line"]
        for stage in audit.STAGES:
            rows = []
            for idx, raw in enumerate(samples):
                _, answer = audit.c.split_reasoning_and_final(raw or "")
                rows.append({"uid": str(idx), audit.runner.ANSWER_FIELD[stage]: answer,
                             stage: {"raw_output": raw, "format_status": audit.c.answer_format(raw or "")}})
            actual = audit.format_diagnostics(rows, stage)
            self.assertEqual(actual["missing_raw"]["uids"], ["0"])
            self.assertEqual(actual["empty_raw"]["uids"], ["1"])
            self.assertEqual(actual["empty_extracted"]["uids"], ["1","5"])
            self.assertEqual(actual["fallback"]["uids"], ["2","6"])
            self.assertEqual(actual["nonempty_lines_after_last_final_marker"], {"count": 2, "uids": ["4","5"]})
