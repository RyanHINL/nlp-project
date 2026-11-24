"""Synthetic scoring/integrity fixtures; the two real prompts are used only for identity."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation import report
from evaluation.contract import MANIFEST, ROOT, load_retrieval, parse_judgment, sha256_file


SAMPLE = ROOT / "examples/bm25_development_sample.jsonl"


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "judge.jsonl"
        self.rows = load_retrieval(SAMPLE, MANIFEST, "sample")
        self.provenance = {"retrieval_sha256": sha256_file(SAMPLE), "manifest_sha256": sha256_file(MANIFEST)}
        # Isolate report logic. Runner tests exercise its audit validation using a fake transport.
        patcher = patch.object(report, "validate_result")
        patcher.start()
        self.addCleanup(patcher.stop)

    def judged(self, row, label):
        norm = "".join(c for c in label.lower() if "a" <= c <= "z")
        status = norm if norm in {"correct", "incorrect"} else ("unrecognized" if label else "empty")
        return dict(row, scope="sample", qa={"provenance": self.provenance},
                    qa_status="ok", agent_answer="synthetic abstention",
                    gold_answer="synthetic reference", judge={"provenance": self.provenance},
                    judge_answer=label, judge_online_correct=parse_judgment(label),
                    judge_parse_status=status, judge_status="ok" if label else "empty_output")

    def run_report(self, rows, extra=b""):
        self.path.write_bytes(b"".join((json.dumps(r) + "\n").encode() for r in rows) + extra)
        return report.summarize(SAMPLE, self.path, "sample", MANIFEST)

    def test_official_fallback_online_disagreement_and_offline_repeat(self):
        rows = [self.judged(r, label) for r, label in zip(self.rows, ["C.o.r.r.e.c.t!", "mostly correct"])]
        summary, cases, _ = self.run_report(rows)
        self.assertEqual("complete", summary["status"])
        self.assertEqual({"total": 2, "correct": 1, "accuracy": .5, "correct_over_planned": .5}, summary["overall"])
        self.assertEqual(summary["overall"], summary["by_domain"]["Finance"])
        self.assertEqual(summary["overall"], summary["by_qtype"]["multi_hop"])
        self.assertIsNone(summary["by_qtype"]["temporal"]["accuracy"])
        self.assertEqual(6, len(summary["by_qtype"]))
        self.assertEqual([c["uid"] for c in cases], [d["uid"] for d in summary["online_vs_summary"]])
        self.assertEqual(summary, report.summarize(SAMPLE, self.path, "sample", MANIFEST)[0])

    def test_overall_is_micro_accuracy_on_unequal_question_groups(self):
        synthetic = {**self.rows[0], "uid": "Technology/temporal/report-fixture",
                     "domain": "Technology", "qtype": "temporal"}
        plan = [*self.rows, synthetic]
        labels = ["Correct", "Correct", "Incorrect"]
        rows = [self.judged(r, label) for r, label in zip(plan, labels)]
        with patch.object(report, "load_retrieval", return_value=plan):
            summary, _, _ = self.run_report(rows)
        self.assertEqual(2 / 3, summary["overall"]["accuracy"])
        self.assertEqual(1, summary["by_domain"]["Finance"]["accuracy"])
        self.assertEqual(0, summary["by_domain"]["Technology"]["accuracy"])

    def test_missing_corrupt_duplicate_foreign_keep_planned_denominator(self):
        first = self.judged(self.rows[0], "Correct")
        unknown = {**first, "uid": "synthetic-foreign-id-never-export", "split": "evaluation"}
        summary, cases, _ = self.run_report(
            [first, unknown, first], extra=b'\xff\n{bad\n\n{"x":1,"x":2}\n',
        )
        self.assertEqual("incomplete", summary["status"])
        self.assertIsNone(summary["overall"]["accuracy"])
        self.assertEqual((0, 2), (summary["overall"]["correct"], summary["overall"]["total"]))
        self.assertEqual(6, len(summary["integrity_errors"]))
        self.assertEqual(["duplicate UID", "missing result"], [c["integrity_issue"] for c in cases])
        self.assertNotIn(unknown["uid"], json.dumps([summary, cases]))

    def test_technical_empty_is_a_miss_with_unfinished_status(self):
        rows = [self.judged(r, label) for r, label in zip(self.rows, ["Correct", ""])]
        rows[1].update(qa_status="timeout", judge_status="skipped_qa", agent_answer="")
        summary, _, _ = self.run_report(rows)
        self.assertEqual("incomplete", summary["status"])
        self.assertEqual(1, summary["overall"]["correct"])
        self.assertEqual([self.rows[1]["uid"]], summary["empty_judge_uids"])
        self.assertEqual(1, summary["qa_status_counts"]["timeout"])

    def test_mixing_context_or_changed_input_invalidates_scoring(self):
        rows = [self.judged(r, "Correct") for r in self.rows]
        rows[0]["judge"] = {"provenance": {**self.provenance, "config_sha256": "different"}}
        summary, cases, _ = self.run_report(rows)
        self.assertEqual(0, summary["overall"]["correct"])
        self.assertTrue(all(c["integrity_issue"] == "mixed run provenance" for c in cases))
        rows[0]["question"] += "wrong question"
        summary, cases, _ = self.run_report(rows)
        self.assertEqual(1, summary["overall"]["correct"])
        self.assertIn("differs from input", cases[0]["integrity_issue"])
        self.assertEqual("incomplete", summary["status"])

    def test_review_validates_verbatim_domain_rank_and_preserves_primary_score(self):
        rows = [self.judged(r, "Incorrect") for r in self.rows]
        summary, cases, planned = self.run_report(rows)
        c = cases[0]
        candidate = c["retrieved_excerpts"][0]
        source = self.rows[0]["retrieved"][0]["formatted_passage"]
        self.assertIn(candidate["excerpt"], source)
        change = {"uid": c["uid"], "case_sha256": c["case_sha256"], "review": {
            "category": "suspected_judge_error", "status": "reviewed",
            "explanation": "Synthetic review fixture", "evidence": [
                {"source": "retrieved", "domain": c["domain"], "msg_node": candidate["msg_node"],
                 "rank": candidate["rank"], "quote": candidate["excerpt"]}],
        }}
        path = self.path.with_name("human.jsonl")
        for key, invalid in (("domain", "Technology"), ("rank", 3), ("quote", "<fabricated>")):
            bad = deepcopy(change)
            bad["review"]["evidence"][0][key] = invalid
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                report.apply_reviews(cases, planned, path)
        path.write_text(json.dumps(change) + "\n", encoding="utf-8")
        report.apply_reviews(cases, planned, path)
        self.assertEqual("reviewed", c["review"]["status"])
        self.assertIs(False, c["official_correct"])
        self.assertEqual(0, summary["overall"]["correct"])
        change["case_sha256"] = "stale"
        path.write_text(json.dumps(change) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            report.apply_reviews(cases, planned, path)
