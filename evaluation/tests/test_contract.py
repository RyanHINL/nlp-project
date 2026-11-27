"""CPU tests for pinned GroupMemBench presentation, identities, and scoring."""

import ast
from copy import deepcopy
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import contract as c


SAMPLE = c.ROOT / "examples/bm25_development_sample.jsonl"


def official_user_texts(row, gold, final):
    # Evaluate just the vendored string-building assignments in their original
    # order, without importing the API client or running retrieval / run_qa.
    source = c.VENDOR / "baselines/rag_common/eval_lib.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "run_qa")
    names = {"passages_text", "asker_line", "agent_user", "judge_user"}
    statements = sorted((n for n in ast.walk(function) if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names),
                        key=lambda n: n.lineno)
    scope = dict(asking_user_id=row["asking_user_id"] or "", question_text=row["question"],
                 retrieved_docs=[d["formatted_passage"] for d in row["retrieved"]],
                 gold_answer=gold, agent_final=final)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), scope)
    return scope["agent_user"], scope["judge_user"]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = c.read_jsonl(SAMPLE)
        self.manifest = json.loads(c.MANIFEST.read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def write_rows(self, rows):
        path = self.directory / "retrieval.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def test_two_record_smoke(self):
        with patch.object(c, "_source_rows", side_effect=AssertionError("QA opened the gold source")):
            rows = c.load_retrieval(SAMPLE, scope="sample")
            self.assertEqual(rows, self.rows)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["qtype"] == "multi_hop" and r["domain"] == "Finance" for r in rows))
            for r in rows:
                user = c.qa_messages(r)[1]["content"]
                self.assertEqual(user, official_user_texts(r, "", "")[0])
                self.assertNotIn(r["uid"], user)
                self.assertNotIn(r["method"], user)
        for scope in ("development", "pilot"):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "missing"):
                c.load_retrieval(SAMPLE, scope=scope)

    def test_upstream_prompt_and_field_separation(self):
        source = self.rows[0]
        for asker, passages in ((source["asking_user_id"], source["retrieved"]), (None, [])):
            r = dict(source, asking_user_id=asker, retrieved=passages,
                     uid="UID-SENTINEL", qtype="TYPE-SENTINEL", split="SPLIT-SENTINEL",
                     method="METHOD-SENTINEL", answer="GOLD-SENTINEL", judge_answer="JUDGE-SENTINEL")
            agent, judge = official_user_texts(r, r["answer"], "FINAL-SENTINEL")
            expected_qa = (c.VENDOR / "prompts/hipporag_agent_system.txt").read_text().strip()
            expected_judge = (c.VENDOR / "prompts/hipporag_judge_system.txt").read_text().strip()
            self.assertEqual(c.qa_messages(r), [{"role": "system", "content": expected_qa},
                                                {"role": "user", "content": agent}])
            self.assertNotIn("SENTINEL", agent)
            self.assertEqual(c.judge_messages(r["question"], r["answer"], "FINAL-SENTINEL"),
                             [{"role": "system", "content": expected_judge}, {"role": "user", "content": judge}])
            self.assertNotIn("Retrieved passages", judge)

    def test_explicit_empty_and_unsorted_scores(self):
        rows = deepcopy(self.rows)
        rows[0]["retrieved"] = []
        rows[1]["retrieved"][0]["score"] = -9000
        self.assertEqual(c.load_retrieval(self.write_rows(rows), scope="sample"), rows)
        self.assertIn("Retrieved passages:\n\n\n", c.qa_messages(rows[0])[1]["content"])

    def test_invalid_identity_schema_and_ranks(self):
        r = self.rows[0]
        bad_rows = [
            dict(r, uid=self.manifest["evaluation"][0]), dict(r, split="evaluation"),
            dict(r, domain="Technology"), dict(r, qtype="abstention"), dict(r, question="mismatch"),
            dict(r, asking_user_id="another"), dict(r, qa_status="done"), dict(r, retrieved=None),
            dict(r, retrieved=r["retrieved"] + [r["retrieved"][0]]),
            dict(r, retrieved=[dict(r["retrieved"][0], rank=2)]),
            dict(r, retrieved=[dict(r["retrieved"][0], rank=True)]),
            dict(r, retrieved=[dict(r["retrieved"][0], score="0.5")]),
            dict(r, retrieved=[dict(r["retrieved"][0], is_noise=True)]),
            dict(r, answer="unexpected reference"),
            {key: value for key, value in r.items() if key != "retrieved"},
        ]
        for i, invalid in enumerate(bad_rows):
            with self.subTest(case=i), self.assertRaises(ValueError):
                c.load_retrieval(self.write_rows([invalid]), scope="sample")
        with self.assertRaisesRegex(ValueError, "duplicate retrieval"):
            c.load_retrieval(self.write_rows([r, r]), scope="sample")
        with self.assertRaisesRegex(ValueError, "one method"):
            c.load_retrieval(self.write_rows([r, dict(self.rows[1], method="other")]), scope="sample")

    def test_unknown_extension_fields_stay_out_of_qa_artifacts(self):
        rows = deepcopy(self.rows)
        rows[0]["extension"] = {"internal": "unrecognized top-level fields"}
        rows[0]["retrieved"][0]["extra_tags"] = ["unrecognized message metadata"]
        self.assertEqual(c.load_retrieval(self.write_rows(rows), scope="sample"), self.rows)

    def test_json_integrity(self):
        for broken in ('{', '\n', '[]', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            path = self.directory / "broken.jsonl"
            path.write_text('{}\n' + broken + '\n', encoding='utf-8')
            with self.subTest(broken=broken), self.assertRaisesRegex(ValueError, r"broken.jsonl:2"):
                c.read_jsonl(path)

    def test_fixed_pilot(self):
        development = self.manifest["development"]
        pilot = c.pilot_uids(development)
        self.assertEqual(len(pilot), 12)
        for domain in c.DOMAINS:
            for qtype in c.QTYPES:
                prefix = f"{domain}/{qtype}/"
                self.assertEqual([u for u in pilot if u.startswith(prefix)],
                                 [next(u for u in development if u.startswith(prefix))])
        changes = deepcopy(self.manifest)
        changes["development"][0], changes["evaluation"][0] = changes["evaluation"][0], changes["development"][0]
        path = self.directory / "shuffled.json"
        path.write_text(json.dumps(changes))
        with self.assertRaisesRegex(ValueError, "committed"):
            c.load_retrieval(SAMPLE, path, scope="sample")

    def test_official_extract_and_two_verdicts(self):
        cases = [
            ("", "", "empty", None, False), ("  ", "", "empty", None, False),
            ("reason\ntail", "tail", "fallback", None, False),
            ("Final: Incorrect\nFinal ANSWER: C-o-r-r-e-c-t!\npostscript", "C-o-r-r-e-c-t!", "marked", None, True),
            ("Final: Correct explanation", "Correct explanation", "marked", True, False),
            ("Final: not correct", "not correct", "marked", False, False),
            ("Final: CORRECT!!", "CORRECT!!", "marked", True, True),
            ("Final: Correct\nFinal: ", "", "marked", None, False),
        ]
        for raw, final, mode, online, score in cases:
            with self.subTest(raw=raw):
                self.assertEqual(c.split_reasoning_and_final(raw)[1], final)
                self.assertEqual(c.answer_format(raw), mode)
                self.assertIs(c.parse_judgment(final), online)
                self.assertIs(c.is_correct(final), score)

    def test_offline_gold_join_with_synthetic_source(self):
        sample = self.rows[0]
        uid = sample["uid"]
        name = f"questions/{sample['domain']}/{sample['qtype']}.jsonl"
        data = self.directory / "data"
        path = data / name
        path.parent.mkdir(parents=True)
        row = dict(id=uid.split("/")[2], question=sample["question"],
                   asking_user_id=sample["asking_user_id"], answer="SYNTHETIC-ANSWER")
        other_id = next(u.split("/")[2] for u in self.manifest["evaluation"] if u.startswith("Finance/multi_hop/"))
        heldout = dict(id=other_id)
        path.write_text(json.dumps(heldout) + "\n" + json.dumps(row) + "\n")
        (self.directory / "configs").mkdir()
        (self.directory / "configs/data_sources.json").write_text(json.dumps({"files": {name: c.sha256_file(path)}}))
        with patch.object(c, "ROOT", self.directory):
            self.assertEqual(c.load_gold([sample], data, c.MANIFEST), {uid: "SYNTHETIC-ANSWER"})
            for change in (dict(question="mismatch"), dict(asking_user_id=None), dict(qtype="temporal"),
                           dict(uid=self.manifest["evaluation"][0])):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.load_gold([dict(sample, **change)], data, c.MANIFEST)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                c.load_gold([sample], data, c.MANIFEST)

    @unittest.skipUnless((c.ROOT / "runs/bm25_development.jsonl").is_file(), "restored baseline not installed")
    def test_pilot_note_locators(self):
        rows = c.load_retrieval(c.ROOT / "runs/bm25_development.jsonl", scope="pilot")
        notes = (c.ROOT / "evaluation/pilot_notes.md").read_text(encoding="utf-8")
        pattern = r'^\| `([^`]+)` \| .*? \| (Finance|Technology)/(Msg_\d+), rank (\d+)\. “([^”]+)” \|'
        citations = re.findall(pattern, notes, re.MULTILINE)
        self.assertEqual([uid for uid, *_ in citations], [row["uid"] for row in rows])
        for (uid, domain, node, rank, excerpt), row in zip(citations, rows):
            with self.subTest(uid=uid):
                hit = row["retrieved"][int(rank) - 1]
                self.assertEqual((domain, node), (row["domain"], hit["msg_node"]))
                self.assertIn(excerpt, hit["formatted_passage"])

    @unittest.skipUnless((c.DATA / "questions").is_dir(), "pinned question files not installed")
    def test_reproduce_catalog_and_real_join(self):
        catalog = c.build_identity_catalog()
        self.assertEqual(catalog, json.loads(c.IDENTITIES.read_text()))
        self.assertEqual(set(catalog), set(self.manifest["development"]))
        self.assertTrue(all(len(v) == 64 for v in catalog.values()))
        self.assertEqual(set(c.load_gold(self.rows)), {r["uid"] for r in self.rows})


if __name__ == "__main__":
    unittest.main()
