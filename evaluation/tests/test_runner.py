"""CPU-only protocol, failure, checkpoint and judge integration tests using a fake transport."""
from collections import Counter
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from evaluation import contract, local_model, runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.input = contract.ROOT / 'examples/bm25_development_sample.jsonl'
        self.config = self.dir / 'synthetic.json'
        settings = json.loads((contract.ROOT / 'evaluation/config.example.json').read_text())
        settings['runtime_build'] = 'mock-runtime'
        for stage in ('qa', 'judge'):
            settings[stage].update(model=f'fake-{stage}', revision='fixture-v1',
                                   tokenizer_revision='fixture-v1', chat_template_sha256='1' * 64)
        self.config.write_text(json.dumps(settings))
        self.requests = []

    def args(self, stage='qa', name=None):
        args = runner.parser().parse_args([stage, '--input', str(self.input), '--scope', 'sample',
                                           '--output', str(self.dir / (name or stage)), '--config', str(self.config)])
        if stage == 'judge':
            args.qa = self.dir / 'qa'
        return args

    def transport(self, endpoint, path, payload, timeout):
        self.requests.append((path, payload))
        if path == '/tokenize':
            return {'count': 67, 'max_model_len': 8192}
        answer = 'Not supported by the evidence' if payload['model'] == 'fake-qa' else 'Correct, indeed!'
        return {'model': payload['model'], 'choices': [{'message': {'content': 'Synthetic.\nFinal: ' + answer},
                                                        'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 67, 'completion_tokens': 11, 'total_tokens': 78}}

    def test_qa_judge_independent_resume_and_offline_report(self):
        from evaluation.report import summarize
        with patch.object(contract, 'load_gold', side_effect=AssertionError('QA must not read gold')):
            self.assertEqual(runner.run(self.args(), self.transport), {'ok': 2})
        qa_path = self.dir / 'qa'
        qa_hash = contract.sha256_file(qa_path)
        with self.assertRaises(ValueError):
            runner.run(self.args(), self.transport)
        self.requests.clear()
        args = self.args()
        args.resume = True
        self.assertEqual(runner.run(args, self.transport), {'ok': 2})
        self.assertEqual(self.requests, [])
        self.assertEqual(contract.sha256_file(qa_path), qa_hash)
        gold = {r['uid']: 'SYNTHETIC REFERENCE' for r in contract.read_jsonl(self.input)}
        with patch.object(contract, 'load_gold', return_value=gold):
            self.assertEqual(runner.run(self.args('judge'), self.transport), {'ok': 2})
        chat = [p for path, p in self.requests if path == '/v1/chat/completions']
        self.assertEqual(len(chat), 2)
        self.assertTrue(all(p['model'] == 'fake-judge' for p in chat))
        self.assertTrue(all('SYNTHETIC REFERENCE' in p['messages'][1]['content'] for p in chat))
        self.assertEqual(contract.sha256_file(qa_path), qa_hash)
        counts = summarize(self.input, self.dir / 'judge', 'sample', contract.MANIFEST)[0]
        self.assertEqual(counts['status'], 'complete')
        self.assertEqual(counts['overall']['correct'], 0)
        self.assertEqual(len(counts['online_vs_summary']), 2)

    def test_failures_distinct_no_retry_and_no_judge_for_failed_rows(self):
        completions = iter(['', TimeoutError('synthetic timeout')])
        def transport(*args):
            result = self.transport(*args)
            if args[1] == '/v1/chat/completions':
                value = next(completions)
                if isinstance(value, Exception):
                    raise value
                result['choices'][0]['message']['content'] = value
            return result
        self.assertEqual(runner.run(self.args(), transport), {'empty_output': 1, 'timeout': 1})
        gold = {r['uid']: 'synthetic' for r in contract.read_jsonl(self.input)}
        self.requests.clear()
        with patch.object(contract, 'load_gold', return_value=gold):
            self.assertEqual(runner.run(self.args('judge'), transport), {'skipped_qa': 2})
        self.assertEqual(self.requests, [])
        for row in contract.read_jsonl(self.dir / 'judge'):
            self.assertEqual(row['judge_parse_status'], 'empty')
            self.assertEqual(row['judge_answer'], '')
            self.assertFalse(contract.is_correct(row['judge_answer']))

    def test_budget_enforced_before_chat(self):
        def transport(*args):
            result = self.transport(*args)
            result['count'] = 8192
            return result
        self.assertEqual(runner.run(self.args(), transport), {'over_budget': 2})
        self.assertTrue(all(path == '/tokenize' for path, _ in self.requests))

    def test_resume_unacknowledged_call_and_reuse_only_identical_context(self):
        def interrupt(*args):
            if args[1] == '/v1/chat/completions':
                raise KeyboardInterrupt()
            return self.transport(*args)
        args = self.args()
        with self.assertRaises(KeyboardInterrupt):
            runner.run(args, interrupt)
        self.assertEqual([r['qa_status'] for r in contract.read_jsonl(args.output)], ['in_flight', 'pending'])
        args.resume = True
        self.assertEqual(runner.run(args, self.transport), {'interrupted': 1, 'ok': 1})
        args = self.args(name='reuse')
        args.reuse = self.dir / 'qa'
        self.requests.clear()
        runner.run(args, self.transport)
        self.assertFalse(self.requests)
        settings = json.loads(self.config.read_text())
        settings['temperature'] = 0.9
        self.config.write_text(json.dumps(settings))
        args.output = self.dir / 'changed-settings'
        with self.assertRaisesRegex(ValueError, 'differ'):
            runner.run(args, self.transport)
        self.assertFalse(args.output.exists())
        self.assertFalse(self.requests)

    def test_mutated_input_duplicate_tamper_and_overwrite_rejected(self):
        args = self.args()
        runner.run(args, self.transport)
        rows = contract.read_jsonl(args.output)
        with self.assertRaises(ValueError):
            runner.validate_result(dict(rows[0], agent_answer='modified'), 'qa')
        with self.assertRaises(ValueError):
            runner.validate_result(dict(rows[0], question='modified'), 'qa')
        args.resume = True
        args.input = self.dir / 'changed-input'
        source = contract.read_jsonl(self.input)
        source[0]['retrieved'][0]['formatted_passage'] += ' synthetic-only change'
        runner.checkpoint(args.input, source)
        with self.assertRaisesRegex(ValueError, 'differ'):
            runner.run(args, self.transport)
        args.input = self.input
        runner.checkpoint(args.output, [rows[0], rows[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            runner.run(args, self.transport)
        args.output = args.input
        with self.assertRaisesRegex(ValueError, 'read-only'):
            runner.run(args, self.transport)

    def test_truncated_and_malformed_completions(self):
        for case in ['length', 'no_choices', 'aborted', 'invalid_usage']:
            with self.subTest(case=case):
                def transport(*args):
                    result = self.transport(*args)
                    if args[1] == '/tokenize':
                        return result
                    if case == 'length':
                        result['choices'][0]['finish_reason'] = 'length'
                    elif case == 'no_choices':
                        result['choices'] = []
                    elif case == 'aborted':
                        result['choices'][0]['finish_reason'] = 'abort'
                    else:
                        result['usage'] = None
                    return result
                args = self.args(name=case)
                expected = 'truncated' if case == 'length' else 'call_error'
                self.assertEqual(runner.run(args, transport), {expected: 2})
                for record in contract.read_jsonl(args.output):
                    runner.validate_result(record, 'qa')
                    self.assertTrue(record['qa']['response'])
                    self.assertTrue(record['qa']['error'])
                    if case == 'length':
                        self.assertEqual(record['agent_answer'], 'Not supported by the evidence')

    def test_local_configuration_and_technical_limits(self):
        with self.assertRaises(ValueError):
            local_model.load_config(contract.ROOT / 'evaluation/config.example.json')
        for bad in ['https://example.invalid', 'http://192.0.2.2:8000', 'http://example.invalid',
                    'http://user@127.0.0.1', 'http://127.0.0.1:8000/v1']:
            with self.subTest(url=bad), self.assertRaises(ValueError):
                local_model.endpoint_address(bad)
        def malformed(*args):
            result = self.transport(*args)
            if args[1] == '/v1/chat/completions':
                result['usage']['prompt_tokens'] = 50
            return result
        self.assertEqual(runner.run(self.args(), malformed), Counter(call_error=2))
        for row in contract.read_jsonl(self.dir / 'qa'):
            runner.validate_result(row, 'qa')
            self.assertIn('tokenizer count', row['qa']['error'])
            self.assertTrue(row['qa']['response'])

    @unittest.skipUnless(os.environ.get('QA_EVAL_VERIFY_FULL_REUSE') == '1',
                         'opt-in integration with restored 90-question BM25 and pinned source files')
    def test_full_development_reuses_pilot(self):
        from evaluation import report

        input_path = contract.ROOT / 'runs/bm25_development.jsonl'
        manifest = json.loads(contract.MANIFEST.read_text())
        wanted = set(manifest['development'])
        pilot = set(contract.pilot_uids(manifest['development']))
        self.assertEqual(len(pilot), 12)

        def run_stage(stage, scope, output, expected_calls, qa=None, reuse=None):
            opts = [stage, '--input', str(input_path), '--scope', scope,
                    '--output', str(output), '--config', str(self.config)]
            if qa:
                opts.extend(['--qa', str(qa)])
            if reuse:
                opts.extend(['--reuse', str(reuse)])
            self.requests.clear()
            counts = runner.run(runner.parser().parse_args(opts), self.transport)
            self.assertEqual(counts, {'ok': len(pilot) if scope == 'pilot' else len(wanted)})
            self.assertEqual(len(self.requests), 2 * expected_calls)
            chats = [p['model'] for path, p in self.requests if path == '/v1/chat/completions']
            self.assertEqual(chats, [f'fake-{stage}'] * expected_calls)

        pilot_qa, pilot_judge = self.dir / 'pilot-qa', self.dir / 'pilot-judge'
        full_qa, full_judge = self.dir / 'full-qa', self.dir / 'full-judge'
        run_stage('qa', 'pilot', pilot_qa, 12)
        run_stage('judge', 'pilot', pilot_judge, 12, qa=pilot_qa)
        run_stage('qa', 'development', full_qa, 78, reuse=pilot_qa)
        qa_hash = contract.sha256_file(full_qa)
        run_stage('judge', 'development', full_judge, 78, qa=full_qa, reuse=pilot_judge)

        for stage, full, prior in [('qa', full_qa, pilot_qa), ('judge', full_judge, pilot_judge)]:
            rows = contract.read_jsonl(full)
            self.assertEqual(len(rows), 90)
            self.assertEqual({r['uid'] for r in rows}, wanted)
            reused = {r['uid']: r[stage]['reused_from_sha256'] for r in rows
                      if 'reused_from_sha256' in r[stage]}
            self.assertEqual(reused, dict.fromkeys(pilot, contract.sha256_file(prior)))

        score_dir = self.dir / 'report'
        self.assertEqual(report.main(['--input', str(input_path), '--scope', 'development',
                                      '--judge', str(full_judge), '--data-dir', str(contract.DATA),
                                      '--output-dir', str(score_dir)]), 0)
        summary = json.loads((score_dir / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['overall']['total'], 90)
        self.assertEqual({d: v['total'] for d, v in summary['by_domain'].items()},
                         {'Finance': 44, 'Technology': 46})
        self.assertEqual({t: v['total'] for t, v in summary['by_qtype'].items()},
                         dict(Counter(uid.split('/')[1] for uid in wanted)))
        self.assertEqual(len(contract.read_jsonl(score_dir / 'failures.jsonl')), summary['failure_count'])
        run_stage('judge', 'development', self.dir / 'rejudged', 90, qa=full_qa)
        self.assertEqual(contract.sha256_file(full_qa), qa_hash)


if __name__ == '__main__':
    unittest.main()
