#!/usr/bin/env python3
"""Run the fixed pilot's first QA and judge request, then stop for protocol inspection."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import contract, local_model, runner


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'),
                        help='plan reads inputs only; run makes at most one new call per stage')
    parser.add_argument('--config', type=Path, help='agreed deployment config, required for run')
    parser.add_argument('--input', type=Path, default=ROOT / 'runs/bm25_development.jsonl',
                        help='the same full BM25 development file used for the 12 and 90 scopes')
    parser.add_argument('--manifest', type=Path, default=contract.MANIFEST)
    parser.add_argument('--data-dir', type=Path, default=contract.DATA, help='verified data root for judge')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/real-pilot-prep/first')
    args = parser.parse_args()
    if args.action == 'run' and args.config is None:
        parser.error('run requires --config')
    return args


def stage_args(args, stage, output, qa_snapshot=None):
    return argparse.Namespace(stage=stage, input=args.input, manifest=args.manifest,
                              data_dir=args.data_dir, config=args.config, scope='pilot',
                              output=output, qa=qa_snapshot, reuse=None, resume=False)


def write_immutable(path, row):
    with runner.writer(path):
        if path.exists():
            if contract.read_jsonl(path) != [row]:
                raise ValueError(f'existing artifact differs: {path}')
            return
        runner.checkpoint(path, [row])


def first_stage(args, uid):
    """Checkpoint in_flight before contacting the normal local transport. No technical retries."""
    with runner.writer(args.output):
        plan = runner.plan(args)
        if len(plan) != 12 or plan[0]['uid'] != uid:
            raise ValueError('fixed pilot changed')
        rows = plan if args.stage == 'qa' else plan[:1]
        # The complete QA snapshot records 11 intentionally pending questions.
        # Only the actually selected judge is persisted in this first-case artifact.
        if args.output.exists():
            runner.restore(rows, args.output, args.stage, resume=True)
            if args.stage == 'qa' and any(row['qa_status'] != 'pending' for row in rows[1:]):
                raise ValueError('a later pilot QA was attempted in the first-case snapshot')
        else:
            runner.checkpoint(args.output, rows)
        row = rows[0]
        audit = row[args.stage]
        if audit['status'] == 'interrupted':
            # restore converts an unknown-delivery in_flight record into a terminal status.
            runner.checkpoint(args.output, rows)
        if audit['status'] != 'pending':
            return row
        audit['status'] = 'in_flight'
        runner.seal(row, args.stage)
        runner.checkpoint(args.output, rows)
        local_model.call(audit)
        runner.seal(row, args.stage)
        runner.checkpoint(args.output, rows)
        runner.validate_result(row, args.stage)
        return row


def inspect(qa, judge, paths, config):
    flags = []
    for stage, row in [('qa', qa), ('judge', judge)]:
        if row is None:
            flags.append(f'{stage} unattempted')
            continue
        audit = row[stage]
        if audit['status'] != 'ok':
            flags.append(f"{stage} status={audit['status']}")
        if audit['format_status'] != 'marked':
            flags.append(f"{stage} format={audit['format_status']}")
    if judge is not None:
        if judge['judge_parse_status'] not in {'correct', 'incorrect'}:
            flags.append(f"judge verdict format={judge['judge_parse_status']}")
        if judge['judge_online_correct'] != contract.is_correct(judge['judge_answer']):
            flags.append('online and final parsers differ')
    info = {
        'uid': qa['uid'], 'scope': 'first of fixed 12 development pilot',
        'qa_status': qa['qa_status'], 'judge_status': judge['judge_status'] if judge else 'unattempted',
        'qa_format': qa['qa']['format_status'], 'judge_format': judge['judge']['format_status'] if judge else None,
        'judge_parse_status': judge['judge_parse_status'] if judge else None,
        'protocol_flags': flags, 'gate': 'inspect_flags' if flags else 'ready_for_protocol_review',
        'remaining_pilot_uids_unattempted': 11,
        'sha256': {name: contract.sha256_file(path) for name, path in paths.items() if path.is_file()},
        'config_sha256': contract.sha256_file(config), 'orchestrator_sha256': contract.sha256_file(__file__),
    }
    return info


def main():
    args = arguments()
    selection = contract.load_retrieval(args.input, args.manifest, 'pilot')
    uid = selection[0]['uid']
    qa_snapshot = args.output_dir / 'first.qa.snapshot.jsonl'
    qa_cache = args.output_dir / 'first.qa.jsonl'
    judge_path = args.output_dir / 'first.judge.jsonl'
    gate_path = args.output_dir / 'inspection.jsonl'
    paths = {'qa_snapshot': qa_snapshot, 'qa_cache': qa_cache, 'judge_cache': judge_path}
    qa_args = stage_args(args, 'qa', qa_snapshot)
    if args.action == 'plan':
        info = {'scope': 'pilot', 'uid': uid, 'pilot_count': len(selection),
                'planned_completion_cap': {'qa': 1, 'judge': 1},
                'input_sha256': contract.sha256_file(args.input),
                'manifest_sha256': contract.sha256_file(args.manifest)}
        if args.config:
            plan = runner.plan(qa_args)
            info['qa_plan_cache_key'] = plan[0]['qa']['cache_key']
        print(json.dumps(info, indent=2))
        return 0

    inputs = {path.resolve() for path in (args.input, args.manifest, args.config)}
    outputs = [p.resolve() for p in [*paths.values(), gate_path]]
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError('first-case artifacts must be separate from each other and the input files')
    with runner.writer(args.output_dir / 'first_case'):
        # An orphaned later artifact may mean a previous request was already delivered.
        if any(path.exists() for path in (qa_cache, judge_path, gate_path)):
            if not qa_snapshot.is_file():
                raise ValueError('first QA snapshot missing next to existing first-case artifacts')
            saved = contract.read_jsonl(qa_snapshot)
            if not saved or saved[0]['qa_status'] in {'pending', 'in_flight'}:
                raise ValueError('first QA snapshot conflicts with a later artifact')
        if gate_path.exists():
            prior, = contract.read_jsonl(gate_path)
            if any(not paths[key].is_file() or contract.sha256_file(paths[key]) != checksum
                   for key, checksum in prior['sha256'].items()):
                raise ValueError('previously inspected artifact is missing or changed')
        qa = first_stage(qa_args, uid)
        write_immutable(qa_cache, qa)
        judge = None
        if qa['qa_status'] == 'ok':
            judge = first_stage(stage_args(args, 'judge', judge_path, qa_snapshot), uid)
        info = inspect(qa, judge, paths, args.config)
        write_immutable(gate_path, info)
    print(json.dumps(info, indent=2))
    return 1 if info['protocol_flags'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f'first case stopped: {exc}')
