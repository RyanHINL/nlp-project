"""Run or resume one development QA or judge stage against a local service."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from evaluation import contract, local_model
from evaluation.contract import digest_json, sha256_file


STATUSES = {'pending', 'in_flight', 'ok', 'empty_output', 'empty_final', 'over_budget',
            'timeout', 'call_error', 'truncated', 'interrupted', 'skipped_qa'}
ANSWER_FIELD = {'qa': 'agent_answer', 'judge': 'judge_answer'}


def provenance(stage, settings, retrieval_hash, manifest_path, qa_hash=None):
    sources = ['evaluation/runner.py', 'evaluation/local_model.py', 'evaluation/contract.py',
               'evaluation/development_identities.json', 'retrieval/data_tools.py',
               'vendor/GroupMemBench/baselines/rag_common/eval_lib.py',
               'vendor/GroupMemBench/task_synthesis/summarize_typed_eval.py']
    role = 'agent' if stage == 'qa' else 'judge'
    prompt = contract.VENDOR / f'prompts/hipporag_{role}_system.txt'
    result = {
        'retrieval_sha256': retrieval_hash, 'manifest_sha256': sha256_file(manifest_path),
        'settings': settings, 'config_sha256': digest_json(settings),
        'system_prompt_sha256': sha256_file(prompt),
        'code_sha256': digest_json({p: sha256_file(contract.ROOT / p) for p in sources}),
    }
    if stage == 'judge':
        result.update(source_qa_sha256=qa_hash,
                      data_sources_sha256=sha256_file(contract.ROOT / 'configs/data_sources.json'))
    return result


def parse_fields(stage, raw):
    text = raw or ''
    _, answer = contract.split_reasoning_and_final(text)
    fields = {ANSWER_FIELD[stage]: answer}
    if stage == 'judge':
        norm = re.sub('[^a-zA-Z]+', '', answer).lower()
        code = norm if norm in {'correct', 'incorrect'} else ('unrecognized' if answer else 'empty')
        fields.update(judge_parse_status=code, judge_online_correct=contract.parse_judgment(answer))
    return fields


def cache_key(row, stage, info, request):
    source = {k: row[k] for k in ('uid', 'domain', 'qtype', 'split', 'method', 'question',
                                  'asking_user_id', 'retrieved', 'retrieval_seconds')}
    source['qa_status'] = 'not_run'
    # A full QA file may incorporate pilot records. Bind judge reuse to the
    # individual QA request and extracted answer, independent of the container.
    parent = row['qa']['cache_key'] if stage == 'judge' else None
    condition = {k: v for k, v in info.items() if k != 'source_qa_sha256'}
    return digest_json([stage, source, parent, condition, request])


def seal(row, stage):
    audit = row[stage]
    fields = parse_fields(stage, audit['raw_output'])
    row.update(fields)
    if audit['status'] == 'ok' and not row[ANSWER_FIELD[stage]]:
        audit['status'] = 'empty_final'
    row[stage + '_status'] = audit['status']
    audit['format_status'] = contract.answer_format(audit['raw_output'] or '')
    audit.pop('audit_sha256', None)
    audit['audit_sha256'] = digest_json(audit)


def validate_result(row, stage):
    """Check the stored protocol, local record checksum and duplicated summary fields offline."""
    audit = row[stage]
    if not isinstance(audit, dict) or audit['audit_sha256'] != digest_json(
            {k: v for k, v in audit.items() if k != 'audit_sha256'}):
        raise ValueError(f'{stage} audit checksum mismatch')
    status = row[stage + '_status']
    if status not in STATUSES or status != audit['status']:
        raise ValueError(f'{stage} status mismatch')
    raw = audit['raw_output']
    if raw is not None and not isinstance(raw, str):
        raise ValueError(f'invalid {stage} output')
    if raw is not None and raw != (audit['response']['choices'][0]['message']['content'] or ''):
        raise ValueError(f'{stage} output differs from response')
    if status == 'ok' and not row[ANSWER_FIELD[stage]]:
        raise ValueError(f'empty {stage} answer recorded as successful')
    if status == 'skipped_qa' and (stage != 'judge' or row['qa_status'] == 'ok'):
        raise ValueError('invalid skipped_qa status')
    if audit['format_status'] != contract.answer_format(raw or ''):
        raise ValueError(f'{stage} format differs from raw completion')
    for key, value in parse_fields(stage, raw).items():
        if row[key] != value:
            raise ValueError(f'{key} differs from official parsing')
    info = audit['provenance']
    if info['config_sha256'] != digest_json(info['settings']):
        raise ValueError(f'{stage} settings checksum mismatch')
    messages = (contract.qa_messages(row) if stage == 'qa' else
                contract.judge_messages(row['question'], row['gold_answer'], row['agent_answer']))
    if audit['cache_key'] != cache_key(row, stage, info, audit['request']):
        raise ValueError(f'{stage} cache key differs from recorded inputs')
    if audit['request'] != local_model.chat_request(messages, info['settings']):
        raise ValueError(f'{stage} request differs from recorded inputs or settings')


def index_rows(rows):
    by_uid = {}
    for row in rows:
        uid = row['uid']
        if uid in by_uid:
            raise ValueError(f'duplicate UID {uid}')
        by_uid[uid] = row
    return by_uid


def read_stable(path, reader):
    checksum = sha256_file(path)
    value = reader(path)
    if sha256_file(path) != checksum:
        raise ValueError(f'{path} changed during read')
    return value, checksum


def plan(args):
    retrieval, input_hash = read_stable(args.input, lambda p: contract.load_retrieval(p, args.manifest, args.scope))
    config = local_model.load_config(args.config)
    stage = args.stage
    qa_by_uid, gold, qa_hash = {}, {}, None
    if stage == 'judge':
        qa_rows, qa_hash = read_stable(args.qa, contract.read_jsonl)
        qa_by_uid = index_rows(qa_rows)
        if set(qa_by_uid) != {row['uid'] for row in retrieval}:
            raise ValueError('QA artifact must contain every and only the planned UIDs')
        qa_info = qa_rows[0]['qa']['provenance']
        if (qa_info['retrieval_sha256'] != input_hash or
                qa_info['manifest_sha256'] != sha256_file(args.manifest) or
                qa_info['settings'] != local_model.settings_for(config, 'qa')):
            raise ValueError('QA artifact differs from configured QA condition or input')
        gold = contract.load_gold(retrieval, args.data_dir, args.manifest)
    settings = local_model.settings_for(config, stage)
    info = provenance(stage, settings, input_hash, args.manifest, qa_hash)
    rows = []
    for source in retrieval:
        uid = source['uid']
        row = dict(source, scope=args.scope)
        if stage == 'qa':
            messages = contract.qa_messages(source)
        else:
            qa = qa_by_uid[uid]
            validate_result(qa, 'qa')
            if (qa['scope'] != args.scope or 'judge' in qa or qa['qa']['provenance'] != qa_info or
                    any(qa[k] != v for k, v in source.items() if k != 'qa_status')):
                raise ValueError(f'{uid}: QA source scope, retrieval or condition mismatch')
            row.update(qa=qa['qa'], qa_status=qa['qa_status'], agent_answer=qa['agent_answer'], gold_answer=gold[uid])
            messages = contract.judge_messages(row['question'], row['gold_answer'], row['agent_answer'])
        request = local_model.chat_request(messages, settings)
        skipped = stage == 'judge' and row['qa_status'] != 'ok'
        row[stage] = {
            'provenance': info, 'request': request, 'cache_key': cache_key(row, stage, info, request),
            'status': 'skipped_qa' if skipped else 'pending', 'raw_output': None, 'response': None,
            'prompt_tokens': None, 'usage': None, 'seconds': None,
            'error': f"QA status was {row['qa_status']}" if skipped else None,
        }
        seal(row, stage)
        rows.append(row)
    return rows


@contextmanager
def writer(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode stable across atomic output replacement and process death.
    with (output.parent / (output.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'{output} is being written by another process') from exc
        yield


def checkpoint(output, rows):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                     prefix='.' + output.name + '.', suffix='.partial', delete=False) as stream:
        path = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    try:
        os.replace(path, output)
    finally:
        path.unlink(missing_ok=True)


def restore(rows, path, stage, resume):
    saved_rows, checksum = read_stable(path, contract.read_jsonl)
    saved = index_rows(saved_rows)
    wanted = {row['uid'] for row in rows}
    if set(saved) - wanted or (resume and set(saved) != wanted):
        raise ValueError('unexpected checkpoint UID set')
    for row in rows:
        prior = saved.get(row['uid'])
        if prior is None:
            continue
        validate_result(prior, stage)
        audit = prior[stage]
        if audit['cache_key'] != row[stage]['cache_key']:
            raise ValueError(f"{row['uid']}: cached context, inputs or settings differ")
        if stage == 'qa' and 'judge' in prior:
            raise ValueError('expected a QA-only artifact')
        if stage == 'judge':
            validate_result(prior, 'qa')
        mutable = {stage, stage + '_status', *parse_fields(stage, None)}
        if not resume:
            mutable.add('scope')
            if stage == 'judge':
                # A reused pilot QA may carry a new artifact reference. Its request,
                # final, technical status and parent cache key are already bound.
                mutable.add('qa')
        if ({k: v for k, v in prior.items() if k not in mutable} !=
                {k: v for k, v in row.items() if k not in mutable}):
            raise ValueError('checkpoint does not match planned row')
        if resume and audit['provenance'] != row[stage]['provenance']:
            raise ValueError('checkpoint provenance differs')
        expected_status = row[stage]['status']
        if (audit['status'] == 'skipped_qa') != (expected_status == 'skipped_qa'):
            raise ValueError('cached QA technical status differs')
        info = row[stage]['provenance']
        row[stage] = deepcopy(audit)
        row[stage]['provenance'] = info
        if not resume:
            row[stage]['reused_from_sha256'] = checksum
        if audit['status'] == 'in_flight':
            row[stage].update(status='interrupted', error='Process interrupted; delivery/response unknown')
        seal(row, stage)


def run(args, transport=local_model.post):
    stage, output = args.stage, args.output
    sources = (args.input, args.config, args.manifest, args.qa, args.reuse)
    if any(p is not None and p.resolve() == output.resolve() for p in sources):
        raise ValueError('output must differ from the read-only input paths')
    with writer(output):
        if output.exists() != args.resume:
            raise ValueError('resume requires an existing output; new output must not already exist')
        rows = plan(args)
        if args.resume or args.reuse:
            restore(rows, output if args.resume else args.reuse, stage, args.resume)
        checkpoint(output, rows)
        for row in rows:
            audit = row[stage]
            if audit['status'] != 'pending':
                continue
            audit['status'] = 'in_flight'
            seal(row, stage)
            checkpoint(output, rows)
            local_model.call(audit, transport)
            seal(row, stage)
            checkpoint(output, rows)
    return Counter(row[stage + '_status'] for row in rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('qa', 'judge'))
    for name in ['input', 'output', 'config']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scope', choices=('sample', 'pilot', 'development'), required=True)
    p.add_argument('--manifest', type=Path, default=contract.MANIFEST)
    p.add_argument('--qa', type=Path, help='judge only: completed or partial QA artifact from the same scope')
    p.add_argument('--data-dir', type=Path, default=contract.DATA, help='pinned data root used only by judge')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true', help='continue pending UIDs in this output')
    group.add_argument('--reuse', type=Path, help='reuse matching rows from a prior artifact, e.g. pilot')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if bool(args.qa) != (args.stage == 'judge'):
        p.error('--qa is required for judge and forbidden for QA')
    try:
        counts = run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f'{exc}\n')
    complete = set(counts) == {'ok'}
    print(json.dumps({'stage': args.stage, 'scope': args.scope, 'counts': counts,
                      'status': 'complete' if complete else 'incomplete'}))
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
