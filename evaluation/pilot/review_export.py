"""Export the frozen real BM25 first-case or twelve-case pilot for offline review."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
from evaluation import contract as c, local_model, report, runner

CHECKS = PREP / 'bm25_checks.json'
WORKSHEET = ROOT / 'evaluation/pilot_notes.md'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
STAGES = ('qa', 'judge')
UNATTEMPTED = {'pending', 'skipped_qa'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return c.parse_json(path.read_text(encoding='utf-8'))


def frozen(run):
    path = run / 'deployment.frozen.json'
    deployment = read_json(path)
    config = local_model.load_config(run / 'local.json')
    spec = read_json(PREP / 'deployment.json')
    require(deployment['status'] == 'frozen_for_launch' and deployment['model'] == MODEL and
            deployment['revision'] == REVISION and
            all(deployment[key] == value for key, value in spec.items() if key != 'status'),
            'run differs from the reviewed deployment')
    require(config['runtime_build'].endswith('deployment_sha256=' + c.sha256_file(path)),
            'runtime build does not identify this frozen deployment')
    proposal = deployment['config_proposal']
    require({k:v for k,v in config.items() if k != 'runtime_build'} ==
            {k:v for k,v in proposal.items() if k != 'runtime_build'}, 'config differs from frozen proposal')
    for name, key in [('chat_template.jinja', 'chat_template_sha256'),
                      ('model.sha256.json', 'model_manifest_sha256'), ('packages.freeze.txt', 'packages_freeze_sha256')]:
        require(c.sha256_file(run / name) == deployment[key], f'{name} changed')
    return deployment, config


def worksheet(planned):
    notes = {}
    for line in WORKSHEET.read_text(encoding='utf-8').splitlines():
        if not line.startswith('| `'):
            continue
        uid, focus, evidence, check = [x.strip() for x in line.strip('|').split('|')]
        uid = uid.strip('`')
        if uid not in planned:
            continue
        node, rank, quote = re.fullmatch(r'(.+), rank (\d+)\. “(.*)”', evidence).groups()
        hit = planned[uid]['retrieved'][int(rank) - 1]
        require(node == planned[uid]['domain'] + '/' + hit['msg_node'] and quote in hit['formatted_passage'],
                f'{uid}: worksheet citation differs from retrieved passage')
        notes[uid] = (focus, node, rank, quote, check)
    require(notes.keys() == planned.keys(), 'worksheet missing pilot UID')
    return notes


def scan(path, stage, plan, expected, checks, issues, qa_rows=None, prior=None):
    """Validate before exposing raw content. Ambiguous UIDs are quarantined."""
    rows, seen = {}, set()
    if not path.is_file():
        issues.append({'file':path.name, 'kind':'missing_file'})
        return rows
    before = c.sha256_file(path)
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            uid = None
            kind = 'corrupt'
            try:
                row = c.parse_json(raw.decode('utf-8'))
                key = row.get('uid') if isinstance(row, dict) else None
                require(isinstance(key, str) and key in plan, 'foreign or absent UID')
                uid = key
                if uid in seen:
                    kind = 'duplicate'
                    raise ValueError('duplicate UID')
                seen.add(uid)
                allowed = set(c.RETRIEVAL_FIELDS) | {'scope','qa','agent_answer'}
                if stage=='judge':
                    allowed.update({'gold_answer','judge','judge_status','judge_answer','judge_online_correct','judge_parse_status'})
                require(row.keys()==allowed, 'unexpected result fields')
                require(row['scope']=='pilot' and all(row[k]==v for k,v in plan[uid].items() if k!='qa_status'),
                        'retrieval or scope mismatch')
                runner.validate_result(row, stage)
                audit = row[stage]
                require(audit['provenance']==expected, 'deployment, source or code fingerprint mismatch')
                if stage == 'judge':
                    require(c.digest_json(row['gold_answer'])==checks[uid]['gold_sha256'] and uid in qa_rows and
                            row['qa']==qa_rows[uid][0]['qa'] and row['agent_answer']==qa_rows[uid][0]['agent_answer'],
                            'different pinned reference or parent QA')
                if 'reused_from_sha256' in audit:
                    require(prior is not None and uid in prior[1] and audit['reused_from_sha256']==prior[0],
                            'unverified cache source')
                    parent = prior[1][uid][0][stage]
                    ignore = {'provenance','reused_from_sha256','audit_sha256'}
                    require({k:v for k,v in audit.items() if k not in ignore} ==
                            {k:v for k,v in parent.items() if k not in ignore}, 'cache response drift')
                rows[uid] = (row, number)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                if uid:
                    rows.pop(uid, None)
                issues.append({'file':path.name, 'line':number, 'kind':kind, 'uid':uid, 'issue':str(exc)})
    require(c.sha256_file(path)==before, f'{path} changed during export')
    return rows


def counts(phase, stage, planned_n, rows):
    stats = Counter()
    statuses, parsed = Counter(), Counter()
    attempted = set()
    for uid, (row, _) in rows.items():
        audit = row[stage]
        status = audit['status']
        statuses[status] += 1
        if 'reused_from_sha256' in audit:
            stats['cache_hits'] += 1
        elif status not in UNATTEMPTED:
            attempted.add(uid)
            if isinstance(audit['seconds'], (int,float)):
                stats['observed_stage_seconds'] += audit['seconds']
            if audit['usage']:
                for key in ('prompt_tokens', 'completion_tokens'):
                    value = audit['usage'].get(key) if isinstance(audit['usage'],dict) else None
                    if isinstance(value,int):
                        stats[key] += value
            if audit['response'] is not None:
                stats['chat_responses'] += 1
            if status in {'interrupted','in_flight','timeout'}:
                stats['uncertain_delivery_or_completion'] += 1
        if status not in UNATTEMPTED | {'ok','in_flight'}:
            stats['technical_errors'] += 1
        if st_label := (row['judge_parse_status'] if stage=='judge' else audit['format_status']):
            parsed[st_label] += 1
        if audit['format_status'] == 'fallback':
            stats['fallback_final'] += 1
        if stage == 'judge' and row['judge_parse_status'] in {'unrecognized','empty'}:
            stats['empty_or_unrecognized_final'] += 1
    return {'phase':phase,'stage':stage,'planned_records':planned_n,'valid':len(rows),
            'missing_or_quarantined':planned_n-len(rows),'new_stage_attempts':len(attempted),
            **{k:stats[k] for k in ('cache_hits','chat_responses','uncertain_delivery_or_completion',
                                    'fallback_final','empty_or_unrecognized_final','technical_errors',
                                    'observed_stage_seconds','prompt_tokens','completion_tokens')},
            'parse_statuses':dict(parsed),'statuses':dict(statuses)}, attempted


def notes_for(cases, plan, path, data_dir):
    notes = {case['uid']:dict(uid=case['uid'],case_sha256=case['case_sha256'],category=None,
                              assessment='',evidence=[],human_notes='') for case in cases}
    if not path:
        return notes
    reviewed, seen = [], set()
    for supplied in c.read_jsonl(path):
        uid = supplied['uid']
        require(uid in notes and uid not in seen and supplied['case_sha256']==notes[uid]['case_sha256'],
                'unknown, duplicate or stale note')
        seen.add(uid)
        require(set(supplied)==set(notes[uid]) and isinstance(supplied['human_notes'], str)
                and isinstance(supplied['assessment'], str), 'invalid note schema')
        if supplied['assessment']:
            reviewed.append(dict(uid=uid,case_sha256=supplied['case_sha256'],review=dict(
                status='reviewed',category=supplied['category'],explanation=supplied['assessment'],
                evidence=supplied['evidence'])))
        else:
            require(not supplied['category'] and not supplied['evidence'], 'a diagnosis needs an assessment')
        notes[uid]=supplied
    if reviewed:
        # Use the repository's quote/locator checks on a temporary internal projection.
        # The exported assessment and separate human_notes preserve who still needs to review.
        with tempfile.TemporaryDirectory() as td:
            scratch=Path(td)/'annotations.jsonl'
            report._write_jsonl(scratch,reviewed)
            report.apply_reviews([dict(case) for case in cases],plan,scratch,data_dir)
    return notes


def block(value):
    if value is None:
        return '未记录'
    if value == '':
        return '空字符串'
    return '\n'.join('    '+s for s in str(value).splitlines())


def record_path(phase, stage, uid):
    # The accepted UIDs are exactly the fixed pilot manifest entries.
    return Path("records") / phase / stage / (uid + ".json")


def render(cases, planned, records, phase, prepared, notes, audit):
    overall = audit['official']['overall']
    lines=['# BM25 开发集逐题复核', '', f"范围 {len(cases)} 题，状态 {audit['status']}。"
           f"已确认正确数／计划题数 {overall['correct']}/{overall['total']}，"
           + (f"准确率 {overall['accuracy']:.1%}。" if overall['accuracy'] is not None else "正式准确率暂缺。"), '',
           f"部署与调用统计见 [audit.json](audit.json)。复核前的观察来自 [运行前笔记](pilot_notes.md)。"
           '以下引文是人工核对位置，数据集没有官方支持消息 ID。','']
    for case in cases:
        uid=case['uid']
        source=planned[uid]
        focus,node,rank,quote,check=prepared[uid]
        lines+=['## '+uid,'',source['question'],'',f"提问者 {source['asking_user_id']}。检索用时 {source['retrieval_seconds']:.6f} 秒。",'',
                '**已核验参考答案**','',block(case['gold_answer']),'',
                f"官方最终计分 {case['official_correct']}，在线解析 {case['judge_online_correct']}，judge 格式 {case['judge_parse_status']}。"
                f"完整性问题 {case['integrity_issue'] or '无'}。",'']
        for st in STAGES:
            match=records[(phase,st)].get(uid)
            if not match:
                lines+=[f'**{st.upper()}** 缺失或未通过完整性校验。','']
                continue
            row,line=match
            a=row[st]
            usage=a['usage'] if isinstance(a['usage'],dict) else {}
            response=a['response'] if isinstance(a['response'],dict) else {}
            choices=response.get('choices')
            choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
            lines += [f'**{st.upper()}**　[完整有序请求和原始响应]({record_path(phase,st,uid).as_posix()})（原 JSONL 第 {line} 行）。','',
                      f"状态 `{a['status']}`，格式 `{a['format_status']}`，finish reason `{choice.get('finish_reason')}`。"
                      f"请求 token {a['prompt_tokens']}，响应记录输入/输出 token {usage.get('prompt_tokens')} / {usage.get('completion_tokens')}，"
                      f"用时 {a['seconds']} 秒，缓存来源 {a.get('reused_from_sha256','无')}。",'',
                      '原始文本','',block(a['raw_output']),'','上游解析后','',block(row[runner.ANSWER_FIELD[st]]),'']
            if a['error']:
                lines += ['调用错误','',block(a['error']),'']
        lines += ['**检索顺序与消息定位**','']
        for h in source['retrieved']:
            header=h['formatted_passage'].splitlines()[0]
            meta=dict(pair.split('=',1) for pair in header.strip('[]').split(' / ') if '=' in pair)
            lines += [f"- {h['rank']}. {source['domain']}/{h['msg_node']} · {h['channel']} · {h['speaker']} · "
                      f"{meta.get('timestamp','未载明')} · 回复 {meta.get('reply_to','未载明')}"]
        lines += ['',f"运行前的核对位置为 {node}，排名 {rank}。", '', block(quote),'',focus,'',check,'']
        note=notes[uid]
        lines+=['**初步诊断**','',note['category'] or '待核对','',block(note['assessment']) if note['assessment'] else '待填写。','']
        for citation in note['evidence']:
            lines += [f"{citation['source']} · {citation['domain']}/{citation['msg_node']} · 排名 {citation.get('rank','未检索')}。",'',
                      block(citation['quote']),'']
        lines+=['**人工复核备注**','',note['human_notes'] or '____________________','']
    return '\n'.join(lines)+'\n'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def short_report(audit, cases):
    overall = audit['official']['overall']
    accuracy = f"{overall['accuracy']:.1%}" if overall['accuracy'] is not None else '待完整验证'
    lines = [
        '# BM25 开发集 pilot 复核包', '',
        f"共 {len(cases)} 题，状态为 {audit['status']}。官方正确数为 {overall['correct']}/{overall['total']}，准确率为 {accuracy}。",
        '', '逐题预测、参考答案、官方判定、人工复核栏和原始请求链接见 [cases.md](cases.md)。',
        '修改 [notes.jsonl](notes.jsonl) 的副本可记录复核意见。每题的原始 JSON 包含完整有序的 retrieved、实际 request、response、状态、计时和 token 用量。',
        '', '| 范围 | 正确数 | 计划题数 | 准确率 |', '|---|---:|---:|---:|',
    ]
    for name, group in [*audit['official']['by_domain'].items(), *audit['official']['by_qtype'].items()]:
        value = f"{group['accuracy']:.1%}" if group['accuracy'] is not None else '—'
        lines.append(f"| {name} | {group['correct']} | {group['total']} | {value} |")
    lines.extend([
        '', '按 first 和 pilot 阶段统计的尝试、缓存复用、格式、token、错误、完整性及 CPU/服务端 QA 输入 token 对照见 [audit.json](audit.json)。',
        '冻结配置见 [local.json](config/local.json) 和 [deployment.frozen.json](config/deployment.frozen.json)。',
        '文件 SHA256 见 [SHA256SUMS](SHA256SUMS)。解压后可在包根目录运行 `shasum -a 256 -c SHA256SUMS`。',
        '', '问题按 UID 划分，同领域的开发题和保留题共享对话历史。数据来自 Yang 等人的 GroupMemBench，按 CC BY 4.0 许可使用。',
        '[固定数据版本](https://github.com/UCSB-NLP-Chang/GroupMemBench/tree/461f00c3e8a1a4a3a2ed5814b729014502c4702f) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)',
    ])
    return '\n'.join(lines) + '\n'


def portable_zip(directory, destination):
    """Archive the freshly assembled allowlist. Fix ZIP metadata for repeatable bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        try:
            with ZipFile(stream, mode='w') as archive:
                for path in sorted(directory.rglob('*')):
                    if path.is_file():
                        entry = ZipInfo(path.relative_to(directory).as_posix())
                        entry.external_attr = 0o100644 << 16
                        archive.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def write_export(args, run, audit, records, cases, notes, markdown):
    output = args.output_dir.resolve()
    if args.zip:
        dest_zip = args.zip.resolve()
        require(not dest_zip.exists() and dest_zip != output and output not in dest_zip.parents,
                'ZIP must be a new path outside the output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / 'export'
        staging.mkdir()
        (staging/'config').mkdir()
        for name in ('deployment.frozen.json', 'local.json', 'chat_template.jinja', 'model.sha256.json'):
            shutil.copyfile(run/name, staging/'config'/name)
        shutil.copyfile(CHECKS, staging/'bm25_checks.json')
        shutil.copyfile(WORKSHEET, staging/'pilot_notes.md')
        for (phase, stage), rows in records.items():
            for uid, (row, _) in rows.items():
                # Input and scope validation precede every write into the handoff.
                path = staging / record_path(phase, stage, uid)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_json(path, row)
        (staging/'README.md').write_text(short_report(audit, cases), encoding='utf-8')
        (staging/'cases.md').write_text(markdown, encoding='utf-8')
        report._write_jsonl(staging/'notes.jsonl', notes.values())
        save_json(staging/'audit.json', audit)
        checksums = [f"{c.sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                     for path in sorted(staging.rglob('*')) if path.is_file()]
        (staging/'SHA256SUMS').write_text(''.join(checksums), encoding='utf-8')
        if args.zip:
            portable_zip(staging, dest_zip)
        staging.rename(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['first','pilot'], required=True)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/bm25-real-pilot-001')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, help='existing evaluation.report output for cross-checking pilot')
    parser.add_argument('--notes', type=Path, help='edited copy of notes.jsonl')
    parser.add_argument('--data-dir', type=Path, help='optional pinned data root for post-hoc conversation citations')
    parser.add_argument('--zip', type=Path, help='also create a portable ZIP at this new path')
    args=parser.parse_args()
    run=args.run_dir
    deployment,config=frozen(run)
    checks = read_json(CHECKS)
    input_hash = checks['input_sha256']
    require(checks['manifest_sha256']==c.sha256_file(c.MANIFEST) and
            checks['notes_sha256']==c.sha256_file(WORKSHEET) and
            checks['data_sources_sha256']==c.sha256_file(ROOT/'configs/data_sources.json') and
            checks['chat_template_sha256']==config['qa']['chat_template_sha256'] and
            checks['tokenizer_revision']==config['qa']['tokenizer_revision'], 'fixed checks differ from deployment')
    lengths = {item['uid']: item for item in checks['cases']}
    ordered = c.pilot_uids(read_json(c.MANIFEST)['development'])
    require(len(ordered)==len(lengths)==len(checks['cases'])==12 and list(lengths)==ordered, 'pilot selection drift')
    snapshot = run/'first/first.qa.snapshot.jsonl'
    snapshot_rows, snapshot_hash = runner.read_stable(snapshot, c.read_jsonl)
    require([row['uid'] for row in snapshot_rows]==ordered, 'initial snapshot differs from fixed pilot')
    pilot = {}
    for row in snapshot_rows:
        uid = row['uid']
        source = {key: row[key] for key in c.RETRIEVAL_FIELDS}
        source['qa_status'] = 'not_run'
        require(c.digest_json(source)==lengths[uid]['source_sha256'] and
                c.digest_json(c.qa_messages(source))==lengths[uid]['qa_messages_sha256'],
                f'{uid}: frozen retrieval / QA messages differ')
        pilot[uid] = source
    first = {ordered[0]:pilot[ordered[0]]}
    phase = args.scope
    chosen = first if phase=='first' else pilot
    prepared = worksheet(chosen)
    issues,records,paths,stats=[],{},{},[]
    attempts={st:Counter() for st in STAGES}
    qa_info=runner.provenance('qa',local_model.settings_for(config,'qa'),input_hash,c.MANIFEST)
    files={('first',s):run/f'first/first.{s}.jsonl' for s in STAGES}
    files.update({('pilot',s):run/f'pilot.{s}.jsonl' for s in STAGES})
    for part in (('first',) if phase=='first' else ('first','pilot')):
        planned=first if part=='first' else pilot
        for st in STAGES:
            path=files[part,st]
            parent=snapshot if part=='first' else files[part,'qa']
            info=qa_info if st=='qa' else runner.provenance(
                st,local_model.settings_for(config,st),input_hash,c.MANIFEST,c.sha256_file(parent) if parent.is_file() else None)
            prior=None
            if part=='pilot' and files['first',st].is_file():
                prior=(c.sha256_file(files['first',st]),records['first',st])
            rec=scan(path,st,planned,info,lengths,issues,records.get((part,'qa')),prior)
            records[part,st]=rec
            paths[part,st]=path
            counter,new=counts(part,st,len(planned),rec)
            stats.append(counter)
            attempts[st].update(new)
    if snapshot.is_file():
        try:
            require(c.sha256_file(snapshot)==snapshot_hash, 'first snapshot changed during export')
            snap=snapshot_rows
            require([r['uid'] for r in snap]==list(pilot) and all(r['qa_status']=='pending' for r in snap[1:]),
                    'first-case snapshot does not preserve remaining pilot plan')
            if records['first','qa']:
                require(snap[0]==next(iter(records['first','qa'].values()))[0],'first QA cache differs from snapshot')
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            issues.append({'file':snapshot.name,'kind':'corrupt_snapshot','issue':str(exc)})
    else:
        issues.append({'file':snapshot.name,'kind':'missing_file'})
    if phase=='pilot':
        for st in STAGES:
            uid=next(iter(first))
            if uid in records['first',st] and uid in records['pilot',st]:
                if not records['pilot',st][uid][0][st].get('reused_from_sha256'):
                    issues.append({'uid':uid,'file':files['pilot',st].name,'kind':'first_not_reused'})
    for uid,(row,_) in records[phase,'qa'].items():
        actual=row['qa']['prompt_tokens']
        if actual is not None and actual!=lengths[uid]['prompt_tokens']:
            issues.append({'uid':uid,'kind':'tokenizer_mismatch','cpu':lengths[uid]['prompt_tokens'],'actual':actual})
    cases=[]
    for uid,source in chosen.items():
        rec=records[phase,'judge'].get(uid)
        row=rec[0] if rec else None
        issue=('missing or quarantined judgment' if row is None else
               'unfinished QA or judge call' if row['qa_status']!='ok' or row['judge_status']!='ok' else None)
        # A missing judge carries no verified reference in the raw output.
        case=report._case(source,row,row['gold_answer'] if row else None,issue)
        if row is None:
            available=records[phase,'qa'].get(uid)
            case['case_sha256']=c.digest_json([case['case_sha256'],available[0]['qa']['audit_sha256'] if available else None,
                                               c.sha256_file(run/'deployment.frozen.json')])
        cases.append(case)
    complete=not issues and all(case['integrity_issue'] is None for case in cases)
    def group(subset):
        stats=report._statistics(subset)
        return {**stats,'accuracy':stats['correct_over_planned'] if complete else None}
    official={'overall':group(cases),'by_domain':{d:group([r for r in cases if r['domain']==d]) for d in c.DOMAINS},
              'by_qtype':{t:group([r for r in cases if r['qtype']==t]) for t in c.QTYPES}}
    if args.report_dir:
        require(phase=='pilot','--report-dir expects a 12-question pilot report')
        stored=read_json(args.report_dir/'summary.json')
        require(stored['scope']=='pilot' and stored['split']=='development' and
                stored['input_sha256']==input_hash and stored['manifest_sha256']==checks['manifest_sha256'] and
                stored['judge_sha256']==c.sha256_file(paths[phase,'judge']),
                'normal offline report source differs')
        if complete:
            require(stored['status']=='complete' and all(official[key]==stored[key] for key in official),
                    'official counts differ from the normal offline report')
        official_cases = c.read_jsonl(args.report_dir/'review.jsonl')
        by_uid = runner.index_rows(official_cases)
        require(set(by_uid)==set(chosen), 'official report UID set differs')
        for case in cases:
            existing = by_uid[case['uid']]
            if case['integrity_issue'] is None:
                require(case['case_sha256']==existing['case_sha256'] and
                        case['official_correct']==existing['official_correct'], 'official case differs')
    notes=notes_for(cases,chosen,args.notes,args.data_dir)
    duplicates={st:{uid:n for uid,n in attempts[st].items() if n>1} for st in STAGES}
    audit={'scope':phase,'status':'complete' if complete else 'incomplete','planned':len(chosen),
           'intentionally_deferred_pilot_uids':12-len(chosen),'official':official,'accounting':stats,
           'new_recorded_stage_attempts':{s:attempts[s].total() for s in STAGES},
           'observed_repeat_attempts':{s:sum(n-1 for n in dup.values()) for s,dup in duplicates.items()},
           'issue_counts':dict(Counter(x['kind'] for x in issues)),
           'online_vs_strict':[{k:case[k] for k in ('uid','judge_online_correct','official_correct')}
                               for case in cases if case['official_correct'] is not None and
                               case['official_correct']!=case['judge_online_correct']],
           'extra_recorded_attempts_per_uid':{s:{u:n-1 for u,n in dup.items()} for s,dup in duplicates.items()},
           'accounting_scope':'New stage attempts include recorded starts with unknown delivery and tokenization-only failures. A reused first response counts once. '
              'Unknown delivery remains explicit. Counts cover the saved first and pilot artifacts; other runs are unobserved.',
           'integrity_issues':issues,'artifact_sha256':{f'{k[0]}.{k[1]}':c.sha256_file(p) for k,p in paths.items() if p.is_file()},
           'input_sha256':input_hash,'manifest_sha256':c.sha256_file(c.MANIFEST),'deployment_sha256':c.sha256_file(run/'deployment.frozen.json'),
           'deployment_runtime':deployment['runtime_versions'],'hardware_backend':deployment['apple_runtime'],
           'model':MODEL,'revision':REVISION,'config':config,'evaluation_commit':deployment['evaluation_commit']}
    audit['cpu_token_crosscheck'] = {
        'reference': 'bm25_checks.json', 'tokenizer_revision': checks['tokenizer_revision'],
        'chat_template_sha256': checks['chat_template_sha256'],
        'cases': {uid: {'cpu': lengths[uid]['prompt_tokens'],
                        'server': records[phase,'qa'][uid][0]['qa']['prompt_tokens'] if uid in records[phase,'qa'] else None}
                  for uid in chosen},
    }
    require(not args.output_dir.exists(),'output directory already exists')
    markdown=render(cases,chosen,records,phase,prepared,notes,audit)
    write_export(args, run, audit, records, cases, notes, markdown)
    print(f"{audit['status']}: {len(chosen)} cases, {args.output_dir}")
    return 0 if complete else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError,KeyError,TypeError,ValueError,IndexError,AttributeError) as exc:
        sys.exit(f'export stopped: {exc}')
