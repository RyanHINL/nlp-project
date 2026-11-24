"""CPU-only preparation; upstream pure loaders are reused without API imports."""
from __future__ import annotations
import argparse,ast,hashlib,html,json,math
from pathlib import Path
from typing import Any,Dict,List,Optional,Tuple

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'vendor/GroupMemBench'
DATA=ROOT/'data/GroupMemBench'
DOMAINS=('Finance','Technology')
QTYPES=('multi_hop','knowledge_update','temporal','user_implicit','term_ambiguity','abstention')

def load_upstream():
    names={'load_conversation_messages','message_index_text','format_retrieved_message'}
    path=UPSTREAM/'baselines/rag_common/eval_lib.py'
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope=dict(json=json,html=html,Dict=Dict,List=List,Any=Any)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope
UP=load_upstream()

def messages(domain):
    return UP['load_conversation_messages'](str(DATA/f'data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json'))

def questions(domain):
    out=[]
    for typ in QTYPES:
        for line in (DATA/f'questions/{domain}/{typ}.jsonl').read_text().splitlines():
            if not line.strip():continue
            q=json.loads(line)
            q.update(uid=f"{domain}/{typ}/{q['id']}",domain=domain,qtype=typ)
            out.append(q)
    if len({q['uid'] for q in out})!=len(out):raise ValueError('Duplicate question identity')
    return out

def retrieval_input(q):
    # Explicit allowlist: no answers, evaluation labels or synthetic annotations.
    return {'uid':q['uid'],'question':q['question'],'asking_user_id':q.get('asking_user_id') or ''}

def split_questions(all_questions,seed=4051,fraction=.2):
    if not 0<fraction<1:raise ValueError('Invalid development fraction')
    dev=[];test=[]
    for domain in DOMAINS:
        for typ in QTYPES:
            group=[q for q in all_questions if q['domain']==domain and q['qtype']==typ]
            group.sort(key=lambda q:hashlib.sha256(f"{seed}:{q['uid']}".encode()).hexdigest())
            n=max(1,min(len(group)-1,math.ceil(len(group)*fraction))) if len(group)>1 else 0
            dev.extend(q['uid'] for q in group[:n]);test.extend(q['uid'] for q in group[n:])
    normalized=lambda q:(q['domain'],' '.join(q['question'].lower().split()))
    devtext={normalized(q) for q in all_questions if q['uid'] in set(dev)}
    testtext={normalized(q) for q in all_questions if q['uid'] in set(test)}
    if devtext & testtext:raise ValueError('Exact duplicate question crosses split; resolve grouping before proceeding')
    return dict(seed=seed,development_fraction=fraction,unit='question; conversations are shared within domain',development=dev,evaluation=test)

def prepare(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report={};allq=[]
    for domain in DOMAINS:
        ms=messages(domain);qs=questions(domain);allq+=qs
        ids=[m['msg_node'] for m in ms]
        if len(set(ids))!=len(ids):raise ValueError(f'Duplicate message IDs in {domain}')
        idset=set(ids)
        report[domain]=dict(messages=len(ms),channels=len({m['_channel'] for m in ms}),speakers=len({m['author'] for m in ms}),questions=len(qs),question_counts={t:sum(q['qtype']==t for q in qs) for t in QTYPES},missing_reply_targets=sum(bool(m.get('reply_to')) and m['reply_to'] not in idset for m in ms),empty_content=sum(not UP['message_index_text'](m) for m in ms),question_fields=sorted(set().union(*(q.keys() for q in qs))))
    manifest=split_questions(allq)
    (out/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for split in ['development','evaluation']:
        selected=[q for q in allq if q['uid'] in set(manifest[split])]
        with (out/f'{split}_queries.jsonl').open('w') as f:
            for q in selected:f.write(json.dumps(retrieval_input(q))+'\n')
    print(json.dumps({'audit':report,'development':len(manifest['development']),'evaluation':len(manifest['evaluation'])},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();prepare(a.output)
