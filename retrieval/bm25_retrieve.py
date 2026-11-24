"""Official BM25 retrieval logic only. No QA, judging, API keys or paid calls."""
import argparse,json,re,time
from pathlib import Path
from rank_bm25 import BM25Okapi
from retrieval.data_tools import DOMAINS,UP,messages,questions,retrieval_input

TOKEN_RE=re.compile(r'[A-Za-z0-9_]+')
def tokenize(text):return [t.lower() for t in TOKEN_RE.findall(text)] if text else []

class Retriever:
    def __init__(self,ms):
        self.messages=ms
        self.index=BM25Okapi([tokenize(UP['message_index_text'](m)) for m in ms])
    def retrieve(self,query,k=10):
        if type(k)is not int or k<1:raise ValueError('top_k must be positive')
        # Same asker-prefixed query as official run_qa.
        asker=query.get('asking_user_id') or ''
        search=f"{asker} {query['question']}" if asker else query['question']
        tokens=tokenize(search)
        if not tokens:return []
        scores=self.index.get_scores(tokens)
        order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [dict(rank=n+1,msg_node=self.messages[i]['msg_node'],channel=self.messages[i]['_channel'],speaker=self.messages[i]['author'],score=float(scores[i]),formatted_passage=UP['format_retrieved_message'](self.messages[i])) for n,i in enumerate(order)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--split',choices=['development','evaluation'],default='development');p.add_argument('--top-k',type=int,default=10);a=p.parse_args()
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    manifest=json.loads(Path(a.manifest).read_text());wanted=set(manifest[a.split]);seen=set();metrics={}
    with dest.open('x') as f:
        for domain in DOMAINS:
            start=time.perf_counter();retriever=Retriever(messages(domain));build=time.perf_counter()-start
            count=0;seconds=0
            for q in questions(domain):
                if q['uid'] not in wanted:continue
                start=time.perf_counter();docs=retriever.retrieve(retrieval_input(q),a.top_k);elapsed=time.perf_counter()-start
                record=dict(uid=q['uid'],domain=domain,qtype=q['qtype'],split=a.split,method='official_bm25_retrieval',question=q['question'],asking_user_id=q.get('asking_user_id'),retrieved=docs,retrieval_seconds=elapsed,qa_status='not_run')
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();seen.add(q['uid']);count+=1;seconds+=elapsed
            metrics[domain]=dict(index_seconds=build,queries=count,retrieval_seconds=seconds)
    if seen!=wanted:raise ValueError('Unmatched manifest IDs; incomplete output')
    dest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
