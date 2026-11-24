import ast,hashlib,json,re,unittest
from retrieval.data_tools import UPSTREAM,DOMAINS,QTYPES,questions,split_questions,retrieval_input,UP
from retrieval.bm25_retrieve import Retriever,tokenize

class PreparationTests(unittest.TestCase):
    def test_source_hashes(self):
        meta=json.loads((UPSTREAM/'SOURCE.json').read_text())
        for path,sha in meta['files'].items():
            self.assertEqual(hashlib.sha256((UPSTREAM/path).read_bytes()).hexdigest(),sha)
    def test_tokenizer_matches_official(self):
        source=ast.parse((UPSTREAM/'baselines/bm25/eval_benchmark.py').read_text())
        fn=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='tokenize']
        ns={'List':list,'_TOKEN_RE':re.compile(r'[A-Za-z0-9_]+')}
        exec(compile(ast.Module(body=fn,type_ignores=[]),'official_tokenizer','exec'),ns)
        for t in ['User_12: 2025-07-18; R&D','Finance / Phase_2','',None]:
            self.assertEqual(tokenize(t),ns['tokenize'](t))
    def test_no_answer_or_labels_in_retrieval(self):
        q=dict(uid='x',question='deadline?',asking_user_id='User_1',answer='secret',qtype='abstention')
        self.assertEqual(set(retrieval_input(q)),{'uid','question','asking_user_id'})
    def test_index_content_only(self):
        m=dict(content='  A &amp; B ',author='User_SECRET',_channel='SECRET',is_noise=True)
        self.assertEqual(UP['message_index_text'](m),'A & B')
    def test_stratified_disjoint_deterministic_split(self):
        qs=sum((questions(d) for d in DOMAINS),[])
        m=split_questions(qs);self.assertEqual(m,split_questions(qs[::-1]))
        self.assertFalse(set(m['development'])&set(m['evaluation']))
        self.assertEqual(len(m['development'])+len(m['evaluation']),427)
        for d in DOMAINS:
            for t in QTYPES:
                self.assertTrue(any(q.startswith(f'{d}/{t}/') for q in m['development']))
                self.assertTrue(any(q.startswith(f'{d}/{t}/') for q in m['evaluation']))
    def test_topk_and_tie_order(self):
        ms=[dict(msg_node=str(i),_channel='g',author='u',content=f'text {i}') for i in range(4)]
        r=Retriever(ms)
        result=r.retrieve({'question':'absent'},2)
        self.assertEqual([x['msg_node'] for x in result],['0','1'])
        with self.assertRaises(ValueError):r.retrieve({'question':'q'},0)

if __name__=='__main__':unittest.main()
