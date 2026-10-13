"""Compare actual language-model generations, before any runtime fallback."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.runtime import LanguageModel
from scentbridge.contracts import parse_answer,grounded

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--adapter',type=Path)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--per-task',type=int,default=5)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();allrows=[json.loads(l) for l in (args.data/(args.split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    counts=collections.Counter();rows=[]
    for r in allrows:
        if counts[r['task']]<args.per_task:rows.append(r);counts[r['task']]+=1
    args.out.parent.mkdir(parents=True,exist_ok=True)
    progress=args.out.with_suffix('.progress.jsonl');identity=args.out.with_suffix('.run.json')
    config={'base':str(args.base.resolve()),'adapter':str(args.adapter.resolve()) if args.adapter else None,
      'row_ids':[r['id'] for r in rows],'data_sha256':hashlib.sha256((args.data/(args.split+'.jsonl')).read_bytes()).hexdigest(),
      'weight_stat':[(f.name,f.stat().st_size,f.stat().st_mtime_ns) for f in ([args.base] if args.base.is_file() else args.base.glob('*.safetensors'))]}
    config=json.loads(json.dumps(config));results=[]
    if args.resume:
        assert json.loads(identity.read_text())==config,'Evaluation inputs changed'
        results=[json.loads(l) for l in progress.read_text(encoding='utf-8').splitlines()]
        assert [r['id'] for r in results]==[r['id'] for r in rows[:len(results)]]
    else:
        identity.write_text(json.dumps(config,indent=2),encoding='utf-8');progress.write_text('',encoding='utf-8')
    lm=LanguageModel(args.base,args.adapter);started=time.perf_counter()
    for i,row in enumerate(rows[len(results):],start=len(results)):
        raw,stats=lm.generate(row['messages'][:-1],384)
        valid=False;exact=False;schema=False;ground=False;desc=False
        try:
            parsed=json.loads(raw.strip());valid=isinstance(parsed,dict);exact=parsed==row['expected']
            if row['task'] in ['text_to_formula','formula_to_text']:
                answer=parse_answer(raw);schema=True;ground=grounded(answer,row['expected']);desc=answer['description_sv']==row['expected']['description_sv']
        except (ValueError,TypeError,KeyError):pass
        results.append({'id':row['id'],'task':row['task'],'prior_id':row['prior_id'],
          'family':row['accord_family_id'],'valid_json':valid,'exact_json':exact,'domain_schema_valid':schema,
          'formula_grounded':ground,'description_exact':desc,'raw':raw,**stats})
        with progress.open('a',encoding='utf-8') as log:log.write(json.dumps(results[-1],ensure_ascii=False)+'\n')
        print(json.dumps({'i':i+1,'n':len(rows),'task':row['task'],'valid_json':valid,'exact_json':exact}),flush=True)
    bytask={}
    for task in counts:
        rs=[r for r in results if r['task']==task]
        bytask[task]={'n':len(rs),**{k:sum(r[k] for r in rs)/len(rs) for k in ['valid_json','exact_json','domain_schema_valid','formula_grounded','description_exact']}}
    report={'split':args.split,'adapter':str(args.adapter) if args.adapter else None,'n':len(results),
      'context_includes_reference_answer_fields':True,'measures':'schema following and faithful contextual generation, not blind odor prediction',
      'fallback_used':False,'by_task':bytask,'seconds':time.perf_counter()-started,'results':results}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(bytask,indent=2))

if __name__=='__main__':main()
