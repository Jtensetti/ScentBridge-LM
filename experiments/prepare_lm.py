"""Build reproducible family-separated context-grounded language-model examples."""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.contracts import reference,messages,components,compact

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    forms=[json.loads(l) for l in (args.data/'derived/formulas.jsonl').read_text(encoding='utf-8').splitlines()]
    briefs=json.loads((args.data/'derived/target_descriptions.json').read_text(encoding='utf-8'))
    groups=collections.defaultdict(list)
    for r in forms:groups[r['prior_id']].append(r)
    dataset=collections.defaultdict(list);rng=random.Random(20261013)
    for prior,group in sorted(groups.items()):
        group.sort(key=lambda r:r['variant_index']);split=group[0]['split']
        chosen=[group[i] for i in ([0,1,17,33,49,65,81,113] if split=='train' else [0,17])]
        for f in chosen:
            for i,text in enumerate(briefs[prior]):
                ref=reference(f,briefs[prior][0])
                for task,query in [('text_to_formula',text),('formula_to_text',components(f))]:
                    row={'id':f"lm_{f['formula_id']}_{i}_{task}",'task':task,'formula_id':f['formula_id'],
                      'prior_id':prior,'accord_family_id':f['accord_family_id'],'split':split,
                      'messages':messages(task,query,ref,ref),'expected':ref}
                    # Reverse input is identical across paraphrase index; keep only one.
                    if task=='formula_to_text' and i>0:continue
                    dataset[split].append(row)
    math_tasks={'scale_neat_batch','normalize_parts','complete_percent','dilute_blend','replace_stock','audit_percent_sum'}
    per=collections.Counter()
    for split in ['train','validation','test']:
        for line in (args.data/'derived/sft'/f'{split}.jsonl').open(encoding='utf-8'):
            r=json.loads(line);key=(split,r['prior_id'],r['task'])
            if r['task'] not in math_tasks or per[key]>=(2 if split=='train' else 1):continue
            per[key]+=1
            ms=r['messages'];ms[0]['content']='Du är ScentBridge. Lös blandningsuppgiften och svara endast med JSON. Alla andelar är massa/massa.'
            dataset[split].append({'id':'lm_'+r['record_id'],'task':r['task'],'formula_id':r['formula_id'],
              'prior_id':r['prior_id'],'accord_family_id':r['accord_family_id'],'split':split,'messages':ms,
              'expected':json.loads(ms[-1]['content'])})
    report={'seed':20261013,'source_sha256':hashlib.sha256((args.data/'derived/formulas.jsonl').read_bytes()).hexdigest(),
      'purpose':'context-grounded language generation plus arithmetic auxiliary tasks',
      'not_a_blind_olfactory_benchmark':True,'splits':{}}
    famsets=[]
    for split,rows in dataset.items():
        rng.shuffle(rows);path=args.out/(split+'.jsonl')
        path.write_text(''.join(compact(r)+'\n' for r in rows),encoding='utf-8')
        famsets.append({r['accord_family_id'] for r in rows})
        report['splits'][split]={'rows':len(rows),'families':sorted({r['accord_family_id'] for r in rows}),
            'tasks':dict(collections.Counter(r['task'] for r in rows)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert all(not a&b for i,a in enumerate(famsets) for b in famsets[i+1:])
    (args.out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Runtime knows the approved catalog; test labels are not used for training.
    catalog=[dict(f,descriptions_sv=briefs[f['prior_id']]) for f in forms if f['variant_index']==0]
    (args.out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
