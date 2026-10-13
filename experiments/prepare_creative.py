"""Synthetic intent supervision, kept distinct from sensory evidence.

No formula or target intent is included in a training prompt. A split of
unseen facet pairs and unseen wording is reserved for semantic evaluation.
"""
import argparse,hashlib,itertools,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.runtime import Catalog
from scentbridge.creative import FACETS,ALIASES,options,planner_messages

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    catalog=Catalog(a.catalog);labels=options(catalog);rng=random.Random(20261013)
    terms={k:v[1][:5] for k,v in FACETS.items()}
    terms.update({k:ALIASES.get(k,[v['name_sv'].lower()]) for k,v in catalog.by_prior.items()})
    records={'train':[],'validation':[],'test':[]};seen=set()
    def add(split,text,weights,avoid=None,unsupported=None):
        if text in seen:return
        seen.add(text)
        chat=planner_messages(text,labels)
        chat.append({'role':'assistant','content':json.dumps({'weights':weights,'avoid':avoid or [],'unsupported':unsupported or []},ensure_ascii=False,separators=(',',':'))})
        records[split].append({'id':hashlib.sha256(text.encode()).hexdigest()[:16],'task':'odor_intent','messages':chat,'evidence':'synthetic_language_supervision_not_sensory_test'})
    for k,ts in terms.items():
        for t in ts:
            for template in ['Jag vill ha en doft av {}.','Skapa ett ackord som luktar {}.','Det ska dofta {}.']:
                add('train',template.format(t),{k:3})
        add('validation','Doftidén är '+ts[0]+'.',{k:3})
    pairs=list(itertools.combinations(sorted(terms),2));rng.shuffle(pairs)
    held={tuple(sorted(x)) for x in [('smoke','ash'),('jasmine','wood'),('vanilla','smoke'),('citrus','musk'),('rose','earth'),('fig','wood')]}
    used=0
    for x,y in pairs:
        if (x,y) in held:continue
        add('train',f'Jag söker {terms[x][0]} och {terms[y][0]}.',{x:3,y:3})
        add('train',f'Mest {terms[x][0]} med lite {terms[y][0]}.',{x:5,y:1})
        add('train',f'{terms[x][0]}, utan {terms[y][0]}.',{x:3},[y])
        used+=1
        if used>=110:break
    for x,y in sorted(held):
        add('test',f'En doft som påminner om {terms[x][0]} tillsammans med {terms[y][0]}.',{x:3,y:3})
        add('test',f'{terms[x][0]} men ingen {terms[y][0]}.',{x:3},[y])
    unknown=['havssalt','kaffe','bensin','nybakat bröd','ozon','regnvåt asfalt','banan','läder','metall','gurka']
    for u in unknown:
        add('train',f'Jag vill att det luktar {u}.',{},unsupported=[u])
        add('train',f'Rök och {u}.',{'smoke':3},unsupported=[u])
        add('validation',f'Doft av {u}, tack.',{},unsupported=[u])
    # Real supplied descriptions label an existing accord concept only; formulas
    # and sensory approval are not used as labels for these synthetic tasks.
    for k,row in catalog.by_prior.items():
        for desc in row['descriptions_sv'][:2]:add('train',desc,{k:3})
    a.out.mkdir(parents=True,exist_ok=True)
    for split,rows in records.items():
        rng.shuffle(rows)
        (a.out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={'task':'brief_to_odor_intent','label_evidence':'authored_synthetic_language_labels_not_sensory_observations',
        'counts':{k:len(v) for k,v in records.items()},'target_in_prompt':False,'held_out_facet_pairs':[list(x) for x in sorted(held)],
        'limitation':'Intent accuracy does not establish recipe smell or quality; final acceptance prompts kept separate.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest))
if __name__=='__main__':main()
