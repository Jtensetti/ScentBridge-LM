"""Generate traceable hypothetical brief/formula pairs and distill a sparse decoder.

This distils explicit additive design priors; its synthetic test is not sensory
validation. Never merge these records into the user-approved observation set.
"""
import argparse,collections,hashlib,json,random,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.runtime import Catalog
from scentbridge.creative import MODULES,FACETS,ALIASES,options,rounded_composition,planner_messages
from scentbridge.contracts import components,validate_percent

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=6000);a=p.parse_args()
    if a.count<1000:raise ValueError('Use at least 1000 rows for this experiment')
    cat=Catalog(a.catalog);labels=options(cat)
    modules={**{k:components(v) for k,v in cat.by_prior.items()},**MODULES}
    keys=sorted(modules);ingredients=sorted({k for m in modules.values() for k in m})
    rng=random.Random(20261013);rows=[];seen_formula=set();seen_text=set()
    synonyms={k:v[1][:5] for k,v in FACETS.items()}
    synonyms.update({k:ALIASES.get(k,[row['name_sv'].lower()]) for k,row in cat.by_prior.items()})
    templates=['Jag vill ha {parts}.','Doften ska förena {parts}.','Skapa ett ackord med {parts}.',
        'Min doftidé är {parts}.','Jag tänker mig {parts}.','Det ska dofta {parts}.']
    attempts=0
    while len(rows)<a.count:
        attempts+=1
        if attempts>100*a.count:raise RuntimeError('Unique recipe target not reached')
        selected=[keys[len(rows)]] if len(rows)<len(keys) else sorted(rng.sample(keys,rng.choice([2,2,3])))
        weights={k:rng.choice([1,3,5]) for k in selected}
        values={}
        for k,w in weights.items():
            for material,v in modules[k].items():values[material]=values.get(material,Decimal(0))+Decimal(str(v))*w
        formula=rounded_composition(values);validate_percent(formula)
        fingerprint=hashlib.sha256(json.dumps(formula,sort_keys=True).encode()).hexdigest()
        parts=[]
        for k in selected:
            term=rng.choice(synonyms[k]);w=weights[k]
            parts.append(('lite ' if w==1 else 'tydlig ' if w==5 else '')+term)
        text=rng.choice(templates).format(parts=' tillsammans med '.join(parts))
        if fingerprint in seen_formula or text in seen_text:continue
        seen_formula.add(fingerprint);seen_text.add(text)
        group='|'.join(selected);bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10
        split='train' if len(selected)==1 or bucket>1 else 'test' if bucket==0 else 'validation'
        parents=[cat.by_prior[k]['formula_id'] for k in selected if k in cat.by_prior]
        rows.append({'id':'syn_'+fingerprint[:20],'description_sv':text,'intent':weights,
            'components_percent_ww':formula,'split':split,'composition_group':group,
            'sensory_status':'synthetic_not_smell_tested','approval':False,
            'mapping_method':'authored_odor_modules_additive_mass_composition',
            'parent_formula_ids':parents,'authored_modules':[k for k in selected if k in MODULES],
            'prediction_scope':'intended_odor_association_not_measured_mixture_perception'})
    a.out.mkdir(parents=True,exist_ok=True)
    write=lambda path,rs:path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rs),encoding='utf-8')
    write(a.out/'pairs.jsonl',rows)
    for split in ['train','validation','test']:
        subset=[r for r in rows if r['split']==split];write(a.out/f'{split}.jsonl',subset)
        instructions=[]
        for r in subset:
            forward=[{'role':'system','content':'Du är en receptmodell. Föreslå ett doftackord som JSON med components_percent_ww. Receptet är en oprövad hypotes.'},
                {'role':'user','content':r['description_sv']},
                {'role':'assistant','content':json.dumps({'components_percent_ww':r['components_percent_ww'],'sensory_status':'synthetic_not_smell_tested'},ensure_ascii=False)}]
            instructions.append({'id':r['id']+'_forward','task':'text_to_novel_formula','messages':forward,'evidence':'synthetic'})
            reverse=[{'role':'system','content':'Beskriv den avsedda doftbilden i en syntetisk formel. Det är ingen luktobservation.'},
                {'role':'user','content':json.dumps(r['components_percent_ww'])},
                {'role':'assistant','content':r['description_sv']}]
            instructions.append({'id':r['id']+'_reverse','task':'formula_to_intended_description','messages':reverse,'evidence':'synthetic'})
        write(a.out/f'sft-{split}.jsonl',instructions)
    # Stable nonlinguistic sparse intent vector. No test or validation rows fit W.
    def arrays(subset):
        x=np.zeros((len(subset),len(keys)));y=np.zeros((len(subset),len(ingredients)))
        for i,r in enumerate(subset):
            total=sum(r['intent'].values())
            for k,v in r['intent'].items():x[i,keys.index(k)]=v/total
            for k,v in r['components_percent_ww'].items():y[i,ingredients.index(k)]=float(v)
        return x,y
    x,y=arrays([r for r in rows if r['split']=='train'])
    w,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    # Remove numerical residue, enforce nonnegative mass and normalized rows.
    w=np.where(w<.0001,0,w);w=w/w.sum(axis=1,keepdims=True)*100
    metrics={}
    for split in ['train','validation','test']:
        xx,yy=arrays([r for r in rows if r['split']==split]);error=np.abs(xx@w-yy)
        metrics[split]={'rows':len(xx),'mean_absolute_percentage_point_error':float(error.mean()),
            'max_absolute_percentage_point_error':float(error.max())}
    decoder={'version':'0.2','kind':'nonnegative_linear_intent_to_mass_distillation',
        'trained_on':'synthetic_recipe_pairs_only','sensory_quality_validated':False,
        'features':keys,'materials':ingredients,'fit_rank':int(rank),
        'modules':{key:{material:round(float(w[i,j]),9) for j,material in enumerate(ingredients) if w[i,j]>0} for i,key in enumerate(keys)}}
    (a.out/'formula-decoder.json').write_text(json.dumps(decoder,ensure_ascii=False,indent=2),encoding='utf-8')
    groups={s:{r['composition_group'] for r in rows if r['split']==s} for s in metrics}
    assert not groups['train']&groups['test'] and not groups['train']&groups['validation'] and not groups['test']&groups['validation']
    manifest={'pairs':len(rows),'unique_descriptions':len(seen_text),'unique_formulas':len(seen_formula),
        'sensory_status':'synthetic_not_smell_tested','user_approval_inherited':False,'seed':20261013,
        'split_unit':'unordered_set_of_odor_facets_all_ratios_grouped','cross_split_group_overlap':0,
        'decoder_fit_rows':len(x),'decoder_parameters':int(np.count_nonzero(w)),'decoder_fit_rank':int(rank),
        'metrics':metrics,'test_used_for_fitting':False,'direct_lm_sft_rows':len(rows)*2,
        'direct_lm_sft_status':'prepared_not_trained; deployed decoder is the fitted sparse linear specialist',
        'corpus_sha256':hashlib.sha256((a.out/'pairs.jsonl').read_bytes()).hexdigest(),
        'limitation':'Synthetic reconstruction accuracy measures fidelity to authored mixing rules, not real smell quality or empirical generalization.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
