"""Evaluate intent decoding separately from recipe arithmetic and sensory quality."""
import argparse,json,sys,time
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.runtime import Catalog,LanguageModel
from scentbridge.creative import RecipeDesigner,plan_schema
from scentbridge.contracts import validate_percent

ACCEPTANCE=[
 ('jag vill att det ska lukta aska och eld',{'ash','smoke'},set(),False),
 ('Jasmin och trä med lite citrus',{'jasmine','wood','citrus'},set(),False),
 ('Fikon med mjuk mysk',{'fig','musk'},set(),False),
 ('Vanilj med lite rök',{'vanilla','smoke'},set(),False),
 ('Trä utan rök',{'wood'},{'smoke'},False),
 ('Rök men inte vanilj',{'smoke'},{'vanilla'},False),
 ('Utan rök och vanilj men med ros och jord',{'rose','earth'},{'smoke','vanilla'},False),
 ('Ett parfymrecept med trä men utan blommor och mysk',{'wood'},{'floral','musk'},False),
 ('En doft av nyklippt gräs och gröna äpplen',{'green','apples__grona_applen'},set(),False),
 ('En doft av kaffe',set(),set(),True),
 ('Rök med bensin',{'smoke'},set(),True),
 ('zxqxz',set(),set(),True),
]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lm=LanguageModel(a.model);designer=RecipeDesigner(lm,Catalog(a.catalog));results=[];accept=[]
    try:
        for line in (a.data/'test.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);target=json.loads(row['messages'][-1]['content'])
            raw,stats=lm.generate_structured(row['messages'][:-1],plan_schema(designer.labels))
            try:
                actual=json.loads(raw);exact=actual==target
                facets=set(actual['weights'])==set(target['weights']) and set(actual['avoid'])==set(target['avoid'])
            except Exception:actual=None;exact=False;facets=False
            results.append({'id':row['id'],'exact_intent':exact,'facet_sets_correct':facets,'target':target,'actual':actual,**stats})
        for text,expected,avoid,unsupported in ACCEPTANCE:
            response=designer.respond({'text':text,'recipe_type':'perfume','batch_g':'20'})
            correct=response['status']=='needs_detail' if unsupported else response['status']=='ok'
            if not unsupported and response['status']=='ok':
                correct &= set(response['intent']['weights'])==expected and set(response['intent']['avoid'])==avoid
                p=response['answer']['components_percent_ww'];validate_percent(p)
                t=response['trial'];total=sum(Decimal(v) for v in t['stock_masses_g'].values())+Decimal(t['additional_carrier_g'])
                assert total==Decimal(20)
                assert response['answer']['sensory_status']=='generated_not_smell_tested'
            accept.append({'brief':text,'passed':bool(correct),'response':response})
            print(json.dumps({'brief':text,'passed':bool(correct),'status':response['status']},ensure_ascii=True),flush=True)
    finally:lm.close()
    summary={'model':a.model.name,'held_out_intent_rows':len(results),'exact_intent':sum(r['exact_intent'] for r in results),
        'facet_sets_correct':sum(r['facet_sets_correct'] for r in results),'acceptance_cases':len(accept),'acceptance_passed':sum(r['passed'] for r in accept),
        'sensory_quality_tested':False,'decoder':'learned_synthetic_decoder' if designer.learned_modules else 'authored_module_composition',
        'target_in_encoder_prompt':False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'summary':summary,'held_out_intent':results,'application_acceptance':accept},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
