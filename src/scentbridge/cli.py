"""Command-line access to grounded responses or explicitly unvalidated raw generation."""
import argparse,json
from pathlib import Path
from .runtime import Assistant,Catalog,LanguageModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--catalog',type=Path)
    p.add_argument('--text');p.add_argument('--formula-json');p.add_argument('--accord');p.add_argument('--batch',default='10')
    p.add_argument('--raw',action='store_true',help='Free generation without catalog retrieval or formula validation')
    args=p.parse_args();model=LanguageModel(args.model)
    if args.raw:
        if not args.text:p.error('--raw requires --text')
        text,stats=model.generate([{'role':'system','content':'Du är ScentBridge, en svensk assistent för doftackord. Skilj förslag från provade katalogrecept.'},
            {'role':'user','content':args.text}])
        response={'generation_source':'language_model_raw','catalog_validated':False,'text':text,**stats}
    else:
        if not args.catalog:p.error('--catalog is required unless --raw is used')
        request={'task':'formula_to_text' if args.formula_json else 'text_to_formula','text':args.text or '',
          'prior_id':args.accord,'batch_g':args.batch}
        if args.formula_json:request['components_percent_ww']=json.loads(args.formula_json)
        response=Assistant(model,Catalog(args.catalog)).respond(request)
    print(json.dumps(response,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
