"""Merge the selected local LoRA checkpoint into standalone safetensors weights."""
import argparse,json,shutil,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--approved-data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',default='ScentBridge-LM-0.1')
    p.add_argument('--decoder',type=Path)
    args=p.parse_args();report=json.loads((args.training/'training-report.json').read_text())
    checkpoint=args.training/f"checkpoint-{report['selected_step']}";args.out.mkdir(parents=True,exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(model,checkpoint,local_files_only=True)
    model=model.merge_and_unload(safe_merge=True);model.config.use_cache=True
    model.save_pretrained(args.out/'model',safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base,local_files_only=True).save_pretrained(args.out/'model')
    shutil.copytree(checkpoint,args.out/'adapter',dirs_exist_ok=True)
    adapter_config=args.out/'adapter/adapter_config.json'
    cfg=json.loads(adapter_config.read_text(encoding='utf-8'))
    cfg['base_model_name_or_path']='Qwen/Qwen3-0.6B';cfg['revision']=report['base_revision']
    adapter_config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    shutil.copy2(args.base/'LICENSE',args.out/'LICENSE-Qwen.txt')
    shutil.copy2(args.base/'download-manifest.json',args.out/'base-download-manifest.json')
    shutil.copy2(args.training/'training-report.json',args.out/'training-report.json')
    shutil.copy2(args.data/'manifest.json',args.out/'training-data-manifest.json')
    shutil.copy2(args.data/'catalog.json',args.out/'catalog.json')
    if args.decoder:shutil.copy2(args.decoder,args.out/'formula-decoder.json')
    # Compact runtime registry; the full provenance remains in the separate data package.
    with (args.out/'registered-formulas.jsonl').open('w',encoding='utf-8') as dest:
        for l in (args.approved_data/'derived/formulas.jsonl').open(encoding='utf-8'):
            r=json.loads(l);small={k:r[k] for k in ['formula_id','prior_id','name_sv','components']}
            dest.write(json.dumps(small,ensure_ascii=False)+'\n')
    repo=Path(__file__).resolve().parents[1]
    shutil.copytree(repo/'src',args.out/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo/'app',args.out/'app',dirs_exist_ok=True)
    shutil.copy2(repo/'requirements-lm.txt',args.out/'requirements.txt')
    shutil.copy2(repo/'configs/requirements-local.lock.txt',args.out/'requirements-local.lock.txt')
    manifest={'model_name':args.name,'base':'Qwen/Qwen3-0.6B','revision':report['base_revision'],
      'modified_by':'ScentBridge local LoRA fine-tuning','merged_parameters':model.num_parameters(),
      'format':'safetensors','dtype':'bfloat16','standalone_merged_model':True,'selected_step':report['selected_step']}
    (args.out/'export-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('EXPORTED '+str(args.out),flush=True)

if __name__=='__main__':main()
