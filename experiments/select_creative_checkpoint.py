"""Select among saved checkpoints using all development rows, never test rows."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from train_lm import load_rows,evaluate_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--training',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.training/'training-report.json').read_text())
    assert (a.training/f"checkpoint-{report['steps']}").is_dir(),'Training must finish first'
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    dev,stats=load_rows(a.data/'validation.jsonl',tok,report['max_length'])
    base=AutoModelForCausalLM.from_pretrained(a.base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    entries=[];model=None
    for checkpoint in sorted(a.training.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])):
        step=int(checkpoint.name.split('-')[-1]);name=f'step{step}'
        if model is None:model=PeftModel.from_pretrained(base,checkpoint,adapter_name=name,local_files_only=True)
        else:model.load_adapter(checkpoint,adapter_name=name,local_files_only=True)
        model.set_adapter(name);loss=evaluate_loss(model,dev,'cuda')
        entry={'step':step,'validation_loss':loss,'rows':len(dev)};entries.append(entry);print(json.dumps(entry),flush=True)
    best=min(entries,key=lambda x:x['validation_loss'])
    report.update(selected_step=best['step'],selected_validation_loss=best['validation_loss'],
        checkpoint_selection='all_validation_rows_post_training_no_test',full_validation_selection=entries)
    (a.training/'training-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
