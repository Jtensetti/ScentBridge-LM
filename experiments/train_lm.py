"""Local Qwen3 LoRA SFT with assistant-only loss, explicit splits and checkpoints."""
import argparse,collections,hashlib,json,math,os,random,time
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model

BASE_REVISION='c1899de289a04d12100db370d81485cdf75e47ca'

def encode_row(row,tokenizer,max_length):
    prompt=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    answer=tokenizer.encode(row['messages'][-1]['content'],add_special_tokens=False)+[tokenizer.eos_token_id]
    if len(prefix)+len(answer)>max_length:return None
    return {'ids':prefix+answer,'labels':[-100]*len(prefix)+answer,'id':row['id'],'task':row['task']}

def load_rows(path,tokenizer,max_length):
    raw=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
    result=[encode_row(r,tokenizer,max_length) for r in raw];accepted=[r for r in result if r is not None]
    return accepted,{'source':len(raw),'accepted':len(accepted),'excluded_over_length':len(raw)-len(accepted),
      'tasks':dict(collections.Counter(r['task'] for r in accepted)),
      'tokens':sum(len(r['ids']) for r in accepted),'assistant_tokens':sum(sum(x!=-100 for x in r['labels']) for r in accepted)}

def batch(row,device):
    x=torch.tensor([row['ids']],dtype=torch.long,device=device)
    return {'input_ids':x,'attention_mask':torch.ones_like(x),'labels':torch.tensor([row['labels']],device=device)}

@torch.no_grad()
def evaluate_loss(model,rows,device):
    model.eval();total=0.;n=0
    for row in rows:
        nt=sum(v!=-100 for v in row['labels']);loss=model(**batch(row,device)).loss.item()
        total+=loss*nt;n+=nt
    model.train();return total/n

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=320)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--eval-every',type=int,default=80);p.add_argument('--lr',type=float,default=0.00015)
    p.add_argument('--rank',type=int,default=8);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU training requires CUDA; CPU inference is supported separately.')
    if args.steps<=0 or args.accumulation<=0:raise ValueError('Positive step counts required')
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    seed=20261013;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device='cuda';dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype!=torch.bfloat16:raise RuntimeError('This verified trainer requires BF16 support.')
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    train,trstats=load_rows(args.data/'train.jsonl',tokenizer,args.max_length)
    dev,dvstats=load_rows(args.data/'validation.jsonl',tokenizer,args.max_length)
    # Stable balanced development subset; test file is never opened here.
    counts=collections.Counter();selected=[]
    for r in dev:
        if counts[r['task']]<4:selected.append(r);counts[r['task']]+=1
    dev=selected
    assert train and dev
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True).to(device)
    model.config.use_cache=False
    config=LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM')
    model=get_peft_model(model,config);model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr,weight_decay=.01)
    params=sum(v.numel() for v in model.parameters() if v.requires_grad)
    initial=evaluate_loss(model,dev,device);torch.cuda.reset_peak_memory_stats()
    report={'base_model':'Qwen/Qwen3-0.6B','base_revision':BASE_REVISION,'seed':seed,'device':torch.cuda.get_device_name(),
      'torch':torch.__version__,'dtype':str(dtype),'trainable_parameters':params,'rank':args.rank,
      'train_data':trstats,'validation_data':dvstats,'validation_loss_rows':len(dev),'initial_validation_loss':initial,
      'max_length':args.max_length,'steps':args.steps,'accumulation':args.accumulation,'learning_rate':args.lr,
      'assistant_only_loss':True,'test_read_during_training':False,'history':[],
      'dataset_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()}
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    started=time.perf_counter();order=list(range(len(train)));random.shuffle(order);cursor=0;seen=set();best=initial;beststep=0
    # Always save a trained checkpoint, even if held-out loss worsens.
    optimizer.zero_grad(set_to_none=True);window=[];supervised_tokens=0
    for step in range(1,args.steps+1):
        model.train()
        for _ in range(args.accumulation):
            if cursor==len(order):random.shuffle(order);cursor=0
            row=train[order[cursor]];cursor+=1;seen.add(row['id'])
            loss=model(**batch(row,device)).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stopping without exporting invalid weights')
            window.append(loss.item());(loss/args.accumulation).backward()
            supervised_tokens+=sum(v!=-100 for v in row['labels'])
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        factor=min(1,step/10)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for g in optimizer.param_groups:g['lr']=args.lr*factor
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if step%10==0 or step==args.steps:
            elapsed=time.perf_counter()-started
            entry={'step':step,'loss':sum(window)/len(window),'seconds':elapsed,
              'eta_seconds':elapsed/step*(args.steps-step),'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20}
            report['history'].append(entry);window=[];print(json.dumps(entry),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            devloss=evaluate_loss(model,dev,device)
            print(json.dumps({'step':step,'validation_loss':devloss}),flush=True)
            report['history'].append({'step':step,'validation_loss':devloss})
            checkpoint=args.out/f'checkpoint-{step}';model.save_pretrained(checkpoint,safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            if beststep==0 or devloss<best:best=devloss;beststep=step
            report.update(selected_step=beststep,selected_validation_loss=best,unique_training_rows_seen=len(seen),
                supervised_tokens_seen=supervised_tokens,elapsed_seconds=time.perf_counter()-started,
                peak_vram_mb=torch.cuda.max_memory_allocated()/2**20)
            (args.out/'training-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE '+str(args.out/f'checkpoint-{beststep}'),flush=True)

if __name__=='__main__':main()
