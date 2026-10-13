"""Small NumPy contrastive text/formula model; no pretrained language model.

Learns a shared 32-D embedding from approved descriptions and compositions.
Decoding is catalog retrieval, not generation or calibrated sensory prediction.
"""
import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np

def tokens(text):
    words=re.findall(r'[^\W\d_]+',text.lower())
    return words+['#'+w[i:i+3] for w in words for i in range(len(w)-2)]

def text_matrix(texts,vocab,idf):
    ix={w:i for i,w in enumerate(vocab)}; x=np.zeros((len(texts),len(vocab)))
    for j,s in enumerate(texts):
        for w,c in collections.Counter(tokens(s)).items():
            if w in ix:x[j,ix[w]]=1+np.log(c)
    x*=idf
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def formula_matrix(rows,materials):
    ix={m:i for i,m in enumerate(materials)};x=np.zeros((len(rows),len(materials)))
    for j,r in enumerate(rows):
        for c in r['components']:
            if c['ingredient_id'] in ix:x[j,ix[c['ingredient_id']]]=np.sqrt(float(c['weight_percent'])/100)
    return x

def encode(x,w):
    h=np.tanh(x@w);n=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
    return h/n,(x,h,n)

def backward(g,z,cache):
    x,h,n=cache
    return x.T@((g-z*np.sum(g*z,axis=1,keepdims=True))/n*(1-h*h))

def loss_grad(x,y,a,b,temp=.12):
    u,uc=encode(x,a);v,vc=encode(y,b);s=u@v.T/temp;n=len(x)
    def soft(q):
        z=q-q.max(axis=1,keepdims=True);ex=np.exp(z);p=ex/ex.sum(axis=1,keepdims=True)
        return p, -np.mean(np.diag(z)-np.log(ex.sum(axis=1)))
    p,l1=soft(s);q,l2=soft(s.T);g=(p+q.T-2*np.eye(n))/(2*n*temp)
    return (l1+l2)/2, backward(g@v,u,uc),backward(g.T@u,v,vc)

def evaluate(rows, briefs, catalog, vocab,idf,materials,a,b):
    # Every split is evaluated against ALL 39 baseline candidates (same catalog).
    # Held-out candidates are inference inputs, never gradient-training examples.
    catids=[r['prior_id'] for r in catalog]
    fe,_=encode(formula_matrix(catalog,materials),b)
    textc=np.stack([encode(text_matrix(briefs[k],vocab,idf),a)[0].mean(0) for k in catids])
    textc/=np.maximum(np.linalg.norm(textc,axis=1,keepdims=True),1e-12)
    result={}
    for split in ['train','validation','test']:
        subset=[r for r in rows if r['split']==split]; priorids=sorted({r['prior_id'] for r in subset})
        if not subset:continue
        ts=[s for k in priorids for s in briefs[k]];truth=[k for k in priorids for s in briefs[k]]
        te,_=encode(text_matrix(ts,vocab,idf),a);pred=np.argmax(te@fe.T,axis=1)
        tacc=np.mean([catids[p]==k for p,k in zip(pred,truth)])
        re,_=encode(formula_matrix(subset,materials),b);rp=np.argmax(re@textc.T,axis=1)
        facc=np.mean([catids[p]==r['prior_id'] for p,r in zip(rp,subset)])
        unknown=sorted({c['ingredient_id'] for r in subset for c in r['components'] if c['ingredient_id'] not in materials})
        result[split]={'text_to_formula_prior_top1':float(tacc),'formula_to_text_prior_top1':float(facc),
          'formula_queries':len(subset),'text_queries':len(ts),'priors':len(priorids),
          'families':len({r['accord_family_id'] for r in subset}),'unseen_material_ids':unknown,
          'catalog_size':len(catalog),'random_top1':1/len(catalog)}
    return result

def train(args):
    started=time.perf_counter();rng=np.random.default_rng(20261013)
    data=args.data/'derived/formulas.jsonl'
    rows=[json.loads(l) for l in data.read_text(encoding='utf-8').splitlines()]
    briefs=json.loads((args.data/'derived/target_descriptions.json').read_text(encoding='utf-8'))
    trainrows=[r for r in rows if r['split']=='train'];groups=collections.defaultdict(list)
    for r in trainrows:groups[r['prior_id']].append(r)
    priors=sorted(groups);texts=[s for k in priors for s in briefs[k]]
    vocab=sorted({w for s in texts for w in tokens(s)})
    df=collections.Counter(w for s in texts for w in set(tokens(s)))
    idf=np.array([1+np.log((1+len(texts))/(1+df[w])) for w in vocab])
    materials=sorted({c['ingredient_id'] for r in trainrows for c in r['components']})
    x=text_matrix(texts,vocab,idf).reshape(len(priors),3,-1)
    y=np.stack([formula_matrix(sorted(groups[k],key=lambda r:r['variant_index']),materials) for k in priors])
    assert y.shape[1]==128
    a=rng.normal(0,.08,(len(vocab),32)); b=rng.normal(0,.08,(len(materials),32))
    catalog=sorted([r for r in rows if r['variant_index']==0],key=lambda r:r['prior_id'])
    # Validation only during selection: do not compute held-out test metrics yet.
    devrows=[r for r in rows if r['split']!='test']
    def dev_metrics():
        # evaluate handles empty test inputs via separate split list below.
        return evaluate(devrows,briefs,catalog,vocab,idf,materials,a,b)
    m=[np.zeros_like(a),np.zeros_like(b)];v=[np.zeros_like(a),np.zeros_like(b)]
    history=[];best=-1;best_epoch=None;bestw=None
    initial=dev_metrics();seen=set()
    for epoch in range(1,args.epochs+1):
        bi=rng.integers(0,3,len(priors));vi=(epoch-1)%128
        loss,ga,gb=loss_grad(x[np.arange(len(priors)),bi],y[:,vi],a,b)
        seen.update((k,vi) for k in priors)
        for i,(w,g) in enumerate([(a,ga),(b,gb)]):
            g+=1e-4*w;m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
            w-=.003*(m[i]/(1-.9**epoch))/(np.sqrt(v[i]/(1-.999**epoch))+1e-8)
        if epoch%50==0 or epoch==args.epochs:
            metrics=dev_metrics();d=metrics['validation'];score=(d['text_to_formula_prior_top1']+d['formula_to_text_prior_top1'])/2
            history.append({'epoch':epoch,'loss':float(loss),'validation_score':score})
            if (epoch>=128 or epoch==args.epochs) and score>best:best=score;bestw=(a.copy(),b.copy());best_epoch=epoch
            print(json.dumps(history[-1]),flush=True)
    a,b=bestw;metrics=evaluate(rows,briefs,catalog,vocab,idf,materials,a,b)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'weights.npz',text_projection=a,formula_projection=b,idf=idf)
    config={'model_type':'paired_contrastive_catalog_retrieval','version':1,'seed':20261013,
      'vocab':vocab,'materials':materials,'latent_dim':32,'catalog':[
        {'formula_id':r['formula_id'],'prior_id':r['prior_id'],'split':r['split'],'name_sv':r['name_sv'],
         'components':r['components'],'descriptions_sv':briefs[r['prior_id']]} for r in catalog],
      'decoding':'retrieve_approved_baseline_catalog_entry','out_of_distribution_scores_calibrated':False}
    (args.out/'model.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'status':'experimental_retrieval_model_trained','language_model_trained':False,
      'parameters':a.size+b.size,'latent_dim':32,'seed':20261013,'numpy_version':np.__version__,
      'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'training_formulas_seen':len(seen),
      'training_descriptions':len(texts),'unique_training_priors':len(priors),'epochs_run':args.epochs,
      'selected_epoch':best_epoch,'selection':'mean of validation bidirectional prior retrieval accuracy',
      'initial_dev':initial,'metrics':metrics,'history':history,'seconds':time.perf_counter()-started,
      'test_used_for_selection':False,'split_unit':'accord_family_id',
      'limitations':['Only 3 held-out test families; variants share descriptions and are correlated.',
        'Uses approved catalog descriptions, not numeric sensory ratings.',
        'Some descriptions explicitly name ingredients; this is not a blinded olfactory test.',
        'All 39 candidate catalog entries are exposed at inference, including held-out families.',
        'Top1 measures prior identity, not exact proportions; text cannot distinguish all 128 variants.',
        'No novel-formula generator or free-text language decoder; retrieval scores are not confidence.']}
    (args.out/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','initial_dev']},indent=2))

def query(args):
    cfg=json.loads((args.model/'model.json').read_text(encoding='utf-8'));w=np.load(args.model/'weights.npz',allow_pickle=False)
    cat=cfg['catalog'];a=w['text_projection'];b=w['formula_projection']
    if args.text:
        x=text_matrix([args.text],cfg['vocab'],w['idf'])
        if not np.any(x):raise ValueError('No known text features; cannot rank reliably.')
        q,_=encode(x,a);gallery,_=encode(formula_matrix(cat,cfg['materials']),b)
    else:
        comp=json.loads(args.formula)
        if not comp or any(not np.isfinite(float(v)) or float(v)<0 for v in comp.values()) or abs(sum(map(float,comp.values()))-100)>1e-6:
            raise ValueError('Expected finite nonnegative percentages totaling 100.')
        unknown=set(comp)-set(cfg['materials'])
        if unknown:raise ValueError('Untrained material IDs: '+','.join(sorted(unknown)))
        x=formula_matrix([{'components':[{'ingredient_id':k,'weight_percent':v} for k,v in comp.items()]}],cfg['materials'])
        q,_=encode(x,b)
        gallery=np.stack([encode(text_matrix(r['descriptions_sv'],cfg['vocab'],w['idf']),a)[0].mean(0) for r in cat])
        gallery/=np.maximum(np.linalg.norm(gallery,axis=1,keepdims=True),1e-12)
    scores=(q@gallery.T)[0];indices=np.argsort(-scores)[:3]
    print(json.dumps({'experimental':True,'score_is_probability':False,'results':[
        {'score':float(scores[i]),**cat[i]} for i in indices]},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--epochs',type=int,default=600);p.add_argument('--model',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--text');g.add_argument('--formula')
    args=p.parse_args()
    if args.model and (args.text or args.formula):query(args)
    elif args.data and args.out and args.epochs>0:train(args)
    else:p.error('Supply --data and --out for training, or --model and --text/--formula for inference.')
