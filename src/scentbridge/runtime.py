"""Offline inference, approved-catalog grounding and deterministic mass scaling."""
import collections,json,math,re,time,unicodedata
from decimal import Decimal
from pathlib import Path
from .contracts import messages,reference,components,parse_answer,grounded,scale,validate_percent

def words(s):
    s=unicodedata.normalize('NFKC',s).lower()
    aliases={'pistage':'pistasch','pistagenötter':'pistasch','pistagenöt':'pistasch'}
    return [aliases.get(w,w) for w in re.findall(r'[^\W\d_]+',s)]

def formula_key(p):
    return tuple(sorted((k,Decimal(str(v))) for k,v in p.items() if Decimal(str(v))!=0))

class Catalog:
    def __init__(self,path):
        self.rows=json.loads(Path(path).read_text(encoding='utf-8'))
        self.by_prior={r['prior_id']:r for r in self.rows}
        self.exact={formula_key(components(r)):r for r in self.rows}
        full=Path(path).with_name('registered-formulas.jsonl')
        if full.exists():
            with full.open(encoding='utf-8') as source:
                for line in source:
                    r=json.loads(line);r['descriptions_sv']=self.by_prior[r['prior_id']]['descriptions_sv']
                    self.exact[formula_key(components(r))]=r
        self.materials={c['ingredient_id']:c['name_sv'] for r in self.rows for c in r['components']}
        docs=[set(words(r['name_sv']+' '+' '.join(r['descriptions_sv']))) for r in self.rows]
        df=collections.Counter(w for d in docs for w in d)
        self.idf={w:1+math.log((1+len(docs))/(1+n)) for w,n in df.items()};self.docs=docs

    def search(self,text):
        normalized=' '.join(words(text))
        exact=[r for r in self.rows if normalized in [' '.join(words(s)) for s in [r['name_sv'],*r['descriptions_sv']]]]
        if len(exact)==1:return exact[0],exact
        stops={'jag','vill','ett','en','som','med','och','doft','doften','ackord','skapa','den','det','av','till','för','i','söker','parfym','blandning'}
        query=set(words(text))-stops;scores=[]
        for r,doc in zip(self.rows,self.docs):
            name=set(words(r['name_sv']));overlap=query&doc
            score=sum(self.idf[w] for w in overlap)+4*len(query&name)
            # Swedish inflections/compounds, e.g. pistage -> pistagenötter.
            partial={w for w in doc if any(min(len(q),len(w))>=4 and (w.startswith(q) or q.startswith(w)) for q in query)}-overlap
            score+=sum(.65*self.idf[w] for w in partial)+2*len(partial&name)
            # A name match carries more evidence than generic briefing language.
            distinctive=(overlap|partial)-stops
            if not distinctive:score=0
            scores.append((score,r))
        scores.sort(key=lambda x:x[0],reverse=True)
        if not scores or scores[0][0]==0:return None,[]
        # Tied alternatives require a user selection; do not invent confidence.
        if len(scores)>1 and abs(scores[0][0]-scores[1][0])<1e-9:return None,[r for _,r in scores[:3]]
        return scores[0][1],[r for _,r in scores[:3]]

class TransformersLanguageModel:
    def __init__(self,path,adapter=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch;torch.set_num_threads(4)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype=torch.bfloat16 if self.device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=self.dtype,
            attn_implementation='sdpa',local_files_only=True).to(self.device)
        if adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,adapter,local_files_only=True)
        self.model.eval()

    def generate(self,chat,max_new_tokens=384):
        prompt=self.tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=self.tokenizer(prompt,return_tensors='pt').to(self.device)
        if encoded.input_ids.shape[1]>1536:raise ValueError('Förfrågan är för lång.')
        started=time.perf_counter()
        with self.torch.inference_mode():
            out=self.model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,
                temperature=None,top_p=None,top_k=None,pad_token_id=self.tokenizer.eos_token_id)
        new=out[0,encoded.input_ids.shape[1]:]
        return self.tokenizer.decode(new,skip_special_tokens=True),{'seconds':time.perf_counter()-started,'generated_tokens':len(new)}

class NativeLanguageModel:
    """llama.cpp process owned by this instance, exposed only on loopback."""
    def __init__(self,path):
        import atexit,os,socket,subprocess,urllib.request
        self.urllib=urllib.request;self.path=Path(path).resolve();self.process=None
        bundled=self.path.parent/'runtime/llama-server.exe'
        cached_gpu=self.path.parent/'../../work/llama/bin/llama-server.exe'
        override=os.environ.get('SCENTBRIDGE_LLAMA_BINARY')
        candidates=[Path(override)] if override else [cached_gpu.resolve(),bundled]
        error=None
        for exe in candidates:
            if not exe.exists():continue
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            self.url=f'http://127.0.0.1:{port}'
            gpu='cuda' in str(exe).lower() or exe==cached_gpu.resolve()
            self.device='llama.cpp CUDA' if gpu else 'llama.cpp CPU'
            self.log=(self.path.parent/'native-runtime.log').open('w',encoding='utf-8')
            args=[str(exe),'-m',str(self.path),'--host','127.0.0.1','--port',str(port),
              '-c','2048','-np','1','-t','4','-ngl','99' if gpu else '0','--reasoning','off','--no-webui']
            self.process=subprocess.Popen(args,stdout=self.log,stderr=subprocess.STDOUT,
              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            for _ in range(120):
                if self.process.poll() is not None:break
                try:
                    with self.urllib.urlopen(self.url+'/health',timeout=1) as r:
                        if r.status==200:atexit.register(self.close);return
                except OSError:pass
                time.sleep(.25)
            self.close();error='llama.cpp failed to start; see native-runtime.log'
        raise RuntimeError(error or 'Missing bundled llama.cpp runtime.')

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except Exception:self.process.kill()
        if hasattr(self,'log') and not self.log.closed:self.log.close()

    def generate(self,chat,max_new_tokens=384):
        if len(json.dumps(chat,ensure_ascii=False))>16000:raise ValueError('Förfrågan är för lång.')
        payload={'messages':chat,'max_tokens':max_new_tokens,'temperature':0,'top_p':1,'seed':20261013,
          'chat_template_kwargs':{'enable_thinking':False},'stream':False}
        req=self.urllib.Request(self.url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        started=time.perf_counter()
        with self.urllib.urlopen(req,timeout=180) as r:result=json.load(r)
        return result['choices'][0]['message']['content'],{'seconds':time.perf_counter()-started,
            'generated_tokens':result.get('usage',{}).get('completion_tokens',0),'engine':self.device}

def LanguageModel(path,adapter=None):
    if Path(path).suffix=='.gguf':
        if adapter:raise ValueError('GGUF must already contain the merged adapter.')
        return NativeLanguageModel(path)
    return TransformersLanguageModel(path,adapter)

class Assistant:
    def __init__(self,model,catalog):self.lm=model;self.catalog=catalog

    def respond(self,request):
        task=request.get('task','text_to_formula');text=str(request.get('text','')).strip()
        if len(text)>2000:raise ValueError('Beskrivningen får vara högst 2 000 tecken.')
        if task not in {'text_to_formula','formula_to_text'}:raise ValueError('Okänd uppgift.')
        supplied=request.get('components_percent_ww')
        row=None
        if task=='formula_to_text':
            validate_percent(supplied)
            unknown=set(supplied)-set(self.catalog.materials)
            if unknown:raise ValueError('Formeln innehåller råvaror utanför den lokala paletten.')
            row=self.catalog.exact.get(formula_key(supplied))
            if row is None:
                return {'status':'outside_catalog','message':'Den här proportioneringen finns inte bland dina 4 992 godkända katalogformler. Jag kan därför inte koppla en luktprovad beskrivning till den.',
                    'calculation':scale(supplied,request.get('batch_g','10')),'sensory_status':'not_documented_for_this_formula'}
            query=supplied
        else:
            prior=request.get('prior_id')
            if prior:
                if prior not in self.catalog.by_prior:raise ValueError('Okänt ackord.')
                row=self.catalog.by_prior[prior]
            else:
                if not text:raise ValueError('Beskriv en doft eller välj ett ackord.')
                row,alternatives=self.catalog.search(text)
                if row is None:return {'status':'choose_accord','message':'Välj ett ackord ur katalogen för en entydig träff.',
                    'alternatives':[{'prior_id':r['prior_id'],'name_sv':r['name_sv']} for r in alternatives]}
            query=text or row['name_sv']
        ref=reference(row,row['descriptions_sv'][0]);chat=messages(task,query,ref)
        calculation=scale(ref['components_percent_ww'],request.get('batch_g','10'))
        raw,stats=self.lm.generate(chat)
        try:
            answer=parse_answer(raw)
            if not grounded(answer,ref) or answer['description_sv']!=ref['description_sv']:
                raise ValueError('Response deviated from approved catalog evidence')
            source='language_model';reason=None
        except (ValueError,TypeError,KeyError):
            answer=ref;source='catalog_fallback';reason='Modellsvaret avvek från katalogen. Den verifierade katalogposten visas.'
        return {'status':'ok','answer':answer,'formula_id':row['formula_id'],'prior_id':row['prior_id'],
            'generation_source':source,'fallback_reason':reason,'calculation':calculation,'model_stats':stats,
            'evidence':'Luktprovad och godkänd enligt användarens bekräftelse.',
            'retrieval':'approved_catalog; proportions and description supplied as model context'}
