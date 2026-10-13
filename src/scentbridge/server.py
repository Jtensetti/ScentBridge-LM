"""Loopback-only local UI. No remote inference or telemetry."""
import argparse,json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .runtime import Catalog,LanguageModel,Assistant

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--ui',type=Path,required=True);p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    catalog=Catalog(args.catalog);assistant=Assistant(LanguageModel(args.model),catalog);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,value,kind='application/json; charset=utf-8'):
            data=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(data)
        def valid_host(self):return self.headers.get('Host') in {f'127.0.0.1:{args.port}',f'localhost:{args.port}'}
        def do_GET(self):
            if not self.valid_host():self.reply(403,{'error':'Invalid host'});return
            if self.path=='/':self.reply(200,args.ui.read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/api/catalog':self.reply(200,{'accords':[{'id':r['prior_id'],'name':r['name_sv']} for r in catalog.rows],
                'materials':catalog.materials,'registered_formulas':len(catalog.exact),'model':'ScentBridge 0.2 — receptverkstad'})
            elif self.path=='/api/health':self.reply(200,{'ready':True,'device':assistant.lm.device,'version':'0.2',
                'release_root':str(args.model.resolve().parent)})
            else:self.reply(404,{'error':'Not found'})
        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.valid_host() or (origin and origin not in {f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}'}):
                self.reply(403,{'error':'Local requests only'});return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.reply(415,{'error':'JSON required'});return
            if self.path=='/api/shutdown':
                self.reply(200,{'stopping':True});threading.Thread(target=server.shutdown,daemon=True).start();return
            if self.path!='/api/respond':self.reply(404,{'error':'Not found'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16000:raise ValueError('Förfrågan är för stor eller tom.')
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict):raise ValueError('Ogiltig förfrågan.')
                if not lock.acquire(blocking=False):self.reply(429,{'error':'Modellen arbetar. Försök igen om en stund.'});return
                try:response=assistant.respond(request)
                finally:lock.release()
                self.reply(200,response)
            except (ValueError,TypeError,KeyError,ArithmeticError) as exc:self.reply(400,{'error':str(exc)})
            except Exception as exc:
                print(type(exc).__name__,str(exc),flush=True)
                self.reply(500,{'error':'Modellen kunde inte slutföra svaret. Se den lokala loggen.'})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'SCENTBRIDGE_READY http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    finally:
        if hasattr(assistant.lm,'close'):assistant.lm.close()
        server.server_close()

if __name__=='__main__':main()
