"""Hash a local release and create a self-contained CPU/GGUF portable archive."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('release',type=Path);a=p.parse_args();root=a.release
    files=[f for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts
        and f.suffix!='.log' and f.name!='SHA256SUMS.txt']
    hashes={f.relative_to(root).as_posix():digest(f) for f in files}
    manifest=lambda h:''.join(f'{v}  {k}\n' for k,v in h.items())
    (root/'SHA256SUMS.txt').write_text(manifest(hashes),encoding='utf-8')
    portable={k:v for k,v in hashes.items() if not k.startswith(('model/','adapter/'))}
    target=root.with_name(root.name+'-portable.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in portable:z.write(root/name,root.name+'/'+name)
        z.writestr(root.name+'/SHA256SUMS.txt',manifest(portable))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,expected in portable.items():
            h=hashlib.sha256()
            with z.open(root.name+'/'+name) as f:
                while chunk:=f.read(8*1024*1024):h.update(chunk)
            assert h.hexdigest()==expected,name
    print(json.dumps({'archive':str(target),'files':len(portable)+1,'bytes':target.stat().st_size,'all_hashes_verified':True}))
if __name__=='__main__':main()
