"""Verify hashes and catalog arithmetic without loading model weights into memory."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.contracts import validate_percent,components

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        while chunk:=source.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('release',type=Path);args=p.parse_args();root=args.release.resolve()
    n=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1);target=(root/name).resolve()
        if not target.is_relative_to(root) or sha(target)!=digest:raise ValueError('Hash mismatch: '+name)
        n+=1
    count=0;ids=set()
    with (root/'registered-formulas.jsonl').open(encoding='utf-8') as source:
        for line in source:
            r=json.loads(line);validate_percent(components(r));assert r['formula_id'] not in ids
            ids.add(r['formula_id']);count+=1
    print(json.dumps({'passed':True,'hashed_files':n,'valid_registered_formulas':count}))

if __name__=='__main__':main()
