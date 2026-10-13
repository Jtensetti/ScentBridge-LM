"""Convert a merged release with the pinned llama.cpp checkout; bundle CPU runtime."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--llama-source',type=Path,required=True,help='llama.cpp b11429 checkout')
    p.add_argument('--cpu-runtime',type=Path,required=True,help='Verified Windows x64 CPU release extraction')
    args=p.parse_args();converter=args.llama_source/'convert_hf_to_gguf.py'
    if not converter.is_file() or not (args.cpu_runtime/'llama-server.exe').is_file():raise ValueError('Missing converter/runtime')
    subprocess.run([sys.executable,str(converter),str(args.release/'model'),'--outfile',
        str(args.release/'scentbridge-q8.gguf'),'--outtype','q8_0'],check=True)
    shutil.copytree(args.cpu_runtime,args.release/'runtime',dirs_exist_ok=True)
    shutil.copy2(args.llama_source/'LICENSE',args.release/'LICENSE-llama.cpp.txt')
    print('Native Q8 release prepared; run the same held-out evaluation before distribution.')

if __name__=='__main__':main()
