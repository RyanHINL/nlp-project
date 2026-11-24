"""Fetch only pinned Finance/Technology files, verifying each SHA256."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,help='Existing official GroupMemBench checkout, optional');a=p.parse_args()
    meta=json.loads((ROOT/'configs/data_sources.json').read_text());target=ROOT/'data/GroupMemBench'
    for name,digest in meta['files'].items():
        out=target/name;out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            if hashlib.sha256(out.read_bytes()).hexdigest()!=digest:raise ValueError(f'Existing file differs from pinned dataset: {name}')
            continue
        temporary=out.with_suffix(out.suffix+'.partial')
        if a.source_dir:shutil.copyfile(a.source_dir/name,temporary)
        else:
            url=f"https://raw.githubusercontent.com/UCSB-NLP-Chang/GroupMemBench/{meta['commit']}/{name}"
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2','--max-time','180',url,'-o',str(temporary)],check=True)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:raise ValueError(f'Download checksum failed: {name}')
        temporary.replace(out)
    print(f"Verified {len(meta['files'])} pinned data files")
if __name__=='__main__':main()
