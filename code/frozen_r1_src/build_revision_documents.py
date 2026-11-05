"""Compile working TeX documents, optionally only in hardware pre-trial idle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


def idle_budget(campaign):
    if campaign is None or (campaign/'campaign_status.json').exists():
        return 45.
    trials=sorted(campaign.glob('trial_*'))
    if not trials:
        return 0.
    latest=trials[-1]
    ready=latest/'worker_ready.json'
    if not ready.exists() or (latest/'initial_status.json').exists():
        return 0.
    # The immutable runner starts its 20-s idle interval immediately after
    # writing ready. Reserve two seconds before that interval can finish.
    return max(0.,18.-(time.time()-ready.stat().st_mtime))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manuscript',type=Path,default=Path('../manuscript'))
    p.add_argument('--campaign',type=Path)
    p.add_argument('--documents',nargs='+',choices=['supplement','main','main_clean','response','assess'],
                   default=['supplement','main','main_clean','response','assess'])
    args=p.parse_args()
    directory=args.manuscript.resolve()
    flags=['-interaction=nonstopmode','-halt-on-error']
    commands=[]
    for name in args.documents:
        engine='xelatex' if name=='assess' else 'pdflatex'
        target=r'\def\CleanRevision{1}\input{main.tex}' if name=='main_clean' else name+'.tex'
        cmd=[engine]+flags+(['-jobname=main_clean'] if name=='main_clean' else [])+[target]
        commands.append((name,cmd))
        if name in ['main','main_clean']:
            commands += [(name,['bibtex',name]),(name,cmd)]
        commands.append((name,cmd))
    rows=[]
    for index,(name,cmd) in enumerate(commands):
        while idle_budget(args.campaign)<8.:
            time.sleep(.3)
        budget=min(40.,idle_budget(args.campaign))
        began=time.monotonic()
        result=subprocess.run(cmd,cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              timeout=budget,encoding='utf-8',errors='replace')
        row=dict(step=index,document=name,command=cmd,returncode=result.returncode,
                 elapsed_s=time.monotonic()-began,utc=datetime.now(timezone.utc).isoformat())
        rows.append(row)
        (directory/'build_working_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
        if result.returncode:
            print(result.stdout[-6000:])
            raise SystemExit(result.returncode)
    import fitz
    documents=[]
    for name in ['supplement','main','main_clean','response','assess']:
        pdf=fitz.open(directory/(name+'.pdf'))
        text='\n'.join(page.get_text() for page in pdf)
        log=(directory/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        row=dict(document=name,pages=len(pdf),replacement_glyph='\ufffd' in text,
            unresolved_pdf_refs='??' in text,
            overfull=re.findall(r'Overfull[^\n]*',log),
            warnings=[line for line in log.splitlines() if ('Warning:' in line and
                      any(term in line for term in ['undefined','multiply','Rerun','changed']))],
            missing_glyphs=[line for line in log.splitlines() if 'Missing character:' in line],
            type3_fonts=sorted({font[3] for page in pdf for font in page.get_fonts() if font[2]=='Type3'}))
        documents.append(row)
    (directory/'document_qa.json').write_text(json.dumps(documents,indent=2),encoding='utf-8')
    print(json.dumps(documents),flush=True)


if __name__=='__main__':
    main()
