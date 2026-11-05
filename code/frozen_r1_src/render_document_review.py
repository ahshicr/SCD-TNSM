"""Render selected new manuscript pages for visual QA, not scientific data editing."""
import argparse
import json
from pathlib import Path
import time
import fitz
from PIL import Image
from src.build_revision_documents import idle_budget


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manuscript',type=Path,default=Path('../manuscript'))
    p.add_argument('--campaign',type=Path)
    args=p.parse_args()
    out=args.manuscript/'qa_pages'/'current_20260914'
    out.mkdir(parents=True,exist_ok=True)
    selections={
        'main':[1,4,9,10,11,14,15,16],
        'supplement':[1,4,6,7,8,9,10,11,12,13,14],
        'response':[1,4,9,10,11,12,17,26,27,30],
        'assess':[1,5,10],
    }
    for name,pages in selections.items():
        document=fitz.open(args.manuscript/(name+'.pdf'))
        for page in pages:
            while idle_budget(args.campaign)<6.:
                time.sleep(.3)
            target=out/('%s_%02d.png'%(name,page))
            document[page-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(target)
            print(json.dumps(dict(document=name,page=page,path=str(target.resolve()))),flush=True)
        thumbs=[]
        for page in document:
            pix=page.get_pixmap(matrix=fitz.Matrix(.32,.32),alpha=False)
            thumbs.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
        columns=4
        gap=8
        width=columns*thumbs[0].width+(columns-1)*gap
        rows=(len(thumbs)+columns-1)//columns
        height=rows*thumbs[0].height+(rows-1)*gap
        contact=Image.new('RGB',(width,height),'#d0d0d0')
        for index,thumb in enumerate(thumbs):
            x=(index%columns)*(thumb.width+gap)
            y=(index//columns)*(thumb.height+gap)
            contact.paste(thumb,(x,y))
        target=out/(name+'_contact.png')
        contact.save(target)
        print(json.dumps(dict(document=name,pages=len(document),contact=str(target.resolve()))),flush=True)


if __name__=='__main__':
    main()
