"""Render raw/prediction side-by-side panels for ranking JSON files."""
from pathlib import Path
import argparse,json,cv2
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('outputs/full_model_rankings')); a=p.parse_args()
 for f in a.root.glob('rankings/*/*/*_top50.json'):
  parts=f.parts; ds,model=parts[-3],parts[-2]; metric=f.name.split('_')[0]; rows=json.loads(f.read_text())
  for kind,subset in [('top50',rows),('worst',json.loads((f.parent/f'{metric}_worst.json').read_text()))]:
   out=a.root/'images'/metric/ds/model/kind; out.mkdir(parents=True,exist_ok=True)
   for i,r in enumerate(subset,1):
    im=cv2.imread(r['image_path']); pm=cv2.imread(r.get('pred_path',''),cv2.IMREAD_GRAYSCALE) if r.get('pred_path') else None
    if im is None: continue
    right=im.copy()
    if pm is not None:
     if pm.shape!=im.shape[:2]: pm=cv2.resize(pm,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)
     right[pm>0]=(0.55*right[pm>0]+0.45*__import__('numpy').array([60,210,90])).astype('uint8')
    score=r.get(metric); txt=f'{metric}: unavailable' if score is None else f'{metric}: {float(score):.3f}'
    for panel,title in ((im,'RAW'),(right,model.upper()+' PRED')):
     cv2.rectangle(panel,(0,0),(panel.shape[1],48),(20,20,20),-1); cv2.putText(panel,title,(12,30),0,.8,(255,255,255),2)
    cv2.putText(right,txt,(12,78),0,.7,(80,220,255),2)
    cv2.imwrite(str(out/f'{i:02d}_{r["sample_id"]}.jpg'),cv2.hconcat([im,right]),[cv2.IMWRITE_JPEG_QUALITY,92])
if __name__=='__main__': main()
