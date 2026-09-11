"""Compute full-dataset prediction-guided I1/I4 rankings and composites."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np
from lane_eval.manifest import ManifestDataset
from evaluation.readiness_metrics import build_metric_record

def _resolve(p):
    if not p: return None
    q=Path(p)
    if q.exists(): return q
    m=Path('/home/karthikab/Projects/road-readiness-framework')
    try: r=Path.cwd()/q.relative_to(m); return r if r.exists() else q
    except ValueError: return q

def _work(item):
    dataset,model,sid,imgp,predp,extra=item
    im=cv2.imread(str(imgp),cv2.IMREAD_COLOR)
    pm=cv2.imread(str(predp),cv2.IMREAD_GRAYSCALE) if predp else None
    if im is None or pm is None: return {'dataset':dataset,'model':model,'sample_id':sid,'I1':None,'I4':None,'image_path':str(imgp),'reason':'image_or_prediction_unavailable'}
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    try: rec=build_metric_record(type('S',(),{'image_id':sid,'image_path':str(imgp),'gt_mask_path':None,'gt_lane_json':None,'metadata':{}})(),pm,rgb,dataset=dataset,model_name=model)
    except Exception as e: return {'dataset':dataset,'model':model,'sample_id':sid,'I1':None,'I4':None,'image_path':str(imgp),'reason':str(e)}
    return {'dataset':dataset,'model':model,'sample_id':sid,'I1':rec.get('I1_pred_pattern_continuity'),'I4':rec.get('I4_pred_lane_width_stability'),'pattern':rec.get('I1_pred_pattern_type'),'condition':rec.get('I1_pred_condition'),'image_path':str(imgp),'pred_path':str(predp)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('outputs/full_model_rankings')); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--datasets',nargs='*',default=['bdd100k_lane','culane','curvelanes','tusimple']); a=ap.parse_args()
    allout=[]
    for ds in a.datasets:
        samples={str(s.image_id):s for s in ManifestDataset(f'manifests/{ds}/manifest_{ds}.json')}
        sources={'yolopx': list(csv.DictReader((a.root/'results/per_image'/f'yolopx_{ds}_per_image.csv').open())) if (a.root/'results/per_image'/f'yolopx_{ds}_per_image.csv').exists() else json.loads((a.root/'results/per_image'/f'yolopx_{ds}_per_image.json').read_text()), 'clrernet':list(csv.DictReader((Path('outputs/results/per_image_csv')/f'clrernet_{ds}_per_image_metrics.csv').open()))}
        for model,rows in sources.items():
            items=[]
            for row in rows:
                sid=str(row.get('image_id',row.get('sample_id',''))); s=samples.get(sid)
                if not s: continue
                pp=(a.root/'predictions/yolopx'/ds/'masks'/f'{sid}.png') if model=='yolopx' else _resolve(row.get('pred_mask_path'))
                items.append((ds,model,sid,s.image_path,pp,row))
            with ProcessPoolExecutor(max_workers=a.workers) as ex: out=list(ex.map(_work,items,chunksize=16))
            allout.extend(out); print(ds,model,len(out),flush=True)
    outdir=a.root/'rankings'; outdir.mkdir(parents=True,exist_ok=True)
    for ds in a.datasets:
      for model in ('yolopx','clrernet'):
       rows=[r for r in allout if r['dataset']==ds and r['model']==model]
       for metric in ('I1','I4'):
        valid=sorted([r for r in rows if r.get(metric) is not None],key=lambda r:r[metric],reverse=True)
        d=outdir/ds/model; d.mkdir(parents=True,exist_ok=True)
        for name,data in [('top50',valid[:50]),('worst',list(reversed(valid[-50:])))]: (d/f'{metric}_{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False))
       (outdir/ds/model/'summary.json').write_text(json.dumps({'dataset':ds,'model':model,'total':len(rows),'I1_valid':sum(r.get('I1') is not None for r in rows),'I4_valid':sum(r.get('I4') is not None for r in rows)},indent=2))
    (a.root/'i1_i4_full_records.json').write_text(json.dumps(allout,allow_nan=False)); print('saved',len(allout))
if __name__=='__main__': main()
