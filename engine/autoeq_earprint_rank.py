#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

F_MIN=1000.0
F_MAX=12000.0
POINTS=600
SMOOTH_FWHM_OCTAVES=1.0/12.0
REGIONS=((1000.0,6000.0,1.00,"1-6k"),(6000.0,8000.0,0.70,"6-8k"),(8000.0,10000.0,0.40,"8-10k"),(10000.0,12000.0,0.15,"10-12k"))

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--earprint',default='output/pure_earprint_dynamic.txt')
    p.add_argument('--autoeq-root',default='AutoEq')
    p.add_argument('--source',default='Jaytiss')
    p.add_argument('--output-prefix',default='output/iem_earprint_autoeq_jaytiss')
    return p.parse_args()

def read_pairs(path:Path, delimiter:str):
    f=[]; y=[]
    with path.open('r',encoding='utf-8-sig',newline='') as fh:
        for row in csv.reader(fh,delimiter=delimiter):
            if len(row)<2: continue
            try: ff=float(row[0].strip()); yy=float(row[1].strip())
            except (ValueError,AttributeError): continue
            if math.isfinite(ff) and math.isfinite(yy) and ff>0:
                f.append(ff); y.append(yy)
    if len(f)<4: raise ValueError('not enough numeric samples')
    f=np.asarray(f,float); y=np.asarray(y,float)
    order=np.argsort(f); f=f[order]; y=y[order]
    uf,inv=np.unique(f,return_inverse=True)
    if len(uf)!=len(f):
        sums=np.zeros(len(uf)); counts=np.zeros(len(uf))
        np.add.at(sums,inv,y); np.add.at(counts,inv,1.0)
        y=sums/counts; f=uf
    return f,y

def smooth(v,grid):
    step=math.log2(grid[1]/grid[0])
    sigma=SMOOTH_FWHM_OCTAVES/(2.355*step)
    return gaussian_filter1d(v,sigma=sigma,mode='nearest')

def prepare(f,y,grid):
    if f[0]>F_MIN or f[-1]<F_MAX: raise ValueError('measurement does not fully cover 1-12 kHz')
    z=np.interp(grid,f,y); z=smooth(z,grid); return z-float(z[0])

def weights(grid):
    w=np.zeros_like(grid)
    for lo,hi,wt,_ in REGIONS:
        m=(grid>=lo) & ((grid<hi) if hi<F_MAX else (grid<=hi)); w[m]=wt
    w[grid==F_MAX]=REGIONS[-1][2]
    return w

def wrms(e,w): return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))
def wmae(e,w): return float(np.sum(w*np.abs(e))/np.sum(w))

def regional(grid,e):
    out={}
    for lo,hi,_,label in REGIONS:
        m=(grid>=lo) & ((grid<hi) if hi<F_MAX else (grid<=hi))
        out[label]=float(np.sqrt(np.mean(e[m]**2)))
    return out

def corr(a,b):
    if np.std(a)==0 or np.std(b)==0: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def main():
    cfg=args(); root=Path(__file__).resolve().parents[1]
    earprint_path=root/cfg.earprint; autoeq_root=(root/cfg.autoeq_root).resolve()
    source_dir=autoeq_root/'measurements'/cfg.source/'data'/'in-ear'
    if not source_dir.exists(): raise FileNotFoundError(f'Missing AutoEq source directory: {source_dir}')
    grid=np.geomspace(F_MIN,F_MAX,POINTS); w=weights(grid)
    ef,ey=read_pairs(earprint_path,'\t'); ear=prepare(ef,ey,grid)
    rows=[]; skipped=[]
    for path in sorted(source_dir.rglob('*.csv')):
        try:
            f,y=read_pairs(path,','); curve=prepare(f,y,grid)
        except Exception as exc:
            skipped.append({'path':str(path.relative_to(autoeq_root)),'reason':str(exc)}); continue
        e=curve-ear; rr=regional(grid,e)
        rows.append({'name':path.stem,'source':cfg.source,'path':str(path.relative_to(autoeq_root)),
                     'weighted_rms_db':wrms(e,w),'weighted_mae_db':wmae(e,w),'shape_correlation':corr(curve,ear),
                     'rms_1_6k_db':rr['1-6k'],'rms_6_8k_db':rr['6-8k'],'rms_8_10k_db':rr['8-10k'],
                     'rms_10_12k_db':rr['10-12k'],'coverage_hz':f'{f[0]:.1f}-{f[-1]:.1f}'})
    rows.sort(key=lambda r:(r['weighted_rms_db'],r['weighted_mae_db'],-r['shape_correlation'],r['name'].lower()))
    for rank,row in enumerate(rows,1): row['rank']=rank
    prefix=root/cfg.output_prefix; prefix.parent.mkdir(parents=True,exist_ok=True)
    meta={'engine':'AutoEq EarPrint Matcher','earprint_file':cfg.earprint,'autoeq_source':cfg.source,
          'frequency_range_hz':[F_MIN,F_MAX],'points':POINTS,'anchor':'1 kHz','smoothing':'1/12-octave FWHM Gaussian',
          'primary_metric':'weighted RMS','secondary_metric':'weighted MAE',
          'weights':{label:wt for _,_,wt,label in REGIONS},'candidate_count':len(rows),'skipped_count':len(skipped),
          'note':'FR shape match only; not a Harman or target preference score.'}
    fields=['rank','name','source','weighted_rms_db','weighted_mae_db','shape_correlation','rms_1_6k_db','rms_6_8k_db','rms_8_10k_db','rms_10_12k_db','coverage_hz','path']
    with prefix.with_suffix('.csv').open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with prefix.with_suffix('.json').open('w',encoding='utf-8') as fh:
        json.dump({'metadata':meta,'results':rows,'skipped':skipped},fh,ensure_ascii=False,indent=2)
    top=prefix.with_name(prefix.name+'_top100.txt')
    with top.open('w',encoding='utf-8') as fh:
        fh.write('AUTOEQ EARPRINT MATCH — TOP 100\n'); fh.write(f'Source: {cfg.source}\n'); fh.write('Match band: 1–12 kHz\n'); fh.write('Lower RMS = closer FR shape after 1 kHz anchoring\n\n')
        fh.write('Rank | IEM | RMS | MAE | Corr | 1-6k | 6-8k | 8-10k | 10-12k\n'+'-'*115+'\n')
        for r in rows[:100]:
            fh.write(f"{r['rank']:>4} | {r['name']} | {r['weighted_rms_db']:.3f} | {r['weighted_mae_db']:.3f} | {r['shape_correlation']:.4f} | {r['rms_1_6k_db']:.2f} | {r['rms_6_8k_db']:.2f} | {r['rms_8_10k_db']:.2f} | {r['rms_10_12k_db']:.2f}\n")
    print(f'Ranked {len(rows)} usable AutoEq {cfg.source} IEM measurements.')
    print(f'Skipped {len(skipped)} measurements.')
    print(prefix.with_suffix('.csv')); print(prefix.with_suffix('.json')); print(top)

if __name__=='__main__': main()
