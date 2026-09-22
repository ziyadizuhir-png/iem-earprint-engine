/* ============================================================
   MOONDROP PUDDING PEQ ENGINE
   v2026-09-22.5.1

   Squiglink-style AutoEQ PK-only optimizer for MOONDROP Link.

   Principles adapted from AutoEq:
   - level alignment before tonal optimization
   - smooth residual / error
   - largest residual feature first
   - Fc from feature location
   - Q estimated from feature bandwidth
   - gain from feature height
   - residual peeling after each initialized filter
   - constrained continuous refinement
   - MSE objective
   - AutoEq-style sharpness penalty around 18 dB/oct
   - >10 kHz treated as average energy, not point-by-point chasing

   Important:
   This is AutoEq-inspired, NOT a literal copy of AutoEq.
   RBJ peaking response is retained as the provisional DSP model
   because MOONDROP PUDDING's internal DSP implementation is not
   publicly specified.
   ============================================================ */
(function () {
  'use strict';

  const CFG = Object.freeze({
    version: '2026-09-22.5.2',
    bands: 10,
    minFreq: 20,
    maxFreq: 12000,
    optHi: 10000,
    minGain: -12,
    maxGain: 3,
    minQ: 0.50,
    maxQ: 2,
    sampleRate: 48000,
    points: 360,
    alignLo: 100,
    alignHi: 10000,
    smoothSpan: 9,
    minSeedGain: 0.20,
    minSeedSeparationOct: 0.18,
    minActiveGain: 0.05,
    trebleStart: 7000,
    maxPasses: 3
  });

  const $ = id => document.getElementById(id);
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));

  function logspace(a,b,n){
    const out=[];
    const la=Math.log(a), lb=Math.log(b);
    for(let i=0;i<n;i++){
      if(i===0){out.push(a);continue;}
      if(i===n-1){out.push(b);continue;}
      out.push(Math.exp(la+(lb-la)*i/(n-1)));
    }
    return out;
  }

  function parseCurveText(text){
    const rows=[];
    const re=/^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:Hz)?\s*(?:,|;|\t|\s+)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:dB)?\s*$/i;
    for(const raw of String(text||'').replace(/^\uFEFF/,'').split(/\r?\n/)){
      const s=raw.trim();
      if(!s||s[0]==='#'||s.startsWith('//')) continue;
      const m=s.match(re);
      if(!m) continue;
      const f=Number(m[1]), v=Number(m[2]);
      if(Number.isFinite(f)&&Number.isFinite(v)&&f>0) rows.push([f,v]);
    }
    if(rows.length<10) throw Error('Curve needs at least 10 numeric frequency/response points.');
    for(let i=1;i<rows.length;i++) if(rows[i][0]<=rows[i-1][0]) throw Error('Frequencies must be strictly ascending.');
    return rows;
  }

  function interp(c,f){
    const eps=1e-7;
    if(f<c[0][0]-eps||f>c[c.length-1][0]+eps) throw Error('Curve does not cover '+f+' Hz.');
    if(f<=c[0][0]+eps) return c[0][1];
    if(f>=c[c.length-1][0]-eps) return c[c.length-1][1];
    let lo=0,hi=c.length-1;
    while(lo<hi){const m=(lo+hi)>>1;if(c[m][0]<f)lo=m+1;else hi=m;}
    const i=Math.max(1,lo), a=c[i-1], b=c[i];
    const t=(Math.log(f)-Math.log(a[0]))/(Math.log(b[0])-Math.log(a[0])||1);
    return a[1]+(b[1]-a[1])*t;
  }

  const resample=(c,f)=>f.map(x=>interp(c,x));

  function smooth(v,span){
    const out=new Array(v.length), r=Math.max(1,Math.floor(span/2));
    for(let i=0;i<v.length;i++){
      let s=0,n=0;
      for(let j=Math.max(0,i-r);j<=Math.min(v.length-1,i+r);j++){s+=v[j];n++;}
      out[i]=s/n;
    }
    return out;
  }

  function percentile(a,p){
    if(!a.length)return 0;
    const x=a.slice().sort((m,n)=>m-n), q=(x.length-1)*p;
    const i=Math.floor(q),j=Math.ceil(q);
    return i===j?x[i]:x[i]+(x[j]-x[i])*(q-i);
  }

  function rmse(a){
    if(!a.length)return 0;
    let s=0;for(const x of a)s+=x*x;
    return Math.sqrt(s/a.length);
  }

  function rbj(freqs,b){
    const out=new Float64Array(freqs.length);
    if(Math.abs(b.gain)<1e-12)return out;

    const A=Math.pow(10,b.gain/40);
    const w0=2*Math.PI*b.freq/CFG.sampleRate;
    const alpha=Math.sin(w0)/(2*b.q), cw=Math.cos(w0);
    const b0=1+alpha*A,b1=-2*cw,b2=1-alpha*A;
    const a0=1+alpha/A,a1=-2*cw,a2=1-alpha/A;

    for(let i=0;i<freqs.length;i++){
      const w=2*Math.PI*freqs[i]/CFG.sampleRate,c=Math.cos(w),s=Math.sin(w);
      const c2=2*c*c-1, sc=2*s*c;
      const nr=b0+b1*c+b2*c2, ni=-b1*s-b2*sc;
      const dr=a0+a1*c+a2*c2, di=-a1*s-a2*sc;
      const mag=Math.sqrt(nr*nr+ni*ni)/Math.max(Math.sqrt(dr*dr+di*di),1e-15);
      out[i]=20*Math.log10(Math.max(mag,1e-12));
    }
    return out;
  }

  function totalEq(freqs,bands){
    const out=new Float64Array(freqs.length);
    for(const b of bands){
      if(Math.abs(b.gain)<1e-9)continue;
      const r=rbj(freqs,b);
      for(let i=0;i<out.length;i++)out[i]+=r[i];
    }
    return out;
  }

  function cloneBands(b){return b.map(x=>({freq:x.freq,gain:x.gain,q:x.q}));}

  function gridInterp(freqs,vals,f){
    if(f<=freqs[0])return vals[0];
    if(f>=freqs[freqs.length-1])return vals[vals.length-1];
    let lo=0,hi=freqs.length-1;
    while(lo<hi){const m=(lo+hi)>>1;if(freqs[m]<f)lo=m+1;else hi=m;}
    const i=Math.max(1,lo);
    const t=(Math.log(f)-Math.log(freqs[i-1]))/(Math.log(freqs[i])-Math.log(freqs[i-1]));
    return vals[i-1]+(vals[i]-vals[i-1])*t;
  }

  function alignLevel(freqs,raw,target){
    let s=0,n=0;
    for(let i=0;i<freqs.length;i++){
      if(freqs[i]>=CFG.alignLo&&freqs[i]<=CFG.alignHi){s+=raw[i]-target[i];n++;}
    }
    const offset=n?s/n:0;
    return {offset,curve:raw.map(v=>v-offset)};
  }

  function alignLevel(freqs,raw,target){
    let s=0,n=0;
    for(let i=0;i<freqs.length;i++){
      if(freqs[i]>=CFG.alignLo&&freqs[i]<=CFG.alignHi){s+=raw[i]-target[i];n++;}
    }
    const offset=n?s/n:0;
    return {offset,curve:raw.map(v=>v-offset)};
  }

  /*
    Squiglink Lab's equalizer.js AutoEQ algorithm, adapted to the
    MOONDROP PUDDING / MOONDROP Link constraint set.

    Source architecture: squiglink/lab equalizer.js
    - TrebleStartFrom = 7000 Hz
    - AutoEQRange = 20–15000 Hz (Pudding output is capped at 12 kHz)
    - Q = 0.5–2
    - candidate thresholds = 1 dB then 0.5 dB
    - two directional coordinate optimization
    - merge close filters + delete unnecessary filters
    - distance = mean absolute error, ignoring errors < 0.1 dB
  */
  function applyFilters(fr,bands,freqs){
    const out=new Float64Array(fr);
    for(const b of bands){
      if(Math.abs(b.gain)<1e-12)continue;
      const r=rbj(freqs,b);
      for(let i=0;i<out.length;i++)out[i]+=r[i];
    }
    return out;
  }

  function distance(freqs,curve,target){
    let d=0;
    for(let i=0;i<freqs.length;i++){
      const e=Math.abs(curve[i]-target[i]);
      if(e>=0.1)d+=e;
    }
    return d/freqs.length;
  }

  function freqUnit(freq){
    if(freq<100)return 1;
    if(freq<1000)return 10;
    if(freq<10000)return 100;
    return 1000;
  }

  function strip(filters){
    return filters.map(f=>({
      freq:Math.floor(f.freq-f.freq%freqUnit(f.freq)),
      q:clamp(Math.floor(f.q*10)/10,CFG.minQ,CFG.maxQ),
      gain:clamp(Math.floor(f.gain*10)/10,CFG.minGain,CFG.maxGain)
    }));
  }

  function searchCandidates(freqs,fr,frTarget,threshold){
    let state=0,startIndex=-1;
    const candidates=[];
    for(let i=0;i<fr.length;i++){
      const delta=fr[i]-frTarget[i];
      const abs=Math.abs(delta);
      const next=(abs<threshold)?0:(delta/abs);
      if(next===state)continue;
      if(startIndex>=0){
        if(state!==0){
          const start=fr[startIndex]!==undefined?freqs[startIndex]:freqs[startIndex];
          const end=freqs[i];
          const center=Math.sqrt(start*end);
          const gain=gridInterp(freqs.slice(startIndex,i),frTarget.slice(startIndex,i),center)
                    -gridInterp(freqs.slice(startIndex,i),fr.slice(startIndex,i),center);
          const q=center/(end-start);
          if(center>=CFG.minFreq&&center<=CFG.maxFreq)candidates.push({freq:center,q,gain});
        }
        startIndex=-1;
      }else{
        startIndex=i;
      }
      state=next;
    }
    return candidates;
  }

  const OPT_DELTAS=[
    [10,10,10,5,0.1,0.5],
    [10,10,10,2,0.1,0.2],
    [10,10,10,1,0.1,0.1]
  ];

  function optimizePass(freqs,base,target,filters,iteration,dir){
    filters=strip(filters);
    const [maxDF,maxDQ,maxDG,stepDF,stepDQ,stepDG]=OPT_DELTAS[iteration];
    const [minFreq,maxFreq]=[CFG.minFreq,CFG.maxFreq];
    const [minQ,maxQ]=[CFG.minQ,CFG.maxQ];
    const [minGain,maxGain]=[CFG.minGain,CFG.maxGain];
    const begin=dir?filters.length-1:0;
    const end=dir?-1:filters.length;
    const step=dir?-1:1;

    for(let i=begin;i!==end;i+=step){
      const f=filters[i];
      const others=filters.filter((_,fi)=>fi!==i);
      const baseWithout=applyFilters(base,others,freqs);
      let bestFilter=f;
      let bestDistance=distance(freqs,applyFilters(baseWithout,[f],freqs),target);

      const test=(df,dq,dg)=>{
        const freq=f.freq+df*freqUnit(f.freq)*stepDF;
        const q=f.q+dq*stepDQ;
        const gain=f.gain+dg*stepDG;
        if(freq<minFreq||freq>maxFreq||q<minQ||q>maxQ||gain<minGain||gain>maxGain)return false;
        const nf={freq,q,gain};
        const score=distance(freqs,applyFilters(baseWithout,[nf],freqs),target);
        if(score<bestDistance){bestFilter=nf;bestDistance=score;return true;}
        return false;
      };

      for(let df=-maxDF;df<maxDF;df++){
        for(let dq=maxDQ-1;dq>=-maxDQ;dq--){
          for(let dg=1;dg<maxDG;dg++)if(!test(df,dq,dg))break;
          for(let dg=-1;dg>=-maxDG;dg--)if(!test(df,dq,dg))break;
        }
      }
      filters[i]=bestFilter;
    }

    if(!dir)return optimizePass(freqs,base,target,filters,iteration,true);

    filters.sort((a,b)=>a.freq-b.freq);
    for(let i=0;i<filters.length-1;){
      const f1=filters[i],f2=filters[i+1];
      if(Math.abs(f1.freq-f2.freq)<=freqUnit(f1.freq)&&Math.abs(f1.q-f2.q)<=0.1){
        f1.gain+=f2.gain;
        f1.gain=clamp(f1.gain,minGain,maxGain);
        filters.splice(i+1,1);
      }else i++;
    }

    let bestDistance=distance(freqs,applyFilters(base,filters,freqs),target);
    for(let i=0;i<filters.length;){
      if(Math.abs(filters[i].gain)<=0.1){filters.splice(i,1);continue;}
      const reduced=filters.filter((_,fi)=>fi!==i);
      const score=distance(freqs,applyFilters(base,reduced,freqs),target);
      if(score<bestDistance){filters.splice(i,1);bestDistance=score;}else i++;
    }
    return filters;
  }

  function optimize(rawCurve,targetCurve){
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]);
    const hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    if(hi<=lo||hi<2000)throw Error('Raw Pudding and target curves have insufficient frequency overlap.');

    const freqs=logspace(lo,hi,CFG.points);
    const raw=resample(rawCurve,freqs),target=resample(targetCurve,freqs);
    const aligned=alignLevel(freqs,raw,target),rawA=aligned.curve;

    const firstBatchSize=Math.max(Math.floor(CFG.bands/2)-1,1);
    const firstCandidates=searchCandidates(freqs,rawA,target,1)
      .filter(c=>c.freq<=CFG.trebleStart)
      .sort((a,b)=>a.q-b.q)
      .slice(0,firstBatchSize)
      .sort((a,b)=>a.freq-b.freq);

    let firstFilters=firstCandidates;
    for(let i=0;i<OPT_DELTAS.length;i++)firstFilters=optimizePass(freqs,rawA,target,firstFilters,i,false);

    const secondFR=applyFilters(rawA,firstFilters,freqs);
    const secondBatchSize=CFG.bands-firstFilters.length;
    let secondFilters=searchCandidates(freqs,secondFR,target,0.5)
      .sort((a,b)=>a.q-b.q)
      .slice(0,secondBatchSize)
      .sort((a,b)=>a.freq-b.freq);

    for(let i=0;i<OPT_DELTAS.length;i++)secondFilters=optimizePass(freqs,rawA,target,secondFilters,i,false);

    let allFilters=firstFilters.concat(secondFilters);
    for(let i=0;i<OPT_DELTAS.length;i++)allFilters=optimizePass(freqs,rawA,target,allFilters,i,false);
    allFilters=strip(allFilters);

    while(allFilters.length<CFG.bands)allFilters.push({freq:0,gain:0,q:1});

    const corrected=applyFilters(rawA,allFilters,freqs);
    const before=rawA.map((v,i)=>Math.abs(v-target[i]));
    const after=corrected.map((v,i)=>Math.abs(v-target[i]));
    const active=allFilters.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain);

    return {bands:allFilters,metrics:{
      rmseBefore:rmse(before),rmseAfter:rmse(after),
      p95Before:percentile(before,.95),p95After:percentile(after,.95),
      maxBefore:Math.max(...before),maxAfter:Math.max(...after),
      activeBands:active.length,
      maxBoost:Math.max(...allFilters.map(b=>b.gain)),
      maxCut:Math.min(...allFilters.map(b=>b.gain)),
      maxQ:Math.max(...allFilters.map(b=>b.q)),
      levelOffsetDb:aligned.offset,coverage:[lo,hi],
      objective:distance(freqs,corrected,target)
    }};
  }

  const fmt=v=>Math.abs(v-Math.round(v))<1e-9?String(Math.round(v)):v.toFixed(1);
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  function formatPEQ(result){
    return result.bands.map((b,i)=>
      `Filter ${i+1}: PK ${fmt(b.freq)} Hz, ${b.gain>=0?'+':''}${b.gain.toFixed(2)} dB, Q ${b.q.toFixed(2)}`
    ).join('\n')+'\n';
  }

  function jsonPEQ(result,meta){
    return JSON.stringify({
      device:'MOONDROP PUDDING',
      engine:'Squiglink-style AutoEQ PK-only optimizer',
      version:CFG.version,
      filter_type:'PK',
      implementation:'MOONDROP Link-compatible constraint set',
      dsp_model:{type:'RBJ peaking biquad',sample_rate_hz:CFG.sampleRate,status:'PROVISIONAL'},
      level_alignment:{method:'mean raw-target error over 100 Hz–10 kHz',removed_offset_db:result.metrics.levelOffsetDb},
      optimizer:{initialization:'Squiglink candidate segmentation + geometric-centre Fc + bandwidth-derived Q',
                 loss:'mean absolute error with errors below 0.1 dB ignored',
                 batches:'first batch <=7 kHz, second residual batch, then full two-direction optimization'},
      constraints:{bands:CFG.bands,frequency_hz:[CFG.minFreq,CFG.maxFreq],gain_db:[CFG.minGain,CFG.maxGain],q:[CFG.minQ,CFG.maxQ]},
      source:meta,metrics:result.metrics,peq:result.bands
    },null,2)+'\n';
  }

  function download(text,filename,mime='text/plain;charset=utf-8'){
    const u=URL.createObjectURL(new Blob([text],{type:mime}));
    const a=document.createElement('a');a.href=u;a.download=filename;document.body.appendChild(a);a.click();
    setTimeout(()=>{URL.revokeObjectURL(u);a.remove();},0);
  }

  function rootRepo(){
    const p=location.pathname.split('/').filter(Boolean);
    return location.hostname.endsWith('.github.io')&&p[0]
      ? location.hostname.split('.')[0]+'/'+p[0]
      : 'ziyadizuhir-png/iem-earprint-engine';
  }

  async function readRepo(path){
    if(typeof rawText==='function')return rawText(path);
    const r=await fetch('https://raw.githubusercontent.com/'+rootRepo()+'/main/'+path+'?t='+Date.now(),{cache:'no-store'});
    if(!r.ok)throw Error('Could not read '+path);
    return r.text();
  }

  async function ghList(path){
    if(typeof gh==='function')return gh(path);
    const r=await fetch('https://api.github.com/repos/'+rootRepo()+'/contents/'+path,{headers:{Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},cache:'no-store'});
    if(!r.ok)throw Error('GitHub API '+r.status);
    return r.json();
  }

  function status(msg,cls=''){
    const s=$('puddingStatus');if(s){s.className='status small '+cls;s.textContent=msg||'';}
  }

  function fill(sel,names,preferred){
    if(!sel)return;
    sel.innerHTML=names.map(n=>`<option value="${esc(n)}">${esc(n.replace(/\.txt$/i,''))}</option>`).join('');
    if(preferred&&names.includes(preferred))sel.value=preferred;
  }

  async function populate(){
    try{
      const d=await ghList('input/original_711');
      const all=Array.isArray(d)?d.filter(x=>x.name&&/\.txt$/i.test(x.name)).map(x=>x.name):[];
      const names=all.filter(x=>/pudding/i.test(x));
      const preferred=names.find(x=>/^moondrop pudding fr\.txt$/i.test(x))||names[0]||'';
      fill($('puddingRawSelect'),names,preferred);
      status(names.length?'Raw 711 Pudding source loaded from input/original_711/.':'No Pudding 711 source found in input/original_711.');
    }catch(e){status(e.message,'warn');}
    try{
      const d=await ghList('input/targets');
      const names=Array.isArray(d)?d.filter(x=>x.name&&/\.txt$/i.test(x.name)).map(x=>x.name):[];
      fill($('puddingTargetSelect'),names,names[0]||'');
    }catch(e){status(e.message,'warn');}
  }

  function stem(n){
    return String(n).replace(/\.txt$/i,'').replace(/[^\w.-]+/g,'_').replace(/_+/g,'_').replace(/^[_\.]+|[_\.]+$/g,'')||'target';
  }

  function targetPath(n,robust){
    return robust?'output/'+stem(n)+'__robust_target.txt':'input/targets/'+encodeURIComponent(n);
  }

  async function selected(fileId,selectId,pathFn){
    const f=$(fileId)?.files?.[0];
    if(f)return {curve:parseCurveText(await f.text()),source:f.name};
    const n=$(selectId)?.value;
    if(!n)throw Error('Select or upload a curve.');
    return {curve:parseCurveText(await readRepo(pathFn(n))),source:n};
  }

  let last=null,lastMeta=null;

  async function generate(){
    const button=$('generatePuddingPEQ');
    if(button){button.disabled=true;button.textContent='Optimizing…';}
    status('Preparing raw Pudding + target…');
    try{
      const raw=await selected('puddingRawFile','puddingRawSelect',n=>'input/original_711/'+encodeURIComponent(n));
      const robust=$('puddingRobustTarget')?.checked!==false;
      const target=await selected('puddingTargetFile','puddingTargetSelect',n=>targetPath(n,robust));
      status('Squiglink-style candidate search + two-batch optimization + filter cleanup…');
      const result=optimize(raw.curve,target.curve);
      last=result;lastMeta={raw:raw.source,target:target.source,target_mode:robust?'Robust Target':'Original Target'};
      render(result,lastMeta);
      status('Done · '+result.metrics.activeBands+' active bands · RMSE '+result.metrics.rmseBefore.toFixed(2)+' → '+result.metrics.rmseAfter.toFixed(2)+' dB','ok');
    }catch(e){last=null;lastMeta=null;render(null,null);status(e.message,'warn');}
    finally{if(button){button.disabled=false;button.textContent='Generate Pudding PEQ';}}
  }

  function render(result,meta){
    const box=$('puddingResult'),table=$('puddingTable'),stats=$('puddingStats');
    if(!box||!table||!stats)return;
    if(!result){box.hidden=true;table.innerHTML='';stats.innerHTML='';return;}
    box.hidden=false;
    stats.innerHTML=[
      ['RMSE',result.metrics.rmseBefore.toFixed(2)+' → '+result.metrics.rmseAfter.toFixed(2)+' dB'],
      ['P95 error',result.metrics.p95Before.toFixed(2)+' → '+result.metrics.p95After.toFixed(2)+' dB'],
      ['Max error',result.metrics.maxBefore.toFixed(2)+' → '+result.metrics.maxAfter.toFixed(2)+' dB'],
      ['Coverage',fmt(result.metrics.coverage[0])+'–'+fmt(result.metrics.coverage[1])+' Hz'],
      ['Active bands',result.metrics.activeBands+' / '+CFG.bands],
      ['Gain range',result.metrics.maxCut.toFixed(2)+' to '+(result.metrics.maxBoost>=0?'+':'')+result.metrics.maxBoost.toFixed(2)+' dB'],
      ['Max Q',result.metrics.maxQ.toFixed(2)],
      ['Level alignment',(result.metrics.levelOffsetDb>=0?'+':'')+result.metrics.levelOffsetDb.toFixed(2)+' dB removed']
    ].map(x=>`<div class="summary"><span class="k">${esc(x[0])}</span><span class="v">${esc(x[1])}</span></div>`).join('');

    table.innerHTML='<div class="pudding-peq-head"><span>Band</span><span>Type</span><span>Freq</span><span>Gain</span><span>Q</span></div>'+
      result.bands.map((b,i)=>`<div class="pudding-peq-row"><span>${i+1}</span><span>PK</span><span>${fmt(b.freq)} Hz</span><span>${b.gain>=0?'+':''}${b.gain.toFixed(2)} dB</span><span>${b.q.toFixed(2)}</span></div>`).join('')+
      '<div class="pudding-note">Squiglink-style PK only · 20 Hz–12 kHz · -12 to +3 dB · Q 0.50–2.00 · first batch ≤7 kHz · level-align 100 Hz–10 kHz · RBJ @ 48 kHz provisional.</div>';
  }

  function downloadTxt(){
    if(!last){status('Generate the PEQ first.','warn');return;}
    download(formatPEQ(last),'Pudding_'+stem(lastMeta?.target||'RobustTarget')+'_SquiglinkStyle_PEQ.txt');
  }

  function downloadJson(){
    if(!last){status('Generate the PEQ first.','warn');return;}
    download(jsonPEQ(last,lastMeta),'Pudding_'+stem(lastMeta?.target||'RobustTarget')+'_SquiglinkStyle_PEQ.json','application/json;charset=utf-8');
  }

  function ensureUi(){
    if($('puddingEngine'))return;
    const anchor=$('infoPanel')||$('modal');
    const html=`<div class="card section" id="puddingEngine">
      <div class="visualizer-head"><div><h2 style="margin:0">MOONDROP PUDDING PEQ Engine</h2>
      <div class="visualizer-subtitle">Squiglink-style AutoEQ optimizer → 10-band MOONDROP Link PEQ</div></div>
      <div class="viz-actions"><button type="button" id="puddingRefreshSources">Refresh sources</button></div></div>
      <div class="formrow">
        <div class="field"><label for="puddingRawSelect">Raw Pudding 711</label><select id="puddingRawSelect"></select><input id="puddingRawFile" type="file" accept=".txt,text/plain" style="margin-top:6px"></div>
        <div class="field"><label for="puddingTargetSelect">Target</label><select id="puddingTargetSelect"></select><input id="puddingTargetFile" type="file" accept=".txt,text/plain" style="margin-top:6px"></div>
      </div>
      <div class="inline" style="margin-top:8px"><label class="toggle"><input id="puddingRobustTarget" type="checkbox" checked> Use Robust Target</label>
      <button type="button" class="primary" id="generatePuddingPEQ">Generate Pudding PEQ</button>
      <button type="button" id="downloadPuddingPEQ">Download TXT</button><button type="button" id="downloadPuddingJSON">Download JSON</button></div>
      <div id="puddingStatus" class="status small" style="margin-top:8px"></div>
      <div id="puddingResult" hidden style="margin-top:10px"><div id="puddingStats" class="summary-grid"></div><div id="puddingTable" class="pudding-table" style="margin-top:10px"></div></div>
    </div>`;
    if(anchor)anchor.insertAdjacentHTML('beforebegin',html);else document.body.insertAdjacentHTML('beforeend',html);

    const style=document.createElement('style');
    style.textContent=`#puddingEngine{margin-top:12px}.pudding-table{border:1px solid var(--line);border-radius:6px;overflow:hidden}.pudding-peq-head,.pudding-peq-row{display:grid;grid-template-columns:.5fr .6fr 1.3fr 1.2fr 1fr;gap:8px;padding:7px 9px;align-items:center;font-size:12px;font-variant-numeric:tabular-nums}.pudding-peq-head{background:#0d141c;color:#7f8b99;border-bottom:1px solid var(--line);font-size:10px;text-transform:uppercase;letter-spacing:.04em}.pudding-peq-row{border-bottom:1px solid #1b2530}.pudding-note{padding:8px 9px;color:#7f8b99;font-size:10px;border-top:1px solid var(--line)}@media(max-width:600px){.pudding-peq-head,.pudding-peq-row{grid-template-columns:.45fr .55fr 1fr 1.1fr .8fr;font-size:11px;padding:7px 6px}}`;
    document.head.appendChild(style);
  }

  function init(){
    ensureUi();
    if(!$('puddingEngine'))return;
    status('Pudding PEQ Engine v'+CFG.version);
    $('generatePuddingPEQ')?.addEventListener('click',generate);
    $('downloadPuddingPEQ')?.addEventListener('click',downloadTxt);
    $('downloadPuddingJSON')?.addEventListener('click',downloadJson);
    $('puddingRefreshSources')?.addEventListener('click',populate);
    populate();
  }

  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();

  window.MoondropPuddingPEQ={CFG,optimize,formatPEQ};
})();
