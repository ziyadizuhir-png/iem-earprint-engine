/* ============================================================
   MOONDROP PUDDING PEQ ENGINE
   v2026-09-22.4

   AutoEq-inspired PK-only optimizer for MOONDROP Link.

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
    version: '2026-09-22.4',
    bands: 10,
    minFreq: 20,
    maxFreq: 12000,
    optHi: 10000,
    minGain: -12,
    maxGain: 3,
    minQ: 0.30,
    maxQ: 10,
    sampleRate: 48000,
    points: 360,
    alignLo: 100,
    alignHi: 10000,
    smoothSpan: 9,
    minSeedGain: 0.20,
    minSeedSeparationOct: 0.18,
    minActiveGain: 0.05,
    maxPasses: 5,
    sharpnessSlope: 18
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

  /* AutoEq-inspired sharpness penalty:
     AutoEq derives a gain limit for ~18 dB/oct maximum derivative,
     then applies a steep sigmoid coefficient to the filter response. */
  function sharpnessPenalty(freqs,b){
    if(Math.abs(b.gain)<1e-9)return 0;
    const gainLimit=-0.09503189270199464+20.575128011847003*(1/b.q);
    const x=b.gain/gainLimit-1;
    const k=1/(1+Math.exp(-x*100));
    const fr=rbj(freqs,b);
    let s=0;
    for(const v of fr)s+=v*v*k*k;
    return s/fr.length;
  }

  function objective(freqs,raw,target,bands){
    const eq=totalEq(freqs,bands);
    let s=0,n=0;
    for(let i=0;i<freqs.length;i++){
      if(freqs[i]<=CFG.optHi){
        const e=raw[i]+eq[i]-target[i];
        s+=e*e;n++;
      }
    }
    const ix=Math.max(0,freqs.findIndex(f=>f>CFG.optHi));
    if(ix<freqs.length){
      let a=0,b=0;
      for(let i=ix;i<freqs.length;i++){a+=target[i];b+=raw[i]+eq[i];}
      const m=freqs.length-ix;
      const e=(b/m)-(a/m);
      s+=0.25*e*e;
    }
    s/=Math.max(1,n);

    let p=0;
    for(const b of bands)p+=sharpnessPenalty(freqs,b);
    return Math.sqrt(s+p);
  }

  function findFeature(freqs,residual){
    const sm=smooth(residual,CFG.smoothSpan);
    const candidates=[];
    for(let i=2;i<sm.length-2;i++){
      const isMax=sm[i]>=sm[i-1]&&sm[i]>sm[i+1];
      const isMin=sm[i]<=sm[i-1]&&sm[i]<sm[i+1];
      if(isMax||isMin)candidates.push({i,amp:Math.abs(sm[i])});
    }
    candidates.sort((a,b)=>b.amp-a.amp);

    for(const c of candidates){
      if(c.amp<CFG.minSeedGain)break;
      const sign=sm[c.i]>=0?1:-1;
      const peak=Math.abs(sm[c.i]);
      const half=peak*0.5;
      let l=c.i,r=c.i;
      while(l>0&&Math.abs(sm[l])>half)l--;
      while(r<sm.length-1&&Math.abs(sm[r])>half)r++;
      const bw=Math.max(0.10,Math.log2(freqs[r]/freqs[l]));
      let q=Math.sqrt(Math.pow(2,bw))/(Math.pow(2,bw)-1);
      q=clamp(q,CFG.minQ,CFG.maxQ);
      return {freq:freqs[c.i],gain:clamp(sign*peak,CFG.minGain,CFG.maxGain),q};
    }
    return null;
  }

  function initializeAutoEq(freqs,raw,target){
    let residual=target.map((v,i)=>v-raw[i]);
    const bands=[];
    for(let k=0;k<CFG.bands;k++){
      const f=findFeature(freqs,residual);
      if(!f)break;
      if(bands.some(b=>Math.abs(Math.log2(f.freq/b.freq))<CFG.minSeedSeparationOct)){
        const idx=freqs.reduce((best,_,i)=>Math.abs(residual[i])>Math.abs(residual[best])?i:best,0);
        const alt={freq:freqs[idx],gain:clamp(residual[idx],CFG.minGain,CFG.maxGain),q:f.q};
        if(bands.some(b=>Math.abs(Math.log2(alt.freq/b.freq))<CFG.minSeedSeparationOct))break;
        f.freq=alt.freq;f.gain=alt.gain;
      }
      bands.push(f);
      const fr=rbj(freqs,f);
      residual=residual.map((v,i)=>v-fr[i]);
    }
    while(bands.length<CFG.bands){
      bands.push({freq:logspace(35,9000,CFG.bands)[bands.length],gain:0,q:1});
    }
    return bands;
  }

  function optimizeOne(freqs,raw,target,bands,idx){
    let best=cloneBands(bands), bestScore=objective(freqs,raw,target,best);
    const b=best[idx];

    function test(vals,setter){
      for(const v of vals){
        const t=cloneBands(best);
        setter(t[idx],v);
        const s=objective(freqs,raw,target,t);
        if(s+1e-9<bestScore){best=t;bestScore=s;}
      }
    }

    test([b.gain-1,b.gain-0.5,b.gain-0.25,b.gain,b.gain+0.25,b.gain+0.5,b.gain+1,0,CFG.minGain,CFG.maxGain],
      (x,v)=>x.gain=clamp(v,CFG.minGain,CFG.maxGain));

    test([0.30,0.40,0.50,0.60,0.75,1,1.3,1.7,2.2,3,4,5.5,7.5,10],
      (x,v)=>x.q=v);

    test([b.freq/1.35,b.freq/1.20,b.freq/1.10,b.freq,b.freq*1.10,b.freq*1.20,b.freq*1.35],
      (x,v)=>x.freq=clamp(v,CFG.minFreq,CFG.maxFreq));

    return {bands:best,score:bestScore};
  }

  function optimize(rawCurve,targetCurve){
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]);
    const hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    if(hi<=lo||hi<2000)throw Error('Raw Pudding and target curves have insufficient frequency overlap.');

    const freqs=logspace(lo,hi,CFG.points);
    const raw=resample(rawCurve,freqs), target=resample(targetCurve,freqs);
    const aligned=alignLevel(freqs,raw,target), rawA=aligned.curve;

    let bands=initializeAutoEq(freqs,rawA,target);
    let best=objective(freqs,rawA,target,bands);

    for(let pass=0;pass<CFG.maxPasses;pass++){
      let improved=false;
      for(let i=0;i<bands.length;i++){
        const r=optimizeOne(freqs,rawA,target,bands,i);
        if(r.score+1e-8<best){bands=r.bands;best=r.score;improved=true;}
      }
      if(!improved)break;
    }

    for(const b of bands){
      b.freq=Math.round(clamp(b.freq,CFG.minFreq,CFG.maxFreq)*10)/10;
      b.gain=Math.round(clamp(b.gain,CFG.minGain,CFG.maxGain)*100)/100;
      b.q=Math.round(clamp(b.q,CFG.minQ,CFG.maxQ)*100)/100;
    }

    const eq=totalEq(freqs,bands);
    const before=rawA.map((v,i)=>Math.abs(v-target[i]));
    const after=rawA.map((v,i)=>Math.abs(v+eq[i]-target[i]));
    const active=bands.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain);

    return {
      bands,
      metrics:{
        rmseBefore:rmse(before),rmseAfter:rmse(after),
        p95Before:percentile(before,.95),p95After:percentile(after,.95),
        maxBefore:Math.max(...before),maxAfter:Math.max(...after),
        activeBands:active.length,
        maxBoost:Math.max(...bands.map(b=>b.gain)),
        maxCut:Math.min(...bands.map(b=>b.gain)),
        maxQ:Math.max(...bands.map(b=>b.q)),
        levelOffsetDb:aligned.offset,
        coverage:[lo,hi],
        objective:best
      }
    };
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
      engine:'AutoEq-inspired PK-only optimizer',
      version:CFG.version,
      filter_type:'PK',
      implementation:'MOONDROP Link-compatible constraint set',
      dsp_model:{type:'RBJ peaking biquad',sample_rate_hz:CFG.sampleRate,status:'PROVISIONAL'},
      level_alignment:{method:'mean raw-target error over 100 Hz–10 kHz',removed_offset_db:result.metrics.levelOffsetDb},
      optimizer:{initialization:'largest residual feature + bandwidth-derived Q + residual peeling',
                 loss:'MSE 20 Hz–10 kHz + low-weight average-energy term above 10 kHz',
                 sharpness:'AutoEq-style 18 dB/oct sigmoid penalty'},
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
      const d=await ghList('input/preferred');
      const names=Array.isArray(d)?d.filter(x=>x.name&&/\.txt$/i.test(x.name)).map(x=>x.name):[];
      fill($('puddingRawSelect'),names,names.find(x=>/^Pudding\.txt$/i.test(x))||names[0]||'');
      status(names.length?'Raw IEM sources loaded.':'No raw IEM sources found.');
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
      const raw=await selected('puddingRawFile','puddingRawSelect',n=>'input/preferred/'+encodeURIComponent(n));
      const robust=$('puddingRobustTarget')?.checked!==false;
      const target=await selected('puddingTargetFile','puddingTargetSelect',n=>targetPath(n,robust));
      status('AutoEq-inspired initialization + residual peeling + constrained refinement…');
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
      '<div class="pudding-note">AutoEq-inspired PK only · 20 Hz–12 kHz · -12 to +3 dB · Q 0.30–10 · level-align 100 Hz–10 kHz · RBJ @ 48 kHz provisional.</div>';
  }

  function downloadTxt(){
    if(!last){status('Generate the PEQ first.','warn');return;}
    download(formatPEQ(last),'Pudding_'+stem(lastMeta?.target||'RobustTarget')+'_AutoEqInspired_PEQ.txt');
  }

  function downloadJson(){
    if(!last){status('Generate the PEQ first.','warn');return;}
    download(jsonPEQ(last,lastMeta),'Pudding_'+stem(lastMeta?.target||'RobustTarget')+'_AutoEqInspired_PEQ.json','application/json;charset=utf-8');
  }

  function ensureUi(){
    if($('puddingEngine'))return;
    const anchor=$('infoPanel')||$('modal');
    const html=`<div class="card section" id="puddingEngine">
      <div class="visualizer-head"><div><h2 style="margin:0">MOONDROP PUDDING PEQ Engine</h2>
      <div class="visualizer-subtitle">AutoEq-inspired residual-peeling optimizer → 10-band MOONDROP Link PEQ</div></div>
      <div class="viz-actions"><button type="button" id="puddingRefreshSources">Refresh sources</button></div></div>
      <div class="formrow">
        <div class="field"><label for="puddingRawSelect">Raw Pudding</label><select id="puddingRawSelect"></select><input id="puddingRawFile" type="file" accept=".txt,text/plain" style="margin-top:6px"></div>
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