/* ============================================================
   MOONDROP PUDDING PEQ ENGINE
   Browser-side, deterministic, constraint-aware optimizer.

   Pudding's exact internal DSP/biquad implementation is not publicly
   specified. This engine therefore uses a provisional 48 kHz RBJ
   peaking-biquad model for simulation, while enforcing the observed
   MOONDROP Link constraints as hard limits.
   ============================================================ */
(function(){
'use strict';

const CFG = Object.freeze({
  bands:10,minFreq:20,maxFreq:20000,minGain:-12,maxGain:3,minQ:.3,maxQ:10,
  filterType:'PK',sampleRate:48000,points:260,smoothSpan:5,huberDelta:1.25,
  gainPenalty:.010,boostPenalty:.030,qPenalty:.012,narrowQPenalty:.018,
  activePenalty:.045,smoothPenalty:.0025,maxPasses:5,
  gainSteps:[-3,-2,-1.5,-1,-.75,-.5,-.25,0,.25,.5,.75,1,1.5,2,2.5,3],
  qValues:[.3,.45,.6,.8,1,1.3,1.7,2.2,3,4,5.5,7.5,10],
  freqRatios:[1/1.45,1/1.25,1/1.12,1,1.12,1.25,1.45]
});

function clamp(x,a,b){return Math.max(a,Math.min(b,x));}
function logspace(a,b,n){const o=[],la=Math.log(a),lb=Math.log(b);for(let i=0;i<n;i++){const t=i/(n-1);o.push(Math.exp(la+(lb-la)*t));}return o;}
function parseCurveText(text){
  const rows=[],lines=String(text||'').replace(/^\uFEFF/,'').split(/\r?\n/);
  const re=/^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:Hz)?\s*(?:,|;|\t|\s+)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:dB)?\s*$/i;
  for(const raw of lines){const s=raw.trim();if(!s||s[0]==='#'||s.startsWith('//'))continue;const m=s.match(re);if(!m)continue;const f=Number(m[1]),v=Number(m[2]);if(Number.isFinite(f)&&Number.isFinite(v)&&f>0)rows.push([f,v]);}
  if(rows.length<10)throw Error('Curve needs at least 10 numeric frequency/response points.');
  for(let i=1;i<rows.length;i++)if(rows[i][0]<=rows[i-1][0])throw Error('Frequencies must be strictly ascending.');
  return rows;
}
function interp(c,f){
  if(f<c[0][0]||f>c[c.length-1][0])throw Error('Curve does not cover '+f+' Hz.');
  let lo=0,hi=c.length-1;while(lo<hi){const m=(lo+hi)>>1;if(c[m][0]<f)lo=m+1;else hi=m;}
  const i=Math.max(1,lo),a=c[i-1],b=c[i],la=Math.log(a[0]),lb=Math.log(b[0]),t=(Math.log(f)-la)/(lb-la||1);
  return a[1]+(b[1]-a[1])*t;
}
function resample(c,fs){return fs.map(f=>interp(c,f));}
function percentile(a,p){if(!a.length)return 0;const x=a.slice().sort((m,n)=>m-n),q=(x.length-1)*p,i=Math.floor(q),j=Math.ceil(q);return i===j?x[i]:x[i]+(x[j]-x[i])*(q-i);}
function smoothMoving(v,span){const o=new Array(v.length),r=Math.max(1,Math.floor(span/2));for(let i=0;i<v.length;i++){let s=0,n=0;for(let j=Math.max(0,i-r);j<=Math.min(v.length-1,i+r);j++){s+=v[j];n++;}o[i]=s/n;}return o;}
function rbjPeakResponse(freqs,b){
  const o=new Float64Array(freqs.length);if(Math.abs(b.gain)<1e-12)return o;
  const A=Math.pow(10,b.gain/40),w0=2*Math.PI*b.freq/CFG.sampleRate,alpha=Math.sin(w0)/(2*b.q),cw=Math.cos(w0);
  const b0=1+alpha*A,b1=-2*cw,b2=1-alpha*A,a0=1+alpha/A,a1=-2*cw,a2=1-alpha/A;
  for(let i=0;i<freqs.length;i++){const w=2*Math.PI*freqs[i]/CFG.sampleRate,c=Math.cos(w),s=Math.sin(w);
    const nr=b0+b1*c+b2*(2*c*c-1),ni=-b1*s-b2*(2*s*c),dr=a0+a1*c+a2*(2*c*c-1),di=-a1*s-a2*(2*s*c);
    const mag=Math.sqrt(nr*nr+ni*ni)/Math.max(Math.sqrt(dr*dr+di*di),1e-15);o[i]=20*Math.log10(Math.max(mag,1e-12));
  }return o;
}
function totalEq(freqs,bands){const eq=new Float64Array(freqs.length);for(const b of bands){if(Math.abs(b.gain)<1e-9)continue;const r=rbjPeakResponse(freqs,b);for(let i=0;i<eq.length;i++)eq[i]+=r[i];}return eq;}
function huberLoss(e,d){const a=Math.abs(e);return a<=d?.5*a*a:d*(a-.5*d);}
function makeWeights(freqs){return freqs.map(f=>f<80?.85:f<200?1.05:f<1000?1.2:f<4000?1.25:f<8000?1.1:f<12000?.9:.6);}
function interpGrid(freqs,vals,f){
  if(f<=freqs[0])return vals[0];if(f>=freqs[freqs.length-1])return vals[vals.length-1];
  let lo=0,hi=freqs.length-1;while(lo<hi){const m=(lo+hi)>>1;if(freqs[m]<f)lo=m+1;else hi=m;}
  const i=Math.max(1,lo),t=(Math.log(f)-Math.log(freqs[i-1]))/(Math.log(freqs[i])-Math.log(freqs[i-1]));
  return vals[i-1]+(vals[i]-vals[i-1])*t;
}
function localCandidates(freqs,residual){
  const sm=smoothMoving(residual,CFG.smoothSpan),cands=[45,90,180,350,700,1400,2800,5200,8500,13000];
  const extrema=[];for(let i=2;i<sm.length-2;i++){if((sm[i]>sm[i-1]&&sm[i]>=sm[i+1])||(sm[i]<sm[i-1]&&sm[i]<=sm[i+1]))extrema.push({f:freqs[i],a:Math.abs(sm[i])});}
  extrema.sort((a,b)=>b.a-a.a);
  for(const e of extrema){if(e.a<.7)break;if(cands.every(f=>Math.abs(Math.log2(e.f/f))>.22))cands.push(e.f);if(cands.length>=CFG.bands*3)break;}
  return cands.filter(f=>f>=CFG.minFreq&&f<=CFG.maxFreq).sort((a,b)=>a-b).slice(0,CFG.bands*2);
}
function initialBands(freqs,raw,target){
  const residual=target.map((v,i)=>v-raw[i]),cands=localCandidates(freqs,residual),fb=logspace(35,16000,CFG.bands),bands=[];
  for(let i=0;i<CFG.bands;i++){const f=cands[i]||fb[i],err=interpGrid(freqs,residual,f);bands.push({freq:clamp(f,CFG.minFreq,CFG.maxFreq),gain:clamp(err*.65,CFG.minGain,CFG.maxGain),q:Math.abs(err)>3?1.4:.9});}
  return bands;
}
function cloneBands(b){return b.map(x=>({freq:x.freq,gain:x.gain,q:x.q}));}
function score(freqs,raw,target,bands,weights){
  const eq=totalEq(freqs,bands);let loss=0;
  for(let i=0;i<freqs.length;i++)loss+=weights[i]*huberLoss(raw[i]+eq[i]-target[i],CFG.huberDelta);
  loss/=freqs.length;
  let reg=0;
  for(const b of bands){const g=Math.abs(b.gain);if(g<1e-7)continue;reg+=CFG.gainPenalty*g*g;if(b.gain>0)reg+=CFG.boostPenalty*b.gain*b.gain;reg+=CFG.qPenalty*Math.max(0,b.q-3);reg+=CFG.narrowQPenalty*Math.max(0,b.q-5);reg+=CFG.activePenalty;}
  let smooth=0;for(let i=1;i<eq.length-1;i++){const d2=eq[i+1]-2*eq[i]+eq[i-1];smooth+=d2*d2;}smooth/=Math.max(1,eq.length-2);
  return loss+reg+CFG.smoothPenalty*smooth;
}
function optimizeOne(freqs,raw,target,bands,weights,idx){
  let best=cloneBands(bands),bestScore=score(freqs,raw,target,best,weights);

  // Lightweight deterministic coordinate descent: frequency, gain and Q
  // are optimized separately to keep browser/mobile execution practical.
  const base=best[idx];

  const tryValues=(values,assign)=>{
    for(const value of values){
      const trial=cloneBands(best);
      assign(trial[idx],value);
      const s=score(freqs,raw,target,trial,weights);
      if(s+1e-9<bestScore){best=trial;bestScore=s;}
    }
  };

  tryValues([
    base.gain-2,base.gain-1,base.gain-.5,base.gain-.25,
    base.gain,base.gain+.25,base.gain+.5,base.gain+1,base.gain+2,
    0,CFG.minGain,CFG.maxGain
  ],(b,v)=>{b.gain=clamp(v,CFG.minGain,CFG.maxGain);});

  tryValues([
    .3,.45,.6,.8,1,1.3,1.7,2.2,3,4,5.5,7.5,10
  ],(b,v)=>{b.q=v;});

  tryValues([
    base.freq/1.35,base.freq/1.2,base.freq/1.1,
    base.freq,base.freq*1.1,base.freq*1.2,base.freq*1.35
  ],(b,v)=>{b.freq=clamp(v,CFG.minFreq,CFG.maxFreq);});

  return {bands:best,score:bestScore};
}

function optimize(rawCurve,targetCurve){
  const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]),hi=Math.min(CFG.maxFreq,rawCurve[rawCurve.length-1][0],targetCurve[targetCurve.length-1][0]);
  if(hi<=lo||hi<2000)throw Error('Raw Pudding and target curves have insufficient frequency overlap.');
  const freqs=logspace(lo,hi,CFG.points),raw=resample(rawCurve,freqs),target=resample(targetCurve,freqs),weights=makeWeights(freqs);
  let bands=initialBands(freqs,raw,target),bestScore=score(freqs,raw,target,bands,weights);
  for(let pass=0;pass<CFG.maxPasses;pass++){let improved=false;for(let i=0;i<bands.length;i++){const r=optimizeOne(freqs,raw,target,bands,weights,i);if(r.score+1e-8<bestScore){bands=r.bands;bestScore=r.score;improved=true;}}if(!improved)break;}
  for(let i=0;i<bands.length;i++)for(let j=i+1;j<bands.length;j++)if(Math.abs(Math.log2(bands[i].freq/bands[j].freq))<.12){const loser=Math.abs(bands[i].gain)<Math.abs(bands[j].gain)?i:j;bands[loser].gain=0;}
  bands.forEach(b=>{b.freq=Math.round(clamp(b.freq,CFG.minFreq,CFG.maxFreq)*10)/10;b.gain=Math.round(clamp(b.gain,CFG.minGain,CFG.maxGain)*100)/100;b.q=Math.round(clamp(b.q,CFG.minQ,CFG.maxQ)*100)/100;});
  const eq=totalEq(freqs,bands),before=raw.map((v,i)=>Math.abs(v-target[i])),after=raw.map((v,i)=>Math.abs(v+eq[i]-target[i])),active=bands.filter(b=>Math.abs(b.gain)>=.05);
  return {bands,metrics:{rmseBefore:Math.sqrt(before.reduce((s,v)=>s+v*v,0)/before.length),rmseAfter:Math.sqrt(after.reduce((s,v)=>s+v*v,0)/after.length),p95Before:percentile(before,.95),p95After:percentile(after,.95),maxBefore:Math.max(...before),maxAfter:Math.max(...after),activeBands:active.length,maxBoost:Math.max(...bands.map(b=>b.gain)),maxCut:Math.min(...bands.map(b=>b.gain)),maxQ:Math.max(...bands.map(b=>b.q)),coverage:[lo,hi]},score:bestScore};
}
function fmt(v){return Math.abs(v-Math.round(v))<1e-9?String(Math.round(v)):v.toFixed(1);}
function formatPEQ(r){return r.bands.map((b,i)=>'Filter '+(i+1)+': PK '+fmt(b.freq)+' Hz, '+(b.gain>=0?'+':'')+b.gain.toFixed(2)+' dB, Q '+b.q.toFixed(2)).join('\n')+'\n';}
function jsonPEQ(r,meta){return JSON.stringify({device:'MOONDROP PUDDING',filter_type:'PK',implementation:'MOONDROP Link-compatible constraint set',dsp_model:{type:'RBJ peaking biquad',sample_rate_hz:CFG.sampleRate,status:'PROVISIONAL — internal Pudding DSP implementation is not publicly specified'},constraints:{bands:CFG.bands,frequency_hz:[CFG.minFreq,CFG.maxFreq],gain_db:[CFG.minGain,CFG.maxGain],q:[CFG.minQ,CFG.maxQ]},source:meta,metrics:r.metrics,peq:r.bands},null,2)+'\n';}
function blobDownload(text,filename,mime){const blob=new Blob([text],{type:mime||'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(url);a.remove();},0);}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function rootRepo(){const p=location.pathname.split('/').filter(Boolean);if(location.hostname.endsWith('.github.io')&&p[0])return location.hostname.split('.')[0]+'/'+p[0];return 'ziyadizuhir-png/iem-earprint-engine';}
async function readRepo(path){if(typeof rawText==='function')return rawText(path);const r=await fetch('https://raw.githubusercontent.com/'+rootRepo()+'/main/'+path+'?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('Could not read '+path);return r.text();}
async function ghList(path){if(typeof gh==='function')return gh(path);const r=await fetch('https://api.github.com/repos/'+rootRepo()+'/contents/'+path,{headers:{Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},cache:'no-store'});if(!r.ok)throw Error('GitHub API '+r.status);return r.json();}
function E(id){return document.getElementById(id);}
function status(msg,cls){const s=E('puddingStatus');if(s){s.className='status small '+(cls||'');s.textContent=msg||'';}}
function fill(sel,names,preferred){if(!sel)return;sel.innerHTML=names.map(n=>'<option value="'+esc(n)+'">'+esc(n.replace(/\.txt$/i,''))+'</option>').join('');if(preferred&&names.includes(preferred))sel.value=preferred;}
async function populate(){
  try{const d=await ghList('input/preferred');const n=Array.isArray(d)?d.filter(x=>x.name&&/\.txt$/i.test(x.name)).map(x=>x.name):[];fill(E('puddingRawSelect'),n,n.find(x=>/^Pudding\.txt$/i.test(x))||n[0]||'');status(n.length?'Raw IEM sources loaded.':'No raw IEM sources found.');}catch(e){status(e.message,'warn');}
  try{const d=await ghList('input/targets');const n=Array.isArray(d)?d.filter(x=>x.name&&/\.txt$/i.test(x.name)).map(x=>x.name):[];fill(E('puddingTargetSelect'),n,n[0]||'');}catch(e){status(e.message,'warn');}
}
function stem(n){return String(n).replace(/\.txt$/i,'').replace(/[^\w.-]+/g,'_').replace(/_+/g,'_').replace(/^[_\.]+|[_\.]+$/g,'')||'target';}
function targetPath(n,robust){return robust?'output/'+stem(n)+'__robust_target.txt':'input/targets/'+encodeURIComponent(n);}
async function selected(fileId,selectId,pathFn){const f=E(fileId)?.files?.[0];if(f)return {curve:parseCurveText(await f.text()),source:f.name};const n=E(selectId)?.value;if(!n)throw Error('Select or upload a curve.');return {curve:parseCurveText(await readRepo(pathFn(n))),source:n};}
let last=null,lastMeta=null;
async function generate(){
  const b=E('generatePuddingPEQ');if(b){b.disabled=true;b.textContent='Optimizing…';}status('Preparing raw Pudding + target…');
  try{
    const raw=await selected('puddingRawFile','puddingRawSelect',n=>'input/preferred/'+encodeURIComponent(n));
    const robust=E('puddingRobustTarget')?.checked!==false;
    const target=await selected('puddingTargetFile','puddingTargetSelect',n=>targetPath(n,robust));
    status('Running 10-band constrained optimizer…');
    const r=optimize(raw.curve,target.curve);last=r;lastMeta={raw:raw.source,target:target.source,target_mode:robust?'Robust Target':'Original Target'};render(r,lastMeta);
    status('Done · '+r.metrics.activeBands+' active bands · RMSE '+r.metrics.rmseBefore.toFixed(2)+' → '+r.metrics.rmseAfter.toFixed(2)+' dB','ok');
  }catch(e){last=null;lastMeta=null;render(null,null);status(e.message,'warn');}
  finally{if(b){b.disabled=false;b.textContent='Generate Pudding PEQ';}}
}
function render(r,meta){
  const box=E('puddingResult'),table=E('puddingTable'),stats=E('puddingStats');if(!box||!table||!stats)return;
  if(!r){box.hidden=true;table.innerHTML='';stats.innerHTML='';return;}box.hidden=false;
  stats.innerHTML=[['RMSE',r.metrics.rmseBefore.toFixed(2)+' → '+r.metrics.rmseAfter.toFixed(2)+' dB'],['P95 error',r.metrics.p95Before.toFixed(2)+' → '+r.metrics.p95After.toFixed(2)+' dB'],['Max error',r.metrics.maxBefore.toFixed(2)+' → '+r.metrics.maxAfter.toFixed(2)+' dB'],['Coverage',fmt(r.metrics.coverage[0])+'–'+fmt(r.metrics.coverage[1])+' Hz'],['Active bands',r.metrics.activeBands+' / '+CFG.bands],['Gain range',r.metrics.maxCut.toFixed(2)+' to '+(r.metrics.maxBoost>=0?'+':'')+r.metrics.maxBoost.toFixed(2)+' dB'],['Max Q',r.metrics.maxQ.toFixed(2)]].map(x=>'<div class="summary"><span class="k">'+esc(x[0])+'</span><span class="v">'+esc(x[1])+'</span></div>').join('');
  table.innerHTML='<div class="pudding-peq-head"><span>Band</span><span>Type</span><span>Freq</span><span>Gain</span><span>Q</span></div>'+r.bands.map((b,i)=>'<div class="pudding-peq-row"><span>'+String(i+1)+'</span><span>PK</span><span>'+fmt(b.freq)+' Hz</span><span>'+(b.gain>=0?'+':'')+b.gain.toFixed(2)+' dB</span><span>'+b.q.toFixed(2)+'</span></div>').join('')+'<div class="pudding-note">Model: RBJ PK @ '+(CFG.sampleRate/1000)+' kHz (provisional). MOONDROP Link limits are enforced as hard constraints.</div>';
}
function downloadTxt(){if(!last)return status('Generate the PEQ first.','warn');const base='Pudding_'+stem(lastMeta?.target||'RobustTarget');blobDownload(formatPEQ(last),base+'_MOONDROP_Link_PEQ.txt');}
function downloadJson(){if(!last)return status('Generate the PEQ first.','warn');const base='Pudding_'+stem(lastMeta?.target||'RobustTarget');blobDownload(jsonPEQ(last,lastMeta),base+'_MOONDROP_Link_PEQ.json','application/json;charset=utf-8');}
function ensureUi(){
  if(E('puddingEngine')) return;
  const anchor=E('infoPanel') || E('modal');
  const html=`
  <div class="card section" id="puddingEngine">
    <div class="visualizer-head">
      <div>
        <h2 style="margin:0">MOONDROP PUDDING PEQ Engine</h2>
        <div class="visualizer-subtitle">Raw Pudding + Robust Target → 10-band MOONDROP Link PEQ</div>
      </div>
      <div class="viz-actions"><button type="button" id="puddingRefreshSources">Refresh sources</button></div>
    </div>

    <div class="formrow">
      <div class="field">
        <label for="puddingRawSelect">Raw Pudding</label>
        <select id="puddingRawSelect"></select>
        <input id="puddingRawFile" type="file" accept=".txt,text/plain" style="margin-top:6px">
      </div>
      <div class="field">
        <label for="puddingTargetSelect">Target</label>
        <select id="puddingTargetSelect"></select>
        <input id="puddingTargetFile" type="file" accept=".txt,text/plain" style="margin-top:6px">
      </div>
    </div>

    <div class="inline" style="margin-top:8px">
      <label class="toggle"><input id="puddingRobustTarget" type="checkbox" checked> Use Robust Target</label>
      <button type="button" class="primary" id="generatePuddingPEQ">Generate Pudding PEQ</button>
      <button type="button" id="downloadPuddingPEQ">Download TXT</button>
      <button type="button" id="downloadPuddingJSON">Download JSON</button>
    </div>

    <div id="puddingStatus" class="status small" style="margin-top:8px"></div>

    <div id="puddingResult" hidden style="margin-top:10px">
      <div id="puddingStats" class="summary-grid"></div>
      <div id="puddingTable" class="pudding-table" style="margin-top:10px"></div>
    </div>
  </div>`;

  if(anchor) anchor.insertAdjacentHTML('beforebegin',html);
  else document.body.insertAdjacentHTML('beforeend',html);

  const style=document.createElement('style');
  style.textContent=`
    #puddingEngine{margin-top:12px}
    #puddingEngine .viz-actions{display:flex}
    #puddingEngine input[type=file]{font-size:12px}
    .pudding-table{border:1px solid var(--line);border-radius:6px;overflow:hidden}
    .pudding-peq-head,.pudding-peq-row{display:grid;grid-template-columns:.5fr .6fr 1.3fr 1.2fr 1fr;gap:8px;padding:7px 9px;align-items:center;font-size:12px;font-variant-numeric:tabular-nums}
    .pudding-peq-head{background:#0d141c;color:#7f8b99;border-bottom:1px solid var(--line);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
    .pudding-peq-row{border-bottom:1px solid #1b2530}
    .pudding-peq-row:last-of-type{border-bottom:0}
    .pudding-note{padding:8px 9px;color:#7f8b99;font-size:10px;border-top:1px solid var(--line)}
    @media(max-width:600px){.pudding-peq-head,.pudding-peq-row{grid-template-columns:.45fr .55fr 1fr 1.1fr .8fr;font-size:11px;padding:7px 6px}.pudding-note{font-size:10px}}
  `;
  document.head.appendChild(style);
}

function init(){
  ensureUi();
  if(!E('puddingEngine'))return;
  E('generatePuddingPEQ')?.addEventListener('click',generate);
  E('downloadPuddingPEQ')?.addEventListener('click',downloadTxt);
  E('downloadPuddingJSON')?.addEventListener('click',downloadJson);
  E('puddingRefreshSources')?.addEventListener('click',populate);
  populate();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
window.MoondropPuddingPEQ={CFG,optimize,formatPEQ};
})();
