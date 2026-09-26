/* ============================================================
   IEM EARPRINT PEQ SOLVER — MOONDROP PUDDING
   Independent deterministic robust-target constrained PEQ solver.
   Exact response model: standard RBJ peaking biquad, 48 kHz provisional.
   Robust Target is the sole PEQ target authority.
   ============================================================ */
(function () {
  'use strict';

  const CFG = {
    version: '2026-09-26-v4.2.1-Moondrop-Loudness-UI',
    topologyToleranceDb: 0.15,
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
    balancedPoints: 180,
    balancedHuberPoints: 120,
    balancedPasses: 1,
    balancedFreqRadius: 3,
    balancedGainRadius: 2,
    alignLo: 100,
    alignHi: 10000,
    smoothSpan: 9,
    minSeedGain: 0.20,
    minSeedSeparationOct: 0.18,
    minActiveGain: 0.05,
    allowDuplicateFc: true,
    trebleStart: 7000,
    maxPasses: 3,
    huberEnabled: true,
    huberDeltaDb: 1.0,
    huberMinImprovementDb: 0.002,
    huberP95ToleranceDb: 0.15,
    huberMaxToleranceDb: 0.05,
    huberTriggerMaxDb: 3.0,
    huberTriggerP95Db: 0.80,
    // Balanced is the production path: screen both losses at Q<=2, then
    // run high-Q rescue only on the branch that has already passed guards.
    // Set to "exhaustive" for the slower legacy two-branch audit path.
    performanceMode: 'balanced'
    ,resolution: { r0Points: 720, r1Points: 240, r2Points: 96, tonalOctaves: 1/6 }
    ,huberEpsilonDb: 0.10
    ,sharpnessPenaltyWeight: 0.002
    ,complexityPenaltyWeight: 0.003
    ,boostRiskWeight: 0.025
    ,hfValidation: { startHz: 12000, endHz: 20000, maxDeviationDb: 6, maxSlopeDbPerOct: 18 }
    ,quantization: { freqHz: 1, gainDb: 0.01, q: 0.01 }
    ,feature: { prominenceDb: 0.15, minWidthOct: 0.05, supportOct: 0.12 }
    ,solver: {
      mode: 'lm-irls',
      initialBands: 3,
      growth: [3,5,7,10],
      candidateLimit: 24,
      minBandImprovementDb: 0.004,
      lmPoints: 240,
      maxIterations: 8,
      irlsIterations: 2,
      lambdaInitial: 0.03,
      lambdaUp: 6.0,
      lambdaDown: 0.35,
      maxRejectedSteps: 5,
      stepTolerance: 1e-4,
      lossTolerance: 1e-7,
      freqJacobianStep: 0.003,
      gainJacobianStepDb: 0.03,
      qJacobianStep: 0.01,
      gainRegularization: 0.0008,
      highQRegularization: 0.0015,
      boundaryHz: 1000,
      boundaryWidthOct: 0.10,
      boundaryHighQPenalty: 0.004,
      perturbationMaxRmseDeltaDb: 0.08,
      quantizationRescuePasses: 1
    }
  };

  const DOMAIN_BANDS = Object.freeze([
    [20,100],[100,300],[300,1000],[1000,3000],[3000,6000],
    [6000,8000],[8000,10000],[10000,12000],[12000,20000]
  ]);

  // The standard objective remains the reference branch. Huber is evaluated
  // as a robust candidate and can only be committed after the guards pass.
  let LOSS_MODE = 'standard';

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


  // Moondrop Link export validator (v4.1.1)
  // This validates device compatibility only. It does not change the
  // EarPrint optimization boundary (personal region remains 1kHz-12kHz).
  function validateMoondropLinkBands(bands){
    const errors=[];
    if(!Array.isArray(bands)) return ['Bands must be an array'];
    if(bands.length>10) errors.push('Band count exceeds Moondrop Link limit (10).');
    for(let i=0;i<bands.length;i++){
      const b=bands[i]||{};
      if(!(b.freq>=20&&b.freq<=20000)) errors.push(`Band ${i+1}: frequency out of 20Hz-20kHz range.`);
      if(!(b.gain>=-12&&b.gain<=3)) errors.push(`Band ${i+1}: gain out of -12dB/+3dB range.`);
      if(!(b.q>=0.3&&b.q<=10)) errors.push(`Band ${i+1}: Q out of 0.3-10 range.`);
    }
    return errors;
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

  // Apply the exact declared biquad transfer response to a curve.
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
    if(LOSS_MODE==='huber'){
      let s=0; const d=CFG.huberDeltaDb;
      for(let i=0;i<freqs.length;i++){
        const e=Math.abs(curve[i]-target[i]);
        s += e<=d ? 0.5*e*e : d*(e-0.5*d);
      }
      return Math.sqrt(2*s/freqs.length);
    }
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
    // Internal solver state stays continuous. Export quantization occurs only in quantizeBands().
    return filters.map(f=>({freq:clamp(Number(f.freq),CFG.minFreq,CFG.maxFreq),q:clamp(Number(f.q),CFG.minQ,CFG.maxQ),gain:clamp(Number(f.gain),CFG.minGain,CFG.maxGain)}));
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

  function augmentCandidates(freqs,base,target,seedCandidates){
    // Candidate pool is evidence-driven only: residual regions + feature analysis.
    const residual=base.map((v,i)=>v-target[i]);
    const features=featureAnalysis(freqs,residual);
    const out=seedCandidates.slice();
    const seen=new Set(out.map(c=>`${Math.round(c.freq)}:${Math.round(c.q*100)}`));
    for(const f of features){
      if(f.freq<CFG.minFreq||f.freq>CFG.maxFreq||Math.abs(f.gain)<CFG.minSeedGain) continue;
      const bw=Math.max((CFG.feature||{}).minWidthOct||0.05,f.widthOct);
      const x=Math.pow(2,bw);
      const q0=clamp(Math.sqrt(x)/Math.max(1e-9,x-1),CFG.minQ,2.0);
      const c={freq:f.freq,q:q0,gain:clamp(f.gain,CFG.minGain,CFG.maxGain),risk:boostRisk(f),morphologyConfidence:residualMorphologyConfidence(f)};
      const key=`${Math.round(c.freq)}:${Math.round(c.q*100)}`;
      if(!seen.has(key)){out.push(c);seen.add(key);}
    }
    return out;
  }

  function responseDelta(freqs,oldFilter,newFilter){
    const oldR=oldFilter?rbj(freqs,oldFilter):new Float64Array(freqs.length);
    const newR=newFilter?rbj(freqs,newFilter):new Float64Array(freqs.length);
    const d=new Float64Array(freqs.length);
    for(let i=0;i<d.length;i++)d[i]=newR[i]-oldR[i];
    return d;
  }

  // R0 is the numerical authority. R1/R2 are only search representations.
  function resolutionGrids(lo,hi){
    const r=CFG.resolution||{};
    return {R0:logspace(lo,hi,r.r0Points||720),R1:logspace(lo,hi,r.r1Points||240),R2:logspace(lo,hi,r.r2Points||96)};
  }

  function deadZoneHuber(e,epsilon=CFG.huberEpsilonDb,delta=CFG.huberDeltaDb){
    const a=Math.max(0,Math.abs(e)-epsilon);
    return a<=delta ? 0.5*a*a : delta*(a-0.5*delta);
  }

  function featureAnalysis(freqs,residual){
    const features=[];
    const minProm=(CFG.feature||{}).prominenceDb||0.15;
    for(let i=1;i<residual.length-1;i++){
      const v=residual[i], av=Math.abs(v);
      if(av<minProm || av<Math.abs(residual[i-1]) || av<Math.abs(residual[i+1]))continue;
      const sign=Math.sign(v); let l=i,r=i;
      while(l>0 && Math.sign(residual[l-1])===sign && Math.abs(residual[l-1])>=av*0.5)l--;
      while(r<residual.length-1 && Math.sign(residual[r+1])===sign && Math.abs(residual[r+1])>=av*0.5)r++;
      const width=Math.max(1e-9,Math.log2(freqs[r]/freqs[l]));
      const slope=(residual[r]-residual[l])/Math.max(1e-9,Math.log2(freqs[r]/freqs[l]));
      const curvature=(residual[i-1]-2*v+residual[i+1]);
      const support=Math.min(1,width/((CFG.feature||{}).supportOct||0.12));
      features.push({freq:freqs[i],gain:-v,widthOct:width,area:av*width,slope,curvature,support,isolated:support<0.5,continuation:1-support,density:1/Math.max(width,0.01)});
    }
    return features.sort((a,b)=>Math.abs(b.gain)-Math.abs(a.gain));
  }

  // v4.1.2 Residual Morphology Confidence: scores residual shape, not magnitude.
  // This intentionally complements Huber IRLS instead of duplicating outlier rejection.
  function residualMorphologyConfidence(feature){
    if(!feature) return 0.5;
    const broad = clamp(feature.widthOct/0.35,0,1);
    const smooth = clamp(feature.continuation,0,1);
    const isolatedPenalty = feature.isolated ? 0.25 : 0;
    return clamp(0.35*broad + 0.45*smooth + 0.20*(1-isolatedPenalty),0,1);
  }

  function boostRisk(feature){
    if(!feature || feature.gain<=0)return 0;
    const hf=clamp(Math.log2(Math.max(feature.freq,1000)/1000)/Math.log2(12),0,1);
    return clamp((feature.widthOct<0.15?0.35:0.05)+hf*0.25+Math.min(1,feature.gain/3)*0.25+(1-feature.support)*0.15+(feature.isolated?0.1:0),0,1);
  }

  function erbRate(f){ return 21.4*Math.log10(1+0.00437*Math.max(0,f)); }
  function erbBandError(freqs,curve,target){
    if(!freqs.length)return 0; const bins=new Map();
    for(let i=0;i<freqs.length;i++){const k=Math.round(erbRate(freqs[i])*2)/2; const a=bins.get(k)||[0,0]; a[0]+=curve[i]-target[i];a[1]++;bins.set(k,a);}
    const e=[...bins.values()].map(a=>a[0]/a[1]); return rmse(e);
  }
  function narrowFeatureError(freqs,curve,target){
    if(freqs.length<3)return 0; const e=curve.map((v,i)=>v-target[i]); let s=0,n=0;
    for(let i=1;i<e.length-1;i++){const c=e[i-1]-2*e[i]+e[i+1];s+=c*c;n++;} return Math.sqrt(s/Math.max(1,n));
  }
  function topologyError(freqs,curve,target){
    if(freqs.length<3)return 0; let s=0,n=0;
    for(let i=1;i<freqs.length-1;i++){const cc=curve[i-1]-2*curve[i]+curve[i+1], tc=target[i-1]-2*target[i]+target[i+1]; const d=cc-tc;s+=d*d;n++;} return Math.sqrt(s/Math.max(1,n));
  }
  function biquadSafety(b){
    if(!b||![b.freq,b.gain,b.q].every(Number.isFinite)||b.q<=0||b.freq<=0||b.freq>=CFG.sampleRate/2)return {stable:false,reason:'invalid parameters'};
    const A=Math.pow(10,b.gain/40),w0=2*Math.PI*b.freq/CFG.sampleRate,alpha=Math.sin(w0)/(2*b.q),cw=Math.cos(w0),a0=1+alpha/A,a1=-2*cw,a2=1-alpha/A;
    if(![a0,a1,a2].every(Number.isFinite)||Math.abs(a0)<1e-15)return {stable:false,reason:'invalid coefficients'};
    const A1=a1/a0,A2=a2/a0,disc=A1*A1-4*A2;
    let r1,r2;if(disc>=0){const q=Math.sqrt(disc);r1=(-A1+q)/2;r2=(-A1-q)/2;}else{r1=r2=Math.sqrt(Math.abs(A2));}
    return {stable:Math.abs(r1)<1-1e-10&&Math.abs(r2)<1-1e-10,maxPoleRadius:Math.max(Math.abs(r1),Math.abs(r2))};
  }
  function modeledHeadroom(bands){const f=logspace(20,20000,720),eq=totalEq(f,bands),m=Math.max(0,...eq);return {modeledMaxBoostDb:m,recommendedHeadroomDb:-m-0.5};}

  // v4.1: frequency confidence weighting. Keeps EarPrint/Robust Target unchanged.
  // It only changes optimizer loss weighting to reduce over-trusting uncertain regions.
  function frequencyConfidence(freq){
    if(freq<1000) return 0.9;
    if(freq<=5000) return 1.0;
    if(freq<=12000) return 0.75;
    return 0.45;
  }

  function weightedHuberLoss(freqs, errors){
    let total=0;
    for(let i=0;i<errors.length;i++) total += frequencyConfidence(freqs[i])*deadZoneHuber(errors[i]);
    return total/Math.max(1,errors.length);
  }

  function objectiveComponents(freqs,curve,target,bands=[],validation=null){
    const errors=curve.map((v,i)=>v-target[i]), abs=errors.map(Math.abs);
    const h=weightedHuberLoss(freqs,errors);
    const gainEnergy=bands.reduce((s,b)=>s+b.gain*b.gain,0);
    const qComplexity=bands.reduce((s,b)=>s+Math.max(0,b.q-2)*Math.max(0,Math.abs(b.gain)),0);
    const risk=bands.reduce((s,b)=>s+(b.gain>0?boostRisk({freq:b.freq,gain:b.gain,widthOct:1,support:1}):0),0);
    const hf=validation||{maxDeviation:0,p95Deviation:0,maxSlope:0};
    const sharpness=curve.length>2?curve.slice(1,-1).reduce((s,v,i)=>s+Math.abs(curve[i]-2*v+curve[i+2]),0)/curve.length:0;
    const hfPenalty=hf.maxDeviation+0.5*hf.p95Deviation;
    const erbError=erbBandError(freqs,curve,target), narrowError=narrowFeatureError(freqs,curve,target), topology=topologyError(freqs,curve,target);
    return {rmse:rmse(errors),mae:abs.reduce((s,v)=>s+v,0)/Math.max(1,abs.length),p95:percentile(abs,.95),max:Math.max(0,...abs),erbError,narrowError,topologyError:topology,huberLoss:h,gainEnergy,qComplexity,bandCost:bands.length,boostRisk:risk,sharpnessPenalty:sharpness,hfPenalty,score:rmse(errors)+0.08*erbError+0.05*narrowError+0.03*topology+CFG.complexityPenaltyWeight*(gainEnergy+qComplexity+bands.length)+0.002*qComplexity+CFG.sharpnessPenaltyWeight*sharpness+CFG.boostRiskWeight*risk+CFG.complexityPenaltyWeight*hfPenalty};
  }

  function highFrequencyValidation(rawCurve,bands){
    const lo=Math.max(12000,rawCurve[0][0]), hi=Math.min(20000,rawCurve.at(-1)[0]);
    if(hi<=lo)return {status:'SKIP',maxDeviation:0,p95Deviation:0,maxSlope:0,points:0};
    const f=logspace(lo,hi,96), base=resample(rawCurve,f), eq=totalEq(f,bands), d=eq.map(Math.abs);
    let maxSlope=0; for(let i=1;i<eq.length;i++)maxSlope=Math.max(maxSlope,Math.abs((eq[i]-eq[i-1])/Math.log2(f[i]/f[i-1])));
    const lim=CFG.hfValidation||{};
    return {status:Math.max(...d)<=lim.maxDeviationDb&&maxSlope<=lim.maxSlopeDbPerOct?'PASS':'FAIL',maxDeviation:Math.max(...d),p95Deviation:percentile(d,.95),maxSlope,points:f.length};
  }

  function quantizeBands(bands){
    const q=CFG.quantization||{}, active=bands.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain);
    if(active.length>CFG.bands) throw Error('Active solver state exceeds '+CFG.bands+' bands.');
    return active.map(b=>({freq:clamp(Math.round(b.freq/(q.freqHz||1))*(q.freqHz||1),CFG.minFreq,CFG.maxFreq),gain:clamp(Math.round(b.gain/(q.gainDb||0.01))*(q.gainDb||0.01),CFG.minGain,CFG.maxGain),q:clamp(Math.round(b.q/(q.q||0.01))*(q.q||0.01),CFG.minQ,CFG.maxQ)}));
  }

  function responseAwarePrune(freqs,base,target,bands){
    let work=bands.map(b=>({...b})), current=objectiveComponents(freqs,applyFilters(base,work,freqs),target,work).score;
    for(let i=0;i<work.length;){
      const trial=work.filter((_,j)=>j!==i), score=objectiveComponents(freqs,applyFilters(base,trial,freqs),target,trial).score;
      if(score<=current+0.002){work=trial;current=score;}else i++;
    }
    return work;
  }

  function finalizeQuantized(rawCurve,targetCurve,bands){
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]), hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    const freqs=logspace(lo,hi,(CFG.resolution||{}).r0Points||720), raw=resample(rawCurve,freqs), target=resample(targetCurve,freqs);
    const aligned=alignLevel(freqs,raw,target), qbands=responseAwarePrune(freqs,aligned.curve,target,quantizeBands(bands));
    const curve=applyFilters(aligned.curve,qbands,freqs), validation=highFrequencyValidation(rawCurve,qbands);
    const continuousCurve=applyFilters(aligned.curve,bands,freqs);
    const beforeTopology=topologyError(freqs,continuousCurve,target), afterTopology=topologyError(freqs,curve,target);
    const shapeGuard={status:afterTopology<=beforeTopology+CFG.topologyToleranceDb?'PASS':'FAIL',deltaDb:afterTopology-beforeTopology};
    const components=objectiveComponents(freqs,curve,target,qbands,validation);
    const finite=qbands.length<=CFG.bands&&qbands.every(b=>Number.isFinite(b.freq)&&Number.isFinite(b.gain)&&Number.isFinite(b.q)&&b.freq>=20&&b.freq<=12000&&b.gain>=-12&&b.gain<=3&&b.q>=.3&&b.q<=10);
    return {bands:qbands,components,validation,shapeGuard,finite,exactSimulation:{freqs,curve,target}};
  }

  function rankCandidates(freqs,base,target,candidates){
    return candidates.map((candidate,index)=>({candidate,index,score:distance(freqs,applyFilters(base,[candidate],freqs),target)+CFG.boostRiskWeight*(candidate.risk||0)+CFG.complexityPenaltyWeight*Math.max(0,candidate.q-2)}))
      .sort((a,b)=>a.score-b.score || a.index-b.index)
      .map(x=>x.candidate);
  }

  const OPT_DELTAS=[
    [10,10,10,5,0.1,0.5],
    [10,10,10,2,0.1,0.2],
    [10,10,10,1,0.1,0.1]
  ];

  function qSearchValues(current,iteration){
    const anchors=[0.30,0.40,0.50,0.60,0.75,1.00,1.25,1.50,1.75,2.00,2.50,3.00,3.50,4.00,5.00,6.00,7.50,10.00];
    const step=[0.5,0.2,0.1][iteration] || 0.1;
    const vals=new Set();
    for(const a of anchors) vals.add(Math.round(clamp(a,CFG.minQ,CFG.maxQ)*10)/10);
    for(let k=-5;k<=5;k++) vals.add(Math.round(clamp(current+k*step,CFG.minQ,CFG.maxQ)*10)/10);
    return Array.from(vals).sort((a,b)=>Math.abs(a-current)-Math.abs(b-current));
  }

  function optimizePass(freqs,base,target,filters,iteration,dir){
    filters=strip(filters);
    const passDeltas=CFG.performanceMode==='balanced'?OPT_DELTAS.slice(0,CFG.balancedPasses):OPT_DELTAS;
    const [maxDF,maxDQ,maxDG,stepDF,stepDQ,stepDG]=passDeltas[iteration];
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
      let bestCurve=applyFilters(baseWithout,[f],freqs);
      let bestDistance=distance(freqs,bestCurve,target);

      const test=(nf)=>{
        if(nf.freq<minFreq||nf.freq>maxFreq||nf.q<minQ||nf.q>maxQ||nf.gain<minGain||nf.gain>maxGain)return;
        const delta=responseDelta(freqs,bestFilter,nf), trial=new Float64Array(bestCurve);
        for(let k=0;k<trial.length;k++)trial[k]+=delta[k];
        const score=distance(freqs,trial,target);
        if(score<bestDistance-1e-12){bestFilter=nf;bestDistance=score;bestCurve=trial;}
      };

      const qvals=qSearchValues(f.q,iteration);
      const gvals=[];
      for(let dg=-maxDG;dg<=maxDG;dg++){
        gvals.push(Math.round((f.gain+dg*stepDG)*10)/10);
      }
      for(let df=-maxDF;df<maxDF;df++){
        const freq=f.freq+df*freqUnit(f.freq)*stepDF;
        if(freq<minFreq||freq>maxFreq)continue;
        for(const q of qvals){
          for(const gain of gvals)test({freq,q,gain});
        }
      }
      // A final local refinement around the best point at 0.1 Q / small gain/frequency steps.
      const bf=bestFilter;
      for(const df of [-2,-1,0,1,2]){
        for(const q of [bf.q-0.2,bf.q-0.1,bf.q,bf.q+0.1,bf.q+0.2]){
          for(const dg of [-2,-1,0,1,2]){
            test({freq:bf.freq+df*freqUnit(bf.freq),q,gain:bf.gain+dg*0.1});
          }
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

  function responseRmse(freqs,curve,target){
    if(!curve.length)return 0;
    if(LOSS_MODE==='huber'){
      let s=0; const d=CFG.huberDeltaDb;
      for(let i=0;i<curve.length;i++){
        const e=Math.abs(curve[i]-target[i]);
        s += e<=d ? 0.5*e*e : d*(e-0.5*d);
      }
      return Math.sqrt(2*s/curve.length);
    }
    let s=0;
    for(let i=0;i<curve.length;i++){const e=curve[i]-target[i];s+=e*e;}
    return Math.sqrt(s/curve.length);
  }
  /* Response-aware cleanup:
     The merge condition also requires nearly identical Q.
     For Pudding, two nearby same-sign PK filters can still be functionally
     redundant when their combined transfer function is well represented by
     one bounded PK filter. Test the actual response instead of comparing Q only. */
  function fitMergedFilter(freqs,pair){
    const [a,b]=pair;
    const lo=Math.max(CFG.minFreq,Math.min(a.freq,b.freq)/1.7);
    const hi=Math.min(CFG.maxFreq,Math.max(a.freq,b.freq)*1.7);
    const idx=[];
    for(let i=0;i<freqs.length;i++)if(freqs[i]>=lo&&freqs[i]<=hi)idx.push(i);
    if(idx.length<12)return null;

    const ra=rbj(freqs,a), rb=rbj(freqs,b);
    const combined=idx.map(i=>ra[i]+rb[i]);
    const fcLo=Math.max(CFG.minFreq,Math.min(a.freq,b.freq)*0.9);
    const fcHi=Math.min(CFG.maxFreq,Math.max(a.freq,b.freq)*1.1);
    const fGrid=logspace(fcLo,fcHi,18);
    let best=null;
    for(const fc of fGrid){
      for(const q of qSearchValues((a.q+b.q)/2,2)){
        for(let gain=CFG.minGain;gain<=CFG.maxGain+1e-9;gain+=0.2){
          const rr=rbj(freqs,{freq:fc,q,gain});
          let se=0,mx=0;
          for(let j=0;j<idx.length;j++){
            const d=Math.abs(rr[idx[j]]-combined[j]);
            se+=d*d; if(d>mx)mx=d;
          }
          const rms=Math.sqrt(se/idx.length);
          if(!best||rms<best.rms)best={freq:fc,q,gain,rms,max:mx};
        }
      }
    }
    if(!best||best.rms>0.12||best.max>0.25)return null;
    return best;
  }

  function refineMergedGlobal(freqs,base,target,others,seed){
    let best=seed;
    let bestScore=responseRmse(freqs,applyFilters(base,others.concat([seed]),freqs),target);
    const fGrid=logspace(Math.max(CFG.minFreq,seed.freq*0.95),Math.min(CFG.maxFreq,seed.freq*1.05),15);
    const qLo=Math.max(CFG.minQ,seed.q-0.3), qHi=Math.min(CFG.maxQ,seed.q+0.3);
    const gLo=Math.max(CFG.minGain,seed.gain-0.8), gHi=Math.min(CFG.maxGain,seed.gain+0.8);
    for(const freq of fGrid){
      for(let q=qLo;q<=qHi+1e-9;q+=0.1){
        for(let gain=gLo;gain<=gHi+1e-9;gain+=0.1){
          const cand={freq,q,gain};
          const score=responseRmse(freqs,applyFilters(base,others.concat([cand]),freqs),target);
          if(score<bestScore){best=cand;bestScore=score;}
        }
      }
    }
    return {filter:best,score:bestScore};
  }

  function mergeOverlappingFilters(freqs,base,target,filters){
    let work=filters.slice().sort((a,b)=>a.freq-b.freq);
    let changed=true;
    while(changed){
      changed=false;
      for(let i=0;i<work.length-1;i++){
        const a=work[i],b=work[i+1];
        if(Math.sign(a.gain)!==Math.sign(b.gain))continue;
        if(Math.max(a.freq,b.freq)/Math.min(a.freq,b.freq)>1.35)continue;
        // Keep duplicate/near-duplicate Fc when the two filters have
        // materially different Q. This permits paired-filter solutions while
        // still allowing genuinely redundant filters to be merged.
        if(CFG.allowDuplicateFc &&
           Math.abs(a.freq-b.freq)<=freqUnit(Math.min(a.freq,b.freq)) &&
           Math.abs(a.q-b.q)>0.25)continue;

        const seed=fitMergedFilter(freqs,[a,b]);
        if(!seed)continue;

        const others=work.filter((_,j)=>j!==i&&j!==i+1);
        const oldScore=responseRmse(freqs,applyFilters(base,work,freqs),target);
        const refined=refineMergedGlobal(freqs,base,target,others,seed);
        const candidate=others.concat([refined.filter]).sort((x,y)=>x.freq-y.freq);

        // Only accept a merge when the actual global response is no worse
        // than the pre-merge response within a very small tolerance.
        if(refined.score<=oldScore+0.003){
          work=candidate;
          changed=true;
          break;
        }
      }
    }
    return work;
  }

  // bound-aware reoptimization: transactional re-optimization for filters pinned at
  // the MOONDROP Link gain ceiling. Changes are accepted only if the FINAL
  // solution improves globally and does not materially regress P95 or max error.
  function ceilingAwareReoptimize(freqs,base,target,filters){
    let work=filters.map(b=>({...b}));
    const metric=(bands)=>{
      const curve=applyFilters(base,bands,freqs);
      const e=curve.map((v,i)=>Math.abs(v-target[i]));
      return {rmse:rmse(e),p95:percentile(e,.95),max:Math.max(...e)};
    };
    const guard=(oldM,newM)=>
      newM.rmse<=oldM.rmse-0.002 &&
      newM.p95<=oldM.p95+0.05 &&
      newM.max<=oldM.max+0.05;

    const baseline=metric(work);
    let bestWork=work.map(b=>({...b}));
    let bestM=baseline;

    // Search all ceiling-pinned filters from the same baseline, then commit
    // the whole candidate only if the final global guard passes.
    const ceilingPasses=CFG.performanceMode==='balanced'?1:2;
    const ceilingFreqPoints=CFG.performanceMode==='balanced'?15:25;
    const ceilingQStep=CFG.performanceMode==='balanced'?0.2:0.1;
    for(let pass=0;pass<ceilingPasses;pass++){
      for(let i=0;i<work.length;i++){
        if(Math.abs(work[i].gain-CFG.maxGain)>1e-9)continue;
        let local=work[i], localM=metric(work);
        const f0=work[i].freq;
        const fGrid=logspace(
          Math.max(CFG.minFreq,Math.max(7000,f0*0.82)),
          Math.min(CFG.maxFreq,f0*1.18),ceilingFreqPoints
        );
        for(const f of fGrid){
          for(let q=CFG.minQ;q<=CFG.maxQ+1e-9;q+=ceilingQStep){
            const trial=work.map(b=>({...b}));
            trial[i]={freq:f,q,gain:CFG.maxGain};
            const m=metric(trial);
            if(m.rmse<localM.rmse-1e-10){
              local=trial[i];localM=m;
            }
          }
        }
        work[i]=local;
      }
    }

    const finalM=metric(work);
    if(guard(baseline,finalM)){
      return work;
    }
    return bestWork;
  }

  function targetShapeGuard(freqs,base,target,preFilters,finalFilters){
    const pre=applyFilters(base,preFilters,freqs), fin=applyFilters(base,finalFilters,freqs);
    const a=topologyError(freqs,pre,target), b=topologyError(freqs,fin,target), delta=b-a;
    return {filters:delta<=CFG.topologyToleranceDb?finalFilters:preFilters,accepted:delta<=CFG.topologyToleranceDb,reason:delta<=CFG.topologyToleranceDb?'target_relative_pass':'target_relative_rollback',delta,pre:{rms:a},fin:{rms:b}};
  }
  function optimizeSingle(rawCurve,targetCurve){
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]);
    const hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    if(hi<=lo||hi<2000)throw Error('Raw Pudding and target curves have insufficient frequency overlap.');

    const grids=resolutionGrids(lo,hi);
    const freqs=grids.R0;
    const raw=resample(rawCurve,freqs),target=resample(targetCurve,freqs);
    const aligned=alignLevel(freqs,raw,target),rawA=aligned.curve;

    // Discover features on the coarse and tonal views, then hand only their
    // deterministic seeds to the existing candidate/ranking machinery.
    const featureSeeds=[];
    for(const view of [grids.R2,grids.R1]){
      const vr=resample(rawCurve,view), vt=resample(targetCurve,view), va=alignLevel(view,vr,vt).curve;
      for(const f of featureAnalysis(view,va.map((v,i)=>v-vt[i])).slice(0,24)){
        featureSeeds.push({freq:f.freq,q:clamp(1/Math.max(f.widthOct*2,0.3),0.3,2),gain:clamp(f.gain,-12,3),risk:boostRisk(f)});
      }
    }

    const firstBatchSize=Math.max(Math.floor(CFG.bands/2)-1,1);
    const firstCandidates=rankCandidates(freqs,rawA,target,augmentCandidates(freqs,rawA,target,searchCandidates(freqs,rawA,target,1).concat(featureSeeds)))
      .filter(c=>c.freq<=CFG.trebleStart)
      .slice(0,firstBatchSize)
      .sort((a,b)=>a.freq-b.freq);

    let firstFilters=firstCandidates;
    const passCount=CFG.performanceMode==='balanced'?CFG.balancedPasses:OPT_DELTAS.length;
    for(let i=0;i<passCount;i++)firstFilters=optimizePass(freqs,rawA,target,firstFilters,i,false);

    const secondFR=applyFilters(rawA,firstFilters,freqs);
    const secondBatchSize=CFG.bands-firstFilters.length;
    let secondFilters=rankCandidates(freqs,secondFR,target,augmentCandidates(freqs,secondFR,target,searchCandidates(freqs,secondFR,target,0.5).concat(featureSeeds)))
      .slice(0,secondBatchSize)
      .sort((a,b)=>a.freq-b.freq);

    for(let i=0;i<passCount;i++)secondFilters=optimizePass(freqs,rawA,target,secondFilters,i,false);

    let allFilters=firstFilters.concat(secondFilters);
    for(let i=0;i<passCount;i++)allFilters=optimizePass(freqs,rawA,target,allFilters,i,false);
    allFilters=strip(allFilters);
    allFilters=mergeOverlappingFilters(freqs,rawA,target,allFilters);
    allFilters=strip(allFilters);
    const preCeilingFilters=allFilters.map(b=>({...b}));
    allFilters=ceilingAwareReoptimize(freqs,rawA,target,allFilters);
    allFilters=strip(allFilters);

    // Target Topology Guard: the selected Robust Target is the sole reference.
    // Pure EarPrint is upstream-only and is not read by the PEQ solver.
    const sg=targetShapeGuard(freqs,rawA,target,preCeilingFilters,allFilters);
    allFilters=sg.filters;

    const corrected=applyFilters(rawA,allFilters,freqs);
    const before=rawA.map((v,i)=>Math.abs(v-target[i]));
    const after=corrected.map((v,i)=>Math.abs(v-target[i]));
    const active=allFilters.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain);

    return {bands:allFilters,metrics:{
      rmseBefore:rmse(before),rmseAfter:rmse(after),
      p95Before:percentile(before,.95),p95After:percentile(after,.95),
      maxBefore:Math.max(...before),maxAfter:Math.max(...after),
      activeBands:active.length,
      maxBoost:Math.max(0,...allFilters.map(b=>b.gain)),
      maxCut:Math.min(0,...allFilters.map(b=>b.gain)),
      maxQ:Math.max(0,...allFilters.map(b=>b.q)),
      levelOffsetDb:aligned.offset,coverage:[lo,hi],
      objective:distance(freqs,corrected,target),
      shapeGuardEnabled:true,
      shapeGuardToleranceDb:CFG.topologyToleranceDb,
      shapeGuardAccepted:sg.accepted,
      shapeGuardReason:sg.reason,
      shapeGuardDeltaDb:sg.delta===undefined?null:sg.delta,
      shapeGuardPreRmsDb:sg.pre?sg.pre.rms:null,
      shapeGuardFinalRmsDb:sg.fin?sg.fin.rms:null,
      objectiveComponents:objectiveComponents(freqs,corrected,target,allFilters),
      resolution:{R0:grids.R0.length,R1:grids.R1.length,R2:grids.R2.length},
      featureCount:featureSeeds.length
    }};
  }


  // Dual-Q v2 selector: full Q<=2 baseline + fast high-Q rescue.
  // The rescue explores only Q>2 around the conservative solution plus a
  // small residual-seed pass, then applies the same transactional guards.
  function extendHighQFromQ2(rawCurve,targetCurve,q2){
    const oldMinQ=CFG.minQ, oldMaxQ=CFG.maxQ;
    CFG.minQ=0.30; CFG.maxQ=10.00;
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]);
    const hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    const freqs=resolutionGrids(lo,hi).R0;
    const raw=resample(rawCurve,freqs),target=resample(targetCurve,freqs);
    const aligned=alignLevel(freqs,raw,target),rawA=aligned.curve;
    const active=q2.bands.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain).map(b=>({...b}));
    let work=active.slice();
    const highQs=CFG.performanceMode==='balanced'
      ? [2.0,3.0,4.0,6.0,10.0]
      : [2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.5,10.0];
    // Local high-Q rescue around each existing band.
    // Evaluate candidates against a precomputed response with the current
    // band removed; this preserves the objective exactly while avoiding a
    // full multi-filter recomputation for every candidate.
    for(let i=0;i<work.length;i++){
      const f0=work[i].freq, fu=freqUnit(f0), g0=work[i].gain;
      const others=work.filter((_,j)=>j!==i);
      const baseWithout=applyFilters(rawA,others,freqs);
      let best={...work[i]};
      let bestM=responseRmse(freqs,applyFilters(baseWithout,[best],freqs),target);
      const fvals=[];
      const fRadius=CFG.performanceMode==='balanced'?CFG.balancedFreqRadius:4;
      for(let k=-fRadius;k<=fRadius;k++)fvals.push(clamp(f0+k*fu,CFG.minFreq,CFG.maxFreq));
      const gvals=[];
      const gRadius=CFG.performanceMode==='balanced'?CFG.balancedGainRadius:5;
      for(let k=-gRadius;k<=gRadius;k++)gvals.push(clamp(Math.round((g0+k*0.2)*10)/10,CFG.minGain,CFG.maxGain));
      for(const f of fvals){
        for(const q of highQs){
          for(const gain of gvals){
            const cand={freq:f,q,gain};
            const m=responseRmse(freqs,applyFilters(baseWithout,[cand],freqs),target);
            if(m<bestM-1e-12){bestM=m;best=cand;}
          }
        }
      }
      work[i]=best;
    }

    // Residual-seed pass: try up to two new high-Q peaks where the current
    // response has the largest local absolute error.
    const rescuePasses=CFG.performanceMode==='balanced'?1:2;
    for(let pass=0;pass<rescuePasses && work.length<CFG.bands;pass++){
      const cur=applyFilters(rawA,work,freqs);
      const cand=[];
      for(let i=2;i<freqs.length-2;i++){
        const e=Math.abs(cur[i]-target[i]);
        if(e<0.35)continue;
        if(e>=Math.abs(cur[i-1]-target[i-1]) && e>=Math.abs(cur[i+1]-target[i+1])) cand.push({i,e});
      }
      cand.sort((a,b)=>b.e-a.e);
      let added=false;
      for(const c of cand){
        const f=freqs[c.i];
        if(f<CFG.minFreq||f>CFG.maxFreq)continue;
        if(work.some(b=>Math.abs(Math.log2(b.freq/f))<0.15))continue;
        const localGain=clamp(Math.round((target[c.i]-cur[c.i])*10)/10,CFG.minGain,CFG.maxGain);
        const baseExisting=applyFilters(rawA,work,freqs);
        let best=null,bestM=responseRmse(freqs,baseExisting,target);
        for(const q of highQs.slice(1)){
          for(const dg of [-1,-0.5,0,0.5,1]){
            const gain=clamp(localGain+dg,CFG.minGain,CFG.maxGain);
            const cand={freq:f,q,gain};
            const rr=rbj(freqs,cand);
            const candidateCurve=new Float64Array(baseExisting);
            for(let i=0;i<candidateCurve.length;i++)candidateCurve[i]+=rr[i];
            const m=responseRmse(freqs,candidateCurve,target);
            if(m<bestM-1e-12){bestM=m;best=cand;}
          }
        }
        if(best){work.push(best);work.sort((a,b)=>a.freq-b.freq);added=true;break;}
      }
      if(!added)break;
    }

    work=mergeOverlappingFilters(freqs,rawA,target,work);
    work=strip(work);
    const pre=work.map(b=>({...b}));
    work=ceilingAwareReoptimize(freqs,rawA,target,work);
    work=strip(work);
    const sg=targetShapeGuard(freqs,rawA,target,pre,work);
    work=sg.filters;
    CFG.minQ=oldMinQ; CFG.maxQ=oldMaxQ;
    const corrected=applyFilters(rawA,work,freqs), before=rawA.map((v,i)=>Math.abs(v-target[i])), after=corrected.map((v,i)=>Math.abs(v-target[i]));
    return {bands:work,metrics:{
      rmseBefore:rmse(before),rmseAfter:rmse(after),p95Before:percentile(before,.95),p95After:percentile(after,.95),
      maxBefore:Math.max(...before),maxAfter:Math.max(...after),activeBands:work.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain).length,
      maxBoost:Math.max(0,...work.map(b=>b.gain)),maxCut:Math.min(0,...work.map(b=>b.gain)),maxQ:Math.max(0,...work.map(b=>b.q)),
      levelOffsetDb:aligned.offset,coverage:[lo,hi],objective:distance(freqs,corrected,target),shapeGuardEnabled:true,
      shapeGuardToleranceDb:CFG.topologyToleranceDb,shapeGuardAccepted:sg.accepted,shapeGuardReason:sg.reason,shapeGuardDeltaDb:sg.delta===undefined?null:sg.delta,
      shapeGuardPreRmsDb:sg.pre?sg.pre.rms:null,shapeGuardFinalRmsDb:sg.fin?sg.fin.rms:null,
      objectiveComponents:objectiveComponents(freqs,corrected,target,work),
      resolution:{R0:freqs.length,R1:(CFG.resolution||{}).r1Points||240,R2:(CFG.resolution||{}).r2Points||96}
    }};
  }

  function optimizeBranch(rawCurve,targetCurve,lossMode){
    const previousLoss=LOSS_MODE;
    LOSS_MODE=lossMode;
    const savedMinQ=CFG.minQ, savedMaxQ=CFG.maxQ;
    try{
      CFG.minQ=0.30; CFG.maxQ=2.00;
      const q2=optimizeSingle(rawCurve,targetCurve);
      const q10=extendHighQFromQ2(rawCurve,targetCurve,q2);
      const q10PassGlobal =
        q10.metrics.rmseAfter <= q2.metrics.rmseAfter - 0.002 &&
        q10.metrics.p95After <= q2.metrics.p95After + 0.05 &&
        q10.metrics.maxAfter <= q2.metrics.maxAfter + 0.05;
      const branchShapeDelta = (q10.metrics.shapeGuardFinalRmsDb!=null && q2.metrics.shapeGuardFinalRmsDb!=null)
        ? q10.metrics.shapeGuardFinalRmsDb-q2.metrics.shapeGuardFinalRmsDb : null;
      const q10PassShape = q10.metrics.shapeGuardAccepted !== false &&
        (branchShapeDelta==null || branchShapeDelta<=CFG.topologyToleranceDb+1e-12);
      const useExtended=q10PassGlobal&&q10PassShape;
      const chosen=useExtended?q10:q2;
      const finalized=finalizeQuantized(rawCurve,targetCurve,chosen.bands);
      if(!finalized.finite){
        chosen.metrics.quantizationRescue='rejected';
      }else{
        chosen.bands=finalized.bands;
        chosen.metrics.quantizationRescue=finalized.validation.status==='FAIL'||finalized.shapeGuard.status==='FAIL'?'rejected':'accepted';
        chosen.metrics.quantizedObjective=finalized.components;
        chosen.metrics.hfValidation=finalized.validation;
        chosen.metrics.quantizedShapeGuard=finalized.shapeGuard;
        chosen.metrics.exactExportSimulation=true;
        chosen.metrics.activeBands=chosen.bands.length;
        chosen.metrics.maxBoost=Math.max(0,...chosen.bands.map(b=>b.gain));
        chosen.metrics.maxCut=Math.min(0,...chosen.bands.map(b=>b.gain));
        chosen.metrics.maxQ=Math.max(0,...chosen.bands.map(b=>b.q));
      }
      chosen.metrics.lossMode=lossMode;
      chosen.metrics.qAllowed=[0.30,10.00];
      chosen.metrics.qRangeSelected=useExtended?'0.30–10.00':'0.30–2.00';
      chosen.metrics.q2Candidate={rmse:q2.metrics.rmseAfter,p95:q2.metrics.p95After,max:q2.metrics.maxAfter,maxQ:q2.metrics.maxQ,shapeGuardAccepted:q2.metrics.shapeGuardAccepted,shapeGuardDeltaDb:q2.metrics.shapeGuardDeltaDb};
      chosen.metrics.q10Candidate={rmse:q10.metrics.rmseAfter,p95:q10.metrics.p95After,max:q10.metrics.maxAfter,maxQ:q10.metrics.maxQ,shapeGuardAccepted:q10.metrics.shapeGuardAccepted,shapeGuardDeltaDb:q10.metrics.shapeGuardDeltaDb};
      chosen.metrics.q10GlobalGuard=q10PassGlobal;chosen.metrics.q10ShapeGuard=q10PassShape;chosen.metrics.q10BranchShapeDeltaDb=branchShapeDelta;chosen.metrics.q10Committed=useExtended;
      return chosen;
    }finally{
      CFG.minQ=savedMinQ; CFG.maxQ=savedMaxQ; LOSS_MODE=previousLoss;
    }
  }

  // Re-score a reduced-grid candidate on the production grid before any
  // guard can accept it. This lets the balanced Huber screen be cheaper
  // without allowing coarse-grid optimism into the final decision.
  function rescoreCandidate(rawCurve,targetCurve,candidate){
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]);
    const hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    const freqs=logspace(lo,hi,CFG.balancedPoints||CFG.points);
    const raw=resample(rawCurve,freqs),target=resample(targetCurve,freqs);
    const aligned=alignLevel(freqs,raw,target),rawA=aligned.curve;
    const corrected=applyFilters(rawA,candidate.bands,freqs);
    const before=rawA.map((v,i)=>Math.abs(v-target[i])), after=corrected.map((v,i)=>Math.abs(v-target[i]));
    const metrics={...candidate.metrics,
      rmseBefore:rmse(before),rmseAfter:rmse(after),
      p95Before:percentile(before,.95),p95After:percentile(after,.95),
      maxBefore:Math.max(...before),maxAfter:Math.max(...after),
      activeBands:candidate.bands.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain).length,
      maxBoost:Math.max(...candidate.bands.map(b=>b.gain)),
      maxCut:Math.min(...candidate.bands.map(b=>b.gain)),
      maxQ:Math.max(...candidate.bands.map(b=>b.q)),
      levelOffsetDb:aligned.offset,coverage:[lo,hi],
      objective:distance(freqs,corrected,target)
    };
    return {bands:candidate.bands,metrics};
  }

  function huberGuardPass(candidate,reference){
    return candidate.metrics.shapeGuardAccepted !== false &&
      candidate.metrics.rmseAfter <= reference.metrics.rmseAfter-CFG.huberMinImprovementDb &&
      candidate.metrics.p95After <= reference.metrics.p95After+CFG.huberP95ToleranceDb &&
      candidate.metrics.maxAfter <= reference.metrics.maxAfter+CFG.huberMaxToleranceDb;
  }

  // Balanced branch: both objectives get the same conservative Q<=2 search,
  // but only the guarded winner pays for the expensive high-Q rescue. The
  // final Q10 candidate is still checked against its Q2 parent and Shape Guard.
  function optimizeBalanced(rawCurve,targetCurve){
    const previousLoss=LOSS_MODE;
    const savedMinQ=CFG.minQ, savedMaxQ=CFG.maxQ;
    let standardQ2, huberQ2;
    try{
      CFG.minQ=0.30; CFG.maxQ=2.00;
      LOSS_MODE='standard';
      standardQ2=optimizeSingle(rawCurve,targetCurve);
      standardQ2.metrics.lossMode='standard';
      standardQ2.metrics.qAllowed=[0.30,10.00];
      standardQ2.metrics.qRangeSelected='0.30–2.00';
      const robustNeeded=CFG.huberEnabled &&
        (standardQ2.metrics.maxAfter>CFG.huberTriggerMaxDb || standardQ2.metrics.p95After>CFG.huberTriggerP95Db);
      if(robustNeeded){
        LOSS_MODE='huber';
        const fullBalancedPoints=CFG.balancedPoints;
        try{
          CFG.balancedPoints=CFG.balancedHuberPoints;
          huberQ2=optimizeSingle(rawCurve,targetCurve);
        }finally{
          CFG.balancedPoints=fullBalancedPoints;
        }
        huberQ2=rescoreCandidate(rawCurve,targetCurve,huberQ2);
        huberQ2.metrics.lossMode='huber';
        huberQ2.metrics.qAllowed=[0.30,10.00];
        huberQ2.metrics.qRangeSelected='0.30–2.00';
      }
    }finally{
      CFG.minQ=savedMinQ; CFG.maxQ=savedMaxQ; LOSS_MODE=previousLoss;
    }

    const huberPass=!!huberQ2 && huberGuardPass(huberQ2,standardQ2);
    const selectedQ2=huberPass?huberQ2:standardQ2;
    const selectedLoss=huberPass?'huber':'standard';
    const chosen=optimizeBranchFromQ2(rawCurve,targetCurve,selectedQ2,selectedLoss);
    chosen.metrics.performanceMode='balanced';
    chosen.metrics.huberCommitted=huberPass;
    chosen.metrics.huberGuard={
      rmseImprovementDb:standardQ2.metrics.rmseAfter-(huberQ2?huberQ2.metrics.rmseAfter:standardQ2.metrics.rmseAfter),
      p95DeltaDb:huberQ2?huberQ2.metrics.p95After-standardQ2.metrics.p95After:0,
      maxDeltaDb:huberQ2?huberQ2.metrics.maxAfter-standardQ2.metrics.maxAfter:0,
      accepted:huberPass,
      comparisonStage:'q2_screen',
      skipped:!huberQ2,
      skipReason:!huberQ2?'standard_q2_within_robust_trigger':null,
      fallback:huberPass?'none':'standard_loss'
    };
    chosen.metrics.standardCandidate={rmse:standardQ2.metrics.rmseAfter,p95:standardQ2.metrics.p95After,max:standardQ2.metrics.maxAfter,maxQ:standardQ2.metrics.maxQ,qRangeSelected:standardQ2.metrics.qRangeSelected};
    chosen.metrics.huberCandidate=huberQ2?{rmse:huberQ2.metrics.rmseAfter,p95:huberQ2.metrics.p95After,max:huberQ2.metrics.maxAfter,maxQ:huberQ2.metrics.maxQ,qRangeSelected:huberQ2.metrics.qRangeSelected}:null;
    return chosen;
  }

  // Shared high-Q completion used by the balanced path. Keeping this logic
  // identical to the legacy branch preserves the existing transactional
  // Q10 and Target Topology Guard behavior.
  function optimizeBranchFromQ2(rawCurve,targetCurve,q2,lossMode){
    const previousLoss=LOSS_MODE;
    const savedMinQ=CFG.minQ, savedMaxQ=CFG.maxQ;
    LOSS_MODE=lossMode;
    try{
      CFG.minQ=0.30; CFG.maxQ=2.00;
      const q10=extendHighQFromQ2(rawCurve,targetCurve,q2);
      const q10PassGlobal =
        q10.metrics.rmseAfter <= q2.metrics.rmseAfter - 0.002 &&
        q10.metrics.p95After <= q2.metrics.p95After + 0.05 &&
        q10.metrics.maxAfter <= q2.metrics.maxAfter + 0.05;
      const branchShapeDelta = (q10.metrics.shapeGuardFinalRmsDb!=null && q2.metrics.shapeGuardFinalRmsDb!=null)
        ? q10.metrics.shapeGuardFinalRmsDb-q2.metrics.shapeGuardFinalRmsDb : null;
      const q10PassShape = q10.metrics.shapeGuardAccepted !== false &&
        (branchShapeDelta==null || branchShapeDelta<=CFG.topologyToleranceDb+1e-12);
      const useExtended=q10PassGlobal&&q10PassShape;
      const chosen=useExtended?q10:q2;
      const finalized=finalizeQuantized(rawCurve,targetCurve,chosen.bands);
      chosen.metrics.quantizationRescue=finalized.finite&&finalized.validation.status!=='FAIL'&&finalized.shapeGuard.status!=='FAIL'?'accepted':'rejected';
      if(finalized.finite){
        chosen.bands=finalized.bands;
        chosen.metrics.quantizedObjective=finalized.components;
        chosen.metrics.hfValidation=finalized.validation;
        chosen.metrics.quantizedShapeGuard=finalized.shapeGuard;
        chosen.metrics.exactExportSimulation=true;
        chosen.metrics.activeBands=chosen.bands.length;
        chosen.metrics.maxBoost=Math.max(0,...chosen.bands.map(b=>b.gain));
        chosen.metrics.maxCut=Math.min(0,...chosen.bands.map(b=>b.gain));
        chosen.metrics.maxQ=Math.max(0,...chosen.bands.map(b=>b.q));
      }
      chosen.metrics.lossMode=lossMode;
      chosen.metrics.qAllowed=[0.30,10.00];
      chosen.metrics.qRangeSelected=useExtended?'0.30–10.00':'0.30–2.00';
      chosen.metrics.q2Candidate={rmse:q2.metrics.rmseAfter,p95:q2.metrics.p95After,max:q2.metrics.maxAfter,maxQ:q2.metrics.maxQ,shapeGuardAccepted:q2.metrics.shapeGuardAccepted,shapeGuardDeltaDb:q2.metrics.shapeGuardDeltaDb};
      chosen.metrics.q10Candidate={rmse:q10.metrics.rmseAfter,p95:q10.metrics.p95After,max:q10.metrics.maxAfter,maxQ:q10.metrics.maxQ,shapeGuardAccepted:q10.metrics.shapeGuardAccepted,shapeGuardDeltaDb:q10.metrics.shapeGuardDeltaDb};
      chosen.metrics.q10GlobalGuard=q10PassGlobal;chosen.metrics.q10ShapeGuard=q10PassShape;chosen.metrics.q10BranchShapeDeltaDb=branchShapeDelta;chosen.metrics.q10Committed=useExtended;
      return chosen;
    }finally{
      CFG.minQ=savedMinQ; CFG.maxQ=savedMaxQ; LOSS_MODE=previousLoss;
    }
  }

  /* ============================================================
     vNext4 LM + IRLS production solver
     ------------------------------------------------------------
     - Robust Target remains the sole numerical target.
     - Exact RBJ responses are used in every model/Jacobian evaluation.
     - Levenberg-Marquardt jointly refines Fc/Gain/Q.
     - Huber is applied by IRLS as an outlier-robust fitting component.
     - Band count is grown inside the solve and never sliced after solving.
     - Final exported/quantized filters are exactly re-simulated.
     ============================================================ */

  const RESPONSE_CACHE = new WeakMap();
  function cachedRbj(freqs,b){
    let cache=RESPONSE_CACHE.get(freqs);
    if(!cache){cache=new Map();RESPONSE_CACHE.set(freqs,cache);}
    const key=`${Number(b.freq).toFixed(5)}|${Number(b.gain).toFixed(5)}|${Number(b.q).toFixed(5)}`;
    let r=cache.get(key);
    if(!r){r=rbj(freqs,b); if(cache.size>6000)cache.clear(); cache.set(key,r);}
    return r;
  }

  function applyFiltersCached(fr,bands,freqs){
    const out=new Float64Array(fr);
    for(const b of bands){
      if(Math.abs(b.gain)<1e-12)continue;
      const r=cachedRbj(freqs,b);
      for(let i=0;i<out.length;i++)out[i]+=r[i];
    }
    return out;
  }

  function huberWeight(e,delta=CFG.huberDeltaDb,epsilon=CFG.huberEpsilonDb){
    const a=Math.max(0,Math.abs(e)-epsilon);
    return a<=delta||a===0 ? 1 : delta/a;
  }

  function solverFitCost(errors,bands,mode='standard'){
    let fit=0;
    if(mode==='huber'){
      for(const e of errors)fit+=deadZoneHuber(e);
      fit=2*fit/Math.max(1,errors.length);
    }else{
      for(const e of errors){const a=Math.max(0,Math.abs(e)-CFG.huberEpsilonDb);fit+=a*a;}
      fit/=Math.max(1,errors.length);
    }
    const sc=CFG.solver||{};
    let reg=0;
    for(const b of bands){
      reg+=(sc.gainRegularization||0)*b.gain*b.gain;
      reg+=(sc.highQRegularization||0)*Math.max(0,b.q-2)**2*Math.max(0.25,Math.abs(b.gain));
      const d=Math.abs(Math.log2(Math.max(1,b.freq)/(sc.boundaryHz||1000)));
      if(d<(sc.boundaryWidthOct||0.10) && b.q>2)reg+=(sc.boundaryHighQPenalty||0)*Math.pow(b.q-2,2);
    }
    return fit+reg;
  }

  function paramsFromBands(bands){
    const p=[];
    for(const b of bands)p.push(Math.log(clamp(b.freq,CFG.minFreq,CFG.maxFreq)),clamp(b.gain,CFG.minGain,CFG.maxGain),Math.log(clamp(b.q,CFG.minQ,CFG.maxQ)));
    return p;
  }

  function bandsFromParams(params){
    const bands=[];
    for(let i=0;i<params.length;i+=3){
      bands.push({
        freq:clamp(Math.exp(params[i]),CFG.minFreq,CFG.maxFreq),
        gain:clamp(params[i+1],CFG.minGain,CFG.maxGain),
        q:clamp(Math.exp(params[i+2]),CFG.minQ,CFG.maxQ)
      });
    }
    return bands.sort((a,b)=>a.freq-b.freq);
  }

  function solveLinearSystem(A,b){
    const n=b.length, M=A.map((row,i)=>Array.from(row).concat([b[i]]));
    for(let col=0;col<n;col++){
      let pivot=col;
      for(let r=col+1;r<n;r++)if(Math.abs(M[r][col])>Math.abs(M[pivot][col]))pivot=r;
      if(!Number.isFinite(M[pivot][col])||Math.abs(M[pivot][col])<1e-12)return null;
      if(pivot!==col){const tmp=M[col];M[col]=M[pivot];M[pivot]=tmp;}
      const d=M[col][col];
      for(let j=col;j<=n;j++)M[col][j]/=d;
      for(let r=0;r<n;r++){
        if(r===col)continue;
        const f=M[r][col]; if(Math.abs(f)<1e-18)continue;
        for(let j=col;j<=n;j++)M[r][j]-=f*M[col][j];
      }
    }
    const x=M.map(row=>row[n]);
    return x.every(Number.isFinite)?x:null;
  }

  function modelAndErrors(freqs,base,target,bands){
    const curve=applyFiltersCached(base,bands,freqs), errors=new Float64Array(freqs.length);
    for(let i=0;i<errors.length;i++)errors[i]=curve[i]-target[i];
    return {curve,errors};
  }

  function jacobianColumns(freqs,bands){
    const sc=CFG.solver||{}, cols=[];
    for(let bi=0;bi<bands.length;bi++){
      const b=bands[bi];
      const specs=[
        {kind:'freq',step:sc.freqJacobianStep||0.003},
        {kind:'gain',step:sc.gainJacobianStepDb||0.03},
        {kind:'q',step:sc.qJacobianStep||0.01}
      ];
      for(const spec of specs){
        let lo={...b},hi={...b},den=2*spec.step;
        if(spec.kind==='freq'){
          lo.freq=clamp(b.freq*Math.exp(-spec.step),CFG.minFreq,CFG.maxFreq);
          hi.freq=clamp(b.freq*Math.exp(spec.step),CFG.minFreq,CFG.maxFreq);
          den=Math.log(hi.freq)-Math.log(lo.freq);
        }else if(spec.kind==='gain'){
          lo.gain=clamp(b.gain-spec.step,CFG.minGain,CFG.maxGain);
          hi.gain=clamp(b.gain+spec.step,CFG.minGain,CFG.maxGain);
          den=hi.gain-lo.gain;
        }else{
          lo.q=clamp(b.q*Math.exp(-spec.step),CFG.minQ,CFG.maxQ);
          hi.q=clamp(b.q*Math.exp(spec.step),CFG.minQ,CFG.maxQ);
          den=Math.log(hi.q)-Math.log(lo.q);
        }
        const rl=cachedRbj(freqs,lo),rh=cachedRbj(freqs,hi),col=new Float64Array(freqs.length);
        const safe=Math.abs(den)>1e-12?den:1;
        for(let i=0;i<col.length;i++)col[i]=(rh[i]-rl[i])/safe;
        cols.push(col);
      }
    }
    return cols;
  }

  function lmRefine(freqs,base,target,seedBands,mode='standard'){
    const sc=CFG.solver||{};
    let bands=strip(seedBands).filter(b=>Math.abs(b.gain)>=0.01);
    if(!bands.length)return {bands:[],iterations:0,acceptedSteps:0,rejectedSteps:0,cost:Infinity,converged:true};
    let state=modelAndErrors(freqs,base,target,bands);
    let cost=solverFitCost(state.errors,bands,mode), lambda=sc.lambdaInitial||0.03;
    let accepted=0,rejected=0,converged=false,iterations=0;
    const maxOuter=mode==='huber'?(sc.irlsIterations||3):1;
    const maxIter=Math.max(2,Math.ceil((sc.maxIterations||12)/maxOuter));

    for(let outer=0;outer<maxOuter;outer++){
      for(let iter=0;iter<maxIter;iter++){
        iterations++;
        state=modelAndErrors(freqs,base,target,bands);
        const errors=state.errors, weights=new Float64Array(errors.length);
        for(let i=0;i<weights.length;i++)weights[i]=mode==='huber'?huberWeight(errors[i]):1;
        const J=jacobianColumns(freqs,bands), m=J.length;
        if(!m){converged=true;break;}
        const A=Array.from({length:m},()=>new Float64Array(m)), g=new Float64Array(m);
        for(let j=0;j<m;j++){
          const cj=J[j]; let gj=0;
          for(let i=0;i<errors.length;i++)gj+=weights[i]*cj[i]*errors[i];
          g[j]=gj;
          for(let k=0;k<=j;k++){
            const ck=J[k]; let v=0;
            for(let i=0;i<errors.length;i++)v+=weights[i]*cj[i]*ck[i];
            A[j][k]=A[k][j]=v;
          }
        }
        for(let j=0;j<m;j++)A[j][j]+=lambda*(A[j][j]+1e-6);
        const delta=solveLinearSystem(A,Array.from(g,x=>-x));
        if(!delta){lambda*=sc.lambdaUp||6;rejected++;continue;}
        let norm=0;for(const d of delta)norm+=d*d;norm=Math.sqrt(norm);
        if(norm<(sc.stepTolerance||1e-4)){converged=true;break;}
        const p=paramsFromBands(bands);
        for(let j=0;j<p.length;j++)p[j]+=clamp(delta[j],-0.75,0.75);
        const trialBands=bandsFromParams(p);
        const trial=modelAndErrors(freqs,base,target,trialBands);
        const trialCost=solverFitCost(trial.errors,trialBands,mode);
        if(Number.isFinite(trialCost)&&trialCost<cost-(sc.lossTolerance||1e-7)){
          const improvement=cost-trialCost;
          bands=trialBands;cost=trialCost;accepted++;lambda=Math.max(1e-8,lambda*(sc.lambdaDown||0.35));rejected=0;
          if(improvement<(sc.lossTolerance||1e-7)){converged=true;break;}
        }else{
          lambda=Math.min(1e8,lambda*(sc.lambdaUp||6));rejected++;
          if(rejected>=(sc.maxRejectedSteps||5))break;
        }
      }
      if(converged)break;
    }
    return {bands:strip(bands),iterations,acceptedSteps:accepted,rejectedSteps:rejected,cost,converged};
  }

  function candidatePoolForState(freqs,base,target,currentBands=[]){
    const current=applyFiltersCached(base,currentBands,freqs), residual=Array.from(current,(v,i)=>v-target[i]);
    const featureSeeds=featureAnalysis(freqs,residual).map(f=>{
      const bw=Math.max((CFG.feature||{}).minWidthOct||0.05,f.widthOct);
      const x=Math.pow(2,bw), q=clamp(Math.sqrt(x)/Math.max(1e-9,x-1),CFG.minQ,2.0);
      return {freq:f.freq,gain:clamp(f.gain,CFG.minGain,CFG.maxGain),q,risk:boostRisk(f),support:f.support};
    });
    const regions=searchCandidates(freqs,current,target,0.25).map(c=>({...c,q:clamp(c.q,CFG.minQ,2.0)}));
    let pool=augmentCandidates(freqs,current,target,regions.concat(featureSeeds));
    pool=pool.filter(c=>Math.abs(c.gain)>=CFG.minSeedGain && c.freq>=CFG.minFreq && c.freq<=CFG.maxFreq);
    pool=rankCandidates(freqs,current,target,pool);
    const out=[];
    for(const c of pool){
      if(currentBands.some(b=>Math.abs(Math.log2(b.freq/c.freq))<0.055 && Math.abs(b.q-c.q)<0.35))continue;
      if(out.some(b=>Math.abs(Math.log2(b.freq/c.freq))<CFG.minSeedSeparationOct*0.55))continue;
      out.push({freq:c.freq,gain:c.gain,q:c.q});
      if(out.length>=(CFG.solver||{}).candidateLimit)break;
    }
    return out;
  }

  function addBandsToCount(freqs,base,target,bands,count){
    const work=cloneBands(bands);
    let safety=0;
    while(work.length<count && safety++<count*3){
      const pool=candidatePoolForState(freqs,base,target,work);
      if(!pool.length)break;
      work.push(pool[0]);work.sort((a,b)=>a.freq-b.freq);
    }
    return work;
  }

  function solutionMetrics(freqs,base,target,bands){
    const curve=applyFiltersCached(base,bands,freqs), comp=objectiveComponents(freqs,curve,target,bands);
    return {curve,comp};
  }

  function bandGrowthAccept(previous,next){
    if(!previous)return true;
    const p=previous.comp,n=next.comp, min=(CFG.solver||{}).minBandImprovementDb||0.004;
    const rmseGain=p.rmse-n.rmse;
    const bounded=n.p95<=p.p95+0.04 && n.max<=p.max+0.05 && n.erbError<=p.erbError+0.025 && n.narrowError<=p.narrowError+0.035;
    return bounded && (rmseGain>=min || (p.score-n.score)>=min*0.5);
  }

  function responseAwarePruneLM(freqs,base,target,bands,mode){
    // Fast active-set pruning: removal is judged on the exact current response;
    // one joint LM refinement is performed after pruning, not once per trial.
    let work=cloneBands(bands), current=solutionMetrics(freqs,base,target,work);
    // Small/low-value bands are tested first.
    const order=work.map((b,i)=>({i,mag:Math.abs(b.gain)})).sort((a,b)=>a.mag-b.mag).map(x=>x.i);
    const removed=new Set();
    for(const originalIndex of order){
      if(removed.has(originalIndex))continue;
      const trial=work.filter((_,j)=>j!==originalIndex && !removed.has(j));
      const m=solutionMetrics(freqs,base,target,trial);
      if(m.comp.rmse<=current.comp.rmse+0.006 && m.comp.p95<=current.comp.p95+0.04 && m.comp.max<=current.comp.max+0.05){removed.add(originalIndex);current=m;}
    }
    work=work.filter((_,i)=>!removed.has(i));
    if(work.length)work=lmRefine(freqs,base,target,work,mode).bands;
    return work;
  }

  function fastRedundancyCleanup(freqs,base,target,bands,mode){
    let work=cloneBands(bands).sort((a,b)=>a.freq-b.freq), changed=true;
    while(changed){
      changed=false;
      const baseline=solutionMetrics(freqs,base,target,work);
      for(let i=0;i<work.length-1;i++){
        const a=work[i],b=work[i+1];
        if(Math.sign(a.gain)!==Math.sign(b.gain))continue;
        if(Math.abs(Math.log2(a.freq/b.freq))>0.045 || Math.abs(a.q-b.q)>0.25)continue;
        const drop=Math.abs(a.gain)<=Math.abs(b.gain)?i:i+1;
        const trial=work.filter((_,j)=>j!==drop), m=solutionMetrics(freqs,base,target,trial);
        if(m.comp.rmse<=baseline.comp.rmse+0.004 && m.comp.p95<=baseline.comp.p95+0.025 && m.comp.max<=baseline.comp.max+0.04){
          work=trial.length?lmRefine(freqs,base,target,trial,mode).bands:[];changed=true;break;
        }
      }
    }
    return work;
  }

  function stabilityPerturbationTest(freqs,base,target,bands){
    const baseline=solutionMetrics(freqs,base,target,bands).comp.rmse;
    let worst=0,sum=0,n=0;
    for(let i=0;i<bands.length;i++){
      const variants=[
        {...bands[i],freq:clamp(bands[i].freq*0.995,CFG.minFreq,CFG.maxFreq)},
        {...bands[i],freq:clamp(bands[i].freq*1.005,CFG.minFreq,CFG.maxFreq)},
        {...bands[i],gain:clamp(bands[i].gain-0.05,CFG.minGain,CFG.maxGain)},
        {...bands[i],gain:clamp(bands[i].gain+0.05,CFG.minGain,CFG.maxGain)},
        {...bands[i],q:clamp(bands[i].q*0.98,CFG.minQ,CFG.maxQ)},
        {...bands[i],q:clamp(bands[i].q*1.02,CFG.minQ,CFG.maxQ)}
      ];
      for(const v of variants){
        const trial=cloneBands(bands);trial[i]=v;
        const d=Math.max(0,solutionMetrics(freqs,base,target,trial).comp.rmse-baseline);
        worst=Math.max(worst,d);sum+=d;n++;
      }
    }
    const limit=(CFG.solver||{}).perturbationMaxRmseDeltaDb||0.08;
    return {status:worst<=limit?'PASS':'WARN',baselineRmse:baseline,worstRmseDeltaDb:worst,meanRmseDeltaDb:n?sum/n:0,tests:n,limitDb:limit};
  }

  function hfSafetyRescue(rawCurve,freqs,base,target,bands){
    let work=cloneBands(bands), validation=highFrequencyValidation(rawCurve,work);
    if(validation.status!=='FAIL')return {bands:work,validation,rescued:false,scale:1};
    const baseline=solutionMetrics(freqs,base,target,work).comp;
    let best=null;
    const consider=(trial,meta)=>{
      const qtrial=quantizeBands(trial), hf=highFrequencyValidation(rawCurve,qtrial);
      if(hf.status!=='PASS')return;
      const m=solutionMetrics(freqs,base,target,qtrial).comp;
      if(!best || m.rmse<best.metrics.rmse-1e-9 || (Math.abs(m.rmse-best.metrics.rmse)<1e-9&&m.p95<best.metrics.p95))best={bands:qtrial,validation:hf,metrics:m,...meta};
    };
    // First search a coherent transform of all positive near-boundary filters.
    // Moving them slightly down and/or narrowing Q often preserves 10-12 kHz
    // fit better than simply reducing all gains.
    for(const fScale of [0.84,0.88,0.92,0.96,1.00]){
      for(const qScale of [0.8,1.0,1.2,1.5,1.8,2.2]){
        for(const gScale of [0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.98]){
          const trial=work.map(b=>b.freq>=9000&&b.gain>0?{
            ...b,
            freq:clamp(b.freq*fScale,CFG.minFreq,CFG.maxFreq),
            q:clamp(b.q*qScale,CFG.minQ,CFG.maxQ),
            gain:clamp(b.gain*gScale,CFG.minGain,CFG.maxGain)
          }:{...b});
          consider(trial,{scale:gScale,freqScale:fScale,qScale});
        }
      }
    }
    if(best)return {bands:best.bands,validation:best.validation,rescued:true,scale:best.scale,freqScale:best.freqScale,qScale:best.qScale,rmseDeltaDb:best.metrics.rmse-baseline.rmse};
    return {bands:work,validation,rescued:false,scale:1,freqScale:1,qScale:1,rmseDeltaDb:0};
  }

  function quantizationRescue(freqs,base,target,bands){
    let work=quantizeBands(bands), current=solutionMetrics(freqs,base,target,work);
    const qz=CFG.quantization||{}, passes=(CFG.solver||{}).quantizationRescuePasses||1;
    for(let pass=0;pass<passes;pass++){
      let changed=false;
      for(let i=0;i<work.length;i++){
        const b=work[i], others=work.filter((_,j)=>j!==i), baseWithout=applyFiltersCached(base,others,freqs);
        let best={...b}, bestComp=current.comp;
        for(const df of [-1,0,1])for(const dg of [-1,0,1])for(const dq of [-1,0,1]){
          if(df===0&&dg===0&&dq===0)continue;
          const cand={
            freq:clamp(b.freq+df*(qz.freqHz||1),CFG.minFreq,CFG.maxFreq),
            gain:clamp(b.gain+dg*(qz.gainDb||0.01),CFG.minGain,CFG.maxGain),
            q:clamp(b.q+dq*(qz.q||0.01),CFG.minQ,CFG.maxQ)
          };
          if(!biquadSafety(cand).stable)continue;
          const curve=applyFiltersCached(baseWithout,[cand],freqs), comp=objectiveComponents(freqs,curve,target,others.concat([cand]));
          if(comp.rmse<bestComp.rmse-1e-6 && comp.p95<=bestComp.p95+0.01 && comp.max<=bestComp.max+0.02){best=cand;bestComp=comp;}
        }
        if(best.freq!==b.freq||best.gain!==b.gain||best.q!==b.q){work[i]=best;current=solutionMetrics(freqs,base,target,work);changed=true;}
      }
      if(!changed)break;
    }
    return work.sort((a,b)=>a.freq-b.freq);
  }

  function optimizeLMIRLS(rawCurve,targetCurve){
    const started=(typeof performance!=='undefined'&&performance.now)?performance.now():Date.now();
    const lo=Math.max(CFG.minFreq,rawCurve[0][0],targetCurve[0][0]);
    const hi=Math.min(CFG.maxFreq,rawCurve.at(-1)[0],targetCurve.at(-1)[0]);
    if(hi<=lo||hi<2000)throw Error('Raw Pudding and Robust Target have insufficient frequency overlap.');
    const sc=CFG.solver||{}, grids=resolutionGrids(lo,hi);
    const lmN=Math.min(Math.max(48,sc.lmPoints||240),grids.R0.length);
    const lmFreqs=logspace(lo,hi,lmN), rawLM=resample(rawCurve,lmFreqs), targetLM=resample(targetCurve,lmFreqs);
    const alignedLM=alignLevel(lmFreqs,rawLM,targetLM), baseLM=alignedLM.curve;
    const growth=(sc.growth||[3,5,7,10]).map(n=>Math.min(n,CFG.bands)).filter((n,i,a)=>n>0&&a.indexOf(n)===i).sort((a,b)=>a-b);
    if(!growth.includes(CFG.bands))growth.push(CFG.bands);
    let bands=[], acceptedState=null, trace=[], totalIterations=0, totalAccepted=0, totalRejected=0;

    for(const count of growth){
      const seeded=addBandsToCount(lmFreqs,baseLM,targetLM,bands,count);
      if(seeded.length<=bands.length && bands.length)break;
      const standard=lmRefine(lmFreqs,baseLM,targetLM,seeded,'standard');
      totalIterations+=standard.iterations;totalAccepted+=standard.acceptedSteps;totalRejected+=standard.rejectedSteps;
      let chosen=standard, branch='standard';
      if(CFG.huberEnabled){
        const robust=lmRefine(lmFreqs,baseLM,targetLM,standard.bands,'huber');
        totalIterations+=robust.iterations;totalAccepted+=robust.acceptedSteps;totalRejected+=robust.rejectedSteps;
        const sm=solutionMetrics(lmFreqs,baseLM,targetLM,standard.bands), rm=solutionMetrics(lmFreqs,baseLM,targetLM,robust.bands);
        const standardHuberCost=solverFitCost(modelAndErrors(lmFreqs,baseLM,targetLM,standard.bands).errors,standard.bands,'huber');
        const robustHuberCost=solverFitCost(modelAndErrors(lmFreqs,baseLM,targetLM,robust.bands).errors,robust.bands,'huber');
        const robustPass=robustHuberCost<standardHuberCost-1e-8 && rm.comp.rmse<=sm.comp.rmse+0.015 && rm.comp.p95<=sm.comp.p95+CFG.huberP95ToleranceDb && rm.comp.max<=sm.comp.max+CFG.huberMaxToleranceDb && rm.comp.score<=sm.comp.score+0.003;
        if(robustPass){chosen=robust;branch='huber-irls';}
      }
      let candidateBands=responseAwarePruneLM(lmFreqs,baseLM,targetLM,chosen.bands,branch==='huber-irls'?'huber':'standard');
      candidateBands=fastRedundancyCleanup(lmFreqs,baseLM,targetLM,candidateBands,branch==='huber-irls'?'huber':'standard');
      const state=solutionMetrics(lmFreqs,baseLM,targetLM,candidateBands);
      const accepted=bandGrowthAccept(acceptedState,state);
      trace.push({requestedBands:count,actualBands:candidateBands.length,rmse:state.comp.rmse,p95:state.comp.p95,max:state.comp.max,erb:state.comp.erbError,narrow:state.comp.narrowError,score:state.comp.score,branch,accepted});
      if(accepted){bands=candidateBands;acceptedState=state;}else break;
      if(bands.length>=CFG.bands)break;
    }

    if(!bands.length)throw Error('LM/IRLS solver could not initialize a valid PEQ solution.');

    // Dense final joint refinement; Huber IRLS remains active, then exact guards.
    const f=grids.R0, raw=resample(rawCurve,f), target=resample(targetCurve,f), aligned=alignLevel(f,raw,target), base=aligned.curve;
    let final=lmRefine(f,base,target,bands,CFG.huberEnabled?'huber':'standard');
    totalIterations+=final.iterations;totalAccepted+=final.acceptedSteps;totalRejected+=final.rejectedSteps;
    bands=responseAwarePruneLM(f,base,target,final.bands,CFG.huberEnabled?'huber':'standard');
    bands=fastRedundancyCleanup(f,base,target,bands,CFG.huberEnabled?'huber':'standard');
    bands=strip(bands).filter(b=>Math.abs(b.gain)>=CFG.minActiveGain);
    if(bands.length>CFG.bands)throw Error('LM/IRLS active-set invariant violated: more than '+CFG.bands+' bands.');
    for(const b of bands)if(!biquadSafety(b).stable)throw Error('LM/IRLS produced an unstable biquad.');

    const continuous=solutionMetrics(f,base,target,bands);
    let qbands=quantizationRescue(f,base,target,bands);
    qbands=responseAwarePrune(f,base,target,qbands);
    if(qbands.length>CFG.bands)throw Error('Quantized active-set invariant violated.');
    const hfRescue=hfSafetyRescue(rawCurve,f,base,target,qbands);
    qbands=hfRescue.bands;
    const corrected=applyFiltersCached(base,qbands,f), before=Array.from(base,(v,i)=>Math.abs(v-target[i])), after=Array.from(corrected,(v,i)=>Math.abs(v-target[i]));
    const hf=hfRescue.validation, qcomp=objectiveComponents(f,corrected,target,qbands,hf);
    const qtop=topologyError(f,corrected,target), ctop=topologyError(f,continuous.curve,target);
    const shapeGuard={status:qtop<=ctop+CFG.topologyToleranceDb?'PASS':'FAIL',deltaDb:qtop-ctop};
    const stability=stabilityPerturbationTest(f,base,target,qbands);
    const headroom=modeledHeadroom(qbands);
    const elapsed=((typeof performance!=='undefined'&&performance.now)?performance.now():Date.now())-started;
    return {bands:qbands,metrics:{
      rmseBefore:rmse(before),rmseAfter:rmse(after),p95Before:percentile(before,.95),p95After:percentile(after,.95),maxBefore:Math.max(...before),maxAfter:Math.max(...after),
      activeBands:qbands.length,maxBoost:Math.max(0,...qbands.map(b=>b.gain)),maxCut:Math.min(0,...qbands.map(b=>b.gain)),maxQ:Math.max(0,...qbands.map(b=>b.q)),
      levelOffsetDb:aligned.offset,coverage:[lo,hi],objective:qcomp.score,objectiveComponents:qcomp,erbError:qcomp.erbError,narrowError:qcomp.narrowError,
      shapeGuardEnabled:true,shapeGuardAccepted:shapeGuard.status==='PASS',shapeGuardReason:shapeGuard.status==='PASS'?'target_relative_pass':'target_relative_warning',shapeGuardDeltaDb:shapeGuard.deltaDb,
      shapeGuardPreRmsDb:ctop,shapeGuardFinalRmsDb:qtop,quantizationRescue:'accepted',quantizedObjective:qcomp,hfValidation:hf,quantizedShapeGuard:shapeGuard,exactExportSimulation:true,
      qAllowed:[0.30,10.00],qRangeSelected:qbands.some(b=>b.q>2)?'0.30–10.00':'0.30–2.00',q10Committed:qbands.some(b=>b.q>2),lossMode:CFG.huberEnabled?'LM + Huber IRLS':'LM least-squares',performanceMode:'lm-irls',huberCommitted:CFG.huberEnabled,
      solver:{name:'Levenberg-Marquardt + Huber IRLS',jointParameters:['log(Fc)','Gain','log(Q)'],iterations:totalIterations,acceptedSteps:totalAccepted,rejectedSteps:totalRejected,bandGrowthTrace:trace,elapsedMs:elapsed,continuousRmse:continuous.comp.rmse,quantizedRmse:qcomp.rmse,responseCache:true},
      stabilityPerturbation:stability,hfSafetyRescue:{applied:hfRescue.rescued,scale:hfRescue.scale,rmseDeltaDb:hfRescue.rmseDeltaDb||0},modeledHeadroom:headroom,resolution:{R0:f.length,R1:grids.R1.length,R2:grids.R2.length},loudnessCompensation:loudness
    }};
  }

  function optimize(rawCurve,targetCurve){
    if((CFG.solver||{}).mode==='lm-irls'){
      try{return optimizeLMIRLS(rawCurve,targetCurve);}
      catch(e){
        // Transactional fallback keeps the app usable if a numerical edge case
        // defeats LM. The fallback is explicit in metrics and never silent.
        const fallback=CFG.performanceMode==='balanced'?optimizeBalanced(rawCurve,targetCurve):optimizeBranch(rawCurve,targetCurve,'standard');
        fallback.metrics.solverFallback={from:'LM + Huber IRLS',to:'legacy constrained solver',reason:String(e&&e.message||e)};
        fallback.metrics.performanceMode='legacy-fallback';
        return fallback;
      }
    }
    if(CFG.performanceMode==='balanced')return optimizeBalanced(rawCurve,targetCurve);
    const standard=optimizeBranch(rawCurve,targetCurve,'standard');
    if(!CFG.huberEnabled){
      standard.metrics.huberCommitted=false;
      return standard;
    }
    const huber=optimizeBranch(rawCurve,targetCurve,'huber');
    const huberPass = huberGuardPass(huber,standard);
    const chosen=huberPass?huber:standard;
    chosen.metrics.huberCommitted=huberPass;
    chosen.metrics.huberGuard={
      rmseImprovementDb:standard.metrics.rmseAfter-huber.metrics.rmseAfter,
      p95DeltaDb:huber.metrics.p95After-standard.metrics.p95After,
      maxDeltaDb:huber.metrics.maxAfter-standard.metrics.maxAfter,
      accepted:huberPass,
      comparisonStage:'full_branch',
      fallback:huberPass?'none':'standard_loss'
    };
    chosen.metrics.performanceMode='exhaustive';
    chosen.metrics.standardCandidate={rmse:standard.metrics.rmseAfter,p95:standard.metrics.p95After,max:standard.metrics.maxAfter,maxQ:standard.metrics.maxQ,qRangeSelected:standard.metrics.qRangeSelected};
    chosen.metrics.huberCandidate={rmse:huber.metrics.rmseAfter,p95:huber.metrics.p95After,max:huber.metrics.maxAfter,maxQ:huber.metrics.maxQ,qRangeSelected:huber.metrics.qRangeSelected};
    return chosen;
  }
  const fmt=v=>Math.abs(v-Math.round(v))<1e-9?String(Math.round(v)):v.toFixed(1);
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));


  // v4.2.1 Moondrop Loudness Layer: post-solver level preservation + optional ISO226 approximation hook
  function loudnessCompensation(rawCurve, correctedCurve, freqs){
    const band=freqs.map((f,i)=>f>=100&&f<=10000?i:null).filter(i=>i!==null);
    if(!band.length)return {gainDb:0,iso226Applied:false};
    const avg=a=>band.reduce((x,i)=>x+a[i],0)/band.length;
    const loss=avg(rawCurve)-avg(correctedCurve);
    return {gainDb:Math.max(0,Math.min(4,loss)),iso226Applied:false,reference:'100Hz-10kHz mean level match'};
  }

  function formatPEQ(result){
    return result.bands.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain).map((b,i)=>
      `${i===0&&result.metrics?.loudnessCompensation?.gainDb?('Preamp: +'+result.metrics.loudnessCompensation.gainDb.toFixed(2)+' dB\n'):''}Filter ${i+1}: PK ${fmt(b.freq)} Hz, ${b.gain>=0?'+':''}${b.gain.toFixed(2)} dB, Q ${b.q.toFixed(2)}`
    ).join('\n')+'\n';
  }

  function jsonPEQ(result,meta){
    return JSON.stringify({
      device:'MOONDROP PUDDING',
      engine:'IEM EarPrint PEQ Solver',
      version:CFG.version,
      cleanup:{method:'response-aware same-sign adjacent-filter merge',local_rms_limit_db:0.12,local_max_limit_db:0.25,global_rmse_tolerance_db:0.003},
      filter_type:'PK',
      implementation:'MOONDROP Link-compatible constraint set',
      dsp_model:{type:'RBJ peaking biquad',sample_rate_hz:CFG.sampleRate,status:'PROVISIONAL'},
      level_alignment:{method:'mean raw-target error over 100 Hz–10 kHz',removed_offset_db:result.metrics.levelOffsetDb},
      optimizer:{initialization:'Evidence-driven residual features + bandwidth-derived Q; active band growth inside the <=10-band constraint',
                 algorithm:'Joint Levenberg-Marquardt refinement of log(Fc), Gain and log(Q), followed by Huber IRLS robust refinement',
                 loss:'exact pointwise residual fit with Huber IRLS robustness; ERB/narrow/topology/complexity metrics used as bounded validation criteria',
                 robust_loss:{name:'Huber IRLS',delta_db:CFG.huberDeltaDb,enabled:CFG.huberEnabled,selection:'transactional robust refinement with bounded RMSE/P95/max/secondary-metric guards'},
                 performance_mode:result.metrics.performanceMode||CFG.performanceMode,
                 active_set:'3 -> 5 -> 7 -> <=10 bands, stopping when validated marginal improvement is negligible',
                 exact_model:'RBJ peaking biquad at every candidate, Jacobian and final export validation'},
      constraints:{bands:CFG.bands,device_frequency_hz:[20,20000],correction_domain_hz:[CFG.minFreq,CFG.maxFreq],gain_db:[CFG.minGain,CFG.maxGain],q:[0.30,10.00],q_range_selected:result.metrics.qRangeSelected},
      source:meta,metrics:result.metrics,target_topology_guard:{reference:'selected Robust Target',tolerance_db:CFG.topologyToleranceDb},peq:result.bands.filter(b=>Math.abs(b.gain)>=CFG.minActiveGain).slice(0,CFG.bands)
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

  function cleanDisplayName(n){return String(n||'').replace(/\.txt$/i,'').replace(/_/g,' ').replace(/\s+/g,' ').trim();}
  let targetRegistry=[];
  async function populate(){
    try{
      const outputs=await ghList('output');
      targetRegistry=(Array.isArray(outputs)?outputs:[]).filter(x=>x.name&&/__robust_target\.txt$/i.test(x.name)).map(x=>({id:x.name.replace(/__robust_target\.txt$/i,''),displayName:cleanDisplayName(x.name.replace(/__robust_target\.txt$/i,'')),robustTargetPath:'output/'+x.name,status:'READY'}));
      const sel=$('puddingTargetSelect'); if(sel){sel.innerHTML=targetRegistry.map(t=>`<option value="${esc(t.id)}">${esc(t.displayName)}</option>`).join('');}
      status(targetRegistry.length?'Ready · fixed Moondrop Pudding Original 711 · '+targetRegistry.length+' Robust Targets':'No READY Robust Target found.');
    }catch(e){status(e.message,'warn');}
  }
  function selectedTarget(){const id=$('puddingTargetSelect')?.value; const t=targetRegistry.find(x=>x.id===id); if(!t)throw Error('Selected Robust Target is unavailable.'); return t;}
  let last=null,lastMeta=null;

  async function generate(){
    const button=$('generatePuddingPEQ');
    if(button){button.disabled=true;button.textContent='Optimizing…';}
    status('Preparing raw Pudding + target…');
    try{
      const raw={curve:parseCurveText(await readRepo('input/original_711/moondrop pudding fr.txt')),source:'input/original_711/moondrop pudding fr.txt'};
      const reg=selectedTarget();
      const target={curve:parseCurveText(await readRepo(reg.robustTargetPath)),source:reg.displayName,path:reg.robustTargetPath};
      status('IEM EarPrint constrained optimization…');
      const result=optimize(raw.curve,target.curve);
      const loudOn=$('loudnessMode')?.checked!==false;
      if(result.metrics?.loudnessCompensation){ result.metrics.loudnessCompensation.enabled=loudOn; if(!loudOn) result.metrics.loudnessCompensation.gainDb=0; }
      last=result;lastMeta={raw:raw.source,target:target.source,target_mode:'Robust Target',robust_target_path:target.path,loudness_mode:loudOn?'Moondrop Safe':'Off'};
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
      ['Loudness Match',result.metrics.loudnessCompensation?.enabled?'ON · +'+result.metrics.loudnessCompensation.gainDb.toFixed(2)+' dB':'OFF'],
      ['Max Q',result.metrics.maxQ.toFixed(2)],
      ['ERB error',(result.metrics.erbError??0).toFixed(3)+' dB'],
      ['Narrow error',(result.metrics.narrowError??0).toFixed(3)],
      ['Loss',result.metrics.lossMode||'standard'],
      ['Mode',result.metrics.performanceMode||CFG.performanceMode],
      ['Huber guard',result.metrics.huberCommitted===true?'PASS':(result.metrics.huberCommitted===false?'fallback':'—')],
      ['Level alignment',(result.metrics.levelOffsetDb>=0?'+':'')+result.metrics.levelOffsetDb.toFixed(2)+' dB removed'],
      ['Target Guard',result.metrics.shapeGuardAccepted===false?'ROLLBACK':'PASS'],
      ['Shape Δ',result.metrics.shapeGuardDeltaDb===null?'—':(result.metrics.shapeGuardDeltaDb>=0?'+':'')+result.metrics.shapeGuardDeltaDb.toFixed(4)+' dB']
    ].map(x=>`<div class="summary"><span class="k">${esc(x[0])}</span><span class="v">${esc(x[1])}</span></div>`).join('');

    table.innerHTML='<div class="pudding-peq-head"><span>Band</span><span>Type</span><span>Freq</span><span>Gain</span><span>Q</span></div>'+
      result.bands.map((b,i)=>`<div class="pudding-peq-row"><span>${i+1}</span><span>PK</span><span>${fmt(b.freq)} Hz</span><span>${b.gain>=0?'+':''}${b.gain.toFixed(2)} dB</span><span>${b.q.toFixed(2)}</span></div>`).join('')+
      '<div class="pudding-note">Evidence-driven active bands · joint LM Fc/Gain/Q refinement · Huber IRLS robustness · exact RBJ @ 48 kHz provisional · exact post-quantization validation · Robust Target only.</div>';
  }

  function downloadTxt(){
    if(!last){status('Generate the PEQ first.','warn');return;}
    download(formatPEQ(last),'Pudding_'+stem(lastMeta?.target||'RobustTarget')+'_EarPrintSolver_PEQ.txt');
  }

  function downloadJson(){
    if(!last){status('Generate the PEQ first.','warn');return;}
    download(jsonPEQ(last,lastMeta),'Pudding_'+stem(lastMeta?.target||'RobustTarget')+'_EarPrintSolver_PEQ.json','application/json;charset=utf-8');
  }

  function ensureUi(){
    if($('puddingEngine'))return;
    const anchor=$('infoPanel')||$('modal');
    const html=`<div class="card section" id="puddingEngine"><div class="visualizer-head"><div><h2 style="margin:0">IEM EarPrint PEQ Engine</h2><div class="visualizer-subtitle">Moondrop Pudding · Original 711 · Robust Target only</div></div><button type="button" id="puddingRefreshSources">Refresh targets</button></div>
    <div class="peq-source"><b>Device</b><span>Moondrop Pudding</span><b>Measurement</b><span>Original 711</span><b>Source</b><code>input/original_711/moondrop pudding fr.txt</code></div>
    <div class="field" style="margin-top:10px"><label for="puddingTargetSelect">Robust Target — PEQ Reference</label><select id="puddingTargetSelect"></select></div>
    <div class="inline" style="margin-top:10px"><div class="field"><label><input id="loudnessMode" type="checkbox" checked> Loudness level match (Moondrop Safe)</label><div class="pudding-note">Keeps perceived volume closer after PEQ. Max compensation +4 dB. ISO226 remains optional.</div></div><button type="button" class="primary" id="generatePuddingPEQ">Generate PEQ</button><button type="button" id="downloadPuddingPEQ">Download TXT</button><button type="button" id="downloadPuddingJSON">Download JSON</button></div>
    <div id="puddingStatus" class="status small" style="margin-top:8px"></div><div id="puddingResult" hidden style="margin-top:10px"><div id="puddingStats" class="summary-grid"></div><div id="puddingTable" class="pudding-table" style="margin-top:10px"></div></div>
    <details style="margin-top:12px"><summary>Import PEQ preset</summary><div class="field"><label>PK text preset (local only)</label><input id="importPeqFile" type="file" accept=".txt,text/plain"></div><div class="inline"><button id="validateImportedPeq" type="button">Validate</button><button id="downloadImportedPeq" type="button" disabled>Download unchanged</button></div><pre id="importPeqStatus" class="status small"></pre></details></div>`;
    if(anchor)anchor.insertAdjacentHTML('beforebegin',html);else document.body.insertAdjacentHTML('beforeend',html);
    const style=document.createElement('style');style.textContent=`#puddingEngine{margin-top:12px;box-shadow:0 10px 30px rgba(0,0,0,.15)}.peq-source{display:grid;grid-template-columns:auto 1fr;gap:5px 12px;font-size:12px}.peq-source code{overflow-wrap:anywhere}.pudding-table{border:1px solid var(--line);border-radius:6px;overflow:hidden}.pudding-peq-head,.pudding-peq-row{display:grid;grid-template-columns:.5fr .6fr 1.3fr 1.2fr 1fr;gap:8px;padding:7px 9px;align-items:center;font-size:12px;font-variant-numeric:tabular-nums}.pudding-peq-head{background:#0d141c;color:#7f8b99;border-bottom:1px solid var(--line);font-size:10px;text-transform:uppercase}.pudding-peq-row{border-bottom:1px solid #1b2530}.pudding-note{padding:8px 9px;color:#7f8b99;font-size:10px}`;document.head.appendChild(style);
  }
  let importedPeqText=null;
  function parseImportedPeq(text){const rows=[];for(const line of String(text).split(/\r?\n/)){const m=line.match(/PK\s+([\d.]+)\s*Hz\s*,?\s*([+-]?[\d.]+)\s*dB\s*,?\s*Q\s*([\d.]+)/i);if(m)rows.push({freq:+m[1],gain:+m[2],q:+m[3]});}if(!rows.length)throw Error('No valid PK filters found.');if(rows.length>10)throw Error('Imported preset has more than 10 filters.');rows.forEach((b,i)=>{if(b.freq<20||b.freq>12000||b.gain<-12||b.gain>3||b.q<.3||b.q>10||!biquadSafety(b).stable)throw Error('Filter '+(i+1)+' is outside device/stability limits.');});return rows;}
  async function validateImport(){const f=$('importPeqFile')?.files?.[0];if(!f)return;try{const txt=await f.text(),bands=parseImportedPeq(txt);importedPeqText=txt;$('importPeqStatus').textContent='VALID · '+bands.length+' PK filters · no optimizer/repository changes';$('downloadImportedPeq').disabled=false;}catch(e){importedPeqText=null;$('importPeqStatus').textContent='INVALID · '+e.message;$('downloadImportedPeq').disabled=true;}}

  function init(){
    ensureUi();
    if(!$('puddingEngine'))return;
    status('Pudding PEQ Engine v'+CFG.version);
    $('generatePuddingPEQ')?.addEventListener('click',generate);
    $('downloadPuddingPEQ')?.addEventListener('click',downloadTxt);
    $('downloadPuddingJSON')?.addEventListener('click',downloadJson);
    $('puddingRefreshSources')?.addEventListener('click',populate);
    $('validateImportedPeq')?.addEventListener('click',validateImport);
    $('downloadImportedPeq')?.addEventListener('click',()=>{if(importedPeqText)download(importedPeqText,'Imported_PEQ.txt');});
    populate();
  }

  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();

  window.MoondropPuddingPEQ={CFG,optimize,formatPEQ,
    __test:{DOMAIN_BANDS,resolutionGrids,featureAnalysis,boostRisk,deadZoneHuber,huberWeight,objectiveComponents,highFrequencyValidation,quantizeBands,responseAwarePrune,finalizeQuantized,responseDelta,rbj,totalEq,applyFilters,applyFiltersCached,erbRate,erbBandError,narrowFeatureError,topologyError,biquadSafety,modeledHeadroom,parseImportedPeq,solveLinearSystem,lmRefine,candidatePoolForState,stabilityPerturbationTest,hfSafetyRescue,quantizationRescue,optimizeLMIRLS}};
})();
