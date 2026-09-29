/* ISO 226:2003 normal equal-loudness contours.
 * The exported function returns a relative contour in dB. It is a weighting
 * layer only; it never changes a target curve or generates filters.
 */
(function (root) {
  'use strict';

  const FREQUENCIES = [20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500];
  const ALPHA = [0.532,0.506,0.480,0.455,0.432,0.409,0.387,0.367,0.349,0.330,0.315,0.301,0.288,0.276,0.267,0.259,0.253,0.250,0.246,0.244,0.243,0.243,0.243,0.242,0.242,0.245,0.254,0.271,0.301];
  const LU = [-31.6,-27.2,-23.0,-19.0,-15.8,-13.1,-10.8,-8.6,-6.6,-4.8,-3.2,-1.9,-0.8,0,0.6,1,1.2,0,-2.5,-4.7,-6.0,-6.4,-6.6,-6.7,-6.6,-6.1,-5.4,-4.3,-3.0];
  const TF = [78.5,68.7,59.5,51.1,44.0,37.5,31.5,26.5,22.1,17.9,14.4,11.4,8.6,6.2,4.4,3.0,2.2,2.4,3.5,1.7,-1.3,-4.2,-6.0,-5.7,-5.1,-1.3,1.0,-1.2,-2.5];

  function contourAt(frequency, phon) {
    if (!Number.isFinite(frequency) || frequency <= 0) return NaN;
    if (![40, 60, 80].includes(phon)) throw new Error('ISO226 phon must be 40, 60, or 80.');
    const x = Math.log(frequency);
    const values = FREQUENCIES.map((f, i) => {
      const af = 4.47e-3 * (Math.pow(10, 0.025 * phon) - 1.15)
        + Math.pow(0.4 * Math.pow(10, ((TF[i] + LU[i]) / 10) - 9), ALPHA[i]);
      return 10 / ALPHA[i] * Math.log10(af) - LU[i] + 94;
    });
    if (frequency <= FREQUENCIES[0]) return values[0];
    if (frequency >= FREQUENCIES.at(-1)) return values.at(-1);
    let i = 1;
    while (FREQUENCIES[i] < frequency) i++;
    const t = (x - Math.log(FREQUENCIES[i - 1])) / (Math.log(FREQUENCIES[i]) - Math.log(FREQUENCIES[i - 1]));
    return values[i - 1] + t * (values[i] - values[i - 1]);
  }

  function relativeContourDb(frequency, phon) {
    return contourAt(frequency, phon) - contourAt(1000, phon);
  }

  function optimizerWeight(frequency, phon, iemFactor) {
    const elc = relativeContourDb(frequency, phon) * iemFactor;
    return Math.pow(10, -elc / 20);
  }

  root.IEMEqualLoudness = Object.freeze({
    standard: 'ISO226',
    frequencies: FREQUENCIES.slice(),
    contourAt,
    relativeContourDb,
    optimizerWeight
  });
}(typeof window !== 'undefined' ? window : globalThis));
