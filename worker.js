/* Runs OpenCV.js off the main thread so the page stays responsive. */
self.importScripts('vendor/opencv.js', 'detect.js');

const ready = new Promise((resolve, reject) => {
  const mod = self.cv;
  if (!mod) return reject(new Error('OpenCV.js failed to load'));
  if (mod.Mat) return resolve(mod);
  if (typeof mod.then === 'function') {
    // The Emscripten module is thenable and resolves to itself; drop .then afterwards so it doesn't loop.
    mod.then(m => { delete m.then; resolve(m); });
  } else {
    mod.onRuntimeInitialized = () => resolve(mod);
  }
});

ready.then(
  () => self.postMessage({ type: 'ready' }),
  err => self.postMessage({ type: 'error', message: String(err && err.message || err) })
);

self.onmessage = async e => {
  const { id, width, height, buffer, sensitivity, sameSize } = e.data;
  try {
    const cv = await ready;
    const t0 = performance.now();
    const mat = cv.matFromImageData({ data: new Uint8ClampedArray(buffer), width, height });
    let res;
    try {
      res = self.detectCoins(cv, mat, { sensitivity, sameSize });
    } finally {
      mat.delete();
    }
    self.postMessage({ type: 'result', id, circles: res.circles, radius: res.radius, ms: Math.round(performance.now() - t0) });
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String(err && err.message || err) });
  }
};
