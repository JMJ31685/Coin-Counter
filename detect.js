/* Coin detection with OpenCV.js. Shared by worker.js and the Node test. */
(function (root) {
  'use strict';

  const WORK_MAX = 1000;   // longest side used for detection, in px
  const OVERLAP = 0.7;     // centers closer than 0.7*(r1+r2) count as the same coin
  const SIZE_TOL = 0.2;    // "same size" band around the typical radius
  const MIN_R = 0.015;     // smallest coin radius, as a fraction of the short side
  const MAX_R = 0.3;       // largest
  const CANNY = 70;        // edge threshold; higher misses silver coins on light backgrounds

  function hough(cv, gray, minR, maxR, minDist, param2) {
    const m = new cv.Mat();
    try {
      cv.HoughCircles(gray, m, cv.HOUGH_GRADIENT, 1.2, minDist, CANNY, param2, minR, maxR);
      const out = [];
      for (let i = 0; i < m.cols; i++) {
        out.push({ x: m.data32F[i * 3], y: m.data32F[i * 3 + 1], r: m.data32F[i * 3 + 2] });
      }
      return out; // strongest first
    } finally {
      m.delete();
    }
  }

  // Keep the stronger circle wherever two overlap.
  function suppress(circles) {
    const kept = [];
    for (const c of circles) {
      if (kept.every(k => Math.hypot(k.x - c.x, k.y - c.y) >= OVERLAP * (k.r + c.r))) kept.push(c);
    }
    return kept;
  }

  // Radius shared by the most circles (all coins in a photo are the same type).
  function typicalRadius(circles) {
    let best = null, bestN = -1;
    for (const a of circles) {
      const n = circles.filter(b => Math.abs(b.r - a.r) <= 0.15 * a.r).length;
      if (n > bestN) { bestN = n; best = a; }
    }
    const rs = circles.filter(b => Math.abs(b.r - best.r) <= 0.15 * best.r).map(b => b.r).sort((p, q) => p - q);
    return rs[rs.length >> 1];
  }

  /**
   * @param cv        initialized OpenCV.js module
   * @param rgba      cv.Mat, CV_8UC4
   * @param opts      { sensitivity: 1..10 (default 4), sameSize: bool, log: fn(stage, info) }
   * @returns         { circles: [{x, y, r}], radius } in rgba's pixel coordinates
   */
  function detectCoins(cv, rgba, opts) {
    opts = opts || {};
    const sensitivity = Math.min(10, Math.max(1, opts.sensitivity || 4));
    const sameSize = opts.sameSize !== false;
    const log = opts.log || function () {};
    const scale = Math.min(1, WORK_MAX / Math.max(rgba.cols, rgba.rows));
    const gray = new cv.Mat();
    const tmp = new cv.Mat();
    try {
      if (scale < 1) {
        cv.resize(rgba, tmp, new cv.Size(Math.round(rgba.cols * scale), Math.round(rgba.rows * scale)), 0, 0, cv.INTER_AREA);
        cv.cvtColor(tmp, gray, cv.COLOR_RGBA2GRAY);
      } else {
        cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
      }
      // Even out lighting and glare, then smooth away surface texture.
      const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
      clahe.apply(gray, tmp);
      clahe.delete();
      cv.GaussianBlur(tmp, gray, new cv.Size(9, 9), 2, 2);

      const minDim = Math.min(gray.cols, gray.rows);
      const param2 = Math.round(62 - sensitivity * 4); // accumulator threshold: 58 (strict) .. 22 (loose)

      // Pass 1: any plausible coin size.
      const minR = Math.max(6, Math.round(minDim * MIN_R));
      let circles = suppress(hough(cv, gray, minR, Math.round(minDim * MAX_R), minR * 1.5, param2));
      let radius = circles.length ? typicalRadius(circles) : 0;
      log('pass1', { minR, param2, n: circles.length, radius, rs: circles.map(c => Math.round(c.r)) });

      // Pass 2: search again tuned to the typical size, which finds touching coins pass 1 merged.
      if (radius && sameSize) {
        const inBand = c => Math.abs(c.r - radius) <= SIZE_TOL * radius;
        const second = hough(cv, gray, Math.round(radius * (1 - SIZE_TOL)), Math.round(radius * (1 + SIZE_TOL)),
          radius * 1.5, Math.max(15, Math.round(param2 * 0.8)));
        log('pass2', { n: second.length, inBand: second.filter(inBand).length, rs: second.map(c => Math.round(c.r)) });
        circles = suppress(circles.filter(inBand).concat(second.filter(inBand)));
        radius = circles.length ? typicalRadius(circles) : radius;
      }

      const s = 1 / scale;
      return {
        circles: circles.map(c => ({ x: c.x * s, y: c.y * s, r: c.r * s })),
        radius: radius * s,
      };
    } finally {
      gray.delete();
      tmp.delete();
    }
  }

  root.detectCoins = detectCoins;
  if (typeof module !== 'undefined' && module.exports) module.exports = { detectCoins };
})(typeof self !== 'undefined' ? self : this);
