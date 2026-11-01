// node web_check/nms_test.js : unit check of the JS NMS / decode on synthetic data
const D = require('../docs/detect.js'), assert = require('assert');
const b = (x1, y1, x2, y2, score, cls) => ({ x1, y1, x2, y2, score, cls });
assert(Math.abs(D.iou(b(0, 0, 10, 10), b(5, 0, 15, 10)) - 1 / 3) < 1e-6);
let k = D.nms([b(0, 0, 10, 10, .9, 0), b(1, 1, 11, 11, .8, 0), b(1, 1, 11, 11, .7, 1), b(50, 50, 60, 60, .6, 0)], 0.45);
assert.deepStrictEqual(k.map((d) => d.score), [.9, .7, .6]); // overlap suppressed within a class only
// decode: 2 anchors, 80 classes; anchor 0 = class 5 score .8 at centre (320,320) 100x100; anchor 1 below threshold
const n = 2, out = new Float32Array(84 * n);
[320, 100, 100, 100].forEach((v, r) => { out[r * n] = v; }); out[n + 0] = 320; out[1 * n + 1] = 0;
out[(4 + 5) * n] = 0.8; out[(4 + 7) * n + 1] = 0.1;
const lb = D.letterbox(1280, 960); // scale .5, padY 80
assert.deepStrictEqual([lb.scale, lb.padX, lb.padY], [0.5, 0, 80]);
const r = D.decode(out, n, lb, 1280, 960);
assert.strictEqual(r.length, 1); assert.strictEqual(r[0].cls, 5);
assert.deepStrictEqual([r[0].x1, r[0].y1, r[0].x2, r[0].y2], [540, 380, 740, 580]);
console.log('nms/decode unit checks passed');
