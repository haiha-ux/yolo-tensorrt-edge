// Headless check of docs/: loads the page, runs the browser pipeline on docs/sample.jpg and matches the detections
// against web_check/ref.json (Ultralytics, same ONNX). Usage: node run.mjs <playwright-core> <chromium-headless-shell> [url] [png]
import { createRequire } from 'module';
import fs from 'fs';
const require = createRequire(import.meta.url);
const { chromium } = require(process.argv[2]);
const ref = JSON.parse(fs.readFileSync(new URL('./ref.json', import.meta.url)));
const b = await chromium.launch({ executablePath: process.argv[3], args: ['--no-sandbox'] });
const p = await b.newPage({ viewport: { width: 800, height: 900 } });
p.on('pageerror', (e) => console.log('PAGE ERROR', e.message));
await p.goto(process.argv[4] || 'http://localhost:8001/');
await p.waitForFunction(() => /inference/.test(document.getElementById('stats').textContent), null, { timeout: 120000 });
console.log(await p.textContent('#stats'));
const dets = await p.evaluate(async () => {
  const img = new Image(); img.src = 'sample.jpg'; await img.decode();
  const d = await YD.detect(img, img.naturalWidth, img.naturalHeight);
  const t = performance.now(); for (let i = 0; i < 5; i++) await YD.detect(img, img.naturalWidth, img.naturalHeight);
  return { d, ms: (performance.now() - t) / 5, backend: YD.backend };
});
const iou = (a, c) => { const w = Math.max(0, Math.min(a[2], c.x2) - Math.max(a[0], c.x1)), h = Math.max(0, Math.min(a[3], c.y2) - Math.max(a[1], c.y1)), i = w * h;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (c.x2 - c.x1) * (c.y2 - c.y1) - i); };
let matched = 0, minIou = 1, maxDs = 0;
for (const r of ref) {
  const m = dets.d.filter((d) => d.cls === r.cls).map((d) => [iou(r.box, d), d]).sort((a, c) => c[0] - a[0])[0];
  if (m && m[0] > 0.9) { matched++; minIou = Math.min(minIou, m[0]); maxDs = Math.max(maxDs, Math.abs(m[1].score - r.score)); }
}
console.log(`backend ${dets.backend}; JS detections ${dets.d.length}, reference ${ref.length}; matched (same class, IoU>0.9) ${matched}/${ref.length}; min IoU ${minIou.toFixed(3)}; max score diff ${maxDs.toFixed(3)}; avg full-frame ${dets.ms.toFixed(0)} ms`);
await p.screenshot({ path: process.argv[5] || 'shot.png' });
await b.close();
