// YOLO11 pre/post-processing: letterbox geometry, decoding of the raw [1,84,8400] output, per-class NMS.
(function (root) {
  const NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

  // scale and padding that fit a w x h image into size x size, centred (same as Ultralytics LetterBox, auto=False)
  function letterbox(w, h, size = 640) {
    const scale = Math.min(size / w, size / h);
    const nw = Math.round(w * scale), nh = Math.round(h * scale);
    return { scale, nw, nh, padX: Math.floor((size - nw) / 2), padY: Math.floor((size - nh) / 2) };
  }

  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
  }

  // greedy per-class NMS on boxes sorted by score
  function nms(dets, iouThr) {
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) {
      let ok = true;
      for (const k of keep) if (k.cls === d.cls && iou(k, d) > iouThr) { ok = false; break; }
      if (ok) keep.push(d);
    }
    return keep;
  }

  // out: Float32Array [84 * n] (channel-major: cx,cy,w,h then 80 class scores); returns boxes in original image pixels
  function decode(out, n, lb, imgW, imgH, conf = 0.25, iouThr = 0.45, maxDet = 100) {
    const nc = out.length / n - 4, cand = [];
    for (let i = 0; i < n; i++) {
      let best = 0, bc = -1;
      for (let c = 0; c < nc; c++) { const s = out[(4 + c) * n + i]; if (s > best) { best = s; bc = c; } }
      if (best < conf) continue;
      const cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i];
      const cl = (v, m) => Math.min(Math.max(v, 0), m);
      cand.push({ x1: cl((cx - w / 2 - lb.padX) / lb.scale, imgW), y1: cl((cy - h / 2 - lb.padY) / lb.scale, imgH),
                  x2: cl((cx + w / 2 - lb.padX) / lb.scale, imgW), y2: cl((cy + h / 2 - lb.padY) / lb.scale, imgH), score: best, cls: bc });
    }
    return nms(cand, iouThr).slice(0, maxDet);
  }

  const api = { NAMES, letterbox, iou, nms, decode };
  if (typeof module !== 'undefined') module.exports = api; else root.Det = api;
})(this);
