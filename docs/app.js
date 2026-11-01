(function () {
  const SIZE = 640, $ = (id) => document.getElementById(id);
  const cv = $('cv0'), ctx = cv.getContext('2d'), vid = $('vid');
  const pre = document.createElement('canvas'); pre.width = pre.height = SIZE;
  const pctx = pre.getContext('2d', { willReadFrequently: true });
  let session, backend = '?', src = null, stream = null, running = false;

  async function load() {
    const opts = { graphOptimizationLevel: 'all' };
    const tries = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    for (const ep of tries) {
      try {
        if (ep === 'wasm') ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
        session = await ort.InferenceSession.create('yolo11n.onnx', { ...opts, executionProviders: [ep] });
        backend = ep; return;
      } catch (e) { console.warn(ep + ' failed:', e); }
    }
    throw new Error('no backend could load the model');
  }

  // run one frame (image, canvas or video element); draws boxes and returns timings
  async function detect(el, w, h) {
    const t0 = performance.now();
    const lb = Det.letterbox(w, h, SIZE);
    pctx.fillStyle = 'rgb(114,114,114)'; pctx.fillRect(0, 0, SIZE, SIZE);
    pctx.drawImage(el, 0, 0, w, h, lb.padX, lb.padY, lb.nw, lb.nh);
    const px = pctx.getImageData(0, 0, SIZE, SIZE).data, x = new Float32Array(3 * SIZE * SIZE), n = SIZE * SIZE;
    for (let i = 0; i < n; i++) { x[i] = px[4 * i] / 255; x[n + i] = px[4 * i + 1] / 255; x[2 * n + i] = px[4 * i + 2] / 255; }
    const t1 = performance.now();
    const out = (await session.run({ images: new ort.Tensor('float32', x, [1, 3, SIZE, SIZE]) })).output0;
    const data = out.data, t2 = performance.now();
    const dets = Det.decode(data, out.dims[2], lb, w, h, +$('conf').value, 0.45);
    const t3 = performance.now();
    cv.width = w; cv.height = h; ctx.drawImage(el, 0, 0, w, h);
    ctx.lineWidth = Math.max(2, w / 300); ctx.font = Math.max(12, w / 45) + 'px sans-serif';
    for (const d of dets) {
      const hue = (d.cls * 47) % 360, col = 'hsl(' + hue + ',85%,50%)', label = Det.NAMES[d.cls] + ' ' + d.score.toFixed(2);
      ctx.strokeStyle = col; ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      const tw = ctx.measureText(label).width + 6, th = parseInt(ctx.font) + 4;
      ctx.fillStyle = col; ctx.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
      ctx.fillStyle = '#fff'; ctx.fillText(label, d.x1 + 3, Math.max(th - 4, d.y1 - 4));
    }
    $('stats').textContent = 'backend: ' + backend + ' | inference: ' + (t2 - t1).toFixed(1) + ' ms | pre ' + (t1 - t0).toFixed(1) +
      ' ms, NMS ' + (t3 - t2).toFixed(1) + ' ms | ' + dets.length + ' detections';
    return dets;
  }

  function stopCam() {
    running = false;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; $('cam').textContent = 'Start webcam';
  }

  async function showImage(url) {
    stopCam();
    const img = new Image(); img.src = url; await img.decode(); src = img;
    await detect(img, img.naturalWidth, img.naturalHeight);
  }

  async function loop() {
    while (running) {
      if (vid.videoWidth) await detect(vid, vid.videoWidth, vid.videoHeight);
      await new Promise(requestAnimationFrame);
    }
  }

  $('file').onchange = (e) => { const f = e.target.files[0]; if (f) showImage(URL.createObjectURL(f)); };
  $('sample').onclick = () => showImage('sample.jpg');
  $('conf').oninput = () => { $('cv').textContent = (+$('conf').value).toFixed(2); if (src && !running) detect(src, src.naturalWidth, src.naturalHeight); };
  $('cam').onclick = async () => {
    if (running) return stopCam();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 640 } } });
    } catch (e) { $('stats').textContent = 'Camera not available: ' + e.message; return; }
    vid.srcObject = stream; await vid.play(); running = true; $('cam').textContent = 'Stop webcam'; loop();
  };

  window.YD = { detect, get backend() { return backend; } };
  load().then(() => showImage('sample.jpg')).catch((e) => { $('stats').textContent = 'Failed to load model: ' + e.message; });
})();
