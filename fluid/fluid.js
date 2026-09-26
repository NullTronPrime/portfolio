(function () {
  var script = document.getElementById('fluid-loader');
  var base = (script && script.dataset.src) || 'fluid/fluid.wasm';
  var ex = null;
  var canvas = null;
  var ctx = null;
  var img = null;
  var frameTimer = null;
  var lastStats = 0;
  var statsCount = 0;
  var statsEl = null;

  var KEYMAP = {
    '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
    'p': 7, 'm': 8, 'o': 9, 's': 10, 'v': 11, 'r': 12, 'n': 13,
    'ArrowUp': 14, 'ArrowDown': 15, 'f': 16, 'b': 17, 'w': 18
  };

  var FLUID_LABELS = ['Tank', 'Wind Tunnel', 'Paint', 'Hi-Res Tunnel'];
  var MODE_LABELS = ['Fluid','2D Fire','3D Fire'];

  var downKeys = new Set();
  var prevKeys = new Set();
  var pending = 0;
  var mX = 0, mY = 0, lDown = false, rDown = false;
  var scrollAcc = 0;

  function norm(k) { return (k.length === 1 ? k.toLowerCase() : k); }

  function codeFor(k) { return KEYMAP[norm(k)]; }

  window.addEventListener('keydown', function (e) {
    var b = codeFor(e.key);
    if (b !== undefined) downKeys.add(norm(e.key));
  });
  window.addEventListener('keyup', function (e) {
    downKeys.delete(norm(e.key));
  });

  function updateStats() {
    if (!ex || !statsEl) return;
    var paused = ex.fluid_paused() ? 1 : 0;
    var mode = ex.fluid_mode();
    var scene = ex.fluid_scene();
    var g = ex.fluid_grid();
    var label = mode === 0 ? 'Fluid Sim · ' + FLUID_LABELS[scene]
      : (mode === 1 ? '2D Fire' : '3D Fire');
    var pausedTxt = paused ? ' · Paused' : '';
    var grid = (g & 65535) + 'x' + (g >>> 16);
    var fps = statsCount;
    statsCount = 0;
    statsEl.textContent = label + ' · ' + grid + ' · ' + fps + ' FPS' + pausedTxt;
  }

  function loop() {
    if (!ex) return;
    var pressed = 0, down = 0;
    for (var k in KEYMAP) {
      if (downKeys.has(k)) {
        down |= 1 << KEYMAP[k];
        if (!prevKeys.has(k)) pressed |= 1 << KEYMAP[k];
      }
    }
    pressed |= pending;
    pending = 0;
    prevKeys = new Set(downKeys);

    var s = 0;
    if (scrollAcc !== 0) { s = scrollAcc; scrollAcc = 0; }

    ex.fluid_frame(pressed >>> 0, down >>> 0, mX, mY, lDown ? 1 : 0, rDown ? 1 : 0, s);

    var fb = new Uint8ClampedArray(ex.memory.buffer, ex.fluid_fbuf_ptr(), ex.fluid_fbuf_len());
    img.data.set(fb);
    ctx.putImageData(img, 0, 0);

    statsCount += 1;
    var now = Date.now();
    if (now - lastStats >= 1000 && frameTimer) {
      lastStats = now;
      updateStats();
    }
  }

  function pointerPos(e) {
    var r = canvas.getBoundingClientRect();
    mX = (e.clientX - r.left) * canvas.width / r.width;
    mY = (e.clientY - r.top) * canvas.height / r.height;
  }

  function init() {
    canvas = document.getElementById('fluid-canvas');
    if (!canvas) return;
    statsEl = document.getElementById('fluid-stats');
    ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    canvas.addEventListener('pointerdown', function (e) {
      e.preventDefault();
      pointerPos(e);
      if (e.button === 0) lDown = true;
      if (e.button === 2) rDown = true;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', function (e) { pointerPos(e); });
    canvas.addEventListener('pointerup', function (e) {
      if (e.button === 0) lDown = false;
      if (e.button === 2) rDown = false;
    });
    canvas.addEventListener('pointercancel', function () { lDown = false; rDown = false; });
    canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      scrollAcc += e.deltaY / 120;
    }, { passive: false });

    window.FluidSim = {
      press: function (k) { var b = codeFor(k); if (b !== undefined) pending |= 1 << b; },
      running: function () { return !!frameTimer; },
      start: start,
      stop: stop
    };

    var btns = document.querySelectorAll('.fluid-btn');
    for (var i = 0; i < btns.length; i++) {
      (function (b) {
        b.addEventListener('click', function () { window.FluidSim.press(b.getAttribute('data-key')); });
      })(btns[i]);
    }
  }

  function start() {
    if (frameTimer) return;
    lastStats = Date.now();
    statsCount = 0;
    loop();
    frameTimer = setInterval(loop, 33);
  }

  function stop() {
    if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
  }

  document.addEventListener('DOMContentLoaded', function () {
    fetch(base).then(function (r) { return r.arrayBuffer(); })
      .then(function (buf) { return WebAssembly.instantiate(buf, {}); })
      .then(function (res) {
        ex = res.instance.exports;
        ex.fluid_init();
        init();
        if (canvas) {
          canvas.width = ex.fluid_win_w();
          canvas.height = ex.fluid_win_h();
          img = ctx.createImageData(canvas.width, canvas.height);
          start();
        }
      })
      .catch(function (e) {
        var el = document.getElementById('fluid-stats');
        if (el) el.textContent = 'Fluid sim failed to load: ' + e.message;
        if (typeof console !== 'undefined') console.error('fluid wasm error', e);
      });
  });
})();