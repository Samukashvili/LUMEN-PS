// Interactive relight — WebGL2. Samples the recovered normal + albedo + alpha
// on a flat plane and re-shades albedo * (ambient + max(N·L,0)) as you drag the
// light. This is the project's thesis made tangible.

const VERT = `#version 300 es
in vec2 p; out vec2 vUv;
void main(){ vUv = p*0.5+0.5; gl_Position = vec4(p,0.,1.); }`;

const FRAG = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uN, uA, uAlpha;
uniform vec2 uFit;          // aspect 'contain' correction
uniform vec3 uL;            // light dir (normalized)
uniform float uAmbient, uDX, uBackdrop;
void main(){
  vec2 uv = (vUv-0.5)*uFit + 0.5;
  if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0){ o=vec4(0.02,0.02,0.02,1.0); return; }
  // backdrop
  vec3 bg;
  if(uBackdrop>0.5){
    vec2 c = floor(vUv*vec2(34.0,20.0));
    float k = mod(c.x+c.y,2.0);
    bg = mix(vec3(0.14,0.13,0.11), vec3(0.09,0.08,0.07), k);
  } else { bg = vec3(0.04,0.035,0.03); }
  vec3 n = texture(uN, uv).rgb*2.0-1.0;
  if(uDX>0.5) n.y = -n.y;
  n = normalize(n);
  vec3 alb = texture(uA, uv).rgb;
  float a = texture(uAlpha, uv).r;
  float d = max(dot(n, uL), 0.0);
  vec3 lit = alb*(uAmbient + d);
  o = vec4(mix(bg, lit, a), 1.0);
}`;

function sh(gl, type, src){
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class Relight{
  constructor(canvas){
    this.canvas = canvas;
    this.imgAspect = 1;
    this.state = { el: 35, ambient: 0.12, dx: false, backdrop: 1, lx: 0.0, ly: 0.6 };
    this._dirty = true;
    this._tex = {};
    this._imgs = null;
    this._running = true;
    this._raf = null;

    // Survive a lost/restored GL context (GPU switch, driver reset, tab sleep)
    // instead of leaving a permanently black canvas.
    this._onLost = e => { e.preventDefault(); if(this._raf){ cancelAnimationFrame(this._raf); this._raf = null; } };
    this._onRestored = () => {
      this._initGL(); this._uploadTextures(); this._dirty = true;
      if(this._running && this._raf === null) this._loop();
    };
    canvas.addEventListener('webglcontextlost', this._onLost, false);
    canvas.addEventListener('webglcontextrestored', this._onRestored, false);

    this._initGL();

    this._onMove = e => this._pointer(e);
    this._onDown = e => { canvas.setPointerCapture(e.pointerId); this._pointer(e); };
    canvas.addEventListener('pointermove', this._onMove);
    canvas.addEventListener('pointerdown', this._onDown);
    this._loop();
  }

  _initGL(){
    const gl = this.canvas.getContext('webgl2', {antialias:true, premultipliedAlpha:false});
    if(!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog; gl.useProgram(prog);
    this._buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.u = {};
    ['uN','uA','uAlpha','uFit','uL','uAmbient','uDX','uBackdrop'].forEach(k =>
      this.u[k] = gl.getUniformLocation(prog, k));
    this._tex = {};
  }

  async load(nURL, aURL, alphaURL){
    const [n, a, al] = await Promise.all([this._img(nURL), this._img(aURL), this._img(alphaURL)]);
    this._imgs = { uN: n, uA: a, uAlpha: al };
    this.imgAspect = n.naturalWidth / n.naturalHeight;
    this._uploadTextures();
    this._dirty = true;
  }

  _uploadTextures(){
    if(!this.gl || !this._imgs) return;
    this._deleteTextures();
    this._tex.uN = this._texture(this._imgs.uN, 0, 'uN');
    this._tex.uA = this._texture(this._imgs.uA, 1, 'uA');
    this._tex.uAlpha = this._texture(this._imgs.uAlpha, 2, 'uAlpha');
  }

  _deleteTextures(){
    if(!this.gl) return;
    for(const t of Object.values(this._tex)) if(t) this.gl.deleteTexture(t);
    this._tex = {};
  }

  _img(url){ return new Promise((res, rej) => {
    const im = new Image(); im.crossOrigin='anonymous';
    im.onload = () => res(im); im.onerror = () => rej(new Error('img '+url)); im.src = url; }); }

  _texture(img, unit, uni){
    const gl = this.gl; const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(this.u[uni], unit);
    return t;
  }

  _pointer(e){
    const r = this.canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * 2 - 1;
    const y = -((e.clientY - r.top) / r.height * 2 - 1);
    const m = Math.hypot(x, y) || 1;
    this.state.lx = x / m; this.state.ly = y / m;   // azimuth from pointer
    this._dirty = true;
  }

  set(k, v){ this.state[k] = v; this._dirty = true; }

  _resize(){
    const gl = this.gl, c = this.canvas, dpr = Math.min(devicePixelRatio||1, 2);
    const w = Math.round(c.clientWidth*dpr), h = Math.round(c.clientHeight*dpr);
    if(c.width!==w||c.height!==h){ c.width=w; c.height=h; gl.viewport(0,0,w,h); this._dirty=true; }
  }

  _loop(){
    if(!this._running){ this._raf = null; return; }
    this._raf = requestAnimationFrame(() => this._loop());
    const gl = this.gl;
    if(gl.isContextLost()) return;
    this._resize();
    if(!this._dirty || !this._tex.uN) return;
    this._dirty = false;
    const s = this.state;
    const ca = this.canvas.width/this.canvas.height, ia = this.imgAspect;
    const fit = ca > ia ? [ca/ia, 1] : [1, ia/ca];
    const el = s.el*Math.PI/180, ch = Math.cos(el);
    gl.uniform2f(this.u.uFit, fit[0], fit[1]);
    gl.uniform3f(this.u.uL, s.lx*ch, s.ly*ch, Math.sin(el));
    gl.uniform1f(this.u.uAmbient, s.ambient);
    gl.uniform1f(this.u.uDX, s.dx?1:0);
    gl.uniform1f(this.u.uBackdrop, s.backdrop);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Stop the render loop and release GL + DOM resources. Call before dropping a
  // viewer (e.g. leaving the results stage) so instances don't pile up — each
  // one holds a live WebGL2 context, and browsers only allow a handful.
  dispose(){
    this._running = false;
    if(this._raf){ cancelAnimationFrame(this._raf); this._raf = null; }
    this.canvas.removeEventListener('pointermove', this._onMove);
    this.canvas.removeEventListener('pointerdown', this._onDown);
    this.canvas.removeEventListener('webglcontextlost', this._onLost);
    this.canvas.removeEventListener('webglcontextrestored', this._onRestored);
    const gl = this.gl;
    if(gl && !gl.isContextLost()){
      this._deleteTextures();
      if(this.prog) gl.deleteProgram(this.prog);
      if(this._buf) gl.deleteBuffer(this._buf);
      const ext = gl.getExtension('WEBGL_lose_context');
      if(ext) ext.loseContext();
    }
    this._imgs = null;
  }
}
