import p5 from 'p5';
import * as Tone from 'tone';
import {start} from "tone";

const NUM_PARTICLES = 10000;

const sketch = (p) => {

    // tone vars
    let osc1, osc2, osc3, osc4, osc5, gainNode, audioStarted = false;
    // --- ml5 vars ---
    let video, faceMesh, faces = [];
    let options = { maxFaces: 1, refineLandmarks: true, flipped: false };
    let vx, vy, vw, vh;

    // --- WebGPU vars ---
    let device, particleBuffer, uniformBuffer;
    let computePipeline, renderPipeline;
    let computeBindGroup, renderBindGroup;
    let gpuCanvas, gpuContext;

    function calcVideoRect() {
        const aspect = 16 / 9;
        if (p.width / p.height > aspect) {
            vh = p.height; vw = vh * aspect;
        } else {
            vw = p.width; vh = vw / aspect;
        }
        vx = (p.width - vw) / 2;
        vy = (p.height - vh) / 2;
    }

    function gotFaces(results) { faces = results; }

    async function startAudio() {
        await Tone.start();

        const reverb  = new Tone.Reverb({ decay: 20, wet: 0.9 }).toDestination();
        const chorus  = new Tone.Chorus(0.5, 3.5, 0.2).connect(reverb).start();
        gainNode = new Tone.Gain(0).connect(chorus);

        osc1 = new Tone.Oscillator(220,   'sine').connect(gainNode);
        osc2 = new Tone.Oscillator(275,   'sine').connect(gainNode);
        osc3 = new Tone.Oscillator(330,   'sine').connect(gainNode);
        osc4 = new Tone.Oscillator(440,   'sine').connect(gainNode);
        osc5 = new Tone.Oscillator(110,   'sine').connect(gainNode);

        osc1.start(); osc2.start(); osc3.start(); osc4.start(); osc5.start();
        audioStarted = true;
    }



    function keypointToCanvas(kp, scaleX, scaleY) {
        let mx = kp.x * scaleX;
        if (options.flipped) mx = vw - mx;
        const my = kp.y * scaleY;
        return { x: vx + mx, y: vy + my };
    }

    // ─── WebGPU init ────────────────────────────────────────────────────────────
    async function initWebGPU() {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { console.error('WebGPU not supported'); return; }
        device = await adapter.requestDevice();

        // Particle buffer: each particle = 8 floats
        // [posX, posY, velX, velY, lifetime, maxLifetime, sourceEye(0|1), padding]
        const particleData = new Float32Array(NUM_PARTICLES * 8);
        for (let i = 0; i < NUM_PARTICLES; i++) {
            const base = i * 8;
            const isMouth = i % 3 === 2;
            const maxLife = isMouth
                ? Math.random() * 4 + 3   // mouth: 3–7 seconds
                : Math.random() * 3 + 1;  // eyes:  1–4 seconds
            particleData[base + 4] = -Math.random() * maxLife; // staggered start
            particleData[base + 5] = maxLife;
            particleData[base + 6] = i % 3;
        }

        particleBuffer = device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(particleBuffer.getMappedRange()).set(particleData);
        particleBuffer.unmap();

        // Uniform buffer: [leftEyeX, leftEyeY, rightEyeX, rightEyeY, canvasW, canvasH, time, deltaTime]
        uniformBuffer = device.createBuffer({
            size: 12 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        await createComputePipeline();
        await createRenderPipeline();
    }

    // ─── Compute shader ──────────────────────────────────────────────────────────
    async function createComputePipeline() {
        const wgsl = /* wgsl */`
      struct Particle {
        pos     : vec2f,
        vel     : vec2f,
        life    : f32,
        maxLife : f32,
        eye     : f32,
        pad     : f32,
      };

      struct Uniforms {
        leftEye   : vec2f,
        rightEye  : vec2f,
        mouth     : vec2f,
        canvasSize: vec2f,
        time      : f32,
        dt        : f32,
        mouthOpen : f32,
        _pad      : f32,
      };

      @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
      @group(0) @binding(1) var<uniform> u: Uniforms;

      fn rand(seed: f32) -> f32 {
        return fract(sin(seed * 127.1 + u.time * 311.7) * 43758.5453);
      }

      fn hash2(p: vec2f) -> f32 {
        var q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
        return fract(sin(q.x + q.y) * 43758.5453);
      }

      fn smoothNoise(p: vec2f) -> f32 {
        let i  = floor(p);
        let f  = fract(p);
        let u2 = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(hash2(i + vec2f(0.0, 0.0)), hash2(i + vec2f(1.0, 0.0)), u2.x),
          mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), u2.x),
          u2.y
        );
      }

      fn noiseForce(pos: vec2f, t: f32) -> vec2f {
        let scale = 0.002;
        let speed = 0.3;
        let p2    = pos * scale;
        let n1    = smoothNoise(p2 + vec2f(t * speed, 0.0));
        let n2    = smoothNoise(p2 + vec2f(0.0, t * speed) + vec2f(5.2, 1.3));
        return (vec2f(n1, n2) - 0.5) * 2.0;
      }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= ${NUM_PARTICLES}u) { return; }

  var p = particles[i];

  if (p.life >= 0.0) {
    // ── alive: simulate ──
    let noiseStr = select(300.0, 150.0, p.eye > 1.5); // eyes stronger, mouth gentler
    let nf       = noiseForce(p.pos, u.time);
    let force    = nf * noiseStr;
    p.pad    = clamp(length(nf), 0.0, 1.0); // store noise magnitude for render shader
    p.vel.y += 80.0 * u.dt;
    p.vel   += force * u.dt;
    p.vel   *= pow(0.995, u.dt * 60.0);
    p.pos   += p.vel * u.dt;
    p.life  -= u.dt;

  } else {
    // ── dead/waiting: tick toward 0 ──
    p.life += u.dt;

    if (p.life >= 0.0) {
      // ── time to spawn ──
        let emitPos = select(
          select(u.leftEye, u.rightEye, p.eye > 0.5),
          u.mouth,
          p.eye > 1.5
        );
        let isEye = p.eye < 1.5;
        
        if (isEye && u.mouthOpen < 0.5) {
            p.life = -0.01; // mouth closed, suppress eye particles
        } else if (length(emitPos) < 1.0) {
            p.life = -0.01; // emitter inactive, retry next frame
        } else {
        let seed = f32(i) + u.time * 100.0;
        let isMouthParticle = p.eye > 1.5;
        let angle = select(
          rand(seed) * 6.2831853,
          rand(seed) * 1.0471 + 1.0472,
          isMouthParticle
        );
        let baseSpeed = select(
          rand(seed + 1.0) * 150.0 + 50.0,
          rand(seed + 1.0) * 100.0 + 80.0,
          isMouthParticle
        );
        let isFast  = (i % 10u) == 0u;
        let speed   = select(baseSpeed, baseSpeed * 2.0, isFast);
        p.pos = select(
          emitPos + vec2f((rand(seed + 2.0) - 0.5) * 10.0,  (rand(seed + 3.0) - 0.5) * 10.0),
          emitPos + vec2f((rand(seed + 2.0) - 0.5) * 70.0,  (rand(seed + 3.0) - 0.5) * 30.0),
          isMouthParticle
        );
        p.vel  = vec2f(cos(angle), sin(angle)) * speed;
        p.life = p.maxLife;
      }
    }
  }

  particles[i] = p;
}
      
    `;

        const module = device.createShaderModule({ code: wgsl });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' }
        });

        computeBindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } },
            ]
        });
    }

    // ─── Render shader ───────────────────────────────────────────────────────────
    async function createRenderPipeline() {
        // Create a dedicated WebGPU canvas — do NOT reuse p5's canvas
        gpuCanvas = document.createElement('canvas');
        gpuCanvas.width  = p.width;
        gpuCanvas.height = p.height;
        gpuCanvas.style.position = 'absolute';
        gpuCanvas.style.top = '0';
        gpuCanvas.style.left = '0';
        gpuCanvas.style.pointerEvents = 'none';
        document.getElementById('sketch-container').appendChild(gpuCanvas);

        gpuContext = gpuCanvas.getContext('webgpu');
        if (!gpuContext) { console.error('Failed to get WebGPU context'); return; }

        const format = navigator.gpu.getPreferredCanvasFormat();
        gpuContext.configure({ device, format, alphaMode: 'premultiplied' });

        const wgsl = /* wgsl */`
      struct Particle {
        pos     : vec2f,
        vel     : vec2f,
        life    : f32,
        maxLife : f32,
        eye     : f32,
        pad     : f32,
      };

      struct Uniforms {
        leftEye   : vec2f,
        rightEye  : vec2f,
        mouth     : vec2f,
        canvasSize: vec2f,
        time      : f32,
        dt        : f32,
        mouthOpen : f32,
        _pad      : f32,
      };

      @group(0) @binding(0) var<storage, read> particles: array<Particle>;
      @group(0) @binding(1) var<uniform> u: Uniforms;

      struct VSOut {
        @builtin(position) pos : vec4f,
        @location(0)       col : vec4f,
      };

        @vertex
        fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
          let particleIdx = vi / 6u;
          let cornerIdx   = vi % 6u;
        
          let p = particles[particleIdx];
          var out: VSOut;
        
          if (p.life < 0.0) {
            out.pos = vec4f(10.0, 10.0, 0.0, 1.0);
            out.col = vec4f(0.0);
            return out;
          }
        
          let t    = clamp(p.life / p.maxLife, 0.0, 1.0);
          let size = t * 4.0; // starts at 12px, shrinks to 0
        
          // Quad corners in pixel offsets (two triangles)
          var corners = array<vec2f, 6>(
            vec2f(-1.0, -1.0),
            vec2f( 1.0, -1.0),
            vec2f(-1.0,  1.0),
            vec2f(-1.0,  1.0),
            vec2f( 1.0, -1.0),
            vec2f( 1.0,  1.0),
          );
        
          let offset  = corners[cornerIdx] * size;
          let pixPos  = p.pos + offset;
          let ndc     = (pixPos / u.canvasSize) * 2.0 - 1.0;
          out.pos     = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
        
          let isMouth = p.eye > 1.5;

          // per-particle variation: a stable random value 0..1 based on index
          let variation = fract(sin(f32(particleIdx) * 127.1 + 43.0) * 43758.5453);

          // green base with slight hue shift toward cyan or yellow
          let eyeBase = mix(
            vec3f(0.0, 1.0, 0.0),                              // pure green
            select(
              vec3f(0.0, 1.0, 1.0),                            // green→teal
              vec3f(1.0, 1.0, 0.0),                            // green→lime
              variation > 0.5
            ),
            variation * 0.5                                     // max 50% shift
          );

          // blue base with slight hue shift toward cyan or violet
          let mouthBase = mix(
            vec3f(0.0, 0.0, 1.0),                              // pure blue
            select(
              vec3f(0.0, 0.3, 1.0),                            // blue→cyan
              vec3f(0.3, 0.0, 1.0),                            // blue→violet
              variation > 0.5
            ),
            variation * 0.5
          );

          let spawnCol = select(eyeBase, mouthBase, isMouth);
          let col = mix(vec3f(1.0, 0.0, 0.0), spawnCol, t);   // fade to red as t→0

          // mix in yellow based on noise magnitude stored in p.pad
          let yellow = vec3f(1.0, 1.0, 0.0);
          let finalCol = mix(col, yellow, p.pad * 0.6);
          out.col = vec4f(finalCol * t, t);
          return out;
        }


      @fragment
      fn fs(@location(0) col: vec4f) -> @location(0) vec4f {
        return col;
      }
    `;

        const module = device.createShaderModule({ code: wgsl });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            ]
        });

        renderPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex:   { module, entryPoint: 'vs' },
            fragment: {
                module, entryPoint: 'fs',
                targets: [{
                    format,blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    }
                }]
            },primitive: { topology: 'triangle-list' },
        });

        renderBindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } },
            ]
        });
    }

    // ─── Per-frame GPU dispatch ──────────────────────────────────────────────────
    let lastTime = 0;
    let ratio = 0;  // ← declare before the face block
    function dispatchGPU(timestamp) {
        if (!device || !computePipeline || !renderPipeline || !gpuContext) return;
        let mouthX = 0, mouthY = 0;

        const dt = Math.min((timestamp - lastTime) / 1000, 0.05);
        lastTime = timestamp;

        let leftEyeX = 0, leftEyeY = 0, rightEyeX = 0, rightEyeY = 0;

        if (faces.length > 0) {
            const face = faces[0];
            const srcW = (video.elt?.videoWidth) || 640;
            const srcH = (video.elt?.videoHeight) || 480;
            const scaleX = vw / srcW;
            const scaleY = vh / srcH;

            const leftKP = face.keypoints[468] ?? face.keypoints[159];
            const rightKP = face.keypoints[473] ?? face.keypoints[386];

            if (leftKP) {
                const lc = keypointToCanvas(leftKP, scaleX, scaleY);
                leftEyeX = lc.x;
                leftEyeY = lc.y;
            }
            if (rightKP) {
                const rc = keypointToCanvas(rightKP, scaleX, scaleY);
                rightEyeX = rc.x;
                rightEyeY = rc.y;
            }

            const upperLip = face.keypoints[13];
            const lowerLip = face.keypoints[14];
            const lipLeft = face.keypoints[61];
            const lipRight = face.keypoints[291];

            if (upperLip && lowerLip && lipLeft && lipRight) {
                const ul = keypointToCanvas(upperLip, scaleX, scaleY);
                const ll = keypointToCanvas(lowerLip, scaleX, scaleY);
                const openness = Math.abs(ll.y - ul.y);
                const mouthWidthKP  = keypointToCanvas(lipLeft,  scaleX, scaleY);
                const mouthWidthKP2 = keypointToCanvas(lipRight, scaleX, scaleY);
                const mouthWidth = Math.abs(mouthWidthKP2.x - mouthWidthKP.x);

                ratio = openness / mouthWidth; // 0 = closed, ~1 = wide open

                if (audioStarted) {
                    // smoothly ramp gain: 0 when closed, 1 when fully open
                    const targetGain = ratio > 0.2 ? Math.min(ratio, 1.0) / 4 : 0;
                    gainNode.gain.rampTo(targetGain, 0.05); // 50ms smooth
                }

                if (ratio > 0.2) {
                    mouthX = (ul.x + ll.x) / 2;
                    mouthY = (ul.y + ll.y) / 2;
                }
            }

        }

        device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
            leftEyeX, leftEyeY,
            rightEyeX, rightEyeY,
            mouthX, mouthY,
            p.width, p.height,
            timestamp / 1000,
            dt,
            ratio > 0.2 ? 1.0 : 0.0,  // mouthOpen
            0,                          // _pad
        ]));

        const encoder = device.createCommandEncoder();

        // 1. Compute pass
        const compute = encoder.beginComputePass();
        compute.setPipeline(computePipeline);
        compute.setBindGroup(0, computeBindGroup);
        compute.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / 64));
        compute.end();

        // 2. Render pass — onto the dedicated gpuCanvas (transparent bg)
        const render = encoder.beginRenderPass({
            colorAttachments: [{
                view: gpuContext.getCurrentTexture().createView(),
                loadOp:     'clear',
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                storeOp:    'store',
            }]
        });
        render.setPipeline(renderPipeline);
        render.setBindGroup(0, renderBindGroup);
        render.draw(NUM_PARTICLES * 6); // 6 vertices per quad
        render.end();

        device.queue.submit([encoder.finish()]);
    }

    // ─── p5 lifecycle ────────────────────────────────────────────────────────────
    p.setup = async () => {
        const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
        canvas.style('position', 'absolute');
        canvas.style('top', '0');
        canvas.style('left', '0');

        video = p.createCapture(p.VIDEO);
        video.hide();

        faceMesh = await ml5.faceMesh(options);
        calcVideoRect();
        await faceMesh.detectStart(video, gotFaces);

        await initWebGPU();

        document.getElementById('audio-btn').addEventListener('click', async () => {
            await startAudio();
            document.getElementById('audio-btn').textContent = '🔊 Sound On';
            document.getElementById('audio-btn').disabled = true;
        });
    };

    p.draw = () => {
        p.background(0, 20);
        p.image(video, vx, vy, vw, vh);
        dispatchGPU(performance.now());

        // Instructions overlay
        p.textSize(16);
        p.textAlign(p.LEFT, p.TOP);
        p.textStyle(p.BOLD);

        // Draw black stroke/shadow by drawing text offset in multiple directions
        p.fill(0, 0, 0, 220);
        p.text('1. Click the 🔊 Sound button to start', 17, 17);
        p.text('1. Click the 🔊 Sound button to start', 15, 17);
        p.text('1. Click the 🔊 Sound button to start', 17, 15);
        p.text('1. Click the 🔊 Sound button to start', 15, 15);
        p.text('2. Open your mouth to release particles', 17, 41);
        p.text('2. Open your mouth to release particles', 15, 41);
        p.text('2. Open your mouth to release particles', 17, 39);
        p.text('2. Open your mouth to release particles', 15, 39);

        // Draw white text on top
        p.fill(255, 255, 255, 220);
        p.text('1. Click the 🔊 Sound button to start', 16, 16);
        p.text('2. Open your mouth to release particles', 16, 40);
    };

    p.windowResized = () => {
        p.resizeCanvas(p.windowWidth, p.windowHeight);
        if (gpuCanvas) {
            gpuCanvas.width  = p.width;
            gpuCanvas.height = p.height;
        }
        calcVideoRect();
    };
};

new p5(sketch, document.getElementById('sketch-container'));
