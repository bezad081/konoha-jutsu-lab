import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, Camera, Check, ChevronDown, CircleHelp, CirclePlay, Cpu, Expand, Eye, Flame,
  Hand, Info, Lightbulb, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Zap
} from 'lucide-react';
import { FaceLandmarker, FilesetResolver, HandLandmarker, ImageSegmenter, type FaceLandmarkerResult, type HandLandmarkerResult } from '@mediapipe/tasks-vision';

type Jutsu = 'none' | 'rasengan' | 'chidori' | 'clone' | 'lightning-burst' | 'fireball' | 'amaterasu' | 'wind-scythe';
type Gesture = 'none' | 'open' | 'fist' | 'two-open' | 'two-fist' | 'clone-seal' | 'ox-seal' | 'hare-seal' | 'monkey-seal';
type Point = { x: number; y: number; z: number };
type NinjaTool = 'none' | 'shuriken' | 'kunai';
type SequenceKind = 'chidori' | 'fireball';
type ThrownTool = { kind: Exclude<NinjaTool, 'none'>; x: number; y: number; vx: number; vy: number; born: number; spin: number };

// Prefer assets shipped with this GitHub Pages build so the browser does not need to reach a third-party model/CDN.
const LOCAL_MODEL_URL = new URL('models/hand_landmarker.task', document.baseURI).href;
const LOCAL_WASM_URL = new URL('mediapipe/wasm/', document.baseURI).href;
const MODEL_URLS = [
  LOCAL_MODEL_URL,
  'https://huggingface.co/Leo-TX/mediapipe-hand/resolve/main/hand_landmarker.task?download=true',
  'https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task',
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
];
const FACE_MODEL_URLS = [new URL('models/face_landmarker.task', document.baseURI).href, 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'];
const SEGMENTER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite';
const WASM_URLS = [
  LOCAL_WASM_URL,
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm',
  'https://unpkg.com/@mediapipe/tasks-vision@0.10.21/wasm',
];

function describeFailure(cause: unknown): string {
  if (cause instanceof Error) return `${cause.name}: ${cause.message}`;
  if (typeof Event !== 'undefined' && cause instanceof Event) {
    const target = cause.target as { src?: unknown; responseURL?: unknown; status?: unknown } | null;
    const resource = typeof target?.src === 'string'
      ? target.src
      : typeof target?.responseURL === 'string'
        ? target.responseURL
        : '';
    const status = typeof target?.status === 'number' ? ` (status ${target.status})` : '';
    return `browser network event "${cause.type}"${resource ? ` for ${resource}` : ''}${status}`;
  }
  return String(cause);
}

async function createHandTracker() {
  const failures: string[] = [];
  for (const wasmUrl of WASM_URLS) {
    try {
      const vision = await FilesetResolver.forVisionTasks(wasmUrl);
      for (const modelAssetPath of MODEL_URLS) {
        for (const delegate of ['GPU', 'CPU'] as const) {
          try {
            const tracker = await HandLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numHands: 2,
              minHandDetectionConfidence: 0.55,
              minHandPresenceConfidence: 0.5,
              minTrackingConfidence: 0.5,
            });
            return { tracker, vision };
          } catch (cause) {
            failures.push(`${delegate} / ${modelAssetPath}: ${describeFailure(cause)}`);
          }
        }
      }
    } catch (cause) {
      failures.push(`WASM ${wasmUrl}: ${describeFailure(cause)}`);
    }
  }
  const recentFailures = failures.slice(-6).join(' | ') || 'No model source could be initialized.';
  throw new Error(
    `Hand tracking could not initialize after trying local assets and fallbacks (${failures.length} failed attempts). Recent diagnostics: ${recentFailures}`,
  );
}
const LINKS: Array<[number, number]> = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]
];
const INFO: Record<Exclude<Jutsu, 'none'>, { name: string; color: string }> = {
  rasengan: { name: 'Rasengan', color: 'blue' },
  chidori: { name: 'Chidori', color: 'cyan' },
  clone: { name: 'Shadow Clone', color: 'violet' },
  'lightning-burst': { name: 'Raiton Pulse', color: 'violet' },
  fireball: { name: 'Fire Style: Fireball Jutsu', color: 'orange' },
  amaterasu: { name: 'Amaterasu · Black Flames', color: 'black' },
  'wind-scythe': { name: 'Wind Fang · Gale Slash', color: 'green' }
};

const COMBO_STEPS: Array<{ gesture: Gesture; title: string; instruction: string }> = [
  { gesture: 'ox-seal', title: 'Charge · Set your stance', instruction: 'Bring both closed fists close together in front of your upper chest — do not press them into your body' },
  { gesture: 'two-fist', title: 'Release · Lightning thrust', instruction: 'Keep both fists closed; drive one fist forward toward the camera while pulling the other back beside your shoulder' },
];
const FIREBALL_STEPS: Array<{ gesture: Gesture; title: string; instruction: string }> = [
  { gesture: 'hare-seal', title: 'Weave · Gather chakra', instruction: 'Raise the index and middle fingers of one hand; curl the other fingers and hold the sign steady' },
  { gesture: 'ox-seal', title: 'Focus · Compress chakra', instruction: 'Bring both closed fists close together in front of your chest' },
  { gesture: 'two-fist', title: 'Release · Uchiha Fire', instruction: 'Separate both fists; fire streams from your mouth and forms a great fireball' },
];

function dist(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * 0.5);
}
function fingerExtended(points: Point[], tip: number, pip: number, mcp: number) {
  if (points.length < 21) return false;
  const wrist = points[0];
  return dist(points[tip], wrist) > dist(points[pip], wrist) * 1.1
    && dist(points[tip], points[mcp]) > dist(points[pip], points[mcp]) * 1.06;
}
function classify(points: Point[]): Gesture {
  if (points.length < 21) return 'none';
  const extended = [
    fingerExtended(points, 8, 6, 5),
    fingerExtended(points, 12, 10, 9),
    fingerExtended(points, 16, 14, 13),
    fingerExtended(points, 20, 18, 17),
  ];
  const count = extended.filter(Boolean).length;
  if (count >= 3) return 'open';
  if (count <= 1) return 'fist';
  return 'none';
}
function midpoint(a: Point, b: Point) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}
function isShadowCloneSeal(hands: Point[][]) {
  if (hands.length !== 2 || hands.some((hand) => hand.length < 21)) return false;
  const poses = hands.map((hand) => ({
    index: fingerExtended(hand, 8, 6, 5),
    middle: fingerExtended(hand, 12, 10, 9),
    ring: fingerExtended(hand, 16, 14, 13),
    pinky: fingerExtended(hand, 20, 18, 17),
  }));
  // The classic plus/cross seal uses index and middle fingers while the outer fingers stay curled.
  if (poses.some((pose) => !pose.index || !pose.middle || pose.ring || pose.pinky)) return false;

  const [a, b] = hands;
  const palmA = palm(a), palmB = palm(b);
  const scaleA = dist(a[0], a[9]), scaleB = dist(b[0], b[9]);
  const palmsNear = Math.hypot(palmA.x - palmB.x, palmA.y - palmB.y) < (scaleA + scaleB) * 1.9;
  if (!palmsNear) return false;

  // Compare the index/middle finger axes. A visible cross/plus is less likely to be confused
  // with simply holding both palms open side by side.
  const baseA = midpoint(a[5], a[9]), tipA = midpoint(a[8], a[12]);
  const baseB = midpoint(b[5], b[9]), tipB = midpoint(b[8], b[12]);
  const ax = tipA.x - baseA.x, ay = tipA.y - baseA.y;
  const bx = tipB.x - baseB.x, by = tipB.y - baseB.y;
  const lengthA = Math.hypot(ax, ay), lengthB = Math.hypot(bx, by);
  if (lengthA < 0.01 || lengthB < 0.01) return false;
  const cosine = Math.min(1, Math.abs((ax * bx + ay * by) / (lengthA * lengthB)));
  const angle = Math.acos(cosine) * 180 / Math.PI;
  const tipsNear = Math.hypot(tipA.x - tipB.x, tipA.y - tipB.y) < (scaleA + scaleB) * 1.65;
  return angle > 24 && tipsNear;
}
function playJutsuCue(audio: AudioContext, jutsu: Exclude<Jutsu, 'none'>) {
  const now = audio.currentTime, master = audio.createGain();
  master.gain.value = jutsu === 'chidori' ? 0.78 : 0.58; master.connect(audio.destination);
  const chirp = (start: number, frequency: number, duration: number, volume: number) => {
    const osc = audio.createOscillator(), gain = audio.createGain(); osc.type = 'sine';
    osc.frequency.setValueAtTime(frequency, start);
    osc.frequency.exponentialRampToValueAtTime(frequency * (1.2 + Math.random() * .45), start + duration * .38);
    osc.frequency.exponentialRampToValueAtTime(frequency * (.55 + Math.random() * .22), start + duration);
    gain.gain.setValueAtTime(.0001, start); gain.gain.exponentialRampToValueAtTime(volume, start + .009);
    gain.gain.exponentialRampToValueAtTime(.0001, start + duration);
    osc.connect(gain); gain.connect(master); osc.start(start); osc.stop(start + duration + .01);
  };
  if (jutsu === 'chidori') {
    // Short, layered "thousand birds" chirps plus an electric crackle — no spoken voice.
    for (let i = 0; i < 13; i++) chirp(now + i * .027, 1750 + (i % 5) * 330, .095, .12 + (i % 3) * .025);
    const buffer = audio.createBuffer(1, Math.floor(audio.sampleRate * .22), audio.sampleRate), data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
    const source = audio.createBufferSource(), filter = audio.createBiquadFilter(), gain = audio.createGain();
    source.buffer = buffer; filter.type = 'bandpass'; filter.frequency.value = 2700; filter.Q.value = .8;
    gain.gain.setValueAtTime(.0001, now); gain.gain.exponentialRampToValueAtTime(.22, now + .018); gain.gain.exponentialRampToValueAtTime(.0001, now + .23);
    source.connect(filter); filter.connect(gain); gain.connect(master); source.start(now); source.stop(now + .24);
  } else if (jutsu === 'rasengan') {
    const osc = audio.createOscillator(), gain = audio.createGain(); osc.type = 'sine';
    osc.frequency.setValueAtTime(260, now); osc.frequency.exponentialRampToValueAtTime(75, now + .42);
    gain.gain.setValueAtTime(.0001, now); gain.gain.exponentialRampToValueAtTime(.23, now + .045); gain.gain.exponentialRampToValueAtTime(.0001, now + .45);
    osc.connect(gain); gain.connect(master); osc.start(now); osc.stop(now + .46);
  } else {
    const duration = jutsu === 'fireball' ? .32 : .2;
    const buffer = audio.createBuffer(1, Math.floor(audio.sampleRate * duration), audio.sampleRate), data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, jutsu === 'fireball' ? 1.8 : 2.8);
    const source = audio.createBufferSource(), filter = audio.createBiquadFilter(), gain = audio.createGain();
    source.buffer = buffer; filter.type = 'lowpass'; filter.frequency.setValueAtTime(jutsu === 'fireball' ? 1300 : 2100, now); filter.frequency.exponentialRampToValueAtTime(180, now + duration);
    gain.gain.setValueAtTime(.0001, now); gain.gain.exponentialRampToValueAtTime(jutsu === 'fireball' ? .34 : .24, now + .02); gain.gain.exponentialRampToValueAtTime(.0001, now + duration);
    source.connect(filter); filter.connect(gain); gain.connect(master); source.start(now); source.stop(now + duration);
    if (jutsu === 'lightning-burst') for (let i = 0; i < 5; i++) chirp(now + i * .026, 900 + i * 380, .07, .11);
  }
}

function palmOrientationAllowsRasengan(result: HandLandmarkerResult, handIndex: number): boolean {
  const world = (result.worldLandmarks?.[handIndex] ?? []) as Point[];
  if (world.length < 21) return false;
  const wrist = world[0], index = world[5], pinky = world[17];
  const ax = index.x - wrist.x, ay = index.y - wrist.y, az = index.z - wrist.z;
  const bx = pinky.x - wrist.x, by = pinky.y - wrist.y, bz = pinky.z - wrist.z;
  const normalY = az * bx - ax * bz;
  const normalZ = ax * by - ay * bx;
  const side = result.handedness?.[handIndex]?.[0]?.categoryName === 'Left' ? 1 : -1;
  return normalZ * side > 0.00002 || normalY * side > 0.00002;
}
function readJutsu(result: HandLandmarkerResult | null): { jutsu: Jutsu; gesture: Gesture } {
  const hands=(result?.landmarks??[]) as Point[][], gestures=hands.map((hand)=>classify(hand));
  if(isShadowCloneSeal(hands)) return {jutsu:'clone',gesture:'clone-seal'};
  const near=hands.length===2&&hands.every(hand=>hand.length>=21)&&Math.hypot(palm(hands[0]).x-palm(hands[1]).x,palm(hands[0]).y-palm(hands[1]).y)<(dist(hands[0][0],hands[0][9])+dist(hands[1][0],hands[1][9]))*1.25;
  if(near&&gestures.every(g=>g==='fist'))return {jutsu:'none',gesture:'ox-seal'};
  if(hands.length===2&&gestures.every(g=>g==='fist'))return {jutsu:'fireball',gesture:'two-fist'};
  if(hands.length===1&&gestures[0]==='fist')return {jutsu:'amaterasu',gesture:'fist'};
  const hare=hands.some(hand=>hand.length>=21&&fingerExtended(hand,8,6,5)&&fingerExtended(hand,12,10,9)&&!fingerExtended(hand,16,14,13)&&!fingerExtended(hand,20,18,17));
  if(hare)return {jutsu:'wind-scythe',gesture:'hare-seal'};
  const openHands = gestures.map((gesture, index) => gesture === 'open' ? index : -1).filter((index) => index >= 0);
  if(openHands.length > 1) {
    if(result && openHands.every((index) => palmOrientationAllowsRasengan(result, index))) return {jutsu:'lightning-burst',gesture:'two-open'};
    return {jutsu:'none',gesture:'none'};
  }
  if(openHands.length === 1) {
    const index = openHands[0];
    if(result && palmOrientationAllowsRasengan(result, index)) return {jutsu:'rasengan',gesture:'open'};
    return {jutsu:'none',gesture:'none'};
  }
  if(gestures.includes('fist'))return {jutsu:'none',gesture:'fist'};
  return {jutsu:'none',gesture:'none'};
}
function palm(points: Point[]) {
  const ids = [0,5,9,13,17];
  return ids.reduce((p, i) => ({ x: p.x + points[i].x / ids.length, y: p.y + points[i].y / ids.length }), { x: 0, y: 0 });
}

async function createFaceTracker(vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>) {
  const failures: string[] = [];
  for (const modelAssetPath of FACE_MODEL_URLS) for (const delegate of ['GPU', 'CPU'] as const) {
    try { return await FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true }); }
    catch (cause) { failures.push(`${delegate}: ${describeFailure(cause)}`); }
  }
  throw new Error(`Face tracking could not initialize: ${failures.slice(-3).join(' | ')}`);
}
const toolImageCache: Partial<Record<'shuriken'|'kunai', HTMLImageElement>> = {};
function getToolImage(tool: 'shuriken'|'kunai'): HTMLImageElement | null {
  if (typeof Image === 'undefined') return null;
  if (!toolImageCache[tool]) { const image = new Image(); image.src = new URL(`assets/${tool}.svg`, document.baseURI).href; toolImageCache[tool] = image; }
  return toolImageCache[tool] ?? null;
}
function drawEnergyTrail(
  ctx: CanvasRenderingContext2D,
  points: Array<{ x: number; y: number; at: number }>,
  w: number,
  h: number,
  now: number,
  color: string,
) {
  if (points.length < 2) return;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  ctx.lineCap = 'round';
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    const age = Math.max(0, now - b.at);
    const fade = Math.max(0, 1 - age / 520) * (i / points.length);
    if (fade <= 0) continue;
    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.quadraticCurveTo(
      (a.x + b.x) * w * 0.5 + Math.sin(now * 0.012 + i) * 2,
      (a.y + b.y) * h * 0.5 + Math.cos(now * 0.01 + i) * 2,
      b.x * w, b.y * h,
    );
    ctx.strokeStyle = color;
    ctx.globalAlpha = fade * 0.72;
    ctx.lineWidth = 1 + fade * 5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8 + fade * 17;
    ctx.stroke();
    if (i % 3 === 0) {
      const px = b.x * w, py = b.y * h;
      ctx.globalAlpha = fade * 0.8;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(px, py, 1 + fade * 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}


function drawHeldTool(ctx: CanvasRenderingContext2D, hand: Point[], tool: Exclude<NinjaTool, 'none'>, w: number, h: number, t: number) {
  if (hand.length < 21) return;
  const p = palm(hand), x = p.x * w, y = p.y * h;
  const angle = Math.atan2(hand[8].y - hand[5].y, hand[8].x - hand[5].x) + Math.PI / 2;
  const size = Math.min(w, h) * (tool === 'shuriken' ? 0.055 : 0.075);
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
  ctx.globalCompositeOperation = 'screen'; ctx.shadowColor = '#bdeeff'; ctx.shadowBlur = 12;
  const image = getToolImage(tool); if (image?.complete && image.naturalWidth > 0) { ctx.globalCompositeOperation = 'source-over'; ctx.shadowColor = '#a9dfff'; ctx.shadowBlur = 13; ctx.drawImage(image, -size * (tool === 'shuriken' ? 1.02 : .58), -size * (tool === 'shuriken' ? 1.02 : 1.32), size * (tool === 'shuriken' ? 2.04 : 1.16), size * (tool === 'shuriken' ? 2.04 : 2.64)); ctx.restore(); return; }
  if (tool === 'shuriken') {
    ctx.rotate(t * 0.55);
    const metal = ctx.createLinearGradient(-size, -size, size, size);
    metal.addColorStop(0, '#f8ffff'); metal.addColorStop(.45, '#8da9c4'); metal.addColorStop(1, '#eaf8ff');
    ctx.beginPath();
    for (let i = 0; i < 16; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 8, radius = i % 2 === 0 ? size : size * .34;
      const px = Math.cos(a) * radius, py = Math.sin(a) * radius;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath(); ctx.fillStyle = metal; ctx.fill();
    ctx.lineWidth = Math.max(1.5, size * .055); ctx.strokeStyle = 'rgba(255,255,255,.95)'; ctx.stroke();
    ctx.beginPath(); ctx.arc(0, 0, size * .16, 0, Math.PI * 2); ctx.fillStyle = '#263c54'; ctx.fill();
  } else {
    ctx.beginPath(); ctx.moveTo(0, -size * 1.25); ctx.lineTo(size * .23, -size * .52);
    ctx.lineTo(size * .18, size * .48); ctx.lineTo(0, size * .7);
    ctx.lineTo(-size * .18, size * .48); ctx.lineTo(-size * .23, -size * .52); ctx.closePath();
    const metal = ctx.createLinearGradient(-size * .3, 0, size * .3, 0);
    metal.addColorStop(0, '#6e8aa5'); metal.addColorStop(.45, '#f1fbff'); metal.addColorStop(1, '#7795b1');
    ctx.fillStyle = metal; ctx.fill(); ctx.strokeStyle = 'rgba(255,255,255,.92)'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, size * .68); ctx.lineTo(0, size * 1.2);
    ctx.strokeStyle = '#c9a77c'; ctx.lineWidth = Math.max(2, size * .11); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, size * 1.22, size * .13, 0, Math.PI * 2);
    ctx.strokeStyle = '#d8e7f4'; ctx.lineWidth = Math.max(1.5, size * .07); ctx.stroke();
  }
  ctx.restore();
}

function drawSharinganEyes(ctx: CanvasRenderingContext2D, face: Point[], w: number, h: number, t: number) {
  if(face.length<478)return;const eyes=[{center:468,contour:[33,160,158,133,153,144],corners:[33,133]},{center:473,contour:[362,385,387,263,373,380],corners:[362,263]}];ctx.save();ctx.globalCompositeOperation='source-over';
  for(const eye of eyes){const c={x:face[eye.center].x*w,y:face[eye.center].y*h},ew=Math.hypot((face[eye.corners[0]].x-face[eye.corners[1]].x)*w,(face[eye.corners[0]].y-face[eye.corners[1]].y)*h),r=Math.max(3,ew*.255);ctx.save();ctx.beginPath();eye.contour.forEach((id,i)=>{const p=face[id];if(!i)ctx.moveTo(p.x*w,p.y*h);else ctx.lineTo(p.x*w,p.y*h);});ctx.closePath();ctx.clip();const glow=ctx.createRadialGradient(c.x,c.y,r*.1,c.x,c.y,r*1.8);glow.addColorStop(0,'rgba(255,25,40,.54)');glow.addColorStop(1,'rgba(255,0,22,0)');ctx.fillStyle=glow;ctx.beginPath();ctx.arc(c.x,c.y,r*1.8,0,Math.PI*2);ctx.fill();const iris=ctx.createRadialGradient(c.x-r*.24,c.y-r*.3,r*.08,c.x,c.y,r);iris.addColorStop(0,'#ff7777');iris.addColorStop(.34,'#e7192e');iris.addColorStop(.78,'#a90017');iris.addColorStop(1,'#3b0612');ctx.fillStyle=iris;ctx.beginPath();ctx.arc(c.x,c.y,r,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#190007';ctx.lineWidth=Math.max(1,r*.12);ctx.beginPath();ctx.arc(c.x,c.y,r*.9,0,Math.PI*2);ctx.stroke();ctx.strokeStyle='rgba(40,0,7,.72)';ctx.lineWidth=Math.max(.8,r*.045);for(let ray=0;ray<18;ray++){const a=ray*Math.PI/9+t*.025;ctx.beginPath();ctx.moveTo(c.x+Math.cos(a)*r*.28,c.y+Math.sin(a)*r*.28);ctx.lineTo(c.x+Math.cos(a)*r*.82,c.y+Math.sin(a)*r*.82);ctx.stroke();}for(let m=0;m<3;m++){const a=m*Math.PI*2/3-Math.PI/2,mx=c.x+Math.cos(a)*r*.59,my=c.y+Math.sin(a)*r*.59;ctx.save();ctx.translate(mx,my);ctx.rotate(a+Math.PI/2);ctx.fillStyle='#160208';ctx.beginPath();ctx.arc(0,0,r*.19,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(r*.04,r*.13);ctx.quadraticCurveTo(r*.37,r*.08,r*.24,-r*.25);ctx.quadraticCurveTo(r*.18,-r*.04,r*.04,r*.13);ctx.fill();ctx.restore();}ctx.fillStyle='#100107';ctx.beginPath();ctx.arc(c.x,c.y,r*.22,0,Math.PI*2);ctx.fill();ctx.fillStyle='rgba(255,255,255,.9)';ctx.beginPath();ctx.arc(c.x-r*.25,c.y-r*.28,Math.max(1,r*.11),0,Math.PI*2);ctx.fill();ctx.restore();}ctx.restore();
}

function drawThrownTools(ctx: CanvasRenderingContext2D, projectiles: ThrownTool[], w: number, h: number, now: number) {
  for (const projectile of projectiles) {
    const age = (now - projectile.born) / 1000;
    const fade = Math.max(0, Math.min(1, (1.8 - age) / .28));
    const x = projectile.x * w, y = projectile.y * h;
    const size = Math.min(w, h) * (projectile.kind === 'shuriken' ? .035 : .047);
    ctx.save();
    ctx.globalAlpha = fade * .78; ctx.globalCompositeOperation = 'screen';
    ctx.beginPath();
    ctx.moveTo((projectile.x - projectile.vx * .065) * w, (projectile.y - projectile.vy * .065) * h);
    ctx.lineTo(x, y);
    ctx.strokeStyle = projectile.kind === 'shuriken' ? 'rgba(195,242,255,.82)' : 'rgba(167,211,255,.8)';
    ctx.lineWidth = Math.max(2, size * .18); ctx.shadowColor = '#9edfff'; ctx.shadowBlur = 13; ctx.stroke();
    ctx.translate(x, y);
    ctx.rotate(projectile.spin + age * (projectile.kind === 'shuriken' ? 18 : 8));
    ctx.globalAlpha = fade; ctx.globalCompositeOperation = 'screen';
    ctx.shadowColor = '#9edfff'; ctx.shadowBlur = 16;
    const image = getToolImage(projectile.kind); if (image?.complete && image.naturalWidth > 0) { ctx.globalCompositeOperation = 'source-over'; ctx.shadowColor = '#b7e6ff'; ctx.shadowBlur = 13; ctx.drawImage(image, -size * (projectile.kind === 'shuriken' ? .95 : .54), -size * (projectile.kind === 'shuriken' ? .95 : 1.24), size * (projectile.kind === 'shuriken' ? 1.9 : 1.08), size * (projectile.kind === 'shuriken' ? 1.9 : 2.48)); ctx.restore(); continue; }
    if (projectile.kind === 'shuriken') {
      ctx.beginPath();
      for (let i = 0; i < 16; i++) {
        const a = -Math.PI / 2 + i * Math.PI / 8, radius = i % 2 === 0 ? size : size * .32;
        const px = Math.cos(a) * radius, py = Math.sin(a) * radius;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath(); ctx.fillStyle = '#dff7ff'; ctx.fill();
      ctx.lineWidth = Math.max(1, size * .08); ctx.strokeStyle = '#ffffff'; ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, size * .15, 0, Math.PI * 2); ctx.fillStyle = '#526b82'; ctx.fill();
    } else {
      ctx.beginPath(); ctx.moveTo(0, -size * 1.35); ctx.lineTo(size * .25, -size * .42);
      ctx.lineTo(size * .16, size * .52); ctx.lineTo(0, size * .72);
      ctx.lineTo(-size * .16, size * .52); ctx.lineTo(-size * .25, -size * .42); ctx.closePath();
      const metal = ctx.createLinearGradient(-size * .3, 0, size * .3, 0);
      metal.addColorStop(0, '#7896b3'); metal.addColorStop(.5, '#f7ffff'); metal.addColorStop(1, '#7896b3');
      ctx.fillStyle = metal; ctx.fill(); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, size * .7); ctx.lineTo(0, size * 1.25);
      ctx.strokeStyle = '#c7a078'; ctx.lineWidth = Math.max(2, size * .11); ctx.stroke();
    }
    ctx.restore();
  }
}

function drawLandmarks(ctx: CanvasRenderingContext2D, hands: Point[][], w: number, h: number) {
  ctx.save(); ctx.lineWidth = Math.max(1.5, w * 0.002); ctx.lineCap = 'round';
  for (const hand of hands) {
    ctx.strokeStyle = 'rgba(130,235,255,.62)';
    for (const pair of LINKS) {
      ctx.beginPath(); ctx.moveTo(hand[pair[0]].x * w, hand[pair[0]].y * h);
      ctx.lineTo(hand[pair[1]].x * w, hand[pair[1]].y * h); ctx.stroke();
    }
    for (const p of hand) {
      ctx.beginPath(); ctx.fillStyle = '#e5fbff'; ctx.shadowColor = '#47dcff'; ctx.shadowBlur = 7;
      ctx.arc(p.x * w, p.y * h, Math.max(2, w * 0.003), 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.restore();
}
function drawRasengan(
  ctx: CanvasRenderingContext2D,
  p: {x:number;y:number},
  w: number,
  h: number,
  t: number,
  charge = 0.72,
) {
  const x = p.x * w, y = p.y * h;
  const pulse = 1 + Math.sin(t * 5.2) * 0.035 + charge * 0.035;
  const r = Math.min(w, h) * (0.027 + charge * 0.073) * pulse;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';

  // Soft spill light first, then a bright, dense chakra core.
  const halo = ctx.createRadialGradient(x, y, r * 0.12, x, y, r * 2.15);
  halo.addColorStop(0, 'rgba(245,255,255,.98)');
  halo.addColorStop(0.16, 'rgba(105,225,255,.82)');
  halo.addColorStop(0.47, 'rgba(37,146,255,.34)');
  halo.addColorStop(1, 'rgba(20,105,255,0)');
  ctx.fillStyle = halo;
  ctx.beginPath(); ctx.arc(x, y, r * 2.15, 0, Math.PI * 2); ctx.fill();

  const core = ctx.createRadialGradient(x - r * 0.22, y - r * 0.25, r * 0.03, x, y, r * 0.77);
  core.addColorStop(0, '#ffffff');
  core.addColorStop(0.22, '#d8fbff');
  core.addColorStop(0.55, '#5bcfff');
  core.addColorStop(0.83, '#197eff');
  core.addColorStop(1, 'rgba(20,86,255,.08)');
  ctx.shadowColor = '#38bfff'; ctx.shadowBlur = 30 + charge * 22;
  ctx.fillStyle = core; ctx.beginPath(); ctx.arc(x, y, r * 0.78, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;

  // Multiple counter-rotating, incomplete energy ribbons create the turbulent spinning look.
  for (let ring = 0; ring < 7; ring++) {
    const direction = ring % 2 ? -1 : 1;
    const rotation = t * (1.25 + ring * 0.17) * direction + ring * 1.37;
    ctx.beginPath();
    ctx.ellipse(x, y, r * (0.43 + ring * 0.095), r * (0.25 + ring * 0.072), rotation, 0.22 + ring * 0.08, Math.PI * (1.18 + (ring % 3) * 0.2));
    ctx.strokeStyle = ring % 3 === 0 ? 'rgba(247,255,255,.92)' : ring % 2 ? 'rgba(65,170,255,.72)' : 'rgba(113,224,255,.86)';
    ctx.lineWidth = Math.max(1, r * (ring % 2 ? 0.026 : 0.04));
    ctx.shadowColor = '#5bcfff'; ctx.shadowBlur = 8 + charge * 9; ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Curled filaments spiral inward instead of making the effect look like static concentric circles.
  for (let strand = 0; strand < 5; strand++) {
    const phase = t * (1.9 + strand * 0.18) + strand * Math.PI * 0.4;
    ctx.beginPath();
    for (let i = 0; i <= 28; i++) {
      const k = i / 28;
      const a = phase + k * Math.PI * 2.2;
      const rr = r * (0.12 + k * (0.62 + 0.04 * Math.sin(t * 3 + strand)));
      const px = x + Math.cos(a) * rr;
      const py = y + Math.sin(a) * rr * (0.8 + 0.15 * Math.sin(a * 2 + strand));
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = strand % 2 ? 'rgba(255,255,255,.64)' : 'rgba(73,193,255,.78)';
    ctx.lineWidth = Math.max(1, r * 0.022); ctx.stroke();
  }

  for (let i = 0; i < 34; i++) {
    const a = t * (2.5 + (i % 4) * 0.3) + i * Math.PI * 2 / 34;
    const rr = r * (0.22 + 0.63 * ((Math.sin(t * 2.1 + i * 2.17) + 1) / 2));
    const size = Math.max(1, r * (i % 6 === 0 ? 0.045 : 0.021));
    ctx.fillStyle = i % 4 === 0 ? '#ffffff' : i % 2 ? '#75e2ff' : '#368dff';
    ctx.shadowColor = '#8beaff'; ctx.shadowBlur = i % 4 === 0 ? 12 : 5;
    ctx.beginPath(); ctx.arc(x + Math.cos(a) * rr, y + Math.sin(a) * rr * 0.86, size, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}
function drawChidori(
  ctx: CanvasRenderingContext2D,
  p: {x:number;y:number},
  movement: {x:number;y:number},
  w: number,
  h: number,
  t: number,
  charge = 0.72,
  hand?: Point[],
) {
  const x = p.x * w, y = p.y * h;
  const pulse = 1 + Math.sin(t * 12.5) * 0.055;
  const r = Math.min(w, h) * (0.073 + charge * 0.027) * pulse;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';

  // Charged electric haze and a white-hot center make the lightning read as one energy mass.
  const haze = ctx.createRadialGradient(x, y, r * 0.04, x, y, r * 1.7);
  haze.addColorStop(0, 'rgba(249,255,255,.96)');
  haze.addColorStop(0.17, 'rgba(151,233,255,.86)');
  haze.addColorStop(0.48, 'rgba(49,159,255,.38)');
  haze.addColorStop(1, 'rgba(22,112,255,0)');
  ctx.fillStyle = haze; ctx.beginPath(); ctx.arc(x, y, r * 1.7, 0, Math.PI * 2); ctx.fill();

  // Lightning coats the detected hand: palm, knuckles, fingers and fingertips.
  if (hand && hand.length >= 21) {
    ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const [a, b] of LINKS) {
      const ax=hand[a].x*w, ay=hand[a].y*h, bx=hand[b].x*w, by=hand[b].y*h;
      const dx=bx-ax, dy=by-ay, length=Math.hypot(dx,dy)||1, nx=-dy/length, ny=dx/length;
      const jitter=Math.sin(t*31+a*3.7+b)*Math.min(8,length*.17), mx=(ax+bx)/2+nx*jitter, my=(ay+by)/2+ny*jitter;
      ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(mx,my);ctx.lineTo(bx,by);ctx.strokeStyle='rgba(62,166,255,.96)';ctx.lineWidth=Math.max(4,Math.min(13,length*.28));ctx.shadowColor='#2fa8ff';ctx.shadowBlur=17+charge*10;ctx.stroke();
      ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(mx,my);ctx.lineTo(bx,by);ctx.strokeStyle='rgba(225,250,255,.94)';ctx.lineWidth=Math.max(1.2,Math.min(3.5,length*.07));ctx.shadowColor='#d9f7ff';ctx.shadowBlur=8;ctx.stroke();
      if((a+b)%3===0){const side=Math.sin(t*27+a*5)>0?1:-1;ctx.beginPath();ctx.moveTo(mx,my);ctx.lineTo(mx+nx*side*(5+charge*8),my+ny*side*(5+charge*8));ctx.lineTo(mx+nx*side*(10+charge*13)+dx*.12,my+ny*side*(10+charge*13)+dy*.12);ctx.strokeStyle='rgba(168,229,255,.9)';ctx.lineWidth=1.2;ctx.shadowBlur=9;ctx.stroke();}
    }
    for(const id of [4,8,12,16,20]){const tip=hand[id],tx=tip.x*w,ty=tip.y*h,g=ctx.createRadialGradient(tx,ty,0,tx,ty,12+charge*7);g.addColorStop(0,'rgba(255,255,255,.98)');g.addColorStop(.28,'rgba(105,209,255,.78)');g.addColorStop(1,'rgba(42,145,255,0)');ctx.fillStyle=g;ctx.beginPath();ctx.arc(tx,ty,12+charge*7,0,Math.PI*2);ctx.fill();}
    ctx.restore();
  }

  // Fine jagged bolts branch out from a compact, bright core.
  for (let bolt = 0; bolt < 17; bolt++) {
    const a = bolt * Math.PI * 2 / 17 + Math.sin(t * 3.1 + bolt) * 0.12;
    const length = r * (0.9 + 0.52 * ((Math.sin(t * 10 + bolt * 7.7) + 1) / 2));
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(a) * r * 0.15, y + Math.sin(a) * r * 0.15);
    let lastX = x, lastY = y;
    for (let step = 1; step <= 7; step++) {
      const k = step / 7;
      const jitter = step === 7 ? 0 : Math.sin(t * 22 + bolt * 4.3 + step * 8.1) * r * (0.1 + 0.08 * charge);
      const nx = x + Math.cos(a) * length * k + Math.cos(a + Math.PI / 2) * jitter;
      const ny = y + Math.sin(a) * length * k + Math.sin(a + Math.PI / 2) * jitter;
      ctx.lineTo(nx, ny);
      lastX = nx; lastY = ny;
      if (step === 4 && bolt % 3 === 0) {
        ctx.moveTo(nx, ny);
        ctx.lineTo(nx + Math.cos(a + 0.8) * r * 0.22, ny + Math.sin(a + 0.8) * r * 0.22);
        ctx.moveTo(nx, ny);
      }
    }
    ctx.strokeStyle = bolt % 4 === 0 ? 'rgba(255,255,255,.98)' : bolt % 2 ? 'rgba(80,182,255,.92)' : 'rgba(141,228,255,.94)';
    ctx.lineWidth = bolt % 4 === 0 ? 2.3 + charge : 1 + charge * 0.7;
    ctx.shadowColor = '#49beff'; ctx.shadowBlur = 8 + charge * 13; ctx.stroke();
    // Tiny hot point sparks ride the tips of longer bolts.
    if (bolt % 3 === 0) {
      ctx.fillStyle = '#e9fdff'; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(lastX, lastY, 1.2 + charge, 0, Math.PI * 2); ctx.fill();
    }
  }

  // The hand's recent motion leaves a short, directional lightning tail.
  const speed = Math.hypot(movement.x, movement.y);
  if (speed > 0.0015) {
    const dx = movement.x / speed, dy = movement.y / speed;
    const tail = r * (1.35 + Math.min(1, speed * 24) * 1.7);
    for (let i = 0; i < 4; i++) {
      const offset = (i - 1.5) * r * 0.1;
      ctx.beginPath();
      ctx.moveTo(x - dx * r * 0.15 + -dy * offset, y - dy * r * 0.15 + dx * offset);
      ctx.lineTo(x - dx * tail * 0.46 + -dy * offset * 1.4, y - dy * tail * 0.46 + dx * offset * 1.4);
      ctx.lineTo(x - dx * tail * (0.7 + i * 0.08) + -dy * offset * 0.4, y - dy * tail * (0.7 + i * 0.08) + dx * offset * 0.4);
      ctx.strokeStyle = i % 2 ? 'rgba(65,166,255,.48)' : 'rgba(197,246,255,.8)';
      ctx.lineWidth = i % 2 ? 1.6 : 2.4; ctx.shadowBlur = 13; ctx.stroke();
    }
  }
  for (let i = 0; i < 12; i++) {
    const a = t * 4.2 + i * Math.PI * 2 / 12;
    const rr = r * (0.18 + 0.88 * ((Math.sin(t * 8 + i * 1.9) + 1) / 2));
    ctx.fillStyle = i % 3 ? '#71dcff' : '#ffffff'; ctx.shadowBlur = 7;
    ctx.beginPath(); ctx.arc(x + Math.cos(a) * rr, y + Math.sin(a) * rr, 1 + (i % 3) * 0.6, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}
function drawLightningBurst(ctx: CanvasRenderingContext2D, p: {x:number;y:number}, w: number, h: number, t: number, charge = 0.72) {
  const x=p.x*w, y=p.y*h, base=Math.min(w,h)*(0.075+charge*0.035), pulse=1+Math.sin(t*9)*0.08;
  ctx.save(); ctx.globalCompositeOperation='screen'; ctx.lineCap='round'; ctx.lineJoin='round';
  const glow=ctx.createRadialGradient(x,y,base*.08,x,y,base*2.5);
  glow.addColorStop(0,'rgba(255,255,255,.96)'); glow.addColorStop(.2,'rgba(171,244,255,.82)'); glow.addColorStop(.55,'rgba(58,160,255,.32)'); glow.addColorStop(1,'rgba(45,115,255,0)');
  ctx.fillStyle=glow;ctx.beginPath();ctx.arc(x,y,base*2.5*pulse,0,Math.PI*2);ctx.fill();
  for(let ring=0;ring<3;ring++){
    ctx.beginPath();ctx.ellipse(x,y,base*(.65+ring*.38)*pulse,base*(.35+ring*.24),t*(ring%2?-.9:1.1)+ring,0,Math.PI*2);
    ctx.strokeStyle=ring===0?'rgba(255,255,255,.95)':'rgba(83,196,255,.78)';ctx.lineWidth=ring===0?3:1.8;ctx.shadowColor='#50caff';ctx.shadowBlur=14;ctx.stroke();
  }
  for(let bolt=0;bolt<22;bolt++){
    const a=bolt*Math.PI*2/22+t*(bolt%2?1.7:-1.3),len=base*(.75+((Math.sin(t*11+bolt*2.8)+1)/2)*1.25);
    ctx.beginPath();ctx.moveTo(x+Math.cos(a)*base*.16,y+Math.sin(a)*base*.16);
    for(let step=1;step<=5;step++){const k=step/5,j=step===5?0:Math.sin(t*24+bolt+step*3)*base*.12;ctx.lineTo(x+Math.cos(a)*len*k-Math.sin(a)*j,y+Math.sin(a)*len*k+Math.cos(a)*j);}
    ctx.strokeStyle=bolt%4===0?'rgba(255,255,255,.98)':'rgba(91,191,255,.92)';ctx.lineWidth=bolt%4===0?2.4:1.3;ctx.shadowBlur=10;ctx.stroke();
  }
  ctx.restore();
}

function drawUchihaMouthFire(ctx: CanvasRenderingContext2D, face: Point[], w: number, h: number, t: number, charge = 0.8): {x:number;y:number} {
  if(face.length<292)return {x:.5,y:.5};
  const mouth={x:(face[13].x+face[14].x)*.5*w,y:(face[13].y+face[14].y)*.5*h},mouthWidth=Math.max(12,Math.hypot((face[61].x-face[291].x)*w,(face[61].y-face[291].y)*h));
  const eyeMid={x:(face[33].x+face[263].x)*.5*w,y:(face[33].y+face[263].y)*.5*h},nose={x:face[1].x*w,y:face[1].y*h};
  const angle=Math.atan2(nose.y-eyeMid.y,nose.x-eyeMid.x)+.04*Math.sin(t*1.4),dx=Math.cos(angle),dy=Math.sin(angle),nx=-dy,ny=dx,length=Math.min(w,h)*(.23+charge*.08),base=mouthWidth*(.22+charge*.1);
  ctx.save();ctx.globalCompositeOperation='screen';ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(mouth.x,mouth.y);ctx.quadraticCurveTo(mouth.x+dx*length*.48+nx*base,mouth.y+dy*length*.48+ny*base,mouth.x+dx*length,mouth.y+dy*length);ctx.strokeStyle='rgba(255,76,12,.48)';ctx.lineWidth=base*4.8;ctx.shadowColor='#ff4b0b';ctx.shadowBlur=base*3.8;ctx.stroke();
  for(let r=0;r<7;r++){const side=(r-3)/3;ctx.beginPath();for(let s=0;s<=12;s++){const p=s/12,wiggle=Math.sin(t*(7+r*.19)+p*15+r*1.7)*base*(.12+p*.42),spread=side*base*(.15+p*2.6),along=p*length,x=mouth.x+dx*along+nx*(spread+wiggle),y=mouth.y+dy*along+ny*(spread+wiggle);if(!s)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.strokeStyle=r%3===0?'rgba(255,246,190,.96)':r%3===1?'rgba(255,167,42,.94)':'rgba(255,58,12,.88)';ctx.lineWidth=base*(r%3===0?.36:.72);ctx.shadowColor=r%3===0?'#fff1a3':'#ff4b0b';ctx.shadowBlur=base*(1.2+charge);ctx.stroke();}
  for(let i=0;i<24;i++){const p=((i*.61803398875+t*.11)%1),side=Math.sin(i*2.37+t*5.5)*base*(.15+p*2.3),along=p*length,x=mouth.x+dx*along+nx*side,y=mouth.y+dy*along+ny*side,q=base*(.16+((Math.sin(t*13+i*3.1)+1)*.5)*.55)*(1-p*.45);ctx.beginPath();ctx.moveTo(x-nx*q*.2,y-ny*q*.2);ctx.quadraticCurveTo(x+dx*q*.55+nx*q,y+dy*q*.55+ny*q,x+dx*q*1.15-nx*q*.18,y+dy*q*1.15-ny*q*.18);ctx.strokeStyle=i%4===0?'#fff6c5':i%2?'#ffb43c':'#ff4a16';ctx.lineWidth=Math.max(1.2,q*.32);ctx.shadowColor='#ff6b1b';ctx.shadowBlur=9;ctx.stroke();}
  const glow=ctx.createRadialGradient(mouth.x,mouth.y,1,mouth.x,mouth.y,base*2.1);glow.addColorStop(0,'rgba(255,255,224,.98)');glow.addColorStop(.25,'rgba(255,179,61,.82)');glow.addColorStop(1,'rgba(255,50,9,0)');ctx.fillStyle=glow;ctx.beginPath();ctx.ellipse(mouth.x,mouth.y,base*1.5,base*.9,angle,0,Math.PI*2);ctx.fill();ctx.restore();return {x:(mouth.x+dx*length)/w,y:(mouth.y+dy*length)/h};
}

function drawFireball(ctx: CanvasRenderingContext2D, p: {x:number;y:number}, w:number, h:number, t:number, charge=0.8) {
  const x=p.x*w,y=p.y*h,r=Math.min(w,h)*(.07+charge*.045),pulse=1+Math.sin(t*13)*.07;
  ctx.save();ctx.globalCompositeOperation='screen';ctx.lineCap='round';ctx.lineJoin='round';
  const glow=ctx.createRadialGradient(x,y,r*.1,x,y,r*2.8);
  glow.addColorStop(0,'rgba(255,255,220,.98)');glow.addColorStop(.18,'rgba(255,203,85,.92)');glow.addColorStop(.48,'rgba(255,94,24,.48)');glow.addColorStop(1,'rgba(255,48,12,0)');
  ctx.fillStyle=glow;ctx.beginPath();ctx.arc(x,y,r*2.8*pulse,0,Math.PI*2);ctx.fill();
  for(let ring=0;ring<5;ring++){
    const phase=t*(ring%2?1.8:-1.45)+ring*1.25;ctx.beginPath();
    for(let i=0;i<=42;i++){const k=i/42,a=phase+k*Math.PI*2*(1.1+ring*.12),rr=r*(.2+k*(.5+ring*.08)),px=x+Math.cos(a)*rr,py=y+Math.sin(a)*rr*(.7+.12*Math.sin(a*3+ring));if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);}
    ctx.strokeStyle=ring%2?'rgba(255,236,151,.9)':'rgba(255,112,35,.86)';ctx.lineWidth=ring%2?2.2:3.3;ctx.shadowColor='#ff5a16';ctx.shadowBlur=15;ctx.stroke();
  }
  const core=ctx.createRadialGradient(x-r*.22,y-r*.28,r*.04,x,y,r*.78);
  core.addColorStop(0,'#fffde2');core.addColorStop(.3,'#ffe477');core.addColorStop(.62,'#ff8b24');core.addColorStop(1,'rgba(255,62,12,.08)');
  ctx.fillStyle=core;ctx.shadowColor='#ff6b20';ctx.shadowBlur=28;ctx.beginPath();ctx.arc(x,y,r*.75*pulse,0,Math.PI*2);ctx.fill();
  for(let i=0;i<28;i++){const a=t*(1.8+(i%4)*.4)+i*Math.PI*2/28,rr=r*(.72+((Math.sin(t*9+i*2.3)+1)/2)*.82),len=r*(.08+((Math.sin(t*12+i)+1)/2)*.16);ctx.beginPath();ctx.moveTo(x+Math.cos(a)*rr,y+Math.sin(a)*rr);ctx.lineTo(x+Math.cos(a)*(rr+len),y+Math.sin(a)*(rr+len));ctx.strokeStyle=i%3?'rgba(255,171,56,.9)':'rgba(255,246,194,.98)';ctx.lineWidth=i%3?1.5:2.4;ctx.shadowBlur=10;ctx.stroke();}
  ctx.restore();
}

function drawCloneEcho(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
  ctx.save(); ctx.globalCompositeOperation='screen';
  for(let i=0;i<3;i++) {
    const x=w*.5+Math.sin(t*2.5+i*2)*w*.025+(i-1)*w*.14, y=h*.52;
    ctx.globalAlpha=.1+.04*Math.sin(t*4+i); ctx.fillStyle=i%2?'#bd9aff':'#75dfff';
    ctx.shadowColor=ctx.fillStyle; ctx.shadowBlur=28;
    ctx.beginPath(); ctx.ellipse(x,y-h*.12,w*.035,h*.075,0,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.roundRect(x-w*.036,y-h*.03,w*.072,h*.22,w*.025); ctx.fill();
    ctx.strokeStyle='rgba(180,225,255,.45)'; ctx.lineWidth=1.1; ctx.stroke();
  }
  ctx.restore();
}

type ConfidenceMask = { width: number; height: number; getAsFloat32Array(): Float32Array };

function updateCloneSprite(
  video: HTMLVideoElement,
  mask: ConfidenceMask,
  sprite: HTMLCanvasElement,
) {
  const width = mask.width, height = mask.height;
  if (!width || !height) return;
  if (sprite.width !== width || sprite.height !== height) {
    sprite.width = width;
    sprite.height = height;
  }
  const ctx = sprite.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(video, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  const scores = mask.getAsFloat32Array();
  const count = Math.min(scores.length, width * height);
  for (let i = 0; i < count; i++) {
    const confidence = Math.max(0, Math.min(1, (scores[i] - 0.18) / 0.58));
    const soft = confidence * confidence * (3 - 2 * confidence);
    pixels.data[i * 4 + 3] = Math.round(pixels.data[i * 4 + 3] * soft);
  }
  ctx.putImageData(pixels, 0, 0);
}

function drawWindScythe(ctx: CanvasRenderingContext2D, p: {x:number;y:number}, move: {x:number;y:number}, w:number, h:number, t:number, charge=0.8) {
  const x=p.x*w,y=p.y*h,base=Math.min(w,h)*(.11+charge*.045),speed=Math.hypot(move.x,move.y);
  ctx.save();ctx.globalCompositeOperation='screen';ctx.lineCap='round';ctx.lineJoin='round';
  const aura=ctx.createRadialGradient(x,y,base*.05,x,y,base*2.2);
  aura.addColorStop(0,'rgba(235,255,247,.65)');aura.addColorStop(.35,'rgba(100,255,190,.32)');aura.addColorStop(1,'rgba(50,255,170,0)');
  ctx.fillStyle=aura;ctx.beginPath();ctx.arc(x,y,base*2.2,0,Math.PI*2);ctx.fill();
  const direction=Math.atan2(move.y,move.x)+(speed>.012?0:-.22);
  for(let blade=0;blade<3;blade++){
    const phase=t*(2.1+blade*.38)+blade*2.05,angle=direction+phase*.14;
    const radius=base*(.8+blade*.27),start=angle-.92,end=angle+.92;
    ctx.beginPath();ctx.ellipse(x+Math.cos(phase)*base*.2,y+Math.sin(phase)*base*.2,radius*1.12,radius*.48,angle,start,end);
    ctx.strokeStyle=blade===0?'rgba(255,255,255,.96)':blade===1?'rgba(137,255,204,.92)':'rgba(53,206,151,.72)';
    ctx.lineWidth=blade===0?3.2:2.1;ctx.shadowColor='#6cffbd';ctx.shadowBlur=18+charge*10;ctx.stroke();
  }
  for(let i=0;i<18;i++){
    const a=i*2.399+t*(i%2?1.8:-1.2),r=base*(.42+((i*7)%11)/11*1.55);
    const px=x+Math.cos(a)*r,py=y+Math.sin(a)*r;
    ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px+Math.cos(a+.7)*(3+(i%4)*2),py+Math.sin(a+.7)*(3+(i%4)*2));
    ctx.strokeStyle=i%3?'rgba(101,255,190,.86)':'rgba(255,255,255,.95)';ctx.lineWidth=i%3?1.5:2.4;ctx.shadowBlur=8;ctx.stroke();
  }
  ctx.restore();
}

function drawAmaterasu(ctx: CanvasRenderingContext2D, p: {x:number;y:number}, w:number, h:number, t:number, charge=0.8) {
  const x=p.x*w,y=p.y*h,base=Math.min(w,h)*(.06+charge*.025);
  ctx.save();ctx.globalCompositeOperation='source-over';ctx.lineCap='round';ctx.lineJoin='round';
  const aura=ctx.createRadialGradient(x,y,base*.05,x,y,base*2.2);
  aura.addColorStop(0,'rgba(255,255,255,.8)');aura.addColorStop(.18,'rgba(100,58,150,.58)');aura.addColorStop(.52,'rgba(38,14,58,.42)');aura.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=aura;ctx.beginPath();ctx.arc(x,y,base*2.2,0,Math.PI*2);ctx.fill();
  for(let i=0;i<13;i++){
    const a=i*Math.PI*2/13+t*(i%2?-.55:.7),flicker=.72+.28*Math.sin(t*8+i*2.4);
    const inner=base*(.34+(i%3)*.08),outer=base*(1.05+(i%4)*.19)*flicker;
    const sx=x+Math.cos(a)*inner,sy=y+Math.sin(a)*inner;
    const ex=x+Math.cos(a+.12*Math.sin(t*6+i))*outer,ey=y+Math.sin(a+.12*Math.cos(t*5+i))*outer;
    ctx.beginPath();ctx.moveTo(sx,sy);ctx.quadraticCurveTo(x+Math.cos(a+.28)*outer*.55,y+Math.sin(a+.28)*outer*.55,ex,ey);
    ctx.strokeStyle=i%3===0?'rgba(155,110,255,.96)':'rgba(8,5,16,.98)';ctx.lineWidth=Math.max(3,base*(i%3===0?.12:.2));ctx.shadowColor='#6c38a8';ctx.shadowBlur=10+charge*12;ctx.stroke();
    ctx.beginPath();ctx.arc(ex,ey,Math.max(1.5,base*.035),0,Math.PI*2);ctx.fillStyle=i%3===0?'#bba0ff':'#08050e';ctx.fill();
  }
  ctx.restore();
}

function drawSegmentedClones(
  ctx: CanvasRenderingContext2D,
  sprite: HTMLCanvasElement,
  w: number,
  h: number,
  t: number,
) {
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  const echoes = [
    { x: -0.23 + Math.sin(t * 1.4) * 0.012, y: 0.006, scale: 0.96, alpha: 0.76, hue: 135 },
    { x: 0.23 + Math.sin(t * 1.7 + 1) * 0.012, y: -0.004, scale: 0.94, alpha: 0.72, hue: 205 },
  ];
  for (const [index, echo] of echoes.entries()) {
    const cloneX = w * (0.5 + echo.x);
    const cloneY = h * (0.72 + echo.y);
    const smokePulse = Math.pow(Math.max(0, Math.sin(t * 3.4 + index * 2.15)), 2);
    if (smokePulse > 0.02) {
      ctx.save();
      ctx.globalAlpha = smokePulse * 0.19;
      ctx.filter = 'blur(7px)';
      ctx.fillStyle = '#dff7ff';
      for (let puff = 0; puff < 7; puff++) {
        const angle = puff * Math.PI * 2 / 7 + t * 0.7;
        const drift = h * (0.012 + smokePulse * 0.018);
        ctx.beginPath();
        ctx.ellipse(
          cloneX + Math.cos(angle) * (w * 0.018 + drift),
          cloneY - smokePulse * h * 0.025 + Math.sin(angle) * drift,
          w * (0.012 + smokePulse * 0.009),
          h * (0.014 + smokePulse * 0.01),
          angle * 0.4, 0, Math.PI * 2,
        );
        ctx.fill();
      }
      ctx.restore();
    }
    ctx.save();
    ctx.globalAlpha = echo.alpha;
    ctx.filter = 'drop-shadow(0 0 13px rgba(130,205,255,.62))';
    ctx.drawImage(
      sprite,
      w * echo.x + (w - w * echo.scale) / 2,
      h * echo.y + (h - h * echo.scale) / 2,
      w * echo.scale,
      h * echo.scale,
    );
    ctx.restore();
  }
  ctx.restore();
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackerRef = useRef<HandLandmarker | null>(null);
  const segmenterRef = useRef<ImageSegmenter | null>(null);
  const cloneSpriteRef = useRef<HTMLCanvasElement | null>(null);
  const cloneSpriteReadyRef = useRef(false);
  const faceTrackerRef = useRef<FaceLandmarker | null>(null);
  const sharinganRef = useRef(false);
  const smoothedLandmarksRef = useRef<Point[][]>([]);
  const smoothedHandednessRef = useRef<string[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const activeRef = useRef<Jutsu>('none');
  const audioRef = useRef<AudioContext | null>(null);
  const jutsuStartedAtRef = useRef(0);
  const castUntilRef = useRef(0);
  const cloneVisualUntilRef = useRef(0);
  const soundRef = useRef(false);
  const activeToolRef = useRef<NinjaTool>('none');
  const thrownToolsRef = useRef<ThrownTool[]>([]);
  const landmarksRef = useRef(true);
  const [cameraState, setCameraState] = useState<'idle'|'loading'|'ready'|'camera-only'|'error'>('idle');
  const [error, setError] = useState('');
  const [activeJutsu, setActiveJutsu] = useState<Jutsu>('none');
  const [activeTool, setActiveTool] = useState<NinjaTool>('none');
  const [chargeLevel, setChargeLevel] = useState(0);
  const [gestureLabel, setGestureLabel] = useState('Waiting for hands');
  const [handCount, setHandCount] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [fps, setFps] = useState(0);
  const [cloneStatus, setCloneStatus] = useState<'idle'|'loading'|'ready'|'error'>('idle');
  const [faceTrackingState, setFaceTrackingState] = useState<'loading'|'ready'|'unavailable'>('loading');
  const [sharinganActive, setSharinganActive] = useState(false);
  const [sequenceType, setSequenceType] = useState<SequenceKind>('chidori');
  const [comboMode, setComboMode] = useState(false);
  const [comboProgress, setComboProgress] = useState(0);
  const [comboHoldProgress, setComboHoldProgress] = useState(0);
  const [comboFeedback, setComboFeedback] = useState('Start Chidori, bring both fists close together, then thrust one fist forward while pulling the other back.');
  const comboModeRef = useRef(false);
  const sequenceTypeRef = useRef<SequenceKind>('chidori');
  const comboProgressRef = useRef(0);
  const comboCandidateRef = useRef<Gesture>('none');
  const comboCandidateAtRef = useRef(0);
  const comboStepStartedAtRef = useRef(0);
  const comboProgressUiAtRef = useRef(0);
  useEffect(() => { soundRef.current=soundEnabled; if(soundEnabled)try{const a=audioRef.current??new window.AudioContext();audioRef.current=a;void a.resume();}catch{/* optional */} }, [soundEnabled]);
  useEffect(() => { landmarksRef.current = showLandmarks; }, [showLandmarks]);
  useEffect(() => {
    if (activeJutsu === 'none') { setChargeLevel(0); return; }
    const tick = () => setChargeLevel(Math.min(100, Math.round(Math.max(0, performance.now() - jutsuStartedAtRef.current) / 2600 * 100)));
    tick();
    const timer = window.setInterval(tick, 90);
    return () => window.clearInterval(timer);
  }, [activeJutsu]);

  const switchJutsu = useCallback((next: Jutsu) => {
    if(activeRef.current===next)return; activeRef.current=next;setActiveJutsu(next);
    if(next!=='none') {
      jutsuStartedAtRef.current=performance.now();
      // Reuse the cinematic impact idea: a brief camera-stage flash/shake on every cast.
      const stage = stageRef.current;
      if (stage) {
        stage.classList.remove('jutsu-impact');
        void stage.offsetWidth;
        stage.classList.add('jutsu-impact');
        window.setTimeout(() => stage.classList.remove('jutsu-impact'), 420);
      }
    }
    if(next!=='none'&&soundRef.current)try{
      const audio=audioRef.current??new window.AudioContext();audioRef.current=audio;void audio.resume();playJutsuCue(audio,next);
    }catch{/* optional audio */}
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    trackerRef.current?.close(); trackerRef.current = null;
    faceTrackerRef.current?.close(); faceTrackerRef.current = null;
    segmenterRef.current?.close(); segmenterRef.current = null;
    cloneSpriteRef.current = null; cloneSpriteReadyRef.current = false;
    smoothedLandmarksRef.current = []; smoothedHandednessRef.current = [];
    sharinganRef.current = false; setSharinganActive(false); setFaceTrackingState('loading');
    setCloneStatus('idle');
    comboModeRef.current = false; setComboMode(false); comboCandidateRef.current = 'none';
    castUntilRef.current = 0; cloneVisualUntilRef.current = 0;
    if (videoRef.current) videoRef.current.srcObject = null;
    switchJutsu('none'); setCameraState('idle'); setHandCount(0); setGestureLabel('Camera stopped');
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) ctx.clearRect(0,0,canvasRef.current.width,canvasRef.current.height);
  }, [switchJutsu]);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setError('This browser does not expose camera access. Use a current browser on localhost or HTTPS.');
      return;
    }
    stopCamera(); setCameraState('loading'); setError(''); setGestureLabel('Loading hand-tracking model…');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } }
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Video preview is not available. Refresh and try again.');
      video.srcObject = stream;
      await new Promise<void>((resolve) => {
        if (video.readyState >= 2) resolve();
        else video.onloadedmetadata = () => resolve();
      });
      await video.play();
      if (stageRef.current && video.videoWidth && video.videoHeight) stageRef.current.style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
      setGestureLabel('Camera is on · loading hand tracking from available mirrors…');
      const { tracker, vision } = await createHandTracker();
      trackerRef.current = tracker;
      const cameraStream = stream; setFaceTrackingState('loading');
      void createFaceTracker(vision).then((faceTracker) => { if (streamRef.current !== cameraStream) { faceTracker.close(); return; } faceTrackerRef.current = faceTracker; setFaceTrackingState('ready'); }).catch(() => setFaceTrackingState('unavailable'));
      setCameraState('ready');
      setGestureLabel('Rasengan: open palm · Chidori: two closed-fist moves · Fireball: two fists apart · Amaterasu: one fist');
      let segmenterLoading = false, segmenterFailed = false, personMaskIndex = 0;
      let lastSegmentationAt = 0, lastSegmentationVideoTime = -1;
      const cloneSprite = document.createElement('canvas');
      cloneSpriteRef.current = cloneSprite; cloneSpriteReadyRef.current = false;
      setCloneStatus('idle');
      const loadSegmenter = async () => {
        if (segmenterLoading || segmenterRef.current || segmenterFailed) return;
        segmenterLoading = true;
        setCloneStatus('loading');
        try {
          let loaded: ImageSegmenter;
          try {
            loaded = await ImageSegmenter.createFromOptions(vision, {
              baseOptions: { modelAssetPath: SEGMENTER_MODEL_URL, delegate: 'GPU' },
              runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
            });
          } catch {
            loaded = await ImageSegmenter.createFromOptions(vision, {
              baseOptions: { modelAssetPath: SEGMENTER_MODEL_URL, delegate: 'CPU' },
              runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
            });
          }
          segmenterRef.current = loaded;
          const labels = loaded.getLabels();
          const personLabel = labels.findIndex((label) => /person|foreground|human/i.test(label));
          const nonBackgroundLabel = labels.findIndex((label) => !/background/i.test(label));
          // Some model builds label the background last; never assume the final mask is the person.
          personMaskIndex = personLabel >= 0 ? personLabel : nonBackgroundLabel >= 0 ? nonBackgroundLabel : 0;
          setCloneStatus('ready');
        } catch {
          segmenterFailed = true;
          setCloneStatus('error');
          setGestureLabel('Shadow Clone cutout unavailable; using the visual fallback');
        } finally {
          segmenterLoading = false;
        }
      };
      let lastVideoTime = -1, lastWrist: Point | null = null, lastWristAt = 0, lastToolThrowAt = 0, lastProjectileFrameAt = performance.now(), lastFaceDetectAt = 0, move = {x:.2,y:-.05};
      let candidate: Jutsu = 'none', candidateAt = 0, noPoseAt = 0, fpsAt = performance.now(), frames = 0;
      let latest: HandLandmarkerResult | null = null, latestFace: FaceLandmarkerResult | null = null;
      let eyesClosed = false, eyeClosedAt = 0, lastBlinkAt = 0;
      const energyTrail: Array<{ x: number; y: number; at: number }> = [];
      const frame = () => {
        const v = videoRef.current, canvas = canvasRef.current, model = trackerRef.current;
        if (!v || !canvas || !model || v.readyState < 2) { rafRef.current = requestAnimationFrame(frame); return; }
        const w = v.videoWidth || 1280, h = v.videoHeight || 720;
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        const now = performance.now();
        if (v.currentTime !== lastVideoTime) {
          lastVideoTime = v.currentTime; latest = model.detectForVideo(v, now);
          if (faceTrackerRef.current && now - lastFaceDetectAt > 42) { lastFaceDetectAt = now; try { const fr = faceTrackerRef.current.detectForVideo(v, now); latestFace = fr; const cats = fr.faceBlendshapes?.[0]?.categories ?? []; const left = cats.find((c) => c.categoryName === 'eyeBlinkLeft')?.score ?? 0, right = cats.find((c) => c.categoryName === 'eyeBlinkRight')?.score ?? 0, closed = left > .48 && right > .48; if (closed && !eyesClosed) eyeClosedAt = now; if (!closed && eyesClosed) { const duration = now - eyeClosedAt; if (duration >= 55 && duration <= 720 && now - lastBlinkAt > 950) { lastBlinkAt = now; sharinganRef.current = !sharinganRef.current; setSharinganActive(sharinganRef.current); setGestureLabel(sharinganRef.current ? 'Sharingan awakened · blink again to seal the eye' : 'Sharingan sealed · blink to awaken'); } } eyesClosed = closed; } catch { faceTrackerRef.current?.close(); faceTrackerRef.current = null; setFaceTrackingState('unavailable'); } }
          const rawHands = (latest.landmarks ?? []) as Point[][];
          const previousHands = smoothedLandmarksRef.current;
          const oldHandedness = smoothedHandednessRef.current;
          const currentHandedness = latest.handedness?.map((entry, index) => entry[0]?.categoryName ?? `Unknown-${index}`) ?? rawHands.map((_, index) => `Unknown-${index}`);
          // Match left/right hands by handedness to prevent sudden smoothing jumps.
          smoothedLandmarksRef.current = rawHands.map((hand, handIndex) => {
            const priorIndex = oldHandedness.indexOf(currentHandedness[handIndex]);
            const previous = previousHands[priorIndex >= 0 ? priorIndex : handIndex];
            if (!previous || previous.length !== hand.length) return hand.map((point) => ({ ...point }));
            return hand.map((point, pointIndex) => {
              const prior = previous[pointIndex];
              const speed = Math.hypot(point.x - prior.x, point.y - prior.y);
              // Smooth slow jitter, but let fast gestures respond quickly enough for throwing.
              const alpha = Math.min(0.88, 0.28 + speed * 7.5);
              return {
                x: prior.x + (point.x - prior.x) * alpha,
                y: prior.y + (point.y - prior.y) * alpha,
                z: prior.z + (point.z - prior.z) * alpha,
              };
            });
          });
          smoothedHandednessRef.current = currentHandedness;
          const detected = readJutsu(latest), count = rawHands.length;
          setHandCount(count);
          if (!comboModeRef.current && activeRef.current !== 'none') {
            // Lock the current technique while either hand remains in frame. Intermediate poses cannot trigger a different jutsu.
            candidate = 'none';
            if (count > 0) {
              noPoseAt = 0;
              if (activeRef.current === 'clone' && detected.gesture === 'clone-seal') cloneVisualUntilRef.current = now + 6000;
              setGestureLabel(`${INFO[activeRef.current].name} active · move both hands out of frame to switch techniques`);
            } else {
              if (!noPoseAt) noPoseAt = now;
              const cloneStillVisible = activeRef.current === 'clone' && now < cloneVisualUntilRef.current;
              if (cloneStillVisible) setGestureLabel('Shadow Clone · afterimages linger for six seconds');
              else if (now > castUntilRef.current && now - noPoseAt > 650) {
                switchJutsu('none'); castUntilRef.current = 0; noPoseAt = 0;
                setGestureLabel('Hands out of frame · ready for the next jutsu');
              }
            }
          } else if (!comboModeRef.current && activeRef.current === 'chidori' && now < castUntilRef.current) {
            candidate = 'none'; noPoseAt = 0;
            setGestureLabel('Chidori released · lightning surrounds your hand!');
          } else if (comboModeRef.current) {
            const steps = sequenceTypeRef.current === 'fireball' ? FIREBALL_STEPS : COMBO_STEPS;
            let expected = steps[comboProgressRef.current], timedOut = false;
            candidate = 'none'; noPoseAt = 0;
            if (comboStepStartedAtRef.current && now - comboStepStartedAtRef.current > 6500) {
              comboProgressRef.current = 0; comboCandidateRef.current = 'none'; comboCandidateAtRef.current = 0;
              comboStepStartedAtRef.current = now; setComboProgress(0); setComboHoldProgress(0);
              setComboFeedback('Sequence timed out. Restart from the first hand sign.');
              setGestureLabel('Technique sequence timed out · restart from step 1');
              expected = steps[0]; timedOut = true;
            }
            if (!timedOut && expected && detected.gesture === expected.gesture) {
              if (comboCandidateRef.current !== expected.gesture) {
                comboCandidateRef.current = expected.gesture; comboCandidateAtRef.current = now;
                setComboHoldProgress(0); setComboFeedback(`Hold ${expected.title} steady…`);
              }
              const heldFor = now - comboCandidateAtRef.current;
              if (now - comboProgressUiAtRef.current > 70) {
                setComboHoldProgress(Math.min(100, heldFor / 500 * 100)); comboProgressUiAtRef.current = now;
              }
              if (heldFor >= 500) {
                const nextProgress = comboProgressRef.current + 1;
                comboProgressRef.current = nextProgress; setComboProgress(nextProgress);
                comboCandidateRef.current = 'none'; comboCandidateAtRef.current = 0; setComboHoldProgress(0);
                comboStepStartedAtRef.current = now;
                if (nextProgress >= steps.length) {
                  const completedJutsu = sequenceTypeRef.current;
                  comboModeRef.current = false; setComboMode(false);
                  setComboFeedback(completedJutsu === 'fireball' ? 'Uchiha Fire released! Flames erupt from your mouth.' : 'Chidori released! Lightning gathers around your hand.');
                  setGestureLabel(completedJutsu === 'fireball' ? 'Uchiha Fire sequence complete · Weave → Focus → Release' : 'Chidori sequence complete · Charge → Release');
                  castUntilRef.current = now + 2800; switchJutsu(completedJutsu);
                } else {
                  setComboFeedback(`Step ${nextProgress} complete. Next: ${steps[nextProgress].instruction}.`);
                  setGestureLabel(`Technique sequence · step ${nextProgress + 1} of ${steps.length}`);
                }
              }
            } else if (!timedOut) {
              comboCandidateRef.current = 'none'; comboCandidateAtRef.current = 0; setComboHoldProgress(0);
              if (detected.gesture !== 'none' && expected) setComboFeedback(`Keep going — step ${comboProgressRef.current + 1}: ${expected.instruction}.`);
              setGestureLabel(expected ? `Technique sequence · step ${comboProgressRef.current + 1} of ${steps.length}` : 'Finishing sequence…');
            }
          } else if (detected.jutsu !== 'none') {
            noPoseAt = 0;
            if (candidate !== detected.jutsu) { candidate = detected.jutsu; candidateAt = now; }
            if (now - candidateAt > 420) {
              if (detected.gesture === 'clone-seal') { castUntilRef.current = now + 6000; cloneVisualUntilRef.current = now + 6000; }
              switchJutsu(detected.jutsu);
              setGestureLabel(
                detected.gesture === 'clone-seal'
                  ? 'Shadow Clone · clones stay for six seconds'
                  : detected.gesture === 'two-open'
                    ? 'Raiton Pulse · two palms detected'
                    : detected.gesture === 'open'
                      ? 'Open palm detected'
                      : detected.gesture === 'hare-seal'
                        ? 'Hare seal · Wind Fang activated'
                        : 'Fist detected',
              );
            } else setGestureLabel('Hold the pose steady…');
          } else {
            candidate = 'none'; candidateAt = now; if (!noPoseAt) noPoseAt = now;
            setGestureLabel(count ? 'Pose detected · show your palm to cast Rasengan, or start the two-move Chidori sequence' : 'Move your hands into the frame');
            if (now > castUntilRef.current && now - noPoseAt > 700) switchJutsu('none');
          }
          const first = latest.landmarks?.[0] as Point[] | undefined;
          if (first) {
            if (lastWrist) {
              move = {x:first[0].x-lastWrist.x,y:first[0].y-lastWrist.y};
              const elapsed = Math.max(12, now - lastWristAt);
              const vx = move.x / elapsed * 1000, vy = move.y / elapsed * 1000;
              const speed = Math.hypot(vx, vy);
              if (activeToolRef.current !== 'none' && speed > 0.95 && now - lastToolThrowAt > 430) {
                const launch = palm(smoothedLandmarksRef.current[0] ?? first);
                const scale = Math.min(1.75, Math.max(.7, speed * .72));
                thrownToolsRef.current.push({
                  kind: activeToolRef.current, x: launch.x, y: launch.y,
                  vx: vx * scale, vy: vy * scale, born: now, spin: Math.random() * Math.PI * 2,
                });
                if (thrownToolsRef.current.length > 12) thrownToolsRef.current.shift();
                lastToolThrowAt = now;
                setGestureLabel(activeToolRef.current === 'shuriken' ? 'Shuriken thrown · make another quick motion to throw again' : 'Kunai thrown · make another quick motion to throw again');
                if (soundRef.current) {
                  try {
                    const audio = audioRef.current ?? new window.AudioContext(); audioRef.current = audio; void audio.resume();
                    const oscillator = audio.createOscillator(), gain = audio.createGain();
                    oscillator.type = 'triangle'; oscillator.frequency.setValueAtTime(820, audio.currentTime);
                    oscillator.frequency.exponentialRampToValueAtTime(240, audio.currentTime + .11);
                    gain.gain.setValueAtTime(.0001, audio.currentTime); gain.gain.exponentialRampToValueAtTime(.12, audio.currentTime + .012);
                    gain.gain.exponentialRampToValueAtTime(.0001, audio.currentTime + .12);
                    oscillator.connect(gain); gain.connect(audio.destination); oscillator.start(); oscillator.stop(audio.currentTime + .13);
                  } catch { /* optional tool throw cue */ }
                }
              }
            }
            lastWrist = { ...first[0] }; lastWristAt = now;
          } else { lastWrist = null; lastWristAt = 0; }
        }
        if (activeRef.current === 'clone') {
          if (!segmenterRef.current && !segmenterLoading && !segmenterFailed) void loadSegmenter();
          if (segmenterRef.current && v.currentTime !== lastSegmentationVideoTime && now - lastSegmentationAt > 125) {
            lastSegmentationVideoTime = v.currentTime;
            lastSegmentationAt = now;
            try {
              segmenterRef.current.segmentForVideo(v, now, (result) => {
                const masks = result.confidenceMasks ?? [];
                const personMask = masks[personMaskIndex] ?? masks[0];
                if (personMask && cloneSpriteRef.current) {
                  updateCloneSprite(v, personMask, cloneSpriteRef.current);
                  cloneSpriteReadyRef.current = true;
                }
              });
            } catch {
              setCloneStatus('error');
            }
          }
        }
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0,0,w,h);
          const hands = smoothedLandmarksRef.current;
          if (landmarksRef.current) drawLandmarks(ctx,hands,w,h);
          const charge = Math.min(1, Math.max(0, (now - jutsuStartedAtRef.current) / 2600));
          if (activeRef.current !== 'none' && hands.length) {
            const focus = activeRef.current === 'lightning-burst' && hands.length > 1
              ? { x: (palm(hands[0]).x + palm(hands[1]).x) / 2, y: (palm(hands[0]).y + palm(hands[1]).y) / 2 }
              : palm(hands[0]);
            const previous = energyTrail[energyTrail.length - 1];
            if (!previous || Math.hypot(focus.x - previous.x, focus.y - previous.y) > 0.0025 || now - previous.at > 45) {
              energyTrail.push({ ...focus, at: now });
            }
          }
          while (energyTrail.length && now - energyTrail[0].at > 520) energyTrail.shift();
          const trailColor = activeRef.current === 'fireball' ? 'rgba(255,150,48,.95)'
            : activeRef.current === 'amaterasu' ? 'rgba(166,119,255,.95)'
            : activeRef.current === 'wind-scythe' ? 'rgba(104,255,190,.96)'
            : activeRef.current === 'clone' ? 'rgba(199,165,255,.92)'
            : activeRef.current === 'chidori' || activeRef.current === 'lightning-burst' ? 'rgba(115,222,255,.98)'
            : 'rgba(70,190,255,.95)';
          if (activeRef.current !== 'none') drawEnergyTrail(ctx, energyTrail, w, h, now, trailColor);
          else energyTrail.length = 0;
          if (activeRef.current === 'rasengan' && hands.length) drawRasengan(ctx,palm(hands[0]),w,h,now/1000,charge);
          if (activeRef.current === 'chidori' && hands.length) drawChidori(ctx,palm(hands[0]),move,w,h,now/1000,charge,hands[0]);
          if (activeRef.current === 'lightning-burst' && hands.length > 1) drawLightningBurst(ctx,{x:(palm(hands[0]).x+palm(hands[1]).x)/2,y:(palm(hands[0]).y+palm(hands[1]).y)/2},w,h,now/1000,charge);
          if (activeRef.current === 'fireball') { const face = latestFace?.faceLandmarks?.[0] as Point[] | undefined; if (face && face.length >= 292) { const target = drawUchihaMouthFire(ctx, face, w, h, now / 1000, charge); drawFireball(ctx, target, w, h, now / 1000, charge); } else { drawFireball(ctx, hands.length > 1 ? {x:(palm(hands[0]).x+palm(hands[1]).x)/2,y:(palm(hands[0]).y+palm(hands[1]).y)/2} : {x:.5,y:.5},w,h,now/1000,charge); } }
          if (activeRef.current === 'clone' && now < cloneVisualUntilRef.current) {
            const cloneFade = Math.min(1, Math.max(0, (cloneVisualUntilRef.current - now) / 550));
            ctx.save(); ctx.globalAlpha = cloneFade;
            if (cloneSpriteReadyRef.current && cloneSpriteRef.current) drawSegmentedClones(ctx,cloneSpriteRef.current,w,h,now/1000);
            else drawCloneEcho(ctx,w,h,now/1000);
            ctx.restore();
          }
          if (activeRef.current === 'amaterasu' && hands.length) drawAmaterasu(ctx,palm(hands[0]),w,h,now/1000,charge);
          if (activeRef.current === 'wind-scythe' && hands.length) drawWindScythe(ctx,palm(hands[0]),move,w,h,now/1000,charge);
          const facePoints = latestFace?.faceLandmarks?.[0] as Point[] | undefined;
          if (sharinganRef.current && facePoints && facePoints.length >= 478) drawSharinganEyes(ctx, facePoints, w, h, now / 1000);
          if (activeToolRef.current !== 'none' && hands.length) drawHeldTool(ctx, hands[0], activeToolRef.current, w, h, now / 1000);
          const dt = Math.min(.05, Math.max(0, (now - lastProjectileFrameAt) / 1000));
          lastProjectileFrameAt = now;
          thrownToolsRef.current.forEach((projectile) => {
            projectile.x += projectile.vx * dt;
            projectile.y += projectile.vy * dt;
            projectile.vy += .18 * dt;
          });
          thrownToolsRef.current = thrownToolsRef.current.filter((projectile) =>
            now - projectile.born < 1800 && projectile.x > -.2 && projectile.x < 1.2 && projectile.y > -.2 && projectile.y < 1.2
          );
          drawThrownTools(ctx, thrownToolsRef.current, w, h, now);
        }
        frames++;
        if (now - fpsAt >= 1000) { setFps(Math.round(frames*1000/(now-fpsAt))); fpsAt=now; frames=0; }
        rafRef.current = requestAnimationFrame(frame);
      };
      rafRef.current = requestAnimationFrame(frame);
    } catch (cause) {
      const message = describeFailure(cause);
      if (message.includes('Hand tracking could not initialize')) {
        // Keep the camera preview available for diagnostics; a model download failure should not turn the camera itself off.
        setCameraState('camera-only');
        setHandCount(0);
        setGestureLabel('Camera is active, but hand tracking did not initialize.');
        setError('The webcam is working, but hand tracking could not start. This build now loads its model and WASM from the same site first. Reload once; if this remains, share the diagnostic details below. Details: ' + message);
        return;
      }

      stopCamera();
      setCameraState('error');
      setError(
        message.includes('NotAllowedError')
          ? 'Camera permission was blocked. Allow camera access in browser settings, then try again.'
          : message.includes('NotFoundError')
            ? 'No camera was found. Connect a webcam and try again.'
            : 'Could not start the camera. Check browser permission and camera availability. Details: ' + message,
      );
    }
  }, [stopCamera, switchJutsu]);

  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    trackerRef.current?.close();
    segmenterRef.current?.close();
    audioRef.current?.close().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (cameraState === 'ready' || activeJutsu === 'none' || !canvasRef.current) return;
    const canvas = canvasRef.current, rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * window.devicePixelRatio));
    const h = Math.max(1, Math.round(rect.height * window.devicePixelRatio));
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    let id = 0;
    const draw = (t: number) => {
      ctx.clearRect(0,0,w,h);
      if (activeJutsu === 'rasengan') drawRasengan(ctx,{x:.5,y:.52},w,h,t/1000);
      if (activeJutsu === 'chidori') drawChidori(ctx,{x:.5,y:.52},{x:.8,y:-.2},w,h,t/1000);
      if (activeJutsu === 'clone') drawCloneEcho(ctx,w,h,t/1000);
      id = requestAnimationFrame(draw);
    };
    id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [cameraState, activeJutsu]);

  const preview = (jutsu: Exclude<Jutsu, 'none'>) => {
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    switchJutsu(jutsu);
    setGestureLabel('Preview mode · ' + INFO[jutsu].name);
    timeoutRef.current = window.setTimeout(() => {
      if (activeRef.current === jutsu) switchJutsu('none');
      setGestureLabel(cameraState === 'ready' ? 'Show a gesture to cast' : 'Preview complete · start camera for tracking');
    }, 2200);
  };
  const activeInfo = activeJutsu === 'none' ? null : INFO[activeJutsu];
  const displayedSequenceSteps = sequenceType === 'fireball' ? FIREBALL_STEPS : COMBO_STEPS;
  const toggleCombo = (requestedType: SequenceKind = sequenceTypeRef.current) => {
    if (comboModeRef.current) {
      comboModeRef.current = false; setComboMode(false); comboCandidateRef.current = 'none';
      setComboFeedback('Sequence cancelled. Start again whenever you are ready.');
      return;
    }
    if (cameraState !== 'ready') {
      setComboFeedback('Start the camera before beginning a live technique sequence.');
      return;
    }
    sequenceTypeRef.current = requestedType; setSequenceType(requestedType);
    comboModeRef.current = true; comboProgressRef.current = 0;
    comboCandidateRef.current = 'none'; comboCandidateAtRef.current = 0;
    comboStepStartedAtRef.current = performance.now();
    setComboProgress(0); setComboHoldProgress(0); setComboMode(true); switchJutsu('none');
    setComboFeedback(requestedType === 'fireball' ? `Step 1: ${FIREBALL_STEPS[0].instruction}.` : `Step 1: ${COMBO_STEPS[0].instruction}.`);
    setGestureLabel(requestedType === 'fireball' ? 'Uchiha Fire sequence · step 1 of 3' : 'Chidori sequence · step 1 of 2');
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Konoha Jutsu Lab home">
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div><strong>KONOHA <span>JUTSU LAB</span></strong><small>CAMERA-TRACKED CHAKRA PROTOTYPE</small></div>
        </a>
        <div className="topbar-right"><div className="topbar-portraits" aria-label="Featured shinobi"><img src="https://wallpapers.com/images/hd/naruto-blue-sky-ready-for-battle-w7rfnmhxkzu441oh.jpg" alt="Naruto Uzumaki anime art" title="Naruto Uzumaki" /><img src="https://wallpapers.com/images/hd/sensei-kakashi-hatake-png-jja-oqkhqqgluqkoq0q6.png" alt="Kakashi Hatake anime art" title="Kakashi Hatake" /></div><div className="build-chip"><span className="status-dot" /> PROTOTYPE / 0.5</div>
          <button className="icon-button" onClick={() => setHelpOpen((v) => !v)} aria-label="Toggle instructions"><CircleHelp size={18} /></button>
        </div>
      </header>
      <section className="intro-row" id="top">
        <div><div className="eyebrow"><span className="eyebrow-line" /> THE HIDDEN LEAF, IN YOUR BROWSER</div>
          <h1>Make your hands<br /><span>feel like jutsu.</span></h1>
          <p className="intro-copy">A first step toward real-time hand seals and chakra effects — powered by your webcam, not a keyboard combo.</p>
        </div>
        <div className="intro-stats">
          <div className="stat-item"><span className="stat-icon"><Hand size={17} /></span><div><b>2 hands</b><small>tracked live</small></div></div>
          <div className="stat-item"><span className="stat-icon violet"><Cpu size={17} /></span><div><b>On-device</b><small>pose inference</small></div></div>
          <div className="stat-item"><span className="stat-icon green"><ShieldCheck size={17} /></span><div><b>No upload</b><small>video stays local</small></div></div>
        </div>
      </section>
      {helpOpen && <section className="help-panel"><Info size={18} /><div><strong>Quick start · How to cast</strong><p><b>1.</b> Start the camera. <b>2. Rasengan:</b> show one open palm. <b>3. Chidori:</b> press Start Chidori, bring both closed fists close together in front of your upper chest, then thrust one fist toward the camera while pulling the other back by your shoulder. <b>4. Uchiha Fire:</b> use the guided sequence: hare sign, fists close to focus chakra, then separate them to release fire from your mouth. <b>5. Amaterasu:</b> show one closed fist. <b>6. Shadow Clone:</b> cross the index/middle fingers of both hands into a plus (+) shape. <b>7. Wind Fang:</b> raise the index and middle fingers of one hand while curling the ring and little fingers. Move that hand to sweep the green wind blade. Once a jutsu activates, it stays locked while your hands remain in frame; move both hands fully out of frame to reset. Clones linger for six seconds after the seal is released. Video is processed in your browser.</p></div><button className="text-button" onClick={() => setHelpOpen(false)}>Got it</button></section>}
      <section className="workspace">
        <div className="camera-column">
          <div className="panel-heading"><div><span className="section-kicker">01 / LIVE INPUT</span><h2>Chakra viewport</h2></div>
            <div className={'live-status ' + (cameraState === 'ready' || cameraState === 'camera-only' ? 'is-live' : cameraState === 'error' ? 'is-error' : '')}><span />{cameraState === 'ready' ? 'LIVE TRACKING' : cameraState === 'camera-only' ? 'CAMERA ON · TRACKING OFF' : cameraState === 'loading' ? 'INITIALIZING' : cameraState === 'error' ? 'CAMERA ERROR' : 'STANDBY'}</div>
          </div>
          <div ref={stageRef} className={'camera-stage ' + (activeInfo ? 'effect-' + activeInfo.color : '')}>
            <div className="stage-grid" />{sharinganActive && <div className="sharingan-badge"><Eye size={12} /> SHARINGAN AWAKENED <small>BLINK TO SEAL</small></div>}<div className="stage-corner corner-tl" /><div className="stage-corner corner-tr" /><div className="stage-corner corner-bl" /><div className="stage-corner corner-br" />
            <div className="stage-topline"><span><Activity size={13} /> INPUT FEED</span><span>{cameraState === 'ready' ? (fps || '—') + ' FPS' : '1280 × 720 TARGET'}</span></div>
            {activeJutsu !== 'none' && <div className="chakra-charge-hud" aria-live="polite">
              <div className="charge-hud-top"><span><Zap size={12} /> CHAKRA OUTPUT</span><strong>{chargeLevel}%</strong></div>
              <div className="charge-hud-track"><span style={{ width: chargeLevel + '%' }} /></div>
              <small>{chargeLevel < 35 ? 'GATHERING ENERGY' : chargeLevel < 78 ? 'CHAKRA RESONANCE' : 'MAXIMUM OUTPUT'}</small>
            </div>}
            <div className="video-wrap">
              <video ref={videoRef} className="camera-video" autoPlay playsInline muted aria-label="Mirrored webcam preview" />
              <canvas ref={canvasRef} className="effects-canvas" />
              {cameraState !== 'ready' && cameraState !== 'camera-only' && <div className="camera-placeholder">
                {cameraState === 'loading' ? <LoaderCircle className="spin" size={38} /> : cameraState === 'error' ? <Camera size={35} /> : <div className="camera-orb"><Camera size={28} /></div>}
                <strong>{cameraState === 'loading' ? 'Warming up the chakra flow' : cameraState === 'error' ? (error.includes('hand tracking') ? 'Hand tracking unavailable' : 'Camera not ready') : 'Your training space awaits'}</strong>
                <span>{cameraState === 'loading' ? 'Requesting camera and loading hand tracking…' : cameraState === 'error' ? error : 'Start the camera to see your hands and effects live.'}</span>
              </div>}
            </div>
            <div className="stage-bottomline"><span><span className={'small-dot ' + (cameraState === 'ready' ? 'active' : '')} /> {cameraState === 'ready' ? handCount + ' HAND' + (handCount === 1 ? '' : 'S') + ' DETECTED' : cameraState === 'camera-only' ? 'CAMERA ON · TRACKING UNAVAILABLE' : 'AWAITING VIDEO SOURCE'}</span><span className="mono">MP / HAND LANDMARKER</span></div>
          </div>
          <div className="camera-controls">
            {cameraState === 'ready' || cameraState === 'camera-only' ? <button className="primary-button stop-button" onClick={stopCamera}><span className="stop-square" /> Stop camera</button> :
              <button className="primary-button" onClick={startCamera} disabled={cameraState === 'loading'}>{cameraState === 'loading' ? <LoaderCircle className="spin" size={17} /> : <Camera size={17} />}{cameraState === 'loading' ? 'Starting camera…' : 'Start camera'}</button>}
            <button className={'secondary-button ' + (showLandmarks ? 'selected' : '')} onClick={() => setShowLandmarks((v) => !v)}><Hand size={16} /> Landmark overlay <span className="toggle-pill">{showLandmarks ? 'ON' : 'OFF'}</span></button>
            <button className={'secondary-button ' + (soundEnabled ? 'selected' : '')} onClick={() => { const next = !soundEnabled; setSoundEnabled(next); if (next) { try { const audio = audioRef.current ?? new window.AudioContext(); audioRef.current = audio; void audio.resume().then(() => playJutsuCue(audio, 'chidori')); } catch { /* optional audio */ } } }} title="Tap to turn on and hear a sample; tap again to turn off"><Zap size={16} /> Jutsu sound · test <span className="toggle-pill">{soundEnabled ? 'ON' : 'OFF'}</span></button>
            <button className="reset-button" onClick={() => { switchJutsu('none'); setGestureLabel('Effects cleared'); }} title="Clear current effect"><RotateCcw size={16} /></button>
          </div>
          <div className="ninja-tool-controls" aria-label="Ninja tool selection">
            <div className="tool-controls-heading"><span className="section-kicker">HAND-HELD TOOLS</span><small>Select a tool, then make a quick throwing motion</small></div>
            <div className="tool-buttons">
              <button className={'tool-button shuriken-tool ' + (activeTool === 'shuriken' ? 'is-selected' : '')} onClick={() => {
                const next = activeTool === 'shuriken' ? 'none' : 'shuriken'; activeToolRef.current = next; setActiveTool(next);
                setGestureLabel(next === 'shuriken' ? 'Shuriken ready · move your hand quickly to throw' : 'Tool cleared');
              }} aria-pressed={activeTool === 'shuriken'}>
                <span className="tool-symbol shuriken-symbol">✳</span><span><strong>Shuriken</strong><small>{activeTool === 'shuriken' ? 'READY · quick hand motion throws' : 'Select & throw'}</small></span>
              </button>
              <button className={'tool-button kunai-tool ' + (activeTool === 'kunai' ? 'is-selected' : '')} onClick={() => {
                const next = activeTool === 'kunai' ? 'none' : 'kunai'; activeToolRef.current = next; setActiveTool(next);
                setGestureLabel(next === 'kunai' ? 'Kunai ready · move your hand quickly to throw' : 'Tool cleared');
              }} aria-pressed={activeTool === 'kunai'}>
                <span className="tool-symbol kunai-symbol">➤</span><span><strong>Kunai</strong><small>{activeTool === 'kunai' ? 'READY · quick hand motion throws' : 'Select & throw'}</small></span>
              </button>
              <button className="tool-clear-button" onClick={() => { activeToolRef.current = 'none'; setActiveTool('none'); setGestureLabel('Tool cleared'); }}>Clear</button>
            </div>
          </div>
          {(cameraState === 'error' || cameraState === 'camera-only') && <p className="error-hint"><Info size={14} /> {cameraState === 'camera-only' ? 'The camera preview is being kept active. Hand tracking is unavailable until the model/runtime loads; use the GitHub Actions link to check that the newest build was deployed.' : 'Camera access requires browser permission and a secure context (localhost or HTTPS).'}</p>}
        </div>
        <aside className="right-column">
          <div className="panel-heading"><div><span className="section-kicker">02 / TECHNIQUE SELECTOR</span><h2>Choose your jutsu</h2></div><span className="mini-count">06</span></div>
          <div className="jutsu-list">
            <button className={'jutsu-card blue-card ' + (activeJutsu === 'rasengan' ? 'is-active' : '')} onClick={() => preview('rasengan')}>
              <div className="jutsu-art rasengan-art"><span className="orb-core" /><span className="orb-ring ring-one" /><span className="orb-ring ring-two" /><span className="orb-spark s1" /><span className="orb-spark s2" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">WIND / CHAKRA</div><h3>Rasengan</h3><p>Show your palm, not the back of your hand · the orb grows into a spinning chakra sphere</p><div className="jutsu-foot"><span className="difficulty easy">BEGINNER</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">01</span>
            </button>
            <button className={'jutsu-card cyan-card ' + (activeJutsu === 'chidori' ? 'is-active' : '')} onClick={() => preview('chidori')}>
              <div className="jutsu-art lightning-art"><Zap className="lightning-symbol" size={44} /><i className="bolt b1" /><i className="bolt b2" /><i className="bolt b3" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">LIGHTNING STYLE</div><h3>Chidori</h3><p>Two anime-style closed-fist moves · thrust one fist forward and pull the other back</p><div className="jutsu-foot"><span className="difficulty medium">INTERMEDIATE</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">02</span>
            </button>
            <button className={'jutsu-card violet-card ' + (activeJutsu === 'clone' ? 'is-active' : '')} onClick={() => preview('clone')}>
              <div className="jutsu-art clone-art"><span className="clone-figure figure-one" /><span className="clone-figure figure-two" /><span className="clone-figure figure-three" /><Sparkles className="clone-star" size={17} /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">NINJUTSU / DECOY</div><h3>Shadow Clone</h3><p>Cross index/middle fingers into a + seal; clones linger for six seconds</p><div className="jutsu-foot"><span className="difficulty advanced">CROSSED SEAL</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">03</span>
            </button>
            <button className={'jutsu-card fire-card ' + (activeJutsu === 'fireball' ? 'is-active' : '')} onClick={() => preview('fireball')}>
              <div className="jutsu-art fireball-art"><span className="fire-core" /><i className="fire-tongue f1" /><i className="fire-tongue f2" /><i className="fire-tongue f3" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">FIRE STYLE · KATON</div><h3>Fireball Jutsu</h3><p>Use the guided three-step seal sequence to breathe a flame stream from your mouth, followed by a swirling fireball. Two separated fists remain available as a quick-cast shortcut.</p><div className="jutsu-foot"><span className="difficulty medium">NEW TECHNIQUE</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">04</span>
            </button>
            <button className={'jutsu-card amaterasu-card ' + (activeJutsu === 'amaterasu' ? 'is-active' : '')} onClick={() => preview('amaterasu')}>
              <div className="jutsu-art amaterasu-art"><span className="black-flame flame-a" /><span className="black-flame flame-b" /><span className="black-flame flame-c" /><span className="amaterasu-core" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">MANGEKYŌ SHARINGAN</div><h3>Amaterasu</h3><p>Raise one closed fist to summon writhing black flames edged with violet chakra.</p><div className="jutsu-foot"><span className="difficulty advanced">ADVANCED</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">05</span>
            </button>
          </div>
            <button className={'jutsu-card wind-card ' + (activeJutsu === 'wind-scythe' ? 'is-active' : '')} onClick={() => preview('wind-scythe')}>
              <div className="jutsu-art wind-art"><span className="wind-ring wind-ring-a" /><span className="wind-ring wind-ring-b" /><i className="wind-blade blade-a" /><i className="wind-blade blade-b" /><Sparkles className="wind-star" size={19} /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">WIND STYLE · GALE</div><h3>Wind Fang</h3><p>Raise two fingers and sweep your hand to spin a bright green crescent slash with drifting wind particles.</p><div className="jutsu-foot"><span className="difficulty easy">NEW · HAND SIGN</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">06</span>
            </button>
          <div className="combo-card">
            <div className="combo-header">
              <div><span className="section-kicker">03 / TECHNIQUE SEQUENCES</span><h3>{sequenceType === 'fireball' ? 'Uchiha Fire · Three-step cast' : 'Chidori · Two-step cast'}</h3></div>
              <span className="combo-count">{comboProgress}/{displayedSequenceSteps.length}</span>
            </div>
            <div className="combo-steps">
              {displayedSequenceSteps.map((step, index) => (
                <div className={'combo-step ' + (index < comboProgress ? 'complete' : index === comboProgress && comboMode ? 'current' : '')} key={step.title}>
                  <span className="combo-step-index">{index < comboProgress ? <Check size={13} /> : String(index + 1).padStart(2, '0')}</span>
                  <div><strong>{step.title}</strong><small>{step.instruction}</small></div>
                </div>
              ))}
            </div>
            <div className="combo-hold-meter" aria-label={`Pose hold progress ${Math.round(comboHoldProgress)} percent`}><span style={{ width: `${comboHoldProgress}%` }} /></div>
            <p className="combo-feedback">{comboFeedback}</p>
            <div className="combo-actions">
              {comboMode ? <button className="secondary-button combo-cancel" onClick={() => toggleCombo()} disabled={cameraState === 'loading'}><RotateCcw size={15} /> Cancel sequence</button> : <>
                <button className="primary-button combo-start" onClick={() => toggleCombo('chidori')} disabled={cameraState === 'loading'}><Hand size={15} /> Start Chidori</button>
                <button className="secondary-button combo-fire-start" onClick={() => toggleCombo('fireball')} disabled={cameraState === 'loading'}><Flame size={15} /> Start Uchiha Fire</button>
              </>}
              <span>Hold each pose · 0.5 sec · 6.5 sec timeout</span>
            </div>
            <p className="combo-note">Chidori: bring both fists close, then thrust one forward. Uchiha Fire: form the hare sign, focus with both fists close, then separate them to release a mouth-origin flame stream and fireball.</p>
          </div>
          <div className="detection-card">
            <div className="detection-heading"><div><span className="section-kicker">LIVE READOUT</span><h3>Detection status</h3></div><Activity size={17} /></div>
            <div className="readout-row"><span>Current signal</span><strong className={activeJutsu !== 'none' ? 'signal-active' : ''}>{activeJutsu === 'none' ? 'No active jutsu' : INFO[activeJutsu].name}</strong></div>
            <div className="readout-row"><span>Face / eye tracking</span><strong>{faceTrackingState === "ready" ? (sharinganActive ? "Sharingan active" : "Ready · blink to awaken") : faceTrackingState === "loading" ? "Loading model…" : "Unavailable · manual toggle"}</strong></div><div className="readout-row"><span>Hand landmarks</span><strong>{cameraState === 'ready' ? (handCount ? handCount + ' hand' + (handCount === 1 ? '' : 's') + ' tracked' : 'Searching…') : 'Not connected'}</strong></div>
            <div className="readout-row"><span>Gesture classifier</span><strong>{cameraState === 'ready' ? 'Running locally' : 'Awaiting camera'}</strong></div>
            <div className="readout-row"><span>Clone cutout</span><strong>{cloneStatus === 'ready' ? 'Segmentation ready' : cloneStatus === 'loading' ? 'Loading model…' : cloneStatus === 'error' ? 'Visual fallback' : 'Loads on first clone'}</strong></div>
            <div className="signal-track"><span style={{ width: cameraState === 'ready' ? (handCount ? '86%' : '32%') : '5%' }} /></div>
            <p className="signal-caption">{gestureLabel}</p>
          </div>
        </aside>
      </section>
      <section className="bottom-grid">
        <div className="bottom-card mission-card"><div className="bottom-icon"><Lightbulb size={18} /></div><div><div className="section-kicker">YOUR FIRST MISSION</div><h3>Land your first clean cast</h3><p>Move into frame and hold a single open palm steady to cast Rasengan. For Shadow Clone, keep both open hands visible while the browser loads body segmentation. Try two open palms together to cast Raiton Pulse, two closed fists apart for Fireball, one closed fist for Amaterasu, or the two-finger hare seal for Wind Fang. Show your palm—not the back of your hand—for Rasengan. Move both hands out of frame before switching jutsu.</p></div><ChevronDown size={18} className="muted-chevron" /></div>
        <div className="bottom-card privacy-card"><div className="bottom-icon green-icon"><ShieldCheck size={18} /></div><div><div className="section-kicker">PRIVACY BY DESIGN</div><h3>Your video stays on your device</h3><p>Hand landmarks are processed in the browser. The model is fetched from a public CDN the first time; camera footage is not sent to a server by this prototype.</p></div><Check size={18} className="privacy-check" /></div>
      </section>
      <section className="shinobi-strip" aria-label="Featured shinobi">
        <div className="shinobi-heading"><span className="section-kicker">HIDDEN LEAF ROSTER</span><span className="shinobi-note">CHAKRA · TEAM 7 · KONOHA</span></div>
        <div className="shinobi-cards">
          <a className="shinobi-card naruto-profile" href="https://naruto-official.com/en/anime/naruto2/list/01_923" target="_blank" rel="noreferrer">
            <img src="https://wallpapers.com/images/hd/naruto-blue-sky-ready-for-battle-w7rfnmhxkzu441oh.jpg" alt="Naruto Uzumaki anime portrait" loading="lazy" />
            <span className="shinobi-glow" /><div className="shinobi-copy"><small>THE NINE-TAILS JINCHURIKI</small><strong>Naruto Uzumaki</strong><span>RASENGAN · SHADOW CLONE</span></div><span className="shinobi-symbol">うずまき</span>
          </a>
          <a className="shinobi-card kakashi-profile" href="https://naruto-official.com/en/news/01_1903" target="_blank" rel="noreferrer">
            <img src="https://wallpapers.com/images/hd/sensei-kakashi-hatake-png-jja-oqkhqqgluqkoq0q6.png" alt="Kakashi Hatake anime portrait" loading="lazy" />
            <span className="shinobi-glow" /><div className="shinobi-copy"><small>THE COPY NINJA</small><strong>Kakashi Hatake</strong><span>CHIDORI · LIGHTNING BLADE</span></div><span className="shinobi-symbol">写輪眼</span>
          </a>
        </div>
      </section>
      <footer className="footer"><div><span className="footer-mark">木ノ葉</span><span> KONOHA JUTSU LAB</span><span className="footer-dim"> · Early access prototype</span></div><div><span className="footer-dim">Built with MediaPipe · React · Canvas</span><a href="https://github.com/FavorPan/naruto" target="_blank" rel="noreferrer">Inspired by open-source experiments <Expand size={12} /></a></div></footer>
    </main>
  );
}
