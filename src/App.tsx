import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, Camera, Check, ChevronDown, CircleHelp, CirclePlay, Cpu, Expand,
  Hand, Info, Lightbulb, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Zap
} from 'lucide-react';
import { FilesetResolver, HandLandmarker, ImageSegmenter, type HandLandmarkerResult } from '@mediapipe/tasks-vision';

type Jutsu = 'none' | 'rasengan' | 'chidori' | 'clone';
type Gesture = 'none' | 'open' | 'fist' | 'two-open' | 'clone-seal';
type Point = { x: number; y: number; z: number };

// Prefer assets shipped with this GitHub Pages build so the browser does not need to reach a third-party model/CDN.
const LOCAL_MODEL_URL = new URL('models/hand_landmarker.task', document.baseURI).href;
const LOCAL_WASM_URL = new URL('mediapipe/wasm/', document.baseURI).href;
const MODEL_URLS = [
  LOCAL_MODEL_URL,
  'https://huggingface.co/Leo-TX/mediapipe-hand/resolve/main/hand_landmarker.task?download=true',
  'https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task',
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
];
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
  clone: { name: 'Shadow Clone', color: 'violet' }
};

const COMBO_STEPS: Array<{ gesture: Gesture; title: string; instruction: string }> = [
  { gesture: 'open', title: 'Charge Chakra', instruction: 'Show one open palm' },
  { gesture: 'fist', title: 'Lightning Stance', instruction: 'Close one hand into a fist' },
  { gesture: 'clone-seal', title: 'Shadow Clone Seal', instruction: 'Bring both hands together and cross the fingers into a plus (+)' },
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
function readJutsu(result: HandLandmarkerResult | null): { jutsu: Jutsu; gesture: Gesture } {
  const hands = (result?.landmarks ?? []) as Point[][];
  const gestures = hands.map((hand) => classify(hand));
  if (isShadowCloneSeal(hands)) return { jutsu: 'clone', gesture: 'clone-seal' };
  if (hands.length > 1 && gestures.filter((g) => g === 'open').length > 1) return { jutsu: 'none', gesture: 'two-open' };
  if (gestures.includes('open')) return { jutsu: 'rasengan', gesture: 'open' };
  if (gestures.includes('fist')) return { jutsu: 'chidori', gesture: 'fist' };
  return { jutsu: 'none', gesture: 'none' };
}
function palm(points: Point[]) {
  const ids = [0,5,9,13,17];
  return ids.reduce((p, i) => ({ x: p.x + points[i].x / ids.length, y: p.y + points[i].y / ids.length }), { x: 0, y: 0 });
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
  const r = Math.min(w, h) * (0.064 + charge * 0.018) * pulse;
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

function drawSegmentedClones(
  ctx: CanvasRenderingContext2D,
  sprite: HTMLCanvasElement,
  w: number,
  h: number,
  t: number,
) {
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  const echoes = [
    { x: -0.16 + Math.sin(t * 1.4) * 0.008, y: 0.018, scale: 0.91, alpha: 0.37, hue: 135 },
    { x: 0.16 + Math.sin(t * 1.7 + 1) * 0.008, y: -0.008, scale: 0.88, alpha: 0.31, hue: 205 },
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
    ctx.filter = `hue-rotate(${echo.hue}deg) saturate(1.75) brightness(1.3) drop-shadow(0 0 12px rgba(92,190,255,.85))`;
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
  const smoothedLandmarksRef = useRef<Point[][]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const activeRef = useRef<Jutsu>('none');
  const audioRef = useRef<AudioContext | null>(null);
  const soundRef = useRef(false);
  const landmarksRef = useRef(true);
  const [cameraState, setCameraState] = useState<'idle'|'loading'|'ready'|'camera-only'|'error'>('idle');
  const [error, setError] = useState('');
  const [activeJutsu, setActiveJutsu] = useState<Jutsu>('none');
  const [gestureLabel, setGestureLabel] = useState('Waiting for hands');
  const [handCount, setHandCount] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [fps, setFps] = useState(0);
  const [cloneStatus, setCloneStatus] = useState<'idle'|'loading'|'ready'|'error'>('idle');
  const [comboMode, setComboMode] = useState(false);
  const [comboProgress, setComboProgress] = useState(0);
  const [comboFeedback, setComboFeedback] = useState('Start the camera, then practise the three-pose chain.');
  const comboModeRef = useRef(false);
  const comboProgressRef = useRef(0);
  const comboCandidateRef = useRef<Gesture>('none');
  const comboCandidateAtRef = useRef(0);
  useEffect(() => { soundRef.current = soundEnabled; }, [soundEnabled]);
  useEffect(() => { landmarksRef.current = showLandmarks; }, [showLandmarks]);

  const switchJutsu = useCallback((next: Jutsu) => {
    if (activeRef.current === next) return;
    activeRef.current = next; setActiveJutsu(next);
    if (next !== 'none' && soundRef.current) {
      try {
        const AudioCtor = window.AudioContext;
        const audio = audioRef.current ?? new AudioCtor();
        audioRef.current = audio;
        const osc = audio.createOscillator(), gain = audio.createGain();
        osc.type = next === 'chidori' ? 'sawtooth' : 'sine';
        osc.frequency.setValueAtTime(next === 'clone' ? 520 : next === 'chidori' ? 760 : 420, audio.currentTime);
        osc.frequency.exponentialRampToValueAtTime(150, audio.currentTime + .22);
        gain.gain.setValueAtTime(.0001, audio.currentTime);
        gain.gain.exponentialRampToValueAtTime(.045, audio.currentTime + .025);
        gain.gain.exponentialRampToValueAtTime(.0001, audio.currentTime + .25);
        osc.connect(gain); gain.connect(audio.destination); osc.start(); osc.stop(audio.currentTime + .26);
      } catch { /* Audio is an optional enhancement. */ }
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    trackerRef.current?.close(); trackerRef.current = null;
    segmenterRef.current?.close(); segmenterRef.current = null;
    cloneSpriteRef.current = null; cloneSpriteReadyRef.current = false;
    smoothedLandmarksRef.current = [];
    setCloneStatus('idle');
    comboModeRef.current = false; setComboMode(false); comboCandidateRef.current = 'none';
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
      setCameraState('ready');
      setGestureLabel('Show an open palm or make a fist');
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
          const personLabel = labels.findIndex((label) => /person|foreground/i.test(label));
          personMaskIndex = personLabel >= 0 ? personLabel : Math.max(0, labels.length - 1);
          setCloneStatus('ready');
        } catch {
          segmenterFailed = true;
          setCloneStatus('error');
          setGestureLabel('Shadow Clone cutout unavailable; using the visual fallback');
        } finally {
          segmenterLoading = false;
        }
      };
      let lastVideoTime = -1, lastWrist: Point | null = null, move = {x:.2,y:-.05};
      let candidate: Jutsu = 'none', candidateAt = 0, noPoseAt = 0, fpsAt = performance.now(), frames = 0;
      let latest: HandLandmarkerResult | null = null;
      const frame = () => {
        const v = videoRef.current, canvas = canvasRef.current, model = trackerRef.current;
        if (!v || !canvas || !model || v.readyState < 2) { rafRef.current = requestAnimationFrame(frame); return; }
        const w = v.videoWidth || 1280, h = v.videoHeight || 720;
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        const now = performance.now();
        if (v.currentTime !== lastVideoTime) {
          lastVideoTime = v.currentTime; latest = model.detectForVideo(v, now);
          const rawHands = (latest.landmarks ?? []) as Point[][];
          const previousHands = smoothedLandmarksRef.current;
          // A light exponential low-pass filter damps landmark jitter without changing gesture classification.
          smoothedLandmarksRef.current = rawHands.map((hand, handIndex) => {
            const previous = previousHands[handIndex];
            if (!previous || previous.length !== hand.length) return hand.map((point) => ({ ...point }));
            const alpha = 0.46;
            return hand.map((point, pointIndex) => ({
              x: previous[pointIndex].x + (point.x - previous[pointIndex].x) * alpha,
              y: previous[pointIndex].y + (point.y - previous[pointIndex].y) * alpha,
              z: previous[pointIndex].z + (point.z - previous[pointIndex].z) * alpha,
            }));
          });
          const detected = readJutsu(latest), count = rawHands.length;
          setHandCount(count);
          if (comboModeRef.current) {
            const expected = COMBO_STEPS[comboProgressRef.current];
            candidate = 'none'; noPoseAt = 0;
            if (expected && detected.gesture === expected.gesture) {
              if (comboCandidateRef.current !== expected.gesture) {
                comboCandidateRef.current = expected.gesture;
                comboCandidateAtRef.current = now;
                setComboFeedback(`Hold ${expected.title} steady…`);
              } else if (now - comboCandidateAtRef.current >= 500) {
                const nextProgress = comboProgressRef.current + 1;
                comboProgressRef.current = nextProgress;
                setComboProgress(nextProgress);
                comboCandidateRef.current = 'none';
                if (nextProgress >= COMBO_STEPS.length) {
                  comboModeRef.current = false;
                  setComboMode(false);
                  setComboFeedback('Combo complete — Shadow Clone released!');
                  setGestureLabel('Sequence complete · Shadow Clone released');
                  switchJutsu('clone');
                } else {
                  setComboFeedback(`Step ${nextProgress} complete. Next: ${COMBO_STEPS[nextProgress].instruction}.`);
                  setGestureLabel(`Combo practice · step ${nextProgress + 1} of ${COMBO_STEPS.length}`);
                }
              }
            } else {
              comboCandidateRef.current = 'none';
              if (detected.gesture !== 'none' && expected) {
                setComboFeedback(`Not quite — step ${comboProgressRef.current + 1}: ${expected.instruction}.`);
              }
              setGestureLabel(expected ? `Combo practice · step ${comboProgressRef.current + 1} of ${COMBO_STEPS.length}` : 'Finishing combo…');
            }
          } else if (detected.jutsu !== 'none') {
            noPoseAt = 0;
            if (candidate !== detected.jutsu) { candidate = detected.jutsu; candidateAt = now; }
            if (now - candidateAt > 420) {
              switchJutsu(detected.jutsu);
              setGestureLabel(
                detected.gesture === 'clone-seal'
                  ? 'Shadow Clone plus-seal detected'
                  : detected.gesture === 'two-open'
                    ? 'Two open hands detected'
                    : detected.gesture === 'open'
                      ? 'Open palm detected'
                      : 'Fist detected',
              );
            } else setGestureLabel('Hold the pose steady…');
          } else {
            candidate = 'none'; candidateAt = now; if (!noPoseAt) noPoseAt = now;
            setGestureLabel(count ? 'Pose not mapped — try an open palm or fist' : 'Move your hands into the frame');
            if (now - noPoseAt > 700) switchJutsu('none');
          }
          const first = latest.landmarks?.[0] as Point[] | undefined;
          if (first) {
            if (lastWrist) move = {x:first[0].x-lastWrist.x,y:first[0].y-lastWrist.y};
            lastWrist = { ...first[0] };
          } else lastWrist = null;
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
          const charge = Math.min(1, 0.48 + (now - candidateAt) / 2400);
          if (activeRef.current === 'rasengan' && hands.length) drawRasengan(ctx,palm(hands[0]),w,h,now/1000,charge);
          if (activeRef.current === 'chidori' && hands.length) drawChidori(ctx,palm(hands[0]),move,w,h,now/1000,charge);
          if (activeRef.current === 'clone') {
            if (cloneSpriteReadyRef.current && cloneSpriteRef.current) drawSegmentedClones(ctx,cloneSpriteRef.current,w,h,now/1000);
            else drawCloneEcho(ctx,w,h,now/1000);
          }
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
  const toggleCombo = () => {
    if (comboModeRef.current) {
      comboModeRef.current = false;
      setComboMode(false);
      comboCandidateRef.current = 'none';
      setComboFeedback('Combo cancelled. Start again whenever you are ready.');
      return;
    }
    if (cameraState !== 'ready') {
      setComboFeedback('Start the camera before beginning the live combo.');
      return;
    }
    comboModeRef.current = true;
    comboProgressRef.current = 0;
    comboCandidateRef.current = 'none';
    setComboProgress(0);
    setComboMode(true);
    switchJutsu('none');
    setComboFeedback('Step 1: show one open palm and hold it for half a second.');
    setGestureLabel('Combo practice · step 1 of 3');
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Konoha Jutsu Lab home">
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div><strong>KONOHA <span>JUTSU LAB</span></strong><small>CAMERA-TRACKED CHAKRA PROTOTYPE</small></div>
        </a>
        <div className="topbar-right"><div className="build-chip"><span className="status-dot" /> PROTOTYPE / 0.3</div>
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
      {helpOpen && <section className="help-panel"><Info size={18} /><div><strong>Quick start · How to cast</strong><p><b>1.</b> Press Start camera and allow permission. Keep your hands well lit and fully inside the frame. <b>2. Rasengan:</b> show one open palm toward the camera and hold it steady. <b>3. Chidori:</b> close one hand into a clear fist and hold. <b>4. Shadow Clone:</b> bring both hands together and cross the index/middle fingers into a visible plus (+) shape; keep ring and little fingers curled. Hold the seal still while the clones appear. <b>5. Training:</b> press Start combo, then hold open palm → fist → both open palms, about half a second each. The cards below are visual previews only, not casts. Video is processed in your browser.</p></div><button className="text-button" onClick={() => setHelpOpen(false)}>Got it</button></section>}
      <section className="workspace">
        <div className="camera-column">
          <div className="panel-heading"><div><span className="section-kicker">01 / LIVE INPUT</span><h2>Chakra viewport</h2></div>
            <div className={'live-status ' + (cameraState === 'ready' || cameraState === 'camera-only' ? 'is-live' : cameraState === 'error' ? 'is-error' : '')}><span />{cameraState === 'ready' ? 'LIVE TRACKING' : cameraState === 'camera-only' ? 'CAMERA ON · TRACKING OFF' : cameraState === 'loading' ? 'INITIALIZING' : cameraState === 'error' ? 'CAMERA ERROR' : 'STANDBY'}</div>
          </div>
          <div ref={stageRef} className={'camera-stage ' + (activeInfo ? 'effect-' + activeInfo.color : '')}>
            <div className="stage-grid" /><div className="stage-corner corner-tl" /><div className="stage-corner corner-tr" /><div className="stage-corner corner-bl" /><div className="stage-corner corner-br" />
            <div className="stage-topline"><span><Activity size={13} /> INPUT FEED</span><span>{cameraState === 'ready' ? (fps || '—') + ' FPS' : '1280 × 720 TARGET'}</span></div>
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
            <button className={'secondary-button ' + (soundEnabled ? 'selected' : '')} onClick={() => setSoundEnabled((v) => !v)}><Zap size={16} /> Cast sound <span className="toggle-pill">{soundEnabled ? 'ON' : 'OFF'}</span></button>
            <button className="reset-button" onClick={() => { switchJutsu('none'); setGestureLabel('Effects cleared'); }} title="Clear current effect"><RotateCcw size={16} /></button>
          </div>
          {(cameraState === 'error' || cameraState === 'camera-only') && <p className="error-hint"><Info size={14} /> {cameraState === 'camera-only' ? 'The camera preview is being kept active. Hand tracking is unavailable until the model/runtime loads; use the GitHub Actions link to check that the newest build was deployed.' : 'Camera access requires browser permission and a secure context (localhost or HTTPS).'}</p>}
        </div>
        <aside className="right-column">
          <div className="panel-heading"><div><span className="section-kicker">02 / TECHNIQUE SELECTOR</span><h2>Choose your jutsu</h2></div><span className="mini-count">03</span></div>
          <div className="jutsu-list">
            <button className={'jutsu-card blue-card ' + (activeJutsu === 'rasengan' ? 'is-active' : '')} onClick={() => preview('rasengan')}>
              <div className="jutsu-art rasengan-art"><span className="orb-core" /><span className="orb-ring ring-one" /><span className="orb-ring ring-two" /><span className="orb-spark s1" /><span className="orb-spark s2" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">WIND / CHAKRA</div><h3>Rasengan</h3><p>Open one palm to the camera and hold steady · blue chakra orb</p><div className="jutsu-foot"><span className="difficulty easy">BEGINNER</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">01</span>
            </button>
            <button className={'jutsu-card cyan-card ' + (activeJutsu === 'chidori' ? 'is-active' : '')} onClick={() => preview('chidori')}>
              <div className="jutsu-art lightning-art"><Zap className="lightning-symbol" size={44} /><i className="bolt b1" /><i className="bolt b2" /><i className="bolt b3" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">LIGHTNING STYLE</div><h3>Chidori</h3><p>Make a clear fist and hold steady · electric-blue lightning</p><div className="jutsu-foot"><span className="difficulty medium">INTERMEDIATE</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">02</span>
            </button>
            <button className={'jutsu-card violet-card ' + (activeJutsu === 'clone' ? 'is-active' : '')} onClick={() => preview('clone')}>
              <div className="jutsu-art clone-art"><span className="clone-figure figure-one" /><span className="clone-figure figure-two" /><span className="clone-figure figure-three" /><Sparkles className="clone-star" size={17} /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">NINJUTSU / DECOY</div><h3>Shadow Clone</h3><p>Both hands · cross index/middle fingers into a + seal</p><div className="jutsu-foot"><span className="difficulty advanced">CROSSED SEAL</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">03</span>
            </button>
          </div>
          <div className="combo-card">
            <div className="combo-header">
              <div><span className="section-kicker">03 / SEAL SEQUENCE</span><h3>Three-pose combo</h3></div>
              <span className="combo-count">{comboProgress}/{COMBO_STEPS.length}</span>
            </div>
            <div className="combo-steps">
              {COMBO_STEPS.map((step, index) => (
                <div className={'combo-step ' + (index < comboProgress ? 'complete' : index === comboProgress && comboMode ? 'current' : '')} key={step.title}>
                  <span className="combo-step-index">{index < comboProgress ? <Check size={13} /> : String(index + 1).padStart(2, '0')}</span>
                  <div><strong>{step.title}</strong><small>{step.instruction}</small></div>
                </div>
              ))}
            </div>
            <p className="combo-feedback">{comboFeedback}</p>
            <div className="combo-actions">
              <button className={comboMode ? 'secondary-button combo-cancel' : 'primary-button combo-start'} onClick={toggleCombo} disabled={cameraState === 'loading'}>
                {comboMode ? <RotateCcw size={15} /> : <Hand size={15} />}
                {comboMode ? 'Cancel practice' : comboProgress === COMBO_STEPS.length ? 'Practise again' : 'Start combo'}
              </button>
              <span>Hold each pose · 0.5 sec</span>
            </div>
            <p className="combo-note">Practice sequence: one open palm → one fist → both open palms. This is a starter gesture combo, not yet the traditional Naruto hand-seal sequence.</p>
          </div>
          <div className="detection-card">
            <div className="detection-heading"><div><span className="section-kicker">LIVE READOUT</span><h3>Detection status</h3></div><Activity size={17} /></div>
            <div className="readout-row"><span>Current signal</span><strong className={activeJutsu !== 'none' ? 'signal-active' : ''}>{activeJutsu === 'none' ? 'No active jutsu' : INFO[activeJutsu].name}</strong></div>
            <div className="readout-row"><span>Hand landmarks</span><strong>{cameraState === 'ready' ? (handCount ? handCount + ' hand' + (handCount === 1 ? '' : 's') + ' tracked' : 'Searching…') : 'Not connected'}</strong></div>
            <div className="readout-row"><span>Gesture classifier</span><strong>{cameraState === 'ready' ? 'Running locally' : 'Awaiting camera'}</strong></div>
            <div className="readout-row"><span>Clone cutout</span><strong>{cloneStatus === 'ready' ? 'Segmentation ready' : cloneStatus === 'loading' ? 'Loading model…' : cloneStatus === 'error' ? 'Visual fallback' : 'Loads on first clone'}</strong></div>
            <div className="signal-track"><span style={{ width: cameraState === 'ready' ? (handCount ? '86%' : '32%') : '5%' }} /></div>
            <p className="signal-caption">{gestureLabel}</p>
          </div>
        </aside>
      </section>
      <section className="bottom-grid">
        <div className="bottom-card mission-card"><div className="bottom-icon"><Lightbulb size={18} /></div><div><div className="section-kicker">YOUR FIRST MISSION</div><h3>Land your first clean cast</h3><p>Move into frame and hold a single open palm steady to cast Rasengan. For Shadow Clone, keep both open hands visible while the browser loads body segmentation. Try the preview cards first if your camera is not available.</p></div><ChevronDown size={18} className="muted-chevron" /></div>
        <div className="bottom-card privacy-card"><div className="bottom-icon green-icon"><ShieldCheck size={18} /></div><div><div className="section-kicker">PRIVACY BY DESIGN</div><h3>Your video stays on your device</h3><p>Hand landmarks are processed in the browser. The model is fetched from a public CDN the first time; camera footage is not sent to a server by this prototype.</p></div><Check size={18} className="privacy-check" /></div>
      </section>
      <footer className="footer"><div><span className="footer-mark">木ノ葉</span><span> KONOHA JUTSU LAB</span><span className="footer-dim"> · Early access prototype</span></div><div><span className="footer-dim">Built with MediaPipe · React · Canvas</span><a href="https://github.com/FavorPan/naruto" target="_blank" rel="noreferrer">Inspired by open-source experiments <Expand size={12} /></a></div></footer>
    </main>
  );
}
