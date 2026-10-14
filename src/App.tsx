import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, Camera, Check, ChevronDown, CircleHelp, CirclePlay, Cpu, Expand,
  Hand, Info, Lightbulb, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Zap
} from 'lucide-react';
import { FilesetResolver, HandLandmarker, ImageSegmenter, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import ThreeEffects, { type ThreeTrackingState } from './ThreeEffects';

type Jutsu = 'none' | 'rasengan' | 'clone';
type Gesture = 'none' | 'open' | 'fist' | 'clone-seal';
type Point = { x: number; y: number; z: number };
type OneEuroAxis = { raw: number; filtered: number; derivative: number; at: number };
type OneEuroPoint = { x: OneEuroAxis; y: OneEuroAxis; z: OneEuroAxis };
type WorkerHandPayload = { landmarks: Point[][]; handedness?: Array<Array<{ categoryName: string }>>; worldLandmarks?: Point[][] };
type VisionWorkerOutput =
  | { type: 'ready'; delegate: string }
  | { type: 'result'; timestamp: number; hands: WorkerHandPayload }
  | { type: 'tracking-error'; message: string }
  | { type: 'init-error'; message: string };

// Prefer assets shipped with this GitHub Pages build so the browser does not need to reach a third-party model/CDN.
const LOCAL_MODEL_URL = new URL('models/hand_landmarker.task', document.baseURI).href;
const LOCAL_WASM_URL = new URL('mediapipe/wasm/', document.baseURI).href;
const MODEL_URLS = [
  LOCAL_MODEL_URL,
  'https://huggingface.co/Leo-TX/mediapipe-hand/resolve/main/hand_landmarker.task?download=true',
  'https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task',
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
];
const LOCAL_SEGMENTER_MODEL_URL = new URL('models/selfie_segmenter_landscape.tflite', document.baseURI).href;
const SEGMENTER_MODEL_URLS = [
  LOCAL_SEGMENTER_MODEL_URL,
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
];
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
            return { tracker };
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
  clone: { name: 'Shadow Clone', color: 'violet' },
};

function dist(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * 0.5);
}
function fingerExtended(points: Point[], tip: number, pip: number, mcp: number) {
  if (points.length < 21) return false;
  const dip = points[tip - 1];
  const base = points[mcp], joint = points[pip];
  // Prefer the PIP bend angle to wrist-distance alone, which is more stable
  // when the palm is rotated toward or away from the camera.
  const ax = base.x - joint.x, ay = base.y - joint.y, az = (base.z - joint.z) * 0.35;
  const bx = dip.x - joint.x, by = dip.y - joint.y, bz = (dip.z - joint.z) * 0.35;
  const la = Math.hypot(ax, ay, az), lb = Math.hypot(bx, by, bz);
  if (la < 1e-5 || lb < 1e-5) return false;
  const cosine = Math.max(-1, Math.min(1, (ax * bx + ay * by + az * bz) / (la * lb)));
  const angle = Math.acos(cosine) * 180 / Math.PI;
  return angle > 148 && dist(points[tip], points[mcp]) > dist(points[pip], points[mcp]) * 1.08;
}
function filterOneEuro(value: number, previous: OneEuroAxis | undefined, now: number): { value: number; state: OneEuroAxis } {
  if (!previous) return { value, state: { raw: value, filtered: value, derivative: 0, at: now } };
  const dt = Math.max(1 / 120, Math.min(0.12, (now - previous.at) / 1000));
  const alpha = (cutoff: number) => {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  };
  const rawDerivative = (value - previous.raw) / dt;
  const derivativeAlpha = alpha(1.0);
  const derivative = previous.derivative + derivativeAlpha * (rawDerivative - previous.derivative);
  // Normalized video coordinates need a larger beta than pixel-space examples.
  const cutoff = 3.2 + 11.0 * Math.abs(derivative);
  const valueAlpha = alpha(cutoff);
  const filtered = previous.filtered + valueAlpha * (value - previous.filtered);
  return { value: filtered, state: { raw: value, filtered, derivative, at: now } };
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
  // Make the seal tolerant of small landmark errors: both index/middle pairs
  // should be extended, but the ring/pinky fingers need not be perfectly curled.
  if (poses.some((pose) => !pose.index || !pose.middle || (pose.ring && pose.pinky))) return false;

  const [a, b] = hands;
  const palmA = palm(a), palmB = palm(b);
  const scaleA = dist(a[0], a[9]), scaleB = dist(b[0], b[9]);
  const palmsNear = Math.hypot(palmA.x - palmB.x, palmA.y - palmB.y) < (scaleA + scaleB) * 2.25;
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
  const tipsNear = Math.hypot(tipA.x - tipB.x, tipA.y - tipB.y) < (scaleA + scaleB) * 1.9;
  return angle > 12 && tipsNear;
}
function playJutsuCue(audio: AudioContext, jutsu: Exclude<Jutsu, 'none'>) {
  const now = audio.currentTime;
  const master = audio.createGain();
  master.gain.value = 0.34;
  master.connect(audio.destination);
  if (jutsu === 'rasengan') {
    const oscillator = audio.createOscillator();
    const overtone = audio.createOscillator();
    const filter = audio.createBiquadFilter();
    const gain = audio.createGain();
    oscillator.type = 'sine';
    overtone.type = 'triangle';
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(720, now);
    filter.frequency.exponentialRampToValueAtTime(190, now + 0.55);
    oscillator.frequency.setValueAtTime(180, now);
    oscillator.frequency.exponentialRampToValueAtTime(62, now + 0.56);
    overtone.frequency.setValueAtTime(360, now);
    overtone.frequency.exponentialRampToValueAtTime(124, now + 0.52);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.42, now + 0.06);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.58);
    oscillator.connect(filter); overtone.connect(filter); filter.connect(gain); gain.connect(master);
    oscillator.start(now); overtone.start(now);
    oscillator.stop(now + 0.6); overtone.stop(now + 0.6);
  } else {
    const duration = 0.34;
    const buffer = audio.createBuffer(1, Math.floor(audio.sampleRate * duration), audio.sampleRate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / samples.length, 2.1);
    const source = audio.createBufferSource();
    const lowpass = audio.createBiquadFilter();
    const gain = audio.createGain();
    source.buffer = buffer;
    lowpass.type = 'lowpass';
    lowpass.frequency.setValueAtTime(1500, now);
    lowpass.frequency.exponentialRampToValueAtTime(240, now + duration);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.3, now + 0.035);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    source.connect(lowpass); lowpass.connect(gain); gain.connect(master);
    source.start(now); source.stop(now + duration);
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
function readJutsu(result: HandLandmarkerResult | null): { jutsu: Jutsu; gesture: Gesture; handIndex?: number } {
  const hands = (result?.landmarks ?? []) as Point[][];
  const gestures = hands.map((hand) => classify(hand));

  // Keep the live classifier intentionally narrow: these are the only two
  // techniques in the focused experience.
  if (isShadowCloneSeal(hands)) return { jutsu: 'clone', gesture: 'clone-seal' };

  const openHands = gestures
    .map((gesture, index) => gesture === 'open' ? index : -1)
    .filter((index) => index >= 0);

  if (openHands.length === 1) {
    const index = openHands[0];
    if (result && palmOrientationAllowsRasengan(result, index)) {
      return { jutsu: 'rasengan', gesture: 'open', handIndex: index };
    }
  }
  if (gestures.includes('fist')) return { jutsu: 'none', gesture: 'fist' };
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
function createVisionBitmap(video: HTMLVideoElement, maxWidth = 640): Promise<ImageBitmap> {
  const sourceWidth = video.videoWidth || maxWidth;
  const sourceHeight = video.videoHeight || Math.round(maxWidth * 9 / 16);
  const scale = Math.min(1, maxWidth / sourceWidth);
  return createImageBitmap(video, {
    resizeWidth: Math.max(1, Math.round(sourceWidth * scale)),
    resizeHeight: Math.max(1, Math.round(sourceHeight * scale)),
    resizeQuality: 'low',
  });
}

type CloneMask = { width:number; height:number; getAsFloat32Array():Float32Array; close():void };
function updateCloneSprite(video:HTMLVideoElement, mask:CloneMask, sprite:HTMLCanvasElement):boolean {
 const width=mask.width,height=mask.height;
 if(!width||!height)return false;
 const scores=new Float32Array(mask.getAsFloat32Array());
 if(scores.length<width*height)return false;
 let peak=0;for(let i=0;i<width*height;i++)if(scores[i]>peak)peak=scores[i];
 if(peak<0.2)return false;
 let minX=width,minY=height,maxX=-1,maxY=-1;
 for(const threshold of [0.38,0.27,0.16]){
  minX=width;minY=height;maxX=-1;maxY=-1;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(scores[y*width+x]>threshold){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
  if(maxX>=minX&&maxY>=minY&&(maxX-minX+1)*(maxY-minY+1)>width*height*0.025)break;
 }
 if(maxX<minX||maxY<minY)return false;
 const bw=maxX-minX+1,bh=maxY-minY+1,padX=Math.max(4,Math.round(bw*.1)),padTop=Math.max(3,Math.round(bh*.04)),padBottom=Math.max(3,Math.round(bh*.035));
 const left=Math.max(0,minX-padX),top=Math.max(0,minY-padTop),right=Math.min(width,maxX+1+padX),bottom=Math.min(height,maxY+1+padBottom);
 const frame=document.createElement('canvas');frame.width=width;frame.height=height;
 const fc=frame.getContext('2d',{willReadFrequently:true});if(!fc)return false;
 fc.drawImage(video,0,0,width,height);
 const pixels=fc.getImageData(0,0,width,height);
 for(let i=0;i<width*height;i++){const v=Math.max(0,Math.min(1,(scores[i]-.12)/.58));const a=v*v*(3-2*v);pixels.data[i*4+3]=Math.round(pixels.data[i*4+3]*a);}
 fc.putImageData(pixels,0,0);
 const cw=Math.max(1,right-left),ch=Math.max(1,bottom-top);
 if(sprite.width!==cw||sprite.height!==ch){sprite.width=cw;sprite.height=ch;}
 const sc=sprite.getContext('2d');if(!sc)return false;
 sc.clearRect(0,0,cw,ch);sc.imageSmoothingEnabled=true;sc.imageSmoothingQuality='high';sc.drawImage(frame,left,top,cw,ch,0,0,cw,ch);
 sprite.dataset.revision=String((Number(sprite.dataset.revision)||0)+1);
 frame.width=1;frame.height=1;return true;
}

export default function App() {
export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const threeTrackingRef = useRef<ThreeTrackingState>({ hands: [], effectHands: [], effectHandIndex: 0, cloneSprite: null, cloneRevision: 0, cloneVisibleUntil: 0 });
  const effectHandIndexRef = useRef(0);
  const trackerRef = useRef<HandLandmarker | null>(null);
  const visionWorkerRef = useRef<Worker | null>(null);
  const segmenterRef = useRef<ImageSegmenter | null>(null);
  const cloneSpriteRef = useRef<HTMLCanvasElement | null>(null);
  const cloneSpriteReadyRef = useRef(false);
  const smoothedLandmarksRef = useRef<Point[][]>([]);
  // Effects receive raw landmarks so they can react immediately to fast movements.
  const effectLandmarksRef = useRef<Point[][]>([]);
  const oneEuroFiltersRef = useRef<OneEuroPoint[][]>([]);
  const smoothedHandednessRef = useRef<string[]>([]);
  // A short rolling vote rejects one-frame gesture flicker between the two core signs.
  const gestureVoteHistoryRef = useRef<Array<{ jutsu: Jutsu; gesture: Gesture; at: number }>>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const activeRef = useRef<Jutsu>('none');
  const audioRef = useRef<AudioContext | null>(null);
  const jutsuStartedAtRef = useRef(0);
  const castUntilRef = useRef(0);
  const cloneVisualUntilRef = useRef(0);
  const cloneStatusRef = useRef<'idle'|'loading'|'ready'|'error'>('idle');
  const soundRef = useRef(false);
  const previewUntilRef = useRef(0);
  const landmarksRef = useRef(true);
  const [cameraState, setCameraState] = useState<'idle'|'loading'|'ready'|'camera-only'|'error'>('idle');
  const [error, setError] = useState('');
  const [activeJutsu, setActiveJutsu] = useState<Jutsu>('none');
  const [chargeLevel, setChargeLevel] = useState(0);
  const [gestureLabel, setGestureLabel] = useState('Waiting for hands');
  const [handCount, setHandCount] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [fps, setFps] = useState(0);
  const [trackingEngine, setTrackingEngine] = useState<'idle'|'starting'|'worker-gpu'|'worker-cpu'|'main-thread'|'unavailable'>('idle');
  const [cloneStatus, setCloneStatus] = useState<'idle'|'loading'|'ready'|'error'>('idle');
  useEffect(() => { cloneStatusRef.current = cloneStatus; }, [cloneStatus]);
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
    previewUntilRef.current = 0;
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    timeoutRef.current = null;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    trackerRef.current?.close(); trackerRef.current = null;
    visionWorkerRef.current?.terminate(); visionWorkerRef.current = null;
    segmenterRef.current?.close(); segmenterRef.current = null;
    cloneSpriteRef.current = null; cloneSpriteReadyRef.current = false;
    smoothedLandmarksRef.current = []; effectLandmarksRef.current = []; smoothedHandednessRef.current = []; oneEuroFiltersRef.current = [];
    gestureVoteHistoryRef.current = [];
    setTrackingEngine('idle');
    setCloneStatus('idle');
    castUntilRef.current = 0; cloneVisualUntilRef.current = 0;
    if (videoRef.current) videoRef.current.srcObject = null;
    threeTrackingRef.current = { hands: [], effectHands: [], effectHandIndex: 0, cloneSprite: null, cloneRevision: 0, cloneVisibleUntil: 0 };
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
    stopCamera(); setCameraState('loading'); setError(''); setGestureLabel('Loading hand-tracking model…'); setTrackingEngine('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false, video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 30, max: 30 } }
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
      setGestureLabel('Camera is on · starting the background vision worker…');
      const cameraStream = stream;
      let worker: Worker | null = null;
      let workerMode = false;
      let workerBusy = false;
      let inferencePending = false;
      let latest: HandLandmarkerResult | null = null;
      let workerDelegate = 'GPU';

      if (typeof Worker !== 'undefined' && typeof createImageBitmap === 'function') {
        try {
          worker = new Worker(new URL('./vision.worker.ts', import.meta.url), { type: 'module' });
          const activeWorker = worker;
          await new Promise<void>((resolve, reject) => {
            const timer = window.setTimeout(() => reject(new Error('Background vision worker timed out while loading models.')), 45000);
            activeWorker.onmessage = (event: MessageEvent<VisionWorkerOutput>) => {
              const message = event.data;
              if (message.type === 'ready') {
                workerDelegate = message.delegate;
                window.clearTimeout(timer);
                resolve();
              } else if (message.type === 'init-error') {
                window.clearTimeout(timer);
                reject(new Error(message.message));
              } else if (message.type === 'tracking-error') {
                setGestureLabel('Vision worker: ' + message.message);
              }
            };
            activeWorker.onerror = (event) => {
              window.clearTimeout(timer);
              reject(new Error(event.message || 'Background vision worker failed to start.'));
            };
            activeWorker.postMessage({ type: 'init', wasmUrls: WASM_URLS, handModelUrls: MODEL_URLS });
          });
          visionWorkerRef.current = activeWorker;
          workerMode = true;
          setTrackingEngine(workerDelegate === 'GPU' ? 'worker-gpu' : 'worker-cpu');
          activeWorker.onmessage = (event: MessageEvent<VisionWorkerOutput>) => {
            const message = event.data;
            if (visionWorkerRef.current !== activeWorker) return;
            if (message.type === 'result') {
              latest = message.hands as unknown as HandLandmarkerResult;
              inferencePending = true;
              workerBusy = false;
            } else if (message.type === 'tracking-error') {
              workerBusy = false;
              setGestureLabel('Tracking hiccup · retrying in background');
            }
          };
          activeWorker.onerror = () => {
            workerBusy = false;
            if (visionWorkerRef.current !== activeWorker) return;
            visionWorkerRef.current = null;
            activeWorker.terminate();
            workerMode = false;
            setTrackingEngine('starting');
            setGestureLabel('Background worker stopped · starting compatibility tracking');
            void createHandTracker().then((initialized) => {
              if (streamRef.current !== cameraStream) { initialized.tracker.close(); return; }
              trackerRef.current = initialized.tracker;
              setTrackingEngine('main-thread');
              setGestureLabel('Tracking ready · form Rasengan or Shadow Clone');
            }).catch((cause) => {
              setCameraState('camera-only');
              setTrackingEngine('unavailable');
              setError('Background tracking failed and the compatibility tracker could not start: ' + describeFailure(cause));
            });
          };
        } catch (workerCause) {
          worker?.terminate();
          worker = null;
          visionWorkerRef.current = null;
          workerMode = false;
          setGestureLabel('Background tracking unavailable · using compatible fallback');
          setTrackingEngine('starting');
        }
      }

      if (!workerMode) {
        const initialized = await createHandTracker();
        trackerRef.current = initialized.tracker;
        setTrackingEngine('main-thread');
      }
      setCameraState('ready');
      setGestureLabel('Tracking ready · open palm for Rasengan or cross-finger seal for Shadow Clone');
      let segmenterLoading=false,segmenterFailed=false,segmenterReady=false,segmentationBusy=false;
      let lastSegmentationAt=0,lastSegmentationVideoTime=-1;
      const cloneSprite=document.createElement('canvas');
      cloneSpriteRef.current=cloneSprite;cloneSpriteReadyRef.current=false;setCloneStatus('loading');
      const loadSegmenter=async()=>{
       if(segmenterLoading||segmenterRef.current||segmenterFailed)return;
       segmenterLoading=true;setCloneStatus('loading');
       const failures:string[]=[];
       try{
        let vision:Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>|null=null;
        for(const wasmUrl of WASM_URLS){try{vision=await FilesetResolver.forVisionTasks(wasmUrl);break;}catch(cause){failures.push('WASM '+wasmUrl+': '+describeFailure(cause));}}
        if(!vision)throw new Error('Could not load MediaPipe vision runtime. '+failures.slice(-2).join(' | '));
        for(const modelUrl of SEGMENTER_MODEL_URLS)for(const delegate of ['GPU','CPU'] as const){
         try{
          const seg=await ImageSegmenter.createFromOptions(vision,{baseOptions:{modelAssetPath:modelUrl,delegate},runningMode:'VIDEO',outputCategoryMask:false,outputConfidenceMasks:true});
          if(streamRef.current!==cameraStream){seg.close();return;}
          segmenterRef.current=seg;segmenterReady=true;
          setGestureLabel('Person segmentation ready · form the crossed-finger Shadow Clone seal');return;
         }catch(cause){failures.push(delegate+' / '+modelUrl+': '+describeFailure(cause));}
        }
        throw new Error('Could not initialize person segmentation. '+failures.slice(-4).join(' | '));
       }catch(cause){
        if(streamRef.current===cameraStream){segmenterFailed=true;setCloneStatus('error');setGestureLabel('Shadow Clone image cutout unavailable: '+describeFailure(cause));}
       }finally{segmenterLoading=false;}
      };
      void loadSegmenter();
      let lastVideoTime = -1;      let lastVideoTime = -1;
      let overlayDirty = true;
      let candidate: Jutsu = 'none', candidateAt = 0, noPoseAt = 0, fpsAt = performance.now(), frames = 0;
      const frame = () => {
        const v = videoRef.current, canvas = canvasRef.current, model = trackerRef.current;
        if (!v || !canvas || (!model && !workerMode) || v.readyState < 2) { rafRef.current = requestAnimationFrame(frame); return; }
        const w = v.videoWidth || 960, h = v.videoHeight || 540;
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; overlayDirty = true; }
        const now = performance.now();
        if (v.currentTime !== lastVideoTime) {
          lastVideoTime = v.currentTime;
          if (workerMode && worker && visionWorkerRef.current === worker) {
            if (!workerBusy) {
              workerBusy = true;
              const timestamp = now;
              void createVisionBitmap(v, 640).then((bitmap) => {
                if (visionWorkerRef.current !== worker) { bitmap.close(); workerBusy = false; return; }
                worker!.postMessage({ type: 'frame', timestamp, bitmap }, [bitmap]);
              }).catch(() => {
                workerBusy = false;
                setGestureLabel('Could not copy a camera frame · retrying');
              });
            }
          } else if (model) {
            latest = model.detectForVideo(v, now);
            inferencePending = true;
          }
        }
        if (inferencePending && latest) {
          inferencePending = false;
          frames++;
          if (now - fpsAt >= 1000) { setFps(Math.round(frames * 1000 / (now - fpsAt))); fpsAt = now; frames = 0; }
          const rawHands = (latest.landmarks ?? []) as Point[][];
          overlayDirty = true;
          effectLandmarksRef.current = rawHands;
          const oldHandedness = smoothedHandednessRef.current;
          const previousFilters = oneEuroFiltersRef.current;
          const currentHandedness = latest.handedness?.map((entry, index) => entry[0]?.categoryName ?? `Unknown-${index}`) ?? rawHands.map((_, index) => `Unknown-${index}`);
          // Match filters to handedness so left/right order changes do not jump the overlay.
          const nextFilters: OneEuroPoint[][] = [];
          smoothedLandmarksRef.current = rawHands.map((hand, handIndex) => {
            const priorIndex = oldHandedness.indexOf(currentHandedness[handIndex]);
            const priorFilters = previousFilters[priorIndex >= 0 ? priorIndex : handIndex];
            const currentFilters: OneEuroPoint[] = [];
            const filteredHand = hand.map((point, pointIndex) => {
              const previous = priorFilters?.[pointIndex];
              const x = filterOneEuro(point.x, previous?.x, now);
              const y = filterOneEuro(point.y, previous?.y, now);
              const z = filterOneEuro(point.z, previous?.z, now);
              currentFilters.push({ x: x.state, y: y.state, z: z.state });
              return { x: x.value, y: y.value, z: z.value };
            });
            nextFilters.push(currentFilters);
            return filteredHand;
          });
          oneEuroFiltersRef.current = nextFilters;
          smoothedHandednessRef.current = currentHandedness;
          const rawDetected = readJutsu(latest), count = rawHands.length;
          if (rawDetected.jutsu === 'rasengan' && rawDetected.handIndex !== undefined) effectHandIndexRef.current = rawDetected.handIndex;
          const voteHistory = gestureVoteHistoryRef.current;
          voteHistory.push({ ...rawDetected, at: now });
          // A short temporal vote suppresses single-frame flicker without adding much lag.
          while (voteHistory.length > 5 || (voteHistory[0] && now - voteHistory[0].at > 320)) voteHistory.shift();
          const votes = new Map<string, { jutsu: Jutsu; gesture: Gesture; count: number }>();
          for (const sample of voteHistory) {
            const key = sample.jutsu + ':' + sample.gesture;
            const vote = votes.get(key);
            if (vote) vote.count++;
            else votes.set(key, { jutsu: sample.jutsu, gesture: sample.gesture, count: 1 });
          }
          let winner: { jutsu: Jutsu; gesture: Gesture; count: number } | undefined;
          for (const vote of votes.values()) if (!winner || vote.count > winner.count) winner = vote;
          const detected = winner && winner.count >= 3
            ? { jutsu: winner.jutsu, gesture: winner.gesture }
            : { jutsu: 'none' as const, gesture: rawDetected.gesture };
          setHandCount(count);
          if (activeRef.current !== 'none') {
            if (count > 0) {
              noPoseAt = 0;
              if (detected.jutsu !== 'none' && detected.jutsu !== activeRef.current) {
                if (candidate !== detected.jutsu) { candidate = detected.jutsu; candidateAt = now; }
                if (now - candidateAt > 175) {
                  const next = detected.jutsu;
                  if (next === 'clone') {
                    const cloneWindow = cloneSpriteReadyRef.current ? 6000 : 15000;
                    castUntilRef.current = now + cloneWindow;
                    cloneVisualUntilRef.current = castUntilRef.current;
                  }
                  previewUntilRef.current = 0;
                  switchJutsu(next);
                  candidate = 'none'; candidateAt = 0;
                  setGestureLabel(next === 'clone'
                    ? 'Shadow Clone summoned · clones linger for six seconds'
                    : 'Rasengan formed · keep your palm visible to build chakra');
                } else setGestureLabel('New hand sign detected · hold steady for a moment to switch');
              } else {
                candidate = 'none'; candidateAt = 0;
                if (activeRef.current === 'clone' && detected.gesture === 'clone-seal') {
                  const cloneWindow = cloneSpriteReadyRef.current ? 6000 : 15000;
                  cloneVisualUntilRef.current = now + cloneWindow;
                  castUntilRef.current = cloneVisualUntilRef.current;
                }
                setGestureLabel(detected.jutsu === activeRef.current
                  ? activeRef.current === 'clone'
                    ? 'Shadow Clone active · hold the crossed-finger seal to keep summoning'
                    : 'Rasengan charging · keep your open palm facing the camera'
                  : activeRef.current === 'clone'
                    ? 'Shadow Clone active · form the crossed-finger seal again to refresh'
                    : 'Rasengan active · keep your palm in frame to sustain the chakra');
              }
            } else {
              candidate = 'none';
              if (!noPoseAt) noPoseAt = now;
              const cloneStillVisible = activeRef.current === 'clone' && now < cloneVisualUntilRef.current;
              const waitingForClone = activeRef.current === 'clone' && !cloneSpriteReadyRef.current && now < castUntilRef.current && !segmenterFailed;
              const previewStillVisible = now < previewUntilRef.current;
              if (cloneStillVisible && cloneSpriteReadyRef.current) setGestureLabel('Shadow Clone · your live image is duplicated');
              else if (waitingForClone) setGestureLabel('Shadow Clone · isolating your image, keep your upper body in frame');
              else if (cloneStillVisible) setGestureLabel('Shadow Clone · preparing the person cutout');
              else if (previewStillVisible) setGestureLabel(activeRef.current === 'clone' ? 'Shadow Clone preview · loading your cutout' : 'Rasengan preview · building chakra');
              else if (now > castUntilRef.current && now - noPoseAt > 500) {
                switchJutsu('none'); castUntilRef.current = 0; noPoseAt = 0;
                setGestureLabel('Ready · form Rasengan or the crossed-finger Shadow Clone seal');
              }
            }
          } else if (detected.jutsu !== 'none') {
            noPoseAt = 0;
            if (candidate !== detected.jutsu) { candidate = detected.jutsu; candidateAt = now; }
            if (now - candidateAt > 175) {
              const next = detected.jutsu;
              if (next === 'clone') {
                const cloneWindow = cloneSpriteReadyRef.current ? 6000 : 15000;
                castUntilRef.current = now + cloneWindow;
                cloneVisualUntilRef.current = castUntilRef.current;
              }
              previewUntilRef.current = 0;
              switchJutsu(next);
              setGestureLabel(next === 'clone'
                ? 'Shadow Clone summoned · clones linger for six seconds'
                : 'Rasengan formed · keep your palm visible to build chakra');
              candidate = 'none'; candidateAt = 0;
            } else setGestureLabel('Hand sign recognized · hold steady to summon');
          } else {
            candidate = 'none'; candidateAt = now;
            if (!noPoseAt) noPoseAt = now;
            setGestureLabel(count
              ? 'Pose detected · face one open palm toward the camera for Rasengan, or cross both index/middle fingers for Shadow Clone'
              : 'Move your hands into the frame');
            if (now > castUntilRef.current && now - noPoseAt > 700) switchJutsu('none');
          }
        }
        if(!segmenterRef.current&&!segmenterLoading&&!segmenterFailed)void loadSegmenter();
        const shouldSegment=activeRef.current==='clone'||!cloneSpriteReadyRef.current;
        if(segmenterReady&&segmenterRef.current&&shouldSegment&&!segmentationBusy&&v.currentTime!==lastSegmentationVideoTime){
         const interval=activeRef.current==='clone'?125:280;
         if(now-lastSegmentationAt>=interval){
          lastSegmentationVideoTime=v.currentTime;lastSegmentationAt=now;segmentationBusy=true;
          try{
           const seg=segmenterRef.current;
           seg.segmentForVideo(v,now,(result)=>{
            const masks=result.confidenceMasks??[];
            try{
             const mask=masks[0];
             if(mask&&cloneSpriteRef.current){
              const first=!cloneSpriteReadyRef.current;
              if(updateCloneSprite(v,mask,cloneSpriteRef.current)){
               cloneSpriteReadyRef.current=true;setCloneStatus('ready');
               if(first&&activeRef.current==='clone'){
                const until=performance.now()+6000;cloneVisualUntilRef.current=until;castUntilRef.current=until;
                if(previewUntilRef.current>0)previewUntilRef.current=until;
               }
              }else if(!cloneSpriteReadyRef.current&&activeRef.current==='clone')setGestureLabel('Shadow Clone · finding your outline, keep your upper body in frame');
             }
            }catch(cause){if(!cloneSpriteReadyRef.current)setGestureLabel('Shadow Clone · retrying cutout: '+describeFailure(cause));}
            finally{
             for(const mask of masks){try{mask.close();}catch{/* already released */}}
             try{result.categoryMask?.close();}catch{/* category mask disabled */}
             segmentationBusy=false;
            }
           });
          }catch(cause){segmentationBusy=false;if(!cloneSpriteReadyRef.current&&activeRef.current==='clone')setGestureLabel('Shadow Clone · retrying segmentation: '+describeFailure(cause));}
         }
        }
        if (overlayDirty) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, w, h);
            const hands = smoothedLandmarksRef.current;
            if (landmarksRef.current) drawLandmarks(ctx, hands, w, h);
          }
          overlayDirty = false;
        }
        threeTrackingRef.current = {
          hands: smoothedLandmarksRef.current,
          effectHands: effectLandmarksRef.current,
          effectHandIndex: effectHandIndexRef.current,
          cloneSprite: cloneSpriteReadyRef.current ? cloneSpriteRef.current : null,
          cloneRevision: cloneSpriteRef.current ? Number(cloneSpriteRef.current.dataset.revision ?? 0) : 0,
          cloneVisibleUntil: cloneVisualUntilRef.current,
        }
        rafRef.current = requestAnimationFrame(frame);
      };
      rafRef.current = requestAnimationFrame(frame);
    } catch (cause) {
      const message = describeFailure(cause);
      if (message.includes('Hand tracking could not initialize')) {
        // Keep the camera preview available for diagnostics; a model download failure should not turn the camera itself off.
        setCameraState('camera-only');
        setTrackingEngine('unavailable');
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
    visionWorkerRef.current?.terminate(); visionWorkerRef.current = null;
    segmenterRef.current?.close(); segmenterRef.current = null;
    audioRef.current?.close().catch(() => undefined);
  }, []);

  const preview = (jutsu: Exclude<Jutsu, 'none'>) => {
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    switchJutsu(jutsu);
    const previewStartedAt = performance.now();
    const cloneWaitLimit = 15000;
    const previewDuration = jutsu === 'clone'
      ? (cloneSpriteReadyRef.current ? 6000 : cloneWaitLimit)
      : 2200;
    previewUntilRef.current = previewStartedAt + previewDuration;
    if (jutsu === 'clone') {
      castUntilRef.current = previewStartedAt + previewDuration;
      cloneVisualUntilRef.current = castUntilRef.current;
    }
    setGestureLabel(jutsu === 'clone'
      ? cameraState === 'ready'
        ? 'Shadow Clone preview · preparing your live cutout'
        : 'Shadow Clone preview · start the camera to clone your image'
      : 'Rasengan preview · swirling chakra');

    const finishPreview = () => {
      const now = performance.now();
      if (jutsu === 'clone' && activeRef.current === 'clone') {
        // Don't let a fixed preview timeout kill the first summon while the
        // segmenter is still preparing. Keep polling briefly, then linger for
        // six seconds from the first valid person cutout.
        const waitingForImage = !cloneSpriteReadyRef.current &&
          cloneStatusRef.current !== 'error' && now - previewStartedAt < cloneWaitLimit;
        if (waitingForImage) {
          const keepAliveUntil = now + 400;
          castUntilRef.current = Math.max(castUntilRef.current, keepAliveUntil);
          cloneVisualUntilRef.current = Math.max(cloneVisualUntilRef.current, keepAliveUntil);
          timeoutRef.current = window.setTimeout(finishPreview, 250);
          return;
        }
        if (cloneSpriteReadyRef.current && now < cloneVisualUntilRef.current) {
          timeoutRef.current = window.setTimeout(finishPreview, Math.min(500, Math.max(50, cloneVisualUntilRef.current - now)));
          return;
        }
      }
      previewUntilRef.current = 0;
      if (activeRef.current === jutsu) switchJutsu('none');
      setGestureLabel(cameraState === 'ready'
        ? jutsu === 'clone' && !cloneSpriteReadyRef.current
          ? 'Shadow Clone could not isolate your image · check Detection status'
          : 'Ready · form Rasengan or Shadow Clone'
        : 'Preview complete · start camera for tracking');
    };
    timeoutRef.current = window.setTimeout(finishPreview, jutsu === 'clone' ? 250 : previewDuration);
  };
  const activeInfo = activeJutsu === 'none' ? null : INFO[activeJutsu];


  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Konoha Jutsu Lab home">
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div><strong>KONOHA <span>JUTSU LAB</span></strong><small>RASENGAN · SHADOW CLONE TRAINING</small></div>
        </a>
        <div className="topbar-right"><div className="build-chip"><span className="status-dot" /> TWO SIGNATURE JUTSU</div>
          <button className="icon-button" onClick={() => setHelpOpen((v) => !v)} aria-label="Toggle instructions"><CircleHelp size={18} /></button>
        </div>
      </header>
      <section className="intro-row" id="top">
        <div><div className="eyebrow"><span className="eyebrow-line" /> YOUR HIDDEN LEAF TRAINING GROUND</div>
          <h1>Bring chakra<br /><span>to life.</span></h1>
          <p className="intro-copy">Enter the Hidden Leaf training ground. Charge a Rasengan that locks onto your hand, then split into true shadow clones cut from your own live image.</p>
        </div>
        <div className="intro-stats">
          <div className="stat-item"><span className="stat-icon"><Hand size={17} /></span><div><b>2 jutsu</b><small>one focused experience</small></div></div>
          <div className="stat-item"><span className="stat-icon violet"><Cpu size={17} /></span><div><b>Live tracking</b><small>hands in real time</small></div></div>
          <div className="stat-item"><span className="stat-icon green"><ShieldCheck size={17} /></span><div><b>Private</b><small>video stays local</small></div></div>
        </div>
      </section>
      {helpOpen && <section className="help-panel"><Info size={18} /><div><strong>Quick start · Your two signature jutsu</strong><p><b>1.</b> Start the camera and keep your hands in frame. <b>2. Rasengan:</b> face one open palm toward the camera and hold it steady while chakra gathers. <b>3. Shadow Clone:</b> bring both hands together and cross the index and middle fingers into a clear plus (+) seal. Keep your upper body visible and give the segmentation model a moment to load. The camera feed is processed in your browser.</p></div><button className="text-button" onClick={() => setHelpOpen(false)}>Got it</button></section>}
      <section className="workspace">
        <div className="camera-column">
          <div className="panel-heading"><div><span className="section-kicker">01 / LIVE INPUT</span><h2>Your training mirror</h2></div>
            <div className={'live-status ' + (cameraState === 'ready' || cameraState === 'camera-only' ? 'is-live' : cameraState === 'error' ? 'is-error' : '')}><span />{cameraState === 'ready' ? 'LIVE TRACKING' : cameraState === 'camera-only' ? 'CAMERA ON · TRACKING OFF' : cameraState === 'loading' ? 'INITIALIZING' : cameraState === 'error' ? 'CAMERA ERROR' : 'STANDBY'}</div>
          </div>
          <div ref={stageRef} className={'camera-stage ' + (activeInfo ? 'effect-' + activeInfo.color : '')}>
            <div className="stage-grid" /><div className="stage-corner corner-tl" /><div className="stage-corner corner-tr" /><div className="stage-corner corner-bl" /><div className="stage-corner corner-br" />
            <div className="stage-topline"><span><Activity size={13} /> INPUT FEED</span><span>{cameraState === 'ready' ? (fps || '—') + ' FPS' : '1280 × 720 TARGET'}</span></div>
            {activeJutsu === 'rasengan' && <div className="chakra-charge-hud" aria-live="polite">
              <div className="charge-hud-top"><span><Zap size={12} /> CHAKRA OUTPUT</span><strong>{chargeLevel}%</strong></div>
              <div className="charge-hud-track"><span style={{ width: chargeLevel + '%' }} /></div>
              <small>{chargeLevel < 35 ? 'GATHERING ENERGY' : chargeLevel < 78 ? 'CHAKRA RESONANCE' : 'MAXIMUM OUTPUT'}</small>
            </div>}
            {activeJutsu === 'clone' && <div className="clone-summon-hud" aria-live="polite"><Sparkles size={13} /><span>SHADOW CLONE</span><small>CHAKRA SPLIT IN PROGRESS</small></div>}
            <div className="video-wrap">
              <video ref={videoRef} className="camera-video" autoPlay playsInline muted aria-label="Mirrored webcam preview" />
              <canvas ref={canvasRef} className="effects-canvas" />
              {(cameraState === 'ready' || activeJutsu !== 'none') && <ThreeEffects trackingRef={threeTrackingRef} activeJutsu={activeJutsu} chargeLevel={chargeLevel} />}
              {cameraState !== 'ready' && cameraState !== 'camera-only' && <div className="camera-placeholder">
                {cameraState === 'loading' ? <LoaderCircle className="spin" size={38} /> : cameraState === 'error' ? <Camera size={35} /> : <div className="camera-orb"><Camera size={28} /></div>}
                <strong>{cameraState === 'loading' ? 'Warming up the chakra flow' : cameraState === 'error' ? (error.includes('hand tracking') ? 'Hand tracking unavailable' : 'Camera not ready') : 'Your training space awaits'}</strong>
                <span>{cameraState === 'loading' ? 'Requesting camera and loading hand tracking…' : cameraState === 'error' ? error : 'Start the camera, then choose Rasengan or Shadow Clone.'}</span>
              </div>}
            </div>
            <div className="stage-bottomline"><span><span className={'small-dot ' + (cameraState === 'ready' ? 'active' : '')} /> {cameraState === 'ready' ? handCount + ' HAND' + (handCount === 1 ? '' : 'S') + ' DETECTED' : cameraState === 'camera-only' ? 'CAMERA ON · TRACKING UNAVAILABLE' : 'AWAITING VIDEO SOURCE'}</span><span className="mono">MP / HAND LANDMARKER</span></div>
          </div>
          <div className="camera-controls">
            {cameraState === 'ready' || cameraState === 'camera-only' ? <button className="primary-button stop-button" onClick={stopCamera}><span className="stop-square" /> Stop camera</button> :
              <button className="primary-button" onClick={startCamera} disabled={cameraState === 'loading'}>{cameraState === 'loading' ? <LoaderCircle className="spin" size={17} /> : <Camera size={17} />}{cameraState === 'loading' ? 'Starting camera…' : 'Start camera'}</button>}
            <button className={'secondary-button ' + (showLandmarks ? 'selected' : '')} onClick={() => setShowLandmarks((v) => !v)}><Hand size={16} /> Landmark overlay <span className="toggle-pill">{showLandmarks ? 'ON' : 'OFF'}</span></button>
            <button className={'secondary-button ' + (soundEnabled ? 'selected' : '')} onClick={() => { const next = !soundEnabled; setSoundEnabled(next); if (next) { try { const audio = audioRef.current ?? new window.AudioContext(); audioRef.current = audio; void audio.resume().then(() => playJutsuCue(audio, 'rasengan')); } catch { /* optional audio */ } } }} title="Turn the chakra sound on or off"><Zap size={16} /> Chakra sound <span className="toggle-pill">{soundEnabled ? 'ON' : 'OFF'}</span></button>
            <button className="reset-button" onClick={() => { switchJutsu('none'); setGestureLabel('Effects cleared'); }} title="Clear current effect"><RotateCcw size={16} /></button>
          </div>
          {(cameraState === 'error' || cameraState === 'camera-only') && <p className="error-hint"><Info size={14} /> {cameraState === 'camera-only' ? 'The camera preview is being kept active. Hand tracking is unavailable until the model/runtime loads; use the GitHub Actions link to check that the newest build was deployed.' : 'Camera access requires browser permission and a secure context (localhost or HTTPS).'}</p>}
        </div>
        <aside className="right-column">
          <div className="panel-heading"><div><span className="section-kicker">02 / YOUR TECHNIQUES</span><h2>Choose your jutsu</h2></div><span className="mini-count">02</span></div>
          <div className="jutsu-list">
            <button className={'jutsu-card blue-card ' + (activeJutsu === 'rasengan' ? 'is-active' : '')} onClick={() => preview('rasengan')}>
              <div className="jutsu-art rasengan-art"><span className="orb-core" /><span className="orb-ring ring-one" /><span className="orb-ring ring-two" /><span className="orb-spark s1" /><span className="orb-spark s2" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">WIND / CHAKRA</div><h3>Rasengan</h3><p>Face your palm forward. The chakra sphere follows your hand and surges as you move.</p><div className="jutsu-foot"><span className="difficulty easy">BEGINNER</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">01</span>
            </button>
            <button className={'jutsu-card violet-card ' + (activeJutsu === 'clone' ? 'is-active' : '')} onClick={() => preview('clone')}>
              <div className="jutsu-art clone-art"><span className="clone-figure figure-one" /><span className="clone-figure figure-two" /><span className="clone-figure figure-three" /><Sparkles className="clone-star" size={17} /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">NINJUTSU / DECOY</div><h3>Shadow Clone</h3><p>Form the crossed-finger seal and summon copies cut from your own live camera image.</p><div className="jutsu-foot"><span className="difficulty advanced">CROSSED SEAL</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">02</span>
            </button>
          </div>
          <div className="detection-card">
            <div className="detection-heading"><div><span className="section-kicker">LIVE READOUT</span><h3>Detection status</h3></div><Activity size={17} /></div>
            <div className="readout-row"><span>Current signal</span><strong className={activeJutsu !== 'none' ? 'signal-active' : ''}>{activeJutsu === 'none' ? 'No active jutsu' : INFO[activeJutsu].name}</strong></div>
            <div className="readout-row"><span>Hand landmarks</span><strong>{cameraState === 'ready' ? (handCount ? handCount + ' hand' + (handCount === 1 ? '' : 's') + ' tracked' : 'Searching…') : 'Not connected'}</strong></div>
            <div className="readout-row"><span>Inference engine</span><strong>{trackingEngine === 'worker-gpu' ? 'Web Worker · GPU' : trackingEngine === 'worker-cpu' ? 'Web Worker · CPU' : trackingEngine === 'main-thread' ? 'Compatibility fallback' : trackingEngine === 'starting' ? 'Initializing…' : trackingEngine === 'unavailable' ? 'Unavailable' : 'Idle'}</strong></div>
            <div className="readout-row"><span>Gesture classifier</span><strong>{cameraState === 'ready' ? 'Running locally' : 'Awaiting camera'}</strong></div>
            <div className="readout-row"><span>Clone cutout</span><strong>{cloneStatus === 'ready' ? 'Your image · ready' : cloneStatus === 'loading' ? 'Preparing your image…' : cloneStatus === 'error' ? 'Image cutout unavailable' : 'Loads on first clone'}</strong></div>
            <div className="signal-track"><span style={{ width: cameraState === 'ready' ? (handCount ? '86%' : '32%') : '5%' }} /></div>
            <p className="signal-caption">{gestureLabel}</p>
          </div>
        </aside>
      </section>
      <section className="bottom-grid">
        <div className="bottom-card mission-card"><div className="bottom-icon"><Lightbulb size={18} /></div><div><div className="section-kicker">YOUR FIRST MISSION</div><h3>Land your first clean cast</h3><p>First, land a clean Rasengan: face one open palm toward the camera and hold steady as the chakra sphere builds. Then summon your own shadow clones: make the classic crossed-finger seal with both hands, keep your upper body in frame, and allow a moment for the cutout model to load.</p></div><ChevronDown size={18} className="muted-chevron" /></div>
        <div className="bottom-card privacy-card"><div className="bottom-icon green-icon"><ShieldCheck size={18} /></div><div><div className="section-kicker">PRIVACY BY DESIGN</div><h3>Your video stays on your device</h3><p>Hand landmarks are processed in the browser. The model is fetched from a public CDN the first time; camera footage is not sent to a server by this prototype.</p></div><Check size={18} className="privacy-check" /></div>
      </section>
      <footer className="footer"><div><span className="footer-mark">木ノ葉</span><span> KONOHA JUTSU LAB</span><span className="footer-dim"> · Early access prototype</span></div><div><span className="footer-dim">Built with MediaPipe · Three.js · WebGL</span><a href="https://github.com/FavorPan/naruto" target="_blank" rel="noreferrer">Inspired by open-source experiments <Expand size={12} /></a></div></footer>
    </main>
  );
}
