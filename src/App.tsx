import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, Camera, Check, ChevronDown, CircleHelp, CirclePlay, Cpu, Expand,
  Hand, Info, Lightbulb, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Zap
} from 'lucide-react';
import type { HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision';
import type { ThreeTrackingState } from './ThreeEffects';
import { readJutsu, type Gesture, type Jutsu, type Point } from './vision/gestureClassifier';

// Keep the initial screen light: load WebGL effects only when the user previews or casts Rasengan.
const ThreeEffects = lazy(() => import('./ThreeEffects'));

type OneEuroAxis = { raw: number; filtered: number; derivative: number; at: number };
type OneEuroPoint = { x: OneEuroAxis; y: OneEuroAxis; z: OneEuroAxis };
type WorkerHandPayload = { landmarks: Point[][]; handedness?: Array<Array<{ categoryName: string }>>; worldLandmarks?: Point[][] };
type VisionWorkerOutput =
  | { type: 'ready'; delegate: string }
  | { type: 'result'; timestamp: number; hands: WorkerHandPayload }
  | { type: 'tracking-error'; message: string }
  | { type: 'delegate-changed'; delegate: 'GPU' | 'CPU'; message: string }
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
  // The compatibility path is uncommon; defer MediaPipe's main-thread bundle until it is needed.
  const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
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
              numHands: 1,
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
};

// Try public image hosts in sequence so a blocked hotlink does not leave the card blank.
const RASENGAN_IMAGES = [
  // Build-time assets are same-origin on GitHub Pages; missing candidates fall through to public hosts.
  new URL('images/rasengan-reference.png', document.baseURI).href,
  new URL('images/rasengan-reference.jpg', document.baseURI).href,
  new URL('images/rasengan-reference.jpeg', document.baseURI).href,
  'https://zefirka.club/uploads/posts/2022-12/1672261699_75-zefirka-club-p-viuchit-tekhniki-iz-naruto-rasengan-87.png',
  'https://www.looper.com/img/gallery/the-most-important-jutsu-that-naruto-ever-learned/the-rasengan-is-no-slouch-either-1655876917.jpg',
  'https://wallpaper.forfun.com/fetch/f4/f4bc7b8352674e59dbee35fbfb0dcfb8.jpeg',
];

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

function playJutsuCue(audio: AudioContext, _jutsu: 'rasengan') {
  const now = audio.currentTime;
  const master = audio.createGain(); master.gain.setValueAtTime(0.0001, now); master.gain.exponentialRampToValueAtTime(0.26, now + 0.035); master.connect(audio.destination);
  const oscillator = audio.createOscillator(), overtone = audio.createOscillator();
  const toneFilter = audio.createBiquadFilter(), toneGain = audio.createGain();
  oscillator.type = 'sine'; overtone.type = 'triangle'; toneFilter.type = 'lowpass';
  toneFilter.frequency.setValueAtTime(520, now); toneFilter.frequency.exponentialRampToValueAtTime(1450, now + 0.16); toneFilter.frequency.exponentialRampToValueAtTime(360, now + 0.62);
  oscillator.frequency.setValueAtTime(86, now); oscillator.frequency.exponentialRampToValueAtTime(205, now + 0.16); oscillator.frequency.exponentialRampToValueAtTime(74, now + 0.62);
  overtone.frequency.setValueAtTime(172, now); overtone.frequency.exponentialRampToValueAtTime(365, now + 0.16); overtone.frequency.exponentialRampToValueAtTime(118, now + 0.57);
  toneGain.gain.setValueAtTime(0.0001, now); toneGain.gain.exponentialRampToValueAtTime(0.48, now + 0.075); toneGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.64);
  oscillator.connect(toneFilter); overtone.connect(toneFilter); toneFilter.connect(toneGain); toneGain.connect(master);
  const noiseBuffer = audio.createBuffer(1, Math.floor(audio.sampleRate * 0.68), audio.sampleRate);
  const noiseData = noiseBuffer.getChannelData(0);
  for (let i = 0; i < noiseData.length; i++) noiseData[i] = (Math.random() * 2 - 1) * 0.65;
  const noise = audio.createBufferSource(); noise.buffer = noiseBuffer;
  const airFilter = audio.createBiquadFilter(), airGain = audio.createGain();
  airFilter.type = 'bandpass'; airFilter.Q.value = 0.8;
  airFilter.frequency.setValueAtTime(420, now); airFilter.frequency.exponentialRampToValueAtTime(1650, now + 0.14); airFilter.frequency.exponentialRampToValueAtTime(390, now + 0.62);
  airGain.gain.setValueAtTime(0.0001, now); airGain.gain.exponentialRampToValueAtTime(0.15, now + 0.055); airGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.62);
  noise.connect(airFilter); airFilter.connect(airGain); airGain.connect(master);
  oscillator.start(now); overtone.start(now); noise.start(now);
  oscillator.stop(now + 0.66); overtone.stop(now + 0.66); noise.stop(now + 0.66);
}

type RasenganSoundLoop = { setIntensity: (level: number) => void; stop: () => void };

function createRasenganLoop(audio: AudioContext): RasenganSoundLoop {
  const now = audio.currentTime;
  const master = audio.createGain();
  master.gain.setValueAtTime(0.0001, now);
  master.gain.exponentialRampToValueAtTime(0.30, now + 0.32);
  master.connect(audio.destination);

  const low = audio.createOscillator(), rotor = audio.createOscillator();
  const lowGain = audio.createGain(), rotorGain = audio.createGain();
  const rotorFilter = audio.createBiquadFilter();
  low.type = 'sine'; low.frequency.setValueAtTime(58, now); lowGain.gain.value = 0.28;
  rotor.type = 'triangle'; rotor.frequency.setValueAtTime(112, now); rotor.detune.value = -4;
  rotorFilter.type = 'lowpass'; rotorFilter.frequency.value = 740; rotorFilter.Q.value = 0.7; rotorGain.gain.value = 0.13;
  low.connect(lowGain); lowGain.connect(master);
  rotor.connect(rotorFilter); rotorFilter.connect(rotorGain); rotorGain.connect(master);

  const noiseBuffer = audio.createBuffer(1, Math.floor(audio.sampleRate * 2), audio.sampleRate);
  const noiseData = noiseBuffer.getChannelData(0);
  for (let i = 0; i < noiseData.length; i++) noiseData[i] = (Math.random() * 2 - 1) * 0.7;
  const noise = audio.createBufferSource(); noise.buffer = noiseBuffer; noise.loop = true;
  const noiseFilter = audio.createBiquadFilter(), noiseGain = audio.createGain();
  noiseFilter.type = 'bandpass'; noiseFilter.Q.value = 0.72; noiseFilter.frequency.setValueAtTime(560, now);
  noiseGain.gain.value = 0.115; noise.connect(noiseFilter); noiseFilter.connect(noiseGain); noiseGain.connect(master);

  const spinLfo = audio.createOscillator(), spinDepth = audio.createGain();
  spinLfo.type = 'sine'; spinLfo.frequency.setValueAtTime(4.1, now); spinDepth.gain.value = 0.034;
  spinLfo.connect(spinDepth); spinDepth.connect(noiseGain.gain);
  const sweepLfo = audio.createOscillator(), sweepDepth = audio.createGain();
  sweepLfo.type = 'sine'; sweepLfo.frequency.setValueAtTime(0.22, now); sweepDepth.gain.value = 215;
  sweepLfo.connect(sweepDepth); sweepDepth.connect(noiseFilter.frequency);

  const sources: AudioScheduledSourceNode[] = [low, rotor, noise, spinLfo, sweepLfo];
  let stopped = false;
  const setIntensity = (level: number) => {
    if (stopped) return;
    const amount = Math.max(0, Math.min(1, level)), at = audio.currentTime;
    low.frequency.setTargetAtTime(54 + amount * 31, at, 0.10);
    rotor.frequency.setTargetAtTime(104 + amount * 82, at, 0.10);
    noiseFilter.frequency.setTargetAtTime(455 + amount * 525, at, 0.12);
    spinLfo.frequency.setTargetAtTime(3.4 + amount * 4.6, at, 0.10);
    master.gain.setTargetAtTime(0.27 + amount * 0.035, at, 0.13);
  };
  const stop = () => {
    if (stopped) return;
    stopped = true;
    const at = audio.currentTime;
    master.gain.cancelScheduledValues(at);
    master.gain.setTargetAtTime(0.0001, at, 0.045);
    window.setTimeout(() => {
      sources.forEach((source) => { try { source.stop(); } catch { /* already stopped */ } });
      sources.forEach((source) => { try { source.disconnect(); } catch { /* optional cleanup */ } });
      [master, lowGain, rotorGain, rotorFilter, noiseFilter, noiseGain, spinDepth, sweepDepth].forEach((node) => { try { node.disconnect(); } catch { /* optional cleanup */ } });
    }, 320);
  };
  sources.forEach((source) => source.start(now));
  return { setIntensity, stop };
}

function drawLandmarks(ctx: CanvasRenderingContext2D, hands: Point[][], w: number, h: number) {
  ctx.save(); ctx.lineWidth = Math.max(1.5, w * 0.002); ctx.lineCap = 'round';
  for (const hand of hands) {
    ctx.strokeStyle = 'rgba(130,235,255,.62)';
    for (const pair of LINKS) {
      ctx.beginPath(); ctx.moveTo((1 - hand[pair[0]].x) * w, hand[pair[0]].y * h);
      ctx.lineTo((1 - hand[pair[1]].x) * w, hand[pair[1]].y * h); ctx.stroke();
    }
    for (const p of hand) {
      ctx.beginPath(); ctx.fillStyle = '#e5fbff'; ctx.shadowColor = '#47dcff'; ctx.shadowBlur = 7;
      ctx.arc((1 - p.x) * w, p.y * h, Math.max(2, w * 0.003), 0, Math.PI * 2); ctx.fill();
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

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const threeTrackingRef = useRef<ThreeTrackingState>({ hands: [], effectHands: [], worldHands: [], effectHandIndex: 0, sampledAt: 0 });
  const effectHandIndexRef = useRef(0);
  const trackerRef = useRef<HandLandmarker | null>(null);
  const visionWorkerRef = useRef<Worker | null>(null);
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
  const soundLoopRef = useRef<RasenganSoundLoop | null>(null);
  const jutsuStartedAtRef = useRef(0);
  const castUntilRef = useRef(0);
  const soundRef = useRef(false);
  const previewUntilRef = useRef(0);
  const landmarksRef = useRef(true);
  const [cameraState, setCameraState] = useState<'idle'|'loading'|'ready'|'camera-only'|'error'>('idle');
  const [error, setError] = useState('');
  const [activeJutsu, setActiveJutsu] = useState<Jutsu>('none');
  const [chargeLevel, setChargeLevel] = useState(0);
  const [rasenganImageIndex, setRasenganImageIndex] = useState(0);
  const [gestureLabel, setGestureLabel] = useState('Waiting for hands');
  const [handCount, setHandCount] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [fps, setFps] = useState(0);
  const [trackingEngine, setTrackingEngine] = useState<'idle'|'starting'|'worker-gpu'|'worker-cpu'|'main-thread'|'unavailable'>('idle');
  useEffect(() => {
    soundRef.current = soundEnabled;
    if (!soundEnabled) {
      soundLoopRef.current?.stop();
      soundLoopRef.current = null;
      return;
    }
    try {
      const audio = audioRef.current ?? new window.AudioContext();
      audioRef.current = audio;
      void audio.resume().then(() => {
        if (activeRef.current === 'rasengan' && soundRef.current && !soundLoopRef.current) {
          soundLoopRef.current = createRasenganLoop(audio);
          soundLoopRef.current.setIntensity(chargeLevel / 100);
        }
      }).catch(() => undefined);
    } catch { /* optional audio */ }
  }, [soundEnabled]);
  useEffect(() => { soundLoopRef.current?.setIntensity(chargeLevel / 100); }, [chargeLevel]);
  useEffect(() => { landmarksRef.current = showLandmarks; }, [showLandmarks]);
  useEffect(() => {
    if (activeJutsu === 'none') { setChargeLevel(0); return; }
    const tick = () => setChargeLevel(Math.min(100, Math.round(Math.max(0, performance.now() - jutsuStartedAtRef.current) / 1850 * 100)));
    tick();
    const timer = window.setInterval(tick, 50);
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
    if (next === 'none') {
      soundLoopRef.current?.stop();
      soundLoopRef.current = null;
    }
    if(next!=='none'&&soundRef.current)try{
      const audio=audioRef.current??new window.AudioContext();audioRef.current=audio;
      void audio.resume().then(()=>{
        if(activeRef.current!=='rasengan'||!soundRef.current)return;
        playJutsuCue(audio,next);
        soundLoopRef.current?.stop();
        soundLoopRef.current=createRasenganLoop(audio);
        soundLoopRef.current.setIntensity(0);
      }).catch(()=>undefined);
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
    smoothedLandmarksRef.current = []; effectLandmarksRef.current = []; smoothedHandednessRef.current = []; oneEuroFiltersRef.current = [];
    gestureVoteHistoryRef.current = [];
    setTrackingEngine('idle');
    castUntilRef.current = 0;
    if (videoRef.current) videoRef.current.srcObject = null;
    threeTrackingRef.current = { hands: [], effectHands: [], worldHands: [], effectHandIndex: 0 };
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
    stopCamera(); setCameraState('loading'); setError(''); setGestureLabel('Loading the Rasengan hand tracker…'); setTrackingEngine('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false, video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 60, max: 60 } }
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
      setGestureLabel('Camera is on · starting the hand tracker…');
      const cameraStream = stream;
      let worker: Worker | null = null;
      let workerMode = false;
      let workerBusy = false;
      let inferencePending = false;
      let latestTimestamp = 0;
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
              latestTimestamp = message.timestamp;
              inferencePending = true;
              workerBusy = false;
            } else if (message.type === 'delegate-changed') {
              setTrackingEngine(message.delegate === 'GPU' ? 'worker-gpu' : 'worker-cpu');
              setGestureLabel(message.message);
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
              setGestureLabel('Tracking ready · show one open palm for Rasengan');
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
      setGestureLabel('Tracking ready · show one open palm for Rasengan');

      let lastVideoTime = -1;
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
            latestTimestamp = now;
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
          const rawDetected = readJutsu(rawHands, currentHandedness), count = rawHands.length;
          if (rawDetected.jutsu === 'rasengan' && rawDetected.handIndex !== undefined) effectHandIndexRef.current = rawDetected.handIndex;
          const voteHistory = gestureVoteHistoryRef.current;
          voteHistory.push({ ...rawDetected, at: now });
          // A short temporal vote suppresses single-frame flicker without adding much lag.
          while (voteHistory.length > 3 || (voteHistory[0] && now - voteHistory[0].at > 220)) voteHistory.shift();
          const votes = new Map<string, { jutsu: Jutsu; gesture: Gesture; count: number }>();
          for (const sample of voteHistory) {
            const key = sample.jutsu + ':' + sample.gesture;
            const vote = votes.get(key);
            if (vote) vote.count++;
            else votes.set(key, { jutsu: sample.jutsu, gesture: sample.gesture, count: 1 });
          }
          let winner: { jutsu: Jutsu; gesture: Gesture; count: number } | undefined;
          for (const vote of votes.values()) if (!winner || vote.count > winner.count) winner = vote;
          const detected = winner && winner.count >= 2
            ? { jutsu: winner.jutsu, gesture: winner.gesture }
            : { jutsu: 'none' as const, gesture: rawDetected.gesture };
          setHandCount(count);
          if (detected.jutsu === 'rasengan') {
            noPoseAt = 0;
            if (timeoutRef.current !== null) { window.clearTimeout(timeoutRef.current); timeoutRef.current = null; }
            previewUntilRef.current = 0;
            castUntilRef.current = 0;
            if (activeRef.current !== 'rasengan') {
              if (candidate !== 'rasengan') { candidate = 'rasengan'; candidateAt = now; }
              if (now - candidateAt >= 55) {
                switchJutsu('rasengan'); candidate = 'none'; candidateAt = 0;
                setGestureLabel('Rasengan formed · keep your open palm in frame to build chakra');
              } else setGestureLabel('Open palm detected · focus chakra');
            } else {
              candidate = 'none'; candidateAt = 0;
              setGestureLabel('Rasengan charging · move your palm to drive the chakra');
            }
          } else {
            candidate = 'none'; candidateAt = 0;
            if (!noPoseAt) noPoseAt = now;
            const previewStillVisible = now < previewUntilRef.current;
            // Avoid cancelling the orb during a brief angle, occlusion, or motion-blur hiccup.
            // Only a clear, sustained fist should release quickly. Transient angle misses
            // and slightly curled fingers should not cancel an otherwise stable Rasengan.
            const releaseDelay = detected.gesture === 'fist' ? 300 : count > 0 ? 1550 : 1200;
            if (activeRef.current === 'rasengan' && !previewStillVisible && now > castUntilRef.current && now - noPoseAt > releaseDelay) {
              switchJutsu('none'); castUntilRef.current = 0; noPoseAt = 0;
              setGestureLabel('Chakra released · show your palm to form Rasengan again');
            } else if (activeRef.current === 'rasengan') setGestureLabel(detected.gesture === 'fist' ? 'Rasengan released · open your fingers to form it again' : count ? 'Rasengan stable · slight finger bends and palm turns are okay' : 'Rasengan holding briefly · bring your hand back into view');
            else setGestureLabel(count ? 'Relax your fingers and face your palm toward the camera' : 'Move one hand into the frame');
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
          worldHands: (latest?.worldLandmarks ?? []) as Point[][],
          effectHandIndex: effectHandIndexRef.current,
          sampledAt: latestTimestamp,
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
    soundLoopRef.current?.stop();
    soundLoopRef.current = null;
    audioRef.current?.close().catch(() => undefined);
  }, []);

  const preview = (_jutsu: 'rasengan') => {
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    switchJutsu('rasengan');
    const started = performance.now(), duration = cameraState === 'ready' ? 4600 : 4200;
    previewUntilRef.current = started + duration; castUntilRef.current = previewUntilRef.current;
    setGestureLabel(cameraState === 'ready'
      ? 'Rasengan preview · hold an open palm near the camera to lock on'
      : 'Rasengan preview · start the camera to attach chakra to your hand');
    const finishPreview = () => {
      previewUntilRef.current = 0; castUntilRef.current = 0;
      if (activeRef.current === 'rasengan') switchJutsu('none');
      timeoutRef.current = null;
      setGestureLabel(cameraState === 'ready'
        ? 'Ready · face one open palm toward the camera to form Rasengan'
        : 'Preview complete · start the camera for live hand tracking');
    };
    timeoutRef.current = window.setTimeout(finishPreview, duration);
  };
  const activeInfo = activeJutsu === 'none' ? null : INFO[activeJutsu];


  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Konoha Jutsu Lab home">
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div><strong>KONOHA <span>RASENGAN LAB</span></strong><small>RASENGAN · CHAKRA CONTROL</small></div>
        </a>
        <div className="topbar-right"><div className="build-chip"><span className="status-dot" /> ONE SIGNATURE JUTSU</div>
          <button className="icon-button" onClick={() => setHelpOpen((v) => !v)} aria-label="Toggle instructions"><CircleHelp size={18} /></button>
        </div>
      </header>
      <section className="intro-row" id="top">
        <div><div className="eyebrow"><span className="eyebrow-line" /> YOUR HIDDEN LEAF TRAINING GROUND</div>
          <h1>Bring chakra<br /><span>to life.</span></h1>
          <p className="intro-copy">Enter the Hidden Leaf training ground. Form a live Rasengan that locks onto your palm, gathers chakra, and spins faster as you move.</p>
        </div>
        <div className="intro-stats">
          <div className="stat-item"><span className="stat-icon"><Hand size={17} /></span><div><b>1 signature jutsu</b><small>one technique, refined</small></div></div>
          <div className="stat-item"><span className="stat-icon violet"><Cpu size={17} /></span><div><b>Live tracking</b><small>hands in real time</small></div></div>
          <div className="stat-item"><span className="stat-icon green"><ShieldCheck size={17} /></span><div><b>Private</b><small>video stays local</small></div></div>
        </div>
      </section>
      {helpOpen && <section className="help-panel"><Info size={18} /><div><strong>Rasengan · quick start</strong><p><b>1.</b> Start the camera. <b>2.</b> Raise one open hand with your palm visible. <b>3.</b> Hold the pose to gather chakra, then move your hand to make the orb follow. Close your hand or move it out of frame to release the Rasengan. Video stays on your device.</p></div><button className="text-button" onClick={() => setHelpOpen(false)}>Got it</button></section>}
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
              <small>{chargeLevel < 35 ? 'STAGE 1 · GATHER CHAKRA' : chargeLevel < 78 ? 'STAGE 2 · SPIRAL COMPRESSION' : 'STAGE 3 · PERFECT RESONANCE'}</small>
            </div>}
            <div className="video-wrap">
              <video ref={videoRef} className="camera-video" autoPlay playsInline muted aria-label="Mirrored webcam preview" />
              <canvas ref={canvasRef} className="effects-canvas" />
              {(cameraState === 'ready' || activeJutsu !== 'none') && <Suspense fallback={null}><ThreeEffects trackingRef={threeTrackingRef} activeJutsu={activeJutsu} chargeLevel={chargeLevel} /></Suspense>}
              {cameraState !== 'ready' && cameraState !== 'camera-only' && <div className="camera-placeholder">
                {cameraState === 'loading' ? <LoaderCircle className="spin" size={38} /> : cameraState === 'error' ? <Camera size={35} /> : <div className="camera-orb"><Camera size={28} /></div>}
                <strong>{cameraState === 'loading' ? 'Warming up the chakra flow' : cameraState === 'error' ? (error.includes('hand tracking') ? 'Hand tracking unavailable' : 'Camera not ready') : 'Your training space awaits'}</strong>
                <span>{cameraState === 'loading' ? 'Requesting camera and loading hand tracking…' : cameraState === 'error' ? error : 'Start the camera, then preview Rasengan or show an open palm.'}</span>
              </div>}
            </div>
            <div className="stage-bottomline"><span><span className={'small-dot ' + (cameraState === 'ready' ? 'active' : '')} /> {cameraState === 'ready' ? handCount + ' HAND' + (handCount === 1 ? '' : 'S') + ' DETECTED' : cameraState === 'camera-only' ? 'CAMERA ON · TRACKING UNAVAILABLE' : 'AWAITING VIDEO SOURCE'}</span><span className="mono">MP / HAND LANDMARKER</span></div>
          </div>
          <div className="camera-controls">
            {cameraState === 'ready' || cameraState === 'camera-only' ? <button className="primary-button stop-button" onClick={stopCamera}><span className="stop-square" /> Stop camera</button> :
              <button className="primary-button" onClick={startCamera} disabled={cameraState === 'loading'}>{cameraState === 'loading' ? <LoaderCircle className="spin" size={17} /> : <Camera size={17} />}{cameraState === 'loading' ? 'Starting camera…' : 'Start camera'}</button>}
            <button className={'secondary-button ' + (showLandmarks ? 'selected' : '')} onClick={() => setShowLandmarks((v) => !v)}><Hand size={16} /> Landmark overlay <span className="toggle-pill">{showLandmarks ? 'ON' : 'OFF'}</span></button>
            <button className={'secondary-button ' + (soundEnabled ? 'selected' : '')} onClick={() => { const next = !soundEnabled; setSoundEnabled(next); if (next) { try { const audio = audioRef.current ?? new window.AudioContext(); audioRef.current = audio; void audio.resume().then(() => { if (activeRef.current === 'rasengan') { if (!soundLoopRef.current) soundLoopRef.current = createRasenganLoop(audio); soundLoopRef.current.setIntensity(chargeLevel / 100); } else playJutsuCue(audio, 'rasengan'); }).catch(() => undefined); } catch { /* optional audio */ } } }} title="Turn the chakra sound on or off"><Zap size={16} /> Chakra sound <span className="toggle-pill">{soundEnabled ? 'ON' : 'OFF'}</span></button>
            <button className="reset-button" onClick={() => { switchJutsu('none'); setGestureLabel('Rasengan released'); }} title="Clear current effect"><RotateCcw size={16} /></button>
          </div>
          {(cameraState === 'error' || cameraState === 'camera-only') && <p className="error-hint"><Info size={14} /> {cameraState === 'camera-only' ? 'The camera preview is being kept active. Hand tracking is unavailable until the model/runtime loads; use the GitHub Actions link to check that the newest build was deployed.' : 'Camera access requires browser permission and a secure context (localhost or HTTPS).'}</p>}
        </div>
        <aside className="right-column">
          <div className="panel-heading"><div><span className="section-kicker">02 / SIGNATURE TECHNIQUE</span><h2>Rasengan control</h2></div><span className="mini-count">01</span></div>
          <div className="jutsu-list">
            <button className={'jutsu-card blue-card ' + (activeJutsu === 'rasengan' ? 'is-active' : '')} onClick={() => preview('rasengan')}>
              <div className="jutsu-art rasengan-art">
                {rasenganImageIndex < RASENGAN_IMAGES.length
                  ? <img
                      className="rasengan-reference-photo"
                      src={RASENGAN_IMAGES[rasenganImageIndex] as string}
                      alt="Naruto forming a blue Rasengan"
                      loading="eager"
                      decoding="async"
                      referrerPolicy="no-referrer"
                      onError={() => setRasenganImageIndex((index) => Math.min(index + 1, RASENGAN_IMAGES.length))}
                    />
                  : <div className="rasengan-photo-fallback" role="img" aria-label="Glowing blue Rasengan chakra orb"><div className="rasengan-photo-fallback-core" /></div>}
              </div>
              <div className="jutsu-copy"><div className="jutsu-overline">WIND / CHAKRA</div><h3>Rasengan</h3><p>Open your palm to summon chakra. The orb follows your hand, intensifies as it charges, and releases in a burst.</p><div className="jutsu-foot"><span className="difficulty easy">BEGINNER</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">01</span>
            </button>

          </div>
          <div className="detection-card">
            <div className="detection-heading"><div><span className="section-kicker">LIVE READOUT</span><h3>Detection status</h3></div><Activity size={17} /></div>
            <div className="readout-row"><span>Current signal</span><strong className={activeJutsu !== 'none' ? 'signal-active' : ''}>{activeJutsu === 'none' ? 'No active jutsu' : INFO[activeJutsu].name}</strong></div>
            <div className="readout-row"><span>Hand landmarks</span><strong>{cameraState === 'ready' ? (handCount ? handCount + ' hand' + (handCount === 1 ? '' : 's') + ' tracked' : 'Searching…') : 'Not connected'}</strong></div>
            <div className="readout-row"><span>Inference engine</span><strong>{trackingEngine === 'worker-gpu' ? 'Web Worker · GPU' : trackingEngine === 'worker-cpu' ? 'Web Worker · CPU' : trackingEngine === 'main-thread' ? 'Compatibility fallback' : trackingEngine === 'starting' ? 'Initializing…' : trackingEngine === 'unavailable' ? 'Unavailable' : 'Idle'}</strong></div>
            <div className="readout-row"><span>Gesture classifier</span><strong>{cameraState === 'ready' ? 'Running locally' : 'Awaiting camera'}</strong></div>
            <div className="signal-track"><span style={{ width: cameraState === 'ready' ? (handCount ? '86%' : '32%') : '5%' }} /></div>
            <p className="signal-caption">{gestureLabel}</p>
          </div>
        </aside>
      </section>
      <section className="bottom-grid">
        <div className="bottom-card mission-card"><div className="bottom-icon"><Lightbulb size={18} /></div><div><div className="section-kicker">YOUR FIRST MISSION</div><h3>Land your first clean cast</h3><p>Raise one open palm toward the camera and hold it steady while the orb forms. Once it starts glowing, move your hand smoothly: the Rasengan follows your palm, accelerates with your motion, and flashes outward when you release it.</p></div><ChevronDown size={18} className="muted-chevron" /></div>
        <div className="bottom-card privacy-card"><div className="bottom-icon green-icon"><ShieldCheck size={18} /></div><div><div className="section-kicker">PRIVACY BY DESIGN</div><h3>Your video stays on your device</h3><p>Hand landmarks are processed in the browser. The model is fetched from a public CDN the first time; camera footage is not sent to a server by this prototype.</p></div><Check size={18} className="privacy-check" /></div>
      </section>
      <footer className="footer"><div><span className="footer-mark">木ノ葉</span><span> KONOHA RASENGAN LAB</span><span className="footer-dim"> · Early access prototype</span></div><div><span className="footer-dim">ONE HAND · MEDIAPIPE · THREE.JS / WEBGL</span><a href="https://github.com/FavorPan/naruto" target="_blank" rel="noreferrer">Inspired by open-source experiments <Expand size={12} /></a></div></footer>
    </main>
  );
}
