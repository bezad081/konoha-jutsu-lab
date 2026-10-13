import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, Camera, Check, ChevronDown, CircleHelp, CirclePlay, Cpu, Expand,
  Hand, Info, Lightbulb, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Zap
} from 'lucide-react';
import { FilesetResolver, HandLandmarker, ImageSegmenter, type HandLandmarkerResult } from '@mediapipe/tasks-vision';

type Jutsu = 'none' | 'rasengan' | 'chidori' | 'clone';
type Gesture = 'none' | 'open' | 'fist' | 'two-open';
type Point = { x: number; y: number; z: number };

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const SEGMENTER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
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
  { gesture: 'open', title: 'Open Palm', instruction: 'Show one open palm' },
  { gesture: 'fist', title: 'Fist', instruction: 'Close one hand into a fist' },
  { gesture: 'two-open', title: 'Double Palm', instruction: 'Show both palms together in frame' },
];

function dist(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * 0.5);
}
function classify(points: Point[]): Gesture {
  if (points.length < 21) return 'none';
  const wrist = points[0];
  const pairs: Array<[number, number]> = [[8,6],[12,10],[16,14],[20,18]];
  const count = pairs.filter(([tip, joint]) => dist(points[tip], wrist) > dist(points[joint], wrist) * 1.12).length;
  if (count >= 3) return 'open';
  if (count <= 1) return 'fist';
  return 'none';
}
function readJutsu(result: HandLandmarkerResult | null): { jutsu: Jutsu; gesture: Gesture } {
  const hands = result?.landmarks ?? [];
  const gestures = hands.map((hand) => classify(hand as Point[]));
  if (hands.length > 1 && gestures.filter((g) => g === 'open').length > 1) return { jutsu: 'clone', gesture: 'two-open' };
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
function drawRasengan(ctx: CanvasRenderingContext2D, p: {x:number;y:number}, w: number, h: number, t: number) {
  const x = p.x * w, y = p.y * h, r = Math.min(w,h) * 0.085;
  ctx.save(); ctx.globalCompositeOperation = 'screen';
  const glow = ctx.createRadialGradient(x,y,r*0.08,x,y,r*1.7);
  glow.addColorStop(0,'rgba(235,255,255,.98)'); glow.addColorStop(.22,'rgba(58,194,255,.8)');
  glow.addColorStop(.65,'rgba(27,113,255,.3)'); glow.addColorStop(1,'rgba(22,98,255,0)');
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x,y,r*1.7,0,Math.PI*2); ctx.fill();
  for (let i=0;i<5;i++) {
    ctx.beginPath(); ctx.strokeStyle = i%2 ? 'rgba(100,190,255,.72)' : 'rgba(235,255,255,.9)';
    ctx.lineWidth = Math.max(1.2,r*.04);
    ctx.ellipse(x,y,r*(.35+i*.13),r*(.55+i*.08),t*(i%2 ? -1 : 1)+i,0,Math.PI*2); ctx.stroke();
  }
  for (let i=0;i<18;i++) {
    const a=t*2.5+i*Math.PI*2/18, rr=r*(.6+.25*Math.sin(t*2+i));
    ctx.fillStyle = i%3 ? '#75dfff' : '#ffffff';
    ctx.beginPath(); ctx.arc(x+Math.cos(a)*rr,y+Math.sin(a)*rr,Math.max(1.5,r*.03),0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}
function drawChidori(ctx: CanvasRenderingContext2D, p: {x:number;y:number}, movement: {x:number;y:number}, w: number, h: number, t: number) {
  const x=p.x*w, y=p.y*h, r=Math.min(w,h)*.095;
  ctx.save(); ctx.globalCompositeOperation='screen'; ctx.shadowColor='#48bfff'; ctx.shadowBlur=24;
  for(let b=0;b<11;b++) {
    const a=b*Math.PI*2/11+t*.7, length=r*(.8+.45*((Math.sin(t*8+b*13)+1)/2));
    ctx.beginPath(); ctx.moveTo(x+Math.cos(a)*r*.12,y+Math.sin(a)*r*.12);
    for(let s=1;s<=5;s++) {
      const k=s/5, jitter=s===5?0:Math.sin(t*15+b*4+s*7)*r*.22;
      ctx.lineTo(x+Math.cos(a)*length*k+Math.cos(a+Math.PI/2)*jitter,y+Math.sin(a)*length*k+Math.sin(a+Math.PI/2)*jitter);
    }
    ctx.strokeStyle=b%2?'rgba(64,176,255,.9)':'rgba(230,250,255,.98)'; ctx.lineWidth=b%3===0?2.6:1.4; ctx.stroke();
  }
  const mag=Math.hypot(movement.x,movement.y)||1, dx=movement.x/mag, dy=movement.y/mag;
  ctx.beginPath(); ctx.moveTo(x+dx*r*.2,y+dy*r*.2); ctx.lineTo(x+dx*r*2.8,y+dy*r*2.8);
  ctx.strokeStyle='rgba(88,197,255,.62)'; ctx.lineWidth=r*.22; ctx.stroke(); ctx.restore();
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
  for (const echo of echoes) {
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
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const activeRef = useRef<Jutsu>('none');
  const audioRef = useRef<AudioContext | null>(null);
  const soundRef = useRef(false);
  const landmarksRef = useRef(true);
  const [cameraState, setCameraState] = useState<'idle'|'loading'|'ready'|'error'>('idle');
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
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      let tracker: HandLandmarker;
      try {
        tracker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' }, runningMode: 'VIDEO', numHands: 2,
          minHandDetectionConfidence: .55, minHandPresenceConfidence: .5, minTrackingConfidence: .5
        });
      } catch {
        tracker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' }, runningMode: 'VIDEO', numHands: 2,
          minHandDetectionConfidence: .55, minHandPresenceConfidence: .5, minTrackingConfidence: .5
        });
      }
      trackerRef.current = tracker; setCameraState('ready'); setGestureLabel('Show an open palm or make a fist');
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
          const detected = readJutsu(latest), count = latest.landmarks?.length ?? 0;
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
              setGestureLabel(detected.gesture === 'two-open' ? 'Two open hands detected' : detected.gesture === 'open' ? 'Open palm detected' : 'Fist detected');
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
          const hands = (latest?.landmarks ?? []) as Point[][];
          if (landmarksRef.current) drawLandmarks(ctx,hands,w,h);
          if (activeRef.current === 'rasengan' && hands.length) drawRasengan(ctx,palm(hands[0]),w,h,now/1000);
          if (activeRef.current === 'chidori' && hands.length) drawChidori(ctx,palm(hands[0]),move,w,h,now/1000);
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
      stopCamera(); setCameraState('error');
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message.includes('NotAllowedError') ? 'Camera permission was blocked. Allow camera access in browser settings, then try again.' :
        message.includes('NotFoundError') ? 'No camera was found. Connect a webcam and try again.' :
        'Could not start the camera or load the hand model. Check your connection and camera permission. Details: ' + message);
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
      {helpOpen && <section className="help-panel"><Info size={18} /><div><strong>Quick start</strong><p>Start the camera and allow permission. Hold one open palm for Rasengan or a fist for Chidori for about half a second. Hold two open palms for Shadow Clone. The browser lazily downloads a selfie-segmentation model and uses it to render two tinted copies of your cutout. Start the three-pose practice chain to test ordered gestures. Video frames stay in the browser.</p></div><button className="text-button" onClick={() => setHelpOpen(false)}>Got it</button></section>}
      <section className="workspace">
        <div className="camera-column">
          <div className="panel-heading"><div><span className="section-kicker">01 / LIVE INPUT</span><h2>Chakra viewport</h2></div>
            <div className={'live-status ' + (cameraState === 'ready' ? 'is-live' : cameraState === 'error' ? 'is-error' : '')}><span />{cameraState === 'ready' ? 'LIVE TRACKING' : cameraState === 'loading' ? 'INITIALIZING' : cameraState === 'error' ? 'CAMERA ERROR' : 'STANDBY'}</div>
          </div>
          <div ref={stageRef} className={'camera-stage ' + (activeInfo ? 'effect-' + activeInfo.color : '')}>
            <div className="stage-grid" /><div className="stage-corner corner-tl" /><div className="stage-corner corner-tr" /><div className="stage-corner corner-bl" /><div className="stage-corner corner-br" />
            <div className="stage-topline"><span><Activity size={13} /> INPUT FEED</span><span>{cameraState === 'ready' ? (fps || '—') + ' FPS' : '1280 × 720 TARGET'}</span></div>
            <div className="video-wrap">
              <video ref={videoRef} className="camera-video" autoPlay playsInline muted aria-label="Mirrored webcam preview" />
              <canvas ref={canvasRef} className="effects-canvas" />
              {cameraState !== 'ready' && <div className="camera-placeholder">
                {cameraState === 'loading' ? <LoaderCircle className="spin" size={38} /> : cameraState === 'error' ? <Camera size={35} /> : <div className="camera-orb"><Camera size={28} /></div>}
                <strong>{cameraState === 'loading' ? 'Warming up the chakra flow' : cameraState === 'error' ? 'Camera not ready' : 'Your training space awaits'}</strong>
                <span>{cameraState === 'loading' ? 'Requesting camera and loading hand tracking…' : cameraState === 'error' ? error : 'Start the camera to see your hands and effects live.'}</span>
              </div>}
            </div>
            <div className="stage-bottomline"><span><span className={'small-dot ' + (cameraState === 'ready' ? 'active' : '')} /> {cameraState === 'ready' ? handCount + ' HAND' + (handCount === 1 ? '' : 'S') + ' DETECTED' : 'AWAITING VIDEO SOURCE'}</span><span className="mono">MP / HAND LANDMARKER</span></div>
          </div>
          <div className="camera-controls">
            {cameraState === 'ready' ? <button className="primary-button stop-button" onClick={stopCamera}><span className="stop-square" /> Stop camera</button> :
              <button className="primary-button" onClick={startCamera} disabled={cameraState === 'loading'}>{cameraState === 'loading' ? <LoaderCircle className="spin" size={17} /> : <Camera size={17} />}{cameraState === 'loading' ? 'Starting camera…' : 'Start camera'}</button>}
            <button className={'secondary-button ' + (showLandmarks ? 'selected' : '')} onClick={() => setShowLandmarks((v) => !v)}><Hand size={16} /> Landmark overlay <span className="toggle-pill">{showLandmarks ? 'ON' : 'OFF'}</span></button>
            <button className={'secondary-button ' + (soundEnabled ? 'selected' : '')} onClick={() => setSoundEnabled((v) => !v)}><Zap size={16} /> Cast sound <span className="toggle-pill">{soundEnabled ? 'ON' : 'OFF'}</span></button>
            <button className="reset-button" onClick={() => { switchJutsu('none'); setGestureLabel('Effects cleared'); }} title="Clear current effect"><RotateCcw size={16} /></button>
          </div>
          {cameraState === 'error' && <p className="error-hint"><Info size={14} /> Camera access requires browser permission and a secure context (localhost or HTTPS).</p>}
        </div>
        <aside className="right-column">
          <div className="panel-heading"><div><span className="section-kicker">02 / TECHNIQUE SELECTOR</span><h2>Choose your jutsu</h2></div><span className="mini-count">03</span></div>
          <div className="jutsu-list">
            <button className={'jutsu-card blue-card ' + (activeJutsu === 'rasengan' ? 'is-active' : '')} onClick={() => preview('rasengan')}>
              <div className="jutsu-art rasengan-art"><span className="orb-core" /><span className="orb-ring ring-one" /><span className="orb-ring ring-two" /><span className="orb-spark s1" /><span className="orb-spark s2" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">WIND / CHAKRA</div><h3>Rasengan</h3><p>Open palm · swirling energy orb</p><div className="jutsu-foot"><span className="difficulty easy">BEGINNER</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">01</span>
            </button>
            <button className={'jutsu-card cyan-card ' + (activeJutsu === 'chidori' ? 'is-active' : '')} onClick={() => preview('chidori')}>
              <div className="jutsu-art lightning-art"><Zap className="lightning-symbol" size={44} /><i className="bolt b1" /><i className="bolt b2" /><i className="bolt b3" /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">LIGHTNING STYLE</div><h3>Chidori</h3><p>Fist · crackling lightning field</p><div className="jutsu-foot"><span className="difficulty medium">INTERMEDIATE</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">02</span>
            </button>
            <button className={'jutsu-card violet-card ' + (activeJutsu === 'clone' ? 'is-active' : '')} onClick={() => preview('clone')}>
              <div className="jutsu-art clone-art"><span className="clone-figure figure-one" /><span className="clone-figure figure-two" /><span className="clone-figure figure-three" /><Sparkles className="clone-star" size={17} /></div>
              <div className="jutsu-copy"><div className="jutsu-overline">NINJUTSU / DECOY</div><h3>Shadow Clone</h3><p>Two open hands · segmented body echoes</p><div className="jutsu-foot"><span className="difficulty advanced">IN DEVELOPMENT</span><span className="card-action">Preview <CirclePlay size={13} /></span></div></div><span className="card-number">03</span>
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
            <p className="combo-note">Starter gesture chain for testing. Traditional Naruto hand seals are a later milestone.</p>
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
