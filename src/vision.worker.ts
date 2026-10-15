import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

type Point = { x: number; y: number; z: number };
type InitMessage = { type: 'init'; wasmUrls: string[]; handModelUrls: string[] };
type FrameMessage = { type: 'frame'; timestamp: number; bitmap: ImageBitmap };
type Incoming = InitMessage | FrameMessage;
type SerializedHands = {
  landmarks: Point[][];
  handedness?: Array<Array<{ categoryName: string }>>;
  worldLandmarks?: Point[][];
};
type Outgoing =
  | { type: 'ready'; delegate: 'GPU' | 'CPU' }
  | { type: 'result'; timestamp: number; hands: SerializedHands }
  | { type: 'tracking-error'; message: string }
  | { type: 'init-error'; message: string };

type WorkerScope = {
  onmessage: ((event: MessageEvent<Incoming>) => void) | null;
  postMessage(message: Outgoing): void;
};
const scope = self as unknown as WorkerScope;

let handTracker: HandLandmarker | null = null;
let vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
let handModelPath = '';
let handDelegate: 'GPU' | 'CPU' = 'GPU';
let lastTimestamp = 0;
let busy = false;

const errorText = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);

async function initialize(message: InitMessage) {
  const failures: string[] = [];
  for (const wasmUrl of message.wasmUrls) {
    try {
      vision = await FilesetResolver.forVisionTasks(wasmUrl);
    } catch (cause) {
      failures.push('WASM ' + wasmUrl + ': ' + errorText(cause));
      continue;
    }
    let loaded = false;
    for (const modelPath of message.handModelUrls) {
      for (const delegate of ['GPU', 'CPU'] as const) {
        try {
          handTracker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: modelPath, delegate },
            runningMode: 'VIDEO',
            numHands: 1,
            minHandDetectionConfidence: 0.55,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          handModelPath = modelPath;
          handDelegate = delegate;
          loaded = true;
          break;
        } catch (cause) {
          failures.push(delegate + ' / ' + modelPath + ': ' + errorText(cause));
        }
      }
      if (loaded) break;
    }
    if (loaded) break;
  }
  if (!handTracker || !vision) {
    scope.postMessage({ type: 'init-error', message: 'Worker hand tracking could not initialize. ' + failures.slice(-5).join(' | ') });
    return;
  }
  scope.postMessage({ type: 'ready', delegate: handDelegate });
}

function serializeHands(result: ReturnType<HandLandmarker['detectForVideo']>): SerializedHands {
  return {
    landmarks: (result.landmarks ?? []).map(hand => hand.map(p => ({ x: p.x, y: p.y, z: p.z }))),
    handedness: (result.handedness ?? []).map(hand => hand.map(c => ({ categoryName: c.categoryName }))),
    worldLandmarks: (result.worldLandmarks ?? []).map(hand => hand.map(p => ({ x: p.x, y: p.y, z: p.z }))),
  };
}

async function recoverGpuFailure() {
  if (!vision || handDelegate !== 'GPU') return false;
  try {
    handTracker?.close();
    handTracker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: handModelPath, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    handDelegate = 'CPU';
    scope.postMessage({ type: 'tracking-error', message: 'GPU inference failed; hand tracking switched to CPU.' });
    return true;
  } catch (cause) {
    scope.postMessage({ type: 'tracking-error', message: 'Hand tracker GPU→CPU fallback failed: ' + errorText(cause) });
    return false;
  }
}

async function processFrame(message: FrameMessage) {
  const bitmap = message.bitmap;
  if (!handTracker || busy) { bitmap.close(); return; }
  busy = true;
  try {
    const timestamp = Math.max(message.timestamp, lastTimestamp + 1);
    lastTimestamp = timestamp;
    let handResult: ReturnType<HandLandmarker['detectForVideo']>;
    try {
      handResult = handTracker.detectForVideo(bitmap, timestamp);
    } catch (cause) {
      const recovered = await recoverGpuFailure();
      if (!recovered || !handTracker) throw cause;
      handResult = handTracker.detectForVideo(bitmap, timestamp);
    }
    scope.postMessage({ type: 'result', timestamp, hands: serializeHands(handResult) });
  } catch (cause) {
    scope.postMessage({ type: 'tracking-error', message: errorText(cause) });
  } finally {
    bitmap.close();
    busy = false;
  }
}

scope.onmessage = (event) => {
  const message = event.data;
  if (message.type === 'init') void initialize(message);
  else void processFrame(message);
};
