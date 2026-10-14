import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

type InitMessage = { type: 'init'; wasmUrls: string[]; modelUrl: string };
type FrameMessage = { type: 'frame'; timestamp: number; bitmap: ImageBitmap };
type Incoming = InitMessage | FrameMessage;
type Outgoing =
  | { type: 'ready'; delegate: 'GPU' | 'CPU' }
  | { type: 'mask'; width: number; height: number; scores: Float32Array }
  | { type: 'error'; message: string };
type WorkerScope = {
  onmessage: ((event: MessageEvent<Incoming>) => void) | null;
  postMessage(message: Outgoing, transfer?: Transferable[]): void;
};
const scope = self as unknown as WorkerScope;

let segmenter: ImageSegmenter | null = null;
let busy = false;
let lastTimestamp = 0;

const describe = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);

async function initialize(message: InitMessage) {
  const failures: string[] = [];
  let vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  for (const wasmUrl of message.wasmUrls) {
    try { vision = await FilesetResolver.forVisionTasks(wasmUrl); break; }
    catch (cause) { failures.push(wasmUrl + ': ' + describe(cause)); }
  }
  if (!vision) {
    scope.postMessage({ type: 'error', message: 'Could not load MediaPipe WASM: ' + failures.slice(-3).join(' | ') });
    return;
  }
  for (const delegate of ['GPU', 'CPU'] as const) {
    try {
      segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: message.modelUrl, delegate },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
      scope.postMessage({ type: 'ready', delegate });
      return;
    } catch (cause) {
      failures.push(delegate + ': ' + describe(cause));
    }
  }
  scope.postMessage({ type: 'error', message: 'Could not initialize segmentation: ' + failures.slice(-3).join(' | ') });
}

function processFrame(message: FrameMessage) {
  const bitmap = message.bitmap;
  if (!segmenter || busy) { bitmap.close(); return; }
  busy = true;
  try {
    const timestamp = Math.max(message.timestamp, lastTimestamp + 1);
    lastTimestamp = timestamp;
    segmenter.segmentForVideo(bitmap, timestamp, (result) => {
      try {
        const masks = result.confidenceMasks ?? [];
        if (!masks.length) {
          scope.postMessage({ type: 'error', message: 'Segmentation returned no confidence masks.' });
          return;
        }
        // The landscape selfie model normally returns a single foreground confidence mask.
        // If it returns multiple masks, prefer the first non-background class when labels exist.
        const labels = segmenter?.getLabels() ?? [];
        const foreground = labels.findIndex((label) => /person|foreground|human/i.test(label));
        const nonBackground = labels.findIndex((label) => !/background/i.test(label));
        const maskIndex = foreground >= 0 ? foreground : nonBackground >= 0 ? nonBackground : 0;
        const mask = masks[maskIndex] ?? masks[0];
        const scores = new Float32Array(mask.getAsFloat32Array());
        scope.postMessage({ type: 'mask', width: mask.width, height: mask.height, scores }, [scores.buffer]);
      } catch (cause) {
        scope.postMessage({ type: 'error', message: 'Could not serialize person mask: ' + describe(cause) });
      } finally {
        busy = false;
      }
    });
  } catch (cause) {
    busy = false;
    scope.postMessage({ type: 'error', message: 'Segmentation frame failed: ' + describe(cause) });
  } finally {
    bitmap.close();
  }
}

scope.onmessage = (event) => {
  if (event.data.type === 'init') void initialize(event.data);
  else processFrame(event.data);
};
