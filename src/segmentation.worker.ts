import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

type InitMessage = { type: 'init'; wasmUrls: string[]; modelUrls: string[] };
type FrameMessage = { type: 'frame'; timestamp: number; bitmap: ImageBitmap };
type Incoming = InitMessage | FrameMessage;
type Outgoing =
  | { type: 'ready'; delegate: 'GPU' | 'CPU'; modelUrl: string }
  | { type: 'cutout'; cutout: ImageBitmap; inferenceMs: number }
  | { type: 'skipped' }
  | { type: 'error'; message: string };
type WorkerScope = {
  onmessage: ((event: MessageEvent<Incoming>) => void) | null;
  postMessage(message: Outgoing, transfer?: Transferable[]): void;
};
const scope = self as unknown as WorkerScope;

let segmenter: ImageSegmenter | null = null;
let busy = false;
let lastTimestamp = 0;
let activeDelegate: 'GPU' | 'CPU' = 'GPU';
let activeModelUrl = '';
let labels: string[] = [];

const describe = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

async function initialize(message: InitMessage) {
  const failures: string[] = [];
  let vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  for (const wasmUrl of message.wasmUrls) {
    try { vision = await FilesetResolver.forVisionTasks(wasmUrl); break; }
    catch (cause) { failures.push('WASM ' + wasmUrl + ': ' + describe(cause)); }
  }
  if (!vision) {
    scope.postMessage({ type: 'error', message: 'Could not load MediaPipe WASM: ' + failures.slice(-3).join(' | ') });
    return;
  }

  // Prefer a model bundled with GitHub Pages; try the public Google-hosted model
  // only if the local file is absent or cannot initialize on this browser.
  for (const modelUrl of message.modelUrls) {
    for (const delegate of ['GPU', 'CPU'] as const) {
      try {
        segmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: modelUrl, delegate },
          runningMode: 'VIDEO',
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        });
        activeDelegate = delegate;
        activeModelUrl = modelUrl;
        labels = segmenter.getLabels();
        scope.postMessage({ type: 'ready', delegate, modelUrl });
        return;
      } catch (cause) {
        failures.push(delegate + ' / ' + modelUrl + ': ' + describe(cause));
        segmenter?.close();
        segmenter = null;
      }
    }
  }
  scope.postMessage({ type: 'error', message: 'Could not initialize person segmentation: ' + failures.slice(-4).join(' | ') });
}

function selectPersonMask(masks: Array<{ width: number; height: number; getAsFloat32Array: () => Float32Array; close: () => void }>) {
  const person = labels.findIndex((label) => /person|foreground|human/i.test(label));
  if (person >= 0 && person < masks.length) return masks[person];
  const nonBackground = labels.findIndex((label) => !/background/i.test(label));
  if (nonBackground >= 0 && nonBackground < masks.length) return masks[nonBackground];
  // The official SelfieSegmenter model has background at index 0 and person at index 1.
  return masks[masks.length > 1 ? 1 : 0];
}

/** Build a compact, cropped RGBA portrait off the main thread.
 * Confidence-to-alpha conversion runs at the model's small mask resolution;
 * Canvas compositing then scales the mask smoothly onto the camera frame.
 */
async function createPersonCutout(source: ImageBitmap, scores: Float32Array, maskWidth: number, maskHeight: number): Promise<ImageBitmap> {
  if (typeof OffscreenCanvas === 'undefined') throw new Error('OffscreenCanvas is unavailable in this browser.');
  if (!maskWidth || !maskHeight || scores.length < maskWidth * maskHeight) throw new Error('The person mask is empty or has invalid dimensions.');

  let minX = maskWidth, minY = maskHeight, maxX = -1, maxY = -1;
  for (let y = 0; y < maskHeight; y++) {
    const row = y * maskWidth;
    for (let x = 0; x < maskWidth; x++) {
      if (scores[row + x] > 0.38) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) {
    throw new Error('No person detected in the segmentation mask. Improve lighting and keep your upper body in frame.');
  }

  // Crop around the detected person so each clone is a person-sized sprite, not
  // a whole transparent copy of the camera frame.
  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;
  const padX = Math.max(5, Math.round(boxWidth * 0.13));
  const padTop = Math.max(4, Math.round(boxHeight * 0.06));
  const padBottom = Math.max(4, Math.round(boxHeight * 0.035));
  const cropLeft = Math.max(0, minX - padX);
  const cropTop = Math.max(0, minY - padTop);
  const cropRight = Math.min(maskWidth, maxX + 1 + padX);
  const cropBottom = Math.min(maskHeight, maxY + 1 + padBottom);
  const cropMaskWidth = cropRight - cropLeft;
  const cropMaskHeight = cropBottom - cropTop;

  const sourceX = cropLeft / maskWidth * source.width;
  const sourceY = cropTop / maskHeight * source.height;
  const sourceWidth = cropMaskWidth / maskWidth * source.width;
  const sourceHeight = cropMaskHeight / maskHeight * source.height;
  const outputScale = Math.min(1, 480 / sourceWidth, 640 / sourceHeight);
  const outputWidth = Math.max(1, Math.round(sourceWidth * outputScale));
  const outputHeight = Math.max(1, Math.round(sourceHeight * outputScale));

  const maskCanvas = new OffscreenCanvas(maskWidth, maskHeight);
  const maskCtx = maskCanvas.getContext('2d');
  if (!maskCtx) throw new Error('Could not create a person-mask canvas.');
  const maskPixels = maskCtx.createImageData(maskWidth, maskHeight);
  for (let i = 0; i < maskWidth * maskHeight; i++) {
    const confidence = clamp((scores[i] - 0.14) / 0.58, 0, 1);
    const alpha = confidence * confidence * (3 - 2 * confidence);
    const offset = i * 4;
    maskPixels.data[offset] = 255;
    maskPixels.data[offset + 1] = 255;
    maskPixels.data[offset + 2] = 255;
    maskPixels.data[offset + 3] = Math.round(alpha * 255);
  }
  maskCtx.putImageData(maskPixels, 0, 0);

  const personCanvas = new OffscreenCanvas(outputWidth, outputHeight);
  const personCtx = personCanvas.getContext('2d');
  if (!personCtx) throw new Error('Could not create a clone image canvas.');
  personCtx.imageSmoothingEnabled = true;
  personCtx.imageSmoothingQuality = 'high';
  personCtx.drawImage(source, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, outputWidth, outputHeight);
  personCtx.globalCompositeOperation = 'destination-in';
  personCtx.drawImage(maskCanvas, cropLeft, cropTop, cropMaskWidth, cropMaskHeight, 0, 0, outputWidth, outputHeight);
  personCtx.globalCompositeOperation = 'source-over';
  return createImageBitmap(personCanvas);
}

function closeMasks(result: { confidenceMasks?: Array<{ close: () => void }> | null; categoryMask?: { close: () => void } | null }) {
  for (const mask of result.confidenceMasks ?? []) {
    try { mask.close(); } catch { /* Ignore an already-released native mask. */ }
  }
  try { result.categoryMask?.close(); } catch { /* Category output is disabled, but close defensively. */ }
}

function processFrame(message: FrameMessage) {
  const bitmap = message.bitmap;
  if (!segmenter || busy) {
    bitmap.close();
    // Every submitted frame must release the main thread's busy gate, even if
    // it arrived during initialization or while an inference was in flight.
    scope.postMessage({ type: 'skipped' });
    return;
  }

  busy = true;
  const startedAt = performance.now();
  try {
    const timestamp = Math.max(message.timestamp, lastTimestamp + 1);
    lastTimestamp = timestamp;
    segmenter.segmentForVideo(bitmap, timestamp, (result) => {
      void (async () => {
        try {
          const masks = result.confidenceMasks ?? [];
          if (!masks.length) throw new Error('Segmentation returned no confidence masks.');
          const mask = selectPersonMask(masks);
          if (!mask) throw new Error('The person mask is missing. Check the loaded segmentation model.');
          const scores = mask.getAsFloat32Array();
          const cutout = await createPersonCutout(bitmap, scores, mask.width, mask.height);
          scope.postMessage({ type: 'cutout', cutout, inferenceMs: performance.now() - startedAt }, [cutout]);
        } catch (cause) {
          scope.postMessage({ type: 'error', message: describe(cause) });
        } finally {
          closeMasks(result);
          bitmap.close();
          busy = false;
        }
      })();
    });
  } catch (cause) {
    bitmap.close();
    busy = false;
    scope.postMessage({ type: 'error', message: 'Segmentation frame failed: ' + describe(cause) });
  }
}

scope.onmessage = (event) => {
  if (event.data.type === 'init') void initialize(event.data);
  else processFrame(event.data);
};
