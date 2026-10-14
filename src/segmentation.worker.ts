import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

type InitMessage = { type: 'init'; wasmUrls: string[]; modelUrls: string[] };
type FrameMessage = { type: 'frame'; timestamp: number; bitmap: ImageBitmap };
type Incoming = InitMessage | FrameMessage;
type Outgoing =
  | { type: 'ready'; delegate: 'GPU' | 'CPU'; modelUrl: string }
  | { type: 'cutout'; cutout: ImageBitmap; inferenceMs: number }
  | { type: 'skipped' }
  | { type: 'frame-error'; message: string }
  | { type: 'error'; message: string };
type WorkerScope = {
  onmessage: ((event: MessageEvent<Incoming>) => void) | null;
  postMessage(message: Outgoing, transfer?: Transferable[]): void;
};
const scope = self as unknown as WorkerScope;

let segmenter: ImageSegmenter | null = null;
let busy = false;
let lastTimestamp = 0;
let labels: string[] = [];

const describe = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

async function initialize(message: InitMessage) {
  if (typeof OffscreenCanvas === 'undefined') {
    scope.postMessage({ type: 'error', message: 'This browser cannot create transparent person cutouts in a worker.' });
    return;
  }
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
  // Only trust label metadata when it maps one-to-one to the returned masks.
  // getLabels() is an empty array for models without a label map.
  const validLabels = labels.length === masks.length && labels.every((label) => label.trim().length > 0);
  if (validLabels) {
    const personIndex = labels.findIndex((label) => /person|foreground|human/i.test(label));
    if (personIndex >= 0) return masks[personIndex];
    const backgroundIndex = labels.findIndex((label) => /background/i.test(label));
    if (backgroundIndex >= 0 && masks.length > 1) {
      return masks[backgroundIndex === 0 ? 1 : 0];
    }
  }

  if (masks.length === 1) return masks[0];

  // If labels are missing, select the most person-like mask: a foreground
  // mask has much more confidence near the centre than along the frame edges.
  let best = masks[0];
  let bestContrast = -Infinity;
  for (const candidate of masks) {
    const values = candidate.getAsFloat32Array();
    const w = candidate.width, h = candidate.height;
    if (!w || !h || values.length < w * h) continue;
    let centreSum = 0, centreN = 0, edgeSum = 0, edgeN = 0;
    const edgeX = Math.max(1, Math.floor(w * 0.09));
    const edgeY = Math.max(1, Math.floor(h * 0.07));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const value = values[y * w + x];
        const edge = x < edgeX || x >= w - edgeX || y < edgeY || y >= h - edgeY;
        if (edge) { edgeSum += value; edgeN++; }
        else if (x > w * 0.12 && x < w * 0.88 && y > h * 0.06 && y < h * 0.96) {
          centreSum += value; centreN++;
        }
      }
    }
    const contrast = centreSum / Math.max(1, centreN) - edgeSum / Math.max(1, edgeN);
    if (contrast > bestContrast) { bestContrast = contrast; best = candidate; }
  }
  return best;
}

/** Build a compact, cropped RGBA portrait off the main thread.
 * Confidence-to-alpha conversion runs at the model's small mask resolution;
 * Canvas compositing then scales the mask smoothly onto the camera frame.
 */
async function createPersonCutout(source: ImageBitmap, scores: Float32Array, maskWidth: number, maskHeight: number): Promise<ImageBitmap> {
  if (typeof OffscreenCanvas === 'undefined') throw new Error('OffscreenCanvas is unavailable in this browser.');
  if (!maskWidth || !maskHeight || scores.length < maskWidth * maskHeight) throw new Error('The person mask is empty or has invalid dimensions.');

  let minX = maskWidth, minY = maskHeight, maxX = -1, maxY = -1;
  // Use progressively softer thresholds for dim rooms and low-confidence masks.
  for (const threshold of [0.38, 0.25, 0.12]) {
    minX = maskWidth; minY = maskHeight; maxX = -1; maxY = -1;
    for (let y = 0; y < maskHeight; y++) {
      const row = y * maskWidth;
      for (let x = 0; x < maskWidth; x++) {
        if (scores[row + x] > threshold) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX >= minX && maxY >= minY) break;
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
    scope.postMessage({ type: 'skipped' });
    return;
  }

  busy = true;
  const startedAt = performance.now();
  try {
    const timestamp = Math.max(message.timestamp, lastTimestamp + 1);
    lastTimestamp = timestamp;

    segmenter.segmentForVideo(bitmap, timestamp, (result) => {
      let scores: Float32Array | null = null;
      let maskWidth = 0, maskHeight = 0;
      try {
        const masks = result.confidenceMasks ?? [];
        if (!masks.length) throw new Error('Segmentation returned no confidence masks.');
        const mask = selectPersonMask(masks);
        if (!mask) throw new Error('The person mask is missing. Check the loaded segmentation model.');
        // Copy all mask data while inside the callback; MediaPipe documents that
        // result masks are only valid for the duration of this callback.
        scores = new Float32Array(mask.getAsFloat32Array());
        maskWidth = mask.width;
        maskHeight = mask.height;
      } catch (cause) {
        closeMasks(result);
        bitmap.close();
        busy = false;
        scope.postMessage({ type: 'frame-error', message: describe(cause) });
        return;
      }
      closeMasks(result);

      // Canvas work uses the copied scores and is moved to OffscreenCanvas in
      // this worker; the main UI only receives the finished transparent person.
      void createPersonCutout(bitmap, scores, maskWidth, maskHeight)
        .then((cutout) => {
          scope.postMessage({ type: 'cutout', cutout, inferenceMs: performance.now() - startedAt }, [cutout]);
        })
        .catch((cause) => {
          scope.postMessage({ type: 'frame-error', message: describe(cause) });
        })
        .finally(() => {
          bitmap.close();
          busy = false;
        });
    });
  } catch (cause) {
    bitmap.close();
    busy = false;
    scope.postMessage({ type: 'frame-error', message: 'Segmentation frame failed: ' + describe(cause) });
  }
}

scope.onmessage = (event) => {
  if (event.data.type === 'init') void initialize(event.data);
  else processFrame(event.data);
};
