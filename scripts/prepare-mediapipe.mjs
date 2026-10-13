import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modelPath = path.join(root, 'public', 'models', 'hand_landmarker.task');
const wasmSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmTarget = path.join(root, 'public', 'mediapipe', 'wasm');

// SHA-256 matches the Apache-2.0 MediaPipe Hand Landmarker model mirrored from Google's official model.
const expectedSha256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1';
const modelSources = [
  'https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task',
  'https://huggingface.co/Leo-TX/mediapipe-hand/resolve/main/hand_landmarker.task?download=true',
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
];

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

async function validLocalModel() {
  try {
    const bytes = await readFile(modelPath);
    return bytes.length > 6_000_000 && sha256(bytes) === expectedSha256;
  } catch {
    return false;
  }
}

async function prepareModel() {
  await mkdir(path.dirname(modelPath), { recursive: true });
  if (await validLocalModel()) {
    console.log('Using verified local MediaPipe hand model.');
    return;
  }

  const errors = [];
  for (const url of modelSources) {
    try {
      console.log(`Downloading MediaPipe hand model from ${new URL(url).host}…`);
      const response = await fetch(url, {
        redirect: 'follow',
        signal: AbortSignal.timeout(90_000),
        headers: { 'user-agent': 'Konoha-Jutsu-Lab-build/1.0' },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);

      const bytes = Buffer.from(await response.arrayBuffer());
      if (bytes.length < 6_000_000) {
        throw new Error(`unexpectedly small download (${bytes.length} bytes)`);
      }
      const digest = sha256(bytes);
      if (digest !== expectedSha256) {
        throw new Error(`SHA-256 mismatch: ${digest}`);
      }

      const temporaryPath = `${modelPath}.download`;
      await writeFile(temporaryPath, bytes);
      const { rename } = await import('node:fs/promises');
      await rename(temporaryPath, modelPath);
      console.log(`Verified MediaPipe hand model (${(bytes.length / 1_000_000).toFixed(2)} MB).`);
      return;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      errors.push(`${new URL(url).host}: ${message}`);
      console.warn(`Model source failed: ${errors[errors.length - 1]}`);
    }
  }

  throw new Error(
    'Could not prepare the MediaPipe hand model from any source. Build stopped so a broken camera-tracking site is not deployed.\n' +
    errors.join('\n'),
  );
}

async function prepareWasm() {
  const sourceFiles = await readdir(wasmSource);
  if (!sourceFiles.some((file) => file === 'vision_wasm_internal.js') ||
      !sourceFiles.some((file) => file === 'vision_wasm_internal.wasm')) {
    throw new Error(`Expected MediaPipe WASM files were not found in ${wasmSource}`);
  }

  await rm(wasmTarget, { recursive: true, force: true });
  await mkdir(path.dirname(wasmTarget), { recursive: true });
  await cp(wasmSource, wasmTarget, { recursive: true });

  const copiedFiles = await readdir(wasmTarget);
  console.log(`Copied ${copiedFiles.length} MediaPipe WASM runtime files into the site for same-origin loading.`);
}

await prepareModel();
await prepareWasm();
