# Konoha Rasengan Lab

A focused React + TypeScript + Vite browser experience: use one open palm to summon a live Rasengan, charge its chakra, move the sphere with your hand, and release it in an energy burst.

## This version

- One technique only: Rasengan.
- Live mirrored webcam preview with an optional hand-landmark overlay.
- MediaPipe Hand Landmarker processes one hand in the browser, using a Web Worker with GPU/CPU fallback and a main-thread compatibility fallback.
- Relaxed-palm detection allows moderate finger bends, one more-curled finger, and palm angles including a true side-on pose. Deeply tucked fingers across most of the hand are required for a fist; a sustained fist releases the orb while brief tracking misses are tolerated.
- Procedural Three.js/WebGL Rasengan: a compact saturated-blue vortex with a bright hotspot, moving specular reflections, four close rotating chakra rings, six spiral filaments, tight orbiting particles, palm wisps, and a soft palm-surface glow. The orb grows in four smooth stages while its rotation speeds up; Bloom is tuned to emphasize the hot core without spreading into a large white disc.
- Palm-aware positioning: the orb anchor is the palm-surface center; its hover offset follows the projected 3D palm normal, moving roughly one compact sphere radius above the hand in near-profile poses. A subtle elliptical glow marks the surface below the orb. The effect uses matched MediaPipe world landmarks and follows hand tilt.
- Adaptive rendering targets 60 FPS and drops to 45 FPS with a reduced pixel ratio when the post-processing workload is heavy.
- Three.js effects load on demand; the main-thread MediaPipe compatibility path is dynamically imported only when the worker cannot run.
- Camera frames and hand landmarks stay in the browser; video is not uploaded to an application backend.
- Responsive desktop/mobile UI, camera-permission messaging, a short summoning whoosh plus continuous synthesized spinning/airflow that rises with chakra charge, and a manual preview.
- The Rasengan technique card attempts to cache an anime still as a same-origin site asset during the build, then falls back across public image hosts if the download is unavailable. Check asset permissions before redistribution or commercial use.

## Run locally

Requires Node.js 20 or later and npm.

```bash
npm install
npm run dev
```

Camera access requires permission and a secure context; localhost is treated as secure by modern browsers.

For a production build:

```bash
npm run build
npm run preview
```

## Performance and runtime

The production build packages the MediaPipe hand model and WASM runtime for same-origin use. Camera capture targets 960×540 at 30 FPS and sends downscaled frames to the tracking worker. Only one hand is tracked; the lightweight version no longer loads a separate body-mask model or runs multi-hand classification.

## GitHub Pages

Pushes to `main` build and deploy `dist/` through GitHub Actions. Set Pages source to **GitHub Actions** in repository settings if needed.

## Design and licensing notes

The live Rasengan effect uses procedural Three.js geometry and shaders; the separate technique-card thumbnail uses an externally hosted anime still. Naruto and Rasengan are franchise terms owned by their respective rights holders. Audit dependencies, branding, and asset licenses before public or commercial release.
