# Konoha Rasengan Lab

A focused React + TypeScript + Vite browser experience: use one open palm to summon a live Rasengan, charge its chakra, move the sphere with your hand, and release it in an energy burst.

## This version

- One technique only: Rasengan.
- Live mirrored webcam preview with an optional hand-landmark overlay.
- MediaPipe Hand Landmarker processes one hand in the browser, using a Web Worker with GPU/CPU fallback and a main-thread compatibility fallback.
- Relaxed-palm detection allows slight finger curls and palm angles including a true side-on pose. Clearly back-facing hands are rejected when orientation is measurable. A closed fist never casts, and a detected fist releases an existing orb quickly.
- Procedural Three.js/WebGL Rasengan: a saturated-blue vortex with a bright hotspot, rotating chakra rings, spiral filaments, orbiting particles, palm wisps, and controlled Bloom. The orb grows in four smooth stages from a small core to full size while its spin accelerates.
- Palm-aware positioning: the anchor blends wrist and knuckle landmarks, adds a small offset along the wrist-to-fingertip axis, and aligns the ring motion as the hand turns. This helps the orb stay above the palm instead of slipping to the edge of the hand.
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
