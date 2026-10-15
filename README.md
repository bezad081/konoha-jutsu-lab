# Konoha Rasengan Lab

A focused React + TypeScript + Vite browser experience: use one open palm to summon a live Rasengan, charge its chakra, move the sphere with your hand, and release it in an energy burst.

## This version

- One technique only: Rasengan.
- Live mirrored webcam preview with an optional hand-landmark overlay.
- MediaPipe Hand Landmarker processes one hand in the browser, using a Web Worker with GPU/CPU fallback and a main-thread compatibility fallback.
- Open-palm detection with short temporal confirmation to reduce accidental casts without adding much lag.
- Procedural Three.js/WebGL Rasengan: animated shader core, counter-rotating orbit rings, spiral chakra filaments, orbiting particles, a soft energy halo, and a release burst.
- Motion-aware positioning: the orb follows the palm with smoothed velocity prediction and adapts its follow rate to hand speed.
- Adaptive rendering targets 60 FPS and falls back to 30 FPS with a lower pixel ratio on slower devices.
- Camera frames and hand landmarks stay in the browser; video is not uploaded to an application backend.
- Responsive desktop/mobile UI, camera-permission messaging, synthesized chakra sound, and a manual preview.

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

The live Rasengan uses procedural Three.js geometry and shaders; no anime footage or external character sprites are used by the effect. Naruto and Rasengan are franchise terms owned by their respective rights holders. Audit dependencies, branding, and asset licenses before public or commercial release.
