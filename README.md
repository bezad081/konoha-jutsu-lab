# Konoha Jutsu Lab — browser prototype

A React + TypeScript + Vite prototype for webcam-based hand tracking and Naruto-inspired chakra effects. This is an independent starting point rather than a copy of the reference repositories.

## Current prototype

- Live webcam preview, mirrored like a selfie camera.
- MediaPipe Hand Landmarker inference in the browser, with GPU and CPU fallback.
- Hand landmark overlay and stable-pose gating to reduce accidental casts.
- Guided three-pose combo practice (open palm → fist → two open palms), with timed holds, step-by-step feedback, cancel/restart controls, and a Shadow Clone finisher.
- Open palm → Rasengan-style orb effect.
- Fist → Chidori-style lightning effect.
- Two open palms → lazily loaded MediaPipe selfie segmentation creates two tinted, offset cutout echoes of the live camera image for Shadow Clone.
- Preview controls when camera access is unavailable.
- Responsive UI, camera permission errors, camera cleanup, and optional generated sound.

## Run locally

Requirements: Node.js 20 or later and npm.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. Camera access requires permission and a secure context; `localhost` is treated as secure by modern browsers.

For a production build:

```bash
npm run build
npm run preview
```

## Runtime and privacy

The browser downloads the MediaPipe WASM runtime and hand-landmark model from public Google/jsDelivr URLs on first use. Video frames and detected hand landmarks are processed in the browser in this prototype; they are not uploaded to an application backend.

## Known limits / next milestones

1. The current guided combo validates an ordered sequence of broad poses (open palm → fist → two open palms); it is a prototype training chain, not yet a recognizer for the canonical Naruto hand-seal vocabulary.
2. Rasengan and Chidori are 2D Canvas effects attached to a detected hand; they are not production-grade 3D assets.
3. Shadow Clone now uses selfie segmentation to draw live cutout echoes. Hair, sleeves, motion blur and background edges can still produce artifacts; this is a prototype effect, not a studio-quality rotoscope.
4. Add per-user calibration, confidence smoothing, pose capture/testing, and device performance profiling.
5. Before public or commercial release, audit dependencies and use original or properly licensed visual/audio assets. Naruto-related branding and character assets may have separate rights.

## Design choices

- React, TypeScript, Vite and CSS for a browser-first UI.
- `@mediapipe/tasks-vision` for hand landmarks and on-demand selfie segmentation.
- Canvas for procedural effects to avoid a large asset bundle at this stage.
- Relative Vite base path to support subpath hosting.

## GitHub Pages deployment

A deployment workflow is included at `.github/workflows/deploy.yml`. After the project is pushed to `main`, open **Settings → Pages** and set source to **GitHub Actions** if required. The workflow builds the app and deploys `dist/` on pushes to `main` or a manual trigger from Actions.
