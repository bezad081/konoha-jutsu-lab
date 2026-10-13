# Konoha Jutsu Lab — browser prototype

A React + TypeScript + Vite prototype for webcam-based hand tracking and Naruto-inspired chakra effects. This is an independent starting point rather than a copy of the reference repositories.

## Current prototype

- Live webcam preview, mirrored like a selfie camera.
- MediaPipe Hand Landmarker inference in the browser, with GPU and CPU fallback.
- Hand landmark overlay, low-pass visual landmark smoothing, and stable-pose gating to reduce jitter and accidental casts.
- Guided Chidori hand-seal sequence (Ox → Hare → Monkey) with timed holds and step-by-step feedback; webcam poses are simplified geometric approximations of the anime seals.
- Open palm → Rasengan effect that grows from a compact palm-sized orb over roughly 2.6 seconds, with counter-rotating chakra ribbons, spiral filaments and orbiting particles.
- Chidori is released after the three guided seals and produces a white-hot electric core, branching jagged bolts, moving sparks, and a short trail.
- Shadow Clone seal heuristic: two hands close together, index/middle fingers extended in a crossed/plus-like arrangement, outer fingers curled; selfie segmentation creates more visible tinted, offset cutout copies of the live subject with procedural smoke pulses.
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

The production build downloads and checksum-verifies the MediaPipe hand-landmark model, then ships it along with the MediaPipe WASM runtime from the same site origin. Video frames and detected hand landmarks are processed in the browser in this prototype; they are not uploaded to an application backend. The optional selfie-segmentation model for Shadow Clone is still fetched from Google Storage on first use.

## Known limits / next milestones

1. Shadow Clone uses a geometric crossed-finger plus-seal heuristic rather than a custom-trained gesture model. Hand orientation, camera angle, and lighting can affect recognition; a browser-based training workflow with positive and negative samples would improve reliability.
2. Rasengan and Chidori are procedural 2D Canvas effects attached to a detected hand; they are richer and motion-responsive, but not production-grade 3D assets.
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

## Technique inspirations and boundaries

- The cross-seal interaction is a geometric prototype inspired by the custom hand-sign training workflow in [FavorPan/naruto](https://github.com/FavorPan/naruto). That project records positive and negative gesture samples and trains a small browser model; this app does not copy its model or assets. A similar in-app trainer is the next step if the geometric seal detector needs more tolerance.
- The effects use procedural Canvas drawing rather than reusing copyrighted anime footage. The charge ramp and motion-responsive sparks borrow general interaction ideas from open-source hand-controlled particle demos such as [Wlarskog/gesture-particles](https://github.com/Wlarskog/gesture-particles), while keeping this project lightweight and 2D for now.


## Character image credits

- Naruto Uzumaki cosplay photo: Z, [Cosplay - AWA15 - Naruto Uzumaki](https://commons.wikimedia.org/wiki/File:Cosplay_-_AWA15_-_Naruto_Uzumaki_(3982533553).jpg), CC BY 2.0.
- Kakashi Hatake cosplay photo: [Kakashi Hatake Cosplay](https://commons.wikimedia.org/wiki/File:Kakashi_Hatake_Cosplay.jpg), CC0 1.0.
- The images are external Wikimedia Commons assets, not official anime key art. Naruto character names and franchise references belong to their respective rights holders.
