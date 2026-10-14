# Konoha Jutsu Lab — browser prototype

A React + TypeScript + Vite prototype for webcam-based hand tracking and Naruto-inspired chakra effects. This is an independent starting point rather than a copy of the reference repositories.

## Current prototype

- Live webcam preview, mirrored like a selfie camera.
- MediaPipe Hand Landmarker inference in the browser, with GPU and CPU fallback; Face Landmarker supports blink activation and face-anchored effects.
- Hand landmark overlay, low-pass visual landmark smoothing, and stable-pose gating to reduce jitter and accidental casts.
- Reusable, data-driven pose-sequence flow currently powering the guided two-pose Chidori combo: each pose has a visible hold meter, a 0.5-second stability threshold, and a 6.5-second step timeout so an abandoned sequence resets cleanly. This deliberately avoids the open-palm Rasengan gesture.
- Open palm → Rasengan effect that grows from a compact palm-sized orb over roughly 2.6 seconds, with counter-rotating chakra ribbons, spiral filaments and orbiting particles.
- Chidori produces a white-hot electric core, branching jagged bolts, moving sparks, and a short trail. Its short synthesized sound layers bird-like chirps with electric crackle; no spoken callout is used.
- Shadow Clone seal heuristic: two hands close together, index/middle fingers extended in a crossed/plus-like arrangement, outer fingers curled; selfie segmentation creates natural-color cutout copies of the live subject that remain for about six seconds after the seal is released.
- Uchiha Fire Style / Great Fireball: a face-anchored stream of turbulent flames starts at the detected mouth and is followed by a swirling fireball. The original hand-position fireball remains as fallback if face tracking is unavailable.\n- Sharingan eye effect: blink both eyes to toggle a red iris with three black tomoe marks; the overlay follows face/iris landmarks and has a manual toggle.
- Raiton Pulse: two open palms trigger a blue lightning burst.
- Three.js/WebGL now drives the live Rasengan, Chidori, Raiton Pulse, Fireball, Amaterasu, Wind Fang, Sharingan and ninja-weapon visuals. The transparent renderer uses procedural 3D geometry, additive glow layers, animated particles, electric branches and extruded metallic tools.
- Canvas remains for hand-landmark guides and segmentation-based Shadow Clone cutouts, where pixel-level compositing is still the right tool. The primary live technique effects have moved out of the old 2D drawing routines.
- The sound toggle plays a sample when enabled, making it easy to check the audio before casting.
- Preview controls when camera access is unavailable.
- Responsive UI, camera permission errors, camera cleanup, and optional generated sound.
- Selectable virtual Shuriken and Kunai tools rendered over the tracked hand using locally authored metallic SVG illustrations, with improved held and thrown rendering.
- A quick hand movement launches the selected tool as a spinning Canvas projectile with a short trail-like glow and a brief synthesized throw cue when sound is enabled. This is a visual prototype: it does not simulate real object contact or damage.
- Adaptive landmark smoothing damps jitter and responds quickly to fast gestures; hands are matched by handedness before smoothing to reduce jumps when detection order changes.

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

The production build downloads and checksum-verifies the hand model, attempts to package the face model for blink/eye tracking, and ships the MediaPipe WASM runtime from the same site origin. Video frames and detected hand landmarks are processed in the browser in this prototype; they are not uploaded to an application backend. The optional selfie-segmentation model for Shadow Clone is still fetched from Google Storage on first use.

## Known limits / next milestones

1. Shadow Clone uses a geometric crossed-finger plus-seal heuristic rather than a custom-trained gesture model. Hand orientation, camera angle, and lighting can affect recognition; a browser-based training workflow with positive and negative samples would improve reliability.
2. The core technique visuals are now procedural Three.js geometry rather than Canvas drawings. They are an experimental WebGL pass, not final artist-authored 3D assets; advanced bloom/post-processing and richer fire simulation remain future upgrades.
3. Shadow Clone now uses selfie segmentation to draw live cutout echoes. Hair, sleeves, motion blur and background edges can still produce artifacts; this is a prototype effect, not a studio-quality rotoscope.
4. Add per-user calibration, confidence smoothing, pose capture/testing, and device performance profiling.
5. Before public or commercial release, audit dependencies and use original or properly licensed visual/audio assets. Naruto-related branding and character assets may have separate rights.

## Design choices

- React, TypeScript, Vite and CSS for a browser-first UI.
- `@mediapipe/tasks-vision` for hand landmarks and on-demand selfie segmentation.
- Three.js/WebGL for the main live technique effects, particle systems, weapon meshes and eye overlays; Canvas remains for landmarks and segmentation-based clone compositing.
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


## Eye and tool assets\n\n- `public/assets/shuriken.svg` and `public/assets/kunai.svg` are original local illustrations with metallic gradients, edge highlights and grip details. They rotate with the tool during throws.\n- Sharingan is rendered procedurally from face/iris landmarks rather than a static sticker; blink recognition depends on camera quality and model availability.\n\n## Technique updates and inspiration sources

- Live chakra HUD shows cast charge from gathering to maximum output; a motion-reactive energy trail follows the detected hand/palm and changes color by technique. These are original procedural Canvas/CSS effects, not copied sprites.
- Reference-project boundary: FavorPan/naruto and Vaibhavs006/Naruto informed effect layering, motion feedback, and the future seal-training direction. Their assets/models are not silently copied; license and provenance must be checked before any reuse. Jutsu-Hero's non-commercial license is not suitable for an unrestricted commercial release.
- The current prototype still recognizes a small set of geometric poses, not the full canon set of twelve Naruto hand seals. A robust seal system needs a labelled sample workflow and validation across camera angles; we should add this as a dedicated next milestone rather than falsely label a fist/open-palm heuristic as full seal recognition.
- The current-jutsu lock is a deliberate temporal interaction rule: a recognized technique stays active while a hand remains in frame, preventing transient open-palm/fist classifications from switching effects. Leave the camera frame with both hands to reset.
- Chidori uses a two-pose closed-fist combo, plus a directional lightning tail and a brief impact flash/shake.
- Amaterasu is an original Canvas/CSS fan-effect implementation, with animated black flame shapes and violet highlights.
- Public implementation references reviewed for interaction/effect ideas: [FavorPan/naruto](https://github.com/FavorPan/naruto) (MIT; gesture model and smoke/clone effects) and [Vaibhavs006/Naruto](https://github.com/Vaibhavs006/Naruto) (hand pose variety, Canvas effects, screen flash/shake). No third-party character videos or unlicensed sound recordings were copied into this build.
- Character art currently hotlinks anime-style PNG illustrations from third-party image hosts; those image files are not part of this repository.
