import { useEffect, useRef, type RefObject } from 'react';
import * as T from 'three';

export type EffectPoint = { x: number; y: number; z?: number };
export type ThreeTrackingState = {
  hands: EffectPoint[][];
  effectHands?: EffectPoint[][];
  cloneSprite: HTMLCanvasElement | null;
  cloneRevision: number;
  cloneVisibleUntil: number;
};
type Props = {
  trackingRef: RefObject<ThreeTrackingState>;
  activeJutsu: string;
  chargeLevel: number;
};
type RasenganFx = {
  root: T.Group;
  core: T.Mesh;
  innerCore: T.Mesh;
  rings: T.Mesh[];
  spirals: T.Line[];
  particles: T.Points;
};

const additive = T.AdditiveBlending;
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const basic = (color: number, opacity: number, blending = additive) =>
  new T.MeshBasicMaterial({ color, transparent: true, opacity, blending, depthWrite: false, side: T.DoubleSide });

function makeLine(points: T.Vector3[], color: number, opacity = 0.9) {
  return new T.Line(
    new T.BufferGeometry().setFromPoints(points),
    new T.LineBasicMaterial({ color, transparent: true, opacity, blending: additive, depthWrite: false }),
  );
}

function createRasengan(scene: T.Scene): RasenganFx {
  const root = new T.Group();
  const shader = new T.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uCharge: { value: 0 },
      uBaseColor: { value: new T.Color(0x168dff) },
      uHotColor: { value: new T.Color(0xf0fdff) },
    },
    vertexShader: `varying vec3 vLocalPosition; varying vec3 vNormalView; varying vec3 vViewPosition;
      void main() {
        vLocalPosition = position;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewPosition = -mvPosition.xyz;
        vNormalView = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * mvPosition;
      }`,
    fragmentShader: `uniform float uTime; uniform float uCharge; uniform vec3 uBaseColor; uniform vec3 uHotColor;
      varying vec3 vLocalPosition; varying vec3 vNormalView; varying vec3 vViewPosition;
      void main() {
        vec3 n = normalize(vNormalView);
        vec3 viewDir = normalize(vViewPosition);
        float angle = atan(vLocalPosition.y, vLocalPosition.x);
        float turbulence = sin(vLocalPosition.y * 0.31 + uTime * 2.1)
                        * cos(vLocalPosition.x * 0.24 - uTime * 1.6);
        float phase = angle * 12.0 + vLocalPosition.z * 0.38
                    - uTime * (3.5 + uCharge * 5.0) + turbulence * (1.1 + uCharge * 1.8);
        float bands = 0.5 + 0.5 * sin(phase);
        float filaments = smoothstep(0.64 - uCharge * 0.12, 0.98, bands);
        float fineLines = pow(0.5 + 0.5 * sin(angle * 29.0 - vLocalPosition.z * 0.25 - uTime * 5.0), 10.0);
        float fresnel = pow(1.0 - max(0.0, dot(n, viewDir)), 2.15);
        float core = pow(max(0.0, 1.0 - length(vLocalPosition.xy) * 0.041), 2.8);
        float surge = 0.5 + 0.5 * sin(uTime * (7.0 + uCharge * 5.0) + angle * 4.0);
        vec3 color = mix(uBaseColor, uHotColor,
          clamp(filaments * (0.48 + uCharge * 0.27) + fineLines * 0.2 + fresnel * 0.55 + core * (0.15 + uCharge * 0.3), 0.0, 1.0));
        float alpha = clamp(0.24 + fresnel * 0.55 + filaments * 0.3 + fineLines * 0.13 + core * (0.12 + surge * 0.12), 0.0, 0.96);
        gl_FragColor = vec4(color * (0.7 + fresnel * 1.25 + filaments * 0.65 + uCharge * 0.25), alpha);
      }`,
    transparent: true,
    blending: additive,
    depthWrite: false,
    side: T.DoubleSide,
  });
  const core = new T.Mesh(new T.SphereGeometry(25, 36, 28), shader);
  root.add(core);
  const innerCore = new T.Mesh(new T.SphereGeometry(10, 20, 16), basic(0xd7f8ff, 0.18));
  innerCore.position.z = 2;
  root.add(innerCore);

  const ringSpecs = [
    { radius: 19, tube: 0.85, color: 0xf0fdff, tilt: [0.15, 0.28, 0.18] },
    { radius: 27, tube: 0.7, color: 0x39adff, tilt: [0.8, 0.2, 0.84] },
    { radius: 34, tube: 0.62, color: 0x86dfff, tilt: [1.1, 0.92, 0.2] },
    { radius: 39, tube: 0.48, color: 0xffffff, tilt: [0.35, 1.3, 1.2] },
  ];
  const rings = ringSpecs.map((spec) => {
    const mesh = new T.Mesh(new T.TorusGeometry(spec.radius, spec.tube, 7, 72), basic(spec.color, 0.68));
    mesh.rotation.set(spec.tilt[0], spec.tilt[1], spec.tilt[2]);
    root.add(mesh);
    return mesh;
  });

  const spirals: T.Line[] = [];
  for (let strand = 0; strand < 4; strand++) {
    const points: T.Vector3[] = [];
    for (let i = 0; i < 116; i++) {
      const q = i / 115;
      const angle = q * Math.PI * 6.2 + strand * Math.PI * 0.5;
      const radius = 5 + q * 34;
      points.push(new T.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        Math.sin(angle * 0.46 + strand * 1.7) * (3 + q * 4),
      ));
    }
    const spiral = makeLine(points, strand % 2 ? 0x409cff : 0xe8fbff, strand === 1 ? 0.66 : 0.91);
    spirals.push(spiral);
    root.add(spiral);
  }

  const particleCount = 104;
  const positions = new Float32Array(particleCount * 3);
  const seeds: Array<{ angle: number; radius: number; depth: number; speed: number }> = [];
  for (let i = 0; i < particleCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = 27 + Math.random() * 29;
    const depth = (Math.random() - 0.5) * 14;
    seeds.push({ angle, radius, depth, speed: (Math.random() * 0.55 + 0.35) * (Math.random() < 0.5 ? -1 : 1) });
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = Math.sin(angle) * radius;
    positions[i * 3 + 2] = depth;
  }
  const particleGeometry = new T.BufferGeometry();
  particleGeometry.setAttribute('position', new T.BufferAttribute(positions, 3));
  particleGeometry.userData.seeds = seeds;
  const particles = new T.Points(particleGeometry, new T.PointsMaterial({
    color: 0x9deaff, size: 2.8, transparent: true, opacity: 0.68,
    blending: additive, depthWrite: false, sizeAttenuation: false,
  }));
  root.add(particles);
  scene.add(root);
  return { root, core, innerCore, rings, spirals, particles };
}


export default function ThreeEffects({ trackingRef, activeJutsu, chargeLevel }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef({ activeJutsu, chargeLevel });
  latest.current = { activeJutsu, chargeLevel };

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let renderer: T.WebGLRenderer;
    try {
      renderer = new T.WebGLRenderer({
        alpha: true, antialias: false, powerPreference: 'high-performance', premultipliedAlpha: false,
      });
    } catch {
      element.dataset.renderer = 'unavailable';
      return;
    }
    renderer.setClearColor(0, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1));
    renderer.outputColorSpace = T.SRGBColorSpace;
    renderer.domElement.className = 'three-effects-canvas';
    element.appendChild(renderer.domElement);

    const scene = new T.Scene();
    const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
    camera.position.z = 400;
    const rasengan = createRasengan(scene);

    const cloneGeometry = new T.PlaneGeometry(1, 1);
    const cloneMaterials = [
      new T.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: T.DoubleSide }),
      new T.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: T.DoubleSide }),
    ];
    const glowMaterials = [
      new T.MeshBasicMaterial({ color: 0x39bfff, transparent: true, opacity: 0, blending: additive, depthWrite: false, side: T.DoubleSide }),
      new T.MeshBasicMaterial({ color: 0xffa64d, transparent: true, opacity: 0, blending: additive, depthWrite: false, side: T.DoubleSide }),
    ];
    const glowPlanes = glowMaterials.map((material) => {
      const mesh = new T.Mesh(cloneGeometry, material);
      mesh.visible = false;
      scene.add(mesh);
      return mesh;
    });
    const clonePlanes = cloneMaterials.map((material) => {
      const mesh = new T.Mesh(cloneGeometry, material);
      mesh.visible = false;
      scene.add(mesh);
      return mesh;
    });
    const summonRings = [
      new T.Mesh(new T.TorusGeometry(18, 1.2, 7, 56), basic(0x65e2ff, 0.8)),
      new T.Mesh(new T.TorusGeometry(18, 1.2, 7, 56), basic(0xffb45e, 0.8)),
    ];
    summonRings.forEach((mesh) => { mesh.visible = false; scene.add(mesh); });

    const releaseRing = new T.Mesh(new T.TorusGeometry(13, 1.4, 7, 64), basic(0xbcefff, 0));
    releaseRing.visible = false;
    scene.add(releaseRing);
    const burstCount = 68;
    const burstPositions = new Float32Array(burstCount * 3);
    const burstDirections = Array.from({ length: burstCount }, (_, i) => ({
      angle: (i / burstCount) * Math.PI * 2 + (Math.random() - 0.5) * 0.12,
      speed: 55 + Math.random() * 110,
      depth: (Math.random() - 0.5) * 20,
    }));
    const burstGeometry = new T.BufferGeometry();
    burstGeometry.setAttribute('position', new T.BufferAttribute(burstPositions, 3));
    const burstMaterial = new T.PointsMaterial({
      color: 0x94eaff, size: 3.2, transparent: true, opacity: 0,
      blending: additive, depthWrite: false, sizeAttenuation: false,
    });
    const burstParticles = new T.Points(burstGeometry, burstMaterial);
    burstParticles.visible = false;
    scene.add(burstParticles);

    let cloneTexture: T.CanvasTexture | null = null;
    let cloneSource: HTMLCanvasElement | null = null;
    let lastCloneRevision = -1;
    let cloneEntryAt = 0;
    let wasCloneActive = false;
    let wasCloneTextureReady = false;
    let previousJutsu = 'none';
    let lastRenderAt = 0;
    let wasDrawing = false;
    let frameId = 0;
    let pixelRatio = Math.min(window.devicePixelRatio || 1, 1);
    let slowFrames = 0;
    let fastFrames = 0;
    let targetFrameMs = 1000 / 60;
    let burstStartedAt = -Infinity;
    let burstStrength = 0;
    let burstCenter = new T.Vector3();
    let anchor = new T.Vector3();
    let anchorReady = false;
    let lastObservedTarget = new T.Vector3();
    let lastObservedTargetAt = 0;
    let trackedMotionSpeed = 0;

    const resize = () => {
      const width = Math.max(1, element.clientWidth);
      const height = Math.max(1, element.clientHeight);
      camera.left = -width / 2;
      camera.right = width / 2;
      camera.top = height / 2;
      camera.bottom = -height / 2;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const resizeObserver = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null;
    resizeObserver?.observe(element);
    resize();

    const toScreen = (point: EffectPoint, width: number, height: number) =>
      new T.Vector3((point.x - 0.5) * width, (0.5 - point.y) * height, 0);
    const palmCenter = (hand: EffectPoint[] | undefined) => {
      if (!hand || hand.length < 21) return null;
      const ids = [0, 5, 9, 13, 17];
      return ids.reduce((center, id) => ({
        x: center.x + hand[id].x / ids.length,
        y: center.y + hand[id].y / ids.length,
      }), { x: 0, y: 0 });
    };

    const draw = () => {
      frameId = requestAnimationFrame(draw);
      const now = performance.now();
      if (now - lastRenderAt < targetFrameMs) return;
      const deltaMs = lastRenderAt ? now - lastRenderAt : 16.7;
      lastRenderAt = now;
      const time = now / 1000;
      const width = Math.max(1, element.clientWidth);
      const height = Math.max(1, element.clientHeight);
      const minDimension = Math.min(width, height);
      const state = trackingRef.current;
      const props = latest.current;
      const hands = state?.hands ?? [];
      const cloneUntil = state?.cloneVisibleUntil ?? 0;
      const cloneActive = props.activeJutsu === 'clone' && (cloneUntil === 0 || now < cloneUntil);
      const cloneCanvas = state?.cloneSprite ?? null;
      const scale = minDimension / 360;
      const charge = clamp(props.chargeLevel / 100, 0, 1);

      if (props.activeJutsu === 'rasengan') {
        const effectHands = state?.effectHands?.length ? state.effectHands : hands;
        const rawCenter = palmCenter(effectHands[0]);
        if (rawCenter) {
          const target = toScreen(rawCenter, width, height);
          if (!anchorReady || previousJutsu !== 'rasengan') {
            anchor.copy(target);
            lastObservedTarget.copy(target);
            lastObservedTargetAt = now;
            trackedMotionSpeed = 0;
            anchorReady = true;
          } else {
            // Estimate velocity from the live hand target, not the trailing effect.
            // Fast movement raises the follow rate and keeps the orb attached to the palm.
            const observedDelta = target.distanceTo(lastObservedTarget);
            if (observedDelta > 0.75) {
              const elapsed = Math.max(0.016, (now - lastObservedTargetAt) / 1000);
              trackedMotionSpeed = observedDelta / Math.max(1, minDimension) / elapsed;
              lastObservedTarget.copy(target);
              lastObservedTargetAt = now;
            } else {
              trackedMotionSpeed *= Math.exp(-deltaMs / 140);
            }
            const followRate = clamp(44 + trackedMotionSpeed * 9, 44, 105);
            const follow = 1 - Math.exp(-followRate * Math.max(0.008, deltaMs / 1000));
            anchor.lerp(target, follow);
          }
        }
      }

      if (previousJutsu === 'rasengan' && props.activeJutsu !== 'rasengan' && anchorReady) {
        burstStartedAt = now;
        burstCenter.copy(anchor);
        burstStrength = clamp(charge, 0.2, 1);
        releaseRing.position.copy(burstCenter);
        releaseRing.position.z = 18;
        releaseRing.visible = true;
        burstParticles.position.copy(burstCenter);
        burstParticles.position.z = 19;
        burstParticles.visible = true;
      }
      previousJutsu = props.activeJutsu;

      if (cloneCanvas && cloneCanvas !== cloneSource) {
        cloneTexture?.dispose();
        cloneSource = cloneCanvas;
        cloneTexture = new T.CanvasTexture(cloneCanvas);
        cloneTexture.colorSpace = T.SRGBColorSpace;
        cloneTexture.generateMipmaps = false;
        cloneTexture.minFilter = T.LinearFilter;
        cloneTexture.magFilter = T.LinearFilter;
        [...cloneMaterials, ...glowMaterials].forEach((material) => {
          material.map = cloneTexture;
          material.needsUpdate = true;
        });
        lastCloneRevision = -1;
      }
      if (cloneTexture && (state?.cloneRevision ?? 0) !== lastCloneRevision) {
        cloneTexture.needsUpdate = true;
        lastCloneRevision = state?.cloneRevision ?? 0;
      }
      const hasCloneTexture = !!cloneTexture;
      if (cloneActive && (!wasCloneActive || (hasCloneTexture && !wasCloneTextureReady))) {
        cloneEntryAt = now;
        const centers = [-1, 1];
        summonRings.forEach((ringMesh, i) => {
          ringMesh.position.set(width * centers[i] * 0.29, height * (i === 0 ? -0.005 : 0.005), 14 + i);
          ringMesh.scale.setScalar(scale * 0.3);
          ringMesh.visible = true;
        });
      }
      if (!cloneActive) summonRings.forEach((ringMesh) => { ringMesh.visible = false; });
      wasCloneActive = cloneActive;
      wasCloneTextureReady = hasCloneTexture;

      const entryProgress = cloneActive ? clamp((now - cloneEntryAt) / 680, 0, 1) : 1;
      const entryEase = 1 - Math.pow(1 - entryProgress, 3);
      const cloneOffsets = [-0.29, 0.29];
      const cloneY = [0.006, -0.004];
      const cloneAlpha = [0.92, 0.88];
      const sourceWidth = cloneSource?.width || 3;
      const sourceHeight = cloneSource?.height || 5;
      // The worker returns a cropped person cutout rather than a full camera frame.
      // Size each duplicate by height and preserve the actual person's aspect ratio.
      const clonePlaneHeight = height * 0.84;
      const clonePlaneWidth = clonePlaneHeight * sourceWidth / sourceHeight;
      clonePlanes.forEach((mesh, i) => {
        mesh.visible = cloneActive && hasCloneTexture;
        const offset = cloneOffsets[i] * (0.28 + 0.72 * entryEase);
        const bob = Math.sin(time * (1.4 + i * 0.15) + i) * 0.006;
        mesh.position.set(width * offset, -height * (cloneY[i] + bob), 10.2 + i * 0.01);
        const cloneScale = 0.84 + 0.12 * entryEase;
        mesh.scale.set(clonePlaneWidth * cloneScale, clonePlaneHeight * cloneScale, 1);
        cloneMaterials[i].opacity = cloneAlpha[i] * (0.12 + 0.88 * entryEase);
        glowPlanes[i].visible = mesh.visible;
        glowPlanes[i].position.copy(mesh.position);
        glowPlanes[i].position.z = 9.8 + i * 0.01;
        glowPlanes[i].scale.copy(mesh.scale).multiplyScalar(1.035);
        glowMaterials[i].opacity = 0.2 * (0.12 + 0.88 * entryEase);
      });
      summonRings.forEach((ringMesh, i) => {
        if (!ringMesh.visible) return;
        const material = ringMesh.material as T.MeshBasicMaterial;
        const progress = clamp((now - cloneEntryAt) / 520, 0, 1);
        ringMesh.scale.setScalar(scale * (0.35 + progress * 4.2));
        ringMesh.rotation.z = (i ? -1 : 1) * time * 0.55;
        material.opacity = 0.8 * (1 - progress);
        if (progress >= 1) ringMesh.visible = false;
      });

      const burstProgress = clamp((now - burstStartedAt) / 440, 0, 1);
      const burstActive = now - burstStartedAt < 440;
      if (burstActive) {
        const ringMaterial = releaseRing.material as T.MeshBasicMaterial;
        releaseRing.scale.setScalar(scale * (0.45 + burstProgress * 3.4) * (0.65 + burstStrength * 0.55));
        releaseRing.rotation.z = time * 1.7;
        ringMaterial.opacity = (1 - burstProgress) * (0.24 + burstStrength * 0.55);
        const positions = burstGeometry.getAttribute('position') as T.BufferAttribute;
        const array = positions.array as Float32Array;
        for (let i = 0; i < burstCount; i++) {
          const particle = burstDirections[i];
          const distance = particle.speed * burstProgress * (0.45 + burstStrength * 0.7);
          array[i * 3] = Math.cos(particle.angle) * distance;
          array[i * 3 + 1] = Math.sin(particle.angle) * distance;
          array[i * 3 + 2] = particle.depth * burstProgress;
        }
        positions.needsUpdate = true;
        burstMaterial.opacity = (1 - burstProgress) * (0.28 + burstStrength * 0.48);
        releaseRing.visible = true;
        burstParticles.visible = true;
      } else {
        releaseRing.visible = false;
        burstParticles.visible = false;
        burstMaterialReset(burstParticles);
      }

      const rasenganActive = props.activeJutsu === 'rasengan';
      rasengan.root.visible = rasenganActive;
      if (rasenganActive) {
        const position = anchorReady ? anchor : new T.Vector3(0, 0, 0);
        rasengan.root.position.copy(position);
        rasengan.root.scale.setScalar(scale * (0.62 + charge * 0.53 + Math.sin(time * 8) * (0.008 + charge * 0.018)));
        const shader = rasengan.core.material as T.ShaderMaterial;
        shader.uniforms.uTime.value = time;
        shader.uniforms.uCharge.value = charge;
        rasengan.innerCore.scale.setScalar(0.88 + charge * 0.42 + Math.sin(time * 8.4) * 0.08);
        rasengan.rings.forEach((ringMesh, i) => {
          const direction = i % 2 ? -1 : 1;
          const motionBoost = 1 + clamp(trackedMotionSpeed * 0.7, 0, 1.3);
          ringMesh.rotation.x = time * (0.45 + i * 0.13) * motionBoost + i * 0.7;
          ringMesh.rotation.y = time * direction * (0.62 + charge * 1.45 + i * 0.14) * motionBoost;
          ringMesh.rotation.z = time * (direction * (1.05 + charge * 1.7)) * motionBoost + i * 1.3;
          ringMesh.scale.setScalar(0.7 + charge * 0.35 + Math.sin(time * 6 + i) * 0.035);
          (ringMesh.material as T.MeshBasicMaterial).opacity = 0.36 + charge * 0.45;
        });
        rasengan.spirals.forEach((spiral, i) => {
          spiral.rotation.z = time * (i % 2 ? -2.3 : 2.0) * (0.68 + charge * 0.62) * (1 + clamp(trackedMotionSpeed * 0.65, 0, 1.2));
          spiral.rotation.x = Math.sin(time * 1.3 + i) * 0.28;
          (spiral.material as T.LineBasicMaterial).opacity = 0.5 + charge * 0.43;
        });
        const particles = rasengan.particles;
        particles.rotation.z = -time * (0.45 + charge * 1.2) * (1 + clamp(trackedMotionSpeed * 0.7, 0, 1.3));
        particles.rotation.y = time * (0.35 + charge * 0.5) * (1 + clamp(trackedMotionSpeed * 0.45, 0, 0.9));
        (particles.material as T.PointsMaterial).opacity = 0.34 + charge * 0.52 + Math.sin(time * 9) * 0.05;
        const particlePositions = particles.geometry.getAttribute('position') as T.BufferAttribute;
        const particleArray = particlePositions.array as Float32Array;
        const seeds = particles.geometry.userData.seeds as Array<{ angle: number; radius: number; depth: number; speed: number }>;
        for (let i = 0; i < seeds.length; i++) {
          const seed = seeds[i];
          const angle = seed.angle + time * seed.speed * (0.8 + charge * 1.4);
          const radius = seed.radius * (0.82 + charge * 0.18) + Math.sin(time * 5 + i) * 1.7;
          particleArray[i * 3] = Math.cos(angle) * radius;
          particleArray[i * 3 + 1] = Math.sin(angle) * radius;
          particleArray[i * 3 + 2] = seed.depth + Math.sin(time * 3 + i) * 2.4;
        }
        particlePositions.needsUpdate = true;
      }

      const shouldDraw = rasenganActive || cloneActive || burstActive;
      if (!shouldDraw) {
        if (wasDrawing) {
          renderer.clear();
          wasDrawing = false;
        }
        return;
      }
      wasDrawing = true;
      renderer.render(scene, camera);
      // Count JS scene updates as well as the WebGL submit; adapting only to the
      // GPU call can miss stalls caused by CPU-side particle and clone updates.
      const cost = performance.now() - now;
      if (cost > 21) { slowFrames++; fastFrames = 0; }
      else if (cost < 10) { fastFrames++; slowFrames = 0; }
      else { slowFrames = 0; fastFrames = 0; }
      if (slowFrames >= 3) {
        targetFrameMs = 1000 / 30;
        if (pixelRatio > 0.72) {
          pixelRatio = 0.72;
          renderer.setPixelRatio(pixelRatio);
          resize();
        }
        slowFrames = 0;
        fastFrames = 0;
      } else if (fastFrames >= 90) {
        targetFrameMs = 1000 / 60;
        const bestRatio = Math.min(window.devicePixelRatio || 1, 1);
        if (pixelRatio < bestRatio) {
          pixelRatio = bestRatio;
          renderer.setPixelRatio(pixelRatio);
          resize();
        }
        fastFrames = 0;
      }
    };

    frameId = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      scene.traverse((object) => {
        const mesh = object as T.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const material = mesh.material;
        if (material) {
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material.dispose();
        }
      });
      cloneTexture?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [trackingRef]);

  return <div ref={host} className="three-effects-layer" aria-hidden="true" />;
}

function burstMaterialReset(points: T.Points) {
  (points.material as T.PointsMaterial).opacity = 0;
}
