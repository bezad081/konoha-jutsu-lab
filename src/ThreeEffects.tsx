import { useEffect, useRef, type RefObject } from 'react';
import * as T from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
export type EffectPoint = { x:number; y:number; z?:number };
export type ThreeTrackingState = { hands:EffectPoint[][]; effectHands?:EffectPoint[][]; effectHandIndex?:number; sampledAt?:number };
type Props = { trackingRef:RefObject<ThreeTrackingState>; activeJutsu:string; chargeLevel:number };
type RasenganFx = { root:T.Group; halo:T.Mesh; shell:T.Mesh; core:T.Mesh; innerCore:T.Mesh; rings:T.Mesh[]; spirals:T.Mesh[]; particles:T.Points; palmStreams:T.Line[] };
const additive=T.AdditiveBlending;
const clamp=(v:number,min:number,max:number)=>Math.max(min,Math.min(max,v));
const basic=(color:number,opacity:number,blending=additive)=>new T.MeshBasicMaterial({color,transparent:true,opacity,blending,depthWrite:false,side:T.DoubleSide});
function makeLine(points: T.Vector3[], color: number, opacity = 0.9): T.Mesh {
  // A slim volumetric tube reads like a strand of chakra instead of a flat debug line.
  const curve = new T.CatmullRomCurve3(points);
  const geometry = new T.TubeGeometry(curve, Math.max(140, points.length), 0.34, 5, false);
  const material = new T.MeshBasicMaterial({
    color, transparent: true, opacity, blending: additive, depthWrite: false, side: T.DoubleSide,
  });
  return new T.Mesh(geometry, material);
}

function createRasengan(scene: T.Scene): RasenganFx {
  const root = new T.Group();
  const shader = new T.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uCharge: { value: 0 },
      uPhase: { value: 0 },
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
    fragmentShader: `uniform float uTime; uniform float uCharge; uniform float uPhase; uniform vec3 uBaseColor; uniform vec3 uHotColor;
      varying vec3 vLocalPosition; varying vec3 vNormalView; varying vec3 vViewPosition;
      float hash31(vec3 p) {
        p = fract(p * vec3(0.1031, 0.1030, 0.0973));
        p += dot(p, p.yxz + 33.33);
        return fract((p.x + p.y) * p.z);
      }
      float noise3(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float x00 = mix(hash31(i + vec3(0.0, 0.0, 0.0)), hash31(i + vec3(1.0, 0.0, 0.0)), f.x);
        float x10 = mix(hash31(i + vec3(0.0, 1.0, 0.0)), hash31(i + vec3(1.0, 1.0, 0.0)), f.x);
        float x01 = mix(hash31(i + vec3(0.0, 0.0, 1.0)), hash31(i + vec3(1.0, 0.0, 1.0)), f.x);
        float x11 = mix(hash31(i + vec3(0.0, 1.0, 1.0)), hash31(i + vec3(1.0, 1.0, 1.0)), f.x);
        return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
      }
      void main() {
        vec3 n = normalize(vNormalView);
        vec3 viewDir = normalize(vViewPosition);
        float angle = atan(vLocalPosition.y, vLocalPosition.x);
        float broadFlow = sin(vLocalPosition.y * 0.31 + uTime * 3.8)
                        * cos(vLocalPosition.x * 0.24 - uTime * 3.1);
        float flowingNoise = noise3(vLocalPosition * 0.11 + vec3(uTime * 0.27, -uTime * 0.19, uTime * 0.33));
        float turbulence = broadFlow * 0.36 + (flowingNoise - 0.5) * (2.1 + uCharge * 1.1);
        float radial=length(vLocalPosition.xy);
        float phase=angle*5.4+radial*0.19+vLocalPosition.z*0.16-uTime*(4.2+uCharge*4.6)+turbulence*(1.0+uCharge*1.3);
        float bands=0.5+0.5*sin(phase+sin(angle*2.0+uTime*1.2)*0.42);
        float filaments=smoothstep(0.69-uCharge*0.10,0.98,bands);
        float fineLines=pow(0.5+0.5*sin(angle*11.0+radial*0.34-uTime*5.5),13.0);
        float fresnel=pow(1.0-max(0.0,dot(n,viewDir)),2.0);
        float core=pow(max(0.0,1.0-radial*0.052),7.5);
        float surge=0.5+0.5*sin(uTime*(8.0+uCharge*5.0)+angle*3.0);
        float ultimateGlow=smoothstep(1.4,2.0,uPhase);
        vec3 deepBlue=vec3(0.008,0.055,0.28);
        vec3 color=mix(deepBlue,uBaseColor,clamp(0.18+bands*0.74+fresnel*0.12,0.0,1.0));
        color=mix(color,vec3(0.25,0.79,1.0),clamp(filaments*0.66+fineLines*0.12,0.0,0.76));
        color=mix(color,uHotColor,clamp(core*0.78+ultimateGlow*0.12,0.0,0.82));
        float alpha=clamp(0.26+fresnel*0.34+filaments*0.25+fineLines*0.08+core*(0.12+surge*0.08)+ultimateGlow*0.05,0.0,0.79);
        gl_FragColor=vec4(color*(0.74+fresnel*0.24+filaments*0.72+uCharge*0.12+core*0.40+ultimateGlow*0.14),alpha);
      }`,
    transparent: true,
    blending: additive,
    depthWrite: false,
    side: T.DoubleSide,
  });
  const haloMaterial = new T.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uCharge: { value: 0 }, uTint: { value: new T.Color(0x167fff) } },
    vertexShader: `varying vec2 vUv; void main() { vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: `uniform float uTime; uniform float uCharge; uniform vec3 uTint; varying vec2 vUv;
      void main() {
        vec2 p=(vUv-0.5)*2.0; float r=length(p);
        float auraRing=exp(-pow((r-(0.69+sin(uTime*2.2)*0.012))/0.19,2.0));
        float innerAura=exp(-r*r*7.0); float edgeFade=1.0-smoothstep(0.82,1.0,r);
        float alpha=(auraRing*(0.105+uCharge*0.045)+innerAura*0.045)*edgeFade;
        vec3 color=uTint*(0.84+auraRing*0.34+uCharge*0.20);
        gl_FragColor=vec4(color,alpha);
      }`,
    transparent:true,blending:additive,depthWrite:false,side:T.DoubleSide,
  });
  const halo = new T.Mesh(new T.PlaneGeometry(76, 76), haloMaterial);
  halo.position.z=-4;
  root.add(halo);
  const shellShader = new T.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uCharge: { value: 0 } },
    vertexShader: `varying vec3 vLocalPosition; varying vec3 vNormalView; varying vec3 vViewPosition;
      void main() {
        vLocalPosition = position;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewPosition = -mvPosition.xyz;
        vNormalView = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * mvPosition;
      }`,
    fragmentShader: `uniform float uTime; uniform float uCharge;
      varying vec3 vLocalPosition; varying vec3 vNormalView; varying vec3 vViewPosition;
      float hash31(vec3 p) {
        p = fract(p * vec3(0.1031, 0.1030, 0.0973));
        p += dot(p, p.yxz + 33.33);
        return fract((p.x + p.y) * p.z);
      }
      float noise3(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float x00 = mix(hash31(i + vec3(0.0, 0.0, 0.0)), hash31(i + vec3(1.0, 0.0, 0.0)), f.x);
        float x10 = mix(hash31(i + vec3(0.0, 1.0, 0.0)), hash31(i + vec3(1.0, 1.0, 0.0)), f.x);
        float x01 = mix(hash31(i + vec3(0.0, 0.0, 1.0)), hash31(i + vec3(1.0, 0.0, 1.0)), f.x);
        float x11 = mix(hash31(i + vec3(0.0, 1.0, 1.0)), hash31(i + vec3(1.0, 1.0, 1.0)), f.x);
        return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
      }
      void main() {
        vec3 n = normalize(vNormalView);
        vec3 viewDir = normalize(vViewPosition);
        float angle = atan(vLocalPosition.y, vLocalPosition.x);
        float broadFlow = sin(vLocalPosition.y * 0.32 + uTime * 4.2) * cos(vLocalPosition.x * 0.27 - uTime * 3.3);
        float flowingNoise = noise3(vLocalPosition * 0.105 + vec3(uTime * 0.24, -uTime * 0.32, uTime * 0.41));
        float turbulence = broadFlow * 0.24 + (flowingNoise - 0.5) * (2.4 + uCharge * 0.9);
        float swirl = sin(angle * 10.0 + vLocalPosition.z * 0.55 - uTime * (6.5 + uCharge * 9.0) + turbulence * 1.35);
        float streak = pow(max(0.0, 0.5 + 0.5 * swirl), 7.0);
        float fresnel = pow(1.0 - max(0.0, dot(n, viewDir)), 2.0);
        float hot = clamp(streak * 0.68 + fresnel * 0.45 + uCharge * 0.18, 0.0, 1.0);
        vec3 color = mix(vec3(0.035, 0.24, 0.92), vec3(0.55, 0.94, 1.0), hot);
        float alpha = clamp(fresnel * 0.32 + streak * (0.12 + uCharge * 0.18) + uCharge * 0.035, 0.0, 0.52);
        gl_FragColor = vec4(color * (1.1 + streak * 0.85 + uCharge * 0.35), alpha);
      }`,
    transparent: true,
    blending: additive,
    depthWrite: false,
    side: T.DoubleSide,
  });
  const shell = new T.Mesh(new T.SphereGeometry(33, 28, 22), shellShader);
  root.add(shell);
  const core = new T.Mesh(new T.SphereGeometry(23, 40, 32), shader);
  root.add(core);
  const innerCore = new T.Mesh(new T.SphereGeometry(6.4, 24, 20), basic(0xe8fcff, 0.13));
  innerCore.position.z = 8;
  root.add(innerCore);

  const ringSpecs = [
    { radius: 16.5, tube: 0.58, color: 0xdefaff, tilt: [0.15, 0.28, 0.18] },
    { radius: 21.5, tube: 0.52, color: 0x248dff, tilt: [0.8, 0.2, 0.84] },
    { radius: 26.0, tube: 0.45, color: 0x58cfff, tilt: [1.1, 0.92, 0.2] },
    { radius: 29.0, tube: 0.36, color: 0xbdefff, tilt: [0.35, 1.3, 1.2] },
  ];
  const rings = ringSpecs.map((spec) => {
    const mesh = new T.Mesh(new T.TorusGeometry(spec.radius, spec.tube, 7, 72), basic(spec.color, 0.68));
    mesh.rotation.set(spec.tilt[0], spec.tilt[1], spec.tilt[2]);
    root.add(mesh);
    return mesh;
  });

  const spirals: T.Mesh[] = [];
  for (let strand = 0; strand < 6; strand++) {
    const points: T.Vector3[] = [];
    for (let i = 0; i < 136; i++) {
      const q = i / 135;
      const angle = q * Math.PI * 8.0 + strand * Math.PI / 3;
      const radius = 7 + q * 21;
      points.push(new T.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        Math.sin(angle * 0.46 + strand * 1.7) * (2 + q * 2.8),
      ));
    }
    const spiral = makeLine(points, strand % 2 ? 0x409cff : 0xe8fbff, strand === 1 ? 0.66 : 0.91);
    spirals.push(spiral);
    root.add(spiral);
  }

  // Fine energy streams sit on the front hemisphere near the knuckles.
  const palmStreams: T.Line[] = [];
  for (let i = 0; i < 3; i++) {
    const geometry = new T.BufferGeometry();
    const attribute = new T.BufferAttribute(new Float32Array(9 * 3), 3);
    attribute.setUsage(T.DynamicDrawUsage);
    geometry.setAttribute('position', attribute);
    const material = new T.LineBasicMaterial({
      color: i === 1 ? 0xe8fbff : 0x64cfff, transparent: true, opacity: 0.42,
      blending: additive, depthTest: false, depthWrite: false,
    });
    const stream = new T.Line(geometry, material);
    stream.frustumCulled = false;
    stream.renderOrder = 5;
    stream.visible = false;
    root.add(stream);
    palmStreams.push(stream);
  }

  const particleCount = 104;
  const positions = new Float32Array(particleCount * 3);
  const seeds: Array<{ angle: number; radius: number; depth: number; speed: number }> = [];
  for (let i = 0; i < particleCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = 22 + Math.random() * 10;
    const depth = (Math.random() - 0.5) * 9;
    seeds.push({ angle, radius, depth, speed: (Math.random() * 0.55 + 0.35) * (Math.random() < 0.5 ? -1 : 1) });
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = Math.sin(angle) * radius;
    positions[i * 3 + 2] = depth;
  }
  const particleGeometry = new T.BufferGeometry();
  particleGeometry.setAttribute('position', new T.BufferAttribute(positions, 3));
  particleGeometry.userData.seeds = seeds;
  const particles = new T.Points(particleGeometry, new T.PointsMaterial({
    color: 0x9deaff, size: 2.1, transparent: true, opacity: 0.52,
    blending: additive, depthWrite: false, sizeAttenuation: false,
  }));
  root.add(particles);
  scene.add(root);
  return { root, halo, shell, core, innerCore, rings, spirals, particles, palmStreams };
}

export default function ThreeEffects({trackingRef,activeJutsu,chargeLevel}:Props){
 const host=useRef<HTMLDivElement>(null),latest=useRef({activeJutsu,chargeLevel});latest.current={activeJutsu,chargeLevel};
 useEffect(()=>{
  const element=host.current;if(!element)return;let renderer:T.WebGLRenderer;
  try{renderer=new T.WebGLRenderer({alpha:true,antialias:false,powerPreference:'high-performance',premultipliedAlpha:false})}catch{element.dataset.renderer='unavailable';return}
  renderer.setClearColor(0x000000,0);renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1));renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.12;renderer.domElement.className='three-effects-canvas';element.appendChild(renderer.domElement);
  const scene=new T.Scene(),camera=new T.OrthographicCamera(-1,1,1,-1,.1,1000);camera.position.z=400;
  const rasengan=createRasengan(scene),releaseRing=new T.Mesh(new T.TorusGeometry(13,1.4,7,64),basic(0xbcefff,0));releaseRing.visible=false;scene.add(releaseRing);
  const releaseRing2=new T.Mesh(new T.TorusGeometry(8,0.72,6,64),basic(0x62cfff,0));releaseRing2.visible=false;scene.add(releaseRing2);
  const impactFlash=new T.Mesh(new T.SphereGeometry(13,18,14),basic(0xf0fdff,0));impactFlash.visible=false;scene.add(impactFlash);
  const count=84,burstGeometry=new T.BufferGeometry();burstGeometry.setAttribute('position',new T.BufferAttribute(new Float32Array(count*3),3));
  const directions=Array.from({length:count},(_,i)=>({angle:i/count*Math.PI*2+(Math.random()-.5)*.12,speed:70+Math.random()*135,depth:(Math.random()-.5)*24}));
  const burstMaterial=new T.PointsMaterial({color:0x94eaff,size:3.4,transparent:true,opacity:0,blending:additive,depthWrite:false,sizeAttenuation:false});
  const burstParticles=new T.Points(burstGeometry,burstMaterial);burstParticles.visible=false;scene.add(burstParticles);

  // Reused point buffers give the moving orb an energy trail without per-frame allocations.
  const trailCount=28,trailPositions=new Float32Array(trailCount*3),trailLife=new Float32Array(trailCount),trailSizes=new Float32Array(trailCount);
  trailSizes.fill(4.2);
  const trailGeometry=new T.BufferGeometry();
  const trailPositionAttribute=new T.BufferAttribute(trailPositions,3);trailPositionAttribute.setUsage(T.DynamicDrawUsage);trailGeometry.setAttribute('position',trailPositionAttribute);
  const trailLifeAttribute=new T.BufferAttribute(trailLife,1);trailLifeAttribute.setUsage(T.DynamicDrawUsage);trailGeometry.setAttribute('aLife',trailLifeAttribute);
  const trailSizeAttribute=new T.BufferAttribute(trailSizes,1);trailGeometry.setAttribute('aSize',trailSizeAttribute);
  const trailMaterial=new T.ShaderMaterial({
   transparent:true,blending:additive,depthWrite:false,
   vertexShader:`attribute float aLife;attribute float aSize;varying float vLife;
    void main(){vLife=aLife;vec4 mvPosition=modelViewMatrix*vec4(position,1.0);gl_PointSize=aSize*(0.35+aLife*1.25);gl_Position=projectionMatrix*mvPosition;}`,
   fragmentShader:`varying float vLife;
    void main(){vec2 p=gl_PointCoord-vec2(0.5);float r=length(p);float glow=1.0-smoothstep(0.04,0.5,r);float core=1.0-smoothstep(0.0,0.16,r);float alpha=glow*vLife*0.62;gl_FragColor=vec4(vec3(0.24,0.72,1.0)*(0.85+core*1.2),alpha);}`,
  });
  const trail=new T.Points(trailGeometry,trailMaterial);trail.visible=false;trail.frustumCulled=false;scene.add(trail);

  // Bloom is applied only to this VFX canvas; CSS screen blending lets the webcam pass through the black areas.
  const composer=new EffectComposer(renderer);
  composer.setPixelRatio(renderer.getPixelRatio());
  const bloomPass=new UnrealBloomPass(new T.Vector2(1,1),0.52,0.52,0.32);
  const renderPass=new RenderPass(scene,camera);
  const outputPass=new OutputPass();
  composer.addPass(renderPass);
  composer.addPass(bloomPass);
  composer.addPass(outputPass);

  let raf=0,lastAt=0,targetMs=1000/60,slow=0,fast=0,pixelRatio=Math.min(window.devicePixelRatio||1,1),wasDrawing=false,previous='none',burstAt=-Infinity,burstPower=0,lastCharge=0,chargePhaseLast=-1,trailClock=0,trailCursor=0;
  const burstCenter=new T.Vector3(),anchor=new T.Vector3(),observed=new T.Vector3(),predicted=new T.Vector3(),velocity=new T.Vector3(),targetPoint=new T.Vector3(),motionDelta=new T.Vector3(),lastTrailAnchor=new T.Vector3(1e6,1e6,1e6);let anchorReady=false,observedAt=0,motionSpeed=0;
  const resize=()=>{const w=Math.max(1,element.clientWidth),h=Math.max(1,element.clientHeight);camera.left=-w/2;camera.right=w/2;camera.top=h/2;camera.bottom=-h/2;camera.updateProjectionMatrix();composer.setSize(w,h)};
  const ro=typeof ResizeObserver!=='undefined'?new ResizeObserver(resize):null;ro?.observe(element);resize();
  const palmCenter=(hand:EffectPoint[]|undefined)=>{if(!hand||hand.length<21)return null;return [0,5,9,13,17].reduce((p,i)=>({x:p.x+hand[i].x/5,y:p.y+hand[i].y/5}),{x:0,y:0})};
  // Mirror the landmark X coordinate to match the selfie-preview flip.
  const screen=(p:{x:number;y:number},w:number,h:number,out:T.Vector3)=>out.set((.5-p.x)*w,(.5-p.y)*h,0);
  const draw=()=>{
   raf=requestAnimationFrame(draw);const now=performance.now();if(now-lastAt<targetMs)return;const dt=lastAt?now-lastAt:16.7;lastAt=now;const time=now/1000,w=Math.max(1,element.clientWidth),h=Math.max(1,element.clientHeight),min=Math.min(w,h),state=trackingRef.current,props=latest.current;
   const hands=state?.effectHands?.length?state.effectHands:(state?.hands??[]),handForEffects=hands[state?.effectHandIndex??0]??hands[0],active=props.activeJutsu==='rasengan',scale=min/410,charge=clamp(props.chargeLevel/100,0,1),sampleAt=state?.sampledAt&&state.sampledAt>0?state.sampledAt:now;
   if(active){
    const center=palmCenter(handForEffects);
    if(center){const target=screen(center,w,h,targetPoint);
     if(!anchorReady||previous!=='rasengan'){anchor.copy(target);observed.copy(target);predicted.copy(target);velocity.set(0,0,0);observedAt=sampleAt;motionSpeed=0;anchorReady=true;lastCharge=0;lastTrailAnchor.set(1e6,1e6,1e6);trailClock=0;trailLife.fill(0);trailLifeAttribute.needsUpdate=true}
     else{const moved=target.distanceTo(observed);
      if(moved>.12){const elapsed=clamp((sampleAt-observedAt)/1000,.012,.12);motionDelta.subVectors(target,observed).divideScalar(elapsed);velocity.lerp(motionDelta,.82);motionSpeed=velocity.length()/Math.max(1,min);observed.copy(target);observedAt=sampleAt}
      else{velocity.multiplyScalar(Math.exp(-dt/100));motionSpeed=velocity.length()/Math.max(1,min)}
      predicted.copy(target);const pred=Math.min(.026,Math.max(.010,dt/1000*1.05));let lx=velocity.x*pred,ly=velocity.y*pred;const lead=Math.hypot(lx,ly),cap=min*.048;
      if(lead>cap&&lead>0){lx*=cap/lead;ly*=cap/lead}predicted.x+=lx;predicted.y+=ly;
      anchor.lerp(predicted,1-Math.exp(-clamp(132+motionSpeed*28,132,260)*Math.max(.008,dt/1000)));
     }
    }else if(!anchorReady){anchor.set(0,0,0);anchorReady=true}
    lastCharge=Math.max(lastCharge,charge);
   }
   if(previous==='rasengan'&&!active&&anchorReady){burstAt=now;burstCenter.copy(anchor);burstPower=clamp(Math.max(lastCharge,charge)*.82+clamp(motionSpeed*.15,0,.18),.2,1);releaseRing.position.copy(burstCenter);releaseRing.position.z=18;releaseRing.visible=true;releaseRing2.position.copy(burstCenter);releaseRing2.position.z=16;releaseRing2.visible=true;impactFlash.position.copy(burstCenter);impactFlash.position.z=21;impactFlash.visible=true;burstParticles.position.copy(burstCenter);burstParticles.position.z=19;burstParticles.visible=true}
   previous=active?'rasengan':'none';
   const bp=clamp((now-burstAt)/620,0,1),burstActive=now-burstAt<620;
   if(burstActive){
    const mat=releaseRing.material as T.MeshBasicMaterial;releaseRing.scale.setScalar(scale*(.45+bp*3.7)*(.65+burstPower*.55));releaseRing.rotation.z=time*1.7;mat.opacity=(1-bp)*(.28+burstPower*.58);
    const flashProgress=clamp((now-burstAt)/150,0,1),flashMaterial=impactFlash.material as T.MeshBasicMaterial;
    impactFlash.scale.setScalar(scale*(.24+flashProgress*2.8));flashMaterial.opacity=(1-flashProgress)*(.22+burstPower*.68);
    const bp2=clamp((now-burstAt-72)/560,0,1),mat2=releaseRing2.material as T.MeshBasicMaterial;
    releaseRing2.scale.setScalar(scale*(.2+bp2*4.3)*(.72+burstPower*.62));releaseRing2.rotation.z=-time*1.15;mat2.opacity=(now-burstAt<72?0:(1-bp2)*(.26+burstPower*.43));
    const pos=burstGeometry.getAttribute('position').array as Float32Array;
    for(let i=0;i<count;i++){const p=directions[i],d=p.speed*bp*(.45+burstPower*.7);pos[3*i]=Math.cos(p.angle)*d;pos[3*i+1]=Math.sin(p.angle)*d;pos[3*i+2]=p.depth*bp}
    burstGeometry.getAttribute('position').needsUpdate=true;burstMaterial.opacity=(1-bp)*(.3+burstPower*.55);releaseRing.visible=true;releaseRing2.visible=true;burstParticles.visible=true;
   }else{releaseRing.visible=false;releaseRing2.visible=false;impactFlash.visible=false;burstParticles.visible=false;burstMaterial.opacity=0}

   // Fade the persistent palm trail in-place; the geometry buffers are reused every frame.
   const trailFade=Math.max(0,dt/1000)/.23;let trailActive=false;
   for(let i=0;i<trailCount;i++){trailLife[i]=Math.max(0,trailLife[i]-trailFade);if(trailLife[i]>.01)trailActive=true}
   if(trailActive)trailLifeAttribute.needsUpdate=true;
   if(active&&lastTrailAnchor.x>1e5){lastTrailAnchor.copy(anchor);trailClock=0}
   trailClock+=dt/1000;
   if(active&&anchorReady&&anchor.distanceTo(lastTrailAnchor)>.85&&trailClock>=.02){
    trailClock=0;trailCursor=(trailCursor+1)%trailCount;
    trailPositions[trailCursor*3]=lastTrailAnchor.x;trailPositions[trailCursor*3+1]=lastTrailAnchor.y;trailPositions[trailCursor*3+2]=8;
    trailLife[trailCursor]=1;trailSizes[trailCursor]=3.2+Math.random()*4.8;
    trailPositionAttribute.needsUpdate=true;trailLifeAttribute.needsUpdate=true;trailSizeAttribute.needsUpdate=true;
    trailActive=true;lastTrailAnchor.copy(anchor);
   }else if(!active){trailClock=0}
   trail.visible=trailActive;
   rasengan.root.visible=active;
   if(active){
    rasengan.root.position.copy(anchor);const motion=1+clamp(motionSpeed*.72,0,1.35),pulse=Math.sin(time*13.5)*(.012+charge*.025);
    const chargePhase=charge<.35?0:charge<.78?1:2;
    const outerShell=rasengan.shell.material as T.ShaderMaterial;
    outerShell.uniforms.uTime.value=time;outerShell.uniforms.uCharge.value=charge;
    rasengan.shell.rotation.set(Math.sin(time*1.1)*.12,time*(.16+charge*.34),time*(.42+charge*.6));
    rasengan.shell.scale.setScalar(1+Math.sin(time*7.5)*.012+charge*.045);
    rasengan.root.scale.setScalar(scale*(.90+charge*.28+pulse*.45)*(chargePhase===2?1.025:1));rasengan.halo.scale.setScalar(.92+charge*.10+Math.sin(time*5.2)*.018);
    // Three short animated filaments hug the orb near the tracked knuckles,
    // visually tying the spherical chakra to the hand that is holding it.
    if(handForEffects&&handForEffects.length>=21){
      const knuckles=[5,9,17],rootScale=Math.max(0.001,rasengan.root.scale.x),shellRadius=28.8;
      rasengan.palmStreams.forEach((stream,streamIndex)=>{
        const tracked=screen(handForEffects[knuckles[streamIndex]],w,h,targetPoint);
        let dx=(tracked.x-anchor.x)/rootScale,dy=(tracked.y-anchor.y)/rootScale;
        const length=Math.hypot(dx,dy);
        if(length<0.001){dx=0;dy=-1}
        const safeLength=Math.max(0.001,length),ux=dx/safeLength,uy=dy/safeLength;
        const startDistance=18.5,endDistance=clamp(Math.max(25,safeLength+2),25,32);
        const attribute=stream.geometry.getAttribute('position');
        const array=attribute.array as Float32Array;
        for(let pointIndex=0;pointIndex<9;pointIndex++){
          const t=pointIndex/8,dist=startDistance+(endDistance-startDistance)*t;
          const lateral=Math.sin(t*Math.PI)*(Math.sin(time*5.8+streamIndex*1.7)*0.9+Math.sin(time*2.4+streamIndex)*0.45);
          array[pointIndex*3]=ux*dist-uy*lateral;
          array[pointIndex*3+1]=uy*dist+ux*lateral;
          array[pointIndex*3+2]=Math.sqrt(Math.max(0,shellRadius*shellRadius-Math.min(shellRadius,dist)**2))+0.7+Math.sin(t*Math.PI)*0.6;
        }
        attribute.needsUpdate=true;
        (stream.material as T.LineBasicMaterial).opacity=0.30+charge*0.28+Math.sin(time*7+streamIndex)*0.035;
        stream.visible=true;
      });
    }else rasengan.palmStreams.forEach(stream=>{stream.visible=false});
    const haloMaterial=rasengan.halo.material as T.ShaderMaterial;
    if(chargePhase!==chargePhaseLast){
      const haloHex=chargePhase===0?0x167fff:chargePhase===1?0x22bdff:0x8ae9ff;
      haloMaterial.uniforms.uTint.value.setHex(haloHex);
      (rasengan.core.material as T.ShaderMaterial).uniforms.uBaseColor.value.setHex(chargePhase===0?0x087aff:chargePhase===1?0x159fff:0x50d5ff);
      chargePhaseLast=chargePhase;
    }
    haloMaterial.uniforms.uTime.value=time;haloMaterial.uniforms.uCharge.value=charge;
    const shader=rasengan.core.material as T.ShaderMaterial;shader.uniforms.uTime.value=time;shader.uniforms.uCharge.value=charge;shader.uniforms.uPhase.value=chargePhase;rasengan.innerCore.scale.setScalar(.92+charge*.20+Math.sin(time*9.2)*.035+(chargePhase===2?.04:0));
    rasengan.rings.forEach((ring,i)=>{const d=i%2?-1:1;ring.rotation.x=time*(.8+i*.22)*motion+i*.7;ring.rotation.y=time*d*(1.0+charge*2.1+i*.20)*motion;ring.rotation.z=time*d*(1.8+charge*2.6)+i*1.3;ring.scale.setScalar(.7+charge*.38+Math.sin(time*10+i)*.035);(ring.material as T.MeshBasicMaterial).opacity=(.24+charge*.24)*(chargePhase===2?1.06:1)});
    rasengan.spirals.forEach((line,i)=>{line.rotation.z=time*(i%2?-4.8:4.2)*(.68+charge*.7)*(1+clamp(motionSpeed*.7,0,1.2));line.rotation.x=Math.sin(time*2.0+i)*.28;(line.material as T.MeshBasicMaterial).opacity=.36+charge*.34});
    const particles=rasengan.particles;particles.rotation.z=-time*(.9+charge*2.0)*motion;particles.rotation.y=time*(.7+charge*1.0)*(1+clamp(motionSpeed*.5,0,.9));(particles.material as T.PointsMaterial).opacity=.34+charge*.54+Math.sin(time*9)*.05;
    const pos=particles.geometry.getAttribute('position').array as Float32Array,seeds=particles.geometry.userData.seeds as Array<{angle:number;radius:number;depth:number;speed:number}>;
    for(let i=0;i<seeds.length;i++){const s=seeds[i],angle=s.angle+time*s.speed*(1.4+charge*2.0),radius=s.radius*(.82+charge*.18)+Math.sin(time*5+i)*1.7;pos[3*i]=Math.cos(angle)*radius;pos[3*i+1]=Math.sin(angle)*radius;pos[3*i+2]=s.depth+Math.sin(time*3+i)*2.4}particles.geometry.getAttribute('position').needsUpdate=true;
   }
   if(!(active||burstActive||trailActive)){if(wasDrawing){renderer.clear(true,true,true);wasDrawing=false}return}
   bloomPass.strength=active?0.46+charge*0.24+clamp(motionSpeed*0.04,0,0.06):0.68+burstPower*0.30;
   bloomPass.radius=active?0.42+charge*0.13:0.62;
   wasDrawing=true;composer.render(Math.min(dt,50)/1000);const cost=performance.now()-now;
   if(cost>20){slow++;fast=0}else if(cost<12){fast++;slow=0}else{slow=0;fast=0}
   if(slow>=3){targetMs=1000/45;if(pixelRatio>.72){pixelRatio=.72;composer.setPixelRatio(pixelRatio);resize()}slow=0;fast=0}
   else if(fast>=45){targetMs=1000/60;const best=Math.min(window.devicePixelRatio||1,1);if(pixelRatio<best){pixelRatio=best;composer.setPixelRatio(pixelRatio);resize()}fast=0}
  };
  raf=requestAnimationFrame(draw);
  return()=>{cancelAnimationFrame(raf);ro?.disconnect();composer.dispose();bloomPass.dispose();scene.traverse(o=>{const m=o as T.Mesh;if(m.geometry)m.geometry.dispose();const mat=m.material;if(mat){if(Array.isArray(mat))mat.forEach(x=>x.dispose());else mat.dispose()}});renderer.dispose();renderer.domElement.remove()};
 },[trackingRef]);
 return <div ref={host} className="three-effects-layer" aria-hidden="true"/>;
}
