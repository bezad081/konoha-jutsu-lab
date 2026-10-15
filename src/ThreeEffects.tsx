import { useEffect, useRef, type RefObject } from 'react';
import * as T from 'three';
export type EffectPoint = { x:number; y:number; z?:number };
export type ThreeTrackingState = { hands:EffectPoint[][]; effectHands?:EffectPoint[][]; effectHandIndex?:number };
type Props = { trackingRef:RefObject<ThreeTrackingState>; activeJutsu:string; chargeLevel:number };
type RasenganFx = { root:T.Group; halo:T.Mesh; core:T.Mesh; innerCore:T.Mesh; rings:T.Mesh[]; spirals:T.Line[]; particles:T.Points };
const additive=T.AdditiveBlending;
const clamp=(v:number,min:number,max:number)=>Math.max(min,Math.min(max,v));
const basic=(color:number,opacity:number,blending=additive)=>new T.MeshBasicMaterial({color,transparent:true,opacity,blending,depthWrite:false,side:T.DoubleSide});
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
  const halo = new T.Mesh(new T.SphereGeometry(44, 24, 18), basic(0x258cff, 0.065));
  root.add(halo);
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
  return { root, halo, core, innerCore, rings, spirals, particles };
}

export default function ThreeEffects({trackingRef,activeJutsu,chargeLevel}:Props){
 const host=useRef<HTMLDivElement>(null),latest=useRef({activeJutsu,chargeLevel});latest.current={activeJutsu,chargeLevel};
 useEffect(()=>{
  const element=host.current;if(!element)return;let renderer:T.WebGLRenderer;
  try{renderer=new T.WebGLRenderer({alpha:true,antialias:false,powerPreference:'high-performance',premultipliedAlpha:false})}catch{element.dataset.renderer='unavailable';return}
  renderer.setClearColor(0,0);renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1));renderer.outputColorSpace=T.SRGBColorSpace;renderer.domElement.className='three-effects-canvas';element.appendChild(renderer.domElement);
  const scene=new T.Scene(),camera=new T.OrthographicCamera(-1,1,1,-1,.1,1000);camera.position.z=400;
  const rasengan=createRasengan(scene),releaseRing=new T.Mesh(new T.TorusGeometry(13,1.4,7,64),basic(0xbcefff,0));releaseRing.visible=false;scene.add(releaseRing);
  const count=48,burstGeometry=new T.BufferGeometry();burstGeometry.setAttribute('position',new T.BufferAttribute(new Float32Array(count*3),3));
  const directions=Array.from({length:count},(_,i)=>({angle:i/count*Math.PI*2+(Math.random()-.5)*.12,speed:55+Math.random()*110,depth:(Math.random()-.5)*20}));
  const burstMaterial=new T.PointsMaterial({color:0x94eaff,size:3.2,transparent:true,opacity:0,blending:additive,depthWrite:false,sizeAttenuation:false});
  const burstParticles=new T.Points(burstGeometry,burstMaterial);burstParticles.visible=false;scene.add(burstParticles);
  let raf=0,lastAt=0,targetMs=1000/60,slow=0,fast=0,pixelRatio=Math.min(window.devicePixelRatio||1,1),wasDrawing=false,previous='none',burstAt=-Infinity,burstPower=0,lastCharge=0;
  const burstCenter=new T.Vector3(),anchor=new T.Vector3(),observed=new T.Vector3(),predicted=new T.Vector3(),velocity=new T.Vector3();let anchorReady=false,observedAt=0,motionSpeed=0;
  const resize=()=>{const w=Math.max(1,element.clientWidth),h=Math.max(1,element.clientHeight);camera.left=-w/2;camera.right=w/2;camera.top=h/2;camera.bottom=-h/2;camera.updateProjectionMatrix();renderer.setSize(w,h,false)};
  const ro=typeof ResizeObserver!=='undefined'?new ResizeObserver(resize):null;ro?.observe(element);resize();
  const palmCenter=(hand:EffectPoint[]|undefined)=>{if(!hand||hand.length<21)return null;return [0,5,9,13,17].reduce((p,i)=>({x:p.x+hand[i].x/5,y:p.y+hand[i].y/5}),{x:0,y:0})};
  // Mirror the landmark X coordinate to match the selfie-preview flip.
  const screen=(p:{x:number;y:number},w:number,h:number)=>new T.Vector3((.5-p.x)*w,(.5-p.y)*h,0);
  const draw=()=>{
   raf=requestAnimationFrame(draw);const now=performance.now();if(now-lastAt<targetMs)return;const dt=lastAt?now-lastAt:16.7;lastAt=now;const time=now/1000,w=Math.max(1,element.clientWidth),h=Math.max(1,element.clientHeight),min=Math.min(w,h),state=trackingRef.current,props=latest.current;
   const hands=state?.effectHands?.length?state.effectHands:(state?.hands??[]),active=props.activeJutsu==='rasengan',scale=min/360,charge=clamp(props.chargeLevel/100,0,1);
   if(active){
    const hand=hands[state?.effectHandIndex??0]??hands[0],center=palmCenter(hand);
    if(center){const target=screen(center,w,h);
     if(!anchorReady||previous!=='rasengan'){anchor.copy(target);observed.copy(target);predicted.copy(target);velocity.set(0,0,0);observedAt=now;motionSpeed=0;anchorReady=true;lastCharge=0}
     else{const moved=target.distanceTo(observed);
      if(moved>.12){const elapsed=clamp((now-observedAt)/1000,.012,.12);velocity.lerp(target.clone().sub(observed).divideScalar(elapsed),.62);motionSpeed=velocity.length()/Math.max(1,min);observed.copy(target);observedAt=now}
      else{velocity.multiplyScalar(Math.exp(-dt/170));motionSpeed=velocity.length()/Math.max(1,min)}
      predicted.copy(target);const pred=Math.min(.014,Math.max(.004,dt/1000*.72));let lx=velocity.x*pred,ly=velocity.y*pred;const lead=Math.hypot(lx,ly),cap=min*.03;
      if(lead>cap&&lead>0){lx*=cap/lead;ly*=cap/lead}predicted.x+=lx;predicted.y+=ly;
      anchor.lerp(predicted,1-Math.exp(-clamp(76+motionSpeed*12,76,158)*Math.max(.008,dt/1000)));
     }
    }else if(!anchorReady){anchor.set(0,0,0);anchorReady=true}
    lastCharge=Math.max(lastCharge,charge);
   }
   if(previous==='rasengan'&&!active&&anchorReady){burstAt=now;burstCenter.copy(anchor);burstPower=clamp(Math.max(lastCharge,charge),.2,1);releaseRing.position.copy(burstCenter);releaseRing.position.z=18;releaseRing.visible=true;burstParticles.position.copy(burstCenter);burstParticles.position.z=19;burstParticles.visible=true}
   previous=active?'rasengan':'none';
   const bp=clamp((now-burstAt)/460,0,1),burstActive=now-burstAt<460;
   if(burstActive){const mat=releaseRing.material as T.MeshBasicMaterial;releaseRing.scale.setScalar(scale*(.45+bp*3.7)*(.65+burstPower*.55));releaseRing.rotation.z=time*1.7;mat.opacity=(1-bp)*(.28+burstPower*.52);
    const pos=burstGeometry.getAttribute('position').array as Float32Array;
    for(let i=0;i<count;i++){const p=directions[i],d=p.speed*bp*(.45+burstPower*.7);pos[3*i]=Math.cos(p.angle)*d;pos[3*i+1]=Math.sin(p.angle)*d;pos[3*i+2]=p.depth*bp}
    burstGeometry.getAttribute('position').needsUpdate=true;burstMaterial.opacity=(1-bp)*(.3+burstPower*.5);releaseRing.visible=true;burstParticles.visible=true;
   }else{releaseRing.visible=false;burstParticles.visible=false;burstMaterial.opacity=0}
   rasengan.root.visible=active;
   if(active){
    rasengan.root.position.copy(anchor);const motion=1+clamp(motionSpeed*.72,0,1.35),pulse=Math.sin(time*8.2)*(.012+charge*.025);
    rasengan.root.scale.setScalar(scale*(.78+charge*.64+pulse));rasengan.halo.scale.setScalar(.95+charge*.24+Math.sin(time*4.4)*.035);(rasengan.halo.material as T.MeshBasicMaterial).opacity=.055+charge*.09+motionSpeed*.025;
    const shader=rasengan.core.material as T.ShaderMaterial;shader.uniforms.uTime.value=time;shader.uniforms.uCharge.value=charge;rasengan.innerCore.scale.setScalar(.88+charge*.46+Math.sin(time*8.4)*.08);
    rasengan.rings.forEach((ring,i)=>{const d=i%2?-1:1;ring.rotation.x=time*(.45+i*.13)*motion+i*.7;ring.rotation.y=time*d*(.62+charge*1.5+i*.14)*motion;ring.rotation.z=time*d*(1.05+charge*1.85)+i*1.3;ring.scale.setScalar(.7+charge*.38+Math.sin(time*6+i)*.035);(ring.material as T.MeshBasicMaterial).opacity=.36+charge*.48});
    rasengan.spirals.forEach((line,i)=>{line.rotation.z=time*(i%2?-2.45:2.15)*(.68+charge*.7)*(1+clamp(motionSpeed*.7,0,1.2));line.rotation.x=Math.sin(time*1.3+i)*.28;(line.material as T.LineBasicMaterial).opacity=.5+charge*.44});
    const particles=rasengan.particles;particles.rotation.z=-time*(.45+charge*1.25)*motion;particles.rotation.y=time*(.35+charge*.56)*(1+clamp(motionSpeed*.5,0,.9));(particles.material as T.PointsMaterial).opacity=.34+charge*.54+Math.sin(time*9)*.05;
    const pos=particles.geometry.getAttribute('position').array as Float32Array,seeds=particles.geometry.userData.seeds as Array<{angle:number;radius:number;depth:number;speed:number}>;
    for(let i=0;i<seeds.length;i++){const s=seeds[i],angle=s.angle+time*s.speed*(.8+charge*1.5),radius=s.radius*(.82+charge*.18)+Math.sin(time*5+i)*1.7;pos[3*i]=Math.cos(angle)*radius;pos[3*i+1]=Math.sin(angle)*radius;pos[3*i+2]=s.depth+Math.sin(time*3+i)*2.4}particles.geometry.getAttribute('position').needsUpdate=true;
   }
   if(!(active||burstActive)){if(wasDrawing){renderer.clear();wasDrawing=false}return}
   wasDrawing=true;renderer.render(scene,camera);const cost=performance.now()-now;
   if(cost>21){slow++;fast=0}else if(cost<10){fast++;slow=0}else{slow=0;fast=0}
   if(slow>=3){targetMs=1000/30;if(pixelRatio>.72){pixelRatio=.72;renderer.setPixelRatio(pixelRatio);resize()}slow=0;fast=0}
   else if(fast>=90){targetMs=1000/60;const best=Math.min(window.devicePixelRatio||1,1);if(pixelRatio<best){pixelRatio=best;renderer.setPixelRatio(pixelRatio);resize()}fast=0}
  };
  raf=requestAnimationFrame(draw);
  return()=>{cancelAnimationFrame(raf);ro?.disconnect();scene.traverse(o=>{const m=o as T.Mesh;if(m.geometry)m.geometry.dispose();const mat=m.material;if(mat){if(Array.isArray(mat))mat.forEach(x=>x.dispose());else mat.dispose()}});renderer.dispose();renderer.domElement.remove()};
 },[trackingRef]);
 return <div ref={host} className="three-effects-layer" aria-hidden="true"/>;
}
