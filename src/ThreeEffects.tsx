import { useEffect, useRef, type RefObject } from 'react';
import * as T from 'three';

export type EffectPoint={x:number;y:number;z?:number};
export type EffectTool='none'|'shuriken'|'kunai';
export type EffectProjectile={kind:Exclude<EffectTool,'none'>;x:number;y:number;vx:number;vy:number;born:number;spin:number};
export type ThreeTrackingState={hands:EffectPoint[][];face:EffectPoint[];move:{x:number;y:number};activeTool:EffectTool;thrownTools:EffectProjectile[];trail:Array<{x:number;y:number;at:number}>;cloneSprite:HTMLCanvasElement|null;cloneRevision:number;cloneVisibleUntil:number};
type Props={trackingRef:RefObject<ThreeTrackingState>;activeJutsu:string;sharinganActive:boolean;chargeLevel:number};
type FX={root:T.Group;spin:T.Group;rings:T.Mesh[];flames:T.Mesh[];bolts:T.LineSegments|null;dust:T.Points;energy:T.Mesh|null};
const additive=T.AdditiveBlending;
const mat=(color:number,opacity=1)=>new T.MeshBasicMaterial({color,transparent:opacity<1,opacity,blending:additive,depthWrite:false,side:T.DoubleSide});
const orb=(r:number,c:number,o=1)=>new T.Mesh(new T.SphereGeometry(r,20,14),mat(c,o));
const ring=(r:number,t:number,c:number)=>new T.Mesh(new T.TorusGeometry(r,t,6,64),mat(c,.78));
function line(points:T.Vector3[],color:number){return new T.Line(new T.BufferGeometry().setFromPoints(points),new T.LineBasicMaterial({color,transparent:true,opacity:.9,blending:additive,depthWrite:false}));}
function cloud(n:number,spread:number,color:number){const p=new Float32Array(n*3);for(let i=0;i<n;i++){const a=Math.random()*Math.PI*2,r=spread*(.2+Math.random()*.8);p[i*3]=Math.cos(a)*r;p[i*3+1]=Math.sin(a)*r;p[i*3+2]=(Math.random()-.5)*spread*.6;}const g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(p,3));return new T.Points(g,new T.PointsMaterial({color,size:2.5,transparent:true,opacity:.9,blending:additive,depthWrite:false,sizeAttenuation:false}));}
function effect(scene:T.Scene,name:string):FX{
 const root=new T.Group(),spin=new T.Group();root.add(spin);
 let c=0x42bfff,hot=0xeaffff,dustColor=0x8beaff;
 if(name==='fireball'){c=0xff5d16;hot=0xfff1b1;dustColor=0xff8a32;}
 if(name==='amaterasu'){c=0x170b27;hot=0x9b72ff;dustColor=0x9b72ff;}
 if(name==='wind-scythe'){c=0x32e99b;hot=0xdffff0;dustColor=0x70ffc0;}
 let energy:T.Mesh|null=null;
 if(name!=='wind-scythe'){
  spin.add(orb(name==='fireball'?30:28,c,.09));
  const mode=name==='fireball'?1:name==='amaterasu'?2:0;
  const material=new T.ShaderMaterial({
   uniforms:{uTime:{value:0},uBaseColor:{value:new T.Color(c)},uHotColor:{value:new T.Color(hot)},uMode:{value:mode}},
   vertexShader:`varying vec3 vLocalPosition; varying vec3 vNormalView; varying vec3 vViewPosition;
    void main(){vLocalPosition=position;vec4 mvPosition=modelViewMatrix*vec4(position,1.0);vViewPosition=-mvPosition.xyz;vNormalView=normalize(normalMatrix*normal);gl_Position=projectionMatrix*mvPosition;}`,
   fragmentShader:`uniform float uTime;uniform vec3 uBaseColor;uniform vec3 uHotColor;uniform float uMode;
    varying vec3 vLocalPosition;varying vec3 vNormalView;varying vec3 vViewPosition;
    void main(){
     vec3 n=normalize(vNormalView);vec3 viewDir=normalize(vViewPosition);
     float angle=atan(vLocalPosition.y,vLocalPosition.x);
     float phase=angle*10.0+vLocalPosition.z*0.42-uTime*(uMode>0.5?3.0:4.8)+sin(vLocalPosition.y*0.34+uTime*1.7)*1.35;
     float bands=0.5+0.5*sin(phase);
     float filaments=smoothstep(0.72,0.98,bands);
     float ribs=pow(0.5+0.5*sin(angle*23.0-vLocalPosition.z*0.22-uTime*3.5),9.0);
     float fresnel=pow(1.0-max(0.0,dot(n,viewDir)),2.35);
     float core=pow(max(0.0,1.0-length(vLocalPosition.xy)*0.035),3.0);
     vec3 color=mix(uBaseColor,uHotColor,clamp(filaments*0.72+ribs*0.26+fresnel*0.5+core*0.2,0.0,1.0));
     if(uMode>0.5&&uMode<1.5){float flame=0.5+0.5*sin(vLocalPosition.y*0.38+vLocalPosition.x*0.12+uTime*3.2+sin(angle*7.0)*1.5);color=mix(uBaseColor,vec3(1.0,0.24,0.025),clamp(flame*0.65+fresnel*0.5,0.0,1.0));}
     if(uMode>1.5){float ember=smoothstep(0.48,0.98,bands+fresnel*0.25);color=mix(vec3(0.025,0.008,0.045),uHotColor,ember*0.9);}
     float alpha=clamp(0.35+fresnel*0.58+filaments*0.28+ribs*0.12,0.0,0.96);
     gl_FragColor=vec4(color*(0.75+fresnel*1.5+filaments*0.5),alpha);
    }`,
   transparent:true,blending:additive,depthWrite:false,side:T.DoubleSide
  });
  energy=new T.Mesh(new T.SphereGeometry(name==='fireball'?24:21,32,24),material);energy.position.z=1;spin.add(energy);
 }
 const rings:T.Mesh[]=[];const ringCount=name==='wind-scythe'?3:4;
 for(let i=0;i<ringCount;i++){const r=ring(name==='wind-scythe'?25+i*7:18+i*4,.7+i%2*.45,i%2?hot:c);r.rotation.set(i*.7,i*1.1,i*.45);if(name==='wind-scythe')r.scale.set(1.6,.52,1);rings.push(r);spin.add(r);}
 const flames:T.Mesh[]=[];
 const n=name==='fireball'?16:name==='amaterasu'?13:name==='wind-scythe'?5:0;
 for(let i=0;i<n;i++){const a=i/n*Math.PI*2,len=name==='wind-scythe'?34:14+Math.random()*23;const f=new T.Mesh(new T.ConeGeometry(name==='wind-scythe'?2.3:3+Math.random()*2,len,6),mat(name==='amaterasu'&&i%4?0x0b0714:i%3?c:hot,.86));f.position.set(Math.cos(a)*17,Math.sin(a)*17,(Math.random()-.5)*7);f.rotation.z=a-Math.PI/2;f.userData.base=.7+Math.random()*.7;flames.push(f);spin.add(f);}
 if(name==='wind-scythe'){
  const shape=new T.Shape();shape.moveTo(-68,-22);shape.quadraticCurveTo(-4,-9,70,13);shape.quadraticCurveTo(16,18,-45,36);shape.quadraticCurveTo(-20,14,-68,-22);
  const blade=new T.Mesh(new T.ShapeGeometry(shape,22),new T.MeshBasicMaterial({color:0xa5ffd5,transparent:true,opacity:.76,side:T.DoubleSide,blending:additive,depthWrite:false}));blade.position.set(11,0,8);spin.add(blade);
 }
 if(name==='rasengan')for(let s=0;s<3;s++){const p:T.Vector3[]=[];for(let i=0;i<80;i++){const q=i/79,a=q*Math.PI*5+s*Math.PI*2/3;p.push(new T.Vector3(Math.cos(a)*(10+q*22),Math.sin(a)*(10+q*22),Math.sin(a*.7+s)*5));}spin.add(line(p,s%2?0x4b9dff:0xd5ffff));}
 let bolts:T.LineSegments|null=null;
 if(name==='chidori'||name==='lightning-burst'){
  const positions=new Float32Array(18*3*2*3),g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(positions,3));
  bolts=new T.LineSegments(g,new T.LineBasicMaterial({color:0x74e5ff,transparent:true,opacity:.9,blending:additive,depthWrite:false}));root.add(bolts);
 }
 const dust=cloud(name==='rasengan'?72:name==='fireball'?72:46,name==='fireball'?43:36,dustColor);root.add(dust);scene.add(root);
 return{root,spin,rings,flames,bolts,dust,energy};
}
function tool(kind:'shuriken'|'kunai'){
 const g=new T.Group(),metal=new T.MeshStandardMaterial({color:0xc6d9e8,metalness:.94,roughness:.2,emissive:0x315a78,emissiveIntensity:.45}),edge=new T.MeshStandardMaterial({color:0xf4fcff,metalness:.98,roughness:.15});
 if(kind==='shuriken'){const s=new T.Shape();for(let i=0;i<8;i++){const a=-Math.PI/2+i*Math.PI/4,r=i%2===0?19:5.5,x=Math.cos(a)*r,y=Math.sin(a)*r;if(!i)s.moveTo(x,y);else s.lineTo(x,y);}s.closePath();const h=new T.Path();h.absarc(0,0,3,0,Math.PI*2,true);s.holes.push(h);g.add(new T.Mesh(new T.ExtrudeGeometry(s,{depth:2,bevelEnabled:true,bevelSize:.6,bevelThickness:.4,bevelSegments:1}),metal));}
 else{const s=new T.Shape();s.moveTo(0,25);s.lineTo(6,2);s.lineTo(4,-11);s.lineTo(0,-16);s.lineTo(-4,-11);s.lineTo(-6,2);s.closePath();g.add(new T.Mesh(new T.ExtrudeGeometry(s,{depth:2.5,bevelEnabled:true,bevelSize:.6,bevelThickness:.4,bevelSegments:1}),edge));const h=new T.Mesh(new T.CylinderGeometry(1.5,2,15,8),metal);h.position.y=-23;g.add(h);const loop=new T.Mesh(new T.TorusGeometry(3,1,6,18),edge);loop.position.y=-31;g.add(loop);}
 return g;
}
function cloneGhost(color:number):T.Group{
 const g=new T.Group(),m=new T.MeshBasicMaterial({color,transparent:true,opacity:.32,blending:additive,depthWrite:false,side:T.DoubleSide});
 const head=new T.Mesh(new T.SphereGeometry(8,14,10),m);head.position.y=31;g.add(head);
 const torso=new T.Mesh(new T.SphereGeometry(13,14,10),m);torso.scale.set(.9,1.55,.62);torso.position.y=6;g.add(torso);
 const armGeometry=new T.CylinderGeometry(3.2,4.2,27,7),legGeometry=new T.CylinderGeometry(4,5,29,7);
 for(const side of [-1,1]){const arm=new T.Mesh(armGeometry,m);arm.position.set(side*15,7,0);arm.rotation.z=side*.42;g.add(arm);const leg=new T.Mesh(legGeometry,m);leg.position.set(side*6,-21,0);g.add(leg);}
 return g;
}
function eye(){const g=new T.Group();g.add(new T.Mesh(new T.CircleGeometry(10,32),new T.MeshBasicMaterial({color:0xc91532,transparent:true,opacity:.95})),new T.Mesh(new T.TorusGeometry(10,1.1,6,32),new T.MeshBasicMaterial({color:0x30050d})));const p=new T.Mesh(new T.CircleGeometry(2.7,18),new T.MeshBasicMaterial({color:0x08050a}));p.position.z=1.5;g.add(p);for(let i=0;i<3;i++){const t=new T.Group(),d=new T.Mesh(new T.CircleGeometry(1.7,12),new T.MeshBasicMaterial({color:0x08050a}));d.position.set(0,5.8,1.4);t.add(d);t.rotation.z=i*Math.PI*2/3;g.add(t);}return g;}
export default function ThreeEffects({trackingRef,activeJutsu,sharinganActive,chargeLevel}:Props){
 const host=useRef<HTMLDivElement>(null),latest=useRef({activeJutsu,sharinganActive,chargeLevel});latest.current={activeJutsu,sharinganActive,chargeLevel};
 useEffect(()=>{const el=host.current;if(!el)return;let renderer:T.WebGLRenderer;try{renderer=new T.WebGLRenderer({alpha:true,antialias:false,powerPreference:'default',premultipliedAlpha:false});}catch{el.dataset.renderer='unavailable';return;}
 renderer.setClearColor(0,0);renderer.setPixelRatio(Math.min(devicePixelRatio||1,1));renderer.outputColorSpace=T.SRGBColorSpace;renderer.domElement.className='three-effects-canvas';el.appendChild(renderer.domElement);
 const scene=new T.Scene(),camera=new T.OrthographicCamera(-1,1,1,-1,.1,1000);camera.position.z=300;scene.add(new T.AmbientLight(0xffffff,.8));
 const names=['rasengan','chidori','lightning-burst','fireball','amaterasu','wind-scythe'],fx:Record<string,FX>={};names.forEach(n=>fx[n]=effect(scene,n));
 const heldS=tool('shuriken'),heldK=tool('kunai');scene.add(heldS,heldK);heldS.visible=heldK.visible=false;const poolS:T.Group[]=[],poolK:T.Group[]=[];for(let i=0;i<12;i++){const s=heldS.clone(true),k=heldK.clone(true);s.visible=k.visible=false;scene.add(s,k);poolS.push(s);poolK.push(k);}
 const eyes=[eye(),eye()];eyes.forEach(e=>{e.visible=false;scene.add(e);});
 const cloneGhosts=[cloneGhost(0x9b79ff),cloneGhost(0x70dcff)];cloneGhosts.forEach(g=>{g.visible=false;scene.add(g);});
 const cloneGeometry=new T.PlaneGeometry(1,1);
 const cloneMaterials=[new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false,side:T.DoubleSide}),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false,side:T.DoubleSide})];
 const clonePlanes=cloneMaterials.map(m=>{const mesh=new T.Mesh(cloneGeometry,m);mesh.visible=false;scene.add(mesh);return mesh;});
 let cloneTexture:T.CanvasTexture|null=null,cloneSource:HTMLCanvasElement|null=null,lastCloneRevision=-1;const trPos=new Float32Array(192),trGeo=new T.BufferGeometry();trGeo.setAttribute('position',new T.BufferAttribute(trPos,3));const trMat=new T.LineBasicMaterial({color:0x69dfff,transparent:true,opacity:.7,blending:additive,depthWrite:false}),trLine=new T.Line(trGeo,trMat);trLine.frustumCulled=false;scene.add(trLine);
 const resize=()=>{const w=Math.max(1,el.clientWidth),h=Math.max(1,el.clientHeight);camera.left=-w/2;camera.right=w/2;camera.top=h/2;camera.bottom=-h/2;camera.updateProjectionMatrix();renderer.setSize(w,h,false);};const ro=typeof ResizeObserver!=='undefined'?new ResizeObserver(resize):null;ro?.observe(el);resize();
 const xy=(p:EffectPoint,w:number,h:number)=>new T.Vector3((p.x-.5)*w,(.5-p.y)*h,0),clock=new T.Clock();let raf=0,lastRenderAt=0,wasDrawing=false,windAngle=0,renderRatio=Math.min(devicePixelRatio||1,1),slowFrames=0,fastFrames=0;
 const draw=()=>{raf=requestAnimationFrame(draw);const now=performance.now();if(now-lastRenderAt<1000/30)return;lastRenderAt=now;const t=clock.getElapsedTime(),w=Math.max(1,el.clientWidth),h=Math.max(1,el.clientHeight),s=trackingRef.current,p=latest.current,hands=s?.hands??[],face=s?.face??[];
 const toolNow=s?.activeTool??'none',projectileNow=s?.thrownTools??[],needsEyes=p.sharinganActive&&face.length>=478,needsTrail=(s?.trail?.length??0)>1;
 const shouldDraw=p.activeJutsu!=='none'||needsEyes||(toolNow!=='none'&&hands.length>0)||projectileNow.length>0||needsTrail;
 if(!shouldDraw){if(wasDrawing){renderer.clear();wasDrawing=false;}return;}wasDrawing=true;
 const palm=(a:EffectPoint[]|undefined)=>a&&a.length>=21?{x:[0,5,9,13,17].reduce((v,i)=>v+a[i].x/5,0),y:[0,5,9,13,17].reduce((v,i)=>v+a[i].y/5,0)}:{x:.5,y:.54};
 const a=palm(hands[0]),b=hands[1]?palm(hands[1]):null,mouth=face.length>14?{x:(face[13].x+face[14].x)/2,y:(face[13].y+face[14].y)/2}:null,focus=p.activeJutsu==='fireball'&&mouth?mouth:p.activeJutsu==='lightning-burst'&&b?{x:(a.x+b.x)/2,y:(a.y+b.y)/2}:a,pos=xy(focus,w,h),scale=Math.min(w,h)/360,q=Math.max(.2,Math.min(1.25,p.chargeLevel/100||.2));
 if(p.activeJutsu==='wind-scythe'){const mv=s?.move??{x:0,y:0};if(Math.hypot(mv.x,mv.y)>.0008){const target=Math.atan2(mv.y,mv.x);windAngle+=Math.atan2(Math.sin(target-windAngle),Math.cos(target-windAngle))*.25;}pos.x+=Math.cos(windAngle)*20*scale;pos.y+=Math.sin(windAngle)*20*scale;}
 Object.entries(fx).forEach(([name,f])=>{f.root.visible=p.activeJutsu===name;if(!f.root.visible)return;f.root.position.copy(pos);f.root.scale.setScalar(scale*(name==='wind-scythe'?1.05:name==='fireball'?1.12+q*.18:name==='rasengan'?.88+q*.2:.86+q*.12));if(name==='wind-scythe'){f.spin.rotation.y=0;f.spin.rotation.z=t*.42;f.root.rotation.z=windAngle;}else{f.spin.rotation.y=t*1.6;f.spin.rotation.z=t*(name==='rasengan'?2.3:1.5);}if(f.energy){(f.energy.material as T.ShaderMaterial).uniforms.uTime.value=t;}f.rings.forEach((r,i)=>{r.rotation.x+=.009;r.rotation.y+=.013+i*.002;r.rotation.z+=(i%2?-1:1)*.02;});f.flames.forEach((m,i)=>{const base=Number(m.userData.base||1),v=.8+Math.sin(t*13+i*2)*.2;m.scale.set(base*v,base*(.8+v*.35),1);m.rotation.z+=Math.sin(t*4+i)*.009;});if(f.bolts){const at=f.bolts.geometry.getAttribute('position') as T.BufferAttribute,arr=at.array as Float32Array;for(let i=0;i<18;i++){const ang=i/18*Math.PI*2+t*.5+Math.sin(t*3+i)*.04,len=28+(i*17)%24,x1=Math.cos(ang+Math.sin(t*28+i)*.2)*len*.4,y1=Math.sin(ang+Math.sin(t*25+i)*.2)*len*.4,x2=Math.cos(ang+.22)*len*.7,y2=Math.sin(ang+.22)*len*.7,x3=Math.cos(ang-.08)*len,y3=Math.sin(ang-.08)*len,k=i*18;arr[k]=0;arr[k+1]=0;arr[k+2]=0;arr[k+3]=x1;arr[k+4]=y1;arr[k+5]=2;arr[k+6]=x1;arr[k+7]=y1;arr[k+8]=2;arr[k+9]=x2;arr[k+10]=y2;arr[k+11]=-3;arr[k+12]=x2;arr[k+13]=y2;arr[k+14]=-3;arr[k+15]=x3;arr[k+16]=y3;arr[k+17]=3;}at.needsUpdate=true;}f.dust.rotation.z=-t*.9;f.dust.rotation.y=t*.7;(f.dust.material as T.PointsMaterial).opacity=.58+Math.sin(t*7)*.2;});
 const cloneUntil=s?.cloneVisibleUntil??0,cloneActive=p.activeJutsu==='clone'&&(cloneUntil===0||now<cloneUntil);
 const cloneCanvas=s?.cloneSprite??null;
 if(cloneCanvas&&cloneCanvas!==cloneSource){cloneTexture?.dispose();cloneSource=cloneCanvas;cloneTexture=new T.CanvasTexture(cloneCanvas);cloneTexture.colorSpace=T.SRGBColorSpace;cloneTexture.generateMipmaps=false;cloneTexture.minFilter=T.LinearFilter;cloneTexture.magFilter=T.LinearFilter;cloneMaterials.forEach(m=>{m.map=cloneTexture;m.needsUpdate=true;});lastCloneRevision=-1;}
 if(cloneTexture&&(s?.cloneRevision??0)!==lastCloneRevision){cloneTexture.needsUpdate=true;lastCloneRevision=s?.cloneRevision??0;}
 const cloneFade=cloneUntil===0?1:Math.max(0,Math.min(1,(cloneUntil-now)/550));
 const cloneOffsets=[-.23+Math.sin(t*1.4)*.012,.23+Math.sin(t*1.7+1)*.012],cloneY=[.006,-.004],cloneScale=[.96,.94],cloneAlpha=[.76,.72];
 clonePlanes.forEach((mesh,i)=>{mesh.visible=cloneActive&&!!cloneTexture;mesh.position.set(w*cloneOffsets[i],-h*cloneY[i],10+i*.01);mesh.scale.set(w*cloneScale[i],h*cloneScale[i],1);cloneMaterials[i].opacity=cloneAlpha[i]*cloneFade;});
 cloneGhosts.forEach((ghost,i)=>{ghost.visible=cloneActive&&!cloneTexture;ghost.position.set(w*cloneOffsets[i],-h*.025,6+i*.01);ghost.scale.setScalar(Math.min(w,h)/275);});
 const activeTool=s?.activeTool??'none';heldS.visible=activeTool==='shuriken'&&hands.length>0;heldK.visible=activeTool==='kunai'&&hands.length>0;if(heldS.visible||heldK.visible){const m=activeTool==='shuriken'?heldS:heldK;m.position.copy(xy(a,w,h));m.position.z=24;m.scale.setScalar(scale*.75);m.rotation.z=Math.atan2((hands[0]?.[8]?.y??a.y)-(hands[0]?.[5]?.y??a.y-.1),(hands[0]?.[8]?.x??a.x)-(hands[0]?.[5]?.x??a.x))+Math.PI/2;if(activeTool==='shuriken')m.rotation.y=t*2;}
 const usedS=new Set<number>(),usedK=new Set<number>();(s?.thrownTools??[]).forEach(v=>{const pool=v.kind==='shuriken'?poolS:poolK,used=v.kind==='shuriken'?usedS:usedK;let i=0;while(used.has(i)&&i<pool.length)i++;if(i>=pool.length)return;used.add(i);const m=pool[i];m.visible=true;m.position.copy(xy(v,w,h));m.position.z=28;m.scale.setScalar(scale*.75);m.rotation.z=v.spin+(now-v.born)*.014;});poolS.forEach((m,i)=>{if(!usedS.has(i))m.visible=false;});poolK.forEach((m,i)=>{if(!usedK.has(i))m.visible=false;});
 eyes.forEach((e,i)=>{
  const valid=face.length>=478;
  const iris=valid?face[i===0?468:473]:null;
  const corners=valid?(i===0?[33,133]:[362,263]):null;
  const outer=corners?face[corners[0]]:null,inner=corners?face[corners[1]]:null;
  const wasVisible=e.visible;
  e.visible=!!(p.sharinganActive&&iris&&outer&&inner);
  if(iris&&outer&&inner){
    const center=xy(iris,w,h);
    const left=xy(outer,w,h),right=xy(inner,w,h);
    const eyeWidth=Math.hypot(right.x-left.x,right.y-left.y);
    const eyeAngle=Math.atan2(right.y-left.y,right.x-left.x);
    const targetScale=Math.max(.18,Math.min(1.15,eyeWidth*.38/20));
    if(!wasVisible){e.position.set(center.x,center.y,40);e.rotation.z=eyeAngle;e.scale.setScalar(targetScale);}
    else{
      e.position.x+=(center.x-e.position.x)*.42;
      e.position.y+=(center.y-e.position.y)*.42;
      const angleDelta=Math.atan2(Math.sin(eyeAngle-e.rotation.z),Math.cos(eyeAngle-e.rotation.z));
      e.rotation.z+=angleDelta*.38;
      e.scale.setScalar(e.scale.x+(targetScale-e.scale.x)*.32);
    }
    e.position.z=40;
    // Scale the iris overlay from that eye's own width and smooth small landmark jumps.
  }
 });
 const recent=(s?.trail??[]).slice(-64),at=trGeo.getAttribute('position') as T.BufferAttribute,arr=at.array as Float32Array;recent.forEach((v,i)=>{const z=xy(v,w,h);arr[i*3]=z.x;arr[i*3+1]=z.y;arr[i*3+2]=8;});at.needsUpdate=true;trGeo.setDrawRange(0,recent.length);trMat.color.setHex(p.activeJutsu==='fireball'?0xff8731:p.activeJutsu==='amaterasu'?0x9a6aff:p.activeJutsu==='wind-scythe'?0x68ffc0:p.activeJutsu==='chidori'||p.activeJutsu==='lightning-burst'?0x74e5ff:0x50bfff);trMat.opacity=p.activeJutsu==='none'?0:.66;
 const renderStarted=performance.now();renderer.render(scene,camera);const cost=performance.now()-renderStarted;if(cost>23){slowFrames++;fastFrames=0;}else if(cost<11){fastFrames++;slowFrames=0;}else{slowFrames=0;fastFrames=0;}if(slowFrames>=4&&renderRatio>.75){renderRatio=.75;renderer.setPixelRatio(renderRatio);resize();slowFrames=0;}else if(fastFrames>=180&&renderRatio<Math.min(devicePixelRatio||1,1)){renderRatio=Math.min(devicePixelRatio||1,1);renderer.setPixelRatio(renderRatio);resize();fastFrames=0;}
 };
 raf=requestAnimationFrame(draw);
 return()=>{cancelAnimationFrame(raf);ro?.disconnect();scene.traverse(o=>{const m=o as T.Mesh;if(m.geometry)m.geometry.dispose();const z=m.material;if(z){if(Array.isArray(z))z.forEach(a=>a.dispose());else z.dispose();}});cloneTexture?.dispose();renderer.dispose();renderer.domElement.remove();};
 },[trackingRef]);
 return <div ref={host} className="three-effects-layer" aria-hidden="true"/>;
}
