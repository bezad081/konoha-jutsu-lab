export type Jutsu = 'none' | 'rasengan';
export type Gesture = 'none' | 'open' | 'fist';
export type Point = { x: number; y: number; z: number };
function distance(a:Point,b:Point){return Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z)}
type FingerPose={extended:boolean;folded:boolean};
function readFingerPose(points:Point[],tip:number,pip:number,mcp:number):FingerPose{
  if(points.length<21)return {extended:false,folded:false};
  const dip=points[tip-1],base=points[mcp],joint=points[pip];
  const ax=base.x-joint.x,ay=base.y-joint.y,az=base.z-joint.z;
  const bx=dip.x-joint.x,by=dip.y-joint.y,bz=dip.z-joint.z;
  const proximalLength=Math.hypot(ax,ay,az),distalLength=Math.hypot(bx,by,bz);
  if(proximalLength<1e-5||distalLength<1e-5)return {extended:false,folded:false};
  const cosine=Math.max(-1,Math.min(1,(ax*bx+ay*by+az*bz)/(proximalLength*distalLength)));
  const angle=Math.acos(cosine)*180/Math.PI;
  const reachRatio=distance(points[tip],points[mcp])/Math.max(1e-5,distance(points[pip],points[mcp]));
  // A naturally curved finger is okay; require clear curling before calling it folded.
  return {extended:angle>128&&reachRatio>0.96,folded:angle<108||reachRatio<0.74};
}
/** Classify the visible palm from raw, unmirrored camera landmarks. */
export function isPalmFacingCamera(points:Point[],mediaPipeHandedness?:string):boolean{
  if(points.length<21||(mediaPipeHandedness!=='Left'&&mediaPipeHandedness!=='Right'))return false;
  const wrist=points[0],indexBase=points[5],pinkyBase=points[17];
  const ax=indexBase.x-wrist.x,ay=indexBase.y-wrist.y,az=indexBase.z-wrist.z;
  const bx=pinkyBase.x-wrist.x,by=pinkyBase.y-wrist.y,bz=pinkyBase.z-wrist.z;
  const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
  const normalLength=Math.hypot(nx,ny,nz);
  if(normalLength<1e-6)return false;
  const facingScore=nz/normalLength;
  // At a true side angle, front/back cannot be reliably separated from the palm normal.
  // Let the visible open-finger pose pass through this narrow ambiguous band.
  if(Math.abs(facingScore)<0.12)return true;
  const expectedSign=mediaPipeHandedness==='Left'?1:-1;
  return facingScore*expectedSign>0.12;
}
/** Slightly curled fingers are valid; only a deep, clear fist is a different pose. */
export function classifyHand(points:Point[],mediaPipeHandedness?:string):Gesture{
  if(points.length<21)return 'none';
  const fingers=[readFingerPose(points,8,6,5),readFingerPose(points,12,10,9),readFingerPose(points,16,14,13),readFingerPose(points,20,18,17)];
  const extended=fingers.filter(f=>f.extended).length,folded=fingers.filter(f=>f.folded).length;
  const relaxedOrExtended=4-folded;
  if(folded>=3&&extended<=1)return 'fist';
  if(relaxedOrExtended>=3&&(extended>=1||folded===0))
    return isPalmFacingCamera(points,mediaPipeHandedness)?'open':'none';
  return 'none';
}
export function readJutsu(hands:Point[][],handedness:Array<string|undefined>=[]):{jutsu:Jutsu;gesture:Gesture;handIndex?:number}{
  const gestures=hands.map((hand,index)=>classifyHand(hand,handedness[index]));
  const openHands=gestures.map((gesture,index)=>gesture==='open'?index:-1).filter(index=>index>=0);
  if(openHands.length===1)return {jutsu:'rasengan',gesture:'open',handIndex:openHands[0]};
  if(gestures.includes('fist'))return {jutsu:'none',gesture:'fist'};
  return {jutsu:'none',gesture:'none'};
}