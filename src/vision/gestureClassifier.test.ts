import { describe, expect, it } from 'vitest';
import { classifyHand, isPalmFacingCamera, readJutsu, type Point } from './gestureClassifier';

function makeHand(extended: boolean[]): Point[] {
  const points: Point[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  points[0] = { x: 0.5, y: 0.9, z: 0 };
  const xs = [0.28, 0.43, 0.58, 0.72];
  const mcp = [5, 9, 13, 17], pip = [6, 10, 14, 18], dip = [7, 11, 15, 19], tip = [8, 12, 16, 20];
  for (let i = 0; i < 4; i++) {
    points[mcp[i]] = { x: xs[i], y: 0.58, z: 0 };
    points[pip[i]] = { x: xs[i], y: 0.43, z: 0 };
    points[dip[i]] = { x: xs[i], y: extended[i] ? 0.30 : 0.47, z: 0 };
    points[tip[i]] = { x: xs[i], y: extended[i] ? 0.18 : 0.50, z: 0 };
  }
  return points;
}

function rotateHand(points: Point[], depthTilt: number, imageRoll: number): Point[] {
  const depthCos = Math.cos(depthTilt), depthSin = Math.sin(depthTilt);
  const rollCos = Math.cos(imageRoll), rollSin = Math.sin(imageRoll);
  return points.map((point) => {
    const y = point.y * depthCos - point.z * depthSin;
    const z = point.y * depthSin + point.z * depthCos;
    return { x: point.x * rollCos - y * rollSin, y: point.x * rollSin + y * rollCos, z };
  });
}

function showBackOfHand(points: Point[]): Point[] {
  return points.map((point) => ({ ...point, x: 1 - point.x }));
}
function gentlyCurlFingers(points:Point[],degrees:number[]):Point[]{
  const result=points.map(p=>({...p})),pip=[6,10,14,18],dip=[7,11,15,19],tip=[8,12,16,20];
  degrees.forEach((deg,i)=>{const a=deg*Math.PI/180,seg=0.13;
    result[dip[i]]={x:result[pip[i]].x+Math.sin(a)*seg,y:result[pip[i]].y-Math.cos(a)*seg,z:result[pip[i]].z};
    const b=a+0.18;
    result[tip[i]]={x:result[dip[i]].x+Math.sin(b)*0.105,y:result[dip[i]].y-Math.cos(b)*0.105,z:result[dip[i]].z};
  });return result;
}

describe('Rasengan gesture classifier', () => {
  const palm = () => makeHand([true, true, true, true]);

  it('recognizes an open palm facing the camera', () => {
    const hand = palm();
    expect(isPalmFacingCamera(hand, 'Left')).toBe(true);
    expect(classifyHand(hand, 'Left')).toBe('open');
  });

  it('does not activate for the back of the same hand', () => {
    const back = showBackOfHand(palm());
    expect(isPalmFacingCamera(back, 'Left')).toBe(false);
    expect(classifyHand(back, 'Left')).toBe('none');
  });

  it('recognizes the other hand when its palm faces the camera', () => {
    const otherPalm = showBackOfHand(palm());
    expect(isPalmFacingCamera(otherPalm, 'Right')).toBe(true);
    expect(classifyHand(otherPalm, 'Right')).toBe('open');
  });

  it('keeps working through an oblique palm angle', () => {
    const obliquePalm = rotateHand(palm(), 68 * Math.PI / 180, 77 * Math.PI / 180);
    expect(classifyHand(obliquePalm, 'Left')).toBe('open');
  });

  it('accepts an oblique palm almost sideways to the camera', () => {
    const sidePalm = rotateHand(palm(), 84.5 * Math.PI / 180, 12 * Math.PI / 180);
    expect(isPalmFacingCamera(sidePalm, 'Left')).toBe(true);
    expect(classifyHand(sidePalm, 'Left')).toBe('open');
  });
  it('rejects an exactly edge-on palm where front/back is ambiguous', () => {
    const edgeOnPalm = rotateHand(palm(), 89.5 * Math.PI / 180, 0);
    expect(classifyHand(edgeOnPalm, 'Left')).toBe('none');
  });
  it('recognizes a relaxed palm with partly curled fingers', () => {
    const relaxedPalm = gentlyCurlFingers(palm(), [48, 48, 58, 58]);
    expect(classifyHand(relaxedPalm, 'Left')).toBe('open');
    expect(readJutsu([relaxedPalm], ['Left']).jutsu).toBe('rasengan');
  });

  it('recognizes a closed fist even without a reliable facing estimate', () => {
    expect(classifyHand(makeHand([false, false, false, false]))).toBe('fist');
  });

  it('keeps ambiguous two-finger poses neutral', () => {
    expect(classifyHand(makeHand([true, true, false, false]), 'Left')).toBe('none');
  });

  it('ignores incomplete landmark arrays', () => {
    expect(classifyHand([{ x: 0.5, y: 0.5, z: 0 }], 'Left')).toBe('none');
  });

  it('casts with exactly one camera-facing open palm', () => {
    expect(readJutsu([palm()], ['Left'])).toEqual({ jutsu: 'rasengan', gesture: 'open', handIndex: 0 });
  });

  it('does not cast if the shown hand is a back-of-hand pose', () => {
    expect(readJutsu([showBackOfHand(palm())], ['Left'])).toEqual({ jutsu: 'none', gesture: 'none' });
  });

  it('does not cast when both palms face the camera', () => {
    expect(readJutsu([palm(), showBackOfHand(palm())], ['Left', 'Right'])).toEqual({ jutsu: 'none', gesture: 'none' });
  });
});
