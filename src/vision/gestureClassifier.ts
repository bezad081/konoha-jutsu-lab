export type Jutsu = 'none' | 'rasengan';
export type Gesture = 'none' | 'open' | 'fist';
export type Point = { x: number; y: number; z: number };

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

type FingerPose = { extended: boolean; folded: boolean };

function readFingerPose(points: Point[], tip: number, pip: number, mcp: number): FingerPose {
  if (points.length < 21) return { extended: false, folded: false };
  const dip = points[tip - 1], base = points[mcp], joint = points[pip];
  const ax = base.x - joint.x, ay = base.y - joint.y, az = base.z - joint.z;
  const bx = dip.x - joint.x, by = dip.y - joint.y, bz = dip.z - joint.z;
  const proximalLength = Math.hypot(ax, ay, az), distalLength = Math.hypot(bx, by, bz);
  if (proximalLength < 1e-5 || distalLength < 1e-5) return { extended: false, folded: false };

  const cosine = Math.max(-1, Math.min(1, (ax * bx + ay * by + az * bz) / (proximalLength * distalLength)));
  const angle = Math.acos(cosine) * 180 / Math.PI;
  const reachRatio = distance(points[tip], points[mcp]) / Math.max(1e-5, distance(points[pip], points[mcp]));
  return { extended: angle > 142 && reachRatio > 1.04, folded: angle < 132 || reachRatio < 0.82 };
}

/**
 * Face-side gate for the live, unmirrored camera frames.
 * MediaPipe's handedness labels assume a mirrored selfie frame, so invert the label
 * before comparing the palm-plane normal. If the plane is nearly edge-on, fail closed.
 */
export function isPalmFacingCamera(points: Point[], mediaPipeHandedness?: string): boolean {
  if (points.length < 21 || (mediaPipeHandedness !== 'Left' && mediaPipeHandedness !== 'Right')) return false;
  const wrist = points[0], indexBase = points[5], pinkyBase = points[17];
  const ax = indexBase.x - wrist.x, ay = indexBase.y - wrist.y, az = indexBase.z - wrist.z;
  const bx = pinkyBase.x - wrist.x, by = pinkyBase.y - wrist.y, bz = pinkyBase.z - wrist.z;
  const nx = ay * bz - az * by;
  const ny = az * bx - ax * bz;
  const nz = ax * by - ay * bx;
  const normalLength = Math.hypot(nx, ny, nz);
  if (normalLength < 1e-6) return false;
  const facingScore = nz / normalLength;
  // Require a clear camera-facing component; avoids treating an edge-on hand as an open palm.
  if (Math.abs(facingScore) < 0.12) return false;
  // Raw frames are not mirrored before inference: output "Left" maps to the person's right hand.
  const expectedSign = mediaPipeHandedness === 'Left' ? 1 : -1;
  return facingScore * expectedSign > 0;
}

/** A palm, not just four extended fingers, is required to cast Rasengan. */
export function classifyHand(points: Point[], mediaPipeHandedness?: string): Gesture {
  if (points.length < 21) return 'none';
  const fingers = [
    readFingerPose(points, 8, 6, 5),
    readFingerPose(points, 12, 10, 9),
    readFingerPose(points, 16, 14, 13),
    readFingerPose(points, 20, 18, 17),
  ];
  const extended = fingers.filter((finger) => finger.extended).length;
  const folded = fingers.filter((finger) => finger.folded).length;

  if (extended >= 3) return isPalmFacingCamera(points, mediaPipeHandedness) ? 'open' : 'none';
  if (folded >= 3 && extended <= 1) return 'fist';
  return 'none';
}

export function readJutsu(
  hands: Point[][],
  handedness: Array<string | undefined> = [],
): { jutsu: Jutsu; gesture: Gesture; handIndex?: number } {
  const gestures = hands.map((hand, index) => classifyHand(hand, handedness[index]));
  const openHands = gestures.map((gesture, index) => gesture === 'open' ? index : -1).filter(index => index >= 0);
  if (openHands.length === 1) return { jutsu: 'rasengan', gesture: 'open', handIndex: openHands[0] };
  if (gestures.includes('fist')) return { jutsu: 'none', gesture: 'fist' };
  return { jutsu: 'none', gesture: 'none' };
}
