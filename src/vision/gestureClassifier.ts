export type Jutsu = 'none' | 'rasengan';
export type Gesture = 'none' | 'open' | 'fist';
export type Point = { x: number; y: number; z: number };

function distance(a: Point, b: Point) {
  // MediaPipe z carries useful depth when the hand turns edge-on to the camera.
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

type FingerPose = { extended: boolean; folded: boolean };

function readFingerPose(points: Point[], tip: number, pip: number, mcp: number): FingerPose {
  if (points.length < 21) return { extended: false, folded: false };

  const dip = points[tip - 1];
  const base = points[mcp];
  const joint = points[pip];
  const ax = base.x - joint.x, ay = base.y - joint.y, az = base.z - joint.z;
  const bx = dip.x - joint.x, by = dip.y - joint.y, bz = dip.z - joint.z;
  const proximalLength = Math.hypot(ax, ay, az), distalLength = Math.hypot(bx, by, bz);
  if (proximalLength < 1e-5 || distalLength < 1e-5) return { extended: false, folded: false };

  // Full 3D angle stays stable as the palm rolls or turns away from the camera.
  const cosine = Math.max(-1, Math.min(1, (ax * bx + ay * by + az * bz) / (proximalLength * distalLength)));
  const angle = Math.acos(cosine) * 180 / Math.PI;
  const reachRatio = distance(points[tip], points[mcp]) / Math.max(1e-5, distance(points[pip], points[mcp]));

  // Use both straightness and reach: foreshortening or one noisy joint should not
  // instantly turn an open palm into a fist; uncertain fingers stay neutral.
  return { extended: angle > 142 && reachRatio > 1.04, folded: angle < 132 || reachRatio < 0.82 };
}

/** Classify four fingers using 3D geometry to tolerate hand rotation and foreshortening. */
export function classifyHand(points: Point[]): Gesture {
  if (points.length < 21) return 'none';
  const fingers = [
    readFingerPose(points, 8, 6, 5),
    readFingerPose(points, 12, 10, 9),
    readFingerPose(points, 16, 14, 13),
    readFingerPose(points, 20, 18, 17),
  ];
  const extended = fingers.filter((finger) => finger.extended).length;
  const folded = fingers.filter((finger) => finger.folded).length;
  if (extended >= 3) return 'open';
  if (folded >= 3 && extended <= 1) return 'fist';
  return 'none';
}

export function readJutsu(hands: Point[][]): { jutsu: Jutsu; gesture: Gesture; handIndex?: number } {
  const gestures = hands.map(classifyHand);
  const openHands = gestures.map((gesture, index) => gesture === 'open' ? index : -1).filter(index => index >= 0);
  // Requiring exactly one open palm prevents two-hand poses from accidentally casting.
  if (openHands.length === 1) return { jutsu: 'rasengan', gesture: 'open', handIndex: openHands[0] };
  if (gestures.includes('fist')) return { jutsu: 'none', gesture: 'fist' };
  return { jutsu: 'none', gesture: 'none' };
}
