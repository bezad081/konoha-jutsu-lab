export type Jutsu = 'none' | 'rasengan';
export type Gesture = 'none' | 'open' | 'fist';
export type Point = { x: number; y: number; z: number };

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * 0.5);
}

function fingerExtended(points: Point[], tip: number, pip: number, mcp: number) {
  if (points.length < 21) return false;

  // The distal interphalangeal joint gives a better bend signal than wrist distance alone.
  const dip = points[tip - 1];
  const base = points[mcp];
  const joint = points[pip];
  const ax = base.x - joint.x;
  const ay = base.y - joint.y;
  const az = (base.z - joint.z) * 0.35;
  const bx = dip.x - joint.x;
  const by = dip.y - joint.y;
  const bz = (dip.z - joint.z) * 0.35;
  const la = Math.hypot(ax, ay, az);
  const lb = Math.hypot(bx, by, bz);
  if (la < 1e-5 || lb < 1e-5) return false;

  const cosine = Math.max(-1, Math.min(1, (ax * bx + ay * by + az * bz) / (la * lb)));
  const angle = Math.acos(cosine) * 180 / Math.PI;
  return angle > 148 && distance(points[tip], points[mcp]) > distance(points[pip], points[mcp]) * 1.08;
}

/** Classify the four fingers that are most reliable under a mirrored selfie camera. */
export function classifyHand(points: Point[]): Gesture {
  if (points.length < 21) return 'none';

  const extended = [
    fingerExtended(points, 8, 6, 5),
    fingerExtended(points, 12, 10, 9),
    fingerExtended(points, 16, 14, 13),
    fingerExtended(points, 20, 18, 17),
  ];
  const count = extended.filter(Boolean).length;

  // Leave two-finger and partially occluded poses neutral; temporal voting in App
  // confirms a stable open palm before a cast and avoids turning ambiguity into a release.
  if (count >= 3) return 'open';
  if (count <= 1) return 'fist';
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
