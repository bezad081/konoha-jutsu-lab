import { describe, expect, it } from 'vitest';
import { classifyHand, readJutsu, type Point } from './gestureClassifier';

const mcp = [5, 9, 13, 17];
const pip = [6, 10, 14, 18];
const dip = [7, 11, 15, 19];
const tip = [8, 12, 16, 20];

function makeHand(extended: boolean[]): Point[] {
  const points: Point[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  points[0] = { x: 0.5, y: 0.9, z: 0 };
  const xs = [0.28, 0.43, 0.58, 0.72];

  for (let i = 0; i < 4; i++) {
    points[mcp[i]] = { x: xs[i], y: 0.58, z: 0 };
    points[pip[i]] = { x: xs[i], y: 0.43, z: 0 };
    points[dip[i]] = { x: xs[i], y: extended[i] ? 0.30 : 0.47, z: 0 };
    points[tip[i]] = { x: xs[i], y: extended[i] ? 0.18 : 0.50, z: 0 };
  }
  return points;
}

describe('Rasengan gesture classifier', () => {
  it('recognizes a stable open palm', () => {
    expect(classifyHand(makeHand([true, true, true, true]))).toBe('open');
  });

  it('recognizes a closed fist', () => {
    expect(classifyHand(makeHand([false, false, false, false]))).toBe('fist');
  });

  it('keeps ambiguous two-finger poses neutral', () => {
    expect(classifyHand(makeHand([true, true, false, false]))).toBe('none');
  });

  it('ignores incomplete landmark arrays', () => {
    expect(classifyHand([{ x: 0.5, y: 0.5, z: 0 }])).toBe('none');
  });

  it('casts with exactly one open palm', () => {
    expect(readJutsu([makeHand([true, true, true, true])])).toEqual({
      jutsu: 'rasengan',
      gesture: 'open',
      handIndex: 0,
    });
  });

  it('does not cast when both hands are open', () => {
    expect(readJutsu([
      makeHand([true, true, true, true]),
      makeHand([true, true, true, true]),
    ])).toEqual({ jutsu: 'none', gesture: 'none' });
  });

  it('reports the fist pose as the release gesture', () => {
    expect(readJutsu([makeHand([false, false, false, false])])).toEqual({
      jutsu: 'none',
      gesture: 'fist',
    });
  });
});
