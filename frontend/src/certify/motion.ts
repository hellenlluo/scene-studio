export type Quaternion = readonly [number, number, number, number];

function gravityInBody(q: Quaternion): [number, number, number] {
  const norm = Math.hypot(...q);
  const [w, x, y, z] = q.map((value) => value / norm);
  return [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}

/** Gravity-relative tilt. A rotation around world-up is deliberately invisible. */
export function quaternionTiltDeg(a: Quaternion, b: Quaternion): number {
  const gravityA = gravityInBody(a);
  const gravityB = gravityInBody(b);
  const dot = gravityA.reduce(
    (sum, value, index) => sum + value * gravityB[index],
    0,
  );
  return (Math.acos(Math.max(-1, Math.min(1, dot))) * 180) / Math.PI;
}
