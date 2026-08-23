import { describe, expect, it } from "vitest";
import * as THREE from "three";

import { VIEW_DIRECTION, VIEW_DISTANCE, defaultAim, frame } from "./aim";

/** A camera set up exactly as the viewer's is. */
const camera = (aim: { position: THREE.Vector3; target: THREE.Vector3 }) => {
  const cam = new THREE.PerspectiveCamera(45, 1.25, 0.05, 500);
  cam.up.set(0, 1, 0);
  cam.position.copy(aim.position);
  cam.lookAt(aim.target);
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  return cam;
};

const box = (x: number, z: number, size = 0.4) =>
  new THREE.Box3(
    new THREE.Vector3(x - size, 0, z - size),
    new THREE.Vector3(x + size, size * 2, z + size),
  );

describe("defaultAim", () => {
  it("sits at equal distance along all three axes and looks at the world origin", () => {
    const { position, target } = defaultAim();
    expect(target.toArray()).toEqual([0, 0, 0]);
    expect(position.x).toBeCloseTo(position.y, 9);
    expect(position.y).toBeCloseTo(position.z, 9);
    expect(position.length()).toBeCloseTo(VIEW_DISTANCE, 9);
  });

  it("puts the world origin at the centre of the frame", () => {
    // The reason the pose is a constant: you can read an object's position off the
    // screen because you always know where the origin is.
    const ndc = new THREE.Vector3(0, 0, 0).project(camera(defaultAim()));
    expect(ndc.x).toBeCloseTo(0, 9);
    expect(ndc.y).toBeCloseTo(0, 9);
  });

  it("is isometric: the horizontal world axes land at plus and minus 30 degrees", () => {
    const cam = camera(defaultAim());
    const screenAngle = (v: THREE.Vector3) => {
      const a = new THREE.Vector3(0, 0, 0).project(cam);
      const b = v.clone().project(cam);
      return THREE.MathUtils.radToDeg(
        Math.atan2(-(b.y - a.y) / 1.25, b.x - a.x),
      );
    };
    expect(screenAngle(new THREE.Vector3(1, 0, 0))).toBeCloseTo(30, 1);
    expect(screenAngle(new THREE.Vector3(0, 0, 1))).toBeCloseTo(150, 1);
  });

  it("is level — no roll", () => {
    const cam = camera(defaultAim());
    const right = new THREE.Vector3(1, 0, 0).transformDirection(
      cam.matrixWorld,
    );
    expect(right.y).toBeCloseTo(0, 9);
  });
});

describe("frame (fit to content)", () => {
  const fit = (boxes: THREE.Box3[], direction = VIEW_DIRECTION) =>
    frame(boxes, direction, 45, 1.25);

  it("fills the frame for a small scene and a large one alike", () => {
    for (const size of [0.2, 1, 8]) {
      const boxes = [box(0, 0, size), box(size * 2, size, size)];
      const aim = fit(boxes);
      const xs = boxes.flatMap((b) =>
        [0, 1, 2, 3, 4, 5, 6, 7].map((i) =>
          new THREE.Vector3(
            i & 1 ? b.max.x : b.min.x,
            i & 2 ? b.max.y : b.min.y,
            i & 4 ? b.max.z : b.min.z,
          ).project(camera(aim)),
        ),
      );
      const worst = Math.max(
        ...xs.map((v) => Math.max(Math.abs(v.x), Math.abs(v.y))),
      );
      // Inside the frustum, and actually filling it rather than a speck in the middle.
      expect(worst).toBeLessThan(1);
      expect(worst).toBeGreaterThan(0.8);
    }
  });

  it("scales the distance with the scene", () => {
    const near = fit([box(0, 0, 0.5)]);
    const far = fit([box(0, 0, 5)]);
    expect(far.position.distanceTo(far.target)).toBeGreaterThan(
      near.position.distanceTo(near.target) * 5,
    );
  });

  it("keeps the direction it was given, so framing does not reset the view", () => {
    const direction = new THREE.Vector3(-2, 1, 0.5);
    const aim = fit([box(1, 1), box(-2, 3)], direction);
    const offset = aim.position.clone().sub(aim.target).normalize();
    expect(offset.angleTo(direction.clone().normalize())).toBeLessThan(1e-6);
  });

  it("honours a fixed distance when one is given", () => {
    const aim = frame([box(0, 0, 4)], VIEW_DIRECTION, 45, 1.25, VIEW_DISTANCE);
    expect(aim.position.distanceTo(aim.target)).toBeCloseTo(VIEW_DISTANCE, 4);
  });
});
