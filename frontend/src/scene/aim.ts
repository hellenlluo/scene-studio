import * as THREE from "three";

/**
 * Direction from the world origin to the camera: equal along X, Y and Z.
 *
 * Equal components are the definition of an isometric view — the three axes are then
 * equally foreshortened, which puts them 120 degrees apart on screen and the two
 * horizontal ones at plus and minus 30 degrees from horizontal. In angles that is 45
 * round and 35.26 up.
 */
export const VIEW_DIRECTION = new THREE.Vector3(1, 1, 1).normalize();

/**
 * How far out along that diagonal, so the camera sits at `(d, d, d)` for
 * `d = VIEW_DISTANCE / sqrt(3)` — about 2.6 m on each axis. Frames a 3 m room with
 * air around it.
 */
export const VIEW_DISTANCE = 4.5;

/** Air left around the content when fitting distance to it. */
const FIT_MARGIN = 1.1;

export interface Aim {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

/**
 * Where to point a fixed-direction camera so the scene lands in the middle of the
 * frame.
 *
 * Recentring the scene graph on the origin — which `geometry.recentre` does — is not
 * enough, and the reason is worth writing down. An isometric camera's screen-x axis is
 * the world diagonal `(x + y)/sqrt(2)`, not world x. A scene can have a perfectly
 * centred axis-aligned bounding box and still sit far off-centre along that diagonal,
 * and rooms routinely do: on room.jpg the union AABB centres at x=0.000 and y=0.000
 * while the same objects project to a screen-x range centred at +0.488 m, because the
 * floor lamp sits at (+x, +y) and nothing occupies (-x, -y). Measured, that was 95 px
 * of a 1060 px viewport.
 *
 * Taking `boxes` per object rather than one union box is the other half of it. The
 * union box has corners no object reaches — its extremes project further out than
 * anything visible, so centring it centres a region containing empty space.
 *
 * Only the *aim* is derived; direction and distance are constants. Two consequences
 * worth keeping: every scene is seen from the same angle at the same size, so two
 * reconstructions are directly comparable; and this never runs once the user has
 * touched the camera.
 */
/**
 * Position a camera along `direction` so `boxes` sit centred in the frame.
 *
 * Pass `fixedDistance` to hold the camera at a set distance and solve only the aim —
 * that is the default view, where every scene is seen from the same angle at the same
 * size so two reconstructions are directly comparable. Omit it to fit the distance to
 * the content as well, which is what an explicit "frame this" does.
 *
 * Two things make this harder than averaging the corners.
 *
 * **An isometric camera's screen-x axis is the world diagonal `(x + y)/sqrt(2)`, not
 * world x.** A scene can have a perfectly centred axis-aligned bounding box and still
 * sit far off-centre along that diagonal, and rooms routinely do: on room.jpg the union
 * AABB centres at x=0.000 and y=0.000 while the same objects project to a screen-x
 * range centred at +0.488 m, because the floor lamp sits at (+x, +y) and nothing
 * occupies (-x, -y). Measured, that was 95 px of a 1060 px viewport.
 *
 * **Boxes are taken per object, not as one union.** The union box has corners no object
 * reaches; its extremes project further out than anything visible, so centring it
 * centres a region containing empty space.
 *
 * Solved by iteration rather than in closed form, because the projection is
 * perspective. Averaging world offsets along the screen axes centres an *orthographic*
 * view and leaves real error — measured on the room.jpg layout, a quarter of a frame.
 * Each pass moves the target by the offset the current projection shows and scales the
 * distance by the overflow it shows, both applied at the target's own depth where they
 * are very nearly exact; what remains is second order. Converges in two or three passes.
 */
export function frame(
  boxes: THREE.Box3[],
  direction: THREE.Vector3,
  fov: number,
  aspect: number,
  fixedDistance?: number,
): Aim {
  const unit = direction.clone().normalize();
  const target = new THREE.Vector3();
  let distance = fixedDistance ?? VIEW_DISTANCE;
  const place = () => target.clone().addScaledVector(unit, distance);
  if (!boxes.length) return { position: place(), target };

  // The camera's own screen axes. `up` is world up by construction — the scene is
  // gravity-aligned, so the horizon must stay level.
  const forward = unit.clone().negate();
  const right = new THREE.Vector3()
    .crossVectors(forward, new THREE.Vector3(0, 1, 0))
    .normalize();
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();

  const corners = boxes.flatMap((box) =>
    [0, 1, 2, 3, 4, 5, 6, 7].map(
      (i) =>
        new THREE.Vector3(
          i & 1 ? box.max.x : box.min.x,
          i & 2 ? box.max.y : box.min.y,
          i & 4 ? box.max.z : box.min.z,
        ),
    ),
  );

  const camera = new THREE.PerspectiveCamera(fov, aspect, 0.05, 500);
  camera.up.set(0, 1, 0);

  for (let pass = 0; pass < 8; pass++) {
    camera.position.copy(place());
    camera.lookAt(target);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();

    let lowX = Infinity;
    let highX = -Infinity;
    let lowY = Infinity;
    let highY = -Infinity;
    for (const corner of corners) {
      const ndc = corner.clone().project(camera);
      lowX = Math.min(lowX, ndc.x);
      highX = Math.max(highX, ndc.x);
      lowY = Math.min(lowY, ndc.y);
      highY = Math.max(highY, ndc.y);
    }

    const offX = (lowX + highX) / 2;
    const offY = (lowY + highY) / 2;
    const overflow =
      Math.max((highX - lowX) / 2, (highY - lowY) / 2) * FIT_MARGIN;

    const centred = Math.abs(offX) < 1e-4 && Math.abs(offY) < 1e-4;
    const sized = fixedDistance !== undefined || Math.abs(overflow - 1) < 1e-3;
    if (centred && sized) break;

    const halfHeight = Math.tan(THREE.MathUtils.degToRad(fov) / 2) * distance;
    target.addScaledVector(right, offX * halfHeight * aspect);
    target.addScaledVector(up, offY * halfHeight);
    if (fixedDistance === undefined) distance *= overflow;
  }

  return { position: place(), target };
}

/**
 * The pose every scene opens at: `(d, d, d)`, looking at the world origin.
 *
 * A constant, and nothing about the scene enters it. That is the point — it is the
 * default view of a 3D editor, where the camera sits at equal distance along all three
 * axes and looks at the world centre, and where you can therefore read an object's
 * position off the screen because you know exactly where the origin is.
 *
 * This deliberately replaces an earlier version that solved for the aim that centred
 * the scene's projection. That was correct about a real problem — a scene whose
 * bounding box is centred can still project off-centre, because an isometric view
 * projects along the world diagonal — but it bought centred pixels at the cost of a
 * camera that pointed somewhere unnameable, 0.586 m off the origin on room.jpg. The
 * origin being *at the centre of the viewport* is worth more than the content being
 * perfectly balanced within it, and `geometry.recentre` already puts the scene there.
 *
 * `frame()` still exists and still solves properly; it is what `F` uses.
 */
export function defaultAim(): Aim {
  return {
    position: VIEW_DIRECTION.clone().multiplyScalar(VIEW_DISTANCE),
    target: new THREE.Vector3(0, 0, 0),
  };
}
