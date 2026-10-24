import type { ObjectEdit, SceneObject } from "../api/client";
import type { Pose } from "./physics.worker";

// Ignore contact jitter. Committing every sub-millimetre numerical nudge would pin
// nearly every object as a user edit when the user only wanted the clock that moved.
const POSITION_EPSILON_M = 0.0005;
const ORIENTATION_EPSILON_DEG = 0.05;

function orientationDifferenceDeg(
  first: readonly number[],
  second: readonly number[],
): number {
  const firstNorm = Math.hypot(...first);
  const secondNorm = Math.hypot(...second);
  if (firstNorm === 0 || secondNorm === 0) return Number.POSITIVE_INFINITY;
  const dot = Math.abs(
    first.reduce(
      (sum, value, index) => sum + value * second[index],
      0,
    ) /
      (firstNorm * secondNorm),
  );
  return (2 * Math.acos(Math.min(1, dot)) * 180) / Math.PI;
}

/** Convert the visible paused simulation into the smallest exact pose edit. */
export function physicsPoseEdits(
  objects: SceneObject[],
  poses: Map<string, Pose>,
): ObjectEdit[] {
  const edits: ObjectEdit[] = [];
  for (const object of objects) {
    const pose = poses.get(object.object_id);
    if (!pose) continue;

    const positionChanged = Math.hypot(
      pose.position[0] - object.position_m[0],
      pose.position[1] - object.position_m[1],
      pose.position[2] - object.position_m[2],
    ) > POSITION_EPSILON_M;
    const orientationChanged =
      orientationDifferenceDeg(pose.quaternion, object.orientation) >
      ORIENTATION_EPSILON_DEG;
    if (!positionChanged && !orientationChanged) continue;

    edits.push({
      object_id: object.object_id,
      ...(positionChanged ? { position_m: [...pose.position] } : {}),
      ...(orientationChanged ? { orientation: [...pose.quaternion] } : {}),
    });
  }
  return edits;
}

/** Preserve non-pose draft fields and replace draft poses with the paused result.
 *
 * Physics reports every root pose, including an edited object which happened to
 * return exactly to its committed pose. Removing the draft pose before merging is
 * important in that case: spreading only the non-empty physics delta over it would
 * accidentally re-commit the pre-simulation position.
 */
export function mergeDraftWithPhysicsEdits(
  objects: SceneObject[],
  poses: Map<string, Pose>,
  drafts: ObjectEdit[],
): ObjectEdit[] {
  const merged = new Map<string, ObjectEdit>();
  for (const draft of drafts) {
    if (!poses.has(draft.object_id)) {
      merged.set(draft.object_id, { ...draft });
      continue;
    }
    const nonPose: ObjectEdit = { object_id: draft.object_id };
    if (draft.scale !== undefined) nonPose.scale = draft.scale;
    if (draft.mass_kg !== undefined) nonPose.mass_kg = draft.mass_kg;
    if (draft.supported_by !== undefined) {
      nonPose.supported_by = draft.supported_by;
    }
    if (Object.keys(nonPose).length > 1) merged.set(draft.object_id, nonPose);
  }

  for (const poseEdit of physicsPoseEdits(objects, poses)) {
    merged.set(poseEdit.object_id, {
      ...merged.get(poseEdit.object_id),
      ...poseEdit,
    });
  }
  return [...merged.values()];
}
