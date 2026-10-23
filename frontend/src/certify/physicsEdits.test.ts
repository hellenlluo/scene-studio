import { describe, expect, it } from "vitest";

import type { SceneObject } from "../api/client";
import type { Pose } from "./physics.worker";
import { physicsPoseEdits } from "./physicsEdits";

function object(objectId: string): SceneObject {
  return {
    object_id: objectId,
    label: { object_id: objectId, category: "clock", material: "other" },
    frame: { source: "test", up_axis: "+z", front_axis: "-y", rotation: [1, 0, 0, 0], normalization_scale: 1 },
    parts: [{ part_id: "body", name: "body", proxy_tier: "obb", dims_m: [1, 1, 1], origin_m: [0, 0, 0], collision_mesh_paths: [], source_watertight: false }],
    scale: 1,
    position_m: [0, 0, 0.5],
    orientation: [1, 0, 0, 0],
    provenance: {},
  };
}

describe("physicsPoseEdits", () => {
  it("commits a settled rotation and meaningful translation", () => {
    const pose: Pose = {
      objectId: "clock",
      position: [0.01, 0, 0.49],
      quaternion: [0.995, 0.1, 0, 0],
    };

    expect(physicsPoseEdits([object("clock")], new Map([["clock", pose]]))).toEqual([
      {
        object_id: "clock",
        position_m: pose.position,
        orientation: pose.quaternion,
      },
    ]);
  });

  it("ignores sub-millimetre contact jitter and quaternion sign", () => {
    const pose: Pose = {
      objectId: "clock",
      position: [0.0001, 0, 0.5001],
      quaternion: [-1, 0, 0, 0],
    };

    expect(physicsPoseEdits([object("clock")], new Map([["clock", pose]]))).toEqual([]);
  });
});
