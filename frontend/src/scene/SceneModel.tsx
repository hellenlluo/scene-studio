import { useGLTF } from "@react-three/drei";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";

import type { Certificate, SceneSpec } from "../api/client";
import type { Pose } from "../certify/physics.worker";
import { failingObjectIds, nodeToObjectId } from "../certify/status";
import { useSceneStore } from "./store";

const COLOURS = {
  // Red is a *certified failure* — measured and objective. Amber (v2 uncertainty) and
  // grey (uncommitted edits) are deliberately different channels: an object can be
  // confidently wrong or uncertainly fine.
  failed: new THREE.Color("#d94b4b"),
  selected: new THREE.Color("#4a9eff"),
};

/**
 * How the status tint is applied, and why it is mostly emissive.
 *
 * The obvious approach — lerp the base colour toward red — competes with the texture
 * for the same channel and loses badly in one direction: red is a mid-dark colour, so
 * blending 65% of it into a reconstructed albedo turned a bright sofa and a patterned
 * rug into near-black shapes. Every failing object read as "unlit", which on a scene
 * where eight of nine objects fail is the whole scene.
 *
 * Emissive does not multiply with the lighting, so it adds the status colour without
 * taking away the albedo. A small lerp still goes in alongside it for saturation on
 * pale objects, where emissive alone washes out rather than reads as red.
 */
const TINT_LERP = 0.12;
const TINT_EMISSIVE = 0.22;

interface Props {
  url: string;
  certificate: Certificate;
  graph: SceneSpec["graph"];
  /** Reports the per-object groups a transform gizmo can attach to. */
  onObjects?: (groups: Map<string, THREE.Object3D>) => void;
  /**
   * Held false until the camera has been aimed at this scene.
   *
   * Without it the model is on screen for the frames between the glTF resolving and
   * the aim being applied, so a scene opens with a visible jump as the view snaps to
   * where it should have started.
   */
  visible?: boolean;
  /**
   * Live physics poses by object id, or undefined when physics is off.
   *
   * Applied over the solved pose rather than replacing it: the groups are rebuilt
   * from `graph` whenever the scene changes, so turning physics off restores the
   * server's placement without anything having to be undone.
   */
  poses?: Map<string, Pose>;
}

export function SceneModel({
  url,
  certificate,
  graph,
  onObjects,
  visible = true,
  poses,
}: Props) {
  const { scene } = useGLTF(url);
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const select = useSceneStore((s) => s.select);

  const failing = useMemo(() => failingObjectIds(certificate), [certificate]);
  // GLTFLoader strips `/` from node names, so the object id cannot be recovered
  // by splitting. Build the mapping forward from the graph instead.
  const nodeOwner = useMemo(() => nodeToObjectId(graph), [graph]);

  // useGLTF caches by URL, so the loaded scene is shared across every component that
  // asks for it. Cloning gives this instance its own nodes and materials to touch.
  const { model, groups } = useMemo(() => {
    const model = scene.clone(true);
    const owner = nodeToObjectId(graph);

    // The exporter emits one node per *part*, flat under the root, because that is
    // what the physics export does and the two have to agree on names. A gizmo edits
    // an *object*, so the parts of each one are gathered under a group first.
    //
    // The group is placed at the object's position and the parts are moved in with
    // `attach`, which rewrites each child's local transform to preserve its world
    // placement. Without that the parts would jump by the group's offset; with it the
    // scene looks identical and the gizmo has a handle at the object's own origin
    // rather than at the world origin.
    const groups = new Map<string, THREE.Object3D>();
    for (const object of graph.objects) {
      const parts: THREE.Object3D[] = [];
      model.traverse((node) => {
        if (owner.get(node.name) === object.object_id && node !== model)
          parts.push(node);
      });
      if (!parts.length) continue;

      const parent = parts[0].parent ?? model;
      const group = new THREE.Group();
      group.name = `object:${object.object_id}`;
      // Scene coordinates, not three.js coordinates: this group lives under the glTF
      // root node, which is what carries the Z-up to Y-up conversion.
      //
      // Orientation as well as position, so the group *is* the object frame — the
      // same frame MJCF gives the root body, which is what lets a physics pose be
      // applied by writing these two fields and nothing else. `attach` below
      // compensates either way, so the render is identical; without the rotation
      // here the parts would carry it and a physics quaternion would compound with
      // it instead of replacing it.
      group.position.set(...(object.position_m as [number, number, number]));
      group.quaternion.set(
        object.orientation[1],
        object.orientation[2],
        object.orientation[3],
        object.orientation[0], // MuJoCo and the schema are w,x,y,z; three.js is x,y,z,w
      );
      parent.add(group);
      for (const part of parts) group.attach(part);
      groups.set(object.object_id, group);
    }
    return { model, groups };
  }, [scene, graph]);

  // Layout effect, not effect: this feeds the camera aim, and the setState it triggers
  // has to run and re-render before the browser paints. On a passive effect the paint
  // lands first and the jump is visible however the visibility gate is wired.
  useLayoutEffect(() => onObjects?.(groups), [groups, onObjects]);

  // Physics poses land straight on the object frames. Writing the group rather than
  // going through React state keeps a 60 Hz stream off the reconciler; the groups
  // are the same objects three.js is already drawing.
  useLayoutEffect(() => {
    if (!poses) return;
    for (const [objectId, group] of groups) {
      const pose = poses.get(objectId);
      if (!pose) continue;
      group.position.set(...pose.position);
      group.quaternion.set(
        pose.quaternion[1],
        pose.quaternion[2],
        pose.quaternion[3],
        pose.quaternion[0],
      );
    }
  }, [poses, groups]);

  useEffect(() => {
    model.traverse((node) => {
      if (!(node instanceof THREE.Mesh)) return;
      const objectId = nodeOwner.get(node.name);
      if (objectId === undefined) return;

      // Stash the reconstructed material on first pass and tint from *that* every
      // time. Tinting `node.material` in place compounds: each selection change would
      // lerp the already-lerped colour another 65% toward blue, so an object drifted
      // further from its real appearance the more you clicked it.
      //
      // Cloning is not optional either. `useGLTF` caches by URL and `scene.clone(true)`
      // shares material instances with the cached original, so mutating one leaks
      // colour into every other view of the scene and survives navigation.
      const base = (node.userData.base ??= (
        Array.isArray(node.material) ? node.material : [node.material]
      ).map((material: THREE.Material) =>
        material.clone(),
      )) as THREE.Material[];

      const tint =
        objectId === selectedObjectId
          ? COLOURS.selected
          : failing.has(objectId)
            ? COLOURS.failed
            : null;

      // Materials hold GPU resources that garbage collection does not reclaim, and
      // this runs on every selection change.
      const previous = Array.isArray(node.material)
        ? node.material
        : [node.material];
      for (const material of previous) {
        if (!base.includes(material)) material.dispose();
      }

      const applied = base.map((material) => {
        if (!tint) return material;
        const copy = material.clone();
        if (copy instanceof THREE.MeshStandardMaterial) {
          copy.color.lerp(tint, TINT_LERP);
          copy.emissive = tint.clone();
          copy.emissiveIntensity = TINT_EMISSIVE;
        }
        return copy;
      });

      node.material = applied.length === 1 ? applied[0] : applied;
      node.castShadow = true;
      node.receiveShadow = true;
    });
  }, [model, failing, selectedObjectId, nodeOwner]);

  return (
    <primitive
      object={model}
      visible={visible}
      onClick={(
        event: THREE.Event & {
          stopPropagation: () => void;
          object: THREE.Object3D;
        },
      ) => {
        event.stopPropagation();
        const objectId = nodeOwner.get(event.object.name);
        if (objectId) select(objectId);
      }}
    />
  );
}
