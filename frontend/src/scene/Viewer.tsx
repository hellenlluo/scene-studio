import { Grid, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import {
  Component,
  type ReactNode,
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { SceneEnvelope } from "../api/client";
import { storageUrl } from "../api/client";
import * as THREE from "three";

import { VIEW_DIRECTION, VIEW_DISTANCE, defaultAim, frame } from "./aim";
import type { Pose } from "../certify/physics.worker";
import { SceneModel } from "./SceneModel";
import { useSceneStore } from "./store";

/**
 * One camera, for every scene, for the life of the page.
 *
 * Nothing re-frames on a scene switch, and nothing about the scene enters the default
 * pose: it is `(d, d, d)` looking at the world origin, from `aim.ts`. Two
 * reconstructions are therefore seen from the same place at the same size and are
 * directly comparable, and the world origin is always at the centre of the viewport.
 */
const CAMERA = {
  position: VIEW_DIRECTION.clone().multiplyScalar(VIEW_DISTANCE).toArray(),
  fov: 45,
  up: [0, 1, 0] as [number, number, number],
  near: 0.05,
  far: 200,
};

/**
 * Puts the camera at the default pose, and stops as soon as the user takes over.
 *
 * The guard is the point. "Show a new scene from the default view" and "never move the
 * camera the user set" conflict on a scene switch, so the tie breaks on whether the
 * camera is still a default. Listening to OrbitControls' own `start` event means a drag
 * counts and a scene switch does not.
 *
 * Setting the pose imperatively rather than passing `target` to `<OrbitControls>`: r3f
 * re-applies declarative props on every commit, so that version snapped the view back
 * on every re-render — a repair, a selection.
 */
function SceneAim({
  loaded,
  onAimed,
}: {
  loaded: boolean;
  onAimed: () => void;
}) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as
    | (THREE.EventDispatcher<{ start: object }> & {
        target?: THREE.Vector3;
        update?: () => void;
      })
    | null;
  const touched = useRef(false);

  useEffect(() => {
    if (!controls) return;
    const onStart = () => {
      touched.current = true;
    };
    controls.addEventListener("start", onStart);
    return () => controls.removeEventListener("start", onStart);
  }, [controls]);

  // Layout effect so the camera is in place before the browser paints. The scene is
  // held hidden until `onAimed` fires, and a passive effect would release it one paint
  // too early — which is the flash this is here to prevent.
  useLayoutEffect(() => {
    if (!controls?.target || !loaded) return;
    if (!touched.current) {
      const { position, target } = defaultAim();
      camera.up.set(0, 1, 0);
      camera.position.copy(position);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update?.();
    }
    onAimed();
  }, [controls, camera, loaded, onAimed]);

  return null;
}

/**
 * `F` frames the selection, or the whole scene when nothing is selected.
 *
 * The escape hatch that makes a fixed default camera workable, and the same key Blender,
 * Unity and Maya bind for it. Unlike the default view this keeps whatever direction the
 * user is currently looking from and fits the *distance* to the content — "get me
 * closer to this", not "reset my view", which is why it does not simply re-run
 * `defaultAim`.
 */
function FrameShortcut({
  boxes,
  selected,
}: {
  boxes: THREE.Box3[];
  selected?: THREE.Object3D;
}) {
  const camera = useThree((state) => state.camera);
  const size = useThree((state) => state.size);
  const controls = useThree((state) => state.controls) as {
    target?: THREE.Vector3;
    update?: () => void;
  } | null;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== "f") return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const node = event.target as HTMLElement | null;
      if (node?.isContentEditable) return;
      if (node && /^(INPUT|TEXTAREA|SELECT)$/.test(node.tagName)) return;
      if (!controls?.target) return;

      const subject = selected
        ? [new THREE.Box3().setFromObject(selected)]
        : boxes;
      if (!subject.length) return;

      // The direction the user is currently looking from, not the default one.
      const direction = camera.position.clone().sub(controls.target);
      if (direction.lengthSq() < 1e-9) direction.copy(VIEW_DIRECTION);

      const aim = frame(
        subject,
        direction,
        camera instanceof THREE.PerspectiveCamera ? camera.fov : 45,
        size.width / Math.max(1, size.height),
      );
      camera.position.copy(aim.position);
      camera.lookAt(aim.target);
      controls.target.copy(aim.target);
      controls.update?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [camera, controls, size, boxes, selected]);

  return null;
}

/**
 * Keeps a failed mesh load inside the canvas.
 *
 * `useGLTF` throws when the file is missing, and an uncaught throw during render
 * unmounts the whole tree — observed: one 404 on a scene's `.glb` blanked the entire
 * page, tab bar and certificate panel included, so there was no way to navigate to a
 * scene that *did* load. A scene with no geometry should still show its certificate.
 */
class GeometryBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error("scene geometry failed to load", error);
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

interface Props {
  envelope: SceneEnvelope;
  /** Live physics poses, or an empty map when physics is off. */
  poses?: Map<string, Pose>;
}

export function Viewer({ envelope, poses }: Props) {
  const select = useSceneStore((s) => s.select);
  const mode = useSceneStore((s) => s.mode);
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const recordEdit = useSceneStore((s) => s.recordEdit);
  const gltfPath = envelope.spec.exports?.gltf_path;
  const sceneId = envelope.spec.scene_id;

  // Tagged with the scene it came from. `SceneModel` is keyed on the scene id, so a
  // switch mounts a fresh one — but until its geometry resolves this state still holds
  // the *previous* scene's groups, and aiming the camera at those would frame a scene
  // that is no longer on screen. Null means "nothing loaded for the current scene yet",
  // which is deliberately distinct from an empty map meaning "loaded, and empty".
  const [loaded, setLoaded] = useState<{
    sceneId: string;
    groups: Map<string, THREE.Object3D>;
  } | null>(null);
  const onObjects = useCallback(
    (groups: Map<string, THREE.Object3D>) => setLoaded({ sceneId, groups }),
    [sceneId],
  );
  const groups = loaded?.sceneId === sceneId ? loaded.groups : null;

  // The scene this camera has been aimed at. Compared against the current scene id
  // rather than held as a boolean, so a switch re-hides without a separate reset
  // effect that could run in the wrong order.
  const [aimedFor, setAimedFor] = useState<string | null>(null);
  const onAimed = useCallback(() => setAimedFor(sceneId), [sceneId]);
  const ready = aimedFor === sceneId;

  const selected =
    groups && selectedObjectId ? groups.get(selectedObjectId) : undefined;

  // Per object, not one union box: the union has corners no object reaches, and
  // centring a region that contains empty space centres the empty space too.
  const boxes = useMemo(
    () =>
      groups
        ? [...groups.values()].map((group) =>
            new THREE.Box3().setFromObject(group),
          )
        : [],
    [groups],
  );

  // Read the pose back off the group the gizmo just moved. The group *is* the
  // object frame — position and orientation in scene coordinates — so this is a
  // straight read rather than an inverse of whatever the gizmo did.
  const recordDrag = useCallback(
    (group: THREE.Object3D) => {
      const objectId = group.name.replace(/^object:/, "");
      const base = envelope.spec.graph.objects.find(
        (o) => o.object_id === objectId,
      );
      if (!base) return;
      recordEdit({
        object_id: objectId,
        position_m: [group.position.x, group.position.y, group.position.z],
        // three.js is x,y,z,w; the schema and MuJoCo are w,x,y,z.
        orientation: [
          group.quaternion.w,
          group.quaternion.x,
          group.quaternion.y,
          group.quaternion.z,
        ],
        // The group starts at unit scale, so what the gizmo left there is a factor
        // on the object's own. Averaged because `SceneObject.scale` is a single
        // number by design — anisotropic scale breaks inertia — and a one-axis drag
        // would otherwise be read as a uniform one.
        scale:
          base.scale *
          ((group.scale.x + group.scale.y + group.scale.z) / 3),
      });
    },
    [envelope.spec.graph.objects, recordEdit],
  );

  if (!gltfPath) {
    return (
      <div className="viewer-empty">this scene has no exported geometry</div>
    );
  }

  // updated_at busts the cache. Repair rewrites scene.glb in place at the same URL, so
  // without a version the browser serves the pre-repair copy and the fix looks like it
  // did nothing.
  const url = storageUrl(gltfPath, envelope.updated_at);
  // The same version identifies the geometry, so it is what the subtree is keyed on.
  const geometryKey = `${sceneId}@${envelope.updated_at}`;

  return (
    <Canvas shadows camera={CAMERA} onPointerMissed={() => select(null)}>
      <color attach="background" args={["#15171c"]} />

      {/* Reconstructed base colour maps are dark to begin with — they carry the
          photo's own shading baked in, so the renderer's lighting multiplies shadow by
          shadow. Hence the strong ambient term: it lifts the albedo into view rather
          than modelling a room, which the texture already does.

          The key light is still directional and still casts, because contact shadows
          are what make a resting object look like it is resting rather than floating.
          The fill opposes it so the unlit side does not go to black. */}
      <ambientLight intensity={1.5} />
      <hemisphereLight intensity={1.0} groundColor="#2a2e36" />
      <directionalLight position={[4, 6, 3]} intensity={2.2} castShadow />
      <directionalLight position={[-4, 2, -3]} intensity={0.7} />

      {/* The GLB carries no floor: MJCF's is an infinite plane and a finite box would
          misrepresent its extent, so the grid is drawn here instead. It sits at
          floor_height_m, in the Y-up frame the glTF root establishes. */}
      {/* Hidden along with the model. The grid recedes to the horizon, so a camera
          re-aim is more visible on it than on the geometry — leaving it up during the
          aim would show the very movement the gate exists to hide. */}
      <Grid
        visible={ready}
        args={[12, 12]}
        position={[0, envelope.spec.graph.floor_height_m ?? 0, 0]}
        cellSize={0.25}
        cellColor="#2c313a"
        sectionSize={1}
        sectionColor="#3a414d"
        fadeDistance={18}
        infiniteGrid
      />

      <GeometryBoundary key={`geometry-${sceneId}`}>
        {/* Keyed, so a scene switch tears the old subtree down and falls back to
            nothing. An unkeyed boundary keeps the previous children mounted while the
            next scene's glTF resolves, which put one frame of the *old* geometry on
            screen under the *new* scene's certificate.

            Keyed on `updated_at` as well as the scene id, because a commit or a
            repair rewrites `scene.glb` *in place* — same scene, same URL but for the
            cache-busting version, new geometry. Keyed on the id alone the component
            is never remounted, and its memoised model, its cloned nodes and the
            groups the gizmo has been dragging all survive the change. Committing a
            drag then left the object rendered where it had been dropped while the
            server had already moved it back. */}
        <Suspense key={geometryKey} fallback={null}>
          <SceneModel
            key={geometryKey}
            url={url}
            certificate={envelope.spec.certificate}
            graph={envelope.spec.graph}
            onObjects={onObjects}
            visible={ready}
            poses={poses}
          />
        </Suspense>
      </GeometryBoundary>

      <OrbitControls makeDefault />
      {/* Deliberately not keyed on the scene. Keying it remounts the component and
          resets the ref recording that the user has taken over the camera, so every
          scene switch re-aimed a camera it had no business touching. The effect already
          re-runs per scene: `boxes` and `onAimed` both change with the scene id. */}
      <SceneAim loaded={groups !== null} onAimed={onAimed} />
      <FrameShortcut boxes={boxes} selected={selected} />

      {/* Attaches to the selected object's group, so a drag moves every part of it.
          `makeDefault` on OrbitControls is what lets drei suspend orbiting while a
          handle is held — without it the camera and the object move together and
          neither goes where it was asked. */}
      {selected && (
        <TransformControls
          object={selected}
          mode={mode}
          size={0.8}
          // On mouse-up, not on every frame of the drag: an edit is a statement,
          // and recording one per pointer move would make "how many edits to
          // certification" meaningless as a number.
          onMouseUp={() => recordDrag(selected)}
        />
      )}
    </Canvas>
  );
}
