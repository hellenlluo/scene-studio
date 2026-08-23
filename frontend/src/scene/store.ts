import { create } from "zustand";

/**
 * Selection state, kept out of TanStack Query.
 *
 * The scene itself is server state and lives in the query cache — putting it
 * here too would mean two copies that disagree after a repair. What belongs here
 * is the state the server has no opinion about: which scene is open, and which
 * object the user clicked.
 */
/**
 * Which gizmo a click on an object summons.
 *
 * A mode rather than three separate gizmos: showing translate, rotate and scale
 * handles at once puts nine overlapping grab targets on one object, and picking the
 * one you meant becomes the hard part. Blender, Unity and three.js's own editor all
 * settled on modal W/E/R for the same reason, so the shortcuts are muscle memory
 * rather than a new thing to learn.
 */
export type TransformMode = "translate" | "rotate" | "scale";

interface SceneState {
  sceneId: string | null;
  selectedObjectId: string | null;
  mode: TransformMode;
  setSceneId: (sceneId: string | null) => void;
  select: (objectId: string | null) => void;
  setMode: (mode: TransformMode) => void;
}

export const useSceneStore = create<SceneState>((set) => ({
  sceneId: null,
  selectedObjectId: null,
  // Move by default: it is the edit that is almost always wanted, and the only one
  // whose gizmo cannot damage a scene by being grabbed accidentally — a stray drag
  // shifts an object, where a stray scale silently invalidates its certified size.
  mode: "translate",
  // Clear the selection when the scene changes: object ids are only unique
  // within a scene, so carrying one across would highlight an unrelated object.
  setSceneId: (sceneId) => set({ sceneId, selectedObjectId: null }),
  select: (objectId) => set({ selectedObjectId: objectId }),
  setMode: (mode) => set({ mode }),
}));
