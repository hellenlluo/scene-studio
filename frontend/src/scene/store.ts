import { create } from "zustand";

import type { ObjectEdit } from "../api/client";

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
  /**
   * Drags the user has made but not committed, by object id.
   *
   * Uncommitted deliberately. An edit is a statement about the scene, and a
   * two-pixel nudge that went to the server the moment the mouse came up would
   * re-solve the whole scene on an accident. Committing is also the countable unit
   * that makes "edits to certification" a measurable quantity.
   *
   * These live here rather than in the query cache because the server has no
   * opinion about them yet — that is the definition of uncommitted. While the map
   * is non-empty the certificate on screen describes a scene that is no longer the
   * one being shown, which is what the stale channel says.
   */
  edits: Map<string, ObjectEdit>;
  setSceneId: (sceneId: string | null) => void;
  select: (objectId: string | null) => void;
  setMode: (mode: TransformMode) => void;
  recordEdit: (edit: ObjectEdit) => void;
  clearEdits: () => void;
}

export const useSceneStore = create<SceneState>((set) => ({
  sceneId: null,
  selectedObjectId: null,
  // Move by default: it is the edit that is almost always wanted, and the only one
  // whose gizmo cannot damage a scene by being grabbed accidentally — a stray drag
  // shifts an object, where a stray scale silently invalidates its certified size.
  mode: "translate",
  edits: new Map(),
  // Clear the selection when the scene changes: object ids are only unique
  // within a scene, so carrying one across would highlight an unrelated object.
  // Pending edits go too — they name objects the next scene does not contain, and
  // committing them there would move whatever happened to share an id.
  setSceneId: (sceneId) =>
    set({ sceneId, selectedObjectId: null, edits: new Map() }),
  select: (objectId) => set({ selectedObjectId: objectId }),
  setMode: (mode) => set({ mode }),
  // Merged per object, so moving something and then rotating it is one edit rather
  // than the rotation discarding the move.
  recordEdit: (edit) =>
    set((state) => {
      const edits = new Map(state.edits);
      edits.set(edit.object_id, { ...edits.get(edit.object_id), ...edit });
      return { edits };
    }),
  clearEdits: () => set({ edits: new Map() }),
}));
