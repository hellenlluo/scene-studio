import { create } from 'zustand'

/**
 * Selection state, kept out of TanStack Query.
 *
 * The scene itself is server state and lives in the query cache — putting it
 * here too would mean two copies that disagree after a repair. What belongs here
 * is the state the server has no opinion about: which scene is open, and which
 * object the user clicked.
 */
interface SceneState {
  sceneId: string | null
  selectedObjectId: string | null
  setSceneId: (sceneId: string | null) => void
  select: (objectId: string | null) => void
}

export const useSceneStore = create<SceneState>((set) => ({
  sceneId: null,
  selectedObjectId: null,
  // Clear the selection when the scene changes: object ids are only unique
  // within a scene, so carrying one across would highlight an unrelated object.
  setSceneId: (sceneId) => set({ sceneId, selectedObjectId: null }),
  select: (objectId) => set({ selectedObjectId: objectId }),
}))
