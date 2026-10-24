/**
 * Typed client over the backend.
 *
 * Every type here is pulled from `schema.d.ts`, which is generated from the
 * backend's own OpenAPI document. Hand-writing them would guarantee drift: the
 * scene contract is 33 schemas deep and changes whenever a stage does.
 */
import type { components } from './schema'

/**
 * Fields with a Pydantic default are optional in the OpenAPI document — correct
 * for a request body, wrong for a response, where the server always fills them
 * in. Narrowing here once keeps every component from carrying null checks for a
 * case that cannot happen.
 */
type Always<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>

export type SceneSpec = Always<
  components['schemas']['SceneSpec'],
  | 'certificate'
  | 'exports'
  | 'anchors'
  | 'weights'
  | 'diagnostics'
  | 'repairs_applied'
  | 'outcome'
  | 'outcome_reasons'
>

export type SceneEnvelope = Omit<components['schemas']['SceneEnvelope'], 'spec'> & {
  spec: SceneSpec
}
export type SceneSummary = components['schemas']['SceneSummary']
export type RepairResponse = components['schemas']['RepairResponse']
export type SupportChange = components['schemas']['SupportChange']
export type SceneEditResponse = Omit<
  components['schemas']['SceneEditResponse'],
  'scene' | 'actions' | 'support_changes'
> & { scene: SceneEnvelope; actions: RepairAction[]; support_changes: SupportChange[] }
export type Certificate = components['schemas']['Certificate']
export type SceneObject = components['schemas']['SceneObject']
export type ScaleCheck = components['schemas']['ScaleCheck']
export type BalanceCheck = components['schemas']['BalanceCheck']
export type StabilityCheck = components['schemas']['StabilityCheck']
export type InertialCheck = components['schemas']['InertialCheck']
export type AxisStatus = components['schemas']['AxisStatus']
export type RepairAction = components['schemas']['RepairAction']
export type PhysicsBundle = components['schemas']['PhysicsBundle']
export type PhysicsPreviewRequest = components['schemas']['PhysicsPreviewRequest']
export type ObjectEdit = components['schemas']['ObjectEdit']
export type SceneEditRequest = components['schemas']['SceneEditRequest']
export type JobStatus = components['schemas']['JobStatus']

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    // The body usually carries FastAPI's detail, which is far more useful than
    // the status alone when a scene id is wrong or a scene will not load.
    const body = await response.text().catch(() => '')
    throw new Error(`${init?.method ?? 'GET'} ${path} → ${response.status} ${body}`.trim())
  }
  return response.json() as Promise<T>
}

export const api = {
  createJob: (image: File) => {
    const body = new FormData()
    body.append('image', image)
    // Do not set Content-Type manually: the browser supplies the multipart
    // boundary, without which FastAPI cannot read the uploaded file.
    return request<JobStatus>('/api/jobs', { method: 'POST', body })
  },
  getJob: (jobId: string) => request<JobStatus>(`/api/jobs/${jobId}`),
  listScenes: () => request<SceneSummary[]>('/api/scenes'),
  getScene: (sceneId: string) => request<SceneEnvelope>(`/api/scenes/${sceneId}`),
  repairScene: (sceneId: string) =>
    request<RepairResponse>(`/api/scenes/${sceneId}/repair`, { method: 'POST' }),
  // The MJCF plus its mesh manifest. Built from the stored graph server-side, so it
  // cannot disagree with the scene on screen the way a cached `scene.xml` URL could.
  scenePhysics: (sceneId: string) =>
    request<PhysicsBundle>(`/api/scenes/${sceneId}/physics`),
  previewScenePhysics: (sceneId: string, preview: PhysicsPreviewRequest) =>
    request<PhysicsBundle>(`/api/scenes/${sceneId}/physics/preview`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(preview),
    }),
  // Commit: the server re-derives what each moved object rests on, re-solves,
  // certifies, and repairs — every step bounded to the objects the edit touched
  // and whatever rests on them. Nothing outside that set moves, which was the
  // reason repair used to be refused here; `actions` reports what it did, so a
  // commit is never a silent change.
  editScene: (sceneId: string, edit: SceneEditRequest) =>
    request<SceneEditResponse>(`/api/scenes/${sceneId}`, {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(edit),
    }),
}

/**
 * The URL for a file the backend wrote under its storage directory.
 *
 * Export paths are stored relative to storage_dir precisely so this is
 * derivable. `version` busts the cache: repair rewrites `scene.glb` in place at
 * the same URL, and without it the browser serves the copy from before the
 * repair — which looks exactly like the repair having done nothing.
 */
export function storageUrl(path: string, version?: string): string {
  const url = `/storage/${path}`
  return version ? `${url}?v=${encodeURIComponent(version)}` : url
}
