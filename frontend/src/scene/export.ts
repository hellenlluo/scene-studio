/**
 * Downloading a scene: the glTF as one file, the MJCF as an archive.
 *
 * Both artefacts already exist — the pipeline writes `scene.glb` and `scene.xml`
 * on every solve and every repair, and the viewer and the physics worker read
 * them. What was missing was a way for the user to *leave* with them.
 *
 * **The two are asymmetric, and the asymmetry is the whole design here.** A GLB
 * is self-contained: geometry and base colour textures are in the one binary, so
 * exporting it is a link with a `download` attribute and no bytes through this
 * process at all. An MJCF is not. `mjcf.build_xml` names its mesh assets by bare
 * filename with no `meshdir` — deliberately, so a consumer without a filesystem
 * can satisfy them from a virtual one — which means `scene.xml` alone is a file
 * MuJoCo cannot open. So the MJCF export is a zip of the XML plus every mesh it
 * names, flat, at exactly the names it names them by. Unzip and `mjcf` loads.
 *
 * **The MJCF comes from `/physics`, not from `/storage/scene.xml`.** Same reason
 * the worker does: it is built from the stored graph, so it cannot be the
 * pre-repair copy that a cached URL would hand back. The mesh URLs come with it,
 * so nothing here parses the XML — the naming stays `mjcf`'s business.
 */
import { zip } from "fflate";

import type { SceneEnvelope } from "../api/client";
import { api, storageUrl } from "../api/client";

/** In-flight mesh fetches. Six is the browser's own per-host cap; eight keeps the
 * queue fed without making a 271-asset scene a burst of 271 pending requests. */
const FETCH_CONCURRENCY = 8;

export interface ExportProgress {
  fetched: number;
  total: number;
}

/**
 * Hand a blob to the browser as a download.
 *
 * The object URL is revoked on a later task, not straight after the click:
 * revoking it in the same task cancels the download in Firefox, which is a bug
 * that looks exactly like the button doing nothing.
 */
function save(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Run `tasks` with at most `limit` in flight, and reject as soon as one does.
 *
 * `Promise.all` over every fetch at once would work and is shorter, but a scene
 * with hundreds of collision proxies would open hundreds of requests for the
 * browser to queue anyway — and the progress count would go from 0 to done in
 * one jump, which for a multi-second export is the same as having no count.
 */
async function pooled(limit: number, tasks: (() => Promise<void>)[]): Promise<void> {
  let next = 0;
  const runners = Array.from({ length: Math.min(limit, tasks.length) }, async () => {
    while (next < tasks.length) await tasks[next++]();
  });
  await Promise.all(runners);
}

/** Whether this scene has a glTF export to download. Absent only if the export
 * stage never ran, which a seeded or hand-authored scene can be. */
export function hasGltf(envelope: SceneEnvelope): boolean {
  return Boolean(envelope.spec.exports.gltf_path);
}

/**
 * Download `scene.glb`.
 *
 * Straight off the `/storage` mount rather than fetched and re-blobbed: the file
 * is self-contained and can be tens of megabytes, and there is no reason for it
 * to pass through JS memory to be handed back to the same browser. `updated_at`
 * busts the cache for the same reason the viewer needs it to — repair rewrites
 * this file in place at this URL.
 */
export function downloadGltf(envelope: SceneEnvelope): void {
  const path = envelope.spec.exports.gltf_path;
  if (!path) throw new Error("this scene has no glTF export");
  const link = document.createElement("a");
  link.href = storageUrl(path, envelope.updated_at);
  // Named for the scene rather than kept as `scene.glb`, which is what every
  // scene's file is called and so collides in a downloads folder.
  link.download = `${envelope.spec.scene_id}.glb`;
  link.click();
}

/**
 * Download the MJCF and its meshes as one zip.
 *
 * Deflated rather than stored: the payload is Wavefront OBJ, which is decimal
 * text and compresses several-fold. Zipped off the main thread — `fflate`'s
 * async entry point moves the work to a worker — because the viewer is on it and
 * a scene's proxies add up to enough bytes to drop frames.
 */
export async function downloadMjcf(
  sceneId: string,
  onProgress?: (progress: ExportProgress) => void,
): Promise<void> {
  const bundle = await api.scenePhysics(sceneId);
  const names = Object.keys(bundle.meshes);

  // Flat, and keyed on the name as it appears in `file=`. A subdirectory would
  // need a `meshdir` that the XML does not carry.
  const files: Record<string, Uint8Array> = {
    "scene.xml": new TextEncoder().encode(bundle.mjcf),
  };

  let fetched = 0;
  onProgress?.({ fetched, total: names.length });
  await pooled(
    FETCH_CONCURRENCY,
    names.map((name) => async () => {
      const response = await fetch(storageUrl(bundle.meshes[name]));
      if (!response.ok) {
        // Named, because "export failed" on a scene with hundreds of assets
        // gives no way to tell a missing proxy from an unreachable backend.
        throw new Error(`${name} → ${response.status} ${response.statusText}`);
      }
      files[name] = new Uint8Array(await response.arrayBuffer());
      onProgress?.({ fetched: ++fetched, total: names.length });
    }),
  );

  // Narrowed on the way out rather than copied on the way in: `fflate` types its
  // result as `Uint8Array<ArrayBufferLike>` and `BlobPart` wants an `ArrayBuffer`
  // backing, a difference that only exists so `SharedArrayBuffer` cannot reach a
  // Blob. `zip` allocates its own plain buffer, so this is that buffer.
  const archive = await new Promise<Uint8Array<ArrayBuffer>>((resolve, reject) => {
    zip(files, { level: 6 }, (error, data) =>
      error ? reject(error) : resolve(data as Uint8Array<ArrayBuffer>),
    );
  });

  save(`${sceneId}-mjcf.zip`, new Blob([archive], { type: "application/zip" }));
}
