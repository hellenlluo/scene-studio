import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import type { SceneEnvelope, SceneSummary } from "../api/client";
import {
  type ExportProgress,
  downloadGltf,
  downloadMjcf,
  hasGltf,
} from "./export";
import { useSceneStore } from "./store";

interface Props {
  scenes: SceneSummary[];
  envelope?: SceneEnvelope;
}

/**
 * Left sidebar: which scene is open, and the facts that describe the scene as a whole.
 *
 * Split from the certificate panel on the right by what the information is *about*.
 * This side is the scene — pick one, see where it came from and what frame it is in.
 * The right side is the verdict on it, object by object. Mixing them meant the scene
 * list sat in the header competing with the transform tools, which are a third thing
 * again: they act on the selection, not on the scene.
 *
 * Export lives here for the same reason: it acts on the whole scene, not on the
 * selection and not on the verdict.
 */
export function ScenePanel({ scenes, envelope }: Props) {
  const sceneId = useSceneStore((s) => s.sceneId);
  const setSceneId = useSceneStore((s) => s.setSceneId);
  // Read to warn, not to block. An export of the committed scene is a legitimate
  // thing to want while holding uncommitted drags — it just is not an export of
  // what is on screen, and saying so is cheaper than refusing.
  const edits = useSceneStore((s) => s.edits);

  const graph = envelope?.spec.graph;
  const offset = graph?.world_offset_m;

  // Progress is local rather than mutation state because it arrives during the
  // one `mutationFn`, which react-query has no view inside of.
  const [progress, setProgress] = useState<ExportProgress | null>(null);
  const mjcf = useMutation({
    mutationFn: () => downloadMjcf(sceneId as string, setProgress),
    onSettled: () => setProgress(null),
  });

  return (
    <aside className="panel panel--left">
      <h2>Scenes</h2>
      <ul className="scene-list">
        {scenes.map((summary) => (
          <li key={summary.id}>
            <button
              type="button"
              className={`scene-item ${summary.id === sceneId ? "scene-item--active" : ""}`}
              onClick={() => setSceneId(summary.id)}
            >
              <span className="scene-name">{summary.name}</span>
              <span
                className={`dot dot--${summary.certified ? "pass" : "fail"}`}
              />
              <span className="scene-count">{summary.object_count} obj</span>
            </button>
          </li>
        ))}
      </ul>

      {envelope && (
        <>
          <h2>Scene</h2>
          <dl className="facts">
            <dt>source</dt>
            <dd title={envelope.spec.image_path ?? undefined}>
              {envelope.spec.image_path?.split("/").pop() ?? "hand-authored"}
            </dd>
            <dt>objects</dt>
            <dd>{graph?.objects.length ?? 0}</dd>
            <dt>floor</dt>
            <dd>{((graph?.floor_height_m ?? 0) * 1000).toFixed(0)} mm</dd>
            {/* Non-zero means the scene was moved onto the origin from wherever the
                photographer was standing. Worth showing: it is the one number that
                explains why these coordinates are not the reconstruction's own. */}
            <dt>recentred</dt>
            <dd>
              {offset && (offset[0] || offset[1])
                ? `${offset[0].toFixed(2)}, ${offset[1].toFixed(2)} m`
                : "—"}
            </dd>
            <dt>updated</dt>
            <dd>{new Date(envelope.updated_at).toLocaleTimeString()}</dd>
          </dl>

          <h3>Export</h3>
          <div className="export">
            <button
              type="button"
              className="export-button"
              onClick={() => downloadGltf(envelope)}
              disabled={!hasGltf(envelope)}
              title="Binary glTF: geometry and textures in one self-contained file"
            >
              glTF
            </button>
            <button
              type="button"
              className="export-button"
              onClick={() => mjcf.mutate()}
              disabled={mjcf.isPending}
              title="MuJoCo XML plus every collision mesh it names, zipped"
            >
              {mjcf.isPending
                ? progress && progress.total > 0
                  ? `meshes ${progress.fetched}/${progress.total}…`
                  : "packing…"
                : "MJCF"}
            </button>
          </div>
          {/* Said once, next to the buttons, because the zip is not the single
              file the label implies and a user who unzips it into a folder of
              loose meshes should have been told to expect that. */}
          <p className="note">
            glTF is one <code>.glb</code>; MJCF is a <code>.zip</code> of{" "}
            <code>scene.xml</code> and its meshes, which it names by bare
            filename and so cannot load apart from.
          </p>
          {/* Both artefacts are pure functions of the *stored* graph. An
              uncommitted drag has never been near the server, so it is not in
              either one — and a download that silently disagreed with the
              viewport is the kind of thing found out much later. */}
          {edits.size > 0 && (
            <p className="note">
              exports describe the committed scene — {edits.size} uncommitted
              edit{edits.size === 1 ? "" : "s"} not included
            </p>
          )}
          {mjcf.error && (
            <p className="note note--error">{String(mjcf.error)}</p>
          )}
        </>
      )}
    </aside>
  );
}
