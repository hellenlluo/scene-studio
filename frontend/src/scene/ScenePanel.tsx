import type { SceneEnvelope, SceneSummary } from "../api/client";
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
 */
export function ScenePanel({ scenes, envelope }: Props) {
  const sceneId = useSceneStore((s) => s.sceneId);
  const setSceneId = useSceneStore((s) => s.setSceneId);

  const graph = envelope?.spec.graph;
  const offset = graph?.world_offset_m;

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
        </>
      )}
    </aside>
  );
}
