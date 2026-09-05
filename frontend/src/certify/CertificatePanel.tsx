import { useMutation, useQueryClient } from "@tanstack/react-query";

import type { SceneEnvelope, SceneObject } from "../api/client";
import { api } from "../api/client";
import { useSceneStore } from "../scene/store";
import type { Physics } from "./usePhysics";
import {
  AXES,
  failingObjectIds,
  isCertified,
  reasonsFor,
  uncheckedAxes,
} from "./status";

interface Props {
  envelope: SceneEnvelope;
  physics: Physics;
}

/** `CATEGORIES` in the backend is a closed, snake_case vocabulary — "side_table",
 * "dining_table" — chosen for that, not for display. Formatting is presentation
 * and belongs here rather than in the schema. */
function formatCategory(category: string): string {
  return category.replaceAll("_", " ");
}

/** Human-readable names for the object list, unique within a scene.
 *
 * The category alone is not enough: a room with two cushions produces two rows
 * reading "cushion", and clicking the right one becomes guesswork. Only the
 * repeated categories get a number, so a lone sofa stays "sofa" rather than
 * "sofa 1" — a count of one is noise.
 *
 * Numbered by the graph's own order, which is stable for a given scene, so a row
 * does not renumber itself between renders or after a repair. Keyed on the raw
 * category, before formatting, so two categories that only differ by underscores
 * placement could never collide into one count — not a real case today, but the
 * formatted string is display, not identity.
 */
function objectDisplayNames(objects: SceneObject[]): Map<string, string> {
  const totals = new Map<string, number>();
  for (const object of objects) {
    totals.set(
      object.label.category,
      (totals.get(object.label.category) ?? 0) + 1,
    );
  }

  const seen = new Map<string, number>();
  const names = new Map<string, string>();
  for (const object of objects) {
    const category = object.label.category;
    const index = (seen.get(category) ?? 0) + 1;
    seen.set(category, index);
    const label = formatCategory(category);
    names.set(
      object.object_id,
      totals.get(category)! > 1 ? `${label} ${index}` : label,
    );
  }
  return names;
}

export function CertificatePanel({ envelope, physics }: Props) {
  const certificate = envelope.spec.certificate;
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const select = useSceneStore((s) => s.select);
  const queryClient = useQueryClient();

  const certified = isCertified(certificate);
  const unchecked = uncheckedAxes(certificate);
  const failing = failingObjectIds(certificate);
  const displayNames = objectDisplayNames(envelope.spec.graph.objects);
  const diagnostics = envelope.spec.diagnostics;

  const edits = useSceneStore((s) => s.edits);
  const clearEdits = useSceneStore((s) => s.clearEdits);
  const dirty = edits.size > 0;

  const commit = useMutation({
    mutationFn: () =>
      api.editScene(envelope.spec.scene_id, {
        objects: [...edits.values()],
        anchors: [],
        // Re-solve from the edit rather than merely storing it, so the objects
        // resting on what moved follow it. The server does not repair — that stays
        // a separate, deliberate action.
        resolve: true,
      }),
    onSuccess: (scene) => {
      queryClient.setQueryData(["scene", scene.spec.scene_id], scene);
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
      // Only after the server has them. Clearing on click would lose the edits if
      // the request failed, and the user would have no way to know what to redo.
      clearEdits();
    },
  });

  const repair = useMutation({
    mutationFn: () => api.repairScene(envelope.spec.scene_id),
    onSuccess: (result) => {
      // Write the response straight into the cache instead of invalidating and
      // refetching. The repair endpoint already returns the updated envelope, so the
      // refetch was a wasted round trip — and during it `scene.data` could go
      // undefined, unmounting the Viewer, the `<Canvas>`, and the one camera the whole
      // app shares. That is what made the view jump back to its default on repair.
      queryClient.setQueryData(
        ["scene", result.scene.spec.scene_id],
        result.scene,
      );
      // The list carries the per-scene certified dot, and nothing else does, so it is
      // the one query that genuinely has to be refetched. Scoped, rather than the
      // bare `invalidateQueries()` this replaces, which invalidated every query in the
      // cache including the one just written above.
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
    },
  });

  const statuses: Record<string, string> = {
    scale: certificate.scale_status,
    stability: certificate.stability_status,
    inertial: certificate.inertial_status,
    cost: certificate.cost_status,
  };

  return (
    <aside className="panel">
      <h2>Certificate</h2>

      <ul className="axes">
        {AXES.map((axis) => (
          <li key={axis} className={`axis axis--${statuses[axis]}`}>
            <span>{axis}</span>
            <span className="axis-status">
              {statuses[axis].replace("_", " ")}
            </span>
          </li>
        ))}
      </ul>

      {/* Grey, not red. An uncommitted edit does not mean the scene failed — it
          means the certificate on screen describes a scene that is no longer the
          one being shown. Confidently wrong and not-yet-checked are different
          things and the design keeps them in separate channels. */}
      <p
        className={`verdict verdict--${dirty ? "stale" : certified ? "pass" : "fail"}`}
      >
        {dirty ? "UNCOMMITTED EDITS" : certified ? "CERTIFIED" : "NOT CERTIFIED"}
      </p>
      {unchecked.length > 0 && (
        <p className="note">
          {/* Untested is not passed — an axis nobody ran cannot count toward a pass. */}
          no validator reached: {unchecked.join(", ")}
        </p>
      )}

      {certificate.cost && (
        <p className="note">
          {certificate.cost.mean_step_time_ms.toFixed(3)} ms/step at{" "}
          {certificate.cost.proxy_tier} tier, budget{" "}
          {certificate.cost.budget_ms} ms
        </p>
      )}

      {dirty && (
        <div className="commit">
          <button
            type="button"
            className="commit-button"
            onClick={() => commit.mutate()}
            disabled={commit.isPending}
          >
            {commit.isPending
              ? "committing…"
              : `commit ${edits.size} edit${edits.size === 1 ? "" : "s"}`}
          </button>
          <button type="button" className="commit-discard" onClick={clearEdits}>
            discard
          </button>
          <p className="note">
            the certificate below describes the scene before these edits
          </p>
          {commit.error && (
            <p className="note note--error">{String(commit.error)}</p>
          )}
        </div>
      )}

      {/* The same MuJoCo the backend certifies with — `@mujoco/mujoco` 3.11.0
          against the server's 3.11.0, compiling the MJCF the server built. Off
          until asked: the 9.7 MB module should not be downloaded by someone who
          only wants to look, and the solved pose *is* the answer, so running
          gravity unprompted would show the scene falling and hide it. Nothing here
          writes to the scene — Reset asks the compiled pose again. */}
      <div className="physics">
        <button
          type="button"
          className="physics-toggle"
          disabled={physics.state === "loading"}
          onClick={physics.state === "idle" ? physics.enable : physics.disable}
        >
          {physics.state === "loading"
            ? "loading MuJoCo…"
            : physics.state === "running"
              ? "◼ stop physics"
              : "▶ run physics"}
        </button>
        {physics.state === "running" && (
          <>
            <button type="button" className="physics-reset" onClick={physics.reset}>
              ↺ reset
            </button>
            <span className="note">t = {physics.time.toFixed(2)} s</span>
          </>
        )}
        {physics.error && <p className="reasons-degraded">{physics.error}</p>}
      </div>

      {/* Two flags, not one. `converged` is about the optimiser reaching a fixed
          point; `settled` is about the solved pose already being the equilibrium.
          A scene where one object topples reports converged-but-not-settled, which
          is a very different thing from a solve that failed. */}
      <p className="note">
        solve: {diagnostics.iterations} iterations,{" "}
        <span className={diagnostics.converged ? "flag-ok" : "flag-bad"}>
          {diagnostics.converged ? "converged" : "not converged"}
        </span>
        {", "}
        <span className={diagnostics.settled ? "flag-ok" : "flag-bad"}>
          {diagnostics.settled ? "settled" : "not settled"}
        </span>
        {!diagnostics.settled && diagnostics.max_settle_drift_m > 0 && (
          <> — worst object moves {(diagnostics.max_settle_drift_m * 1000).toFixed(0)} mm
          under gravity</>
        )}
      </p>

      <h3>Objects</h3>
      <ul className="objects">
        {envelope.spec.graph.objects.map((object) => {
          const display = displayNames.get(object.object_id) ?? object.object_id;
          const failed = failing.has(object.object_id);
          const selected = object.object_id === selectedObjectId;
          return (
            <li key={object.object_id}>
              <button
                type="button"
                className={`object ${failed ? "object--failed" : ""} ${
                  selected ? "object--selected" : ""
                }`}
                onClick={() => select(selected ? null : object.object_id)}
              >
                <span>
                  {display}
                  {/* A failed reconstruction renders as a plain box, which is
                      otherwise indistinguishable from a genuinely boxy object. */}
                  {object.degradation_reason && (
                    <span className="badge">box</span>
                  )}
                </span>
              </button>
              {/* The id is the filename the meshes are written under, so it is what
                  you need to go looking on disk — and nothing else. It stays out of
                  the list and appears once the row is open. */}
              {selected && (
                <p className="object-id">{object.object_id}</p>
              )}
              {selected && object.degradation_reason && (
                <ul className="reasons">
                  <li className="reasons-degraded">
                    geometry unavailable: {object.degradation_reason}
                  </li>
                </ul>
              )}
              {selected && (
                <ul className="reasons">
                  {reasonsFor(object.object_id, certificate, displayNames).map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                  {!failed && <li className="reasons-ok">passes every axis</li>}
                </ul>
              )}
            </li>
          );
        })}
      </ul>

      {/* Rendered even when there is nothing to repair, disabled rather than hidden.
          Hiding it made a passing scene look like a missing feature — the same trap
          the physics button falls into, where the success case is indistinguishable
          from the thing being broken. */}
      <button
        type="button"
        className="repair"
        onClick={() => repair.mutate()}
        disabled={repair.isPending || certified}
      >
        {repair.isPending
          ? "Repairing…"
          : certified
            ? "nothing to repair"
            : "Repair"}
      </button>

      {repair.data && (
        <div className="note">
          {repair.data.actions.length === 0
            ? "nothing to repair"
            : repair.data.actions.map((action, index) => (
                <div key={index}>
                  {action.improved ? "✓" : "✗"} {action.kind.replace(/_/g, " ")}{" "}
                  {action.target_id} by {(action.magnitude * 1000).toFixed(0)}{" "}
                  mm
                </div>
              ))}
          {!repair.data.converged && (
            <div>stopped on the round budget, not finished</div>
          )}
        </div>
      )}
      {repair.error && (
        <p className="note note--error">{String(repair.error)}</p>
      )}
    </aside>
  );
}
