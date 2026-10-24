import { useMutation, useQueryClient } from "@tanstack/react-query";

import type { SceneEnvelope, SceneObject } from "../api/client";
import { api } from "../api/client";
import { useSceneStore } from "../scene/store";
import { outcomeReason } from "../scene/outcome";
import type { Physics } from "./usePhysics";
import {
  AXES,
  axisStatuses,
  failingObjectIds,
  reasonsFor,
  supportOf,
  uncheckedAxes,
} from "./status";
import { supportChangeMessage } from "./supportChanges";
import { mergeDraftWithPhysicsEdits } from "./physicsEdits";

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

  const ready = envelope.spec.outcome === "ready";
  const unchecked = uncheckedAxes(certificate);
  const failing = failingObjectIds(certificate);
  const displayNames = objectDisplayNames(envelope.spec.graph.objects);
  const diagnostics = envelope.spec.diagnostics;
  const liveViolations = new Map(
    physics.violations.map((violation) => [violation.objectId, violation]),
  );
  const liveDisplacementLimit =
    certificate.com_displacement_tolerance_m ?? 0.01;
  const liveTiltLimit = certificate.orientation_drift_tolerance_deg ?? 2;

  const liveViolationMessage = (violation: Physics["violations"][number]) => {
    const measurements: string[] = [];
    if (violation.displacementM > liveDisplacementLimit) {
      measurements.push(
        `movement ${(violation.displacementM * 1000).toFixed(1)} mm > ` +
          `${(liveDisplacementLimit * 1000).toFixed(1)} mm`,
      );
    }
    if (violation.tiltDeg > liveTiltLimit) {
      measurements.push(
        `tilt ${violation.tiltDeg.toFixed(1)}° > ` +
          `${liveTiltLimit.toFixed(1)}°`,
      );
    }
    const name = displayNames.get(violation.objectId) ?? violation.objectId;
    return `${name} (${measurements.join(", ")})`;
  };

  const edits = useSceneStore((s) => s.edits);
  const clearEdits = useSceneStore((s) => s.clearEdits);
  const dirty = edits.size > 0;

  const commit = useMutation({
    mutationFn: () =>
      api.editScene(envelope.spec.scene_id, {
        objects: [...edits.values()],
        anchors: [],
        // Re-solve from the edit rather than merely storing it, so the objects
        // resting on what moved follow it. The server also re-derives what each
        // moved object rests on, and repairs — both bounded to the edit.
        resolve: true,
      }),
    onSuccess: (result) => {
      // Written straight into the cache rather than invalidated and refetched. The
      // endpoint already returns the updated envelope, so a refetch is a wasted
      // round trip — and during it `scene.data` could go undefined, unmounting the
      // Viewer, the `<Canvas>`, and the one camera the whole app shares. That is
      // what used to make the view jump back to its default.
      queryClient.setQueryData(
        ["scene", result.scene.spec.scene_id],
        result.scene,
      );
      // The list carries the per-scene certified dot and nothing else does, so it
      // is the one query that genuinely has to be refetched. Scoped rather than a
      // bare `invalidateQueries()`, which would invalidate the entry just written.
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
      // Only after the server has them. Clearing on click would lose the edits if
      // the request failed, and the user would have no way to know what to redo.
      clearEdits();
    },
  });

  const pausedPoseEdits =
    physics.state === "paused"
      ? mergeDraftWithPhysicsEdits(
          envelope.spec.graph.objects,
          physics.poses,
          physics.draftEdits,
        )
      : [];
  const commitPhysics = useMutation({
    mutationFn: () =>
      api.editScene(envelope.spec.scene_id, {
        objects: pausedPoseEdits,
        anchors: [],
        // Preserve exactly what the user approved in MuJoCo. The backend still
        // refreshes support edges and recertifies, but does not solve the pose back
        // toward the original photograph or run an automatic repair over it.
        resolve: false,
      }),
    onSuccess: (result) => {
      queryClient.setQueryData(
        ["scene", result.scene.spec.scene_id],
        result.scene,
      );
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
      // The preview was compiled from this exact draft and the response now
      // contains both it and the approved MuJoCo poses. Leaving the draft in the
      // store would immediately apply the same edits a second time and mark the
      // freshly certified scene stale.
      clearEdits();
      physics.disable();
    },
  });

  const recertify = useMutation({
    mutationFn: () =>
      api.editScene(envelope.spec.scene_id, {
        objects: [],
        anchors: [],
        // A no-op pose commit is the pure re-certification path: the graph and
        // anchors stay unchanged, while every certificate axis runs under the
        // current contract. This is needed when the validator itself changes and
        // an existing scene still carries an older stored certificate.
        resolve: false,
      }),
    onSuccess: (result) => {
      queryClient.setQueryData(
        ["scene", result.scene.spec.scene_id],
        result.scene,
      );
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
      physics.disable();
    },
  });

  const statuses = axisStatuses(certificate);

  return (
    <aside className="panel" aria-label="Scene certificate">
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
        className={`verdict verdict--${dirty ? "stale" : ready ? "pass" : "attention"}`}
        aria-live="polite"
      >
        {dirty ? "Uncommitted edits" : ready ? "Ready" : "Needs attention"}
      </p>
      {!dirty && !ready &&
        envelope.spec.outcome_reasons.map((reason) => (
          <p className="note note--attention" key={reason}>
            {outcomeReason(reason)}
          </p>
        ))}
      {unchecked.length > 0 && (
        <p className="note">
          {/* Untested is not passed — an axis nobody ran cannot count toward a pass. */}
          no validator reached: {unchecked.join(", ")}
        </p>
      )}
      {Object.entries(certificate.axis_errors ?? {}).map(([axis, error]) => (
        <p className="note note--error" key={axis}>
          {axis}: {error}
        </p>
      ))}

      {certificate.scene_graph_sha256 && (
        <p
          className="note"
          title={`graph ${certificate.scene_graph_sha256}\ncollision ${certificate.collision_geometry_sha256 ?? "unknown"}`}
        >
          certificate v{certificate.certificate_version} ·{" "}
          {certificate.simulator_name ?? "simulator"}{" "}
          {certificate.simulator_version ?? "unknown"} · graph{" "}
          {certificate.scene_graph_sha256.slice(0, 8)}
        </p>
      )}
      <button
        type="button"
        className="physics-reset"
        disabled={dirty || recertify.isPending}
        onClick={() => recertify.mutate()}
        title={dirty ? "commit or discard edits before rechecking" : undefined}
      >
        {recertify.isPending ? "rechecking…" : "recheck certificate"}
      </button>
      {recertify.error && (
        <p className="note note--error">{String(recertify.error)}</p>
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

      {/* What committing changed or corrected, shown outside the `dirty` block because it
          describes the commit that just finished — by which point there are no
          uncommitted edits left and that block is gone.

          Shown at all because a commit now repairs, and a repair that moved
          something without saying so is exactly the silent change this endpoint
          used to avoid by never repairing. Reverted proposals are listed too: what
          was tried and did not work is what you need when a scene will not certify.
          Support changes are also reported: they happen before repair when a drag's
          stale support parent is re-derived from its new position. */}
      {commit.data &&
        (commit.data.support_changes.length > 0 || commit.data.actions.length > 0) && (
        <div className="note">
          {commit.data.support_changes.map((change) => (
            <div key={`support:${change.object_id}`}>
              ↳ {supportChangeMessage(change, displayNames)}
            </div>
          ))}
          {commit.data.actions.map((action, index) => (
            <div key={index}>
              {action.improved ? "✓" : "✗"} {action.kind.replace(/_/g, " ")}{" "}
              {displayNames.get(action.target_id) ?? action.target_id} by{" "}
              {(action.magnitude * 1000).toFixed(0)} mm
            </div>
          ))}
        </div>
      )}
      {commitPhysics.data && (
        <div className="note">
          ✓ paused physics pose committed and recertified
          {commitPhysics.data.support_changes.map((change) => (
            <div key={`physics-support:${change.object_id}`}>
              ↳ {supportChangeMessage(change, displayNames)}
            </div>
          ))}
        </div>
      )}

      {/* The same MuJoCo the backend certifies with — `@mujoco/mujoco` 3.11.0
          against the server's 3.11.0, compiling the MJCF the server built. Off
          until asked: the 9.7 MB module should not be downloaded by someone who
          only wants to look, and the solved pose *is* the answer, so running
          gravity unprompted would show the scene falling and hide it. Stepping is
          non-destructive; only the explicit paused-pose commit writes it back. */}
      <div className="physics">
        <button
          type="button"
          className="physics-toggle"
          disabled={physics.state === "loading" || physics.state === "diverged"}
          onClick={
            physics.state === "idle"
              ? physics.enable
              : physics.state === "running"
                ? physics.pause
                : physics.state === "paused"
                  ? physics.resume
                  : physics.disable
          }
        >
          {physics.state === "loading"
            ? "loading MuJoCo…"
            : physics.state === "running"
              ? "Ⅱ pause physics"
              : physics.state === "paused"
                ? "▶ resume physics"
                : physics.state === "diverged"
                  ? "◼ stopped — unstable"
                  : dirty
                    ? "▶ test edits with physics"
                    : "▶ run physics"}
        </button>
        {physics.state === "diverged" && (
          <button type="button" className="physics-reset" onClick={physics.disable}>
            ↩ restore {physics.usingDraft ? "edited" : "stored"} pose
          </button>
        )}
        {physics.state === "running" && (
          <>
            <button type="button" className="physics-reset" onClick={physics.reset}>
              ↺ reset to {physics.usingDraft ? "edits" : "start"}
            </button>
            <button type="button" className="physics-reset" onClick={physics.disable}>
              ↩ stop &amp; restore {physics.usingDraft ? "edits" : "start"}
            </button>
            <span className="note">
              {physics.usingDraft ? "testing draft · " : ""}t ={" "}
              {physics.time.toFixed(2)} s
            </span>
          </>
        )}
        {physics.state === "paused" && (
          <>
            <button
              type="button"
              className="physics-commit"
              disabled={commitPhysics.isPending || pausedPoseEdits.length === 0}
              onClick={() => commitPhysics.mutate()}
            >
              {commitPhysics.isPending
                ? "committing…"
                : `commit paused pose (${pausedPoseEdits.length})`}
            </button>
            <button type="button" className="physics-reset" onClick={physics.reset}>
              ↺ reset to {physics.usingDraft ? "edits" : "start"}
            </button>
            <button type="button" className="physics-reset" onClick={physics.disable}>
              ↩ restore {physics.usingDraft ? "edited" : "stored"} pose
            </button>
            <span className="note">
              {physics.usingDraft ? "draft " : ""}paused at t ={" "}
              {physics.time.toFixed(2)} s
            </span>
          </>
        )}
        {physics.state === "paused" && physics.usingDraft && (
          <p className="note">
            committing saves the draft edits and these simulated poses together
          </p>
        )}
        {commitPhysics.error && (
          <p className="note note--error">{String(commitPhysics.error)}</p>
        )}
        {physics.error && <p className="reasons-degraded">{physics.error}</p>}
        {physics.violations.length > 0 && (
          <p className="note note--error">
            live physics crossed the certificate limit: {physics.violations
              .map(liveViolationMessage)
              .join(", ")}
          </p>
        )}
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

      <h3 className="list-heading">Objects</h3>
      <ul className="objects">
        {envelope.spec.graph.objects.map((object) => {
          const display = displayNames.get(object.object_id) ?? object.object_id;
          const liveViolation = liveViolations.get(object.object_id);
          const failed = failing.has(object.object_id) || liveViolation !== undefined;
          const selected = object.object_id === selectedObjectId;
          return (
            <li key={object.object_id}>
              <button
                type="button"
                className={`object ${failed ? "object--failed" : ""} ${
                  selected ? "object--selected" : ""
                }`}
                aria-pressed={selected}
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
                {/* The support relation, on every row rather than only the open one.
                    It is the thing most of the scale reasons are *about* — "base is
                    not over its support" and "floating above its support" both name
                    a parent the reader otherwise has to guess at — and it is what
                    makes a wrong one (a book resting on a cup) visible at a glance
                    instead of only in the graph. */}
                <span className="object-support">
                  on {supportOf(object, displayNames)}
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
                  {liveViolation && (
                    <li>
                      live physics reaches{" "}
                      {(liveViolation.displacementM * 100).toFixed(1)} cm and{" "}
                      {liveViolation.tiltDeg.toFixed(1)}° of tilt from its stored pose
                    </li>
                  )}
                  {!failed && <li className="reasons-ok">passes every axis</li>}
                </ul>
              )}
            </li>
          );
        })}
      </ul>

      {/* No Repair button. Committing an edit now repairs as part of the same
          action, bounded to the objects the edit touched, so a separate button
          asked the user to distinguish between two things they have no reason to
          think of as different — and left the failure case reachable only by
          knowing to press it. What repair did is reported above, next to the
          commit that caused it. The whole-scene endpoint still exists for a scene
          nobody has edited. */}
    </aside>
  );
}
