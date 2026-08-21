import { useMutation, useQueryClient } from '@tanstack/react-query'

import type { SceneEnvelope } from '../api/client'
import { api } from '../api/client'
import { useSceneStore } from '../scene/store'
import { AXES, failingObjectIds, isCertified, reasonsFor, uncheckedAxes } from './status'

interface Props {
  envelope: SceneEnvelope
}

export function CertificatePanel({ envelope }: Props) {
  const certificate = envelope.spec.certificate
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId)
  const select = useSceneStore((s) => s.select)
  const queryClient = useQueryClient()

  const certified = isCertified(certificate)
  const unchecked = uncheckedAxes(certificate)
  const failing = failingObjectIds(certificate)

  const repair = useMutation({
    mutationFn: () => api.repairScene(envelope.spec.scene_id),
    onSuccess: () => queryClient.invalidateQueries(),
  })

  const statuses: Record<string, string> = {
    scale: certificate.scale_status,
    stability: certificate.stability_status,
    inertial: certificate.inertial_status,
    cost: certificate.cost_status,
  }

  return (
    <aside className="panel">
      <h2>Certificate</h2>

      <ul className="axes">
        {AXES.map((axis) => (
          <li key={axis} className={`axis axis--${statuses[axis]}`}>
            <span>{axis}</span>
            <span className="axis-status">{statuses[axis].replace('_', ' ')}</span>
          </li>
        ))}
      </ul>

      <p className={`verdict verdict--${certified ? 'pass' : 'fail'}`}>
        {certified ? 'CERTIFIED' : 'NOT CERTIFIED'}
      </p>
      {unchecked.length > 0 && (
        <p className="note">
          {/* Untested is not passed — an axis nobody ran cannot count toward a pass. */}
          no validator reached: {unchecked.join(', ')}
        </p>
      )}

      {certificate.cost && (
        <p className="note">
          {certificate.cost.mean_step_time_ms.toFixed(3)} ms/step at{' '}
          {certificate.cost.proxy_tier} tier, budget {certificate.cost.budget_ms} ms
        </p>
      )}

      <h3>Objects</h3>
      <ul className="objects">
        {envelope.spec.graph.objects.map((object) => {
          const failed = failing.has(object.object_id)
          const selected = object.object_id === selectedObjectId
          return (
            <li key={object.object_id}>
              <button
                type="button"
                className={`object ${failed ? 'object--failed' : ''} ${
                  selected ? 'object--selected' : ''
                }`}
                onClick={() => select(selected ? null : object.object_id)}
              >
                <span>{object.object_id}</span>
                <span className="object-category">{object.label.category}</span>
              </button>
              {selected && (
                <ul className="reasons">
                  {reasonsFor(object.object_id, certificate).map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                  {!failed && <li className="reasons-ok">passes every axis</li>}
                </ul>
              )}
            </li>
          )
        })}
      </ul>

      {!certified && (
        <button
          type="button"
          className="repair"
          onClick={() => repair.mutate()}
          disabled={repair.isPending}
        >
          {repair.isPending ? 'Repairing…' : 'Repair'}
        </button>
      )}

      {repair.data && (
        <div className="note">
          {repair.data.actions.length === 0
            ? 'nothing to repair'
            : repair.data.actions.map((action, index) => (
                <div key={index}>
                  {action.improved ? '✓' : '✗'} {action.kind.replace(/_/g, ' ')}{' '}
                  {action.target_id} by {(action.magnitude * 1000).toFixed(0)} mm
                </div>
              ))}
          {!repair.data.converged && <div>stopped on the round budget, not finished</div>}
        </div>
      )}
      {repair.error && <p className="note note--error">{String(repair.error)}</p>}
    </aside>
  )
}
