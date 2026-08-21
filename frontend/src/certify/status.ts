/**
 * Certificate semantics, reimplemented against the wire format.
 *
 * `Certificate.passed`, `.axes` and `.failing_object_ids()` are Python
 * properties on the backend and do **not** serialise. What arrives is the four
 * `*_status` fields plus the per-check `passed` flags, so the rules have to exist
 * here too.
 *
 * That duplication is the reason this file has tests. The two copies silently
 * disagreeing is precisely the bug that would make the viewer's red channel lie —
 * showing an object as fine when the backend has failed it, or the reverse.
 */
import type { AxisStatus, Certificate, SceneSpec } from '../api/client'

export const AXES = ['scale', 'stability', 'inertial', 'cost'] as const
export type AxisName = (typeof AXES)[number]

/** What a single object's geometry should be drawn as. */
export type ObjectStatus = 'failed' | 'ok'

export function axisStatuses(certificate: Certificate): Record<AxisName, AxisStatus> {
  return {
    scale: certificate.scale_status,
    stability: certificate.stability_status,
    inertial: certificate.inertial_status,
    cost: certificate.cost_status,
  }
}

/**
 * Every axis was reached, none failed, and at least one had work to do.
 *
 * All three clauses matter, and the `not_run` one most: without it a scene
 * certifies on the strength of whichever validators happen to be wired up, which
 * is the dishonesty the per-axis contract exists to prevent. Mirrors
 * `Certificate.passed`.
 */
export function isCertified(certificate: Certificate): boolean {
  const statuses = Object.values(axisStatuses(certificate))
  return (
    statuses.every((status) => status !== 'fail') &&
    statuses.every((status) => status !== 'not_run') &&
    statuses.some((status) => status === 'pass')
  )
}

/** Axes no validator reached. Non-empty means `isCertified` cannot be true. */
export function uncheckedAxes(certificate: Certificate): AxisName[] {
  return AXES.filter((axis) => axisStatuses(certificate)[axis] === 'not_run')
}

/**
 * Objects with at least one failing check, across every axis.
 *
 * Mirrors `Certificate.failing_object_ids`. Note that stability measures overlap
 * *symmetrically*, so a mug buried in a table appears here twice over — both
 * bodies genuinely overlap. The panel groups by cause so that reads as one
 * problem rather than two.
 */
export function failingObjectIds(certificate: Certificate): Set<string> {
  const failing = new Set<string>()
  for (const check of certificate.scale ?? []) if (!check.passed) failing.add(check.object_id)
  for (const check of certificate.stability ?? []) if (!check.passed) failing.add(check.object_id)
  for (const check of certificate.inertial ?? []) if (!check.passed) failing.add(check.object_id)
  return failing
}

export function statusFor(objectId: string, certificate: Certificate): ObjectStatus {
  return failingObjectIds(certificate).has(objectId) ? 'failed' : 'ok'
}

/** Human-readable reasons one object failed, in physical units. */
export function reasonsFor(objectId: string, certificate: Certificate): string[] {
  const mm = (metres: number) => `${(metres * 1000).toFixed(0)} mm`
  const tolerance = mm(certificate.penetration_tolerance_m ?? 0.002)
  const reasons: string[] = []

  for (const check of certificate.scale ?? []) {
    if (check.object_id !== objectId || check.passed) continue
    if (Math.abs(check.support_gap_m ?? 0) > 0) {
      const gap = check.support_gap_m ?? 0
      // The sign is the diagnosis: floating and buried are different errors.
      const direction = gap > 0 ? 'floating above' : 'sunk into'
      reasons.push(`${direction} its support by ${mm(Math.abs(gap))} (tolerance ±${tolerance})`)
    }
    const worst = Math.max(...check.deviation_sigma.map(Math.abs))
    if (worst > 3) reasons.push(`${worst.toFixed(1)}σ from its class prior`)
    if (!check.base_inside_parent) reasons.push('base is not over its support')
  }

  for (const check of certificate.stability ?? []) {
    if (check.object_id !== objectId || check.passed) continue
    if (check.initial_penetration_m > 0) {
      reasons.push(`overlaps another object by ${mm(check.initial_penetration_m)} at rest`)
    }
    if (check.com_displacement_m > 0.001) {
      reasons.push(`moves ${(check.com_displacement_m * 100).toFixed(1)} cm while settling`)
    }
    if (check.orientation_drift_deg > 0.1) {
      reasons.push(`tips ${check.orientation_drift_deg.toFixed(1)}° while settling`)
    }
  }

  for (const check of certificate.inertial ?? []) {
    if (check.object_id !== objectId || check.passed) continue
    if (!check.mass_density_volume_consistent) reasons.push('mass disagrees with density × volume')
    if (!check.positive_definite) reasons.push('inertia tensor is not positive definite')
    if (!check.triangle_inequality) reasons.push('inertia violates the triangle inequality')
  }

  return reasons
}

/**
 * three.js's node-name sanitiser, reproduced.
 *
 * `GLTFLoader` runs every node name through `PropertyBinding.sanitizeNodeName`,
 * which strips the characters reserved for animation paths — including `/`. The
 * backend names nodes `{object_id}/{part_id}` via `mjcf.body_name`, so what
 * actually reaches the scene graph is `tabletop`, not `table/top`. Splitting on
 * the separator therefore cannot work: by the time we see the name, it is gone.
 */
export function sanitizeNodeName(name: string): string {
  return name.replace(/\s/g, '_').replace(/[[\]./:]/g, '')
}

/**
 * Sanitised glTF node name to the object it belongs to.
 *
 * Built forward from the graph rather than parsed backward out of the name.
 * Parsing is impossible after sanitisation, and prefix-matching would break the
 * moment one object id is a prefix of another ("mug" and "mug_2"). Constructing
 * the map from the same `{object_id}/{part_id}` rule the exporter used is exact,
 * and a miss shows up as an unmatched node rather than a wrong colour.
 */
export function nodeToObjectId(graph: SceneSpec['graph']): Map<string, string> {
  const map = new Map<string, string>()
  for (const object of graph.objects) {
    for (const part of object.parts) {
      map.set(sanitizeNodeName(`${object.object_id}/${part.part_id}`), object.object_id)
    }
  }
  return map
}
