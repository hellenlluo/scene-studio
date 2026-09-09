/**
 * These mirror backend semantics that do not serialise, so they exist to catch
 * the two copies drifting apart. The assertions deliberately match the ones in
 * `backend/tests/test_schemas.py`.
 */
import { describe, expect, it } from 'vitest'

import type { Certificate, ScaleCheck, StabilityCheck } from '../api/client'
import {
  failingObjectIds,
  isCertified,
  nodeToObjectId,
  reasonsFor,
  sanitizeNodeName,
  statusFor,
  supportOf,
  uncheckedAxes,
} from './status'

function certificate(overrides: Partial<Certificate> = {}): Certificate {
  return {
    scale_status: 'pass',
    stability_status: 'pass',
    inertial_status: 'pass',
    cost_status: 'pass',
    scale: [],
    stability: [],
    inertial: [],
    cost: null,
    penetration_tolerance_m: 0.002,
    support_gap_tolerance_m: 0.005,
    prior_deviation_tolerance_sigma: 3,
    ...overrides,
  } as Certificate
}

function scaleCheck(objectId: string, overrides: Partial<ScaleCheck> = {}): ScaleCheck {
  return {
    object_id: objectId,
    // A fixed 3-tuple, not number[] — the generated types carry Vec3 exactly as
    // the backend declares it.
    deviation_sigma: [0, 0, 0],
    support_gap_m: 0,
    touching_parent: true,
    base_inside_parent: true,
    passed: true,
    ...overrides,
  }
}

function stabilityCheck(
  objectId: string,
  overrides: Partial<StabilityCheck> = {},
): StabilityCheck {
  return {
    object_id: objectId,
    com_displacement_m: 0,
    orientation_drift_deg: 0,
    initial_penetration_m: 0,
    passed: true,
    ...overrides,
  }
}

describe('isCertified', () => {
  it('passes when every axis was reached and none failed', () => {
    expect(isCertified(certificate())).toBe(true)
  })

  it('fails on any failing axis', () => {
    expect(isCertified(certificate({ stability_status: 'fail' }))).toBe(false)
  })

  it('fails when an axis was never run, even if everything else passed', () => {
    // Without this a scene certifies on the strength of whichever validators
    // happen to be wired up.
    expect(isCertified(certificate({ scale_status: 'not_run' }))).toBe(false)
  })

  it('fails when nothing was checked at all', () => {
    const untested = certificate({
      scale_status: 'not_run',
      stability_status: 'not_run',
      inertial_status: 'not_run',
      cost_status: 'not_run',
    })
    expect(isCertified(untested)).toBe(false)
  })

  it('tolerates a not_applicable axis', () => {
    // Nothing of that kind to check is not the same as not having looked.
    expect(isCertified(certificate({ inertial_status: 'not_applicable' }))).toBe(true)
  })
})

describe('uncheckedAxes', () => {
  it('names the gaps', () => {
    expect(uncheckedAxes(certificate({ scale_status: 'not_run' }))).toEqual(['scale'])
  })

  it('is empty for a fully checked scene', () => {
    expect(uncheckedAxes(certificate())).toEqual([])
  })
})

describe('failingObjectIds', () => {
  it('gathers across every axis', () => {
    const cert = certificate({
      scale: [scaleCheck('mug', { passed: false }), scaleCheck('table')],
      stability: [stabilityCheck('cabinet', { passed: false })],
    })
    expect(failingObjectIds(cert)).toEqual(new Set(['mug', 'cabinet']))
  })

  it('is empty for a sound scene', () => {
    expect(failingObjectIds(certificate({ scale: [scaleCheck('mug')] })).size).toBe(0)
  })

  it('reports both bodies of a symmetric overlap', () => {
    // Stability measures penetration symmetrically, so a mug buried in a table
    // genuinely fails both. Honest, and the panel groups by cause so it reads as
    // one problem.
    const cert = certificate({
      stability: [
        stabilityCheck('mug', { initial_penetration_m: 0.05, passed: false }),
        stabilityCheck('table', { initial_penetration_m: 0.05, passed: false }),
      ],
    })
    expect(failingObjectIds(cert)).toEqual(new Set(['mug', 'table']))
  })
})

describe('statusFor', () => {
  it('marks only the objects that failed', () => {
    const cert = certificate({ scale: [scaleCheck('mug', { passed: false })] })
    expect(statusFor('mug', cert)).toBe('failed')
    expect(statusFor('table', cert)).toBe('ok')
  })
})

describe('reasonsFor', () => {
  it('does not report a gap that is inside the tolerance', () => {
    // The regression this exists for. An object over-hanging its support fails on
    // containment while its height is fine, and every scale reason used to be
    // emitted on `|gap| > 0`. Measured on room2: seven of nine failures had gaps
    // under 3 mm against a 5 mm bar and were all reported as floating.
    const cert = certificate({
      scale: [scaleCheck('mug', { support_gap_m: 0.0009, base_inside_parent: false, passed: false })],
    })
    const reasons = reasonsFor('mug', cert)

    expect(reasons).toEqual(['base is not over its support'])
    expect(reasons.some((r) => r.includes('floating'))).toBe(false)
  })

  it('quotes the gap tolerance, not the penetration tolerance', () => {
    // These are different numbers — 5 mm and 2 mm — and reaching for the wrong one
    // reported a measurement as failing against a bar it passed.
    const cert = certificate({
      scale: [scaleCheck('mug', { support_gap_m: 0.05, passed: false })],
    })

    expect(reasonsFor('mug', cert)[0]).toContain('tolerance ±5.0 mm')
  })

  it('takes the sigma tolerance from the certificate rather than hardcoding it', () => {
    const loose = certificate({
      prior_deviation_tolerance_sigma: 6,
      scale: [scaleCheck('mug', { deviation_sigma: [4, 0, 0], passed: false })],
    })
    expect(reasonsFor('mug', loose).some((r) => r.includes('σ'))).toBe(false)

    const tight = certificate({
      prior_deviation_tolerance_sigma: 2,
      scale: [scaleCheck('mug', { deviation_sigma: [4, 0, 0], passed: false })],
    })
    expect(reasonsFor('mug', tight)[0]).toBe('4.0σ from its class prior (tolerance 2σ)')
  })

  it('distinguishes floating from sunk by the sign of the gap', () => {
    const sunk = certificate({
      scale: [scaleCheck('mug', { support_gap_m: -0.05, passed: false })],
    })
    expect(reasonsFor('mug', sunk)[0]).toContain('sunk into')
    expect(reasonsFor('mug', sunk)[0]).toContain('50 mm')

    const floating = certificate({
      scale: [scaleCheck('mug', { support_gap_m: 0.25, passed: false })],
    })
    expect(reasonsFor('mug', floating)[0]).toContain('floating above')
  })

  it('reports measurements in physical units, not verdicts', () => {
    const cert = certificate({
      stability: [
        stabilityCheck('mug', {
          initial_penetration_m: 0.05,
          com_displacement_m: 0.0498,
          passed: false,
        }),
      ],
    })
    const reasons = reasonsFor('mug', cert)
    expect(reasons.some((r) => r.includes('overlaps another object by 50 mm'))).toBe(true)
    expect(reasons.some((r) => r.includes('5.0 cm'))).toBe(true)
    // The tolerance rides along, so a number can be judged without looking it up.
    expect(reasons.some((r) => r.includes('tolerance'))).toBe(true)
  })

  it('only reports the criterion that actually failed', () => {
    // A lamp that topples has huge displacement and micrometres of contact noise.
    // Testing penetration against zero reported both, and rounded the second to
    // "overlaps another object by 0 mm" — a reason that sends you looking for a
    // collision that is not there.
    const cert = certificate({
      stability: [
        stabilityCheck('lamp', {
          initial_penetration_m: 4.7e-6,
          com_displacement_m: 0.295,
          orientation_drift_deg: 26.2,
          passed: false,
        }),
      ],
    })
    const reasons = reasonsFor('lamp', cert)

    expect(reasons.some((r) => r.includes('overlaps'))).toBe(false)
    expect(reasons.some((r) => r.includes('moves 29.5 cm'))).toBe(true)
    expect(reasons.some((r) => r.includes('tips 26.2'))).toBe(true)
  })

  it('names the floor rather than calling it another object', () => {
    const cert = certificate({
      stability: [
        stabilityCheck('lamp', {
          initial_penetration_m: 0.03,
          penetration_against: 'floor',
          passed: false,
        }),
      ],
    })
    const reasons = reasonsFor('lamp', cert)
    expect(reasons.some((r) => r.includes('sinks into the floor by 30 mm'))).toBe(true)
    expect(reasons.some((r) => r.includes('another object'))).toBe(false)
  })

  it('names the neighbour it overlaps, using its display name', () => {
    const cert = certificate({
      stability: [
        stabilityCheck('mug', {
          initial_penetration_m: 0.05,
          penetration_against: 'obj_548_711',
          passed: false,
        }),
      ],
    })
    const names = new Map([['obj_548_711', 'side table']])
    expect(reasonsFor('mug', cert, names).some((r) => r.includes('overlaps side table'))).toBe(
      true,
    )
  })

  it('does not round a sub-millimetre measurement away', () => {
    const cert = certificate({
      stability: [
        stabilityCheck('mug', { initial_penetration_m: 0.0047, passed: false }),
      ],
    })
    expect(reasonsFor('mug', cert).some((r) => r.includes('4.7 mm'))).toBe(true)
  })

  it('says nothing about an object that passed', () => {
    expect(reasonsFor('mug', certificate({ scale: [scaleCheck('mug')] }))).toEqual([])
  })
})

describe('sanitizeNodeName', () => {
  it('strips the separator, exactly as GLTFLoader does', () => {
    // Observed in the running app: the backend writes `table/top` into the GLB
    // and three.js hands back `tabletop`. Splitting on `/` therefore cannot
    // recover the object id — by the time we see the name it is gone.
    expect(sanitizeNodeName('table/top')).toBe('tabletop')
    expect(sanitizeNodeName('mug/body')).toBe('mugbody')
  })

  it('leaves underscores alone and replaces whitespace', () => {
    expect(sanitizeNodeName('drawer_top/panel')).toBe('drawer_toppanel')
    expect(sanitizeNodeName('side table/top')).toBe('side_tabletop')
  })
})

describe('nodeToObjectId', () => {
  const graph = {
    objects: [
      { object_id: 'table', parts: [{ part_id: 'top' }] },
      { object_id: 'mug', parts: [{ part_id: 'body' }] },
      { object_id: 'lamp', parts: [{ part_id: 'base' }, { part_id: 'shade' }] },
    ],
  } as unknown as Parameters<typeof nodeToObjectId>[0]

  it('maps sanitised node names back to their object', () => {
    const map = nodeToObjectId(graph)
    expect(map.get('tabletop')).toBe('table')
    expect(map.get('mugbody')).toBe('mug')
  })

  it('maps every part of a multi-part object to the same owner', () => {
    const map = nodeToObjectId(graph)
    expect(map.get('lampbase')).toBe('lamp')
    expect(map.get('lampshade')).toBe('lamp')
  })

  it('misses rather than guesses for an unknown node', () => {
    // A miss shows up as an untinted object, which is visible. Prefix-matching
    // would instead colour the wrong one, which is not.
    expect(nodeToObjectId(graph).get('somethingelse')).toBeUndefined()
  })
})


describe('supportOf', () => {
  it('names the floor rather than reporting nothing', () => {
    // `supported_by: null` is the most common answer in any scene, not missing data.
    expect(supportOf({ supported_by: null })).toBe('the floor')
  })

  it('uses the display name of the parent', () => {
    const names = new Map([['obj_1', 'side table']])
    expect(supportOf({ supported_by: 'obj_1' }, names)).toBe('side table')
  })

  it('falls back to the raw id when the parent is not in the graph', () => {
    // A dangling parent is a reconcile bug, and `certify.scale` reports the object
    // as a floor contact instead — so printing the id here is the only place it
    // becomes findable.
    expect(supportOf({ supported_by: 'obj_ghost' }, new Map())).toBe('obj_ghost')
  })
})
