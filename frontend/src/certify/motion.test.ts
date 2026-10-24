import { describe, expect, it } from 'vitest'

import { quaternionTiltDeg } from './motion'

describe('quaternionTiltDeg', () => {
  it('ignores a quarter turn around world-up', () => {
    expect(quaternionTiltDeg([1, 0, 0, 0], [Math.SQRT1_2, 0, 0, Math.SQRT1_2])).toBeCloseTo(0)
  })

  it('reports a quarter turn around a horizontal axis as a tip', () => {
    expect(quaternionTiltDeg([1, 0, 0, 0], [Math.SQRT1_2, Math.SQRT1_2, 0, 0])).toBeCloseTo(90)
  })

  it('treats opposite quaternion signs as the same pose', () => {
    expect(quaternionTiltDeg([0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5])).toBeCloseTo(0)
  })
})
