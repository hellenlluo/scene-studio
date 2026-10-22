import { describe, expect, it } from "vitest";

import type { BalanceCheck, Certificate } from "../api/client";
import { balanceEvidence } from "./balanceEvidence";

function check(objectId: string, passed: boolean): BalanceCheck {
  return {
    object_id: objectId,
    centre_of_mass_xy_m: [0, 0],
    centre_of_mass_m: [0, 0, 1],
    contact_count: 4,
    contact_points_m: [],
    support_polygon_xy_m: [],
    evidence_source: "collision_proxy",
    contact_force_feasible: passed,
    support_margin_m: passed ? 0.1 : -0.1,
    passed,
  };
}

function certificate(balance: BalanceCheck[]): Certificate {
  return { balance } as Certificate;
}

describe("balanceEvidence", () => {
  it("isolates the selected object", () => {
    const checks = [check("sofa", false), check("table", true)];
    expect(balanceEvidence(certificate(checks), "table")).toEqual([checks[1]]);
  });

  it("shows failures when no object is selected", () => {
    const checks = [check("sofa", false), check("table", true)];
    expect(balanceEvidence(certificate(checks), null)).toEqual([checks[0]]);
  });

  it("shows all evidence when the scene passes", () => {
    const checks = [check("sofa", true), check("table", true)];
    expect(balanceEvidence(certificate(checks), null)).toEqual(checks);
  });
});
