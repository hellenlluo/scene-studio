import type { BalanceCheck, Certificate } from "../api/client";

/** Selected evidence wins; otherwise keep the unselected view focused on failures. */
export function balanceEvidence(
  certificate: Certificate,
  selectedObjectId: string | null,
): BalanceCheck[] {
  const checks = certificate.balance ?? [];
  if (selectedObjectId)
    return checks.filter((check) => check.object_id === selectedObjectId);
  const failures = checks.filter((check) => !check.passed);
  return failures.length ? failures : checks;
}
