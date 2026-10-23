export function outcomeReason(reason: string): string {
  switch (reason) {
    case "solver_not_converged":
      return "the optimiser did not reach a fixed point";
    case "scene_not_settled":
      return "the stored pose still moves under gravity";
    case "certification_failed":
      return "one or more certification axes failed";
    default:
      return reason.replace(/_/g, " ");
  }
}
