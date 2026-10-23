import type { SupportChange } from "../api/client";

function supportName(
  supportId: string | null,
  displayNames: ReadonlyMap<string, string>,
): string {
  return supportId === null
    ? "floor"
    : (displayNames.get(supportId) ?? supportId);
}

/** Present a changed graph edge without exposing internal IDs when a label exists. */
export function supportChangeMessage(
  change: SupportChange,
  displayNames: ReadonlyMap<string, string>,
): string {
  const objectName = displayNames.get(change.object_id) ?? change.object_id;
  const previous = supportName(change.previous_support_id, displayNames);
  const next = supportName(change.new_support_id, displayNames);
  return `${objectName} support changed: ${previous} → ${next}`;
}
