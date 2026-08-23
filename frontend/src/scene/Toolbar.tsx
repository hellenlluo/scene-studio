import { useEffect } from "react";

import { type TransformMode, useSceneStore } from "./store";

const MODES: { mode: TransformMode; label: string; key: string }[] = [
  { mode: "translate", label: "Move", key: "w" },
  { mode: "rotate", label: "Rotate", key: "e" },
  { mode: "scale", label: "Scale", key: "r" },
];

/**
 * Mode switch for the transform gizmo, with the conventional W/E/R shortcuts.
 *
 * The shortcut listener sits here rather than in the viewer because the mode is a
 * property of the session, not of the canvas: W should work with the pointer over the
 * certificate panel, and it should keep working when no object is selected — picking
 * the tool before picking the object is the normal order.
 */
export function Toolbar() {
  const mode = useSceneStore((s) => s.mode);
  const setMode = useSceneStore((s) => s.setMode);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Modifiers are someone else's shortcuts, and a bare letter belongs to whatever
      // field has focus.
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable) return;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

      const match = MODES.find((m) => m.key === event.key.toLowerCase());
      if (match) setMode(match.mode);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setMode]);

  return (
    <div className="toolbar" role="group" aria-label="Transform mode">
      {MODES.map((m) => (
        <button
          key={m.mode}
          type="button"
          className={`tool ${m.mode === mode ? "tool--active" : ""}`}
          aria-pressed={m.mode === mode}
          onClick={() => setMode(m.mode)}
          title={`${m.label} (${m.key.toUpperCase()})`}
        >
          {m.label}
          <span className="tool-key">{m.key.toUpperCase()}</span>
        </button>
      ))}
    </div>
  );
}
