import {
  QueryClient,
  QueryClientProvider,
  keepPreviousData,
  useQuery,
} from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "./api/client";
import { CertificatePanel } from "./certify/CertificatePanel";
import { usePhysics } from "./certify/usePhysics";
import { ScenePanel } from "./scene/ScenePanel";
import { Toolbar } from "./scene/Toolbar";
import { Viewer } from "./scene/Viewer";
import { useSceneStore } from "./scene/store";
import { useTheme } from "./theme";

const queryClient = new QueryClient();

type WorkspaceView = "scene" | "browser" | "certificate";

function SceneStudio() {
  const sceneId = useSceneStore((s) => s.sceneId);
  const setSceneId = useSceneStore((s) => s.setSceneId);
  const edits = useSceneStore((s) => s.edits);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("scene");
  const { theme, toggleTheme } = useTheme();

  const scenes = useQuery({ queryKey: ["scenes"], queryFn: api.listScenes });
  const scene = useQuery({
    queryKey: ["scene", sceneId],
    queryFn: () => api.getScene(sceneId as string),
    enabled: sceneId !== null,
    // Hold the previous scene on screen while the next one loads, so `scene.data` is
    // never briefly undefined. Without this the `scene.data &&` guard below unmounts
    // the Viewer on every tab click, and with it the `<Canvas>` and the one camera the
    // whole app shares — so the user's orbit was thrown away on every scene switch.
    // Keeping the tree mounted also removes a flash of empty panel.
    placeholderData: keepPreviousData,
  });

  // Lifted here because the button and the poses land in different subtrees: the
  // control is in the certificate panel, next to Repair, and the poses drive meshes
  // inside the Viewer. Keyed on `updated_at` as well as the id, so a repair — which
  // rewrites the graph — tears the worker down rather than stepping a stale model.
  const physics = usePhysics(
    sceneId,
    scene.data?.updated_at,
    scene.data?.spec.certificate.com_displacement_tolerance_m,
    scene.data?.spec.certificate.orientation_drift_tolerance_deg,
    edits,
  );

  // Open the first scene once the list arrives, so there is something on screen
  // without a click.
  useEffect(() => {
    if (sceneId === null && scenes.data?.length) setSceneId(scenes.data[0].id);
  }, [sceneId, scenes.data, setSceneId]);

  // The compact layout behaves like three adjacent pages. Escape is the quick,
  // keyboard-accessible route back to the canvas from either full-screen panel.
  useEffect(() => {
    if (workspaceView === "scene") return;
    const returnToScene = (event: KeyboardEvent) => {
      if (event.key === "Escape") setWorkspaceView("scene");
    };
    window.addEventListener("keydown", returnToScene);
    return () => window.removeEventListener("keydown", returnToScene);
  }, [workspaceView]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="compact-header-slot compact-header-slot--left">
          {workspaceView === "scene" && (
            <button
              type="button"
              className="compact-view-button"
              onClick={() => setWorkspaceView("browser")}
            >
              Scenes
            </button>
          )}
          {workspaceView === "certificate" && (
            <button
              type="button"
              className="compact-view-button"
              onClick={() => setWorkspaceView("scene")}
            >
              ← 3D scene
            </button>
          )}
        </div>
        <div className="app-brand">
          <h1>SceneStudio</h1>
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            {theme === "light" ? (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M20.2 15.2A8.5 8.5 0 0 1 8.8 3.8 8.5 8.5 0 1 0 20.2 15.2Z" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="3.5" />
                <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4" />
              </svg>
            )}
          </button>
        </div>
        <div className="compact-header-slot compact-header-slot--right">
          {workspaceView === "scene" && (
            <button
              type="button"
              className="compact-view-button"
              onClick={() => setWorkspaceView("certificate")}
              disabled={!scene.data}
            >
              Certificate
            </button>
          )}
          {workspaceView === "browser" && (
            <button
              type="button"
              className="compact-view-button"
              onClick={() => setWorkspaceView("scene")}
            >
              3D scene →
            </button>
          )}
        </div>
      </header>

      <main className="workspace" data-compact-view={workspaceView}>
        <ScenePanel
          scenes={scenes.data ?? []}
          envelope={scene.data}
          onSceneSelected={() => setWorkspaceView("scene")}
        />

        <div className="stage">
          <Toolbar />
          <div className="stage-view">
            {scenes.isError && (
              <div className="viewer-empty" role="alert">
                Cannot reach the backend — is uvicorn running on :8000?
              </div>
            )}
            {scenes.data?.length === 0 && (
              <div className="viewer-empty" role="status">
                No scenes yet — upload a photo from the Scenes panel
              </div>
            )}
            {scene.data && (
              <Viewer
                envelope={scene.data}
                poses={physics.poses}
                physicsActive={physics.state !== "idle"}
                theme={theme}
              />
            )}
          </div>
        </div>

        {scene.data && (
          <CertificatePanel envelope={scene.data} physics={physics} />
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SceneStudio />
    </QueryClientProvider>
  );
}
