import {
  QueryClient,
  QueryClientProvider,
  keepPreviousData,
  useQuery,
} from "@tanstack/react-query";
import { useEffect } from "react";

import { api } from "./api/client";
import { CertificatePanel } from "./certify/CertificatePanel";
import { usePhysics } from "./certify/usePhysics";
import { ScenePanel } from "./scene/ScenePanel";
import { Toolbar } from "./scene/Toolbar";
import { Viewer } from "./scene/Viewer";
import { useSceneStore } from "./scene/store";

const queryClient = new QueryClient();

function SceneStudio() {
  const sceneId = useSceneStore((s) => s.sceneId);
  const setSceneId = useSceneStore((s) => s.setSceneId);

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
  const physics = usePhysics(sceneId, scene.data?.updated_at);

  // Open the first scene once the list arrives, so there is something on screen
  // without a click.
  useEffect(() => {
    if (sceneId === null && scenes.data?.length) setSceneId(scenes.data[0].id);
  }, [sceneId, scenes.data, setSceneId]);

  return (
    <div className="app">
      <header>
        <h1>SceneStudio</h1>
      </header>

      <main>
        <ScenePanel scenes={scenes.data ?? []} envelope={scene.data} />

        <div className="stage">
          <Toolbar />
          <div className="stage-view">
            {scenes.isError && (
              <div className="viewer-empty">
                cannot reach the backend — is uvicorn running on :8000?
              </div>
            )}
            {scenes.data?.length === 0 && (
              <div className="viewer-empty">
                no scenes yet — run <code>uv run python -m app.seed</code>
              </div>
            )}
            {scene.data && (
              <Viewer envelope={scene.data} poses={physics.poses} />
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
