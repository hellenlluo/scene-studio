import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'

import { api } from './api/client'
import { CertificatePanel } from './certify/CertificatePanel'
import { Viewer } from './scene/Viewer'
import { useSceneStore } from './scene/store'

const queryClient = new QueryClient()

function SceneStudio() {
  const sceneId = useSceneStore((s) => s.sceneId)
  const setSceneId = useSceneStore((s) => s.setSceneId)

  const scenes = useQuery({ queryKey: ['scenes'], queryFn: api.listScenes })
  const scene = useQuery({
    queryKey: ['scene', sceneId],
    queryFn: () => api.getScene(sceneId as string),
    enabled: sceneId !== null,
  })

  // Open the first scene once the list arrives, so there is something on screen
  // without a click.
  useEffect(() => {
    if (sceneId === null && scenes.data?.length) setSceneId(scenes.data[0].id)
  }, [sceneId, scenes.data, setSceneId])

  return (
    <div className="app">
      <header>
        <h1>SceneStudio</h1>
        <nav>
          {scenes.data?.map((summary) => (
            <button
              key={summary.id}
              type="button"
              className={`tab ${summary.id === sceneId ? 'tab--active' : ''}`}
              onClick={() => setSceneId(summary.id)}
            >
              {summary.name}
              <span className={`dot dot--${summary.certified ? 'pass' : 'fail'}`} />
            </button>
          ))}
        </nav>
      </header>

      <main>
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
          <>
            <Viewer envelope={scene.data} />
            <CertificatePanel envelope={scene.data} />
          </>
        )}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SceneStudio />
    </QueryClientProvider>
  )
}
