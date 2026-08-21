import { Grid, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Suspense } from 'react'

import type { SceneEnvelope } from '../api/client'
import { storageUrl } from '../api/client'
import { SceneModel } from './SceneModel'
import { useSceneStore } from './store'

interface Props {
  envelope: SceneEnvelope
}

export function Viewer({ envelope }: Props) {
  const select = useSceneStore((s) => s.select)
  const gltfPath = envelope.spec.exports?.gltf_path

  if (!gltfPath) {
    return <div className="viewer-empty">this scene has no exported geometry</div>
  }

  // updated_at busts the cache. Repair rewrites scene.glb in place at the same
  // URL, so without a version the browser serves the pre-repair copy and the fix
  // looks like it did nothing.
  const url = storageUrl(gltfPath, envelope.updated_at)

  return (
    <Canvas
      shadows
      camera={{ position: [2.4, 1.8, 2.4], fov: 45, up: [0, 1, 0] }}
      onPointerMissed={() => select(null)}
    >
      <color attach="background" args={['#15171c']} />
      <hemisphereLight intensity={0.55} groundColor="#20232a" />
      <directionalLight position={[3, 5, 2]} intensity={1.6} castShadow />

      {/* The GLB carries no floor: MJCF's is an infinite plane and a finite box
          would misrepresent its extent, so the grid is drawn here instead. It sits
          at floor_height_m, in the Y-up frame the glTF root establishes. */}
      <Grid
        args={[12, 12]}
        position={[0, envelope.spec.graph.floor_height_m ?? 0, 0]}
        cellSize={0.25}
        cellColor="#2c313a"
        sectionSize={1}
        sectionColor="#3a414d"
        fadeDistance={18}
        infiniteGrid
      />

      <Suspense fallback={null}>
        <SceneModel
          url={url}
          certificate={envelope.spec.certificate}
          graph={envelope.spec.graph}
        />
      </Suspense>

      <OrbitControls makeDefault target={[0, 0.4, 0]} />
    </Canvas>
  )
}
