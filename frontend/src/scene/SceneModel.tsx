import { useGLTF } from '@react-three/drei'
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'

import type { Certificate, SceneSpec } from '../api/client'
import { failingObjectIds, nodeToObjectId } from '../certify/status'
import { useSceneStore } from './store'

const COLOURS = {
  // Red is a *certified failure* — measured and objective. Amber (v2
  // uncertainty) and grey (uncommitted edits) are deliberately different
  // channels: an object can be confidently wrong or uncertainly fine.
  failed: new THREE.Color('#d94b4b'),
  ok: new THREE.Color('#b9bec7'),
  selected: new THREE.Color('#4a9eff'),
}

interface Props {
  url: string
  certificate: Certificate
  graph: SceneSpec['graph']
}

export function SceneModel({ url, certificate, graph }: Props) {
  const { scene } = useGLTF(url)
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId)
  const select = useSceneStore((s) => s.select)

  const failing = useMemo(() => failingObjectIds(certificate), [certificate])
  // GLTFLoader strips `/` from node names, so the object id cannot be recovered
  // by splitting. Build the mapping forward from the graph instead.
  const nodeOwner = useMemo(() => nodeToObjectId(graph), [graph])

  // useGLTF caches by URL, so the loaded scene is shared across every component
  // that asks for it. Cloning gives this instance its own materials to tint —
  // mutating the cached ones leaks colour into every other view of the scene and
  // survives navigation.
  const model = useMemo(() => scene.clone(true), [scene])

  useEffect(() => {
    model.traverse((node) => {
      if (!(node instanceof THREE.Mesh)) return
      const objectId = nodeOwner.get(node.name)
      if (objectId === undefined) return

      const colour =
        objectId === selectedObjectId
          ? COLOURS.selected
          : failing.has(objectId)
            ? COLOURS.failed
            : COLOURS.ok

      const material = new THREE.MeshStandardMaterial({
        color: colour,
        roughness: 0.75,
        metalness: 0.0,
      })
      node.material = material
      node.castShadow = true
      node.receiveShadow = true
    })
  }, [model, failing, selectedObjectId, nodeOwner])

  return (
    <primitive
      object={model}
      onClick={(event: THREE.Event & { stopPropagation: () => void; object: THREE.Object3D }) => {
        event.stopPropagation()
        const objectId = nodeOwner.get(event.object.name)
        if (objectId) select(objectId)
      }}
    />
  )
}
