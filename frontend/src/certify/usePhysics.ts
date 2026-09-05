/**
 * Drives the MuJoCo worker: fetch the bundle, step on a rAF loop, hand back poses.
 *
 * **Off until asked, and never destructive.** Physics is a thing the user opts
 * into, for two reasons. The 9.7 MB WASM module should not be downloaded by
 * someone who only wants to look at a scene. And more importantly the solved pose
 * *is* the answer — running gravity on load would show the scene falling and hide
 * what the solver produced. Stepping only ever moves meshes in the viewport;
 * nothing is written back to `SceneSpec`, so "reset" means asking the server's copy
 * again rather than undoing anything.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api, storageUrl } from '../api/client'
import type { Command, Event, Pose } from './physics.worker'
import PhysicsWorker from './physics.worker?worker'

/** Simulated seconds advanced per animation frame. */
const STEP_SECONDS = 1 / 60

export type PhysicsState = 'idle' | 'loading' | 'running' | 'error'

export interface Physics {
  state: PhysicsState
  /** Live poses by object id while running; empty when idle. */
  poses: Map<string, Pose>
  /** Simulated seconds since the scene was compiled. */
  time: number
  error: string | null
  enable: () => void
  disable: () => void
  reset: () => void
}

export function usePhysics(sceneId: string | null, updatedAt?: string): Physics {
  const [state, setState] = useState<PhysicsState>('idle')
  const [poses, setPoses] = useState<Map<string, Pose>>(new Map())
  const [time, setTime] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const worker = useRef<Worker | null>(null)
  const frame = useRef<number | null>(null)
  const stepping = useRef(false)

  const teardown = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current)
    frame.current = null
    stepping.current = false
    worker.current?.terminate()
    worker.current = null
  }, [])

  // A scene change invalidates everything: different MJCF, different bodies, and a
  // stale model would report poses for object ids the new scene does not contain.
  //
  // Reset during render, not in an effect. React's documented pattern for "throw
  // this state away when a prop changes" — an effect that calls setState would
  // render once with the old scene's poses applied to the new scene's objects
  // before correcting itself, and the lint rule against it is right.
  const key = `${sceneId ?? ''}@${updatedAt ?? ''}`
  const [previousKey, setPreviousKey] = useState(key)
  if (key !== previousKey) {
    setPreviousKey(key)
    setState('idle')
    setPoses(new Map())
    setTime(0)
    setError(null)
  }

  // The worker is an external system, so tearing it down belongs in an effect. It
  // runs on a scene change and on unmount, and both want the same thing.
  useEffect(() => teardown, [key, teardown])

  const enable = useCallback(() => {
    if (!sceneId || worker.current) return
    setState('loading')
    setError(null)

    const instance = new PhysicsWorker()
    worker.current = instance

    instance.onmessage = (event: MessageEvent<Event>) => {
      const message = event.data
      if (message.type === 'ready') {
        setState('running')
        stepping.current = true
        const tick = () => {
          if (!stepping.current) return
          instance.postMessage({ type: 'step', seconds: STEP_SECONDS } satisfies Command)
          frame.current = requestAnimationFrame(tick)
        }
        frame.current = requestAnimationFrame(tick)
      } else if (message.type === 'poses') {
        setTime(message.time)
        setPoses(new Map(message.poses.map((pose) => [pose.objectId, pose])))
      } else if (message.type === 'error') {
        setError(message.message)
        setState('error')
        stepping.current = false
      }
    }

    void (async () => {
      try {
        const bundle = await api.scenePhysics(sceneId)
        // Fetched here rather than in the worker so one failed asset surfaces as a
        // message the panel can show, instead of MuJoCo failing to compile with a
        // reason that has to be dug out of the console.
        const entries = await Promise.all(
          Object.entries(bundle.meshes).map(async ([name, path]) => {
            const response = await fetch(storageUrl(path, updatedAt))
            if (!response.ok) throw new Error(`${name}: ${response.status}`)
            return [name, await response.arrayBuffer()] as const
          }),
        )
        const meshes = Object.fromEntries(entries)
        instance.postMessage(
          { type: 'start', mjcf: bundle.mjcf, meshes, reportInterval: STEP_SECONDS },
          // Transferred, not copied: 109 buffers is worth not duplicating.
          Object.values(meshes),
        )
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
        setState('error')
      }
    })()
  }, [sceneId, updatedAt])

  const disable = useCallback(() => {
    teardown()
    setState('idle')
    setPoses(new Map())
    setTime(0)
  }, [teardown])

  const reset = useCallback(() => {
    // Back to the compiled pose — which is the server's, since nothing here ever
    // wrote to it.
    worker.current?.postMessage({ type: 'reset' } satisfies Command)
    setTime(0)
  }, [])

  return useMemo(
    () => ({ state, poses, time, error, enable, disable, reset }),
    [state, poses, time, error, enable, disable, reset],
  )
}
