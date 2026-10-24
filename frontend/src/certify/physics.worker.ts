/// <reference lib="webworker" />
/**
 * MuJoCo in a Web Worker: the same engine the backend certifies with.
 *
 * `@mujoco/mujoco` is 3.11.0 and the backend's Python bindings are 3.11.0 — the
 * same MuJoCo compiled to WASM, not a lookalike. It compiles the MJCF the server
 * built, so what the user sees when they press play is what the stability axis
 * measured, rather than an approximation of it that could disagree with no
 * principled tiebreak.
 *
 * **In a worker because the module is 9.7 MB and the viewer is on the main
 * thread.** Compiling 109 mesh assets and stepping at 500 Hz would drop frames on
 * the thread that has to draw them. Loaded on first use rather than at page load,
 * so a user who never presses the button never pays for it.
 *
 * **Assets go through MuJoCo's virtual filesystem.** `mjcf.build_xml` names them by
 * bare filename for exactly this reason; `mj_loadXML(xml, vfs)` is the overload
 * that accepts them. Absolute paths would be unopenable here, and a browser-only
 * variant of the MJCF would be the drift a single `build_xml` exists to prevent.
 */
import loadMuJoCo, { type MainModule, type MjData, type MjModel } from '@mujoco/mujoco'

import { quaternionTiltDeg } from './motion'

/** Bodies are named `{object_id}/{part_id}` by `mjcf.body_name`. */
const objectOf = (bodyName: string) => bodyName.split('/')[0]

export interface StartMessage {
  type: 'start'
  mjcf: string
  /** Mesh filename, as named in the MJCF, to the bytes for it. */
  meshes: Record<string, ArrayBuffer>
  /** Simulated seconds between pose reports. Coarser than the timestep. */
  reportInterval: number
  maxDisplacementM: number
  maxTiltDeg: number
}

export type Command =
  | StartMessage
  | { type: 'step'; seconds: number }
  | { type: 'reset' }
  | { type: 'pause' }
  | { type: 'resume' }

export interface Pose {
  objectId: string
  position: [number, number, number]
  /** MuJoCo order: w, x, y, z. */
  quaternion: [number, number, number, number]
}

export interface PhysicsViolation {
  objectId: string
  displacementM: number
  tiltDeg: number
}

export type Event =
  | { type: 'ready'; bodies: number; meshes: number }
  | { type: 'poses'; time: number; poses: Pose[]; violations: PhysicsViolation[] }
  /**
   * The scene diverged and MuJoCo threw the state away.
   *
   * Separate from `error` because nothing failed in the usual sense: the model
   * compiled, the assets loaded, and the engine did exactly what it documents.
   * `mj_step` checks `qacc` for Nan/Inf/huge and calls `mj_resetData` when it
   * finds one — silently, to a console the user cannot see. Left undetected the
   * caller keeps stepping a scene that restarts every time it blows up, which on
   * `room2.png` was every 0.446 s, forever, and reads as a hang rather than as
   * the stability failure the certificate already reports.
   */
  | { type: 'unstable'; time: number; message: string }
  | { type: 'error'; message: string }

let mujoco: MainModule | null = null
let model: MjModel | null = null
let data: MjData | null = null
/** Root body id per object, in the order poses are reported. */
let roots: { objectId: string; bodyId: number }[] = []
let running = false
/** Simulated time at the previous report, to catch MuJoCo resetting itself. */
let lastTime = 0
let initial = new Map<string, Pose>()
let initialCom = new Map<string, [number, number, number]>()
let maximumMotion = new Map<string, PhysicsViolation>()
let limits = { displacementM: 0.01, tiltDeg: 2 }

function poseOf(objectId: string, bodyId: number): Pose {
  if (!data) throw new Error('MuJoCo data is not ready')
  const body = data.body(bodyId)
  const p = body.xpos as ArrayLike<number>
  const q = body.xquat as ArrayLike<number>
  return {
    objectId,
    position: [p[0], p[1], p[2]],
    quaternion: [q[0], q[1], q[2], q[3]],
  }
}

function captureInitial() {
  initial = new Map(roots.map(({ objectId, bodyId }) => [objectId, poseOf(objectId, bodyId)]))
  initialCom = new Map(
    roots.map(({ objectId, bodyId }) => {
      if (!data) throw new Error('MuJoCo data is not ready')
      const com = data.body(bodyId).subtree_com as ArrayLike<number>
      return [objectId, [com[0], com[1], com[2]]]
    }),
  )
  maximumMotion = new Map()
}

async function boot(message: StartMessage) {
  // One module for the worker's lifetime. Re-initialising for every scene would
  // re-download nothing (the browser caches the wasm) but would still re-compile
  // and re-instantiate it, which is the slow part.
  mujoco ??= await loadMuJoCo()

  // The VFS wants real files. Writing them under `/` matches the bare filenames in
  // the MJCF, which is the whole reason the MJCF uses bare filenames.
  for (const [name, bytes] of Object.entries(message.meshes)) {
    mujoco.FS.writeFile(`/${name}`, new Uint8Array(bytes))
  }
  mujoco.FS.writeFile('/scene.xml', message.mjcf)

  // `from_xml_path`, not `from_xml_string`: compiled from inside the virtual
  // filesystem, the MJCF's bare mesh filenames resolve against `/`, which is where
  // the assets were just written. Mirrors the Python side's `from_xml_string(xml,
  // assets)` — same overload family, same one MJCF.
  model = mujoco.MjModel.from_xml_path('/scene.xml')
  data = new mujoco.MjData(model)
  limits = {
    displacementM: message.maxDisplacementM,
    tiltDeg: message.maxTiltDeg,
  }

  // Root bodies only: a part's body moves with its parent, and reporting every
  // part would send the viewer transforms it already composes itself.
  roots = []
  const seen = new Set<string>()
  for (let id = 1; id < model.nbody; id += 1) {
    const name = model.body(id).name
    if (!name) continue
    const objectId = objectOf(name)
    if (seen.has(objectId)) continue
    seen.add(objectId)
    roots.push({ objectId, bodyId: id })
  }

  mujoco.mj_forward(model, data)
  captureInitial()

  postMessage({
    type: 'ready',
    bodies: roots.length,
    meshes: Object.keys(message.meshes).length,
  } satisfies Event)
}

function report() {
  if (!model || !data) return
  const current = data
  const poses = roots.map(({ objectId, bodyId }) => poseOf(objectId, bodyId))
  for (let index = 0; index < poses.length; index += 1) {
    const pose = poses[index]
    const start = initial.get(pose.objectId)
    const startCom = initialCom.get(pose.objectId)
    if (!start || !startCom) continue
    // Match the backend certificate exactly: it measures subtree COM motion, not
    // root-body-origin motion. Those points separate when an off-centre object
    // rotates, and comparing different points can invent a frontend/backend
    // disagreement even when both MuJoCo builds produced the same trajectory.
    const com = current.body(roots[index].bodyId).subtree_com as ArrayLike<number>
    const displacementM = Math.hypot(
      com[0] - startCom[0],
      com[1] - startCom[1],
      com[2] - startCom[2],
    )
    const tiltDeg = quaternionTiltDeg(start.quaternion, pose.quaternion)
    const previous = maximumMotion.get(pose.objectId)
    maximumMotion.set(pose.objectId, {
      objectId: pose.objectId,
      displacementM: Math.max(previous?.displacementM ?? 0, displacementM),
      tiltDeg: Math.max(previous?.tiltDeg ?? 0, tiltDeg),
    })
  }
  postMessage({
    type: 'poses',
    time: Number(current.time),
    poses,
    violations: [...maximumMotion.values()].filter(
      (motion) =>
        motion.displacementM > limits.displacementM ||
        motion.tiltDeg > limits.tiltDeg,
    ),
  } satisfies Event)
}

function advance(seconds: number) {
  if (!mujoco || !model || !data) return
  const steps = Math.max(1, Math.round(seconds / model.opt.timestep))
  for (let i = 0; i < steps; i += 1) mujoco.mj_step(model, data)
  // `mj_step` leaves the derived quantities a state behind, so the poses read
  // straight after it are not the ones just computed. The backend's `_settle` calls
  // `mj_forward` for the same reason.
  mujoco.mj_forward(model, data)

  // Time running backwards is the reset: `mj_resetData` puts it back to zero, and
  // it is the one symptom visible from here — the warning itself goes to a console
  // this worker does not own.
  const now = Number(data.time)
  if (now < lastTime) {
    running = false
    postMessage({
      type: 'unstable',
      time: lastTime,
      message:
        `the scene diverged after ${lastTime.toFixed(2)} s and MuJoCo reset it. ` +
        `This is the stability axis failing, not a viewer bug — an object starts ` +
        `inside what it rests on, and the contact force needed to separate them is ` +
        `large enough to break the solver.`,
    } satisfies Event)
    return
  }
  lastTime = now
  report()
}

self.onmessage = async (event: MessageEvent<Command>) => {
  try {
    const command = event.data
    if (command.type === 'start') {
      running = true
      lastTime = 0
      await boot(command)
      // Report the compiled pose before stepping, so the caller can confirm the
      // browser agrees with the server about where things are *before* gravity.
      if (mujoco && model && data) {
        mujoco.mj_forward(model, data)
        report()
      }
      return
    }
    if (!model || !data || !mujoco) return
    if (command.type === 'step') {
      if (running) advance(command.seconds)
    } else if (command.type === 'reset') {
      mujoco.mj_resetData(model, data)
      mujoco.mj_forward(model, data)
      captureInitial()
      lastTime = 0
      running = true
      report()
    } else if (command.type === 'pause') {
      running = false
    } else if (command.type === 'resume') {
      running = true
    }
  } catch (error) {
    postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    } satisfies Event)
  }
}
