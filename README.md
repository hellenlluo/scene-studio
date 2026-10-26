# SceneStudio

SceneStudio turns a single room photograph into an editable, physics-ready 3D scene. It combines vision models, geometric reconstruction, constrained optimization, and MuJoCo validation to recover individual objects, their scale and support relationships, and exportable simulation assets.

It helps to think of the pipeline as increasingly structured representations of the same photograph:

```text
image
  -> object inventory
  -> pixel masks + metric depth
  -> semantic labels + support relationships
  -> separate 3D meshes
  -> metric scene graph
  -> physics-ready objects
  -> optimized scene
  -> certificate + automatic repairs
  -> glTF + MJCF files
```

## What it demonstrates

- An end-to-end asynchronous reconstruction pipeline with cached, typed stages
- Object-level 3D reconstruction, scale/pose solving, collision proxies, and inertial estimation
- Auditable physics certification across scale, balance, stability, inertia, and simulation cost
- Targeted automatic repairs followed by re-certification
- A React/Three.js workspace for inspecting, editing, simulating, and exporting the result
- Self-contained glTF (`.glb`) output and a MuJoCo MJCF bundle for downstream use

## Run locally

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Node.js `20.19+` or `22.12+`
- OpenAI and Fal API credentials

### 1. Configure the backend

Create `backend/.env`:

```dotenv
OPENAI_API_KEY=your_openai_key
FAL_KEY=your_fal_key

# Optional: override the vision model used by the labeling/verification stages
# SCENESTUDIO_OPENAI_MODEL=gpt-5
```

Then install dependencies and start the API:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`; interactive API docs are available at `http://localhost:8000/docs`.

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a JPEG, PNG, or WebP image, and follow the pipeline status in the UI. A full reconstruction uses external model APIs and can take several minutes. The local metric-depth checkpoint is downloaded on first use.

## Project structure

```text
backend/
  app/pipeline/    reconstruction and scene-solving stages
  app/certify/     physics checks and repair strategies
  app/export/      glTF and MJCF exporters
  tests/           unit, API, simulation, and pipeline tests
frontend/
  src/scene/       3D viewer, editing tools, and scene state
  src/certify/     certificate UI and browser-side MuJoCo simulation
```

SceneStudio is an experimental single-user prototype. Jobs run in-process and scene metadata is stored in SQLite, which keeps the demo easy to run while leaving the API boundary ready for a production queue and database.
