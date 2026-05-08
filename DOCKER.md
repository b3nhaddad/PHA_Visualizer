# Docker Operations Guide

## Directory layout

```
3dsubag/
├── docker-compose.yml       ← orchestrates both services
├── frontend/                ← React + Vite viewer
│   ├── Dockerfile
│   ├── src/
│   └── public/
├── backend/                 ← FastAPI algorithm server
│   ├── Dockerfile
│   ├── api/
│   │   ├── main.py          ← app entry point
│   │   ├── models.py        ← Pydantic schema (mirrors src/api/types.ts)
│   │   └── routes/runs.py   ← 3 API endpoints
│   ├── algorithm/           ← plug your Subag implementation here
│   └── tests/
└── tools/
    ├── mock_run.py          ← generate demo JSON without Docker
    └── validate_schema.py   ← validate any run JSON against the schema
```

---

## Starting the apps

### Full stack (recommended)

Requires [Docker Desktop](https://docs.docker.com/get-docker/).

```bash
# First time (or after dependency changes)
docker compose up --build

# Subsequent starts (images already built)
docker compose up
```

| Service  | URL                         |
|----------|-----------------------------|
| Frontend | http://localhost:5173       |
| Backend  | http://localhost:8000       |
| API docs | http://localhost:8000/docs  |

The frontend Vite proxy forwards every `/api/*` request to the backend inside
the Docker network, so the browser never needs to know the backend's address.

### Frontend only (no Docker)

```bat
run.bat
```

Or manually:

```bat
set PATH=C:\Users\schmi\AppData\Local\node-portable;%PATH%
cd frontend
npm install
npm run dev
```

### Backend only (no Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

---

## Common Docker commands

```bash
# View logs from both services
docker compose logs -f

# View backend logs only
docker compose logs -f backend

# Stop everything
docker compose down

# Rebuild a single service after code changes
docker compose up --build backend

# Open a shell inside the backend container (for debugging)
docker compose exec backend bash

# Run backend tests inside the container
docker compose exec backend pytest tests/ -v

# Run tests locally (outside Docker)
cd backend
pytest tests/ -v
```

---

## Connecting your algorithm

1. **Add your code** to `backend/algorithm/`.  
   The recommended interface is a generator that yields `SnapshotBatch` objects:

   ```python
   # backend/algorithm/subag.py
   from api.models import SnapshotBatch, StartRunRequest

   def run(request: StartRunRequest):
       # ... your Subag implementation ...
       for batch in ...:
           yield batch   # SnapshotBatch
   ```

2. **Wire it into the route** — open `backend/api/routes/runs.py` and replace
   the `# ── STUB ──` block in `_run_algorithm()` with:

   ```python
   from algorithm.subag import run as subag_run
   for batch in subag_run(request):
       entry["batches"].append(batch)
   ```

3. **Remove the `time.sleep`** in the stub — it was there only to simulate
   computation time.

4. **Activate the algorithm tests** — remove `@pytest.mark.skip` in
   `backend/tests/test_algorithm.py` and run `pytest tests/ -v`.

5. **Validate a real run file**:

   ```bash
   python tools/validate_schema.py path/to/your_run.json --strict
   ```

---

## Environment variables

| Variable      | Service  | Default                   | Purpose                              |
|---------------|----------|---------------------------|--------------------------------------|
| `BACKEND_URL` | frontend | `http://localhost:8000`   | Vite proxy target (Docker sets this) |

Set in `docker-compose.yml` under `services.frontend.environment`.  
In local dev (no Docker) the proxy falls back to `http://localhost:8000`
via the `??` default in `vite.config.ts`.

---

## Regenerating mock data

```bat
regen_mock.bat
```

Or with options:

```bash
python tools/mock_run.py frontend/public/mock_run.json --model pure --p 4 --N 60 --k 150
python tools/mock_run.py frontend/public/mock_mixed.json --model mixed
```

---

## Integrating your teammate's backend repo

If the algorithm lives in a separate GitHub repository:

```bash
# At the repo root
git submodule add https://github.com/your-org/backend-repo backend/algorithm
git submodule update --init --recursive
```

Then in `docker-compose.yml` the `backend` service already mounts `./backend`
into the container, so the submodule files are available.

Alternative: clone the repo separately and mount it:

```yaml
# docker-compose.yml (override)
services:
  backend:
    volumes:
      - ./backend:/app
      - ../your-backend-repo/algorithm:/app/algorithm
```
