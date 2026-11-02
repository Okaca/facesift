# FaceSift demo web app: plan

A demo where a visitor uploads a selfie and sees every photo of themselves from an event collection. The demo uses the photos in `archive/`. It runs on a Raspberry Pi 5 next to other projects, behind a Cloudflare Tunnel, at **https://onurkagancoskun.com/facesift**.

## 1. Architecture

The expensive work (detecting and encoding every photo) happens once, on the laptop GPU. The Pi only handles the uploaded selfie.

```
LAPTOP (GPU, run once)                         RASPBERRY PI 5 (CPU, always on)
───────────────────────                        ─────────────────────────────────
archive/ photos                                 Docker container "facesift"
   │                                              ┌──────────────────────────────┐
   ▼                                              │ FastAPI (Python)             │
facesift index  ──► data/                         │  • loads embeddings into RAM │
   • detect + encode faces   embeddings.npy  ─rsync─►  • serves the Svelte UI     │
   • make thumbnails         faces.csv            │  • /api/photos  (grid pages) │
   • make display images     photos.json          │  • /api/search  (selfie)     │
                             thumbs/  display/    └──────────────────────────────┘
                                                              ▲
                                               cloudflared (host) → localhost:3003
                                                              ▲
                                                           browser
```

- 2598 faces × 512 floats is about 5 MB. Comparing a selfie against all of them is one matrix multiplication, so no vector database is needed.
- The Pi must run the same model that built the index (`buffalo_l`). Embeddings from different models can't be compared.

## 2. Search flow

1. The browser sends the image to `POST /api/search`.
2. The server decodes it **in memory** (it is never written to disk), shrinks it if it's large, and detects faces.
   - No face: return an error, and the UI says "No face found, try another photo".
   - Several faces: use the largest and return its box, so the UI can show which face was used.
3. Encode that face and compute its cosine similarity with every indexed face.
4. Group by photo: a photo's score is its best-matching face.
5. Keep photos at or above the **threshold (0.40 cosine)** and sort them best first.
6. Return the full match list (IDs and scores only, so it's small). The UI shows 50 at a time.

## 3. API

Every route lives under the base path **`/facesift`**. The tunnel passes the path through unchanged, so the app receives `/facesift/api/search`, not `/api/search`.

| Endpoint                                     | Returns                                                                                                                |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `GET /facesift/api/photos?offset=0&limit=50` | `{total, items: [{id, thumb, display, width, height}]}`                                                                |
| `POST /facesift/api/search` (image upload)   | `{face_box, total, matches: [{id, thumb, display, score}]}`, or an error (no face found, file too large, not an image) |
| `GET /facesift/media/thumbs/{id}.webp`       | Grid thumbnail, about 400 px                                                                                           |
| `GET /facesift/media/display/{id}.jpg`       | Full-view image, about 1600 px (the originals go up to 2807 px)                                                        |
| `GET /facesift/`                             | The built Svelte app                                                                                                   |

The base path is one setting on each side, and local development uses the same prefix, so dev and production behave the same:

- **FastAPI:** every route is registered under a `BASE_PATH` (default `/facesift`).
- **Vite:** `base: '/facesift/'`, so the built HTML loads its JS and CSS from `/facesift/assets/...`. The frontend builds API URLs from `import.meta.env.BASE_URL` instead of hard-coding `/api`.

The full collection is paged by the server. Search results come back all at once, because re-uploading the selfie for each page would be wasteful. In both cases **--More--** shows the next 50.

## 4. Frontend (Svelte + Vite)

```
┌──────────────────┬───────────────────────────────────────────┐
│ ┌──────────┐ ✕  │  All photos (2562)    /    12 matches       │
│ │  selfie  │    │  ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐              │
│ │ preview  │    │  │   ││   ││   ││   ││   ││   │              │
│ │ [face □] │    │  └───┘└───┘└───┘└───┘└───┘└───┘              │
│ └──────────┘    │  ┌───┐┌───┐┌───┐ ...  (50 per batch)         │
│ Upload a photo  │                                             │
│ to find yourself│              -- More --                     │
└──────────────────┴───────────────────────────────────────────┘
```

| Component            | Role                                                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SelfiePanel.svelte` | Upload by click or drag-and-drop, preview with the detected face box, ✕ in the top-right corner. ✕ clears the selfie and the grid goes back to all photos. A new upload replaces the old one and searches again. |
| `PhotoGrid.svelte`   | Thumbnail grid, a heading ("All photos (2562)" / "12 matches"), and the --More-- button, which hides when nothing is left.                                                                                       |
| `Lightbox.svelte`    | Optional: click a thumbnail to view the photo full size.                                                                                                                                                         |
| `api.js`             | All `fetch` calls to the backend.                                                                                                                                                                                |

- **Screen states:** browsing all photos, searching (spinner), showing matches, no face in the selfie, no matches.
- **On a phone,** the selfie panel stacks above the grid.
- Plain Svelte + Vite, without SvelteKit: a single-page demo needs neither routing nor server-side rendering.

## 5. Repo layout

```
facesift/
├── facesift/              # Python package (the pipeline moves here)
│   ├── devices.py, face_cropper.py, face_encoder.py, load_photos.py
│   ├── index.py           # replaces main.py: builds data/ (+ thumbs, display)
│   └── search.py          # FaceIndex: load data/, query(embedding) -> matches
├── server/
│   └── app.py             # FastAPI: the routes above + serves the web build
├── web/                   # Svelte + Vite app
│   ├── src/App.svelte, src/lib/*.svelte, src/lib/api.js
│   └── vite.config.js     # base '/facesift/', dev server forwards /facesift/api to FastAPI
├── data/                  # generated, gitignored, copied to the Pi
├── docs/PLAN.md
├── Dockerfile
├── docker-compose.yml     # reference copy of the Pi's compose file
└── requirements.txt
```

## 6. Deployment

### Target

|            |                                                                                                                                                                              |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Machine    | Raspberry Pi 5, `aarch64`, 16 GB RAM, Pi OS                                                                                                                                  |
| Neighbours | portfolio (:3000), openhouse (:3001), market_webapp (:3002), each with its own compose folder under `/home/okcpi/`                                                           |
| Ingress    | `cloudflared` as a systemd service on the host, rules in `/etc/cloudflared/config.yml`. Projects are routed by **path** on `onurkagancoskun.com` (e.g. `/openhouse` → :3001) |
| FaceSift   | `/home/okcpi/facesift/`, port `127.0.0.1:3003`, public at `onurkagancoskun.com/facesift`                                                                                     |

### Packages

There is no `onnxruntime-gpu` for ARM. One `requirements.txt` picks the right build per machine with environment markers:

```
onnxruntime-gpu[cuda,cudnn]==1.31.1; platform_machine == "x86_64"
onnxruntime==1.31.1;                 platform_machine == "aarch64"
```

`insightface` lists the CPU `onnxruntime` as a dependency, so on the laptop it can get installed again and shadow the GPU build. The fix is `pip uninstall -y onnxruntime`, then `pip install --force-reinstall --no-deps onnxruntime-gpu==1.31.1`. This doesn't affect the Pi image.

### Image

A multi-stage `Dockerfile`:

1. A `node` stage builds `web/` into static files. Node never reaches the final image.
2. A `python:3.12-slim` stage installs the Python packages. `insightface` 2.1, `onnxruntime` and `opencv-python-headless` all ship ready-made ARM64 wheels, so nothing compiles under emulation.
3. The final image holds Python, the packages, the app and the built UI.

It is built on the laptop, like the other projects:

```
docker buildx build --platform linux/arm64 -t metu1onurc1/facesift:latest --push .
```

### On the Pi

`data/` and the `buffalo_l` models are **mounted read-only**, not baked into the image. Updating the demo photos means copying files, not rebuilding.

```
rsync -av data/ okcpi@raspberrypi:~/facesift/data/      # from the laptop
docker compose pull && docker compose up -d             # on the Pi
```

`/etc/cloudflared/config.yml` gets two path rules, one per hostname. They must go **above the portfolio rules**, because rules are matched top to bottom and a portfolio rule (no `path`) matches every path on the domain:

```yaml
ingress:
  # FaceSift: before the catch-all portfolio rules (first match wins)
  - hostname: onurkagancoskun.com
    path: ^/facesift
    service: http://localhost:3003
  - hostname: www.onurkagancoskun.com
    path: ^/facesift
    service: http://localhost:3003
```

The hostnames already exist, so no new DNS record is needed. Run `sudo systemctl restart cloudflared` to apply the change.

### Sharing the Pi

- One uvicorn worker. Each worker loads its own copy of the models.
- ONNX intra-op threads capped at about 2, so a search doesn't take all 4 cores from the other projects.
- Models load once at startup, not per request.
- `mem_limit` in compose as a safety net (RAM is not a constraint).
- Cloudflare can cache `/facesift/media/*` so repeat visits don't hit the Pi.
- **Measure on the Pi:** search latency, and RAM while idle and while searching.

## 7. Privacy and safety

- Selfies are processed in memory only: never stored and never logged.
- Upload limit of about 10 MB, image types only, checked in FastAPI. Cloudflare's own limit is 100 MB.
- A notice under the upload slot: "Your photo is used only for this search and is not stored."
- Real event photos, embeddings and selfies are never committed (`.gitignore` covers `archive/`, `output*/`, `data/`, `*.npy`).

## 8. Build order

| Phase | What                                                                  | Where      | Done when                                                                |
| ----- | --------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------ |
| 0     | Restructure into the `facesift/` package, add `data/` to `.gitignore` | laptop     | the pipeline runs as before from the package                             |
| 1     | Similarity analysis, then choose a threshold                          | laptop     | same-person vs different-person score distributions + a chosen threshold |
| 2     | Indexer: `data/` with thumbnails and display images                   | laptop GPU | `data/` is complete for all 2562 photos                                  |
| 3     | FastAPI backend                                                       | laptop     | both endpoints work with `curl`                                          |
| 4     | Svelte UI                                                             | laptop     | upload, ✕, --More-- and matches work in the browser                      |
| 5     | Docker image, first run locally                                       | laptop     | the container serves the full app on the laptop                          |
| 6     | Deploy: rsync, compose, cloudflared rules                             | Pi         | reachable at `onurkagancoskun.com/facesift`, latency and RAM measured    |
| 7     | README, screenshots, demo GIF                                         | —          | ready for the portfolio                                                  |

## 9. Open questions

- **Threshold:** decided in phase 1: **0.40**. Different people never scored above 0.339; in the search simulation 0.40 returned 0 wrong photos at 99.0% recall (worst person 90.9%).
- **Dataset license:** the demo photos are celebrity images from a Kaggle dataset. Showing them on a public site is a copyright and likeness gray area. Check the license before phase 6, and switch to a dataset with explicit permission if needed. Local development is unaffected.
