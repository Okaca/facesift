import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, Query, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

from facesift.search import FaceIndex
from facesift.selfie import SelfieEncoder

BASE_PATH = os.environ.get("BASE_PATH", "/facesift")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DEVICE = os.environ.get("DEVICE", "auto")
MAX_UPLOAD = 10 * 1024 * 1024


index = FaceIndex(DATA_DIR)
selfie = SelfieEncoder(DEVICE)


def photo_urls(photo_id) -> dict:

    return {
        "thumb": f"{BASE_PATH}/media/thumbs/{photo_id}.webp",
        "display": f"{BASE_PATH}/media/display/{photo_id}.jpg",
    }


app = FastAPI(title="FaceSift")
api = APIRouter(prefix=f"{BASE_PATH}/api")


@api.get("/photos")
async def get_photos(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    page = index.photos[offset : offset + limit]
    items = []
    for photo in page:
        items.append(
            {
                "id": photo["id"],
                "width": photo["width"],
                "height": photo["height"],
                **photo_urls(photo["id"]),
            }
        )
    return {"total": len(index.photos), "items": items}


@api.post("/search")
def search(file: UploadFile):
    # Reading one byte more than the limit is a trick:
    # if you got more than MAX_UPLOAD bytes, the file is too big,
    # and you never had to read a 2 GB upload into memory to find that out.
    data = file.file.read(MAX_UPLOAD + 1)

    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    try:
        result = selfie.embed(data)
        if result is None:
            raise HTTPException(422, detail="No face found, try another photo")
        embedding, box = result
    except ValueError:
        raise HTTPException(400, detail="Not an image")

    matches = index.query(embedding)

    items = []
    for match in matches:
        items.append(
            {
                "id": match["id"],
                "score": match["score"],
                **photo_urls(match["id"]),
            }
        )
    return {"face_box": box, "total": len(matches), "matches": items}


app.include_router(api)
app.mount(f"{BASE_PATH}/media/thumbs", StaticFiles(directory=DATA_DIR / "thumbs"), name="thumbs")
app.mount(f"{BASE_PATH}/media/display", StaticFiles(directory=DATA_DIR / "display"), name="display")
