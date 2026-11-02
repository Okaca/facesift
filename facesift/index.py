"""Build the face index for a folder of photos (e.g. everything guests uploaded).

Writes everything the web app needs into one folder (default data/):
    embeddings.npy   one 512-d vector per face
    faces.csv        one row per face: which photo, where in it, detector score
    photos.json      one entry per photo: id, source path, size
    thumbs/, display/  web-sized copies of every photo, named by photo id

Usage (from the repo root):
    python -m facesift.index "archive/Original Images/Original Images"
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from facesift.devices import check_active, select_providers
from facesift.face_cropper import FaceCropper
from facesift.face_encoder import FaceEncoder
from facesift.images import save_web_images
from facesift.load_photos import PhotoLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder of images (searched recursively)")
    parser.add_argument("--crops", type=Path, help="if given, save every face crop into this folder")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data"), help="Folder to write the index into (default: data)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Where to run the models: auto uses an NVIDIA GPU if it works, else CPU (default: auto)",
    )
    args = parser.parse_args()

    loader = PhotoLoader(args.input_dir)
    if len(loader) == 0:
        parser.error(f"No images found in {args.input_dir}")

    providers, ctx_id = select_providers(args.device)
    cropper = FaceCropper(providers=providers, ctx_id=ctx_id)
    encoder = FaceEncoder(providers=providers, ctx_id=ctx_id)
    try:
        print(f"Detector runs on: {check_active(cropper.provider, args.device)}")
        print(f"Encoder runs on : {check_active(encoder.provider, args.device)}")
    except RuntimeError as e:
        parser.exit(1, f"error: {e}\n")
    for sub in ("thumbs", "display"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    photos = []  # one dict per unique photo, saved as photos.json
    rows = []  # one dict per face, describes the matching row in vectors
    vectors = []  # one (n_faces, 512) array per photo
    seen = {}  # photo id -> first path with those bytes
    duplicates = []  # (skipped path, path it duplicates)
    loaded = total_faces = 0
    no_face = []

    for photo in tqdm(loader, total=len(loader), desc="Indexing", unit="img"):
        loaded += 1
        photo_id = make_photo_id(photo.path)
        if photo_id in seen:
            # byte-identical to a photo already indexed: same id, nothing new to add
            duplicates.append((photo.path, seen[photo_id]))
            continue
        seen[photo_id] = photo.path

        height, width = photo.image.shape[:2]
        photos.append(
            {
                "id": photo_id,
                "source": str(photo.path.relative_to(args.input_dir)),
                "width": width,
                "height": height,
            }
        )
        save_web_images(photo.image, photo_id, args.output)

        faces = cropper.crop_faces(photo.image)
        if not faces:
            no_face.append(photo.path)
        total_faces += len(faces)

        vectors.append(encoder.encode([face.image for face in faces]))
        for face in faces:
            rows.append(
                {
                    "photo_id": photo_id,
                    "face_index": face.face_index,
                    "x1": face.bbox[0],
                    "y1": face.bbox[1],
                    "x2": face.bbox[2],
                    "y2": face.bbox[3],
                    "det_score": round(face.det_score, 4),
                }
            )

        if args.crops:
            # mirror the input folder layout so photos from different guests that
            # happen to share a name (IMG_0001.jpg) don't overwrite each other
            out_dir = args.crops / photo.path.parent.relative_to(args.input_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for face in faces:
                cv2.imwrite(str(out_dir / f"{photo.path.stem}_face{face.face_index}.jpg"), face.image)

    np.save(args.output / "embeddings.npy", np.concatenate(vectors))
    with open(args.output / "faces.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo_id", "face_index", "x1", "y1", "x2", "y2", "det_score"])
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output / "photos.json", "w") as f:
        json.dump(photos, f, indent=1)

    print(f"\nImages found : {len(loader)}")
    print(f"Loaded       : {loaded}")
    print(f"Unreadable   : {len(loader.failed)}")
    for path in loader.failed:
        print(f"  {path}")
    print(f"Duplicates   : {len(duplicates)}")
    for path, original in duplicates:
        print(f"  {path} (same as {original.name})")
    print(f"Indexed      : {len(photos)} photos -> {args.output}")
    print(f"Faces found  : {total_faces}")
    print(f"No face      : {len(no_face)}")


def make_photo_id(path: Path) -> str:
    """Name a photo by its contents: the first 16 hex digits of its SHA-256.

    The same file always gets the same id, so ids survive rebuilds (cached
    thumbnails stay correct) and identical files collapse into one photo.
    16 hex digits = 64 bits, so two different photos colliding is not a concern.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    main()
