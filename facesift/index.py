"""Build the face index for a folder of photos (e.g. everything guests uploaded).

Usage (from the repo root):
    python -m facesift.index "archive/Original Images/Original Images"
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from facesift.devices import check_active, select_providers
from facesift.face_cropper import FaceCropper
from facesift.face_encoder import FaceEncoder
from facesift.load_photos import PhotoLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder of images (searched recursively)")
    parser.add_argument("--crops", type=Path, help="if given, save every face crop into this folder")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output"), help="Folder for embeddings.npy and faces.csv"
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
    rows = []  # one dict per face, describes the matching row in vectors
    vectors = []  # one (n_faces, 512) array per photo
    loaded = total_faces = 0
    no_face = []

    for photo in tqdm(loader, total=len(loader), desc="Cropping", unit="img"):
        loaded += 1
        faces = cropper.crop_faces(photo.image)
        if not faces:
            no_face.append(photo.path)
        total_faces += len(faces)

        vectors.append(encoder.encode([face.image for face in faces]))
        for face in faces:
            rows.append(
                {
                    "photo": str(photo.path),
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

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", np.concatenate(vectors))
    with open(args.output / "faces.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo", "face_index", "x1", "y1", "x2", "y2", "det_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nImages found : {len(loader)}")
    print(f"Loaded       : {loaded}")
    print(f"Unreadable   : {len(loader.failed)}")
    for path in loader.failed:
        print(f"  {path}")
    print(f"Faces found  : {total_faces}")
    print(f"No face      : {len(no_face)}")


if __name__ == "__main__":
    main()
