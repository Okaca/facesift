import csv
import json
from pathlib import Path

import numpy as np

MATCH_THRESHOLD = 0.40


class FaceIndex:
    def __init__(self, data_dir: Path, threshold: float = MATCH_THRESHOLD) -> None:
        self.data_dir = Path(data_dir)
        self.threshold = threshold
        self.embeddings = np.load(self.data_dir / "embeddings.npy")
        with open(self.data_dir / "photos.json") as f:
            self.photos = json.load(f)
        self.photo_pos = {photo["id"]: i for i, photo in enumerate(self.photos)}

        with open(self.data_dir / "faces.csv", newline="") as f:
            reader = csv.DictReader(f)
            positions = [self.photo_pos[row["photo_id"]] for row in reader]
        self.face_photo = np.array(positions, dtype=np.int64)

        if len(self.face_photo) != len(self.embeddings):
            raise ValueError(
                f"faces.csv has {len(self.face_photo)} rows but embeddings.npy has "
                f"{len(self.embeddings)}; rebuild {data_dir} with facesift.index"
            )

    def query(self, embedding: np.ndarray) -> list[dict]:
        scores = (
            self.embeddings @ embedding
        )  # @ is matrix multiplication: (2594, 512) @ (512,) gives (2594,), one score per face, and that score is the cosine similarity.
        best = np.full(len(self.photos), -1.0)

        np.maximum.at(best, self.face_photo, scores)

        keep = np.flatnonzero(best >= self.threshold)  # positions of the matching photos

        order = np.argsort(-best[keep])  # highest score first
        keep = keep[order]

        results = []
        for p in keep:  # p is a photo position
            results.append({"id": self.photos[p]["id"], "score": round(float(best[p]), 4)})

        return results
