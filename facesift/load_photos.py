from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Photo:
    path: Path
    image: np.ndarray  # BGR


class PhotoLoader:
    """Finds images under a folder (recursively) and reads them."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = self.find_images()
        self.failed: list[Path] = []  # filled while iterating

    def find_images(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """Read an image as BGR. Works with non-ASCII paths, unlike
        cv2.imread."""
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return image

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Photo]:
        for path in self.paths:
            try:
                image = self.load_image(path)
            except ValueError:
                self.failed.append(path)  # remember it, keep going
                continue
            yield Photo(path=path, image=image)
