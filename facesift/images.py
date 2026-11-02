"""Make the web copies of a photo: a small grid thumbnail and a larger display image."""

from pathlib import Path

import cv2
import numpy as np

THUMB_SIDE = 400  # longest side of a grid thumbnail, in pixels
DISPLAY_SIDE = 1600  # longest side of the full-view image


def fit_within(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink a BGR image so its longest side is at most max_side. Never enlarges."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    size = (round(w * scale), round(h * scale))  # cv2 wants (width, height)
    # INTER_AREA averages the pixels it merges, so downscaled photos stay smooth
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def save_web_images(image: np.ndarray, photo_id: str, out_dir: Path) -> None:
    """Write out_dir/thumbs/<id>.webp and out_dir/display/<id>.jpg."""
    write_image(out_dir / "thumbs" / f"{photo_id}.webp", fit_within(image, THUMB_SIDE), [cv2.IMWRITE_WEBP_QUALITY, 80])
    write_image(
        out_dir / "display" / f"{photo_id}.jpg",
        fit_within(image, DISPLAY_SIDE),
        [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_PROGRESSIVE, 1],
    )


def write_image(path: Path, image: np.ndarray, params: list[int]) -> None:
    """Encode in memory, then write the bytes. Like PhotoLoader.load_image, this
    works with non-ASCII paths, which cv2.imwrite doesn't."""
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))
