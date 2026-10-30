"""Find faces in a photo and cut each one out as an aligned crop."""

import warnings
from dataclasses import dataclass, field

import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# insightface's alignment code calls a scikit-image function that is being deprecated
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


@dataclass
class FaceCrop:
    face_index: int  # position within its photo, 0 = largest face
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in photo pixels
    det_score: float  # detector confidence, 0..1
    landmarks: np.ndarray = field(repr=False)  # (5, 2): eyes, nose, mouth corners
    image: np.ndarray = field(repr=False)  # aligned BGR crop, crop_size x crop_size


class FaceCropper:
    """Detects faces with RetinaFace and returns aligned crops.

    Args:
        det_size: Resolution the photo is scaled to for detection. Larger finds
            smaller faces (back rows of a group photo) but is slower.
        det_thresh: Minimum detector confidence to keep a face.
        min_face_size: Skip faces whose shorter side is below this many pixels.
        crop_size: Side of the aligned crop. ArcFace expects 112.
    """

    def __init__(
        self,
        det_size: int = 640,
        det_thresh: float = 0.6,
        min_face_size: int = 40,
        crop_size: int = 112,
    ) -> None:
        self.detector = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self.detector.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.min_face_size = min_face_size
        self.crop_size = crop_size

    def crop_faces(self, image: np.ndarray) -> list[FaceCrop]:
        """Return every usable face in a BGR image, largest first."""
        faces = self.detector.get(image)
        faces = [f for f in faces if self._short_side(f.bbox) >= self.min_face_size]
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        crops = []
        for i, face in enumerate(faces):
            aligned = face_align.norm_crop(image, landmark=face.kps, image_size=self.crop_size)
            x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
            crops.append(
                FaceCrop(
                    face_index=i,
                    bbox=(x1, y1, x2, y2),
                    det_score=float(face.det_score),
                    landmarks=face.kps,
                    image=aligned,
                )
            )
        return crops

    @staticmethod
    def _short_side(bbox: np.ndarray) -> float:
        return min(bbox[2] - bbox[0], bbox[3] - bbox[1])
