"""Find faces in a photo and cut each one out as an aligned crop."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import cv2

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
        providers, ctx_id: Where to run the detector; see devices.select_providers().
    """

    def __init__(
        self,
        det_size: int = 640,
        det_thresh: float = 0.6,
        min_face_size: int = 40,
        crop_size: int = 112,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        ctx_id: int = -1,
    ) -> None:
        self.detector = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=list(providers),
        )
        self.detector.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.min_face_size = min_face_size
        self.crop_size = crop_size

    @property
    def provider(self) -> str:
        """Execution provider the detector actually loaded with."""
        return self.detector.models["detection"].session.get_providers()[0]

    def crop_faces(self, image: np.ndarray) -> list[FaceCrop]:
        """Return every usable face in a BGR image, largest first."""
        faces = self._detect(image)

        if not faces:
            """
            a face that fills the whole frame is too big for RetinaFace
            a black border makes it smaller relative to the frame and it gets found
            """
            faces = self._detect_padded(image)

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

    def _detect(self, image: np.ndarray) -> list:
        faces = self.detector.get(image)
        return [f for f in faces if self._short_side(f.bbox) >= self.min_face_size]

    def _detect_padded(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        pad = max(h, w) // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = self._detect(padded)
        for face in faces:
            # coordinates are in padded-image pixels; move them back to the original photo
            face.bbox -= pad
            face.kps -= pad
            face.bbox = np.clip(face.bbox, 0, [w, h, w, h])
        return faces

    @staticmethod
    def _short_side(bbox: np.ndarray) -> float:
        return min(bbox[2] - bbox[0], bbox[3] - bbox[1])
