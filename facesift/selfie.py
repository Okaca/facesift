import cv2
import numpy as np
from facesift.devices import select_providers
from facesift.face_cropper import FaceCropper
from facesift.face_encoder import FaceEncoder
from facesift.images import fit_within

MAX_SIDE = 1280


class SelfieEncoder:

    def __init__(self, device: str = "auto") -> None:
        providers, ctx_id = select_providers(device)
        self.cropper = FaceCropper(providers=providers, ctx_id=ctx_id)
        self.encoder = FaceEncoder(providers=providers, ctx_id=ctx_id)

    def embed(self, data: bytes) -> tuple[np.ndarray, list[float]] | None:
        selfie_array = np.frombuffer(data, dtype=np.uint8)
        selfie_BGR = cv2.imdecode(selfie_array, cv2.IMREAD_COLOR)

        if selfie_BGR is None:
            raise ValueError("Upload was not an image")

        image = fit_within(selfie_BGR, max_side=MAX_SIDE)
        faces = self.cropper.crop_faces(image)

        if not faces:
            return None

        face = faces[0]  # pick largest face

        embedding = self.encoder.encode([face.image])[0]  # result of one embedding

        height, width = image.shape[:2]
        box = [
            round(face.bbox[0] / width, 4),
            round(face.bbox[1] / height, 4),
            round(face.bbox[2] / width, 4),
            round(face.bbox[3] / height, 4),
        ]

        return embedding, box
