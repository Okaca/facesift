"""Turn aligned face crops into 512-d identity vectors with ArcFace."""

from collections.abc import Sequence
from pathlib import Path

import insightface
import numpy as np

DEFAULT_MODEL = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"


class FaceEncoder:
    """Computes L2-normalised ArcFace embeddings.

    The dot product of two embeddings is their cosine similarity:
    ~1 same image, >0.5 usually the same person, ~0 unrelated people.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        ctx_id: int = -1,
    ) -> None:
        self.model = insightface.model_zoo.get_model(str(model_path), providers=list(providers))
        self.model.prepare(ctx_id=ctx_id)  # -1 means force CPU

    @property
    def provider(self) -> str:
        """Execution provider the model actually loaded with."""
        return self.model.session.get_providers()[0]

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed aligned 112x112 BGR crops; returns an (N, 512) float32 array."""
        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        vectors = self.model.get_feat(crops)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.astype(np.float32)
