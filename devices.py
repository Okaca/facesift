"""Decide where the models run: an NVIDIA GPU through CUDA when possible, else the CPU."""

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def select_providers(device: str = "auto") -> tuple[list[str], int]:
    """Return (providers, ctx_id) to hand to insightface for the requested device.

    device: "auto" tries CUDA and falls back to CPU, "gpu" tries CUDA only
    (check_active() then fails if it didn't load), "cpu" never touches the GPU.
    insightface forces CPU whenever ctx_id < 0, so CUDA needs ctx_id=0 (the first GPU).
    """
    if device == "cpu" or CUDA not in ort.get_available_providers():
        return [CPU], -1
    # load the CUDA/cuDNN libraries that pip installed into the venv, so no
    # system-wide CUDA Toolkit is needed
    ort.preload_dlls()
    return [CUDA, CPU], 0


def check_active(provider: str, device: str) -> str:
    """Describe the provider a model actually loaded with.

    Being listed as available only means onnxruntime was built with CUDA; if
    the driver or the CUDA libraries are missing it silently runs on CPU instead.
    """
    if device == "gpu" and provider != CUDA:
        raise RuntimeError(
            "--device gpu was requested but CUDA could not be loaded "
            "(see the onnxruntime error above). Use --device auto to fall back to CPU."
        )
    return "GPU (CUDA)" if provider == CUDA else "CPU"
