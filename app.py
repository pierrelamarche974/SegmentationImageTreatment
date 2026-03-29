from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

try:
    from api.cityscapes_utils import CityscapeUtils
except ImportError:  # When executed from the api/ directory.
    from cityscapes_utils import CityscapeUtils

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter

IMAGE_SIZE = (256, 512)
BASE_DIR = Path(__file__).resolve().parent
TFLITE_MODEL = BASE_DIR / "fpn_full.tflite"

app = FastAPI(
    title="Segmentation API",
    version="1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)



@lru_cache(maxsize=1)
def _load_interpreter():
    if not TFLITE_MODEL.exists():
        raise RuntimeError(f"Missing TFLite model: {TFLITE_MODEL}")
    interpreter = Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x[..., ::-1]


def _prepare_input(x: np.ndarray, input_details) -> np.ndarray:
    dtype = input_details[0]["dtype"]
    if dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        if scale:
            x = x / scale + zero_point
        return np.clip(x, 0, 255).astype(np.uint8)
    return x.astype(dtype)


def _resize_image(img_rgb: np.ndarray) -> np.ndarray:
    height, width = IMAGE_SIZE
    return np.array(Image.fromarray(img_rgb).resize((width, height)))


def _predict_mask(interpreter, input_details, output_details, img_rgb: np.ndarray) -> np.ndarray:
    resized = _resize_image(img_rgb)
    x = preprocess(resized)[np.newaxis]
    x = _prepare_input(x, input_details)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])
    return np.argmax(logits[0], axis=-1).astype(np.uint8)


def _blend(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = _resize_image(img_rgb).astype(np.float32)
    return ((1 - alpha) * base + alpha * CityscapeUtils.PALETTE_8[mask]).clip(0, 255).astype(
        np.uint8
    )


def _encode_png(arr: np.ndarray) -> str:
    image = Image.fromarray(arr)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type and file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG or PNG.")
    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    img_rgb = np.array(img)
    interpreter, input_details, output_details = _load_interpreter()
    mask = _predict_mask(interpreter, input_details, output_details, img_rgb)
    mask_rgb = CityscapeUtils.colorize_mask_8(mask)
    overlay = _blend(img_rgb, mask)

    return {
        "mask": _encode_png(mask_rgb),
        "overlay": _encode_png(overlay),
        "size": {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]},
        "classes": CityscapeUtils.MAIN_CLASSES,
    }
