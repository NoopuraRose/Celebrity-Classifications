from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        ROOT_DIR / "runs" / "classify" / "train" / "weights" / "best.pt",
    )
)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def load_model() -> YOLO:
    """Load the trained classification checkpoint once when the app starts."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Set MODEL_PATH to your trained .pt file."
        )
    return YOLO(str(MODEL_PATH))


model = load_model()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_data_url(image_bytes: bytes, filename: str) -> str:
    extension = filename.rsplit(".", 1)[1].lower()
    mime_type = "image/jpeg" if extension in {"jpg", "jpeg"} else f"image/{extension}"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def classify_image(image_path: Path) -> tuple[str, float, list[dict[str, Any]]]:
    prediction = model(str(image_path), verbose=False)[0]
    if prediction.probs is None:
        raise RuntimeError("The loaded model did not return classification probabilities.")

    names = prediction.names
    top_indices = prediction.probs.top5
    top_confidences = prediction.probs.top5conf
    top_predictions = [
        {
            "label": names[int(index)],
            "confidence": float(confidence),
        }
        for index, confidence in zip(top_indices, top_confidences)
    ]
    best = top_predictions[0]
    return best["label"], best["confidence"], top_predictions


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    image_preview = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or not uploaded_file.filename:
            error = "Choose an image before clicking Predict."
        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a JPG, PNG, WEBP, or BMP image."
        else:
            filename = secure_filename(uploaded_file.filename)
            image_bytes = uploaded_file.read()
            image_preview = image_data_url(image_bytes, filename)
            temporary_path: Path | None = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(filename).suffix.lower(), delete=False
                ) as temporary_file:
                    temporary_file.write(image_bytes)
                    temporary_path = Path(temporary_file.name)

                label, confidence, top_predictions = classify_image(temporary_path)
                result = {
                    "label": label,
                    "confidence": confidence,
                    "top_predictions": top_predictions,
                }
            except Exception as exc:  # Return a useful UI error instead of a server traceback.
                error = f"Prediction failed: {exc}"
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        error=error,
        result=result,
        image_preview=image_preview,
        model_path=MODEL_PATH.name,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="That file is too large. Please upload an image smaller than 10 MB.",
        result=None,
        image_preview=None,
        model_path=MODEL_PATH.name,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
