"""Rate My Feet - minimal Flask app that scores an uploaded photo with NIMA.

Models (loaded once at startup via pyiqa; weights download automatically on first run):
  - "nima"      : NIMA trained on AVA        -> photo aesthetics (1-10)
  - "nima-spaq" : NIMA trained on SPAQ       -> technical quality / sharpness

Note: neither model knows anything about feet. They score the *photo*.
"""
import base64
import io

import pyiqa
import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import to_tensor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on {DEVICE} (first run downloads weights)...")
AESTHETIC = pyiqa.create_metric("nima", device=DEVICE)
TECHNICAL = pyiqa.create_metric("nima-spaq", device=DEVICE)
print("Models ready.")


def to_ten(score: float) -> float:
    """Put a model's raw output on a 0-10 scale (models use 0-1, 0-10 or 0-100)."""
    if score <= 1.0:
        score *= 10
    elif score > 10:
        score /= 10
    return round(max(0.0, min(10.0, score)), 1)


@torch.no_grad()
def score_image(img: Image.Image) -> dict:
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))  # keep inference fast on big phone photos
    x = to_tensor(img).unsqueeze(0).to(DEVICE)  # shape (1, 3, H, W), values 0-1

    aesthetic = to_ten(AESTHETIC(x).item())
    technical = to_ten(TECHNICAL(x).item())
    overall = round(0.6 * aesthetic + 0.4 * technical, 1)
    return {"overall": overall, "aesthetic": aesthetic, "technical": technical}


@app.route("/", methods=["GET", "POST"])
def index():
    scores, image_data, error = None, None, None

    if request.method == "POST":
        file = request.files.get("photo")
        if not file or file.filename == "":
            error = "Please choose a photo."
        else:
            raw = file.read()
            try:
                img = Image.open(io.BytesIO(raw))
                scores = score_image(img)
                image_data = f"data:{file.mimetype};base64,{base64.b64encode(raw).decode()}"
            except UnidentifiedImageError:
                error = "That file isn't an image we can read (try JPG or PNG)."

    return render_template("index.html", scores=scores, image_data=image_data, error=error)


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="Image is too large (max 10 MB)."), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
