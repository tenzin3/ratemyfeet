"""Rate My Feet - minimal Flask app that scores an uploaded photo.

Scores (all 0-10):
  - Overall look : NIMA trained on AVA (photo aesthetics), via pyiqa
  - Sharpness    : NIMA trained on SPAQ (smartphone photo quality), via pyiqa
  - Lighting     : brightness / exposure, measured directly from the pixels
  - Color        : how vivid but natural the colors are (colorfulness metric)
  - Framing      : penalizes black/white bars and very low resolution

Note: none of these know anything about feet yet. They rate the *photo*.
"""
import base64
import io
import os

import numpy as np
import pyiqa
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms.functional import to_tensor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on {DEVICE} (first run downloads weights)...")
AESTHETIC = pyiqa.create_metric("nima", device=DEVICE)
TECHNICAL = pyiqa.create_metric("nima-spaq", device=DEVICE)
print("Models ready.")

# name, plain-language description, weight in the overall score, tip shown when the score is low
FEATURES = {
    "look":      ("Overall look", "How appealing the photo feels at a glance", 0.35,
                  "Keep it simple: one clear subject and an uncluttered background."),
    "sharpness": ("Sharpness", "Is it crisp and in focus?", 0.25,
                  "Hold the phone steady, tap to focus, and avoid zooming in or screenshots."),
    "lighting":  ("Lighting", "Not too dark, not washed out", 0.15,
                  "Use soft natural light, like near a window. Avoid harsh flash and dark rooms."),
    "color":     ("Color", "Vivid but natural colors", 0.10,
                  "Good light brings out color. Skip heavy filters."),
    "framing":   ("Framing", "Fills the shot, no black bars", 0.15,
                  "Upload the original photo, not a screenshot or video frame, and fill the frame."),
}


def clamp10(x: float) -> float:
    return round(float(max(0.0, min(10.0, x))), 1)


def to_ten(score: float) -> float:
    """Put a model's raw output on a 0-10 scale (models use 0-1, 0-10 or 0-100)."""
    if score <= 1.0:
        score *= 10
    elif score > 10:
        score /= 10
    return score


def trim_borders(img: Image.Image):
    """Remove flat black/white bars around the photo. Returns (trimmed image, fraction of area removed)."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape

    def flat(lines):  # a row/column that is almost uniform and very dark or very bright
        mean, std = lines.mean(axis=1), lines.std(axis=1)
        return ((mean < 25) | (mean > 235)) & (std < 12)

    rows, cols = flat(g), flat(g.T)
    top = int(np.argmin(rows)) if not rows.all() else 0
    bottom = h - int(np.argmin(rows[::-1])) if not rows.all() else h
    left = int(np.argmin(cols)) if not cols.all() else 0
    right = w - int(np.argmin(cols[::-1])) if not cols.all() else w

    if (bottom - top) < h * 0.5 or (right - left) < w * 0.5:  # don't over-trim (e.g. dark photos)
        return img, 0.0
    removed = 1 - ((bottom - top) * (right - left)) / (h * w)
    return img.crop((left, top, right, bottom)), removed


def lighting_score(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    mean = g.mean()
    clipped = ((g < 8) | (g > 247)).mean()  # share of pixels that are pure black/white
    contrast = g.std()
    score = 10 - abs(mean - 125) / 12 - clipped * 25
    if contrast < 35:  # flat, hazy look
        score -= (35 - contrast) / 7
    return clamp10(score)


def color_score(img: Image.Image) -> float:
    # Hasler & Suesstrunk colorfulness: ~0 grey, ~35 moderate, ~60 vivid, 100+ extreme
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    m = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    score = 10 * min(m / 60, 1.0)
    if m > 90:  # oversaturated / heavy filter
        score -= (m - 90) / 8
    return clamp10(score)


def framing_score(original: Image.Image, border_fraction: float) -> float:
    score = 10 - border_fraction * 40  # a bar covering 10% of the image costs 4 points
    short_side = min(original.size)
    if short_side < 600:
        score -= (600 - short_side) / 100
    return clamp10(score)


@torch.no_grad()
def score_image(img: Image.Image) -> dict:
    img = ImageOps.exif_transpose(img).convert("RGB")  # respect phone rotation
    original = img
    img, border_fraction = trim_borders(img)
    img.thumbnail((1024, 1024))  # keep inference fast on big phone photos
    x = to_tensor(img).unsqueeze(0).to(DEVICE)  # shape (1, 3, H, W), values 0-1

    raw_aes = AESTHETIC(x).item()
    raw_tech = TECHNICAL(x).item()
    print(f"raw scores -> nima(AVA): {raw_aes:.3f}  nima-spaq: {raw_tech:.3f}  borders removed: {border_fraction:.0%}")

    scores = {
        # NIMA-AVA bunches almost everything into 3-7, so stretch it around the middle
        "look": clamp10(5 + (raw_aes - 5) * 1.5),
        "sharpness": clamp10(to_ten(raw_tech)),
        "lighting": lighting_score(img),
        "color": color_score(img),
        "framing": framing_score(original, border_fraction),
    }
    overall = clamp10(sum(scores[k] * FEATURES[k][2] for k in FEATURES))

    features = [
        {"name": FEATURES[k][0], "desc": FEATURES[k][1], "score": s, "label": label(s),
         "tip": FEATURES[k][3] if s < 6 else None}
        for k, s in scores.items()
    ]
    return {"overall": overall, "label": label(overall), "headline": headline(overall), "features": features}


def label(s: float) -> str:
    if s >= 8.5: return "Excellent"
    if s >= 7: return "Great"
    if s >= 5.5: return "Good"
    if s >= 4: return "Fair"
    return "Needs work"


def headline(s: float) -> str:
    if s >= 8.5: return "Stunning shot. This one's a keeper."
    if s >= 7: return "Really nice photo, just a few small tweaks away from perfect."
    if s >= 5.5: return "Solid photo. A couple of easy fixes would make it pop."
    if s >= 4: return "Decent start. Check the tips below to level it up."
    return "This one needs some work. The tips below will help a lot."


@app.route("/", methods=["GET", "POST"])
def index():
    result, image_data, error = None, None, None

    if request.method == "POST":
        file = request.files.get("photo")
        if not file or file.filename == "":
            error = "Please choose a photo."
        else:
            raw = file.read()
            try:
                result = score_image(Image.open(io.BytesIO(raw)))
                image_data = f"data:{file.mimetype};base64,{base64.b64encode(raw).decode()}"
            except UnidentifiedImageError:
                error = "That file isn't an image we can read. Try a JPG or PNG."

    return render_template("index.html", result=result, image_data=image_data, error=error)


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="That image is too large (max 10 MB)."), 413


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
