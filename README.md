# 🦶 Rate My Feet

> Disclaimer: this project idea was basically seen on social media, and I made it mostly for fun. I personally do not have any kind of fetish or anything of that nature — this is not a secret foot obsession, just a silly little experiment with AI and a bit of internet chaos. Also, no photo is being stored anywhere; this is just a browser-only fun app. Please enjoy it with the same energy I made it: a joke, a gimmick, and absolutely no weirdness.

**Live site: https://tenzin3.github.io/ratemyfeet/**

Upload a feet photo and get an instant score out of 10, a score for each photo feature, and tips to improve it. The AI runs **entirely in your browser**: your photo is never uploaded anywhere.

> 18+ only. Feet only, no nudity, no photos of other people or of minors. See the terms popup on the site.

---

## Features

- **Instant score** out of 10 with a label (*Needs work* → *Excellent*)
- **Five feature scores** with color-coded bars: Overall look, Sharpness, Lighting, Color, Framing
- **Tips** for any feature that scores below 6
- **Feet check**: photos without feet (cars, faces, screenshots…) are rejected instead of scored
- **Automatic clean-up**: fixes phone rotation and trims black bars from screenshots/video frames
- **Private by design**: no server, no uploads, no accounts, no tracking
- **"How it works" page** with a visual walkthrough of the pipeline
- Works on desktop and mobile (tested on desktop; newer iPhones expected to take ~1–2 s per photo)

---

## User guide

### 1. Agree to the terms
The first time you open the site, a popup asks you to confirm you're 18 or older and agree to the rules: only your own feet, no one under 18, no nudity. Tick the box and click **Continue**. Your answer is remembered for 30 days.

<img src="images/Authentication.png" alt="Terms popup: confirm you are 18 or older and agree to the rules" width="600">

### 2. Upload or take a photo
Drag a photo onto the box, click it to choose a file, or tap **📷 Take a photo**. On phones this opens your camera; on laptops it shows a live webcam preview.

<img src="images/First%20Page.png" alt="Home page with the upload box and the Take a photo button" width="600">

### 3. Check the preview and rate it
Your photo appears in the box. Click **Rate my photo**. On your first visit the AI downloads once (about 23 MB), then scoring takes a second or two.

<img src="images/Upload%20a%20feet%20pic.png" alt="A feet photo selected and ready to rate" width="600">

### 4. Read your result
You get an overall score out of 10 with a label and a one-line verdict, plus a colored bar for each of the five features. Any feature below 6 comes with a 💡 tip on how to improve it. Click **Try another photo** to start again.

<img src="images/Get%20Feet%20Rating.png" alt="Result page: overall score 6.1 (Good) with five feature scores and tips" width="700">

### 5. No feet? No score
Photos without feet are not scored. Here a car photo is rejected with a friendly message.

<p>
  <img src="images/Upload%20a%20car%20pic.png" alt="A car photo selected for rating" width="380">
  <img src="images/Good%20Try.png" alt="Result: We couldn't find any feet" width="380">
</p>

### 6. See how it works
Click **How it works?** at the top right for a visual walkthrough of every step, from the feet check to the five judges and how they're combined.

<img src="images/How%20it%20works.png" alt="How it works page: a flowchart of the scoring steps" width="600">

---

## Privacy: what the browser stores

Nothing about you or your photo is sent to a server. Here is everything that is stored, and where:

| What | Where | How long | Why |
|---|---|---|---|
| Your photo | Browser memory only | Until you leave or click "Try another photo" | To score and display it. Never uploaded, never written to disk. |
| AI model (~23 MB) | Browser cache (Cache Storage, managed by Transformers.js) | Until the browser clears it | So it only downloads on the first visit. Safari may clear it after ~7 days without a visit. |
| AI runtime files | Browser HTTP cache | Until the browser clears it | Same reason |
| Terms agreement | `localStorage` key `rmf_agreed_at` (a timestamp) | 30 days, then the popup shows again | So you don't see the popup every visit |

What gets downloaded on a first visit:
- The site itself from GitHub Pages (HTML, JS, and `embeddings.json`, about 100 KB)
- The AI runtime from the jsDelivr CDN
- The MobileCLIP-S0 image model from Hugging Face (`vision_model_fp16.onnx`, ~23 MB)

To wipe everything: clear the site data for `tenzin3.github.io` in your browser settings.

---

## How scoring works

```
Photo → tidy up (rotate, trim bars, shrink) → Is it feet? → 5 scores → weighted overall → result
```

**AI model:** [MobileCLIP-S0](https://huggingface.co/Xenova/mobileclip_s0) (Apple), a small CLIP model that understands images and text, run with [Transformers.js](https://github.com/huggingface/transformers.js) (ONNX Runtime, WebAssembly).

**Feet check:** the photo is compared with 17 descriptions (5 about feet, 12 about other things like "a car" or "a person's face"). If the feet descriptions together get less than 50% of the match, the photo is not scored.

| Score | Weight | How it's measured |
|---|---|---|
| Overall look | 35% | AI: "Good photo." vs "Bad photo." and "A beautiful photo." vs "An ugly photo." ([CLIP-IQA](https://arxiv.org/abs/2207.12396) method) |
| Sharpness | 25% | AI "Sharp photo." vs "Blurry photo.", averaged with edge strength (variance of the Laplacian) |
| Lighting | 15% | Pixels: average brightness, clipped black/white pixels, contrast |
| Color | 10% | Pixels: Hasler–Süsstrunk colorfulness |
| Framing | 15% | Pixels: black/white bars and very low resolution |

Labels: 8.5+ Excellent · 7+ Great · 5.5+ Good · 4+ Fair · below 4 Needs work.

**Text embeddings:** the 23 text prompts are converted to numbers once, ahead of time, and saved in `docs/embeddings.json`. That way visitors only download the image half of the model, not the 43 MB text half.

---

## Project structure

```
ratemyfeet/
├── docs/                    ← the live site (served by GitHub Pages)
│   ├── index.html           rating page (upload, terms popup, results)
│   ├── how.html             "How it works" visual page
│   ├── scoring.js           all scoring logic: model loading, feet check, the 5 scores
│   ├── embeddings.json      pre-computed text prompt embeddings (generated)
│   ├── make-embeddings.html developer tool that generates embeddings.json
│   └── .nojekyll            tells GitHub Pages to serve files as-is
├── images/                  screenshots used in this README
├── app.py                   older server version (Flask + PyTorch), for local experiments
├── templates/               HTML for the Flask version
└── requirements.txt         Python packages for the Flask version
```

---

## Local development

The site is plain static files, so any local web server works. (Opening `index.html` directly as a file won't work, because JavaScript modules need `http://`.)

```sh
python3 -m http.server 8000 -d docs
# open http://localhost:8000
```

Open the browser console to see the feet-check percentage and every score for each photo.

### Changing the prompts

If you edit `FEET_PROMPTS`, `OTHER_PROMPTS` or `QUALITY_PAIRS` in `docs/scoring.js`:

1. Open http://localhost:8000/make-embeddings.html and click **Create embeddings.json**
2. Move the downloaded file into `docs/`
3. Commit both files

If `embeddings.json` is missing or out of date, the site still works but downloads the 43 MB text model to compute it in the browser (a warning appears in the console).

### Changing weights, tips or formulas

All in `docs/scoring.js` (`FEATURES`, and the `*Score` functions). The numbers on `how.html` are written by hand, so update them there too.

### Flask version (optional)

The original server-based version uses larger models (NIMA + CLIP ViT-B/32):

```sh
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py            # http://localhost:5000  (or PORT=5001 python app.py)
```

---

## Deployment

GitHub Pages serves the `docs/` folder from the `main` branch (Settings → Pages → Deploy from a branch → `main` / `/docs`). Every push to `main` updates the live site within a minute or two. Hosting is free and there is no server to maintain.

---

## Development stages

| Stage | Status | What |
|---|---|---|
| 1. Prototype | ✅ Done | Flask app scoring photos with NIMA (photo aesthetics + technical quality) |
| 2. Friendly UI | ✅ Done | Plain-language scores, colored bars, tips, drag-and-drop upload, mobile layout |
| 3. Feet check | ✅ Done | CLIP zero-shot check rejects photos without feet |
| 4. Safety | ✅ Done | 18+ terms popup, no photo storage, clear rules |
| 5. "How it works" | ✅ Done | Visual pipeline page |
| 6. In-browser AI | ✅ Done | Rebuilt with MobileCLIP-S0 + Transformers.js; photos never leave the device; free hosting on GitHub Pages |
| 7. Real-device testing | 🔄 In progress | Test on iPhone/Android Safari & Chrome; tune score scaling (Overall look, Sharpness) on real photos |
| 8. Feet-specific score | 🔜 Next | Collect head-to-head votes ("which photo is better?") and train a small scorer on MobileCLIP embeddings, so the score judges the feet, not just the photo |
| 9. Smarter tips | 🔜 Later | Tips based on the actual cause (e.g. "image is too small, upload the original") instead of one fixed tip per feature |

---

## Known limitations

- **It rates the photo, not the feet (yet).** Lighting, sharpness and composition move the score; a pedicure doesn't.
- **Hand-tuned formulas.** Lighting penalizes intentionally bright photos; sharpness can be harsh on small or compressed images.
- **Feet check edge cases.** Feet in shoes or socks are usually rejected; unusual angles may be missed.
- **Scores are computed on the visitor's device**, so they can't be trusted for leaderboards or competitions.
- **First visit downloads ~23 MB**; use Wi-Fi on phones.

---

## Credits

- [MobileCLIP](https://github.com/apple/ml-mobileclip) by Apple; ONNX conversion [Xenova/mobileclip_s0](https://huggingface.co/Xenova/mobileclip_s0)
- [Transformers.js](https://github.com/huggingface/transformers.js) by Hugging Face
- CLIP-IQA: Wang et al., *Exploring CLIP for Assessing the Look and Feel of Images*, AAAI 2023
- Hasler & Süsstrunk, *Measuring Colourfulness in Natural Images*, 2003
- Flask version: [NIMA](https://arxiv.org/abs/1709.05424) via [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [CLIP](https://arxiv.org/abs/2103.00020)
