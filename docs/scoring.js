// Rate My Feet: all scoring runs in the visitor's browser. The photo never leaves their device.
//
// AI model: MobileCLIP-S0 (Apple), run with Transformers.js. Only the image half of the model is
// downloaded (~23 MB, cached by the browser). The text half is pre-computed into embeddings.json
// by make-embeddings.html, so visitors never download it.
//
// Scores (all 0-10):
//   Overall look : MobileCLIP judges "Good photo." vs "Bad photo." etc. (the CLIP-IQA method)
//   Sharpness    : average of MobileCLIP "Sharp photo." vs "Blurry photo." and a measured edge score
//   Lighting     : brightness / exposure, measured from the pixels
//   Color        : how vivid but natural the colors are (Hasler-Suesstrunk colorfulness)
//   Framing      : penalizes black/white bars and very low resolution

export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
export const MODEL_ID = "Xenova/mobileclip_s0";
// Image model: fp16 (~23 MB). The smaller 8-bit version (q8) is NOT accurate enough for images:
// it scored an obvious feet photo as only ~50% feet, while fp16 matches full precision (~98%).
const VISION_OPTIONS = { dtype: "fp16", device: "wasm" };
// Text model: 8-bit is fine (only used once by make-embeddings.html, or as a fallback).
const TEXT_OPTIONS = { dtype: "q8", device: "wasm" };

// ---------- Prompts ----------
export const FEET_PROMPTS = [
  "a photo of feet", "a photo of bare feet", "a close-up photo of a foot",
  "a photo of toes", "a photo of feet with painted toenails",
];
export const OTHER_PROMPTS = [
  "a photo of a car", "a photo of a person's face", "a photo of hands", "a photo of an animal",
  "a photo of food", "a photo of a landscape", "a photo of a building", "a screenshot of text",
  "a photo of shoes", "a photo of a room", "a photo of an object", "a photo of a person's body",
];
export const FEET_THRESHOLD = 0.5;

// [positive, negative] pairs. Score = how strongly the photo matches the positive side.
export const QUALITY_PAIRS = {
  look: [["Good photo.", "Bad photo."], ["A beautiful photo.", "An ugly photo."]],
  sharpness: [["Sharp photo.", "Blurry photo."]],
};

export const ALL_PROMPTS = [
  ...FEET_PROMPTS, ...OTHER_PROMPTS,
  ...QUALITY_PAIRS.look.flat(), ...QUALITY_PAIRS.sharpness.flat(),
];

// name, weight in the overall score, tip shown when the score is below 6
export const FEATURES = {
  look:      ["Overall look", 0.35, "Keep it simple: one clear subject and an uncluttered background."],
  sharpness: ["Sharpness",    0.25, "Hold the phone steady, tap to focus, and avoid zooming in or screenshots."],
  lighting:  ["Lighting",     0.15, "Use soft natural light, like near a window. Avoid harsh flash and dark rooms."],
  color:     ["Color",        0.10, "Good light brings out color. Skip heavy filters."],
  framing:   ["Framing",      0.15, "Upload the original photo, not a screenshot or video frame, and fill the frame."],
};

// ---------- Model loading ----------
let T = null, processor = null, vision = null, textEmbeds = null;

async function lib() {
  if (!T) {
    T = await import(TRANSFORMERS_URL);
    T.env.allowLocalModels = false; // always load from the Hugging Face Hub
  }
  return T;
}

/** Downloads (or reads from cache) everything needed. onProgress gets 0-100. */
export async function loadModels(onProgress = () => {}) {
  const t = await lib();
  const files = {};
  const progress_callback = (e) => {
    if (e.status === "progress" && e.total) {
      files[e.file] = [e.loaded, e.total];
      const [l, tot] = Object.values(files).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      onProgress(Math.round((100 * l) / tot));
    }
  };
  processor ??= await t.AutoProcessor.from_pretrained(MODEL_ID);
  vision ??= await t.CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { ...VISION_OPTIONS, progress_callback });
  textEmbeds ??= await loadTextEmbeds(progress_callback);
  onProgress(100);
}

async function loadTextEmbeds(progress_callback) {
  try {
    const res = await fetch("embeddings.json", { cache: "no-cache" });
    if (res.ok) {
      const data = await res.json();
      if (data.model === MODEL_ID && JSON.stringify(data.prompts) === JSON.stringify(ALL_PROMPTS)) return data.embeds;
      console.warn("embeddings.json is out of date (prompts changed). Re-run make-embeddings.html.");
    }
  } catch (e) { /* fall through */ }
  console.warn("Computing text embeddings in the browser (downloads the ~43 MB text model). Run make-embeddings.html to avoid this.");
  return computeTextEmbeds(progress_callback);
}

/** Used by make-embeddings.html, and as a fallback if embeddings.json is missing. */
export async function computeTextEmbeds(progress_callback) {
  const t = await lib();
  const tokenizer = await t.AutoTokenizer.from_pretrained(MODEL_ID);
  const textModel = await t.CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { ...TEXT_OPTIONS, progress_callback });
  const inputs = tokenizer(ALL_PROMPTS, { padding: "max_length", truncation: true });
  const { text_embeds } = await textModel(inputs);
  return text_embeds.normalize().tolist();
}

async function imageEmbed(canvas) {
  const t = await lib();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  const image = await t.RawImage.fromBlob(blob);
  const inputs = await processor(image);
  const { image_embeds } = await vision(inputs);
  return image_embeds.normalize().tolist()[0];
}

// ---------- Small math helpers ----------
const clamp10 = (x) => Math.round(Math.max(0, Math.min(10, x)) * 10) / 10;
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
function softmax(xs) {
  const m = Math.max(...xs), e = xs.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
function meanStd(arr) {
  let s = 0, s2 = 0;
  for (const v of arr) { s += v; s2 += v * v; }
  const m = s / arr.length;
  return [m, Math.sqrt(Math.max(0, s2 / arr.length - m * m))];
}

/** Probabilities over a set of prompts (CLIP zero-shot: softmax of 100 x cosine similarity). */
function probs(img, prompts) {
  return softmax(prompts.map((p) => 100 * dot(img, textEmbeds[ALL_PROMPTS.indexOf(p)])));
}
const pairScore = (img, pairs) => pairs.reduce((s, pair) => s + probs(img, pair)[0], 0) / pairs.length;

// ---------- Image preparation ----------
const WORK_SIZE = 512; // longest side used for all measurements

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h));
  return c;
}

export function grayscale(rgba) {
  const g = new Float32Array(rgba.length / 4);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) g[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  return g;
}

/** Finds flat black/white bars around the photo. Returns crop box and fraction of area removed. */
export function findBorders(gray, w, h) {
  const flatRow = (y) => { const [m, s] = meanStd(gray.subarray(y * w, (y + 1) * w)); return (m < 25 || m > 235) && s < 12; };
  const col = new Float32Array(h);
  const flatCol = (x) => { for (let y = 0; y < h; y++) col[y] = gray[y * w + x]; const [m, s] = meanStd(col); return (m < 25 || m > 235) && s < 12; };

  let top = 0, bottom = h, left = 0, right = w;
  while (top < h && flatRow(top)) top++;
  if (top === h) return { x: 0, y: 0, w, h, removed: 0 }; // whole image is flat
  while (bottom > top && flatRow(bottom - 1)) bottom--;
  while (left < w && flatCol(left)) left++;
  while (right > left && flatCol(right - 1)) right--;

  if (bottom - top < h * 0.5 || right - left < w * 0.5) return { x: 0, y: 0, w, h, removed: 0 }; // don't over-trim
  return { x: left, y: top, w: right - left, h: bottom - top, removed: 1 - ((bottom - top) * (right - left)) / (w * h) };
}

/** Loads the file, fixes rotation, trims bars and shrinks it. */
export async function prepare(file) {
  let bitmap;
  try { bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }); }
  catch (e) { bitmap = await createImageBitmap(file); }
  const W = bitmap.width, H = bitmap.height;
  const s = Math.min(1, WORK_SIZE / Math.max(W, H));
  const full = makeCanvas(W * s, H * s);
  full.getContext("2d").drawImage(bitmap, 0, 0, full.width, full.height);
  bitmap.close?.();

  const data = full.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, full.width, full.height).data;
  const box = findBorders(grayscale(data), full.width, full.height);
  const work = makeCanvas(box.w, box.h);
  work.getContext("2d").drawImage(full, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
  return { work, shortSide: Math.min(W, H), borderFraction: box.removed };
}

// ---------- Measured scores ----------
export function lightingScore(gray) {
  const [mean, contrast] = meanStd(gray);
  let clipped = 0;
  for (const v of gray) if (v < 8 || v > 247) clipped++;
  clipped /= gray.length;
  let score = 10 - Math.abs(mean - 125) / 12 - clipped * 25;
  if (contrast < 35) score -= (35 - contrast) / 7; // flat, hazy look
  return clamp10(score);
}

export function colorScore(rgba) {
  // Hasler & Suesstrunk colorfulness: ~0 grey, ~35 moderate, ~60 vivid, 100+ extreme
  const n = rgba.length / 4, rg = new Float32Array(n), yb = new Float32Array(n);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    rg[j] = rgba[i] - rgba[i + 1];
    yb[j] = 0.5 * (rgba[i] + rgba[i + 1]) - rgba[i + 2];
  }
  const [mrg, srg] = meanStd(rg), [myb, syb] = meanStd(yb);
  const m = Math.hypot(srg, syb) + 0.3 * Math.hypot(mrg, myb);
  let score = 10 * Math.min(m / 60, 1);
  if (m > 90) score -= (m - 90) / 8; // oversaturated / heavy filter
  return clamp10(score);
}

export function framingScore(borderFraction, shortSide) {
  let score = 10 - borderFraction * 40; // a bar covering 10% of the image costs 4 points
  if (shortSide < 600) score -= (600 - shortSide) / 100;
  return clamp10(score);
}

/** Edge strength (variance of the Laplacian): blurry photos have weak edges. */
export function edgeScore(gray, w, h) {
  const lap = new Float32Array((w - 2) * (h - 2));
  let k = 0;
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      lap[k++] = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
    }
  const [, sd] = meanStd(lap);
  const variance = sd * sd;
  // ~20 or less = very blurry (0), ~150 = okay (5), ~1000+ = crisp (10)
  return clamp10(((Math.log10(variance + 1) - 1.3) / (3.0 - 1.3)) * 10);
}

// ---------- Labels ----------
export function label(s) {
  if (s >= 8.5) return "Excellent";
  if (s >= 7) return "Great";
  if (s >= 5.5) return "Good";
  if (s >= 4) return "Fair";
  return "Needs work";
}
export function headline(s) {
  if (s >= 8.5) return "Stunning shot. This one's a keeper.";
  if (s >= 7) return "Really nice photo, just a few small tweaks away from perfect.";
  if (s >= 5.5) return "Solid photo. A couple of easy fixes would make it pop.";
  if (s >= 4) return "Decent start. Check the tips below to level it up.";
  return "This one needs some work. The tips below will help a lot.";
}

// ---------- Main entry point ----------
/** Scores a photo File. Returns { notFeet: true } or { overall, label, headline, features: [...] }. */
export async function scorePhoto(file) {
  const { work, shortSide, borderFraction } = await prepare(file);
  const rgba = work.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, work.width, work.height).data;
  const gray = grayscale(rgba);

  const img = await imageEmbed(work);
  const p = probs(img, [...FEET_PROMPTS, ...OTHER_PROMPTS]);
  const feet = p.slice(0, FEET_PROMPTS.length).reduce((a, b) => a + b, 0);
  console.log(`feet check -> ${(feet * 100).toFixed(0)}% feet`);
  if (feet < FEET_THRESHOLD) return { notFeet: true };

  const scores = {
    look: clamp10(10 * pairScore(img, QUALITY_PAIRS.look)),
    sharpness: clamp10((10 * pairScore(img, QUALITY_PAIRS.sharpness) + edgeScore(gray, work.width, work.height)) / 2),
    lighting: lightingScore(gray),
    color: colorScore(rgba),
    framing: framingScore(borderFraction, shortSide),
  };
  console.log("scores", scores, { borderFraction });

  const overall = clamp10(Object.entries(scores).reduce((s, [k, v]) => s + v * FEATURES[k][1], 0));
  return {
    overall, label: label(overall), headline: headline(overall),
    features: Object.entries(scores).map(([k, v]) => ({
      name: FEATURES[k][0], score: v, label: label(v), tip: v < 6 ? FEATURES[k][2] : null,
    })),
  };
}
