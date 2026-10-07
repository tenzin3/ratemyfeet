# ratemyfeet

Run the app on port 5000 by default:

```sh
python app.py
```

If port 5000 is already in use, choose another port:

```sh
PORT=5001 python app.py
```

## Browser version (GitHub Pages)

The `docs/` folder is a version that runs entirely in the visitor's browser: no server, and photos never leave the device. GitHub Pages serves it at `https://tenzin3.github.io/ratemyfeet/`.

- `docs/index.html`: the rating page
- `docs/how.html`: the "How it works" page
- `docs/scoring.js`: all scoring (MobileCLIP-S0 via Transformers.js, plus lighting/color/framing/edge formulas)
- `docs/make-embeddings.html`: one-time tool that creates `docs/embeddings.json` (the prompts as numbers). Re-run it whenever you change the prompts in `scoring.js`.

Preview locally (pages must be served over http, not opened as files):

```sh
python3 -m http.server 8000 -d docs   # then open http://localhost:8000
```
