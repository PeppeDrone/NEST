# NEST project page

Static project page for *NEST: Controllable Neonatal Motion from a Single Frame and Sparse Movement
Specifications*, to be served at <https://peppedrone.github.io/NEST/>.

Plain HTML, CSS and a small script (no build step, no framework, no analytics). Layout and visual identity
follow the [Nerfies project page](https://github.com/nerfies/nerfies.github.io) (CC BY-SA 4.0); the footer
keeps the required attribution. Bulma is included locally (`static/css/bulma.min.css`); the only external
request is the Noto Sans web font from Google Fonts.

## Structure

```
index.html                 the page
static/css/                bulma.min.css (from the template), index.css (page styles)
static/js/index.js         restricted-asset placeholders, command selector, synchronised videos
static/images/             public figures (SVG), favicon, Open Graph image
static/restricted/         RESTRICTED demonstration assets: local preview only, git-ignored
tools/                     scripts that build the figures and check the numbers (not needed to serve the page)
```

## Local preview

The page uses relative paths, so it works both at the site root and under `/NEST/`. To preview it
exactly as on GitHub Pages:

```bash
mkdir -p /tmp/nest-preview && ln -sfn "$PWD" /tmp/nest-preview/NEST
cd /tmp/nest-preview && python3 -m http.server 8000
# open http://localhost:8000/NEST/
```

Opening `index.html` directly from disk also works, except that browsers may block video autoplay.

## Content sources and checks

* Every scientific statement comes from the manuscript (`NEST/manuscript/main.tex`).
* Every number on the page is wrapped as `<span class="num" data-key="MACRO">` with the LaTeX macro the
  manuscript prints. Verify all of them against the manuscript's generated values with:
  ```bash
  python3 tools/check_numbers.py --numbers ../manuscript/generated/numbers.json
  ```
* Public figures are the manuscript PDFs converted to SVG (`pdftocairo -svg`), unchanged. The method figure
  is rebuilt from the manuscript's TikZ code with the starting-pose drawing replaced by a schematic
  (`tools/build_method_figure.py`); its caption says so.

## Restricted assets

Assets derived from recordings of identifiable infants (poses extracted from clinical video, generated
trajectories anchored on them, reference stills, generated videos) are **not public** by default. They are
built locally into `static/restricted/` (git-ignored), and the page shows a labelled placeholder wherever
one is missing, so the deployed site never references a file that is not there.

| asset | content | page section |
|---|---|---|
| `restricted/videos/commands/*.mp4` (26) | generated 2D keypoints of one test infant under 13 commands, manual and automatic models | Teaser, Demonstrations |
| `restricted/images/fig4_single_infant.svg` | Fig. 4: generated trajectories of one test infant | Results D |
| `restricted/videos/stage2_sample.mp4` | control skeleton + generated video, face blurred | From trajectory to video |
| `restricted/images/fig10_stage2.webp` | Fig. 10: reference still and generated frames (face blurred), ankle trajectories | From trajectory to video |

Build them (requires the private NEST analysis outputs and Stage-2 run):

```bash
python3 tools/build_restricted_assets.py --stage1 <stage1 folder with analysis/artifacts> \
  --stage2-run <stage2/run folder> --figures ../manuscript/figures --stage2-clip <Fig. 10 clip id>
```

**To publish an asset after it has been explicitly cleared** (ethics/consent), move it from
`static/restricted/` to the same relative path, commit it, and adjust the `.gitignore` entry, e.g. by
listing only the files that remain restricted. Do not commit the whole folder.

## Deployment (GitHub Pages)

The page is published from this repository by `.github/workflows/pages.yml`, which uploads only
`index.html`, `.nojekyll` and `static/` (never `tools/`, this README or the rest of the repository) and refuses
to run if `static/restricted/` is present.

1. One-time setup: repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Every push to `main` that changes `project-page/` redeploys the site (or run the workflow manually from the
   **Actions** tab). It is served at <https://peppedrone.github.io/NEST/>.
3. Check that the placeholders appear (restricted assets are not in the repository) and that no request in
   the browser's network panel returns 404 for a public file.

## When the paper, code or models become public

Replace the remaining disabled button (Paper) in the hero section of `index.html` (`class="... is-unavailable"`,
`aria-disabled="true"`) with real links, update the author block, the availability table and the BibTeX
section. Keep these unavailable until the corresponding URL exists.
