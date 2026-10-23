# PlotKit

CMS analysis plotting toolkit. A from-scratch redesign of the FLAF plotting code
([cms-flaf/FLAF#171](https://github.com/cms-flaf/FLAF/issues/171)).

- **matplotlib + mplhep by default** — no ROOT required, so it runs anywhere.
- **optional ROOT + [cmsstyle](https://github.com/cms-cat/cmsstyle) backend** — the same plot
  (y range, legend, labels and their sizes, CMS line) through the official CMS ROOT style.
- **drop-in for FLAF** — reads the existing `config/plot/*.yaml` files unchanged and keeps the
  legacy `Plotter` API, so replacing the old PlotKit submodule needs no analysis-side changes.
- **standalone** — render straight from a ROOT file with `uproot`, no FLAF needed.
- **extensible** — new plot styles (dataset comparison, 2D, …) and backends plug into a small
  registry without touching callers.

## Install

```sh
pip install -e .            # core (mplhep backend)
pip install -e .[cmsstyle]  # + ROOT/cmsstyle backend (needs ROOT)
```

In FLAF the package is vendored as the `FLAF/PlotKit` submodule and imported as
`FLAF.PlotKit` — no install step is required there.

## Standalone usage

Two ways to map the histograms in a file to process groups / labels / colours:

**1. Explicit process map** (the legacy `inputs.yaml` the analyses ship):

```sh
python -m PlotKit.cli --style stacked \
    --page-cfg   config/plot/cms_stacked.yaml \
    --page-cfg-custom config/plot/Run3_2022.yaml \
    --hist-cfg   config/plot/histograms.yaml \
    --inputs-cfg config/plot/inputs.yaml \
    --input merged.root --hist-name tau1_pt --output tau1_pt.pdf
```

**2. Auto-discovery from a real FLAF merged file.** FLAF merged files are per-variable and
nest histograms as `<channel>/<region>/<category>/<process>`. Point `--path` at the category
directory and PlotKit classifies the processes itself — observed data via `--data-name`
(defaults try `Data_Full`, `data`, `data_obs`), the signal family via `--signal-regex`
(matches are signals, never backgrounds), and everything else stacked as background.
`--signal-select` optionally narrows which signal mass points are drawn (the rest are
dropped, not demoted to background), and `--signal-scale` scales the overlaid signals — a
fixed factor (e.g. `100`) or **`bkg`** to normalise each signal's integral to the summed
background (handy for shape comparison; the legend then reads `… (norm. to bkg)`):

```sh
python -m PlotKit.cli \
    --page-cfg config/plot/cms_stacked.yaml \
    --page-cfg-custom config/plot/Run3_2023.yaml \
    --hist-cfg config/plot/histograms.yaml \
    --input MT2_bb.root --hist-name MT2_bb --path eMu/SR/res2b \
    --signal-regex 'XtoHHto' --signal-select 'XtoHHto2B2W_2L_(300|600|1000)$' \
    --signal-scale 100 --output MT2_bb.pdf
```

Select the ROOT backend with `--backend cmsstyle` or `PLOTKIT_BACKEND=cmsstyle`.

## Python API

```python
from PlotKit.Plotter import Plotter   # legacy-compatible facade (used by FLAF)
plotter = Plotter(page_cfg, page_cfg_custom, hist_cfg=hist_cfg_dict)
plotter.plot(hist_name, histograms, "out.pdf", want_data=True, custom=custom, scale=scale)
```

`histograms` maps `key -> (hist, label, color, group)` where `hist` is a ROOT `TH1`, an
uproot histogram or a `PlotKit.histogram.Hist1D`, and `group` is `backgrounds`, `signals` or
`data`. Colours and labels may use the legacy ROOT vocabulary (`kAzure-9`, `p_{T}`, `#tau`).

## Configuration (backward compatible)

| File | Role |
|---|---|
| `cms_stacked.yaml` | page/canvas style + per-process histogram styles |
| `<era>.yaml`       | lumi, channel and region labels |
| `histograms.yaml`  | per-variable binning, titles, log/blind flags |
| `inputs.yaml`      | (standalone only) process → group/label/colour map |

ROOT colour names, `TLatex` markup, font codes and draw options are translated to
matplotlib automatically (`PlotKit/rootcompat.py`); when ROOT is importable it is used as the
authoritative colour source.

### Optional style settings

All opt-in: a configuration that sets none of them plots as before (but see the CMS label line
below, which changes for everyone). A value of these settings that cannot be read -- `'45%'`
for a headroom, a `font` that is not a ROOT font code, `true` or `.inf` for a size -- raises a
`ValueError` naming the key, instead of falling back to the default. (Some older settings, e.g.
`signal_plot_scale`, still fall back silently: cms-flaf/PlotKit#9.) Both backends apply them
all. A `text_size` is a ROOT NDC size, as in the legacy configurations: cmsstyle draws it as
such, mplhep at 600 pt per unit (0.03 is 18 pt), clamped to 8-30 pt for a label and 8-20 pt for
the legend.

| Setting (`cms_stacked.yaml`) | Effect |
|---|---|
| `page_setup.label_lines: [[ana_text], [ch_text, cat_text], [customreg_text]]` | The labels top left, one line per entry: the boxes of an entry share a line (joined with `, `), in the `text_size` of its first box, bold if that box's `font` is a bold ROOT font (e.g. 62; bold also inside math, `#bf{}` with cmsstyle; other font codes are drawn in the default font). Each entry must be a non-empty list of box names declared in `page_setup.text_boxes`, or, without that list, of `ana_text`, `ch_text`, `cat_text` and `customreg_text`. Without `label_lines` every box is a line of its own, in the legend's `text_size` (16 pt when that is not set either), in both backends. |
| `legend.n_columns: 2` | Legend columns, a whole number >= 1, filled column by column with the top of the stack first; the two uncertainty-band entries (with the full uncertainty) are kept in one column. cmsstyle sizes the legend box from its widest entry and falls back to one column when the columns do not fit beside the frame. |
| `sgn_hist.show_yield: false` | Drop the `(yield N)` of each signal's legend entry (default `true`; `true` or `false`, not a quoted string). |
| `sgn_hist.norm_legend_note: "..."` | With `bkg` signal scaling, say it once, as a text-only legend entry, instead of `(norm. to bkg)` on every signal. |
| `page_setup.y_headroom: 0.4`, `page_setup.log_y_headroom: 0.45` | The fraction of the y axis (linear / log) kept free above the highest bin, for the legend and labels, instead of scaling the top by `max_y_sf`. On a log axis a fixed factor frees a share of the plot that depends on how many decades it spans; a headroom frees the same share on every plot. Also per variable in `histograms.yaml`. A variable's own setting beats the page's, and at the same level a headroom beats `max_y_sf`: the variable's headroom, then its `max_y_sf`, then the page's headroom, then the page's `max_y_sf`. |
| `page_setup.cms_label_fontsize: 20` | Base font size of the CMS line, in points (> 0) or a matplotlib name such as `large`: mplhep draws the scope at it, "CMS" at 1.3x and the lumi at 1/1.1x, so 20 gives 26 / 20 / 18.2 pt; cmsstyle draws "CMS" and the scope at the same sizes (its lumi keeps cmsstyle's). |

**The CMS label line, for every configuration:** the scope (`scope_text`, e.g. `Preliminary`)
is now drawn -- it is passed as `text=`, while mplhep 1.x ignores the former `label=`, which
dropped it -- as "Simulation <scope>" when no data is shown, in both backends, and after "CMS"
also for a "Private work" scope. Both backends draw it above the frame. "CMS", the scope and the
lumi share one baseline, and a line that would run into the lumi, or come closer to it than half
the scope's font size (a long scope, or "Simulation" added without data), is drawn smaller
until it fits, down to 60% of its size; below that it warns to set a smaller
`cms_label_fontsize`. Tested with **mplhep 1.1.3**, the version flaf_env pins (the floor
declared in `pyproject.toml`).

## Architecture / extension points

```
config.py      legacy YAML -> normalised PlotConfig
histogram.py   Hist1D (+ from_th1 / from_uproot / from_numpy)
rootcompat.py  ROOT colours / TLatex / fonts / binning -> matplotlib
spec.py        StackSpec: engine-neutral description of a plot
plotters/      build a spec from histograms + config   (style registry: STYLES)
backends.py    render a spec                            (mplhep | cmsstyle)
```

- **New plot style**: add a `BasePlotter` subclass that builds a spec, register it in
  `plotters/__init__.py:STYLES`, and add a `render_<style>` to the backends.
- **New backend**: add a `StyleBackend` subclass and register it in `backends.py:_BACKENDS`.
