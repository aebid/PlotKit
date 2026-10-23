"""Rendering backends.

A backend takes an engine-neutral :class:`~PlotKit.spec.StackSpec` and draws it.  Two are
provided, matching issue #171's requirement to "support both mplhep and cmsstyle":

* :class:`MplhepBackend` -- matplotlib + mplhep.  The **default**; pure Python, no ROOT, so
  it works standalone and is the transparent replacement for the legacy ROOT renderer.
* :class:`CmsstyleBackend` -- ROOT + the official ``cmsstyle`` package.  Selected with
  ``backend: cmsstyle`` (or ``PLOTKIT_BACKEND=cmsstyle``).  ``cmsstyle`` is imported lazily;
  if it (or ROOT) is unavailable the backend warns and falls back to mplhep.

New plot styles add a ``render_<style>`` method here (and a plotter that builds the spec).
"""

from __future__ import annotations

import re
import warnings

import numpy as np

from . import rootcompat as rc
from .spec import StackSpec

# Text in a StackSpec is kept as raw ROOT TLatex; ``_t`` converts it for matplotlib.
_t = rc.tlatex_to_mpl


def _check_headroom(name: str, headroom) -> None:
    """A headroom is the fraction of the axis kept free, so strictly between 0 and 1."""
    if headroom is not None and not 0 < headroom < 1:
        raise ValueError(
            f"{name} = {headroom!r}: the fraction of the y axis kept free above the "
            "highest bin must be strictly between 0 and 1"
        )


def _y_max(spec, y_max: float) -> float:
    """Top of a linear y axis: y_headroom of its height free above y_max, else max_y_sf."""
    headroom = spec.y_headroom
    _check_headroom("y_headroom", headroom)
    if headroom is None:
        return y_max * spec.max_y_sf
    return spec.y_min + (y_max - spec.y_min) / (1.0 - headroom)


def _y_highest(spec) -> float:
    """The highest point the y axis has to show, for both backends: the top of the
    uncertainty band, every signal (bkg-normalised curves can peak far above the stack) and
    the data with their errors."""
    highest = [1e-9]
    if spec.bkg_total is not None:
        band = _band_hist(spec)
        highest.append(float(np.max(band.values + band.errors)))
    highest += [float(np.max(e.hist.values)) for e in spec.signals]
    if spec.data is not None:
        highest.append(float(np.max(spec.data.hist.values + spec.data.hist.errors)))
    return max(highest)


def _legend_entries_mplhep_order(backgrounds, others):
    """The legend entries in the order the mplhep backend lists them: the stack top first
    (mplhep reverses the stacked labels), then everything else as drawn."""
    return list(reversed(backgrounds)) + list(others)


def _column_major(entries, ncol, blank):
    """``entries`` laid out as matplotlib fills a legend -- column by column, the first
    ``len % ncol`` columns one entry longer (``np.array_split``) -- in the row-by-row order a
    multi-column TLegend is filled, empty cells as ``blank``."""
    columns = [list(c) for c in np.array_split(np.arange(len(entries)), ncol) if len(c)]
    rows = max((len(c) for c in columns), default=0)
    return [
        entries[c[r]] if r < len(c) else blank for r in range(rows) for c in columns
    ]


def _keep_together(entries, first, count, ncol, blank):
    """``entries`` with blank cells (``blank``) inserted before ``entries[first]`` so that
    the ``count`` entries from there -- the two uncertainty bands -- share one legend column
    when the columns are filled as matplotlib fills them (``np.array_split``). Unchanged when
    they already do, or when they cannot: one column, one entry, or more than a column holds.
    """
    if ncol < 2 or count < 2:
        return list(entries)
    out = list(entries)
    for _ in range(len(entries) + ncol):
        columns = np.array_split(np.arange(len(out)), ncol)
        column_of = {int(i): c for c, cells in enumerate(columns) for i in cells}
        if column_of[first] == column_of[first + count - 1]:
            return out
        out.insert(first, blank)
        first += 1
    return list(entries)


# A ROOT NDC text size times this is the point size the mplhep backend draws it at
# (plotters/stacked.py's mapping of the configurations' text_size), so the cmsstyle backend
# divides by it to draw a configured size in ROOT.
_PT_PER_NDC = 600.0

# PlotKit's TLatex dialect beyond ROOT's: tokens and characters the mplhep backend renders
# (through rootcompat) that ROOT TLatex prints literally or as mojibake, with their ROOT form.
_ROOT_TOKENS = {"#ell": "#font[12]{l}"}
_ROOT_CHARS = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "−": "#minus",
    "→": "#rightarrow",
    "×": "#times",
    "±": "#pm",
    "µ": "#mu",
    "μ": "#mu",
    "ℓ": "#font[12]{l}",  # script small l
}


def _root_text(text) -> str:
    """A label in the ROOT TLatex the cmsstyle backend can draw: PlotKit's extra tokens and
    characters translated (_ROOT_TOKENS, _ROOT_CHARS). Any other non-ASCII character raises,
    naming the label: ROOT would draw it as mojibake."""
    s = "" if text is None else str(text)
    for token, root in _ROOT_TOKENS.items():
        s = s.replace(token, root)
    s = "".join(_ROOT_CHARS.get(c, c) for c in s)
    unknown = sorted({c for c in s if ord(c) > 0x7F})
    if unknown:
        raise ValueError(
            f"label {text!r}: {', '.join(f'U+{ord(c):04X} {c!r}' for c in unknown)} has no "
            "ROOT TLatex form for the cmsstyle backend; write it in TLatex (e.g. #mu, "
            "#rightarrow)"
        )
    return s


def _font_points(size) -> float:
    """A font size as the mplhep backend draws it, in points: a number, or a named size
    ("large") relative to the CMS style's base font."""
    if isinstance(size, (int, float)):
        return float(size)
    import matplotlib
    import mplhep as hep
    from matplotlib.font_manager import FontProperties

    with matplotlib.rc_context(hep.style.CMS):
        return FontProperties(size=size).get_size_in_points()


def _scope_line(spec) -> str:
    """The text after "CMS": the scope, as "Simulation <scope>" when no data is shown --
    built as mplhep's ``cms.label`` builds it, so both backends write the same."""
    scope = spec.scope_text or ""
    if not spec.is_data:
        scope = f"Simulation {scope}"
    return " ".join(scope.split())


def _align_cms_label(ax, texts) -> None:
    """Put the CMS label line -- "CMS", the scope and the lumi, the texts ``hep.cms.label``
    returned -- on one baseline.

    mplhep places "CMS" and the lumi by the bottom of their text box but the scope
    ("Preliminary") by its baseline (and its y by the font's descent), so the scope sits lower
    than "CMS". All three go on the top edge of the axes and on the baseline here, raised a
    quarter of the largest font so descenders clear the frame.
    """
    import matplotlib.transforms as mtransforms

    texts = [t for t in texts if t is not None and t.get_text()]
    if not texts:
        return
    raised = mtransforms.offset_copy(
        ax.transAxes,
        fig=ax.figure,
        y=0.25 * max(t.get_fontsize() for t in texts),
        units="points",
    )
    for t in texts:
        t.set_y(1.0)
        t.set_verticalalignment("baseline")
        t.set_transform(raised)


# The CMS line is shrunk to fit beside the lumi down to this fraction of its size, no further.
_CMS_LABEL_MIN_SCALE = 0.6


def _renderer(fig):
    """The figure's renderer. Only Agg-type canvases have ``get_renderer``; the pdf, svg and
    ps ones that _new_figure keeps do not, so fall back to the figure's, as mplhep does.
    """
    canvas = fig.canvas
    if hasattr(canvas, "get_renderer"):
        return canvas.get_renderer()
    return fig._get_renderer()


def _cms_label_overlap(ax, cms, scope, lumi) -> float:
    """By how many pixels "CMS <scope>" runs into the lumi, a small gap (half the scope's
    font) included; <= 0 when the line fits."""
    if lumi is None or not lumi.get_text():
        return 0.0
    renderer = _renderer(ax.figure)
    left = [t for t in (cms, scope) if t is not None and t.get_text()]
    if not left:
        return 0.0
    right_edge = max(t.get_window_extent(renderer).x1 for t in left)
    gap = renderer.points_to_pixels(0.5 * left[-1].get_fontsize())
    return right_edge + gap - lumi.get_window_extent(renderer).x0


def _cms_label_width(ax, texts) -> float:
    """The width in pixels of the texts of the CMS line, the room they take together."""
    renderer = _renderer(ax.figure)
    return sum(
        t.get_window_extent(renderer).width
        for t in texts
        if t is not None and t.get_text()
    )


def _log_y_max(spec, y_max: float) -> float:
    """Top of a log y axis: log_y_headroom of its height free above y_max, else max_y_sf."""
    headroom = spec.log_y_headroom
    _check_headroom("log_y_headroom", headroom)
    if headroom is None:
        return y_max * spec.max_y_sf
    # The free share is of at least one decade above y_min_log, so a nearly empty plot still
    # gets room, and the top changes smoothly with y_max instead of jumping at y_min_log.
    low = np.log10(spec.y_min_log)
    high = max(np.log10(y_max) if y_max > 0 else low, low + 1.0)
    return float(10 ** (low + (high - low) / (1.0 - headroom)))


# A $...$ math segment of a converted label; an escaped \$ is literal text, not a delimiter.
_MATH_SEGMENT = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")


def _bold_mathtext(text: str) -> str:
    """A converted label made bold inside its math too: mathtext ignores fontweight.

    Each $...$ segment is bolded on its own -- a label joined from several boxes can hold
    more than one. If the result does not parse, the label is returned unbolded (its plain
    text is still drawn in the bold weight), so bolding can never abort a plot.
    """
    bold = _MATH_SEGMENT.sub(
        lambda m: r"$\mathbf{" + m.group(1).replace(r"\mathrm{", r"\mathbf{") + "}$",
        text,
    )
    return bold if bold == text or rc._mathtext_parses(bold) else text


def _want_ratio(spec) -> bool:
    """A ratio panel needs a stack plus something to show in it: data points, or -- when the
    full (stat + syst) uncertainty was computed -- the relative uncertainty band alone.
    """
    if not spec.draw_ratio or spec.bkg_total is None:
        return False
    return spec.data is not None or spec.bkg_total_unc is not None


def _band_hist(spec):
    """The histogram whose errors the outer uncertainty band shows: the full one when
    available, else the stat-only stack error."""
    return spec.bkg_total_unc if spec.bkg_total_unc is not None else spec.bkg_total


def _bands(spec):
    """``(histogram, style)`` pairs for the uncertainty bands, **widest first**.

    Drawing order matters: the narrower band goes on top so it stays visible.  Without the
    full uncertainty there is a single, stat-only band (unchanged legacy behaviour); with it
    the outer band is stat + syst and the inner one is stat alone, so the systematic
    contribution is the region covered by the outer hatch only.
    """
    if spec.bkg_total is None or spec.unc_band is None:
        return []
    bands = [(_band_hist(spec), spec.unc_band)]
    if spec.bkg_total_unc is not None and spec.stat_band is not None:
        bands.append((spec.bkg_total, spec.stat_band))
    return bands


def _relative_band(hist):
    """A histogram flat at 1 whose errors are ``hist``'s relative uncertainties."""
    from .histogram import Hist1D

    with np.errstate(divide="ignore", invalid="ignore"):
        safe = hist.values > 0
        rel = np.where(safe, hist.errors / np.where(safe, hist.values, 1), 0.0)
    return Hist1D(
        edges=hist.edges,
        values=np.ones_like(rel),
        variances=rel**2,
        name="rel_unc",
    )


def _parse_lumi_energy(lumi_text):
    """Extract ``(lumi, energy)`` numbers from a legacy lumi string for cmsstyle."""
    lumi = re.search(r"([\d.]+)\s*fb", str(lumi_text))
    energy = re.search(r"([\d.]+)\s*TeV", str(lumi_text))
    return (
        float(lumi.group(1)) if lumi else None,
        float(energy.group(1)) if energy else None,
    )


class StyleBackend:
    """Abstract backend.  Subclasses implement ``render_stacked`` (and future styles)."""

    name = "base"

    def render_stacked(
        self, spec: StackSpec, output_file: str
    ) -> None:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# matplotlib / mplhep
# --------------------------------------------------------------------------- #
class MplhepBackend(StyleBackend):
    name = "mplhep"

    def _new_figure(self, spec):
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplhep as hep

        plt.style.use(hep.style.CMS)
        want_ratio = _want_ratio(spec)
        if want_ratio:
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=spec.canvas_size,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
            )
        else:
            fig, ax = plt.subplots(figsize=spec.canvas_size)
            rax = None
        return plt, hep, fig, ax, rax

    @staticmethod
    def _step_band(ax, edges, lo, hi, band):
        ax.fill_between(
            edges,
            np.append(lo, lo[-1]),
            np.append(hi, hi[-1]),
            step="post",
            facecolor="none" if band.hatch else band.color,
            hatch=band.hatch,
            edgecolor=band.color,
            linewidth=0.0,
            alpha=band.alpha,
        )

    def _cms_label(self, hep, ax, spec):
        """The CMS line: "CMS <scope>" on the left of the axes top, the lumi on the right.

        A line that would run into the lumi -- a long scope, or "Simulation" prepended without
        data, in mplhep's default sizes -- is redrawn smaller until it fits, keeping mplhep's
        proportions, down to _CMS_LABEL_MIN_SCALE of its size; below that it warns.
        """
        import matplotlib
        from matplotlib.font_manager import FontProperties

        # mplhep >= 1.1.3 (the declared floor) takes the scope ("Preliminary") as text= and
        # ignores label=; it prepends "Simulation" itself when data is False.
        def draw(size):
            texts = hep.cms.label(
                ax=ax,
                data=spec.is_data,
                text=spec.scope_text,
                rlabel=_t(spec.lumi_text),
                loc=0,
                **({} if size is None else {"fontsize": size}),
            )
            _align_cms_label(ax, texts)
            return texts

        texts = draw(spec.cms_label_fontsize)
        # in points, for the shrinking; a named size ("large") is relative to the style's
        current = (
            float(matplotlib.rcParams["font.size"])
            if spec.cms_label_fontsize is None
            else FontProperties(size=spec.cms_label_fontsize).get_size_in_points()
        )
        floor = _CMS_LABEL_MIN_SCALE * current
        for _ in range(4):
            overlap = _cms_label_overlap(ax, *texts[:3])
            if overlap <= 0:
                return
            if current <= floor:
                break
            # every width scales with the font size: shrink by the part that does not fit
            width = _cms_label_width(ax, texts[:3])
            current = max(floor, current * (1.0 - overlap / width) * 0.98)
            for t in texts:
                if t is not None:
                    t.remove()
            texts = draw(current)
        if _cms_label_overlap(ax, *texts[:3]) <= 0:
            return
        warnings.warn(
            f"CMS label {_scope_line(spec)!r} overlaps the lumi text even at "
            f"{_CMS_LABEL_MIN_SCALE:.0%} of its size; set a smaller "
            "page_setup.cms_label_fontsize",
            UserWarning,
        )

    def render_stacked(self, spec: StackSpec, output_file: str) -> None:
        plt, hep, fig, ax, rax = self._new_figure(spec)

        # stacked backgrounds
        if spec.backgrounds:
            edges = spec.backgrounds[0].hist.edges
            hep.histplot(
                [e.hist.values for e in spec.backgrounds],
                bins=edges,
                stack=True,
                histtype="fill",
                label=[_t(e.label) for e in spec.backgrounds],
                color=[e.color for e in spec.backgrounds],
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
            )
        if spec.bkg_total is not None:
            for hist, band in _bands(spec):
                self._step_band(
                    ax,
                    hist.edges,
                    hist.values - hist.errors,
                    hist.values + hist.errors,
                    band,
                )
                # legend proxy for the band
                ax.fill(
                    np.nan,
                    np.nan,
                    facecolor="none",
                    hatch=band.hatch,
                    edgecolor=band.color,
                    label=_t(band.label),
                )

        # signal overlays (already scaled by the plotter)
        for e in spec.signals:
            hep.histplot(
                e.hist.values,
                bins=e.hist.edges,
                histtype="step",
                color=e.line_color or e.color,
                linewidth=e.line_width,
                linestyle=e.line_style,
                label=_t(e.label),
                ax=ax,
            )

        # data points
        if spec.data is not None:
            d = spec.data
            hep.histplot(
                d.hist.values,
                bins=d.hist.edges,
                yerr=d.hist.errors,
                histtype="errorbar",
                color=d.color,
                marker=d.marker or "o",
                markersize=d.marker_size,
                label=_t(d.label),
                ax=ax,
            )

        # axes ranges / scales
        edges = self._any_edges(spec)
        self._set_xrange(ax, edges)
        ax.set_ylabel(_t(spec.y_title))
        if spec.log_x:
            ax.set_xscale("log")
        if spec.log_y:
            ax.set_yscale("log")
            ax.set_ylim(spec.y_min_log, _log_y_max(spec, _y_highest(spec)))
        else:
            ax.set_ylim(spec.y_min, _y_max(spec, _y_highest(spec)))

        # CMS + auxiliary labels
        self._cms_label(hep, ax, spec)
        y = 0.92
        for i, lab in enumerate(spec.extra_labels):
            style = (
                spec.extra_label_styles[i] if i < len(spec.extra_label_styles) else {}
            )
            size = style.get("fontsize") or spec.legend_fontsize or 16
            text = _bold_mathtext(_t(lab)) if style.get("bold") else _t(lab)
            ax.text(
                0.05,
                y,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=size,
                fontweight="bold" if style.get("bold") else "normal",
            )
            # a line per label, further apart for a label larger than 16 pt
            y -= 0.06 * max(1.0, size / 16.0) if style else 0.06
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        if spec.legend_note:
            handles.append(Line2D([], [], linestyle="none"))
            labels.append(_t(spec.legend_note))
        bands = [_t(band.label) for _, band in _bands(spec)]
        if len(bands) > 1 and bands[0] in labels:
            pairs = _keep_together(
                list(zip(handles, labels)),
                labels.index(bands[0]),
                len(bands),
                spec.legend_ncol,
                (Line2D([], [], linestyle="none"), ""),
            )
            handles, labels = [h for h, _ in pairs], [label for _, label in pairs]
        ax.legend(
            handles,
            labels,
            loc=spec.legend_loc,
            frameon=False,
            fontsize=spec.legend_fontsize,
            ncol=spec.legend_ncol,
            borderaxespad=self._legend_axes_pad(spec),
        )

        # ratio panel
        if rax is not None:
            self._draw_ratio(rax, spec)
            rax.set_xlabel(_t(spec.x_title))
            self._set_xrange(rax, edges)
            if spec.log_x:
                rax.set_xscale("log")
        else:
            ax.set_xlabel(_t(spec.x_title))

        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _legend_axes_pad(spec) -> float:
        """The legend's distance from the axes, in units of its font size: matplotlib's 0.5,
        or more when the style's inward major ticks are longer than the padding, so that no
        tick runs into the legend text (CMS style: 12 pt ticks, 10 pt for a 0.017 legend).
        """
        import matplotlib
        from matplotlib.font_manager import FontProperties

        fontsize = FontProperties(
            size=spec.legend_fontsize or matplotlib.rcParams["legend.fontsize"]
        ).get_size_in_points()
        tick = max(
            float(matplotlib.rcParams["xtick.major.size"]),
            float(matplotlib.rcParams["ytick.major.size"]),
        )
        return max(0.5, tick / fontsize)

    @staticmethod
    def _set_xrange(ax, edges):
        if edges is None:
            return
        lo, hi = float(edges[0]), float(edges[-1])
        if hi > lo:
            ax.set_xlim(lo, hi)

    @staticmethod
    def _any_edges(spec):
        for src in (spec.backgrounds, spec.signals):
            if src:
                return src[0].hist.edges
        if spec.data is not None:
            return spec.data.hist.edges
        if spec.bkg_total is not None:
            return spec.bkg_total.edges
        return None

    def _draw_ratio(self, rax, spec):
        total, d = spec.bkg_total, spec.data
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = total.values > 0
            denom = np.where(safe, total.values, 1)
            # Every band shares the same denominator, so the nesting carries over to the ratio.
            for hist, band in _bands(spec):
                rel = np.where(safe, hist.errors / denom, 0.0)
                self._step_band(rax, total.edges, 1 - rel, 1 + rel, band)
        rax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        if d is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(safe, d.hist.values / denom, np.nan)
                ratio_err = np.where(safe, d.hist.errors / denom, np.nan)
            rax.errorbar(
                d.hist.centers,
                ratio,
                yerr=ratio_err,
                fmt="o",
                color=d.color,
                markersize=d.marker_size,
            )
        rax.set_ylim(spec.ratio_min, spec.ratio_max)
        rax.set_ylabel(_t(spec.ratio_title))


# --------------------------------------------------------------------------- #
# ROOT / cmsstyle
# --------------------------------------------------------------------------- #
# cmsstyle's own size of "CMS", relative to the pad's top margin (cmsstyle 0.5.0). Restored on
# every plot: SetCmsText(size=) sets a module global that would outlive the plot.
_CMSSTYLE_CMS_TEXT_SIZE = 0.75
# mplhep draws "CMS" at 1.3 times the base size of the CMS line (cms_label_fontsize).
_MPLHEP_CMS_TEXT_SCALE = 1.3
# The share of a TLegend column that the entry's symbol takes (TLegend's default margin).
_TLEGEND_MARGIN = 0.25


class CmsstyleBackend(StyleBackend):
    name = "cmsstyle"

    def _to_th1(self, ROOT, hist, name, errors=True):
        th = ROOT.TH1D(
            name, name, len(hist.values), np.asarray(hist.edges, dtype="float64")
        )
        for i, (v, e) in enumerate(zip(hist.values, hist.errors), start=1):
            th.SetBinContent(i, float(v))
            th.SetBinError(i, float(e) if errors else 0.0)
        th.SetDirectory(0)
        return th

    @staticmethod
    def _additional_info(spec) -> list:
        """The labels drawn in the frame's top left, as ``(TLatex, size in points, styled)``:
        bold ones in ``#bf{}``, each in the size the mplhep backend draws it at (its
        label_lines text_size, else the legend's, else 16 pt); ``styled`` when the labels
        come from label_lines, whose larger sizes space the lines further apart."""
        labels = []
        for i, lab in enumerate(spec.extra_labels):
            style = (
                spec.extra_label_styles[i] if i < len(spec.extra_label_styles) else {}
            )
            text = _root_text(lab)
            if style.get("bold"):
                text = f"#bf{{{text}}}"
            points = style.get("fontsize") or spec.legend_fontsize or 16
            labels.append((text, points, bool(style)))
        return labels

    @staticmethod
    def _root_color(ROOT, rgb):
        if isinstance(rgb, str):
            return ROOT.TColor.GetColor(rgb)
        return ROOT.TColor.GetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))

    @staticmethod
    def _cms_line_fit(ROOT, pad, size, spec):
        """The cmsstyle size of "CMS" at which "CMS <scope>", drawn above the frame, ends
        clear of the lumi by half the scope's font, as the mplhep backend keeps it: ``size``,
        or smaller down to _CMS_LABEL_MIN_SCALE of it, with mplhep's warning below that.
        Every position and width of the line is proportional to the size, so it is measured
        once, at ``size``, from where cmsstyle's CMS_lumi puts the texts."""
        import ctypes

        import cmsstyle.cmsstyle as cms_module

        left, right, top = pad.GetLeftMargin(), pad.GetRightMargin(), pad.GetTopMargin()
        pad_w = pad.GetWw() * pad.GetAbsWNDC()
        pad_h = pad.GetWh() * pad.GetAbsHNDC()

        def width(text, font, ndc_size):
            latex = ROOT.TLatex(0, 0, text)
            latex.SetTextFont(font)
            latex.SetTextSize(ndc_size)
            w, h = ctypes.c_uint(0), ctypes.c_uint(0)
            latex.GetBoundingBox(w, h)
            return w.value / pad_w

        cms = size * top
        end = left + width("CMS", cms_module.cmsTextFont, cms)
        if cms_module.extraText:
            scope_size = cms_module.extraOverCmsTextSize * cms
            # CMS_lumi's offset of the scope after "CMS", above the frame
            start = left + 0.043 * (cms_module.extraTextFont * cms) * min(
                1.0, pad_h / pad_w
            )
            end = start + width(
                cms_module.extraText, cms_module.extraTextFont, scope_size
            )
            end += 0.5 * scope_size * pad_h / pad_w
        lumi = cms_module.cms_lumi + (
            f" ({cms_module.cms_energy})" if cms_module.cms_energy else ""
        )
        lumi_start = 1 - right - width(lumi, 42, cms_module.lumiTextSize * top)
        if end <= lumi_start:
            return size
        fit = size * (lumi_start - left) / (end - left)
        if fit >= _CMS_LABEL_MIN_SCALE * size:
            return fit
        warnings.warn(
            f"CMS label {_scope_line(spec)!r} overlaps the lumi text even at "
            f"{_CMS_LABEL_MIN_SCALE:.0%} of its size; set a smaller "
            "page_setup.cms_label_fontsize",
            UserWarning,
        )
        return _CMS_LABEL_MIN_SCALE * size

    def _cms_line(self, ROOT, CMS, pad, spec):
        """Draw the CMS line on ``pad`` once, laid out as the mplhep backend lays it out:
        "CMS", the scope and the lumi above the frame, the labels inside its top left.

        cmsCanvas has already drawn the line in the frame, in cmsstyle's sizes, before the
        pad's top margin, which cmsstyle's sizes are relative to, could be read here: those
        texts are removed first. "CMS" is set on every plot, since SetExtraText blanks it for
        a "Private" scope and the blank would persist into every later plot; mplhep draws
        "CMS Private work".
        """
        primitives = pad.GetListOfPrimitives()
        for obj in list(primitives):
            if obj.InheritsFrom("TLatex"):
                primitives.Remove(obj)
        left, right = pad.GetLeftMargin(), pad.GetRightMargin()
        top, bottom = pad.GetTopMargin(), pad.GetBottomMargin()
        size = _CMSSTYLE_CMS_TEXT_SIZE
        if spec.cms_label_fontsize is not None:
            points = _font_points(spec.cms_label_fontsize)
            size = _MPLHEP_CMS_TEXT_SCALE * points / _PT_PER_NDC / top
        CMS.SetCmsText("CMS", size=self._cms_line_fit(ROOT, pad, size, spec))
        CMS.ResetAdditionalInfo()
        CMS.CMS_lumi(pad, 0)  # 0: above the frame
        # the labels at the mplhep backend's positions, in fractions of the frame
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextFont(42)
        latex.SetTextAlign(13)  # left, top
        y = 0.92
        for text, points, styled in self._additional_info(spec):
            latex.SetTextSize(points / _PT_PER_NDC)
            latex.DrawLatex(
                left + 0.05 * (1 - left - right), bottom + y * (1 - top - bottom), text
            )
            y -= 0.06 * max(1.0, points / 16.0) if styled else 0.06

    @staticmethod
    def _legend(ROOT, CMS, pad, spec, entries, bands=(0, 0)):
        """The legend in the top right of ``pad``, sized to its entries: in the configured
        text size (legend.text_size; cmsstyle's 0.04 when unset), in ``legend_ncol`` columns
        as wide as the widest label, and filled column by column as the mplhep backend does.
        Columns that would not fit right of the frame's left edge fall back to one."""
        import ctypes
        import math

        if not entries:
            return None
        text_size = (
            0.04 if spec.legend_fontsize is None else spec.legend_fontsize / _PT_PER_NDC
        )
        pad_width = pad.GetWw() * pad.GetAbsWNDC()

        def width(label):
            latex = ROOT.TLatex(0, 0, label)
            latex.SetTextFont(42)
            latex.SetTextSize(text_size)
            w, h = ctypes.c_uint(0), ctypes.c_uint(0)
            latex.GetBoundingBox(w, h)
            return w.value / pad_width

        column = max(width(label) for _, label, _ in entries) / (1 - _TLEGEND_MARGIN)
        column *= 1.05
        x2 = 1 - pad.GetRightMargin() - 0.03
        y2 = 1 - pad.GetTopMargin() - 0.03
        left = pad.GetLeftMargin() + 0.03
        ncol = max(1, spec.legend_ncol or 1)
        if ncol > 1 and x2 - ncol * column < left:
            ncol = 1
        # the uncertainty bands (``bands``: first entry, count) in one column, as in mplhep
        entries = _keep_together(entries, *bands, ncol, (ROOT.nullptr, "", ""))
        rows = math.ceil(len(entries) / ncol)
        pad.cd()
        leg = CMS.cmsLeg(
            max(left, x2 - ncol * column),
            y2 - rows * 1.3 * text_size,
            x2,
            y2,
            textSize=text_size,
            columns=ncol,
        )
        for obj, label, option in _column_major(entries, ncol, (ROOT.nullptr, "", "")):
            leg.AddEntry(obj, label, option)
        return leg

    def render_stacked(self, spec: StackSpec, output_file: str) -> None:
        try:
            import ROOT
            import cmsstyle as CMS
        except Exception as exc:  # ROOT or cmsstyle missing
            warnings.warn(
                f"cmsstyle backend unavailable ({exc}); falling back to mplhep.",
                RuntimeWarning,
            )
            return MplhepBackend().render_stacked(spec, output_file)

        ROOT.gROOT.SetBatch(True)
        CMS.setCMSStyle()
        # cmsstyle wants numeric lumi/energy; recover them from the legacy string.
        lumi_val, energy_val = _parse_lumi_energy(spec.lumi_text)
        if energy_val is not None:
            CMS.SetEnergy(energy_val)
        CMS.SetLumi(lumi_val if lumi_val is not None else None, run="")
        CMS.SetExtraText(_root_text(_scope_line(spec)))

        edges = MplhepBackend._any_edges(spec)
        x_min, x_max = float(edges[0]), float(edges[-1])
        y_min = spec.y_min_log if spec.log_y else spec.y_min
        highest = _y_highest(spec)
        y_max = _log_y_max(spec, highest) if spec.log_y else _y_max(spec, highest)

        want_ratio = _want_ratio(spec)
        if want_ratio:
            canv = CMS.cmsDiCanvas(
                "c",
                x_min,
                x_max,
                y_min,
                y_max,
                spec.ratio_min,
                spec.ratio_max,
                _root_text(spec.x_title),
                _root_text(spec.y_title),
                _root_text(spec.ratio_title),
                square=True,
            )
            main = canv.cd(1)
        else:
            canv = CMS.cmsCanvas(
                "c",
                x_min,
                x_max,
                y_min,
                y_max,
                _root_text(spec.x_title),
                _root_text(spec.y_title),
                square=True,
            )
            main = canv
        main.cd()
        if spec.log_y:
            main.SetLogy(True)
        if spec.log_x:
            main.SetLogx(True)

        keep = []  # keep python refs alive
        backgrounds, others = [], []
        stack = ROOT.THStack("stack", "")
        for i, e in enumerate(spec.backgrounds):
            th = self._to_th1(ROOT, e.hist, f"bkg_{i}")
            th.SetFillColor(self._root_color(ROOT, e.color))
            th.SetLineColor(ROOT.kBlack)
            th.SetLineWidth(1)
            stack.Add(th)
            keep.append(th)
            backgrounds.append((th, _root_text(e.label), "f"))
        stack.Draw("HIST SAME")
        keep.append(stack)

        # background uncertainty band(s): full (stat + syst) first, stat-only on top
        for i, (hist, style) in enumerate(_bands(spec)):
            band = self._to_th1(ROOT, hist, f"bkg_unc_{i}")
            band.SetFillColor(self._root_color(ROOT, style.color))
            band.SetFillStyle(style.root_fill_style)
            band.SetMarkerSize(0)
            band.Draw("E2 SAME")
            keep.append(band)
            others.append((band, _root_text(style.label), "f"))

        for i, e in enumerate(spec.signals):
            th = self._to_th1(ROOT, e.hist, f"sig_{i}")
            th.SetLineColor(self._root_color(ROOT, e.line_color or e.color))
            th.SetLineWidth(int(e.line_width))
            th.SetLineStyle(2 if e.line_style == "--" else 1)
            th.SetFillStyle(0)
            th.Draw("HIST SAME")
            keep.append(th)
            others.append((th, _root_text(e.label), "l"))

        if spec.data is not None:
            d = self._to_th1(ROOT, spec.data.hist, "data")
            d.SetMarkerStyle(20)
            d.SetLineColor(ROOT.kBlack)
            d.Draw("PE SAME")
            keep.append(d)
            others.append((d, _root_text(spec.data.label), "pe"))
        if spec.legend_note:
            # text-only entry, as in the mplhep legend
            others.append((ROOT.nullptr, _root_text(spec.legend_note), ""))

        # the legend and the CMS line last, so that they are drawn over the histograms
        keep.append(
            self._legend(
                ROOT,
                CMS,
                main,
                spec,
                _legend_entries_mplhep_order(backgrounds, others),
                bands=(len(backgrounds), len(_bands(spec))),
            )
        )
        self._cms_line(ROOT, CMS, main, spec)

        # ratio panel
        if want_ratio:
            canv.cd(2)
            # Obs/Bkg with the data error alone: the background's own error is the band.
            total = self._to_th1(ROOT, spec.bkg_total, "ratio_den", errors=False)
            keep.append(total)
            # Relative uncertainty band(s) around 1, drawn first so the points sit on top.
            for i, (hist, style) in enumerate(_bands(spec)):
                rel_band = self._to_th1(ROOT, _relative_band(hist), f"ratio_unc_{i}")
                rel_band.SetFillColor(self._root_color(ROOT, style.color))
                rel_band.SetFillStyle(style.root_fill_style)
                rel_band.SetMarkerSize(0)
                rel_band.Draw("E2 SAME")
                keep.append(rel_band)
            if spec.data is not None:
                ratio = self._to_th1(ROOT, spec.data.hist, "ratio")
                ratio.Divide(total)
                ratio.SetMarkerStyle(20)
                ratio.Draw("PE SAME")
                keep.append(ratio)

        CMS.SaveCanvas(canv, output_file)


_BACKENDS = {
    "mplhep": MplhepBackend,
    "matplotlib": MplhepBackend,
    "cmsstyle": CmsstyleBackend,
    "root": CmsstyleBackend,
}


def get_backend(name: str) -> StyleBackend:
    """Return a backend instance by name (default mplhep; unknown names warn)."""
    cls = _BACKENDS.get((name or "mplhep").strip().lower())
    if cls is None:
        warnings.warn(f"Unknown backend {name!r}; using mplhep.", RuntimeWarning)
        cls = MplhepBackend
    return cls()
