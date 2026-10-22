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


def _cms_label_overlap(ax, cms, scope, lumi) -> float:
    """By how many pixels "CMS <scope>" runs into the lumi, a small gap (half the scope's
    font) included; <= 0 when the line fits."""
    if lumi is None or not lumi.get_text():
        return 0.0
    renderer = ax.figure.canvas.get_renderer()
    left = [t for t in (cms, scope) if t is not None and t.get_text()]
    if not left:
        return 0.0
    right_edge = max(t.get_window_extent(renderer).x1 for t in left)
    gap = renderer.points_to_pixels(0.5 * left[-1].get_fontsize())
    return right_edge + gap - lumi.get_window_extent(renderer).x0


def _cms_label_width(ax, texts) -> float:
    """The width in pixels of the texts of the CMS line, the room they take together."""
    renderer = ax.figure.canvas.get_renderer()
    return sum(
        t.get_window_extent(renderer).width
        for t in texts
        if t is not None and t.get_text()
    )


def _log_y_max(spec, y_max: float) -> float:
    """Top of a log y axis: log_y_headroom of its height free above y_max, else max_y_sf."""
    headroom = spec.log_y_headroom
    _check_headroom("log_y_headroom", headroom)
    if headroom is None or y_max <= spec.y_min_log:
        return y_max * spec.max_y_sf
    low, high = np.log10(spec.y_min_log), np.log10(y_max)
    return float(10 ** (low + (high - low) / (1.0 - headroom)))


def _bold_mathtext(text: str) -> str:
    """A converted label made bold inside its math too: mathtext ignores fontweight."""
    if len(text) > 1 and text.startswith("$") and text.endswith("$"):
        return r"$\mathbf{" + text[1:-1].replace(r"\mathrm{", r"\mathbf{") + "}$"
    return text


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

        size = spec.cms_label_fontsize
        texts = draw(size)
        start = size if size is not None else float(matplotlib.rcParams["font.size"])
        floor = _CMS_LABEL_MIN_SCALE * start
        for _ in range(4):
            overlap = _cms_label_overlap(ax, *texts[:3])
            if overlap <= 0:
                return
            current = start if size is None else size
            if current <= floor:
                break
            # every width scales with the font size: shrink by the part that does not fit
            width = _cms_label_width(ax, texts[:3])
            size = max(floor, current * (1.0 - overlap / width) * 0.98)
            for t in texts:
                if t is not None:
                    t.remove()
            texts = draw(size)
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

        ymax_candidates = [1e-9]

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
            t = _band_hist(spec)
            ymax_candidates.append(float(np.max(t.values + t.errors)))
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
            ymax_candidates.append(float(np.max(e.hist.values)))

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
            ymax_candidates.append(float(np.max(d.hist.values + d.hist.errors)))

        # axes ranges / scales
        edges = self._any_edges(spec)
        self._set_xrange(ax, edges)
        ax.set_ylabel(_t(spec.y_title))
        if spec.log_x:
            ax.set_xscale("log")
        if spec.log_y:
            ax.set_yscale("log")
            ax.set_ylim(spec.y_min_log, _log_y_max(spec, max(ymax_candidates)))
        else:
            ax.set_ylim(spec.y_min, _y_max(spec, max(ymax_candidates)))

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
        handles, labels = ax.get_legend_handles_labels()
        if spec.legend_note:
            from matplotlib.lines import Line2D

            handles.append(Line2D([], [], linestyle="none"))
            labels.append(_t(spec.legend_note))
        ax.legend(
            handles,
            labels,
            loc=spec.legend_loc,
            frameon=False,
            fontsize=spec.legend_fontsize,
            ncol=spec.legend_ncol,
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
class CmsstyleBackend(StyleBackend):
    name = "cmsstyle"

    def _to_th1(self, ROOT, hist, name):
        th = ROOT.TH1D(
            name, name, len(hist.values), np.asarray(hist.edges, dtype="float64")
        )
        for i, (v, e) in enumerate(zip(hist.values, hist.errors), start=1):
            th.SetBinContent(i, float(v))
            th.SetBinError(i, float(e))
        th.SetDirectory(0)
        return th

    @staticmethod
    def _additional_info(spec) -> list:
        """The labels below "CMS", bold ones as TLatex ``#bf{}``. cmsstyle sizes the CMS line
        and these labels itself, relative to the pad as the CMS guidelines set them, so the
        font-size options are not applied -- and said so, rather than dropped silently.
        """
        ignored = [
            name
            for name, unset in (
                ("cms_label_fontsize", spec.cms_label_fontsize is None),
                (
                    "label_lines text_size",
                    not any(s.get("fontsize") for s in spec.extra_label_styles),
                ),
            )
            if not unset
        ]
        if ignored:
            warnings.warn(
                f"cmsstyle backend: {', '.join(ignored)} not applied; cmsstyle sets the "
                "label sizes itself",
                UserWarning,
            )
        labels = []
        for i, lab in enumerate(spec.extra_labels):
            style = (
                spec.extra_label_styles[i] if i < len(spec.extra_label_styles) else {}
            )
            labels.append(f"#bf{{{lab}}}" if style.get("bold") else lab)
        return labels

    @staticmethod
    def _root_color(ROOT, rgb):
        if isinstance(rgb, str):
            return ROOT.TColor.GetColor(rgb)
        return ROOT.TColor.GetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))

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
        CMS.ResetAdditionalInfo()
        for lab in self._additional_info(spec):
            CMS.AppendAdditionalInfo(lab)  # raw ROOT TLatex -- rendered natively
        # cmsstyle wants numeric lumi/energy; recover them from the legacy string.
        lumi_val, energy_val = _parse_lumi_energy(spec.lumi_text)
        if energy_val is not None:
            CMS.SetEnergy(energy_val)
        CMS.SetLumi(lumi_val if lumi_val is not None else None, run="")
        CMS.SetExtraText(_scope_line(spec))

        edges = MplhepBackend._any_edges(spec)
        x_min, x_max = float(edges[0]), float(edges[-1])
        ymax = 1e-9
        keep = []  # keep python refs alive

        stack = ROOT.THStack("stack", "")
        for i, e in enumerate(spec.backgrounds):
            th = self._to_th1(ROOT, e.hist, f"bkg_{i}")
            th.SetFillColor(self._root_color(ROOT, e.color))
            th.SetLineColor(ROOT.kBlack)
            th.SetLineWidth(1)
            stack.Add(th)
            keep.append(th)
        if spec.bkg_total is not None:
            t = _band_hist(spec)
            ymax = max(ymax, float(np.max(t.values + t.errors)))
        if spec.data is not None:
            ymax = max(
                ymax, float(np.max(spec.data.hist.values + spec.data.hist.errors))
            )

        y_min = spec.y_min_log if spec.log_y else spec.y_min
        y_max = _log_y_max(spec, ymax) if spec.log_y else _y_max(spec, ymax)

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
                spec.x_title,
                spec.y_title,
                spec.ratio_title,
                square=True,
            )
            canv.cd(1)
        else:
            canv = CMS.cmsCanvas(
                "c",
                x_min,
                x_max,
                y_min,
                y_max,
                spec.x_title,
                spec.y_title,
                square=True,
            )
        if spec.log_y:
            ROOT.gPad.SetLogy(True)
        if spec.log_x:
            ROOT.gPad.SetLogx(True)

        leg = CMS.cmsLeg(0.55, 0.6, 0.92, 0.9, columns=spec.legend_ncol)
        stack.Draw("HIST SAME")
        keep.append(stack)
        for i, e in enumerate(spec.backgrounds):
            leg.AddEntry(keep[i], e.label, "f")

        # background uncertainty band(s): full (stat + syst) first, stat-only on top
        for i, (hist, style) in enumerate(_bands(spec)):
            band = self._to_th1(ROOT, hist, f"bkg_unc_{i}")
            band.SetFillColor(self._root_color(ROOT, style.color))
            band.SetFillStyle(style.root_fill_style)
            band.SetMarkerSize(0)
            band.Draw("E2 SAME")
            keep.append(band)
            leg.AddEntry(band, style.label, "f")

        for i, e in enumerate(spec.signals):
            th = self._to_th1(ROOT, e.hist, f"sig_{i}")
            th.SetLineColor(self._root_color(ROOT, e.line_color or e.color))
            th.SetLineWidth(int(e.line_width))
            th.SetLineStyle(2 if e.line_style == "--" else 1)
            th.SetFillStyle(0)
            th.Draw("HIST SAME")
            keep.append(th)
            leg.AddEntry(th, e.label, "l")

        if spec.data is not None:
            d = self._to_th1(ROOT, spec.data.hist, "data")
            d.SetMarkerStyle(20)
            d.SetLineColor(ROOT.kBlack)
            d.Draw("PE SAME")
            keep.append(d)
            leg.AddEntry(d, spec.data.label, "pe")
        if spec.legend_note:
            # text-only entry, as in the mplhep legend
            leg.AddEntry(ROOT.nullptr, spec.legend_note, "")

        CMS.CMS_lumi(ROOT.gPad)  # draw CMS + lumi on the active (main) pad

        # ratio panel
        if want_ratio:
            canv.cd(2)
            total = self._to_th1(ROOT, spec.bkg_total, "ratio_den")
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
