"""Backward-compatible stacked-plot style.

Reproduces the output of the legacy ROOT ``StackedPlotDescriptor`` (stacked backgrounds,
overlaid scaled signals, data points, a background-uncertainty band and an Obs/Bkg ratio
panel) directly from the unchanged analysis ``config/plot/*.yaml`` files.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from .. import rootcompat as rc
from ..backends import _check_headroom
from ..histogram import stack_total, to_hist1d
from ..spec import HistEntry, StackSpec, UncBand
from .base import BasePlotter


class StackedPlotter(BasePlotter):
    def plot(
        self,
        hist_name: str,
        histograms: dict,
        output_file: str,
        want_data: bool = True,
        custom: Optional[dict] = None,
        scale=None,
        total_unc=None,
    ) -> None:
        """Render a stacked plot.

        ``histograms`` maps ``key -> (hist, plot_name, plot_color, process_group)`` where
        ``hist`` is a ROOT ``TH1``, an uproot histogram or a :class:`~PlotKit.histogram.Hist1D`,
        and ``process_group`` is one of ``backgrounds`` / ``signals`` / ``data`` -- exactly the
        structure the FLAF ``HistPlotter`` already builds.

        ``total_unc`` is an optional histogram holding the summed background with its *full*
        per-bin uncertainty (stat + syst).  When given, two nested bands are drawn -- the full
        uncertainty and, on top of it, the stat-only one -- both on the main pad and in the
        ratio panel, and the ratio panel is drawn even when the plot is blinded.
        """
        spec = self.build_spec(
            hist_name, histograms, want_data, custom or {}, scale, total_unc
        )
        self.backend.render_stacked(spec, output_file)

    # ------------------------------------------------------------------ #
    def build_spec(
        self, hist_name, histograms, want_data, custom, scale, total_unc=None
    ) -> StackSpec:
        cfg = self.config
        hd = cfg.hist_desc(hist_name)
        page = cfg.page_setup

        divide = bool(hd.get("divide_by_bin_width", False))
        log_y = bool(hd.get("use_log_y", False))
        log_x = bool(hd.get("use_log_x", False))
        max_y_sf = float(hd.get("max_y_sf", page.get("max_y_sf", 1.5)))

        sgn_style = cfg.group_style("signals")
        data_style = cfg.group_style("data")

        # "bkg" normalisation needs the summed-background integral, which is only known
        # after the loop -- so collect signals here and scale them at the end.
        norm_to_bkg = self._is_bkg_scale(scale)

        backgrounds, signals, data = [], [], None
        raw_signals = []  # (display_hist, raw_integral, plot_name, color)
        bkg_raw_integral = 0.0
        for key, entry in histograms.items():
            hist_obj, plot_name, plot_color, group = entry
            H = to_hist1d(hist_obj, name=str(key))
            raw_integral = H.integral()  # event count, before divide_by_bin_width
            if divide:
                H = H.divided_by_width()
            color = rc.root_color_to_rgba(plot_color)
            # Text is kept raw (ROOT TLatex) in the spec; each backend converts as needed
            # (mplhep -> mathtext, cmsstyle -> native ROOT TLatex).
            label = plot_name

            if group == "backgrounds":
                bkg_raw_integral += raw_integral
                backgrounds.append(
                    HistEntry(hist=H, label=label, color=color, filled=True)
                )
            elif group == "signals":
                raw_signals.append((H, raw_integral, plot_name, color))
            elif group == "data":
                if not want_data:
                    continue
                data = HistEntry(
                    hist=H,
                    label=label,
                    color=rc.root_color_to_rgba(
                        data_style.get("marker_color", "kBlack")
                    ),
                    filled=False,
                    marker="o",
                    marker_size=8.0,
                )
            else:
                raise ValueError(f"Unknown process group: {group!r}")

        bkg_total = stack_total([e.hist for e in backgrounds])

        # Full (stat + syst) uncertainty on the stack, if the caller computed it.
        bkg_total_unc = None
        if total_unc is not None and bkg_total is not None:
            bkg_total_unc = to_hist1d(total_unc, name="total_unc")
            if divide:
                bkg_total_unc = bkg_total_unc.divided_by_width()

        # sgn_hist options: show_yield (default true) adds each signal's yield to its label;
        # norm_legend_note, with "bkg" scaling, says it once as a legend entry instead of
        # "(norm. to bkg)" on every signal
        show_yield = sgn_style.get("show_yield", True)
        if not isinstance(show_yield, bool):
            # a quoted 'false' is a non-empty string, which bool() would read as true
            raise ValueError(
                f"sgn_hist.show_yield = {show_yield!r} must be true or false"
            )
        norm_note = sgn_style.get("norm_legend_note") if norm_to_bkg else None
        for H, raw_integral, plot_name, color in raw_signals:
            if norm_to_bkg:
                # Scale each signal so its integral equals the summed-background integral.
                factor = (bkg_raw_integral / raw_integral) if raw_integral else 1.0
                if norm_note:
                    label = self._SCALE_SUFFIX_RE.sub("", plot_name).rstrip()
                else:
                    label = self._bkg_norm_label(plot_name)
            else:
                factor = self._signal_scale(scale, H.name)
                label = plot_name
            if show_yield:
                label = label + f" (yield {raw_integral:.2f})"
            if factor != 1.0:
                H = H.scaled(factor)
            signals.append(
                HistEntry(
                    hist=H,
                    label=label,
                    color=color,
                    line_color=color,
                    filled=False,
                    line_style=rc.line_style_to_mpl(sgn_style.get("line_style", 2)),
                    line_width=float(sgn_style.get("line_width", 3)),
                )
            )

        max_ratio = float(page.get("max_ratio", 1.5))
        # With the full uncertainty available the ratio panel is meaningful even without
        # data: it then shows the relative (stat + syst) band around the stack.
        draw_ratio = bool(page.get("draw_ratio", True)) and (
            data is not None or bkg_total_unc is not None
        )
        if data is not None:
            ratio_title = page.get("ratio_y_title", "Obs/Bkg")
        else:
            ratio_title = page.get("unc_ratio_y_title", "Unc./Bkg")
        spec = StackSpec(
            backgrounds=backgrounds,
            signals=signals,
            data=data,
            bkg_total=bkg_total,
            bkg_total_unc=bkg_total_unc,
            unc_band=(
                self._unc_band(cfg, full_unc=bkg_total_unc is not None)
                if bkg_total is not None
                else None
            ),
            # Second, narrower band showing the stat-only part, so the systematic
            # contribution reads as the region the outer hatch covers on its own.
            stat_band=self._stat_band(cfg) if bkg_total_unc is not None else None,
            x_title=hd.get("x_title", hist_name),
            y_title=hd.get("y_title", "Events"),
            log_x=log_x,
            log_y=log_y,
            y_min=float(page.get("y_min", 0.0)),
            y_min_log=float(page.get("y_min_log", 1e-2)),
            max_y_sf=max_y_sf,
            log_y_headroom=self._headroom(hist_name, hd, page, "log_y_headroom"),
            y_headroom=self._headroom(hist_name, hd, page, "y_headroom"),
            cms_label_fontsize=self._font_size(
                page.get("cms_label_fontsize"), "page_setup.cms_label_fontsize"
            ),
            legend_note=(norm_note or "") if signals else "",
            draw_ratio=draw_ratio,
            ratio_title=ratio_title,
            ratio_min=max(0.0, 2.0 - max_ratio),
            ratio_max=max_ratio,
            is_data=want_data,
            **self._labels(cfg, custom, want_data),
        )
        return spec

    # ------------------------------------------------------------------ #
    # Aliases that request "scale each signal to the summed-background integral".
    _BKG_SCALE_ALIASES = {"bkg", "bkgs", "background", "backgrounds"}
    # A trailing "(x100)" / "(×100)" suffix baked into a config label, which is
    # meaningless once the signal is normalised to the background instead.
    _SCALE_SUFFIX_RE = re.compile(r"\s*\(\s*[x×]\s*[\d.]+\s*\)\s*$")

    @classmethod
    def _is_bkg_scale(cls, scale) -> bool:
        return (
            isinstance(scale, str) and scale.strip().lower() in cls._BKG_SCALE_ALIASES
        )

    @classmethod
    def _bkg_norm_label(cls, label) -> str:
        return cls._SCALE_SUFFIX_RE.sub("", label).rstrip() + " (norm. to bkg)"

    @staticmethod
    def _signal_scale(scale, hist_name) -> float:
        if scale is None:
            return 1.0
        if isinstance(scale, dict):
            return float(scale.get(hist_name, 1.0))
        try:
            return float(scale)
        except (TypeError, ValueError):
            return 1.0

    def _unc_band(self, cfg, full_unc=False) -> UncBand:
        s = cfg.bkg_unc_style
        fill_style = s.get("fill_style", 3013)
        label = s.get("legend_title", "Bkg. uncertainty")
        if full_unc:
            # A second band is nested inside this one, so take the sparser diagonal hatch
            # and leave the denser cross-hatch to the stat band -- the reverse is hard to
            # read, since the inner band is the one competing with the outer hatch.
            fill_style = s.get("fill_style_full", 3004)
            # Make it explicit that the band is no longer stat-only.
            label = s.get("legend_title_full", f"{label} (stat + syst)")
        return UncBand(
            hatch=rc.fill_style_to_hatch(fill_style) or "///",
            color=rc.root_color_to_rgba(s.get("fill_color", "kCyan-5")),
            label=label,
            root_fill_style=self._root_fill_style(fill_style, 3013),
        )

    def _stat_band(self, cfg) -> UncBand:
        """Style for the inner, stat-only band.

        Defaults deliberately differ from the outer band in *both* hatch and colour: the two
        overlap wherever the stat error reaches, so a shared style would be unreadable.  All
        three are overridable via ``stat_fill_style`` / ``stat_fill_color`` /
        ``stat_legend_title`` in the plot config's ``bkg_unc_hist`` block.
        """
        s = cfg.bkg_unc_style
        fill_style = s.get("stat_fill_style", 3013)
        return UncBand(
            hatch=rc.fill_style_to_hatch(fill_style) or "xxx",
            color=rc.root_color_to_rgba(s.get("stat_fill_color", "kGray+2")),
            label=s.get("stat_legend_title", "Bkg. stat. unc."),
            root_fill_style=self._root_fill_style(fill_style, 3013),
        )

    @staticmethod
    def _root_fill_style(fill_style, default: int) -> int:
        try:
            return int(fill_style)
        except (TypeError, ValueError):
            return default

    # The text boxes under the CMS line, in order; HistPlotter fills them per plot.
    _LABEL_BOXES = ("ana_text", "ch_text", "cat_text", "customreg_text")

    @classmethod
    def _check_label_lines(cls, cfg, label_lines) -> None:
        """page_setup.label_lines must be a non-empty list of lines, each a list of text box
        names declared in page_setup.text_boxes (or, without that list, of the label boxes,
        _LABEL_BOXES): a flat list or a typo would otherwise drop labels silently."""
        if label_lines is None:
            return
        if (
            not isinstance(label_lines, list)
            or not label_lines
            or not all(
                isinstance(line, list)
                and line
                and all(isinstance(b, str) for b in line)
                for line in label_lines
            )
        ):
            raise ValueError(
                f"page_setup.label_lines = {label_lines!r} must be a list of lines, each a "
                "list of text box names, e.g. [[ana_text], [ch_text, cat_text]]"
            )
        declared = set(cfg.text_box_names()) or set(cls._LABEL_BOXES)
        unknown = sorted({b for line in label_lines for b in line} - declared)
        if unknown:
            raise ValueError(
                f"page_setup.label_lines names text boxes that are not declared: {unknown}"
            )

    def _labels(self, cfg, custom, want_data) -> dict:
        """Map the legacy text boxes + per-plot ``custom`` overrides to spec labels."""
        # scope: HistPlotter packs "CMS <scope>" into datasim_text; recover <scope>.
        datasim = custom.get("datasim_text", "")
        scope = datasim[4:].strip() if datasim.startswith("CMS ") else datasim
        if not scope:
            scope = cfg.text_box("scope_text").get("text", "Preliminary")

        cms_text = cfg.text_box("cms_text").get("text", "CMS")
        lumi = cfg.text_box("lumi_text").get("text", "")

        label_lines = cfg.page_setup.get("label_lines")
        self._check_label_lines(cfg, label_lines)
        styles = []
        if label_lines:
            # One label per line of text boxes, in the size and weight of the first box
            extra = []
            for boxes in label_lines:
                texts = [self._box_text(cfg, custom, box) for box in boxes]
                texts = [t for t in texts if t]
                if not texts:
                    continue
                first = cfg.text_box(boxes[0])
                extra.append(", ".join(texts))
                styles.append(
                    {
                        "fontsize": self._text_size_pt(
                            first.get("text_size"), f"{boxes[0]}.text_size"
                        ),
                        "bold": self._is_bold(first.get("font"), f"{boxes[0]}.font"),
                    }
                )
        else:
            extra_sources = [
                self._box_text(cfg, custom, box) for box in self._LABEL_BOXES
            ]
            extra = [t for t in extra_sources if t]

        legend = cfg.legend_cfg
        return {
            "cms_text": cms_text,
            "scope_text": scope,
            "lumi_text": lumi,
            "extra_labels": extra,
            "extra_label_styles": styles,
            "legend_loc": "upper right",
            "legend_fontsize": self._legend_fontsize(legend),
            "legend_ncol": self._n_columns(legend),
        }

    @staticmethod
    def _float_or_none(value, key) -> Optional[float]:
        """An optional number from the config: None when unset, an error naming the key
        when it is not a finite number -- a typo must not silently fall back to the default,
        and a YAML ``true`` or ``.inf`` is not a size."""
        if value is None:
            return None
        try:
            number = None if isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            number = None
        if number is None or not math.isfinite(number):
            raise ValueError(f"{key} = {value!r} is not a number")
        return number

    # The font sizes matplotlib (and so mplhep) accepts by name, relative to the style's.
    _NAMED_FONT_SIZES = {
        "xx-small",
        "x-small",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
        "smaller",
        "larger",
    }

    @classmethod
    def _font_size(cls, value, key):
        """An optional font size: points > 0, or a matplotlib size name such as "large"."""
        if isinstance(value, str) and value in cls._NAMED_FONT_SIZES:
            return value
        size = cls._float_or_none(value, key)
        if size is not None and not size > 0:
            raise ValueError(f"{key} = {value!r} must be a positive font size")
        return size

    @staticmethod
    def _n_columns(legend) -> int:
        n = legend.get("n_columns", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"legend n_columns = {n!r} must be a whole number >= 1")
        return n

    @classmethod
    def _headroom(cls, hist_name, hd, page, name) -> Optional[float]:
        """The headroom fraction (checked to lie in (0, 1)) of a y axis, or None to scale by
        max_y_sf. A variable's own setting beats the page's, and at the same level a headroom
        beats max_y_sf: the variable's headroom, then the variable's max_y_sf (None here), then
        the page's headroom, then the page's max_y_sf (None)."""
        if name in hd:
            key, value = f"{hist_name}.{name}", hd[name]
        elif "max_y_sf" in hd:
            return None
        else:
            key, value = f"page_setup.{name}", page.get(name)
        headroom = cls._float_or_none(value, key)
        _check_headroom(key, headroom)
        return headroom

    @staticmethod
    def _box_text(cfg, custom, box) -> str:
        """A text box's text: the per-plot value when one is passed -- an empty one too, which
        means no label (HistPlotter passes customreg_text="" for a region without a label) --
        else the configured text."""
        if box in custom:
            return custom[box] or ""
        return cfg.text_box(box).get("text", "")

    @classmethod
    def _text_size_pt(cls, text_size, key) -> Optional[float]:
        # Legacy text_size is a ROOT NDC fraction; the same mapping as the legend's.
        size = cls._float_or_none(text_size, key)
        return None if size is None else max(8.0, min(30.0, size * 600.0))

    @staticmethod
    def _is_bold(font, key) -> bool:
        # ROOT font code: precision in the last digit, the font before it; 2, 3, 6, 7, 10
        # and 11 are the bold ones (62 = Helvetica bold).
        if font is None:
            return False
        try:
            if isinstance(font, bool):
                raise TypeError
            return int(font) // 10 in (2, 3, 6, 7, 10, 11)
        except (TypeError, ValueError):
            raise ValueError(f"{key} = {font!r} is not a ROOT font code") from None

    @staticmethod
    def _legend_fontsize(legend) -> Optional[float]:
        # Legacy text_size is a ROOT NDC fraction (~0.012-0.04); map to a sane pt size.
        ts = legend.get("text_size")
        if ts is None:
            return None
        try:
            return max(8.0, min(20.0, float(ts) * 600.0))
        except (TypeError, ValueError):
            return None
