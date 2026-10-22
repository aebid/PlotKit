import numpy as np
import pytest

from PlotKit.Plotter import Plotter
from PlotKit.histogram import Hist1D

PAGE_CFG = {
    "page_setup": {"draw_ratio": True, "max_ratio": 1.5, "legend": "legend"},
    "legend": {"text_size": 0.03},
    "cms_text": {"text": "CMS"},
    "scope_text": {"text": "Preliminary"},
    "lumi_text": {"text": "1 fb^{-1} (13.6 TeV)"},
    "ana_text": {"text": "demo"},
    "bkg_hist": {"unc_hist": "bkg_unc_hist"},
    "bkg_unc_hist": {
        "fill_style": 3013,
        "fill_color": "kCyan-5",
        "legend_title": "Bkg. unc.",
    },
    "sgn_hist": {"line_style": 2, "line_width": 3},
    "data_hist": {"marker_color": "kBlack"},
}
HIST_CFG = {
    "v": {"x_title": "m_{T} (GeV)", "y_title": "Events", "divide_by_bin_width": False}
}


def _synthetic():
    edges = np.linspace(0, 200, 21)
    bkg1 = Hist1D(edges, 40 * np.exp(-edges[:-1] / 80), name="bkg1")
    bkg2 = Hist1D(edges, 20 * np.exp(-edges[:-1] / 120), name="bkg2")
    sig = Hist1D(edges, 5 * np.exp(-((edges[:-1] - 100) ** 2) / 400), name="sig")
    total = bkg1.values + bkg2.values
    data = Hist1D(edges, np.round(total), name="data")
    return {
        "DY": (bkg1, "DY #rightarrow ll", "kAzure-9", "backgrounds"),
        "TT": (bkg2, "t#bar{t}", "kSpring+5", "backgrounds"),
        "HH": (sig, "HH (x10)", "kRed", "signals"),
        "data": (data, "data", "kBlack", "data"),
    }


def test_stacked_render_pdf(tmp_path):
    out = tmp_path / "v.pdf"
    Plotter(PAGE_CFG, hist_cfg=HIST_CFG).plot(
        "v",
        _synthetic(),
        str(out),
        want_data=True,
        custom={"datasim_text": "CMS Preliminary"},
        scale=10.0,
    )
    assert out.exists() and out.stat().st_size > 0


def test_stacked_render_blinded_no_ratio(tmp_path):
    out = tmp_path / "v_blind.png"
    Plotter(PAGE_CFG, hist_cfg=HIST_CFG).plot(
        "v",
        _synthetic(),
        str(out),
        want_data=False,
        custom={"datasim_text": "CMS Simulation"},
    )
    assert out.exists() and out.stat().st_size > 0


def _total_unc(histograms, rel=0.2):
    """Summed background with a stat + syst per-bin uncertainty.

    Mirrors what FLAF's ``GetTotalUncertaintyHist`` builds: the statistical variance of the
    summed background plus the systematic envelopes in quadrature -- which is what makes the
    full band strictly enclose the stat-only one.
    """
    bkgs = [h for h, _, _, group in histograms.values() if group == "backgrounds"]
    values = sum(h.values for h in bkgs)
    stat_var = sum(h.variances for h in bkgs)
    return Hist1D(
        bkgs[0].edges,
        values,
        variances=stat_var + (rel * values) ** 2,
        name="total_unc",
    )


def test_total_unc_drives_band_and_ratio(tmp_path):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    histograms = _synthetic()
    spec = StackedPlotter(PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG), None).build_spec(
        "v", histograms, True, {}, None, _total_unc(histograms)
    )
    # The band histogram carries the full uncertainty, not the stat-only stack error.
    assert spec.bkg_total_unc is not None
    assert np.allclose(spec.bkg_total_unc.values, spec.bkg_total.values)
    assert np.allclose(
        spec.bkg_total_unc.errors,
        np.hypot(spec.bkg_total.errors, 0.2 * spec.bkg_total.values),
    )
    assert not np.allclose(spec.bkg_total_unc.errors, spec.bkg_total.errors)
    assert "stat + syst" in spec.unc_band.label
    assert spec.draw_ratio

    # A second, stat-only band is drawn on top, styled differently from the total one.
    assert spec.stat_band is not None
    assert spec.stat_band.hatch != spec.unc_band.hatch
    assert spec.stat_band.root_fill_style != spec.unc_band.root_fill_style
    assert spec.stat_band.label != spec.unc_band.label

    out = tmp_path / "v_unc.pdf"
    Plotter(PAGE_CFG, hist_cfg=HIST_CFG).plot(
        "v",
        histograms,
        str(out),
        want_data=True,
        custom={"datasim_text": "CMS Preliminary"},
        total_unc=_total_unc(histograms),
    )
    assert out.exists() and out.stat().st_size > 0


def test_total_unc_gives_ratio_panel_when_blinded(tmp_path):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    histograms = _synthetic()
    config = PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG)
    # Blinded and without the full uncertainty there is nothing to put in a ratio panel.
    assert (
        not StackedPlotter(config, None)
        .build_spec("v", histograms, False, {}, None)
        .draw_ratio
    )
    # With it, the panel shows the relative uncertainty band around 1.
    spec = StackedPlotter(config, None).build_spec(
        "v", histograms, False, {}, None, _total_unc(histograms)
    )
    assert spec.draw_ratio and spec.data is None
    assert spec.ratio_title == "Unc./Bkg"

    out = tmp_path / "v_blind_unc.png"
    Plotter(PAGE_CFG, hist_cfg=HIST_CFG).plot(
        "v",
        histograms,
        str(out),
        want_data=False,
        custom={"datasim_text": "CMS Simulation"},
        total_unc=_total_unc(histograms),
    )
    assert out.exists() and out.stat().st_size > 0


def test_bands_are_nested_and_stat_only_without_total_unc():
    from PlotKit.backends import _bands
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    config = PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG)
    histograms = _synthetic()

    # Without the full uncertainty nothing changes: one stat-only band, as before.
    spec = StackedPlotter(config, None).build_spec("v", histograms, True, {}, None)
    bands = _bands(spec)
    assert len(bands) == 1
    assert bands[0][0] is spec.bkg_total
    assert spec.stat_band is None

    # With it: two bands, widest first, so the narrower one stays visible on top.
    spec = StackedPlotter(config, None).build_spec(
        "v", histograms, True, {}, None, _total_unc(histograms)
    )
    bands = _bands(spec)
    assert [h.name for h, _ in bands] == ["total_unc", "total"]
    assert [s.label for _, s in bands] == [spec.unc_band.label, spec.stat_band.label]
    outer, inner = bands[0][0], bands[1][0]
    assert np.all(inner.errors <= outer.errors)


def test_stat_band_style_is_configurable():
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    page = {**PAGE_CFG, "bkg_unc_hist": {**PAGE_CFG["bkg_unc_hist"]}}
    page["bkg_unc_hist"].update(
        stat_fill_style=3005, stat_fill_color="kRed", stat_legend_title="MC stat."
    )
    spec = StackedPlotter(PlotConfig(page, hist_cfg=HIST_CFG), None).build_spec(
        "v", _synthetic(), True, {}, None, _total_unc(_synthetic())
    )
    assert spec.stat_band.root_fill_style == 3005
    assert spec.stat_band.label == "MC stat."


def test_string_backend_name_is_resolved():
    # A backend passed as a name string (e.g. CLI --backend mplhep) must be resolved to a
    # backend instance, not stored verbatim (which would crash in render_stacked).
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    plotter = StackedPlotter(PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG), "mplhep")
    assert hasattr(plotter.backend, "render_stacked")


def test_signal_scaled_to_background_integral():
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    config = PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG)
    spec = StackedPlotter(config, None).build_spec(
        "v", _synthetic(), want_data=True, custom={}, scale="bkg"
    )

    bkg_total = sum(e.hist.integral() for e in spec.backgrounds)
    assert bkg_total > 0
    # Each signal is normalised so its integral equals the summed background.
    for sig in spec.signals:
        assert sig.hist.integral() == pytest.approx(bkg_total, rel=1e-9)
    # The hardcoded "(x10)" suffix is replaced by the normalisation note (then the yield).
    assert spec.signals[0].label.startswith("HH (norm. to bkg) (yield ")


def test_cmsstyle_backend_optional(tmp_path):
    pytest.importorskip("ROOT")
    pytest.importorskip("cmsstyle")
    out = tmp_path / "v_root.pdf"
    Plotter(PAGE_CFG, hist_cfg=HIST_CFG, backend="cmsstyle").plot(
        "v",
        _synthetic(),
        str(out),
        want_data=True,
        custom={"datasim_text": "CMS Preliminary"},
        scale=10.0,
    )
    assert out.exists() and out.stat().st_size > 0


LABEL_LINES_CFG = dict(
    PAGE_CFG,
    page_setup=dict(
        PAGE_CFG["page_setup"],
        label_lines=[["ana_text"], ["ch_text", "cat_text"], ["customreg_text"]],
    ),
    ana_text={"text": "demo", "text_size": 0.03, "font": 62},
    ch_text={"text_size": 0.03, "font": 62},
    customreg_text={"text_size": 0.025, "font": 42},
)
CUSTOM_TEXTS = {
    "ch_text": "#mu#mu",
    "cat_text": "res2b",
    "customreg_text": "Signal Region",
}


def _labels(page_cfg):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    plotter = StackedPlotter(PlotConfig(page_cfg, hist_cfg=HIST_CFG), "mplhep")
    return plotter._labels(plotter.config, CUSTOM_TEXTS, True)


def test_label_lines_join_boxes_in_the_style_of_the_first():
    labels = _labels(LABEL_LINES_CFG)
    assert labels["extra_labels"] == ["demo", "#mu#mu, res2b", "Signal Region"]
    styles = labels["extra_label_styles"]
    assert [s["bold"] for s in styles] == [True, True, False]
    assert [s["fontsize"] for s in styles] == [18.0, 18.0, 15.0]


def test_without_label_lines_each_box_is_a_label_in_the_legend_size():
    labels = _labels(PAGE_CFG)
    assert labels["extra_labels"] == ["demo", "#mu#mu", "res2b", "Signal Region"]
    assert labels["extra_label_styles"] == []


def test_stacked_render_with_label_lines(tmp_path):
    out = tmp_path / "v_lines.pdf"
    Plotter(LABEL_LINES_CFG, hist_cfg=HIST_CFG).plot(
        "v", _synthetic(), str(out), want_data=True, custom=dict(CUSTOM_TEXTS)
    )
    assert out.exists() and out.stat().st_size > 0


def test_signal_label_options_and_legend_columns():
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    page = dict(
        PAGE_CFG,
        legend={"text_size": 0.03, "n_columns": 2},
        sgn_hist=dict(
            PAGE_CFG["sgn_hist"], show_yield=False, norm_legend_note="norm. to bkg"
        ),
    )
    spec = StackedPlotter(PlotConfig(page, hist_cfg=HIST_CFG), None).build_spec(
        "v", _synthetic(), True, {}, "bkg"
    )
    # no yield and no per-signal note: the note is one legend entry
    assert spec.signals[0].label == "HH"
    assert spec.legend_note == "norm. to bkg"
    assert spec.legend_ncol == 2

    # the defaults keep both on every signal
    spec = StackedPlotter(PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG), None).build_spec(
        "v", _synthetic(), True, {}, "bkg"
    )
    assert spec.signals[0].label.startswith("HH (norm. to bkg) (yield ")
    assert spec.legend_note == "" and spec.legend_ncol == 1


def test_headroom_keeps_a_fraction_of_the_axis_free():
    from PlotKit.backends import _log_y_max, _y_max
    from PlotKit.spec import StackSpec

    spec = StackSpec(y_min=0.0, y_min_log=1e-2, y_headroom=0.5, log_y_headroom=0.5)
    assert _y_max(spec, 100.0) == pytest.approx(200.0)
    # log: 1e-2 .. 1e2 is 4 decades, so the top is 4 more decades up
    assert _log_y_max(spec, 1e2) == pytest.approx(1e6)
    # unset: max_y_sf as before
    legacy = StackSpec(max_y_sf=1.5)
    assert _y_max(legacy, 100.0) == pytest.approx(150.0)


def test_the_cms_label_line_shares_one_baseline():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    from PlotKit.backends import MplhepBackend
    from PlotKit.spec import StackSpec

    fig, ax = plt.subplots()
    MplhepBackend()._cms_label(
        hep, ax, StackSpec(scope_text="Preliminary", lumi_text="1 fb^{-1}")
    )
    line = [t for t in ax.texts if t.get_text()]
    assert len(line) >= 2
    assert {t.get_verticalalignment() for t in line} == {"baseline"}
    assert all(t.get_transform() == line[0].get_transform() for t in line)
    plt.close(fig)


def test_a_bold_label_is_bold_inside_its_math_too():
    from PlotKit.backends import _bold_mathtext

    assert (
        _bold_mathtext(r"$\mathrm{hh}\rightarrow\mathrm{bbWW}$")
        == r"$\mathbf{\mathbf{hh}\rightarrow\mathbf{bbWW}}$"
    )
    assert _bold_mathtext("Signal Region") == "Signal Region"


@pytest.mark.parametrize(
    "page_update, key",
    [
        ({"page_setup": {"y_headroom": "45%"}}, "page_setup.y_headroom"),
        ({"page_setup": {"log_y_headroom": 40}}, "page_setup.log_y_headroom"),
        ({"page_setup": {"cms_label_fontsize": "big"}}, "cms_label_fontsize"),
        ({"ana_text": {"text": "demo", "font": "bold"}}, "ana_text.font"),
        ({"ana_text": {"text": "demo", "text_size": "3%"}}, "ana_text.text_size"),
    ],
)
def test_an_unreadable_style_value_raises_naming_its_key(page_update, key):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    page = dict(LABEL_LINES_CFG)
    for block, values in page_update.items():
        page[block] = dict(page.get(block, {}), **values)
    plotter = StackedPlotter(PlotConfig(page, hist_cfg=HIST_CFG), None)
    with pytest.raises(ValueError, match=key):
        plotter.build_spec("v", _synthetic(), True, dict(CUSTOM_TEXTS), None)


def test_a_headroom_outside_0_1_raises():
    from PlotKit.backends import _log_y_max, _y_max
    from PlotKit.spec import StackSpec

    for headroom in (0.0, 1.0, -0.2, 40.0, float("nan")):
        with pytest.raises(ValueError, match="y_headroom"):
            _y_max(StackSpec(y_headroom=headroom), 100.0)
        with pytest.raises(ValueError, match="log_y_headroom"):
            _log_y_max(StackSpec(log_y_headroom=headroom), 100.0)


@pytest.mark.parametrize(
    "is_data, scope_line",
    [(True, "Preliminary"), (False, "Simulation Preliminary")],
)
def test_the_cms_label_draws_the_scope(is_data, scope_line):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    from PlotKit.backends import MplhepBackend, _scope_line
    from PlotKit.spec import StackSpec

    spec = StackSpec(scope_text="Preliminary", lumi_text="1 fb^{-1}", is_data=is_data)
    assert _scope_line(spec) == scope_line
    fig, ax = plt.subplots()
    MplhepBackend()._cms_label(hep, ax, spec)
    # drawn with text= (label= would drop it), as "Simulation <scope>" without data
    assert scope_line in [t.get_text() for t in ax.texts]
    plt.close(fig)


def _cms_line(scope, is_data, lumi="109.9 fb^{-1} (13.6 TeV)"):
    """Draw the CMS line as render_stacked does (CMS style, 10x10 canvas); return the figure,
    its texts by content and the style's default font size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    from PlotKit.backends import MplhepBackend
    from PlotKit.spec import StackSpec

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(10, 10))
        spec = StackSpec(scope_text=scope, lumi_text=lumi, is_data=is_data)
        MplhepBackend()._cms_label(hep, ax, spec)
        default = float(matplotlib.rcParams["font.size"])
    by_text = {t.get_text(): t for t in ax.texts if t.get_text()}
    return fig, by_text, default


def test_a_cms_line_that_fits_keeps_mplhep_sizes():
    import matplotlib.pyplot as plt

    fig, by_text, default = _cms_line("Preliminary", True)
    assert by_text["Preliminary"].get_fontsize() == default
    plt.close(fig)


def test_a_cms_line_that_would_hit_the_lumi_is_shrunk_to_fit():
    import matplotlib.pyplot as plt

    from PlotKit.backends import _CMS_LABEL_MIN_SCALE

    # H_mumu without data: "CMS Simulation Private work" ran 22-58 px into the lumi
    fig, by_text, default = _cms_line("Private work", False)
    size = by_text["Simulation Private work"].get_fontsize()
    assert _CMS_LABEL_MIN_SCALE * default <= size < default
    renderer = fig.canvas.get_renderer()
    left = max(
        by_text[t].get_window_extent(renderer).x1
        for t in ("CMS", "Simulation Private work")
    )
    lumi = next(t for k, t in by_text.items() if "fb" in k)
    assert left < lumi.get_window_extent(renderer).x0
    # still one baseline, in one transform
    assert {t.get_verticalalignment() for t in by_text.values()} == {"baseline"}
    plt.close(fig)


def test_a_cms_line_too_long_to_fit_stops_at_the_floor_and_warns():
    import matplotlib.pyplot as plt

    from PlotKit.backends import _CMS_LABEL_MIN_SCALE

    scope = "Private work " * 6
    with pytest.warns(UserWarning, match="overlaps the lumi"):
        fig, by_text, default = _cms_line(scope, True)
    size = by_text[" ".join(scope.split())].get_fontsize()
    assert size == pytest.approx(_CMS_LABEL_MIN_SCALE * default)
    plt.close(fig)


NOTE_CFG = dict(
    LABEL_LINES_CFG,
    legend={"text_size": 0.03, "n_columns": 2},
    sgn_hist=dict(
        PAGE_CFG["sgn_hist"], show_yield=False, norm_legend_note="Sig. norm."
    ),
)


def test_mplhep_draws_simulation_the_note_and_the_columns(tmp_path, monkeypatch):
    import matplotlib.figure

    seen = {}

    def keep(fig, *args, **kwargs):
        ax = fig.axes[0]
        legend = ax.get_legend()
        seen["texts"] = [t.get_text() for t in ax.texts]
        seen["legend"] = [t.get_text() for t in legend.get_texts()]
        seen["ncol"] = legend._ncols

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", keep)
    Plotter(NOTE_CFG, hist_cfg=HIST_CFG).plot(
        "v", _synthetic(), str(tmp_path / "v.pdf"), False, dict(CUSTOM_TEXTS), "bkg"
    )
    assert "Simulation Preliminary" in seen["texts"]
    assert seen["legend"][-1] == "Sig. norm."
    assert "HH" in seen["legend"]
    assert seen["ncol"] == 2


def test_cmsstyle_bolds_labels_and_says_it_ignores_their_sizes():
    from PlotKit.backends import CmsstyleBackend
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    plotter = StackedPlotter(PlotConfig(LABEL_LINES_CFG, hist_cfg=HIST_CFG), None)
    spec = plotter.build_spec("v", _synthetic(), True, dict(CUSTOM_TEXTS), None)
    with pytest.warns(UserWarning, match="label_lines text_size"):
        labels = CmsstyleBackend._additional_info(spec)
    assert labels == ["#bf{demo}", "#bf{#mu#mu, res2b}", "Signal Region"]


def test_cmsstyle_draws_simulation_the_note_and_the_columns(tmp_path, monkeypatch):
    pytest.importorskip("ROOT")
    CMS = pytest.importorskip("cmsstyle")
    import cmsstyle.cmsstyle as cms_module

    seen = {}

    def keep(canvas, *args, **kwargs):
        pads = [canvas] + [
            p for p in canvas.GetListOfPrimitives() if p.InheritsFrom("TPad")
        ]
        legend = [
            p
            for pad in pads
            for p in pad.GetListOfPrimitives()
            if p.InheritsFrom("TLegend")
        ][0]
        seen["legend"] = [e.GetLabel() for e in legend.GetListOfPrimitives()]
        seen["ncol"] = legend.GetNColumns()
        seen["extra_text"] = cms_module.extraText

    monkeypatch.setattr(CMS, "SaveCanvas", keep)
    with pytest.warns(UserWarning, match="cmsstyle backend"):
        Plotter(NOTE_CFG, hist_cfg=HIST_CFG, backend="cmsstyle").plot(
            "v", _synthetic(), str(tmp_path / "v.pdf"), False, dict(CUSTOM_TEXTS), "bkg"
        )
    assert seen["extra_text"] == "Simulation Preliminary"
    assert seen["legend"][-1] == "Sig. norm."
    assert "HH" in seen["legend"]
    assert seen["ncol"] == 2
