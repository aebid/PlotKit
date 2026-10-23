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
    _need("ROOT")
    _need("cmsstyle")
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


def test_each_math_segment_of_a_bold_label_is_bolded_and_it_renders(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from PlotKit.backends import _bold_mathtext

    # two native-mathtext boxes joined on one label line
    bold = _bold_mathtext(r"$\mu\mu$, $m_{jj} > 400$")
    assert bold == r"$\mathbf{\mu\mu}$, $\mathbf{m_{jj} > 400}$"
    fig, ax = plt.subplots()
    ax.text(0.1, 0.5, bold, fontweight="bold")
    fig.savefig(
        tmp_path / "bold.png"
    )  # raised "Expected '}'" when bolded as one segment
    plt.close(fig)
    # an escaped \$ is text, not a segment
    assert _bold_mathtext(r"cost \$5") == r"cost \$5"


def test_a_bold_label_that_would_not_parse_stays_unbolded(monkeypatch):
    from PlotKit import rootcompat as rc
    from PlotKit.backends import _bold_mathtext

    monkeypatch.setattr(rc, "_mathtext_parses", lambda s: False)
    assert _bold_mathtext(r"$\mu\mu$") == r"$\mu\mu$"


@pytest.mark.parametrize(
    "page_update, key",
    [
        ({"page_setup": {"y_headroom": "45%"}}, "page_setup.y_headroom"),
        ({"page_setup": {"log_y_headroom": 40}}, "page_setup.log_y_headroom"),
        ({"page_setup": {"cms_label_fontsize": "big"}}, "cms_label_fontsize"),
        ({"ana_text": {"text": "demo", "font": "bold"}}, "ana_text.font"),
        ({"ana_text": {"text": "demo", "text_size": "3%"}}, "ana_text.text_size"),
        # a YAML true or .inf is not a size, nor true a font
        ({"page_setup": {"cms_label_fontsize": True}}, "cms_label_fontsize"),
        ({"page_setup": {"cms_label_fontsize": float("inf")}}, "cms_label_fontsize"),
        ({"ana_text": {"text": "demo", "text_size": True}}, "ana_text.text_size"),
        ({"ana_text": {"text": "demo", "font": True}}, "ana_text.font"),
        ({"page_setup": {"y_headroom": float("nan")}}, "page_setup.y_headroom"),
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


@pytest.mark.parametrize(
    "var, page, headroom, max_y_sf",
    [
        # nothing set: the page's max_y_sf default
        ({}, {}, None, 1.5),
        # the page's headroom beats the page's max_y_sf
        ({}, {"y_headroom": 0.4, "max_y_sf": 3}, 0.4, 3.0),
        # a variable's max_y_sf beats the page's headroom (e.g. --wantLogScale's 2000.2)
        ({"max_y_sf": 100}, {"y_headroom": 0.4}, None, 100.0),
        # a variable's headroom beats its own max_y_sf and the page's headroom
        ({"y_headroom": 0.25, "max_y_sf": 100}, {"y_headroom": 0.4}, 0.25, 100.0),
    ],
)
def test_a_variables_own_y_range_setting_beats_the_pages(var, page, headroom, max_y_sf):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    page_cfg = dict(PAGE_CFG, page_setup=dict(PAGE_CFG["page_setup"], **page))
    hist_cfg = {"v": dict(HIST_CFG["v"], **var)}
    spec = StackedPlotter(PlotConfig(page_cfg, hist_cfg=hist_cfg), None).build_spec(
        "v", _synthetic(), True, {}, None
    )
    assert spec.y_headroom == headroom
    assert spec.max_y_sf == max_y_sf


def test_a_bad_variable_headroom_names_the_variable():
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    hist_cfg = {"v": dict(HIST_CFG["v"], y_headroom=1.4)}
    plotter = StackedPlotter(PlotConfig(PAGE_CFG, hist_cfg=hist_cfg), None)
    with pytest.raises(ValueError, match=r"^v\.y_headroom = 1\.4"):
        plotter.build_spec("v", _synthetic(), True, {}, None)


def test_without_the_new_options_the_spec_is_as_before():
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter
    from PlotKit.spec import StackSpec

    spec = StackedPlotter(PlotConfig(PAGE_CFG, hist_cfg=HIST_CFG), None).build_spec(
        "v", _synthetic(), True, {}, 10.0
    )
    defaults = StackSpec()
    for field in (
        "y_headroom",
        "log_y_headroom",
        "cms_label_fontsize",
        "extra_label_styles",
        "legend_ncol",
        "legend_note",
    ):
        assert getattr(spec, field) == getattr(defaults, field), field


def test_the_log_headroom_top_is_smooth_near_y_min_log():
    from PlotKit.backends import _log_y_max
    from PlotKit.spec import StackSpec

    spec = StackSpec(y_min_log=1e-2, log_y_headroom=0.5)
    # within a decade of y_min_log the free share is of that decade: the same top on both
    # sides of y_min_log (it jumped from 3.3 decades to 0.008 at y_min_log)
    tops = [_log_y_max(spec, y) for y in (0.0, 0.0099, 0.0101, 0.1)]
    assert tops == pytest.approx([1.0] * 4)
    # and it grows continuously from there
    assert _log_y_max(spec, 0.1001) == pytest.approx(1.0, rel=1e-2)
    assert _log_y_max(spec, 1e2) == pytest.approx(1e6)


@pytest.mark.parametrize(
    "label_lines, match",
    [
        (["ana_text", "ch_text"], "list of lines"),
        ("ana_text", "list of lines"),
        ([["ana_text"], []], "list of lines"),
        ([], "list of lines"),
        ([["ana_text"], ["ch_txt"]], r"not declared: \['ch_txt'\]"),
        # without text_boxes only the label boxes: not the era file's or a style block
        ([["ana_text"], ["channel_text", "cat_text"]], r"\['channel_text'\]"),
        ([["ana_text"], ["legend"]], r"\['legend'\]"),
    ],
)
def test_label_lines_must_be_lines_of_declared_boxes(label_lines, match):
    page = dict(
        LABEL_LINES_CFG,
        page_setup=dict(LABEL_LINES_CFG["page_setup"], label_lines=label_lines),
    )
    with pytest.raises(ValueError, match=match):
        _labels(page)


def test_label_lines_boxes_are_checked_against_text_boxes():
    page = dict(
        LABEL_LINES_CFG,
        page_setup=dict(
            LABEL_LINES_CFG["page_setup"], text_boxes=["ana_text", "ch_text"]
        ),
    )
    with pytest.raises(ValueError, match="cat_text"):
        _labels(page)


@pytest.mark.parametrize(
    "block, values, key",
    [
        ("sgn_hist", {"show_yield": "false"}, "show_yield"),
        ("legend", {"n_columns": 0}, "n_columns"),
        ("legend", {"n_columns": None}, "n_columns"),
        ("legend", {"n_columns": 1.5}, "n_columns"),
        ("page_setup", {"cms_label_fontsize": 0}, "cms_label_fontsize"),
    ],
)
def test_a_misread_option_raises_naming_its_key(block, values, key):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    page = dict(PAGE_CFG, **{block: dict(PAGE_CFG.get(block, {}), **values)})
    plotter = StackedPlotter(PlotConfig(page, hist_cfg=HIST_CFG), None)
    with pytest.raises(ValueError, match=key):
        plotter.build_spec("v", _synthetic(), True, {}, None)


def test_a_named_cms_label_size_is_used(tmp_path):
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    page = dict(
        PAGE_CFG, page_setup=dict(PAGE_CFG["page_setup"], cms_label_fontsize="large")
    )
    spec = StackedPlotter(PlotConfig(page, hist_cfg=HIST_CFG), None).build_spec(
        "v", _synthetic(), True, {}, None
    )
    assert spec.cms_label_fontsize == "large"
    out = tmp_path / "large.png"
    Plotter(page, hist_cfg=HIST_CFG).plot("v", _synthetic(), str(out), True, {})
    assert out.exists() and out.stat().st_size > 0


def test_an_empty_per_plot_label_is_no_label():
    page = dict(LABEL_LINES_CFG, customreg_text={"text": "Default region"})
    texts = dict(CUSTOM_TEXTS, customreg_text="")
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    plotter = StackedPlotter(PlotConfig(page, hist_cfg=HIST_CFG), "mplhep")
    assert plotter._labels(plotter.config, texts, True)["extra_labels"] == [
        "demo",
        "#mu#mu, res2b",
    ]
    # not passed at all: the configured text
    del texts["customreg_text"]
    labels = plotter._labels(plotter.config, texts, True)["extra_labels"]
    assert labels[-1] == "Default region"


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


def _need(module):
    """A module the cmsstyle tests need: they skip without it, unless PLOTKIT_REQUIRE_ROOT is
    set -- the CI job with ROOT -- where its absence fails them instead."""
    import importlib
    import os

    if os.environ.get("PLOTKIT_REQUIRE_ROOT"):
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _latex_width(ROOT, pad, text, size, font=42):
    """The width of a TLatex text on ``pad``, in NDC of the pad's width."""
    import ctypes

    latex = ROOT.TLatex(0, 0, text)
    latex.SetTextFont(font)
    latex.SetTextSize(size)
    w, h = ctypes.c_uint(0), ctypes.c_uint(0)
    latex.GetBoundingBox(w, h)
    return w.value / (pad.GetWw() * pad.GetAbsWNDC())


def _cmsstyle_plot(
    monkeypatch,
    tmp_path,
    cfg,
    histograms,
    want_data,
    scale,
    custom=None,
    total_unc=None,
):
    """Render through the cmsstyle backend and return what the saved canvas holds: the
    legend (labels row by row, columns, box, text size, entry widths), the frame top, the CMS
    line's texts {text: size}, the ratio points' errors, and cmsstyle's CMS text after.
    """
    ROOT = _need("ROOT")
    CMS = _need("cmsstyle")
    import cmsstyle.cmsstyle as cms_module

    seen = {}

    def keep(canvas, *args, **kwargs):
        pads = [p for p in canvas.GetListOfPrimitives() if p.InheritsFrom("TPad")]
        main = pads[0] if pads else canvas
        primitives = list(main.GetListOfPrimitives())
        (legend,) = [p for p in primitives if p.InheritsFrom("TLegend")]
        labels = [e.GetLabel() for e in legend.GetListOfPrimitives()]
        seen.update(
            legend=labels,
            ncol=legend.GetNColumns(),
            box=(legend.GetX1NDC(), legend.GetX2NDC()),
            margin=legend.GetMargin(),
            text_size=legend.GetTextSize(),
            widths=[
                _latex_width(ROOT, main, label, legend.GetTextSize())
                for label in labels
            ],
            right_edge=1 - main.GetRightMargin(),
            top=main.GetUymax(),
            texts={
                p.GetTitle(): p.GetTextSize()
                for p in primitives
                if p.InheritsFrom("TLatex")
            },
            # {text: (left, right)} in NDC; a right-aligned text (the lumi) ends at its x
            extents={
                p.GetTitle(): (
                    (p.GetX() - w, p.GetX())
                    if p.GetTextAlign() // 10 == 3
                    else (p.GetX(), p.GetX() + w)
                )
                for p in primitives
                if p.InheritsFrom("TLatex")
                for w in [
                    _latex_width(
                        ROOT, main, p.GetTitle(), p.GetTextSize(), p.GetTextFont()
                    )
                ]
            },
            cms_text=cms_module.cmsText,
        )
        if len(pads) > 1:
            (ratio,) = [
                p for p in pads[1].GetListOfPrimitives() if p.GetName() == "ratio"
            ]
            seen["ratio_errors"] = [
                ratio.GetBinError(i) for i in range(1, ratio.GetNbinsX() + 1)
            ]

    monkeypatch.setattr(CMS, "SaveCanvas", keep)
    Plotter(cfg, hist_cfg=HIST_CFG, backend="cmsstyle").plot(
        "v",
        histograms,
        str(tmp_path / "v.pdf"),
        want_data,
        dict(CUSTOM_TEXTS if custom is None else custom),
        scale,
        total_unc=total_unc,
    )
    return seen


def _mplhep_plot(
    monkeypatch,
    tmp_path,
    cfg,
    histograms,
    want_data,
    scale,
    custom=None,
    total_unc=None,
):
    """Render through the mplhep backend and return the legend labels in reading order,
    the y top, the texts on the axes and the ratio points' errors."""
    import matplotlib.figure

    seen = {}

    def keep(fig, *args, **kwargs):
        ax = fig.axes[0]
        seen.update(
            legend=[t.get_text() for t in ax.get_legend().get_texts()],
            ncol=ax.get_legend()._ncols,
            top=ax.get_ylim()[1],
            texts=[t.get_text() for t in ax.texts],
        )
        if len(fig.axes) > 1:
            (points,) = [c for c in fig.axes[1].containers]
            seen["ratio_errors"] = list(points.lines[2][0].get_segments())

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", keep)
    Plotter(cfg, hist_cfg=HIST_CFG).plot(
        "v",
        histograms,
        str(tmp_path / "v.pdf"),
        want_data,
        dict(CUSTOM_TEXTS if custom is None else custom),
        scale,
        total_unc=total_unc,
    )
    return seen


def _three_backgrounds():
    histograms = _synthetic()
    edges = np.linspace(0, 200, 21)
    histograms["VV"] = (
        Hist1D(edges, 5 * np.exp(-edges[:-1] / 60), name="vv"),
        "VV",
        "kOrange",
        "backgrounds",
    )
    return histograms


def test_cmsstyle_labels_are_bold_and_in_their_mplhep_sizes():
    from PlotKit.backends import _PT_PER_NDC, CmsstyleBackend
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    plotter = StackedPlotter(PlotConfig(LABEL_LINES_CFG, hist_cfg=HIST_CFG), None)
    spec = plotter.build_spec("v", _synthetic(), True, dict(CUSTOM_TEXTS), None)
    labels = CmsstyleBackend._additional_info(spec)
    assert [text for text, _, _ in labels] == [
        "#bf{demo}",
        "#bf{#mu#mu, res2b}",
        "Signal Region",
    ]
    # each label at its configured ROOT size
    assert [points / _PT_PER_NDC for _, points, _ in labels] == pytest.approx(
        [0.03, 0.03, 0.025]
    )


def test_column_major_legend_order():
    from PlotKit.backends import _column_major

    # matplotlib: columns [a b c d] [e f g]; a TLegend is filled row by row
    assert _column_major(list("abcdefg"), 2, "_") == list("aebfcgd_")
    assert _column_major(list("abc"), 2, "_") == list("acb_")
    assert _column_major(list("ab"), 3, "_") == list("ab")
    assert _column_major(list("abc"), 1, "_") == list("abc")


def test_root_text_translates_plotkit_labels_and_refuses_the_rest():
    from PlotKit.backends import _root_text

    assert _root_text("W #rightarrow #ell#nu + Jets") == (
        "W #rightarrow #font[12]{l}#nu + Jets"
    )
    assert _root_text("X#rightarrowHH spin‐0 M_{X}=400 GeV") == (
        "X#rightarrowHH spin-0 M_{X}=400 GeV"
    )
    with pytest.raises(ValueError, match=r"U\+00E9"):
        _root_text("café")


# HH_bbWW's plot group names (config/global.yaml, plot_process_groups)
HH_BBWW_GROUP_NAMES = [
    "Diboson + Triboson",
    "Single Higgs",
    "W #rightarrow #ell#nu + Jets",
    "DY m_{#ell#ell}#in[10,50] GeV",
    "DY m_{#ell#ell}>50 GeV",
    "DY",
    "Single top",
    "t#bar{t} + t#bar{t}V",
    "X#rightarrowHH spin‐0 M_{X}=400 GeV",
]


def test_hh_bbww_group_names_render_through_both_backends(tmp_path, monkeypatch):
    edges = np.linspace(0, 200, 21)
    histograms = {
        f"p{i}": (
            Hist1D(edges, (i + 1) * np.exp(-edges[:-1] / 80)),
            name,
            "kAzure-9",
            "signals" if "X#rightarrow" in name else "backgrounds",
        )
        for i, name in enumerate(HH_BBWW_GROUP_NAMES)
    }
    mpl = _mplhep_plot(monkeypatch, tmp_path, PAGE_CFG, histograms, False, 1.0)
    root = _cmsstyle_plot(monkeypatch, tmp_path, PAGE_CFG, histograms, False, 1.0)
    assert len(mpl["legend"]) == len(root["legend"]) == len(HH_BBWW_GROUP_NAMES) + 1
    labels = " ".join(root["legend"])
    assert "#ell" not in labels and labels.isascii()
    assert "W #rightarrow #font[12]{l}#nu + Jets" in root["legend"]


def test_both_backends_keep_the_signals_and_the_headroom_in_the_frame(
    tmp_path, monkeypatch
):
    cfg = dict(NOTE_CFG, page_setup=dict(NOTE_CFG["page_setup"], y_headroom=0.4))
    mpl = _mplhep_plot(monkeypatch, tmp_path, cfg, _synthetic(), True, "bkg")
    root = _cmsstyle_plot(monkeypatch, tmp_path, cfg, _synthetic(), True, "bkg")
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    spec = StackedPlotter(PlotConfig(cfg, hist_cfg=HIST_CFG), None).build_spec(
        "v", _synthetic(), True, dict(CUSTOM_TEXTS), "bkg"
    )
    peak = max(float(np.max(e.hist.values)) for e in spec.signals)
    assert peak > float(np.max(spec.bkg_total.values)), "the signal must be the highest"
    assert root["top"] == pytest.approx(mpl["top"])
    assert root["top"] >= peak / (1 - 0.4) * (1 - 1e-9)


def test_cmsstyle_legend_fits_its_columns_in_the_configured_size(tmp_path, monkeypatch):
    root = _cmsstyle_plot(monkeypatch, tmp_path, NOTE_CFG, _synthetic(), False, "bkg")
    assert root["text_size"] == pytest.approx(0.03)  # legend.text_size
    assert root["ncol"] == 2
    x1, x2 = root["box"]
    assert x2 <= root["right_edge"]
    column = (x2 - x1) / root["ncol"]
    assert max(root["widths"]) <= column * (1 - root["margin"])


def test_cmsstyle_legend_falls_back_to_one_column_when_two_do_not_fit(
    tmp_path, monkeypatch
):
    cfg = dict(NOTE_CFG, legend={"text_size": 0.04, "n_columns": 2})
    histograms = _synthetic()
    histograms["TT"] = (
        histograms["TT"][0],
        "t#bar{t} " + "x" * 40,
        "kSpring+5",
        "backgrounds",
    )
    root = _cmsstyle_plot(monkeypatch, tmp_path, cfg, histograms, False, "bkg")
    assert root["ncol"] == 1


def test_both_backends_list_the_legend_in_the_same_order(tmp_path, monkeypatch):
    from PlotKit.backends import _t

    histograms = _three_backgrounds()
    mpl = _mplhep_plot(monkeypatch, tmp_path, NOTE_CFG, histograms, True, "bkg")
    root = _cmsstyle_plot(monkeypatch, tmp_path, NOTE_CFG, histograms, True, "bkg")
    # read the TLegend's row-by-row cells column by column, as the mplhep legend reads
    ncol, cells = root["ncol"], root["legend"]
    reading = [label for c in range(ncol) for label in cells[c::ncol] if label]
    assert [_t(label) for label in reading] == [t for t in mpl["legend"] if t]
    # the top of the stack first: VV was added last
    assert reading[:3] == ["VV", "t#bar{t}", "DY #rightarrow ll"]


def test_both_backends_write_cms_for_a_private_scope(tmp_path, monkeypatch):
    cfg = dict(PAGE_CFG, scope_text={"text": "Private work"})
    for want_data in (True, False):
        mpl = _mplhep_plot(monkeypatch, tmp_path, cfg, _synthetic(), want_data, 10.0)
        assert "CMS" in mpl["texts"]
        for _ in range(2):  # and the next plot keeps it
            root = _cmsstyle_plot(
                monkeypatch, tmp_path, cfg, _synthetic(), want_data, 10.0
            )
            assert "CMS" in root["texts"]
            assert root["cms_text"] == "CMS"
        scope = "Private work" if want_data else "Simulation Private work"
        assert scope in root["texts"] and scope in mpl["texts"]


def test_cmsstyle_draws_the_configured_cms_and_label_sizes(tmp_path, monkeypatch):
    cfg = dict(
        LABEL_LINES_CFG,
        page_setup=dict(LABEL_LINES_CFG["page_setup"], cms_label_fontsize=20),
    )
    root = _cmsstyle_plot(monkeypatch, tmp_path, cfg, _synthetic(), True, 10.0)
    texts = root["texts"]
    # "CMS" at 1.3 x 20 pt and the scope at about 20 pt, as mplhep draws them
    assert texts["CMS"] == pytest.approx(1.3 * 20 / 600)
    assert texts["Preliminary"] == pytest.approx(0.76 * 1.3 * 20 / 600)
    drawn = {t: texts[t] for t in ("#bf{demo}", "#bf{#mu#mu, res2b}", "Signal Region")}
    assert drawn == pytest.approx(
        {"#bf{demo}": 0.03, "#bf{#mu#mu, res2b}": 0.03, "Signal Region": 0.025}
    )
    # and without the option, cmsstyle's own size again
    root = _cmsstyle_plot(
        monkeypatch, tmp_path, LABEL_LINES_CFG, _synthetic(), True, 10.0
    )
    import cmsstyle.cmsstyle as cms_module

    assert cms_module.cmsTextSize == pytest.approx(0.75)


def test_both_backends_show_the_data_error_alone_in_the_ratio(tmp_path, monkeypatch):
    edges = np.array([0.0, 1.0, 2.0])
    histograms = {
        "B": (
            Hist1D(edges, [100.0, 9.0], [400.0, 36.0]),
            "B",
            "kAzure-9",
            "backgrounds",
        ),
        "data": (Hist1D(edges, [100.0, 9.0], [100.0, 9.0]), "data", "kBlack", "data"),
    }
    root = _cmsstyle_plot(monkeypatch, tmp_path, PAGE_CFG, histograms, True, 1.0)
    assert root["ratio_errors"] == pytest.approx([0.1, 1 / 3])
    mpl = _mplhep_plot(monkeypatch, tmp_path, PAGE_CFG, histograms, True, 1.0)
    half_lengths = [abs(seg[1][1] - seg[0][1]) / 2 for seg in mpl["ratio_errors"]]
    assert half_lengths == pytest.approx([0.1, 1 / 3])


def test_cmsstyle_draws_simulation_the_note_and_the_columns(tmp_path, monkeypatch):
    root = _cmsstyle_plot(monkeypatch, tmp_path, NOTE_CFG, _synthetic(), False, "bkg")
    assert "Simulation Preliminary" in root["texts"]
    ncol, cells = root["ncol"], root["legend"]
    reading = [label for c in range(ncol) for label in cells[c::ncol] if label]
    assert reading[-1] == "Sig. norm."
    assert "HH" in root["legend"]
    assert root["ncol"] == 2


def test_cmsstyle_shrinks_a_cms_line_that_would_run_into_the_lumi(
    tmp_path, monkeypatch
):
    _need("cmsstyle")
    import cmsstyle.cmsstyle as cms_module

    # a line that fits keeps cmsstyle's size of "CMS"
    _cmsstyle_plot(monkeypatch, tmp_path, PAGE_CFG, _synthetic(), True, 10.0)
    assert cms_module.cmsTextSize == pytest.approx(0.75)
    # a long one is shrunk until the scope ends before the lumi
    cfg = dict(PAGE_CFG, scope_text={"text": "Private work, a long scope"})
    root = _cmsstyle_plot(monkeypatch, tmp_path, cfg, _synthetic(), False, 10.0)
    assert cms_module.cmsTextSize < 0.75
    scope_end = root["extents"]["Simulation Private work, a long scope"][1]
    (lumi_start,) = [x for t, (x, _) in root["extents"].items() if "fb" in t]
    assert scope_end < lumi_start
    # and one that cannot fit stops at the floor, with mplhep's warning
    cfg = dict(PAGE_CFG, scope_text={"text": "Private work " + "x" * 60})
    with pytest.warns(UserWarning, match="overlaps the lumi text"):
        _cmsstyle_plot(monkeypatch, tmp_path, cfg, _synthetic(), False, 10.0)
    assert cms_module.cmsTextSize == pytest.approx(0.6 * 0.75)


def _mplhep_axes(monkeypatch, cfg, want_data=True, custom=None):
    """Render through the mplhep backend and return the main axes as saved."""
    import matplotlib.figure

    seen = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda fig, *args, **kwargs: seen.update(ax=fig.axes[0]),
    )
    Plotter(cfg, hist_cfg=HIST_CFG).plot(
        "v",
        _synthetic(),
        "unused.pdf",
        want_data,
        dict(CUSTOM_TEXTS if custom is None else custom),
        10.0,
    )
    return seen["ax"]


def test_mplhep_renders_under_the_pdf_svg_and_ps_backends(tmp_path):
    import matplotlib

    previous = matplotlib.get_backend()
    try:
        for backend in ("pdf", "svg", "ps"):
            matplotlib.use(backend)
            out = tmp_path / f"v.{backend}"
            Plotter(PAGE_CFG, hist_cfg=HIST_CFG).plot(
                "v", _synthetic(), str(out), True, dict(CUSTOM_TEXTS), 10.0
            )
            assert out.exists() and out.stat().st_size > 0
    finally:
        matplotlib.use(previous)


def test_mplhep_draws_the_cms_line_in_cms_label_fontsize(monkeypatch):
    cfg = dict(PAGE_CFG, page_setup=dict(PAGE_CFG["page_setup"], cms_label_fontsize=20))
    ax = _mplhep_axes(monkeypatch, cfg)
    sizes = {t.get_text(): t.get_fontsize() for t in ax.texts}
    # "CMS" at 1.3 x, the scope at 1 x the base size
    assert sizes["CMS"] == pytest.approx(26.0)
    assert sizes["Preliminary"] == pytest.approx(20.0)


def test_an_error_in_mplhep_label_is_not_swallowed(monkeypatch):
    import mplhep

    real = mplhep.cms.label

    def broken(*args, **kwargs):
        # fails as an mplhep without text= would; a retry without it would draw a line
        if "text" in kwargs:
            raise TypeError("label broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(mplhep.cms, "label", broken)
    with pytest.raises(TypeError, match="label broke"):
        _mplhep_axes(monkeypatch, PAGE_CFG)


def test_a_bold_label_is_drawn_bold_inside_its_math(monkeypatch):
    ax = _mplhep_axes(monkeypatch, LABEL_LINES_CFG)
    # "#mu#mu, res2b", converted to mathtext
    (mumu,) = [t for t in ax.texts if r"\mu" in t.get_text()]
    assert r"\mathbf" in mumu.get_text() and r"\mathrm" not in mumu.get_text()
    assert mumu.get_fontweight() == "bold"


def test_the_cms_line_shares_one_baseline_whatever_mplhep_offsets(monkeypatch):
    import mplhep.label

    # mplhep places the scope by the font's descent; with mplhep 1.1.3 and these fonts it is
    # zero, so force one: the line must still share one y
    monkeypatch.setattr(mplhep.label, "_descent_from_layout", lambda layout: 7.0)
    fig, by_text, _ = _cms_line("Preliminary", True)
    assert {t.get_position()[1] for t in by_text.values()} == {1.0}
    fig.clf()


def test_the_cms_line_keeps_half_the_scope_font_from_the_lumi():
    from PlotKit.backends import _cms_label_overlap, _renderer

    fig, by_text, _ = _cms_line("Preliminary", True)
    ax = fig.axes[0]
    cms, scope = by_text["CMS"], by_text["Preliminary"]
    (lumi,) = [t for text, t in by_text.items() if "fb" in text]
    renderer = _renderer(fig)
    gap = renderer.points_to_pixels(0.5 * scope.get_fontsize())
    expected = (
        scope.get_window_extent(renderer).x1 + gap - lumi.get_window_extent(renderer).x0
    )
    assert _cms_label_overlap(ax, cms, scope, lumi) == pytest.approx(expected)
    fig.clf()


def test_the_legend_clears_the_inward_ticks(monkeypatch):
    import matplotlib

    cfg = dict(PAGE_CFG, legend={"text_size": 0.017})  # 10.2 pt, below the 12 pt ticks
    ax = _mplhep_axes(monkeypatch, cfg)
    legend = ax.get_legend()
    fontsize = legend.get_texts()[0].get_fontsize()
    tick = float(matplotlib.rcParams["ytick.major.size"])
    assert tick > 0.5 * fontsize  # matplotlib's default padding would not clear it
    assert legend.borderaxespad * fontsize >= tick


def test_keep_together_moves_a_split_group_to_the_next_column():
    from PlotKit.backends import _keep_together

    # 8 entries in 2 columns: [0 1 2 3] [4 5 6 7]; the bands at 3 and 4 are split
    out = _keep_together(list("abcBBdef"), 3, 2, 2, "_")
    columns = np.array_split(np.arange(len(out)), 2)
    (column,) = [c for c in columns if out.index("B") in c]
    assert out.index("B") + 1 in column
    assert [e for e in out if e != "_"] == list("abcBBdef")
    # already together, one column, or a single band: unchanged
    assert _keep_together(list("aBBcde"), 1, 2, 2, "_") == list("aBBcde")
    assert _keep_together(list("abcBBdef"), 3, 2, 1, "_") == list("abcBBdef")
    assert _keep_together(list("abcBdef"), 3, 1, 2, "_") == list("abcBdef")


def test_both_backends_keep_the_two_bands_in_one_legend_column(tmp_path, monkeypatch):
    from PlotKit.backends import _root_text, _t
    from PlotKit.config import PlotConfig
    from PlotKit.plotters.stacked import StackedPlotter

    histograms = _three_backgrounds()
    total_unc = _total_unc(histograms)
    spec = StackedPlotter(PlotConfig(NOTE_CFG, hist_cfg=HIST_CFG), None).build_spec(
        "v", histograms, True, dict(CUSTOM_TEXTS), "bkg", total_unc
    )
    labels = [spec.unc_band.label, spec.stat_band.label]

    def columns(cells, ncol, row_major):
        if row_major:  # a TLegend's cells, filled row by row
            return [[c for c in cells[i::ncol] if c] for i in range(ncol)]
        return [[c for c in col if c] for col in np.array_split(cells, ncol)]

    mpl = _mplhep_plot(
        monkeypatch, tmp_path, NOTE_CFG, histograms, True, "bkg", total_unc=total_unc
    )
    for cells, ncol, row_major in [(mpl["legend"], mpl["ncol"], False)] + (
        [
            (root["legend"], root["ncol"], True)
            for root in [
                _cmsstyle_plot(
                    monkeypatch,
                    tmp_path,
                    NOTE_CFG,
                    histograms,
                    True,
                    "bkg",
                    total_unc=total_unc,
                )
            ]
        ]
        if _root_available()
        else []
    ):
        assert ncol == 2
        bands = [_root_text(b) if row_major else _t(b) for b in labels]
        assert set(bands) <= set(cells), cells
        (column,) = [c for c in columns(cells, ncol, row_major) if bands[0] in c]
        assert bands[1] in column, columns(cells, ncol, row_major)


def _root_available():
    import os

    if os.environ.get("PLOTKIT_REQUIRE_ROOT"):
        return True
    try:
        import ROOT  # noqa: F401
        import cmsstyle  # noqa: F401
    except Exception:
        return False
    return True
