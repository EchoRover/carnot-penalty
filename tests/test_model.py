"""Unit tests for the Carnot-penalty model.  Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import carnot_penalty as cp  # noqa: E402


# ---------------- psychrometrics ----------------
def test_saturation_pressure_known_points():
    # steam-table values: 20 C -> 2339 Pa, 30 C -> 4246 Pa, 40 C -> 7384 Pa
    assert cp.p_sat(20.0) == pytest.approx(2339, rel=0.005)
    assert cp.p_sat(30.0) == pytest.approx(4246, rel=0.005)
    assert cp.p_sat(40.0) == pytest.approx(7384, rel=0.005)


@pytest.mark.parametrize("T,rh,twb", [(25.0, 50.0, 17.9), (30.0, 50.0, 22.0), (35.0, 40.0, 24.0)])
def test_wet_bulb_matches_psychrometric_chart(T, rh, twb):
    td = cp.dew_point_from_rh(T, rh)
    assert float(cp.wet_bulb(T, td)) == pytest.approx(twb, abs=0.3)


def test_wet_bulb_equals_dry_bulb_when_saturated():
    T = np.array([10.0, 25.0, 40.0])
    assert np.allclose(cp.wet_bulb(T, T), T, atol=1e-6)


def test_wet_bulb_is_between_dew_point_and_dry_bulb():
    rng = np.random.default_rng(0)
    T = rng.uniform(5, 50, 2000)
    td = T - rng.uniform(0, 30, 2000)
    twb = cp.wet_bulb(T, td)
    assert np.all(twb >= td - 1e-6) and np.all(twb <= T + 1e-6)


def test_wet_bulb_agrees_with_independent_stull_formula():
    T, rh = np.meshgrid(np.linspace(20, 45, 26), np.linspace(20, 90, 15))
    twb = cp.wet_bulb(T, cp.dew_point_from_rh(T, rh))
    assert np.max(np.abs(twb - cp.wet_bulb_stull(T, rh))) < 1.0


def test_dew_point_roundtrip():
    T, rh = 33.0, 62.0
    td = cp.dew_point_from_rh(T, rh)
    assert 100 * cp.p_sat(td) / cp.p_sat(T) == pytest.approx(rh, rel=1e-9)


# ---------------- chiller ----------------
@pytest.mark.parametrize("ch", [cp.WATER_COOLED_PHYS, cp.AIR_COOLED_PHYS])
def test_cop_recovers_reference_and_decreases_with_sink(ch):
    assert float(ch.cop(ch.t_ref)) == pytest.approx(ch.cop_ref)
    t = np.linspace(ch.t_ref - 15, ch.t_ref + 15, 50)
    assert np.all(np.diff(ch.cop(t)) < 0)


@pytest.mark.parametrize("ch", [cp.WATER_COOLED_PHYS, cp.AIR_COOLED_PHYS])
def test_cop_never_exceeds_carnot(ch):
    # second law: implied second-law efficiency must be below 1
    assert 0.0 < ch.eta_second_law() < 1.0
    t = np.linspace(ch.t_ref - 15, ch.t_ref + 15, 50)
    carnot = ch.t_evap_K / ch.lift(t)
    assert np.all(ch.cop(t) < carnot)


def test_sensitivity_is_one_over_lift():
    ch = cp.WATER_COOLED_PHYS
    h = 1e-4
    s = -(np.log(ch.cop(ch.t_ref + h)) - np.log(ch.cop(ch.t_ref - h))) / (2 * h)
    assert s == pytest.approx(1.0 / ch.lift_ref, rel=1e-6)


def test_ice_making_costs_more_than_chilled_water():
    ch = cp.WATER_COOLED_PHYS
    assert ch.cop(28.0, dT_charge=12.0) < ch.cop(28.0)


# ---------------- Carnot penalty ----------------
def test_penalty_is_zero_for_flat_sink():
    ch = cp.AIR_COOLED_PHYS
    cop = ch.cop(np.full(24, 35.0))
    w_day = np.r_[np.zeros(8), np.ones(8), np.zeros(8)]
    assert cp.carnot_penalty(cop, w_day, 1 - w_day) == pytest.approx(0.0, abs=1e-12)


def test_closed_form_is_exact_for_two_level_day():
    # sink = 40 C in the solar window, 30 C at night  ->  pi = (40-30)/lift(30) exactly
    ch = cp.AIR_COOLED_PHYS
    w_day = np.r_[np.zeros(8), np.ones(8), np.zeros(8)]
    t = np.where(w_day > 0, 40.0, 30.0)
    exact = cp.carnot_penalty(ch.cop(t), w_day, 1 - w_day)
    closed = cp.carnot_penalty_closed_form(40.0, 30.0, float(ch.lift(30.0)))
    assert exact == pytest.approx(closed, rel=1e-12)
    assert exact > 0


def test_penalty_negative_when_solar_hours_have_cooler_sink():
    ch = cp.WATER_COOLED_PHYS
    w_day = np.r_[np.zeros(8), np.ones(8), np.zeros(8)]
    t = np.where(w_day > 0, 24.0, 27.0)
    assert cp.carnot_penalty(ch.cop(t), w_day, 1 - w_day) < 0


# ---------------- storage choice ----------------
def test_breakeven_consistency():
    eta_t, eta_b = 0.97, 0.85
    pi_star = cp.breakeven_penalty(eta_t, eta_b)
    # exactly at break-even the cold tank has the same effective efficiency as the battery
    assert float(cp.cold_storage_efficiency(pi_star, eta_t)) == pytest.approx(eta_b)
    assert float(cp.cold_storage_efficiency(pi_star - 0.01, eta_t)) > eta_b
    assert float(cp.cold_storage_efficiency(pi_star + 0.01, eta_t)) < eta_b


# ---------------- chiller-library reader ----------------
IDF_SNIPPET = """
Chiller:Electric:EIR,
    Test Chiller,            !- Name
    1000000,                 !- Reference Capacity {W}
    6.0,                     !- Reference COP {W/W}
    6.67,                    !- Reference Leaving Chilled Water Temperature {C}
    29.44,                   !- Reference Entering Condenser Fluid Temperature {C}
    0.04, 0.05,
    Test CAPFT, Test EIRFT, Test EIRFPLR,
    0.1, 1.0, 1.0, 0.2, a, b, c, d,
    WaterCooled;             !- Condenser Type

Curve:Biquadratic,
    Test EIRFT,              !- Name
    0.4, 0.0, 0.0, 0.02, 0.0, 0.0,   !- EIRFT = 0.4 + 0.02*Tcond
    4, 10, 15, 40;
"""


def test_library_reader_and_sensitivity():
    lib = cp.load_chiller_library(IDF_SNIPPET)
    assert len(lib) == 1 and lib[0]["condenser"] == "WaterCooled"
    ch = lib[0]
    assert cp.library_cop_ratio(ch, ch["t_cond_ref"]) == pytest.approx(1.0)
    # EIRFT = 0.4 + 0.02*T  ->  s = 0.02 / (0.4 + 0.02*29.44)
    assert cp.library_sensitivity(ch) == pytest.approx(0.02 / (0.4 + 0.02 * 29.44), rel=1e-3)


# ---------------- the data shipped with the submission ----------------
DATA = Path(__file__).resolve().parents[1] / "data"
STATION_YEARS = ["OMAA_2022", "OMAA_2023", "OMAD_2022", "OMAD_2023", "OMDB_2023", "OMSJ_2023"]


def test_water_cooled_library_is_read():
    lib = cp.load_chiller_library((DATA / "Chillers.idf").read_text(errors="ignore"))
    assert len(lib) >= 150
    assert {ch["condenser"] for ch in lib} == {"WaterCooled"}


def test_air_cooled_library_is_read_and_every_machine_is_plausible():
    lib = cp.load_chiller_library((DATA / "AirCooledChiller.idf").read_text(errors="ignore"))
    assert len(lib) >= 100
    assert {ch["condenser"] for ch in lib} == {"AirCooled"}
    s = np.array([cp.library_sensitivity(ch) for ch in lib])
    assert np.all((s > 0.01) & (s < 0.05))        # each machine loses between 1 and 5 % of COP per kelvin


@pytest.mark.parametrize("name", STATION_YEARS)
def test_station_weather_is_complete_physical_and_in_local_time(name):
    import pandas as pd
    wx = pd.read_csv(DATA / "station_hourly" / f"{name}.csv", comment="#")
    assert len(wx) == 8760
    assert list(wx.columns) == ["month", "day", "hour", "T_C", "Td_C", "ghi_Wm2"]
    assert wx.T_C.between(0, 55).all()
    assert (wx.Td_C <= wx.T_C + 1e-9).all() and (wx.ghi_Wm2 >= 0).all()
    assert wx.ghi_Wm2[wx.hour.isin([0, 1, 2, 3])].max() == 0         # no sun in the small hours
    assert wx.groupby("hour").ghi_Wm2.mean().idxmax() in (11, 12)    # sun highest around local noon
    assert wx.groupby("hour").T_C.mean().idxmax() in (12, 13, 14, 15)   # hottest in the early afternoon


def test_wet_bulb_two_term_response():
    # the wet-bulb follows the dry-bulb weakly and the dew point strongly; the linear estimate is accurate
    T, Td = 34.0, 21.5
    a = float(cp.wet_bulb(T + 0.5, Td) - cp.wet_bulb(T - 0.5, Td))
    b = float(cp.wet_bulb(T, Td + 0.5) - cp.wet_bulb(T, Td - 0.5))
    assert 0.15 < a < 0.40 and 0.45 < b < 0.80
    exact = float(cp.wet_bulb(T + 5.0, Td - 4.6) - cp.wet_bulb(T, Td))
    assert exact == pytest.approx(a * 5.0 + b * (-4.6), abs=0.35)
    assert exact < 0            # hotter but drier air has a LOWER wet-bulb temperature


def test_two_term_coefficients_match_the_analytic_formula():
    # a = 1 / (1 + (h_fg/c_p) Ws'(Twb)),  b = (1 - a) Ws'(Td) / Ws'(Twb)
    T, Td = 34.0, 21.5
    twb = float(cp.wet_bulb(T, Td))

    def dWs(t, h=1e-3):
        return float(cp.humidity_ratio(cp.p_sat(t + h)) - cp.humidity_ratio(cp.p_sat(t - h))) / (2 * h)

    W = float(cp.humidity_ratio(cp.p_sat(Td)))
    a_formula = 1.0 / (1.0 + (2501.0 - 2.326 * twb) / (1.006 + 1.86 * W) * dWs(twb))
    b_formula = (1.0 - a_formula) * dWs(Td) / dWs(twb)
    a = float(cp.wet_bulb(T + 0.5, Td) - cp.wet_bulb(T - 0.5, Td))
    b = float(cp.wet_bulb(T, Td + 0.5) - cp.wet_bulb(T, Td - 0.5))
    assert a == pytest.approx(a_formula, abs=0.02)
    assert b == pytest.approx(b_formula, abs=0.03)
