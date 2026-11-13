#!/usr/bin/env python3
"""
run_analysis.py -- regenerates EVERY number, table and figure in the report.

    python scripts/run_analysis.py

Inputs  (data/):
    Chillers.idf, AirCooledChiller.idf     EnergyPlus chiller libraries (manufacturer-derived performance curves)
    station_hourly/<STATION>_<YEAR>.csv    measured hourly weather, built by scripts/make_station_weather.py
    reanalysis/abudhabi_reanalysis_*.csv   hourly reanalysis years (comparison only)
    weather_hourly.csv                     normals-based typical year from scripts/make_weather.py (comparison only)
    abudhabi_monthly_normals.csv           the 36 monthly normals behind that typical year
Outputs (results/): results.json, monthly_table.csv, datasets_table.csv, chiller_library.csv
        (report/):  figures/*.pdf and *.png, numbers.tex, table_*.tex  (every number quoted in the report)

Deterministic: fixed random seed, no network access.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import carnot_penalty as cp  # noqa: E402
from make_weather import haurwitz_ghi, LON, TZ_MERIDIAN  # noqa: E402

DATA, RES, REP = ROOT / "data", ROOT / "results", ROOT / "report"
FIG = REP / "figures"
for d in (RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

SEED = 2026
ETA_TANK = 0.97        # chilled-water tank round-trip efficiency (daily cycle)
ETA_BATT = 0.85        # Li-ion battery AC-AC round-trip efficiency
TOWER_APPROACH = 4.0   # K, cooling-tower approach: entering condenser water = wet-bulb + approach
COOL_SEASON = [5, 6, 7, 8, 9, 10]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
C_WC, C_AC, C_GREY, C_TD = "#1b6ca8", "#d1495b", "#555555", "#2a9d8f"

# Base case: four measured station-years in Abu Dhabi (airport and island, 2022 and 2023), pooled.
AD_SETS = ["OMAA_2022", "OMAA_2023", "OMAD_2022", "OMAD_2023"]
OTHER_SETS = ["OMDB_2023", "OMSJ_2023"]
LABEL = {"OMAA_2022": "Abu Dhabi airport 2022", "OMAA_2023": "Abu Dhabi airport 2023",
         "OMAD_2022": "Abu Dhabi island 2022", "OMAD_2023": "Abu Dhabi island 2023",
         "OMDB_2023": "Dubai airport 2023", "OMSJ_2023": "Sharjah airport 2023",
         "REAN_2022": "Reanalysis 2022", "REAN_2023": "Reanalysis 2023", "NORMALS": "Normals-based typical year"}
KIND = {**{k: "measured" for k in AD_SETS + OTHER_SETS}, "REAN_2022": "reanalysis", "REAN_2023": "reanalysis",
        "NORMALS": "synthetic"}

# The two chiller classes: library file, condenser type, heat sink, and the offsets used ONLY to estimate
# the physical lift of the library machines (condensing temperature above the entering condenser fluid).
LIBRARIES = {
    "wc": dict(file="Chillers.idf", condenser="WaterCooled", sink="wb", name="Water-cooled",
               approach=TOWER_APPROACH, cond_above=7.0),
    "ac": dict(file="AirCooledChiller.idf", condenser="AirCooled", sink="db", name="Air-cooled",
               approach=0.0, cond_above=14.0),
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.frameon": False})


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


# =============================================================================
# 1. Calibrate the chiller model on the manufacturer-derived libraries
# =============================================================================
def calibrate_chillers(out):
    rng = np.random.default_rng(SEED)
    models, cal, libs, frames = {}, {}, {}, []
    for key in ("wc", "ac"):
        spec = LIBRARIES[key]
        lib = cp.load_chiller_library((DATA / spec["file"]).read_text(errors="ignore"))
        lib = [ch for ch in lib if ch["condenser"] == spec["condenser"]]
        df = pd.DataFrame([dict(name=ch["name"], condenser=ch["condenser"], cop_ref=ch["cop_ref"],
                                t_chw_ref=ch["t_chw_ref"], t_cond_ref=ch["t_cond_ref"],
                                s_per_K=cp.library_sensitivity(ch)) for ch in lib])
        df["valid"] = (df.s_per_K > 0.002) & (df.s_per_K < 0.10) & (df.cop_ref > 1.5) & (df.cop_ref < 10)
        df["lift_eff_K"] = 1.0 / df.s_per_K.where(df.valid)
        frames.append(df)
        n_total = len(df)
        lib = [ch for ch, ok in zip(lib, df.valid) if ok]
        df = df[df.valid].reset_index(drop=True)
        idx = np.arange(len(df))
        s = df.s_per_K.values
        s_med = float(np.median(s))
        lift = 1.0 / s_med
        cop_ref = float(np.median(df.cop_ref.values))
        t_cond_ref = float(np.median(df.t_cond_ref.values))
        t_ref_sink = t_cond_ref - spec["approach"]
        models[key] = cp.Chiller(spec["name"], spec["sink"], cop_ref=cop_ref, t_ref=t_ref_sink, lift_ref=lift)

        # --- how well does 1/COP linear in sink temperature reproduce each real machine's curve? ---
        def curve_errors(lift_used, members):
            errs = []
            for j in members:
                ch = lib[j]
                lo = max(ch["y_min"], ch["t_cond_ref"] - 8.0)
                hi = min(ch["y_max"], ch["t_cond_ref"] + 10.0)
                if hi - lo < 4:
                    continue
                t = np.linspace(lo, hi, 25)
                real = cp.library_cop_ratio(ch, t)
                L = lift_used if np.isscalar(lift_used) else lift_used[j]
                model = L / (L + t - ch["t_cond_ref"])
                errs.append(float(np.sqrt(np.mean((model / real - 1.0) ** 2))))
            return np.array(errs)

        err_class = curve_errors(lift, idx)                             # one lift for the whole class
        err_own = curve_errors({j: 1.0 / s[j] for j in idx}, idx)       # each machine's own lift (tests the FORM)
        # --- out-of-sample: calibrate on a random half, test on the other half ---
        perm = rng.permutation(idx)
        train, test = perm[: len(perm) // 2], perm[len(perm) // 2:]
        lift_train = 1.0 / float(np.median(s[train]))
        err_test = curve_errors(lift_train, test)
        # --- bootstrap CI of the calibrated lift ---
        boot = np.array([1.0 / np.median(rng.choice(s, size=len(s), replace=True)) for _ in range(4000)])
        # physical lift of the library machines at their reference point: what a constant-efficiency
        # Carnot fraction would use (condensing = entering fluid + cond_above, evaporating = chilled water - 1.5 K)
        lp = float(np.median((df.t_cond_ref + spec["cond_above"]) - (df.t_chw_ref - 1.5)))
        cal[key] = dict(n=int(len(idx)), n_total=int(n_total), s_median_pct=100 * s_med,
                        s_q25_pct=100 * float(np.quantile(s, 0.25)), s_q75_pct=100 * float(np.quantile(s, 0.75)),
                        lift_K=lift, lift_ci_lo=float(np.quantile(boot, 0.025)), lift_ci_hi=float(np.quantile(boot, 0.975)),
                        lift_q25_K=1.0 / float(np.quantile(s, 0.75)), lift_q75_K=1.0 / float(np.quantile(s, 0.25)),
                        cop_ref=cop_ref, t_cond_ref=t_cond_ref, t_ref_sink=t_ref_sink,
                        rmse_class_median_pct=100 * float(np.median(err_class)),
                        rmse_class_p90_pct=100 * float(np.quantile(err_class, 0.9)),
                        rmse_own_median_pct=100 * float(np.median(err_own)),
                        lift_train_K=lift_train, rmse_test_median_pct=100 * float(np.median(err_test)),
                        n_train=int(len(train)), n_test=int(len(test)),
                        lift_phys_K=lp, s_ideal_pct=100.0 / lp,
                        real_over_ideal_pct=100.0 * (s_med * lp - 1.0))
        libs[key] = dict(lib=lib, df=df, lift_phys=lp)
    pd.concat(frames, ignore_index=True).to_csv(RES / "chiller_library.csv", index=False)
    out["calibration"] = cal

    # ---------------- Figure 2: calibration / validation ----------------
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3))
    dT = np.linspace(-8, 10, 37)
    for a, key, col, xlabel in ((ax[0], "wc", C_WC, "entering condenser-water temperature minus reference (K)"),
                                (ax[1], "ac", C_AC, "outdoor air temperature minus reference (K)")):
        for ch in libs[key]["lib"]:
            t = ch["t_cond_ref"] + dT
            ok = (t >= ch["y_min"]) & (t <= ch["y_max"])
            a.plot(dT[ok], cp.library_cop_ratio(ch, t[ok]), color="0.6", lw=0.5, alpha=0.6)
        L, Lp = models[key].lift_ref, libs[key]["lift_phys"]
        a.plot([], [], color="0.6", lw=0.8, label=f"{cal[key]['n']} real {LIBRARIES[key]['name'].lower()} chillers")
        a.plot(dT, Lp / (Lp + dT), color=C_GREY, lw=1.5, ls="--", label=f"ideal Carnot fraction, physical lift {Lp:.0f} K")
        a.plot(dT, L / (L + dT), color=col, lw=2.4, label=f"calibrated model, effective lift $L_0$ = {L:.0f} K")
        a.set_ylim(0.6, 1.5)
        a.set_xlabel(xlabel)
        a.legend(fontsize=7.5)
    ax[0].set_ylabel("full-load COP / COP at reference")
    ax[0].set_title("(a) Water-cooled: COP versus heat-sink temperature", loc="left", fontsize=9)
    ax[1].set_title("(b) Air-cooled: COP versus heat-sink temperature", loc="left", fontsize=9)
    bins = np.linspace(0, 6, 41)
    for key, col in (("wc", C_WC), ("ac", C_AC)):
        sv = 100 * libs[key]["df"].s_per_K.values
        ax[2].hist(sv, bins=bins, color=col, alpha=0.6,
                   label=f"{LIBRARIES[key]['name'].lower()}: median {np.median(sv):.1f} %/K")
        ax[2].axvline(np.median(sv), color=col, lw=2)
    ax[2].set_xlabel("COP loss per kelvin of sink temperature, $s$ (%/K)")
    ax[2].set_ylabel("number of chillers")
    ax[2].set_title("(c) Measured sensitivity at the reference point", loc="left", fontsize=9)
    ax[2].set_ylim(0, 47)
    ax[2].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    savefig(fig, "fig2_chiller_calibration")
    return models, libs


# =============================================================================
# 2. Weather: hourly dry-bulb, wet-bulb and solar weights
# =============================================================================
def read_weather(path):
    wx = pd.read_csv(path, comment="#")
    assert len(wx) == 8760, f"expected an 8760-hour year in {path}"
    p = wx["p_Pa"].values if "p_Pa" in wx else cp.P_ATM
    wx["Twb"] = cp.wet_bulb(wx["T_C"].values, wx["Td_C"].values, p=p)
    return wx


def daily_arrays(frames):
    """List of 8760-row frames -> dict of arrays with one row per day; 'src' numbers the frame of each day."""
    shape = (365, 24)
    parts = [dict(month=wx["month"].values.reshape(shape)[:, 0], doy=np.arange(1, 366), src=np.full(365, j),
                  db=wx["T_C"].values.reshape(shape), wb=wx["Twb"].values.reshape(shape),
                  td=wx["Td_C"].values.reshape(shape), g=np.clip(wx["ghi_Wm2"].values.reshape(shape), 0, None))
             for j, wx in enumerate(frames)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def with_clear_sky(D):
    """Replace the irradiance weights by the Haurwitz clear-sky model (tests the choice of solar weights)."""
    solar = (np.arange(24) + 0.5) + (LON - TZ_MERIDIAN) / 15.0
    D2 = dict(D)
    D2["g"] = np.array([haurwitz_ghi(d, solar) for d in D["doy"]])
    return D2


def penalties(chiller, D, use="uniform", make="solar", dT_charge=0.0, lift=None):
    """Daily Carnot penalty (exact, hour by hour) and its closed form."""
    ch = chiller if lift is None else chiller.with_lift(lift)
    sink = D[chiller.sink]
    g = D["g"]
    flat = ((np.arange(24) >= 9) & (np.arange(24) < 15)).astype(float)
    w_make = g if make == "solar" else np.tile(flat, (g.shape[0], 1))
    night = (g <= 0).astype(float)
    w_use = night if use == "uniform" else night * np.clip(D["db"] - 18.0, 0.0, None) + 1e-9 * night
    make_work = (w_make / ch.cop(sink, dT_charge)).sum(1) / w_make.sum(1)   # kWh_e per kWh_th, made in solar hours
    use_work = (w_use / ch.cop(sink)).sum(1) / w_use.sum(1)                 # kWh_e per kWh_th, made on demand at night
    t_make = (w_make * sink).sum(1) / w_make.sum(1)
    t_use = (w_use * sink).sum(1) / w_use.sum(1)
    pi_exact = make_work / use_work - 1.0
    pi_closed = cp.carnot_penalty_closed_form(t_make, t_use, ch.lift(t_use))
    return dict(pi=pi_exact, pi_closed=pi_closed, t_make=t_make, t_use=t_use,
                make_work=make_work, use_work=use_work)


def solar_minus_night(x, g):
    """Daily (irradiance-weighted mean of x) minus (mean of x over the dark hours)."""
    night = (g <= 0)
    return (g * x).sum(1) / g.sum(1) - (x * night).sum(1) / night.sum(1)


def summarise(models, D, pi_star):
    """Headline numbers of one weather data set (or of several pooled)."""
    season = np.isin(D["month"], COOL_SEASON)
    R = {}
    for key in ("ac", "wc"):
        P = penalties(models[key], D)
        sink = D[models[key].sink]
        eff = cp.cold_storage_efficiency(P["pi"], ETA_TANK)
        R[key] = dict(
            pi_season_mean_pct=100 * float(P["pi"][season].mean()),
            pi_season_p10_pct=100 * float(np.quantile(P["pi"][season], 0.1)),
            pi_season_p90_pct=100 * float(np.quantile(P["pi"][season], 0.9)),
            pi_year_mean_pct=100 * float(P["pi"].mean()),
            eff_season_mean_pct=100 * float(eff[season].mean()),
            eff_year_mean_pct=100 * float(eff.mean()),
            share_days_beating_battery_season_pct=100 * float((P["pi"][season] < pi_star).mean()),
            share_days_beating_battery_year_pct=100 * float((P["pi"] < pi_star).mean()),
            swing_solar_minus_night_season_K=float((P["t_make"] - P["t_use"])[season].mean()),
            lift_night_season_K=float(models[key].lift(P["t_use"][season]).mean()),
            closed_form_max_abs_err_pctpt=100 * float(np.max(np.abs(P["pi"] - P["pi_closed"]))),
            cop_night_season=float(1.0 / P["use_work"][season].mean()),
            cop_solar_season=float(1.0 / P["make_work"][season].mean()),
            clean_ratio_breakeven_season=float(cp.breakeven_clean_ratio(P["pi"][season].mean(), ETA_TANK)),
            # kWh_e needed per MWh_th of NIGHT cooling in the cooling season, three routes
            route_grid_night=1000 * float(P["use_work"][season].mean()),
            route_battery=1000 * float(P["use_work"][season].mean()) / ETA_BATT,
            route_cold_tank=1000 * float(P["make_work"][season].mean()) / ETA_TANK,
            breakeven_swing_K=float(models[key].lift(P["t_use"][season]).mean() * pi_star),
            sink_diurnal_range_season_K=float((sink[season].max(1) - sink[season].min(1)).mean()),
        )
        R[key]["_P"] = P
    R["td_solar_minus_night_season_K"] = float(solar_minus_night(D["td"], D["g"])[season].mean())
    return R


# =============================================================================
# 3. Main analysis
# =============================================================================
def main():
    pi_star = cp.breakeven_penalty(ETA_TANK, ETA_BATT)
    out = dict(params=dict(eta_tank=ETA_TANK, eta_batt=ETA_BATT, tower_approach_K=TOWER_APPROACH, seed=SEED,
                           breakeven_penalty_pct=100 * pi_star, base_case=AD_SETS))
    models, libs = calibrate_chillers(out)
    cal = out["calibration"]

    W = {k: read_weather(DATA / "station_hourly" / f"{k}.csv") for k in AD_SETS + OTHER_SETS}
    W["REAN_2022"] = read_weather(DATA / "reanalysis" / "abudhabi_reanalysis_2022.csv")
    W["REAN_2023"] = read_weather(DATA / "reanalysis" / "abudhabi_reanalysis_2023.csv")
    W["NORMALS"] = read_weather(DATA / "weather_hourly.csv")
    D = daily_arrays([W[k] for k in AD_SETS])           # base case: pooled Abu Dhabi station-years
    season = np.isin(D["month"], COOL_SEASON)
    hours = np.arange(24)

    gaps = {}
    for k in AD_SETS + OTHER_SETS:
        head = (DATA / "station_hourly" / f"{k}.csv").read_text().splitlines()[0]
        m = re.search(r"(\d+) of 8760 hours .*longest gap (\d+) h", head)
        gaps[k] = dict(missing_hours=int(m.group(1)), longest_gap_h=int(m.group(2)))
    out["weather"] = dict(n_station_years=len(AD_SETS), n_days=int(D["db"].shape[0]), n_season_days=int(season.sum()),
                          T_max=float(D["db"].max()), T_min=float(D["db"].min()),
                          Twb_max=float(D["wb"].max()), Twb_p99=float(np.quantile(D["wb"], 0.99)),
                          gaps=gaps, max_missing_hours=max(g["missing_hours"] for g in gaps.values()),
                          max_gap_h=max(g["longest_gap_h"] for g in gaps.values()))

    # second-law check of the fitted models over every measured hour, using PHYSICAL temperatures
    # (condensing = sink + approach + cond_above, evaporating = chilled water - 1.5 K), not the fitted lift
    for key in ("wc", "ac"):
        spec, df = LIBRARIES[key], libs[key]["df"]
        t_evap = float(np.median(df.t_chw_ref)) - 1.5
        sink = D[spec["sink"]]
        carnot = (t_evap + 273.15) / (sink + spec["approach"] + spec["cond_above"] - t_evap)
        frac = models[key].cop(sink) / carnot
        assert frac.max() < 1.0, "fitted chiller would beat Carnot"
        cal[key]["second_law_eff_min"] = float(frac.min())
        cal[key]["second_law_eff_max"] = float(frac.max())

    # ---------------- Figure 1: the two heat sinks (measured) ----------------
    Dn = daily_arrays([W["NORMALS"]])
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.4), sharey=True)
    for a, (months, title) in zip(ax, ((COOL_SEASON, "(a) Cooling season, May-Oct"), ([12, 1, 2], "(b) Winter, Dec-Feb"))):
        m = np.isin(D["month"], months)
        a2 = a.twinx()
        a2.fill_between(hours + 0.5, D["g"][m].mean(0), color="#f2c14e", alpha=0.35, lw=0)
        a2.set_ylim(0, 1300)
        a2.grid(False)
        a2.spines["right"].set_visible(True)
        a2.set_ylabel("solar irradiance (W/m$^2$)" if a is ax[1] else "")
        if a is ax[0]:
            a2.set_yticklabels([])
        for key, col, lab in (("db", C_AC, "dry-bulb (air-cooled sink)"), ("wb", C_WC, "wet-bulb (water-cooled sink)")):
            a.plot(hours + 0.5, D[key][m].mean(0), color=col, lw=2.2, label=lab)
            a.fill_between(hours + 0.5, np.quantile(D[key][m], 0.1, 0), np.quantile(D[key][m], 0.9, 0), color=col, alpha=0.15, lw=0)
        a.plot(hours + 0.5, D["td"][m].mean(0), color=C_TD, lw=1.6, ls=":", label="dew point")
        mn = np.isin(Dn["month"], months)
        a.plot(hours + 0.5, Dn["wb"][mn].mean(0), color=C_WC, lw=1.0, ls="--", label="wet-bulb, synthetic year")
        a.set_xlabel("hour of day (local time)")
        a.set_xticks(range(0, 25, 4))
        a.set_title(title, loc="left", fontsize=9)
        a.set_zorder(a2.get_zorder() + 1)
        a.patch.set_visible(False)
    ax[0].set_ylabel("temperature ($^\\circ$C)")
    ax[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    ax[0].set_ylim(5, 52)
    fig.tight_layout()
    savefig(fig, "fig1_heat_sinks")

    # ---------------- core results (base case) ----------------
    R = summarise(models, D, pi_star)
    monthly = {"month": MONTHS}
    for key in ("ac", "wc"):
        P = R[key]["_P"]
        eff = cp.cold_storage_efficiency(P["pi"], ETA_TANK)
        sel = [D["month"] == m for m in range(1, 13)]
        monthly[f"pi_{key}_pct"] = [100 * float(P["pi"][s].mean()) for s in sel]
        monthly[f"eff_{key}_pct"] = [100 * float(eff[s].mean()) for s in sel]
        monthly[f"sink_swing_{key}_K"] = [float((P["t_make"] - P["t_use"])[s].mean()) for s in sel]
        by_src = np.array([[100 * float(P["pi"][s & (D["src"] == j)].mean()) for s in sel] for j in range(len(AD_SETS))])
        monthly[f"pi_{key}_min_pct"] = by_src.min(0).tolist()
        monthly[f"pi_{key}_max_pct"] = by_src.max(0).tolist()
        R[key]["pi_season_by_set_pct"] = [100 * float(P["pi"][season & (D["src"] == j)].mean()) for j in range(len(AD_SETS))]
        R[key]["pi_season_min_pct"] = min(R[key]["pi_season_by_set_pct"])
        R[key]["pi_season_max_pct"] = max(R[key]["pi_season_by_set_pct"])
        R[key]["saving_vs_battery_pct"] = 100 * (1 - R[key]["route_cold_tank"] / R[key]["route_battery"])
    pd.DataFrame(monthly).round(2).to_csv(RES / "monthly_table.csv", index=False)
    # the report says: water-cooled penalty negative from March to November, positive December to February
    wcm = np.array(monthly["pi_wc_pct"])
    assert np.all(wcm[2:11] < 0) and np.all(wcm[[0, 1, 11]] > 0), "monthly sign pattern quoted in the report changed"

    # ---------------- every data set separately ----------------
    ac_design = cp.Chiller("Air-cooled (design values)", "db", cop_ref=3.0, t_ref=35.0, lift_ref=45.0)
    rows = []
    for k in ["NORMALS", "REAN_2022", "REAN_2023"] + AD_SETS + OTHER_SETS:
        Dk = daily_arrays([W[k]])
        sk = np.isin(Dk["month"], COOL_SEASON)
        Rk = summarise(models, Dk, pi_star)
        rows.append(dict(set=k, label=LABEL[k], kind=KIND[k],
                         dT_drybulb_K=float(solar_minus_night(Dk["db"], Dk["g"])[sk].mean()),
                         dT_dewpoint_K=float(solar_minus_night(Dk["td"], Dk["g"])[sk].mean()),
                         dT_wetbulb_K=float(solar_minus_night(Dk["wb"], Dk["g"])[sk].mean()),
                         range_drybulb_K=Rk["ac"]["sink_diurnal_range_season_K"],
                         pi_ac_pct=Rk["ac"]["pi_season_mean_pct"], pi_wc_pct=Rk["wc"]["pi_season_mean_pct"],
                         eff_ac_pct=Rk["ac"]["eff_season_mean_pct"], eff_wc_pct=Rk["wc"]["eff_season_mean_pct"],
                         days_ac_beats_battery_pct=Rk["ac"]["share_days_beating_battery_season_pct"],
                         days_wc_beats_battery_pct=Rk["wc"]["share_days_beating_battery_season_pct"]))
    sets = pd.DataFrame(rows)
    sets.round(3).to_csv(RES / "datasets_table.csv", index=False)
    out["datasets"] = rows
    # first pass (the report's first version): normals-based year, air-cooled chiller from design values
    Dn_season = np.isin(Dn["month"], COOL_SEASON)
    first = dict(pi_ac_pct=100 * float(penalties(ac_design, Dn)["pi"][Dn_season].mean()),
                 pi_wc_pct=100 * float(penalties(models["wc"], Dn)["pi"][Dn_season].mean()),
                 dT_drybulb_K=float(solar_minus_night(Dn["db"], Dn["g"])[Dn_season].mean()),
                 dT_wetbulb_K=float(solar_minus_night(Dn["wb"], Dn["g"])[Dn_season].mean()))
    # the two corrections applied one at a time to the air-cooled first-pass number
    first["pi_ac_measured_weather_design_chiller_pct"] = 100 * float(penalties(ac_design, D)["pi"][season].mean())
    first["pi_ac_normals_weather_library_chiller_pct"] = 100 * float(penalties(models["ac"], Dn)["pi"][Dn_season].mean())
    out["first_pass"] = first

    # ---------------- check of the monthly normals against the airport record ----------------
    norm = pd.read_csv(DATA / "abudhabi_monthly_normals.csv", comment="#")
    obs = []
    for y in (2022, 2023):
        raw = pd.read_csv(DATA / "station" / f"OMAA_{y}.csv")
        raw["tmpc"] = pd.to_numeric(raw["tmpc"], errors="coerce")
        raw = raw.dropna(subset=["tmpc"])
        when = pd.to_datetime(raw["valid"])
        day = raw.groupby(when.dt.date)["tmpc"].agg(["max", "min", "count"])
        day = day[day["count"] >= 20]
        day["month"] = pd.to_datetime(day.index).month
        obs.append(day)
    obs = pd.concat(obs).groupby("month")[["max", "min"]].mean()
    out["normals_check"] = dict(
        obs_tmax=obs["max"].round(2).tolist(), obs_tmin=obs["min"].round(2).tolist(),
        normals_tmax=norm.tmax_C.tolist(), normals_tmin=norm.tmin_C.tolist(),
        mean_tmax_obs_minus_normals_K=float((obs["max"].values - norm.tmax_C.values).mean()),
        mean_tmin_obs_minus_normals_K=float((obs["min"].values - norm.tmin_C.values).mean()),
        min_tmin_obs_minus_normals_K=float((obs["min"].values - norm.tmin_C.values).min()),
        max_tmin_obs_minus_normals_K=float((obs["min"].values - norm.tmin_C.values).max()),
        max_abs_tmax_obs_minus_normals_K=float(np.abs(obs["max"].values - norm.tmax_C.values).max()),
        range_obs_season_K=float((obs["max"] - obs["min"]).loc[COOL_SEASON].mean()),
        range_normals_season_K=float((norm.tmax_C - norm.tmin_C).values[np.array(COOL_SEASON) - 1].mean()))

    # ---------------- why the wet-bulb sink behaves differently: two-term response ----------------
    t_mean, td_mean = float(D["db"][season].mean()), float(D["td"][season].mean())
    a_db = float(cp.wet_bulb(t_mean + 0.5, td_mean) - cp.wet_bulb(t_mean - 0.5, td_mean))      # dTwb/dT at fixed dew point
    b_td = float(cp.wet_bulb(t_mean, td_mean + 0.5) - cp.wet_bulb(t_mean, td_mean - 0.5))      # dTwb/dTd at fixed dry-bulb
    d_db = float(solar_minus_night(D["db"], D["g"])[season].mean())
    d_td = float(solar_minus_night(D["td"], D["g"])[season].mean())
    d_wb = float(solar_minus_night(D["wb"], D["g"])[season].mean())
    out["damping"] = dict(T_mean=t_mean, Td_mean=td_mean, dTwb_dT=a_db, dTwb_dTd=b_td,
                          dT_drybulb_K=d_db, dT_dewpoint_K=d_td, dT_wetbulb_K=d_wb,
                          dT_wetbulb_from_drybulb_K=a_db * d_db, dT_wetbulb_from_dewpoint_K=b_td * d_td,
                          dT_wetbulb_two_term_K=a_db * d_db + b_td * d_td,
                          range_drybulb_K=R["ac"]["sink_diurnal_range_season_K"],
                          range_wetbulb_K=R["wc"]["sink_diurnal_range_season_K"])

    # ---------------- Figure 3: headline result by month ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    xs = np.arange(12)
    for key, col, off in (("ac", C_AC, -0.2), ("wc", C_WC, 0.2)):
        mean = np.array(monthly[f"pi_{key}_pct"])
        lo, hi = np.array(monthly[f"pi_{key}_min_pct"]), np.array(monthly[f"pi_{key}_max_pct"])
        ax[0].bar(xs + off, mean, width=0.38, color=col, alpha=0.85, label=models[key].name,
                  yerr=[mean - lo, hi - mean], error_kw=dict(lw=0.9, capsize=1.5, ecolor="k"))
        ax[1].plot(xs, monthly[f"eff_{key}_pct"], "-o", color=col, lw=2, ms=4, label=f"cold tank, {models[key].name.lower()} plant")
    ax[0].axhline(100 * pi_star, color=C_GREY, ls="--", lw=1.2, label=f"battery break-even $\\pi^*$ = {100 * pi_star:.0f}%")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(MONTHS)
    ax[0].set_ylabel("Carnot penalty $\\pi$ (%)")
    ax[0].set_title("(a) Extra electricity when cold is made at solar hours", loc="left", fontsize=9)
    ax[0].legend(fontsize=7.5, loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.6)
    ax[0].set_ylim(-8, 33)
    ax[1].axhline(100 * ETA_BATT, color=C_GREY, ls="--", lw=1.2, label=f"Li-ion battery ({100 * ETA_BATT:.0f}%)")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(MONTHS)
    ax[1].set_ylabel("effective round-trip efficiency (%)")
    ax[1].set_title("(b) Storing solar energy as cold vs. in a battery", loc="left", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig3_penalty_by_month")

    # ---------------- Figure 4: closed form vs exact, and the emissions rule ----------------
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for key, col in (("ac", C_AC), ("wc", C_WC)):
        P = R[key]["_P"]
        ax[0].scatter(100 * P["pi_closed"], 100 * P["pi"], s=4, color=col, alpha=0.4, label=models[key].name)
    lim = [min(ax[0].get_xlim()[0], ax[0].get_ylim()[0]), max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
    ax[0].plot(lim, lim, color="k", lw=0.8)
    ax[0].set_xlabel("closed form: $\\Delta T_{sink}/L_{night}$ (%)")
    ax[0].set_ylabel("exact hourly calculation (%)")
    ax[0].set_title(f"(a) One formula captures all {D['db'].shape[0]} measured days", loc="left", fontsize=9)
    ax[0].legend(fontsize=8)
    r = np.linspace(0, 1.1, 50)
    for key, col in (("ac", C_AC), ("wc", C_WC)):
        rstar = R[key]["clean_ratio_breakeven_season"]
        ax[1].plot(r, 100 * (1 - r / rstar), color=col, lw=2, label=f"{models[key].name}: break-even $r^*$ = {rstar:.2f}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("$r$ = solar-hour / night emissions (or price) of electricity")
    ax[1].set_ylabel("emissions saved on shifted cooling (%)")
    ax[1].set_title("(b) When does solar-following cold storage pay?", loc="left", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig4_closed_form_and_breakeven")

    # ---------------- sensitivity (one at a time) ----------------
    def season_eff(key, Dv=None, **kw):
        eta = kw.pop("eta_tank", ETA_TANK)
        Dv = D if Dv is None else Dv
        sv = np.isin(Dv["month"], COOL_SEASON)
        return 100 * float(cp.cold_storage_efficiency(penalties(models[key], Dv, **kw)["pi"][sv], eta).mean())

    D_air = daily_arrays([W["OMAA_2022"], W["OMAA_2023"]])
    D_isl = daily_arrays([W["OMAD_2022"], W["OMAD_2023"]])
    D_22 = daily_arrays([W["OMAA_2022"], W["OMAD_2022"]])
    D_23 = daily_arrays([W["OMAA_2023"], W["OMAD_2023"]])
    D_clear = with_clear_sky(D)
    sens = {}
    for key in ("ac", "wc"):
        base = season_eff(key)
        sens[key] = {
            "base": base,
            "Effective lift: lower / upper quartile of the machines": (season_eff(key, lift=cal[key]["lift_q25_K"]), season_eff(key, lift=cal[key]["lift_q75_K"])),
            "Tank efficiency 0.90 / 0.99": (season_eff(key, eta_tank=0.90), season_eff(key, eta_tank=0.99)),
            "Station: airport / island": (season_eff(key, Dv=D_air), season_eff(key, Dv=D_isl)),
            "Year: 2022 / 2023": (season_eff(key, Dv=D_22), season_eff(key, Dv=D_23)),
            "Solar weights: reanalysis / clear-sky model": (base, season_eff(key, Dv=D_clear)),
            "Night load: uniform / degree-hour weighted": (base, season_eff(key, use="degree")),
            "Charging window: solar-weighted / flat 09-15 h": (base, season_eff(key, make="flat")),
            "Colder charging water: 0 K / 2 K": (base, season_eff(key, dT_charge=2.0)),
            "Ice storage (evaporator 12 K colder)": (base, season_eff(key, dT_charge=12.0)),
        }
    out["sensitivity"] = {k: {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in d.items()} for k, d in sens.items()}

    # ---------------- Monte Carlo over machines, storage efficiencies and weather ----------------
    rng = np.random.default_rng(SEED)
    N = 20000
    mc = {}
    eta_t = rng.uniform(0.90, 0.99, N)
    eta_b = rng.uniform(0.80, 0.90, N)
    which = rng.integers(0, len(AD_SETS), N)                  # weather: one of the measured station-years
    for key in ("wc", "ac"):
        Ls = rng.choice(libs[key]["df"].lift_eff_K.values, size=N, replace=True)     # one real machine per draw
        P = R[key]["_P"]
        dT, tu = P["t_make"] - P["t_use"], P["t_use"]
        pis = np.empty(N)
        for j in range(len(AD_SETS)):
            sel, mm = season & (D["src"] == j), which == j
            pis[mm] = (dT[sel][None, :] / (Ls[mm][:, None] + tu[sel][None, :] - models[key].t_ref)).mean(1)
        eff = eta_t / (1 + pis)
        mc[key] = dict(eff=eff, wins=float((eff > eta_b).mean()),
                       eff_p5=float(np.quantile(eff, 0.05)), eff_p50=float(np.quantile(eff, 0.5)), eff_p95=float(np.quantile(eff, 0.95)),
                       pi_p5=100 * float(np.quantile(pis, 0.05)), pi_p50=100 * float(np.quantile(pis, 0.5)), pi_p95=100 * float(np.quantile(pis, 0.95)))
    out["monte_carlo"] = {k: {kk: vv for kk, vv in v.items() if kk != "eff"} for k, v in mc.items()}
    out["monte_carlo"]["n"] = N

    # ---------------- Figure 5: uncertainty ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    bins = np.linspace(60, 110, 51)
    for key, col in (("ac", C_AC), ("wc", C_WC)):
        ax[0].hist(100 * mc[key]["eff"], bins=bins, color=col, alpha=0.6, density=True,
                   label=f"{models[key].name}: beats battery in {100 * mc[key]['wins']:.0f}% of draws")
    ax[0].axvspan(80, 90, color=C_GREY, alpha=0.15, lw=0, label="Li-ion battery range")
    ax[0].set_xlabel("effective round-trip efficiency of solar-made cold (%)")
    ax[0].set_ylabel("probability density")
    ax[0].set_title(f"(a) Monte Carlo, {N:,} draws", loc="left", fontsize=9)
    ax[0].legend(fontsize=7.5, loc="upper left")
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.45)
    labels = [k for k in sens["wc"] if k != "base"]
    y = np.arange(len(labels))[::-1]
    for key, col, off in (("ac", C_AC, -0.18), ("wc", C_WC, 0.18)):
        for yy, lab in zip(y, labels):
            lo, hi = sorted(sens[key][lab])
            ax[1].barh(yy + off, max(hi - lo, 0.15), left=lo, height=0.32, color=col, alpha=0.75)
        ax[1].axvline(sens[key]["base"], color=col, lw=1, ls=":")
    ax[1].axvline(100 * ETA_BATT, color=C_GREY, ls="--", lw=1.2)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(labels, fontsize=7.5)
    ax[1].set_xlim(58, 104)
    ax[1].set_xlabel("cooling-season effective efficiency (%)   [dashed: battery 85%]")
    ax[1].set_title("(b) One-at-a-time sensitivity", loc="left", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig5_uncertainty")

    # ---------------- Figure 6: the same model on every weather data set ----------------
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    xs = np.arange(len(sets))
    ax.axhline(100 * pi_star, color=C_GREY, ls="--", lw=1.2)
    ax.text(len(sets) - 0.55, 100 * pi_star + 0.7, f"battery break-even {100 * pi_star:.0f}%", ha="right", fontsize=7.5, color=C_GREY)
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(xs, sets.pi_ac_pct, "o", color=C_AC, ms=7, label="Air-cooled")
    ax.plot(xs, sets.pi_wc_pct, "s", color=C_WC, ms=6.5, label="Water-cooled")
    for x0, x1, name in ((-0.5, 0.5, "synthetic"), (0.5, 2.5, "reanalysis"), (2.5, 6.5, "measured, Abu Dhabi"), (6.5, 8.5, "measured, other Gulf airports")):
        ax.axvspan(x0, x1, color="0.5", alpha=0.07 if name in ("synthetic", "measured, Abu Dhabi") else 0.0, lw=0)
        ax.text(0.5 * (x0 + x1), 28.3, name, ha="center", fontsize=7.5, color=C_GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace(" 20", "\n20").replace("Normals-based typical year", "Normals-based\ntypical year") for l in sets.label], fontsize=7.5)
    ax.set_ylabel("Carnot penalty $\\pi$, May-Oct (%)")
    ax.set_ylim(-8, 31)
    ax.set_xlim(-0.5, len(sets) - 0.5)
    ax.legend(fontsize=8, loc="center left", ncol=1)
    fig.tight_layout()
    savefig(fig, "fig6_all_datasets")

    # ---------------- write results ----------------
    for key in ("ac", "wc"):
        R[key].pop("_P")
    out["results"] = R
    out["models"] = {k: dict(name=m.name, sink=m.sink, cop_ref=m.cop_ref, t_ref=m.t_ref, lift_ref=m.lift_ref) for k, m in models.items()}
    out["monthly"] = monthly
    (RES / "results.json").write_text(json.dumps(out, indent=2))
    write_report_inputs(out, sets, sens)
    print(json.dumps({k: out[k] for k in ("params", "calibration", "weather", "damping", "first_pass", "normals_check",
                                          "results", "monte_carlo", "sensitivity")}, indent=1))


# =============================================================================
# 4. Numbers and tables for the report (report/numbers.tex, report/table_*.tex)
# =============================================================================
def write_report_inputs(out, sets, sens):
    cal, R, mc, dm, fp, nc, wx = (out["calibration"], out["results"], out["monte_carlo"], out["damping"],
                                  out["first_pass"], out["normals_check"], out["weather"])
    num = {}

    def put(name, value, fmt="{:.1f}"):
        assert name.isalpha(), name
        text = fmt.format(value) if not isinstance(value, str) else value
        num[name] = text.replace("-", "$-$") if text.startswith("-") else text

    for key, K in (("wc", "WC"), ("ac", "AC")):
        c, r, m = cal[key], R[key], mc[key]
        put(f"N{K}", c["n"], "{:d}")
        put(f"NTotal{K}", c["n_total"], "{:d}")
        put(f"Sens{K}", c["s_median_pct"], "{:.2f}")
        put(f"SensLo{K}", c["s_q25_pct"], "{:.2f}")
        put(f"SensHi{K}", c["s_q75_pct"], "{:.2f}")
        put(f"Lift{K}", c["lift_K"])
        put(f"LiftLo{K}", c["lift_ci_lo"])
        put(f"LiftHi{K}", c["lift_ci_hi"])
        put(f"LiftPhys{K}", c["lift_phys_K"], "{:.0f}")
        put(f"SensIdeal{K}", c["s_ideal_pct"], "{:.1f}")
        put(f"RealOverIdeal{K}", abs(c["real_over_ideal_pct"]), "{:.0f}")
        put(f"CopRef{K}", c["cop_ref"], "{:.2f}")
        put(f"TRef{K}", c["t_ref_sink"], "{:.1f}")
        put(f"ErrClass{K}", c["rmse_class_median_pct"])
        put(f"ErrClassHi{K}", c["rmse_class_p90_pct"])
        put(f"ErrOwn{K}", c["rmse_own_median_pct"])
        put(f"LiftTrain{K}", c["lift_train_K"])
        put(f"ErrTest{K}", c["rmse_test_median_pct"])
        put(f"NTrain{K}", c["n_train"], "{:d}")
        put(f"NTest{K}", c["n_test"], "{:d}")
        put(f"Pi{K}", r["pi_season_mean_pct"])
        put(f"PiAbs{K}", abs(r["pi_season_mean_pct"]))
        put(f"PiMin{K}", r["pi_season_min_pct"])
        put(f"PiMax{K}", r["pi_season_max_pct"])
        put(f"PiYear{K}", r["pi_year_mean_pct"])
        put(f"Eff{K}", r["eff_season_mean_pct"])
        put(f"EffRound{K}", r["eff_season_mean_pct"], "{:.0f}")
        put(f"Swing{K}", r["swing_solar_minus_night_season_K"])
        put(f"LiftNight{K}", r["lift_night_season_K"])
        put(f"SwingStar{K}", r["breakeven_swing_K"])
        put(f"Days{K}", r["share_days_beating_battery_season_pct"], "{:.0f}")
        put(f"RouteGrid{K}", r["route_grid_night"], "{:.0f}")
        put(f"RouteBatt{K}", r["route_battery"], "{:.0f}")
        put(f"RouteTank{K}", r["route_cold_tank"], "{:.0f}")
        put(f"Saving{K}", abs(r["saving_vs_battery_pct"]), "{:.0f}")
        put(f"RStar{K}", r["clean_ratio_breakeven_season"], "{:.2f}")
        put(f"Cleaner{K}", 100 * (1 - r["clean_ratio_breakeven_season"]), "{:.0f}")
        put(f"CopNight{K}", r["cop_night_season"], "{:.2f}")
        put(f"CopSolar{K}", r["cop_solar_season"], "{:.2f}")
        put(f"Range{K}", r["sink_diurnal_range_season_K"])
        put(f"McWins{K}", 100 * m["wins"], "{:.0f}")
        put(f"McLo{K}", 100 * m["eff_p5"], "{:.0f}")
        put(f"McMed{K}", 100 * m["eff_p50"], "{:.0f}")
        put(f"McHi{K}", 100 * m["eff_p95"], "{:.0f}")
        put(f"Ice{K}", sens[key]["Ice storage (evaporator 12 K colder)"][1], "{:.0f}")
        put(f"SecondLawLo{K}", 100 * c["second_law_eff_min"], "{:.0f}")
        put(f"SecondLawHi{K}", 100 * c["second_law_eff_max"], "{:.0f}")
        put(f"IceLoss{K}", sens[key]["base"] - sens[key]["Ice storage (evaporator 12 K colder)"][1], "{:.0f}")
        put(f"PiMonthMax{K}", max(out["monthly"][f"pi_{key}_pct"]))
        put(f"PiMonthMin{K}", min(out["monthly"][f"pi_{key}_pct"]))
        put(f"BattMinusTank{K}", out["params"]["eta_batt"] * 100 - r["eff_season_mean_pct"])
    put("IdealOverWC", 100 * (cal["wc"]["s_ideal_pct"] / cal["wc"]["s_median_pct"] - 1), "{:.0f}")
    put("IdealUnderAC", 100 * (1 - cal["ac"]["s_ideal_pct"] / cal["ac"]["s_median_pct"]), "{:.0f}")
    put("DesignUnderAC", 100 * (cal["ac"]["s_median_pct"] / (100.0 / 45.0) - 1), "{:.0f}")
    put("NMachines", cal["wc"]["n"] + cal["ac"]["n"], "{:d}")
    put("PiStar", out["params"]["breakeven_penalty_pct"])
    put("NStationYears", wx["n_station_years"], "{:d}")
    put("NDays", wx["n_days"], "{:d}")
    put("NSeasonDays", wx["n_season_days"], "{:d}")
    put("MaxMissing", wx["max_missing_hours"], "{:d}")
    put("MaxGap", wx["max_gap_h"], "{:d}")
    put("TMean", dm["T_mean"])
    put("TdMean", dm["Td_mean"])
    put("DampA", dm["dTwb_dT"], "{:.2f}")
    put("DampB", dm["dTwb_dTd"], "{:.2f}")
    put("DTdb", dm["dT_drybulb_K"])
    put("DTdp", dm["dT_dewpoint_K"])
    put("DTdpAbs", abs(dm["dT_dewpoint_K"]))
    put("DTwb", dm["dT_wetbulb_K"])
    put("DTwbAbs", abs(dm["dT_wetbulb_K"]))
    put("DTwbFromDb", dm["dT_wetbulb_from_drybulb_K"])
    put("DTwbFromDp", dm["dT_wetbulb_from_dewpoint_K"])
    put("DTwbTwoTerm", dm["dT_wetbulb_two_term_K"])
    put("RangeDb", dm["range_drybulb_K"])
    put("RangeWb", dm["range_wetbulb_K"])
    put("FirstPiAC", fp["pi_ac_pct"])
    put("FirstPiWC", fp["pi_wc_pct"])
    put("FirstDTdb", fp["dT_drybulb_K"])
    put("FirstDTwb", fp["dT_wetbulb_K"])
    put("FirstPiACMeasuredWeather", fp["pi_ac_measured_weather_design_chiller_pct"])
    put("FirstPiACLibraryChiller", fp["pi_ac_normals_weather_library_chiller_pct"])
    put("NormTmaxDiff", nc["mean_tmax_obs_minus_normals_K"])
    put("NormTmaxAbsMax", nc["max_abs_tmax_obs_minus_normals_K"])
    put("NormTminDiff", nc["mean_tmin_obs_minus_normals_K"])
    put("NormTminDiffLo", nc["min_tmin_obs_minus_normals_K"])
    put("NormTminDiffHi", nc["max_tmin_obs_minus_normals_K"])
    put("NormRangeObs", nc["range_obs_season_K"])
    put("NormRangeNormals", nc["range_normals_season_K"])
    by = {r["set"]: r for r in out["datasets"]}
    # the report says the water-cooled penalty is "zero in Dubai"
    assert abs(by["OMDB_2023"]["pi_wc_pct"]) < 0.25, "Dubai water-cooled penalty is no longer zero to one decimal"
    for k, K in (("OMDB_2023", "Dubai"), ("OMSJ_2023", "Sharjah"), ("REAN_2022", "ReanA"), ("REAN_2023", "ReanB"), ("NORMALS", "Normals")):
        put(f"PiAC{K}", by[k]["pi_ac_pct"])
        put(f"PiWC{K}", by[k]["pi_wc_pct"])
        put(f"DTdp{K}", by[k]["dT_dewpoint_K"])
        put(f"DTdb{K}", by[k]["dT_drybulb_K"])
    lines = ["% Generated by scripts/run_analysis.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(num.items())]
    (REP / "numbers.tex").write_text("\n".join(lines) + "\n")

    def f(v, fmt="{:.1f}"):
        s = fmt.format(v)
        if s.strip("+-0.") == "":          # a value that rounds to zero carries no sign
            s = s.lstrip("+-")
        return f"${s}$"

    rows = ["% Generated by scripts/run_analysis.py -- do not edit by hand.",
            "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
            " & & \\multicolumn{3}{c}{Solar hours minus night (K)} & \\multicolumn{2}{c}{Carnot penalty $\\pi$ (\\%)}\\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(l){6-7}",
            "Weather data, May--Oct & Kind & dry-bulb & dew point & wet-bulb & air-cooled & water-cooled\\\\", "\\midrule"]
    for _, r in sets.iterrows():
        if r["set"] in ("OMAA_2022", "OMDB_2023"):
            rows.append("\\addlinespace")
        rows.append(f"{r['label']} & {r['kind']} & {f(r['dT_drybulb_K'], '{:+.1f}')} & {f(r['dT_dewpoint_K'], '{:+.1f}')} & "
                    f"{f(r['dT_wetbulb_K'], '{:+.1f}')} & {f(r['pi_ac_pct'])} & {f(r['pi_wc_pct'], '{:+.1f}')}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (REP / "table_datasets.tex").write_text("\n".join(rows) + "\n")

    rows = ["% Generated by scripts/run_analysis.py -- do not edit by hand.",
            "\\begin{tabular}{@{}lcc@{}}", "\\toprule",
            "Effective round-trip efficiency of the cold tank, May--Oct (\\%) & Air-cooled & Water-cooled\\\\", "\\midrule",
            f"Base case & {sens['ac']['base']:.1f} & {sens['wc']['base']:.1f}\\\\"]
    for lab in [k for k in sens["wc"] if k != "base"]:
        cells = []
        for key in ("ac", "wc"):
            lo, hi = sens[key][lab]
            cells.append(f"{hi:.1f}" if lab.startswith(("Night load", "Charging window", "Colder", "Ice", "Solar weights")) else f"{lo:.1f} / {hi:.1f}")
        rows.append(f"{lab} & {cells[0]} & {cells[1]}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (REP / "table_sensitivity.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
