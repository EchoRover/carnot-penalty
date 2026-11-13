#!/usr/bin/env python3
"""
run_analysis.py -- regenerates EVERY number, table and figure in the report.

    python scripts/run_analysis.py

Inputs  (data/):    Chillers.idf            EnergyPlus chiller performance library (manufacturer-derived curves)
                    weather_hourly.csv      8760-hour typical year for Abu Dhabi (see data/README.md)
Outputs (results/): results.json, monthly_table.csv, chiller_library.csv
        (report/):  figures/*.pdf, figures/*.png

Deterministic: fixed random seed, no network access.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import carnot_penalty as cp  # noqa: E402

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
C_WC, C_AC, C_GREY = "#1b6ca8", "#d1495b", "#555555"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.frameon": False})


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


# =============================================================================
# 1. Calibrate the chiller model on the manufacturer-derived library
# =============================================================================
def calibrate_chillers(out):
    lib = cp.load_chiller_library((DATA / "Chillers.idf").read_text(errors="ignore"))
    rows = []
    for ch in lib:
        s = cp.library_sensitivity(ch)
        rows.append(dict(name=ch["name"], condenser=ch["condenser"], cop_ref=ch["cop_ref"],
                         t_chw_ref=ch["t_chw_ref"], t_cond_ref=ch["t_cond_ref"], s_per_K=s))
    df = pd.DataFrame(rows)
    df["valid"] = (df.s_per_K > 0.002) & (df.s_per_K < 0.10) & (df.cop_ref > 1.5) & (df.cop_ref < 10)
    df["lift_eff_K"] = 1.0 / df.s_per_K.where(df.valid)
    df.to_csv(RES / "chiller_library.csv", index=False)
    lib = [ch for ch, ok in zip(lib, df.valid) if ok]
    df = df[df.valid].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    models, cal = {}, {}
    for cond, key, sink in (("WaterCooled", "wc", "wb"),):
        idx = np.where(df.condenser.values == cond)[0]
        s = df.s_per_K.values[idx]
        s_med = float(np.median(s))
        lift = 1.0 / s_med
        cop_ref = float(np.median(df.cop_ref.values[idx]))
        t_cond_ref = float(np.median(df.t_cond_ref.values[idx]))
        t_ref_sink = t_cond_ref - TOWER_APPROACH if sink == "wb" else t_cond_ref
        label = "Water-cooled" if key == "wc" else "Air-cooled"
        models[key] = cp.Chiller(label, sink, cop_ref=cop_ref, t_ref=t_ref_sink, lift_ref=lift)

        # --- how well does COP ~ 1/lift reproduce each real machine's curve? ---
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
        own = {j: 1.0 / df.s_per_K.values[j] for j in idx}
        err_own = curve_errors(own, idx)                                # each machine's own lift (tests the 1/lift FORM)
        # --- out-of-sample: calibrate on a random half, test on the other half ---
        perm = rng.permutation(idx)
        train, test = perm[: len(perm) // 2], perm[len(perm) // 2:]
        lift_train = 1.0 / float(np.median(df.s_per_K.values[train]))
        err_test = curve_errors(lift_train, test)
        # --- bootstrap CI of the calibrated lift ---
        boot = np.array([1.0 / np.median(rng.choice(s, size=len(s), replace=True)) for _ in range(4000)])

        cal[key] = dict(n=int(len(idx)), s_median_pct=100 * s_med,
                        s_q25_pct=100 * float(np.quantile(s, 0.25)), s_q75_pct=100 * float(np.quantile(s, 0.75)),
                        lift_K=lift, lift_ci_lo=float(np.quantile(boot, 0.025)), lift_ci_hi=float(np.quantile(boot, 0.975)),
                        cop_ref=cop_ref, t_cond_ref=t_cond_ref, t_ref_sink=t_ref_sink,
                        eta_second_law=float(models[key].eta_second_law()),
                        rmse_class_median_pct=100 * float(np.median(err_class)),
                        rmse_class_p90_pct=100 * float(np.quantile(err_class, 0.9)),
                        rmse_own_median_pct=100 * float(np.median(err_own)),
                        lift_train_K=lift_train, rmse_test_median_pct=100 * float(np.median(err_test)),
                        n_train=int(len(train)), n_test=int(len(test)))
    # physical lift of each library machine at its reference point (condensing = entering water + 7 K,
    # evaporating = leaving chilled water - 1.5 K): what an ideal constant-eta Carnot fraction would use
    lp = float(np.median((df.t_cond_ref + 7.0) - (df.t_chw_ref - 1.5)))
    cal["wc"]["lift_phys_K"] = lp
    cal["wc"]["s_ideal_pct"] = 100.0 / lp
    cal["wc"]["n_total"] = int(len(rows))
    # The library holds no air-cooled machines -> standard design values (not library-calibrated):
    # COP 3.0 at 35 C ambient, condensing 14 K above ambient, evaporating 4 C -> lift 45 K
    models["ac"] = cp.Chiller("Air-cooled", "db", cop_ref=3.0, t_ref=35.0, lift_ref=45.0)
    cal["ac"] = dict(lift_K=45.0, cop_ref=3.0, t_ref_sink=35.0, s_median_pct=100.0 / 45.0,
                     eta_second_law=float(models["ac"].eta_second_law()))
    out["calibration"] = cal

    # ---------------- Figure 2: calibration / validation ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    idx = np.where(df.condenser.values == "WaterCooled")[0]
    dT = np.linspace(-8, 10, 37)
    for j in idx:
        ch = lib[j]
        t = ch["t_cond_ref"] + dT
        ok = (t >= ch["y_min"]) & (t <= ch["y_max"])
        ax[0].plot(dT[ok], cp.library_cop_ratio(ch, t[ok]), color="0.6", lw=0.5, alpha=0.6)
    L, Lp = models["wc"].lift_ref, lp
    ax[0].plot([], [], color="0.6", lw=0.8, label=f"{len(idx)} real water-cooled chillers")
    ax[0].plot(dT, Lp / (Lp + dT), color=C_GREY, lw=1.5, ls="--", label=f"ideal Carnot fraction, physical lift {Lp:.0f} K")
    ax[0].plot(dT, L / (L + dT), color=C_WC, lw=2.4, label=f"calibrated model, effective lift $L_0$ = {L:.0f} K")
    ax[0].set_ylim(0.6, 1.5)
    ax[0].set_xlabel("entering condenser-water temperature minus reference (K)")
    ax[0].set_ylabel("full-load COP / COP at reference")
    ax[0].set_title("(a) COP versus heat-sink temperature", loc="left", fontsize=9)
    ax[0].legend(fontsize=7.5)
    sv = 100 * df.s_per_K.values[idx]
    ax[1].hist(sv, bins=np.linspace(0, 6, 31), color=C_WC, alpha=0.6)
    ax[1].axvline(np.median(sv), color=C_WC, lw=2, label=f"library median {np.median(sv):.1f} %/K")
    ax[1].axvline(100 / Lp, color=C_GREY, lw=1.5, ls="--", label=f"ideal Carnot fraction {100 / Lp:.1f} %/K")
    ax[1].set_xlabel("COP loss per kelvin of sink temperature, $s$ (%/K)")
    ax[1].set_ylabel("number of chillers")
    ax[1].set_title("(b) Measured sensitivity at the reference point", loc="left", fontsize=9)
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "fig2_chiller_calibration")
    return models, df


# =============================================================================
# 2. Weather: hourly dry-bulb, wet-bulb and solar weights
# =============================================================================
def load_weather(out):
    wx = pd.read_csv(DATA / "weather_hourly.csv", comment="#")
    assert len(wx) == 8760, "expected an 8760-hour year"
    p = wx["p_Pa"].values if "p_Pa" in wx else cp.P_ATM
    wx["Twb"] = cp.wet_bulb(wx["T_C"].values, wx["Td_C"].values, p=p)
    out["weather"] = dict(n_hours=int(len(wx)),
                          T_max=float(wx.T_C.max()), T_min=float(wx.T_C.min()),
                          Twb_max=float(wx.Twb.max()), Twb_p99=float(wx.Twb.quantile(0.99)))
    return wx


def daily_arrays(wx):
    shape = (365, 24)
    return dict(month=wx["month"].values.reshape(shape)[:, 0],
                db=wx["T_C"].values.reshape(shape), wb=wx["Twb"].values.reshape(shape),
                td=wx["Td_C"].values.reshape(shape), g=np.clip(wx["ghi_Wm2"].values.reshape(shape), 0, None))


def with_td_swing(D, a):
    """Add a diurnal dew-point cycle of amplitude a [K] peaking at 15:00 (a<0: drier afternoons)."""
    h = np.arange(24) + 0.5
    td = np.minimum(D["td"] + a * np.cos(2 * np.pi * (h - 15.0) / 24.0)[None, :], D["db"])
    D2 = dict(D)
    D2["td"], D2["wb"] = td, cp.wet_bulb(D["db"], td)
    return D2


def with_range_scale(D, k):
    """Scale the daily dry-bulb range by k around the daily mean (moisture unchanged)."""
    mean = D["db"].mean(1, keepdims=True)
    db = mean + k * (D["db"] - mean)
    td = np.minimum(D["td"], db)
    D2 = dict(D)
    D2["db"], D2["td"], D2["wb"] = db, td, cp.wet_bulb(db, td)
    return D2


def penalties(chiller, D, use="uniform", make="solar", dT_charge=0.0, lift=None):
    """Daily Carnot penalty (exact, hour by hour) and its closed-form approximation."""
    ch = chiller if lift is None else chiller.with_lift(lift)
    sink = D[chiller.sink]
    g = D["g"]
    w_make = g if make == "solar" else np.tile(((np.arange(24) >= 9) & (np.arange(24) < 15)).astype(float), (365, 1))
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


# =============================================================================
# 3. Main analysis
# =============================================================================
def main():
    out = dict(params=dict(eta_tank=ETA_TANK, eta_batt=ETA_BATT, tower_approach_K=TOWER_APPROACH, seed=SEED,
                           breakeven_penalty_pct=100 * cp.breakeven_penalty(ETA_TANK, ETA_BATT)))
    models, libdf = calibrate_chillers(out)
    wx = load_weather(out)
    D = daily_arrays(wx)
    season = np.isin(D["month"], COOL_SEASON)
    pi_star = cp.breakeven_penalty(ETA_TANK, ETA_BATT)
    hours = np.arange(24)

    # ---------------- Figure 1: the two heat sinks ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
    for a, (months, title) in zip(ax, ((COOL_SEASON, "(a) Cooling season, May-Oct"), ([12, 1, 2], "(b) Winter, Dec-Feb"))):
        m = np.isin(D["month"], months)
        a2 = a.twinx()
        a2.fill_between(hours, D["g"][m].mean(0), color="#f2c14e", alpha=0.35, lw=0)
        a2.set_ylim(0, 1300)
        a2.grid(False)
        a2.spines["right"].set_visible(True)
        a2.set_ylabel("solar irradiance (W/m$^2$)" if a is ax[1] else "")
        if a is ax[0]:
            a2.set_yticklabels([])
        for key, col, lab in (("db", C_AC, "dry-bulb (air-cooled sink)"), ("wb", C_WC, "wet-bulb (water-cooled sink)")):
            a.plot(hours, D[key][m].mean(0), color=col, lw=2.2, label=lab)
            a.fill_between(hours, np.quantile(D[key][m], 0.1, 0), np.quantile(D[key][m], 0.9, 0), color=col, alpha=0.15, lw=0)
        a.set_xlabel("hour of day (local time)")
        a.set_xticks(range(0, 25, 4))
        a.set_title(title, loc="left", fontsize=9)
        a.set_zorder(a2.get_zorder() + 1)
        a.patch.set_visible(False)
    ax[0].set_ylabel("temperature ($^\\circ$C)")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig1_heat_sinks")

    # ---------------- core results ----------------
    R, monthly = {}, {"month": MONTHS}
    for key in ("ac", "wc"):
        P = penalties(models[key], D)
        sink = D[models[key].sink]
        g, night = D["g"], (D["g"] <= 0)
        eff = cp.cold_storage_efficiency(P["pi"], ETA_TANK)
        cop = models[key].cop(sink)
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
            closed_form_r2=float(np.corrcoef(P["pi"], P["pi_closed"])[0, 1] ** 2),
            cop_night_season=float(1.0 / P["use_work"][season].mean()),
            cop_solar_season=float(1.0 / P["make_work"][season].mean()),
            cop_design_min=float(cop.min()),
            clean_ratio_breakeven_season=float(cp.breakeven_clean_ratio(P["pi"][season].mean(), ETA_TANK)),
            # kWh_e needed per MWh_th of NIGHT cooling in the cooling season, three routes
            route_grid_night=1000 * float(P["use_work"][season].mean()),
            route_battery=1000 * float(P["use_work"][season].mean()) / ETA_BATT,
            route_cold_tank=1000 * float(P["make_work"][season].mean()) / ETA_TANK,
            breakeven_swing_K=float(models[key].lift(P["t_use"][season]).mean() * pi_star),
            sink_diurnal_range_season_K=float((sink[season].mean(0).max() - sink[season].mean(0).min())),
        )
        R[key]["_P"] = P
        monthly[f"pi_{key}_pct"] = [100 * float(P["pi"][D["month"] == m].mean()) for m in range(1, 13)]
        monthly[f"eff_{key}_pct"] = [100 * float(eff[D["month"] == m].mean()) for m in range(1, 13)]
        monthly[f"sink_swing_{key}_K"] = [float((P["t_make"] - P["t_use"])[D["month"] == m].mean()) for m in range(1, 13)]
    pd.DataFrame(monthly).round(2).to_csv(RES / "monthly_table.csv", index=False)

    # psychrometric damping: how much of a dry-bulb change shows up in the wet-bulb at constant moisture
    m = season
    t_mean, td_mean = float(D["db"][m].mean()), float(D["td"][m].mean())
    damp = float((cp.wet_bulb(t_mean + 0.5, td_mean) - cp.wet_bulb(t_mean - 0.5, td_mean)))
    out["damping"] = dict(T_mean=t_mean, Td_mean=td_mean, dTwb_dT=damp,
                          td_solar_minus_night_K=float(((D["g"] * D["td"]).sum(1) / D["g"].sum(1)
                                                        - (D["td"] * (D["g"] <= 0)).sum(1) / (D["g"] <= 0).sum(1))[m].mean()))

    # ---------------- Figure 3: headline result by month ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    xs = np.arange(12)
    for key, col, off in (("ac", C_AC, -0.2), ("wc", C_WC, 0.2)):
        P = R[key]["_P"]
        ax[0].bar(xs + off, monthly[f"pi_{key}_pct"], width=0.38, color=col, alpha=0.85, label=models[key].name)
        ax[1].plot(xs, monthly[f"eff_{key}_pct"], "-o", color=col, lw=2, ms=4, label=f"cold tank, {models[key].name.lower()} plant")
    ax[0].axhline(100 * pi_star, color=C_GREY, ls="--", lw=1.2)
    ax[0].text(11.45, 100 * pi_star + 0.6, f"battery break-even $\\pi^*$ = {100 * pi_star:.0f}%", ha="right", fontsize=7.5, color=C_GREY,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(MONTHS)
    ax[0].set_ylabel("Carnot penalty $\\pi$ (%)")
    ax[0].set_title("(a) Extra electricity when cold is made at solar hours", loc="left", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper right", ncol=2)
    ax[0].set_ylim(0, 30)
    ax[1].axhline(100 * ETA_BATT, color=C_GREY, ls="--", lw=1.2, label=f"Li-ion battery ({100 * ETA_BATT:.0f}%)")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(MONTHS)
    ax[1].set_ylabel("effective round-trip efficiency (%)")
    ax[1].set_title("(b) Storing solar energy as cold vs. in a battery", loc="left", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig3_penalty_by_month")

    # ---------------- Figure 4: closed form vs exact ----------------
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for key, col in (("ac", C_AC), ("wc", C_WC)):
        P = R[key]["_P"]
        ax[0].scatter(100 * P["pi_closed"], 100 * P["pi"], s=5, color=col, alpha=0.5, label=models[key].name)
    lim = [min(ax[0].get_xlim()[0], ax[0].get_ylim()[0]), max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
    ax[0].plot(lim, lim, color="k", lw=0.8)
    ax[0].set_xlabel("closed form: $\\Delta T_{sink}/L_{night}$ (%)")
    ax[0].set_ylabel("exact hourly calculation (%)")
    ax[0].set_title("(a) One formula captures all 365 days", loc="left", fontsize=9)
    ax[0].legend(fontsize=8)
    r = np.linspace(0, 1.1, 50)
    for key, col in (("ac", C_AC), ("wc", C_WC)):
        rstar = R[key]["clean_ratio_breakeven_season"]
        ax[1].plot(r, 100 * (1 - r / rstar), color=col, lw=2, label=f"{models[key].name}: break-even $r^*$ = {rstar:.2f}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("$r$ = emissions (or price) of solar-hour electricity / night electricity")
    ax[1].set_ylabel("emissions saved on shifted cooling (%)")
    ax[1].set_title("(b) When does solar-following cold storage pay?", loc="left", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig4_closed_form_and_breakeven")

    # ---------------- sensitivity (one at a time) ----------------
    def season_eff(key, Dv=None, **kw):
        eta = kw.pop("eta_tank", ETA_TANK)
        return 100 * float(cp.cold_storage_efficiency(penalties(models[key], D if Dv is None else Dv, **kw)["pi"][season], eta).mean())

    D_dry, D_hum = with_td_swing(D, -3.0), with_td_swing(D, +3.0)
    D_k08, D_k12 = with_range_scale(D, 0.8), with_range_scale(D, 1.2)

    sens = {}
    for key in ("ac", "wc"):
        L = models[key].lift_ref
        sens[key] = {
            "base": season_eff(key),
            "Lift $L_0$ $-25\\%$ / $+25\\%$": (season_eff(key, lift=0.75 * L), season_eff(key, lift=1.25 * L)),
            "Afternoon dew point 3 K higher / 3 K lower": (season_eff(key, Dv=D_hum), season_eff(key, Dv=D_dry)),
            "Daily temperature range x1.2 / x0.8": (season_eff(key, Dv=D_k12), season_eff(key, Dv=D_k08)),
            "Tank efficiency 0.90 / 0.99": (season_eff(key, eta_tank=0.90), season_eff(key, eta_tank=0.99)),
            "Night load: uniform / degree-hour weighted": (season_eff(key), season_eff(key, use="degree")),
            "Charging window: solar-weighted / flat 09-15 h": (season_eff(key), season_eff(key, make="flat")),
            "Colder charging water: 0 K / 2 K": (season_eff(key), season_eff(key, dT_charge=2.0)),
            "Ice storage (evaporator 12 K colder)": (season_eff(key), season_eff(key, dT_charge=12.0)),
        }
    out["sensitivity"] = {k: {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in d.items()} for k, d in sens.items()}

    # ---------------- Monte Carlo over machine-to-machine and storage uncertainty ----------------
    rng = np.random.default_rng(SEED)
    N = 20000
    mc = {}
    eta_t = rng.uniform(0.90, 0.99, N)
    eta_b = rng.uniform(0.80, 0.90, N)
    for key in ("wc", "ac"):
        wc_lifts = libdf.lift_eff_K.values[libdf.condenser.values == "WaterCooled"]
        rel = rng.choice(wc_lifts, size=N, replace=True) / np.median(wc_lifts)   # machine-to-machine spread
        Ls = models[key].lift_ref * rel
        P = R[key]["_P"]
        dT = (P["t_make"] - P["t_use"])[season]
        tu = P["t_use"][season]
        # weather uncertainty: daily range x U(0.8,1.2); afternoon dew-point shift U(-3,+3) K (wet-bulb sink only)
        k_s, a_s = rng.uniform(0.8, 1.2, N), rng.uniform(-3.0, 3.0, N)
        if key == "wc":
            d_hi = (penalties(models[key], D_hum)["t_make"] - penalties(models[key], D_hum)["t_use"])[season].mean()
            d_lo = (penalties(models[key], D_dry)["t_make"] - penalties(models[key], D_dry)["t_use"])[season].mean()
            slope = (d_hi - d_lo) / 6.0
        else:
            slope = 0.0
        dTs = k_s[:, None] * dT[None, :] + (slope * a_s)[:, None]
        pis = (dTs / (Ls[:, None] + tu[None, :] - models[key].t_ref)).mean(1)
        eff = eta_t / (1 + pis)
        mc[key] = dict(eff=eff, wins=float((eff > eta_b).mean()),
                       eff_p5=float(np.quantile(eff, 0.05)), eff_p50=float(np.quantile(eff, 0.5)), eff_p95=float(np.quantile(eff, 0.95)),
                       pi_p5=100 * float(np.quantile(pis, 0.05)), pi_p50=100 * float(np.quantile(pis, 0.5)), pi_p95=100 * float(np.quantile(pis, 0.95)))
    out["monte_carlo"] = {k: {kk: vv for kk, vv in v.items() if kk != "eff"} for k, v in mc.items()}
    out["monte_carlo"]["n"] = N

    # ---------------- Figure 5: uncertainty ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    bins = np.linspace(60, 105, 46)
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
        base = sens[key]["base"]
        for yy, lab in zip(y, labels):
            lo, hi = sens[key][lab]
            ax[1].barh(yy + off, hi - lo, left=lo, height=0.32, color=col, alpha=0.75)
        ax[1].axvline(base, color=col, lw=1, ls=":")
    ax[1].axvline(100 * ETA_BATT, color=C_GREY, ls="--", lw=1.2)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([l.replace("$", "").replace("\\%", "%").replace("L_0", "L0") for l in labels], fontsize=7.5)
    ax[1].set_xlabel("cooling-season effective efficiency (%)   [dashed: battery 85%]")
    ax[1].set_title("(b) One-at-a-time sensitivity", loc="left", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig5_uncertainty")

    # ---------------- write results ----------------
    for key in R:
        R[key].pop("_P")
    out["results"] = R
    out["models"] = {k: dict(name=m.name, sink=m.sink, cop_ref=m.cop_ref, t_ref=m.t_ref, lift_ref=m.lift_ref) for k, m in models.items()}
    out["monthly"] = monthly
    (RES / "results.json").write_text(json.dumps(out, indent=2))

    print(json.dumps({k: out[k] for k in ("params", "calibration", "weather", "damping", "results", "monte_carlo", "sensitivity")}, indent=1))


if __name__ == "__main__":
    main()
