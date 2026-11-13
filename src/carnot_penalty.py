"""
carnot_penalty.py  --  model core for "The Carnot Penalty" (Carnot 2026 submission)

Everything the report claims is computed from the functions in this file.
The model has four small pieces:

  1. Psychrometrics : dry-bulb + dew point  ->  thermodynamic wet-bulb temperature
  2. Chiller        : COP as a fixed fraction of the Carnot COP between an
                      evaporating and a condensing temperature
  3. Carnot penalty : extra electricity needed when cold is MADE in the solar
                      hours instead of in the (cooler) hours when it is USED
  4. Storage choice : cold tank vs. battery vs. no storage

Units: temperatures in deg C unless a name ends in _K; pressures in Pa.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ----------------------------------------------------------------------------
# 1. Psychrometrics
# ----------------------------------------------------------------------------
P_ATM = 101325.0  # Pa, sea level (Abu Dhabi airport is 27 m above sea level)


def p_sat(T):
    """Saturation vapour pressure of water [Pa] at T [deg C].
    Magnus form with the Alduchov & Eskridge (1996) coefficients."""
    T = np.asarray(T, dtype=float)
    return 610.94 * np.exp(17.625 * T / (T + 243.04))


def humidity_ratio(p_w, p=P_ATM):
    """kg water vapour per kg dry air, from vapour partial pressure p_w [Pa]."""
    return 0.621945 * p_w / (p - p_w)


def dew_point_from_rh(T, rh_percent):
    """Dew point [deg C] from dry-bulb [deg C] and relative humidity [%] (inverse Magnus)."""
    T = np.asarray(T, dtype=float)
    g = np.log(np.clip(np.asarray(rh_percent, dtype=float), 1e-6, 100.0) / 100.0) + 17.625 * T / (T + 243.04)
    return 243.04 * g / (17.625 - g)


def wet_bulb(T, Td, p=P_ATM, n_iter=50):
    """Thermodynamic wet-bulb temperature [deg C].

    Solves the adiabatic-saturation energy balance (ASHRAE Fundamentals, Ch. 1):

        W = [(2501 - 2.326 Twb) Ws(Twb) - 1.006 (T - Twb)] / (2501 + 1.86 T - 4.186 Twb)

    for Twb by bisection.  The root always lies between the dew point and the
    dry-bulb temperature, and the right-hand side increases monotonically with
    Twb, so bisection cannot fail.
    """
    T = np.asarray(T, dtype=float)
    Td = np.minimum(np.asarray(Td, dtype=float), T)
    W = humidity_ratio(p_sat(Td), p)
    lo, hi = Td.copy(), T.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        Ws = humidity_ratio(p_sat(mid), p)
        W_mid = ((2501.0 - 2.326 * mid) * Ws - 1.006 * (T - mid)) / (2501.0 + 1.86 * T - 4.186 * mid)
        too_high = W_mid > W
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def wet_bulb_stull(T, rh_percent):
    """Stull (2011) empirical wet-bulb formula. Used ONLY as an independent test of wet_bulb()."""
    T = np.asarray(T, dtype=float)
    rh = np.asarray(rh_percent, dtype=float)
    return (T * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(T + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


# ----------------------------------------------------------------------------
# 2. Chiller model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Chiller:
    """Carnot-fraction chiller.

    COP = eta * T_e / (T_c - T_e)          (T in kelvin)
      T_c = T_sink + approach   (condensing temperature follows the heat sink)
      T_e = constant            (set by the chilled-water temperature)
      eta = constant second-law efficiency

    Because eta and T_e are constant, COP is inversely proportional to the
    temperature lift (T_c - T_e).  The model is therefore fully described by
      cop_ref   : COP at the reference sink temperature
      t_ref     : reference sink temperature [deg C]
      lift_ref  : lift (T_c - T_e) at the reference sink temperature [K]
    """
    name: str
    sink: str          # "wb" (water-cooled: rejects heat to the wet-bulb) or "db" (air-cooled: dry-bulb)
    cop_ref: float
    t_ref: float
    lift_ref: float
    t_evap_K: float = 277.15  # 4 deg C evaporating temperature

    def lift(self, t_sink):
        return self.lift_ref + (np.asarray(t_sink, dtype=float) - self.t_ref)

    def cop(self, t_sink, dT_charge=0.0):
        """COP at sink temperature t_sink [deg C].
        dT_charge [K] lowers the evaporating temperature (e.g. ~12 K for ice making)."""
        lift = self.lift(t_sink) + dT_charge
        return self.cop_ref * (self.lift_ref / lift) * ((self.t_evap_K - dT_charge) / self.t_evap_K)

    def eta_second_law(self):
        """Implied second-law efficiency = COP / Carnot COP at reference."""
        return self.cop_ref * self.lift_ref / self.t_evap_K

    def with_lift(self, lift_ref):
        return replace(self, lift_ref=float(lift_ref))


# Physics-only defaults (used if the chiller library is unavailable, and as a cross-check)
#   water-cooled @ AHRI 550/590: condensing ~36 C, evaporating ~5 C  -> lift ~31 K
#   air-cooled   @ 35 C ambient: condensing ~49 C, evaporating ~4 C  -> lift ~45 K
WATER_COOLED_PHYS = Chiller("water-cooled (physics default)", "wb", cop_ref=6.1, t_ref=25.4, lift_ref=31.0)
AIR_COOLED_PHYS = Chiller("air-cooled (physics default)", "db", cop_ref=3.0, t_ref=35.0, lift_ref=45.0)


# ----------------------------------------------------------------------------
# 3. The Carnot penalty
# ----------------------------------------------------------------------------
def specific_work(cop, weights):
    """Weighted mean electricity per unit of cold, sum(w/COP)/sum(w)."""
    cop = np.asarray(cop, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w / cop) / np.sum(w))


def carnot_penalty(cop, w_make, w_use):
    """Carnot penalty  pi = (electricity per unit cold when MADE with weights w_make)
                          / (electricity per unit cold when made on demand, weights w_use) - 1.

    pi > 0 : shifting production costs extra electricity (thermodynamic penalty)
    pi < 0 : shifting production actually saves electricity (thermodynamic bonus)
    """
    return specific_work(cop, w_make) / specific_work(cop, w_use) - 1.0


def carnot_penalty_closed_form(t_sink_make, t_sink_use, lift_use):
    """Closed-form result derived in the report:  pi = (T_make - T_use) / lift_use."""
    return (t_sink_make - t_sink_use) / lift_use


# ----------------------------------------------------------------------------
# 4. Storage choice
# ----------------------------------------------------------------------------
def cold_storage_efficiency(pi, eta_tank=0.97):
    """Electricity-equivalent round-trip efficiency of 'solar-made cold':
    kWh of on-demand chiller electricity avoided per kWh of solar electricity used."""
    return eta_tank / (1.0 + np.asarray(pi, dtype=float))


def breakeven_penalty(eta_tank=0.97, eta_batt=0.85):
    """Cold storage beats a battery on energy if pi < eta_tank/eta_batt - 1."""
    return eta_tank / eta_batt - 1.0


def breakeven_clean_ratio(pi, eta_tank=0.97):
    """Solar-following cold storage cuts emissions (or cost) if
    e_solar / e_night < eta_tank / (1 + pi)."""
    return eta_tank / (1.0 + np.asarray(pi, dtype=float))


# ----------------------------------------------------------------------------
# 5. Chiller library (EnergyPlus datasets/Chillers.idf) -- used for calibration
# ----------------------------------------------------------------------------
def parse_idf_objects(text):
    """Minimal IDF reader: returns a list of field-lists, one per object."""
    lines = [ln.split("!")[0] for ln in text.splitlines()]
    objs = []
    for chunk in " ".join(lines).split(";"):
        fields = [f.strip() for f in chunk.split(",")]
        if fields and fields[0]:
            objs.append(fields)
    return objs


def load_chiller_library(text):
    """Return a list of dicts for every Chiller:Electric:EIR object with a usable EIR-f(T) curve.

    Each dict has: name, cop_ref, t_chw_ref, t_cond_ref, condenser ('WaterCooled'/'AirCooled'),
    coef (6 biquadratic coefficients of EIRFT), y_min, y_max (valid condenser-temperature range).
    """
    objs = parse_idf_objects(text)
    curves = {}
    for f in objs:
        if f[0].lower() == "curve:biquadratic" and len(f) >= 12:
            try:
                curves[f[1].lower()] = dict(coef=[float(x) for x in f[2:8]],
                                            x_min=float(f[8]), x_max=float(f[9]),
                                            y_min=float(f[10]), y_max=float(f[11]))
            except ValueError:
                continue
    out = []
    for f in objs:
        if f[0].lower() != "chiller:electric:eir" or len(f) < 10:
            continue
        cond = next((x for x in f if x.lower() in ("watercooled", "aircooled", "evaporativelycooled")), None)
        cur = curves.get(f[9].lower())
        if cond is None or cur is None:
            continue
        try:
            out.append(dict(name=f[1], cop_ref=float(f[3]), t_chw_ref=float(f[4]), t_cond_ref=float(f[5]),
                            condenser=cond, **cur))
        except ValueError:
            continue
    return out


def eirft(ch, t_cond, t_chw=None):
    """Energy-input-ratio temperature modifier of a library chiller (biquadratic curve)."""
    x = ch["t_chw_ref"] if t_chw is None else t_chw
    c = ch["coef"]
    y = np.asarray(t_cond, dtype=float)
    return c[0] + c[1] * x + c[2] * x * x + c[3] * y + c[4] * y * y + c[5] * x * y


def library_cop_ratio(ch, t_cond):
    """Full-load COP of a library chiller at t_cond divided by its COP at its reference point."""
    return eirft(ch, ch["t_cond_ref"]) / eirft(ch, t_cond)


def library_sensitivity(ch, h=0.5):
    """s = -d ln(COP)/dT_cond at the reference point [1/K] (central difference)."""
    t0 = ch["t_cond_ref"]
    return float((np.log(eirft(ch, t0 + h)) - np.log(eirft(ch, t0 - h))) / (2 * h))
