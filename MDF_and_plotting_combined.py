"""
MDF vs Omega — combined script, organised as Spyder cells (#%%)
==================================================================
Run cell-by-cell (Ctrl+Enter in Spyder) instead of the whole script at once.

Scenario design (same 4 combinations for EMP and TCA):
    std_free    — "standard" boundary concentrations (1 mM), no internal bounds   -> Omega = MDF
    std_bound   — standard boundary (1 mM), physiological internal bounds         -> MDF <= Omega
    phys_free   — physiological boundary concentrations, no internal bounds       -> Omega = MDF
    phys_bound  — physiological boundary, physiological internal bounds          -> MDF <= Omega

Layout:
    Cell 0   — imports, constants, style, scenario labels/colors
    Cell 1   — model/helper functions (stoich, dG0_eff, omega, MDF LP, generic 4-scenario runner)
    Cell 2   — RUN: solves all 4 scenarios for EMP and for TCA -> emp_results / tca_results
    Cell 3-6 — analysis: EMP std_free, std_bound, phys_free, phys_bound
    Cell 7-10— analysis: TCA std_free, std_bound, phys_free, phys_bound
    Cell 11  — plot: Figure 1 (EMP, 3 panels)
    Cell 12  — plot: Figure 2 (TCA, 3 panels)
    Cell 13  — plot: Figure 3 (Omega vs MDF scatter, all scenarios x both pathways)
"""

#%% Cell 0 — imports, constants, style
import math
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

RT   = 2.479   # kJ/mol at 298.15 K
V_NC = 1.0     # glucose consumption = min flux (EMP); also used for TCA
REF_CONC_M = 1e-3   # 1 mM — the "m" reference state used for dGm throughout
REF_EXCLUDED = ("h2o", "h")  # species kept at their real values, not shifted to 1 mM

# scenario colors/labels are shared between EMP and TCA (same 4 combinations)
SCENARIO_COLORS = {
    "std_free":   "#2E86AB",   # blue   — 1 mM boundary, no internal bounds
    "std_bound":  "#E84855",   # red    — 1 mM boundary, physiological internal bounds
    "phys_free":  "#3BB273",   # green  — physiological boundary, no internal bounds
    "phys_bound": "#F18F01",   # orange — physiological boundary, physiological internal bounds
}
SCENARIO_LABELS = {
    "std_free":   "1 mM boundary\n(no bounds)",
    "std_bound":  "1 mM boundary\n(physiol. bounds)",
    "phys_free":  "Physiol. boundary\n(no bounds)",
    "phys_bound": "Physiol. boundary\n(physiol. bounds)",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


#%% Cell 1 — model definitions and helper functions

# ── KEGG compound IDs for every metabolite in the model ─────────────────────
# eQuilibrator's free-text lookup of bare shorthand names (e.g. "h", "pi",
# "fdp") can match multiple rows in its local compound cache and raise
# MultipleResultsFound. Namespace-qualified KEGG IDs ("kegg:C00080") are
# unambiguous, so the reaction formulas below use these instead of the
# shorthand acronyms (the shorthand names are still used everywhere else in
# this script — stoichiometric matrices, labels, plots — only the formula
# strings handed to eQuilibrator are KEGG-based).
METABOLITE_KEGG = {
    # EMP glycolysis -> lactate
    "glc-D":  "C00031",  # D-glucose
    "atp":    "C00002",  # ATP
    "g6p":    "C00092",  # glucose 6-phosphate
    "adp":    "C00008",  # ADP
    "h":      "C00080",  # H+
    "f6p":    "C00085",  # fructose 6-phosphate
    "fdp":    "C00354",  # fructose 1,6-bisphosphate
    "dhap":   "C00111",  # dihydroxyacetone phosphate
    "g3p":    "C00118",  # glyceraldehyde 3-phosphate
    "nad":    "C00003",  # NAD+
    "pi":     "C00009",  # orthophosphate
    "13dpg":  "C00236",  # 1,3-bisphosphoglycerate
    "nadh":   "C00004",  # NADH
    "3pg":    "C00197",  # 3-phosphoglycerate
    "2pg":    "C00631",  # 2-phosphoglycerate
    "pep":    "C00074",  # phosphoenolpyruvate
    "pyr":    "C00022",  # pyruvate
    "lac-L":  "C00186",  # L-lactate
    "h2o":    "C00001",  # water
    # TCA cycle
    "oaa":    "C00036",  # oxaloacetate
    "accoa":  "C00024",  # acetyl-CoA
    "cit":    "C00158",  # citrate
    "coa":    "C00010",  # coenzyme A
    "icit":   "C00311",  # isocitrate
    "akg":    "C00026",  # 2-oxoglutarate (alpha-ketoglutarate)
    "co2":    "C00011",  # CO2
    "succoa": "C00091",  # succinyl-CoA
    "succ":   "C00042",  # succinate
    "fad":    "C00016",  # FAD
    "fadh2":  "C01352",  # FADH2
    "fum":    "C00122",  # fumarate
    "mal-L":  "C00149",  # L-malate
}


def to_kegg_formula(formula, kegg_dict=METABOLITE_KEGG):
    """Rewrite a shorthand eQuilibrator formula ('glc-D + atp = g6p + adp')
    into KEGG-qualified tokens ('kegg:C00031 + kegg:C00002 = kegg:C00092 + kegg:C00008')."""
    left, right = formula.split("=")

    def conv_side(side):
        terms = [t.strip() for t in side.split("+")]
        conv = []
        for t in terms:
            parts = t.split()
            coeff, name = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            if name not in kegg_dict:
                raise KeyError(f"'{name}' not in METABOLITE_KEGG — add its KEGG ID")
            token = f"kegg:{kegg_dict[name]}"
            conv.append(f"{coeff} {token}".strip())
        return " + ".join(conv)

    return f"{conv_side(left)} = {conv_side(right)}"


# ── EMP glycolysis -> lactate ───────────────────────────────────────────────
# formulas use KEGG-qualified compound IDs (see METABOLITE_KEGG above) so
# fetch_dgm_equilibrator() doesn't hit ambiguous shorthand-name lookups
REACTIONS = [
    ("HEX",   to_kegg_formula("glc-D + atp = g6p + adp"),                  1),
    ("PGI",   to_kegg_formula("g6p = f6p"),                                1),
    ("PFK",   to_kegg_formula("f6p + atp = fdp + adp"),                    1),
    ("ALD",   to_kegg_formula("fdp = dhap + g3p"),                         1),
    ("TPI",   to_kegg_formula("dhap = g3p"),                               1),
    ("GAPDH", to_kegg_formula("g3p + nad + pi = 13dpg + nadh + h"),        2),
    ("PGK",   to_kegg_formula("13dpg + adp = 3pg + atp"),                  2),
    ("PGM",   to_kegg_formula("3pg = 2pg"),                                2),
    ("ENO",   to_kegg_formula("2pg = pep + h2o"),                          2),
    ("PYK",   to_kegg_formula("pep + adp = pyr + atp"),                    2),
    ("LDH",   to_kegg_formula("pyr + nadh + h = lac-L + nad"),             2),
]
REACTION_NAMES   = [r[0] for r in REACTIONS]
FLUX_MULTIPLIERS = np.array([r[2] for r in REACTIONS], dtype=float)

INTERNAL_METABOLITES = [
    "g6p", "f6p", "fdp", "dhap", "g3p", "13dpg", "3pg", "2pg", "pep", "pyr",
    "nad", "nadh",
]
BOUNDARY_METABOLITES = [
    "glc-D", "lac-L", "atp", "adp", "pi", "h2o", "h",
]
_met_idx_internal = {m: i for i, m in enumerate(INTERNAL_METABOLITES)}
_met_idx_boundary = {m: i for i, m in enumerate(BOUNDARY_METABOLITES)}

# "standard" boundary: all net-conversion metabolites at 1 mM
EMP_STD_BOUNDARY = {
    "glc-D": 1e-3, "lac-L": 1e-3, "atp": 1e-3, "adp": 1e-3, "pi": 1e-3,
    "h2o": 1.0, "h": 1e-7,
}
# physiological boundary concentrations
EMP_PHYS_BOUNDARY = {
    "glc-D": 5e-3, "lac-L": 1e-3, "atp": 3e-3, "adp": 5e-4, "pi": 1e-2,
    "h2o": 1.0, "h": 1e-7,
}
# physiological internal-metabolite bounds (1 uM - 10 mM)
EMP_PHYS_INT_BOUNDS = {m: (1e-6, 10e-3) for m in INTERNAL_METABOLITES}

DG0_LITERATURE = {
    "HEX": -16.7, "PGI": +2.2, "PFK": -14.5, "ALD": +22.8, "TPI": -7.5,
    "GAPDH": +6.3, "PGK": -18.9, "PGM": +4.4, "ENO": -3.6,
    "PYK": -31.7, "LDH": -25.1,
}


def build_stoich_matrices():
    """Returns S_int (n_rxn x n_internal), S_bnd (n_rxn x n_boundary) for EMP."""
    n_rxn = len(REACTIONS)
    S_int = np.zeros((n_rxn, len(INTERNAL_METABOLITES)))
    S_bnd = np.zeros((n_rxn, len(BOUNDARY_METABOLITES)))
    stoich_data = [
        ("HEX",   [("glc-D",-1), ("atp",-1),  ("g6p",  1), ("adp", 1)]),
        ("PGI",   [("g6p",  -1), ("f6p",  1)]),
        ("PFK",   [("f6p",  -1), ("atp", -1),  ("fdp",  1), ("adp", 1)]),
        ("ALD",   [("fdp",  -1), ("dhap", 1),  ("g3p",  1)]),
        ("TPI",   [("dhap", -1), ("g3p",  1)]),
        ("GAPDH", [("g3p",  -1), ("nad", -1),  ("pi",  -1), ("13dpg",1),("nadh",1),("h",1)]),
        ("PGK",   [("13dpg",-1), ("adp", -1),  ("3pg",  1), ("atp",  1)]),
        ("PGM",   [("3pg",  -1), ("2pg",  1)]),
        ("ENO",   [("2pg",  -1), ("pep",  1),  ("h2o",  1)]),
        ("PYK",   [("pep",  -1), ("adp", -1),  ("pyr",  1), ("atp", 1)]),
        ("LDH",   [("pyr",  -1), ("nadh",-1),  ("h",   -1), ("lac-L",1), ("nad", 1)]),
    ]
    for rxn_name, entries in stoich_data:
        i = REACTION_NAMES.index(rxn_name)
        for met, coeff in entries:
            if met in _met_idx_internal:
                S_int[i, _met_idx_internal[met]] = coeff
            elif met in _met_idx_boundary:
                S_bnd[i, _met_idx_boundary[met]] = coeff
    return S_int, S_bnd


def get_dg0_prime_literature():
    vals = np.array([DG0_LITERATURE[n] for n in REACTION_NAMES])
    errs = np.zeros(len(REACTION_NAMES))
    return vals, errs


def fetch_dgm_equilibrator(reactions, pH=7.0, ionic_strength=0.1, temperature=298.15):
    """
    Fetch dGm (physiological standard, all reactants at 1 mM except H2O/H+/
    ionic strength) directly from eQuilibrator — the proper alternative to
    dg0_to_dgm()'s literature-dG0'-plus-conversion approach.
    `reactions` is a list of (name, eQuilibrator-formula, flux_mult) tuples,
    e.g. REACTIONS or TCA_REACTIONS. Requires: pip install equilibrator-api

    If eQuilibrator's local compound cache has duplicate/ambiguous entries for
    a shorthand name, parsing raises sqlalchemy.exc.MultipleResultsFound. This
    wraps each reaction individually so a single bad compound name doesn't
    block the whole batch, and reports which reaction/name failed.
    """
    from equilibrator_api import ComponentContribution, Q_

    cc = ComponentContribution()
    cc.temperature    = Q_(temperature, "K")
    cc.p_h            = Q_(pH, "")
    cc.ionic_strength = Q_(ionic_strength, "M")

    dgm, dgm_err = [], []
    for name, formula, _ in reactions:
        try:
            rxn    = cc.parse_reaction_formula(formula)
            result = cc.physiological_dg_prime(rxn)   # the "m" / 1 mM standard
            val    = result.magnitude
            dgm.append(val.nominal_value)
            dgm_err.append(val.std_dev)
            print(f"  {name:6s}  dGm = {val.nominal_value:+8.2f} +/- {val.std_dev:.2f} kJ/mol")
        except Exception as e:
            print(f"  [FAILED] {name:6s}  formula = '{formula}'  -> {type(e).__name__}: {e}")
            raise

    return np.array(dgm), np.array(dgm_err)


def dg0_to_dgm(dg0_prime, S_int, S_bnd, boundary_metabolites,
               ref_conc=REF_CONC_M, excluded=REF_EXCLUDED, RT=RT):
    """
    Convert standard dG0' (1 M reference) to dGm (1 mM reference, "physiological
    standard"): every reactant is shifted to ref_conc, except species in
    `excluded` (H2O, H+) which keep their real/biochemical-standard treatment.
    This matches eQuilibrator's physiological_dg_prime convention.

        dGm_i = dG0'_i + RT*ln(ref_conc) * sum_j(nu_ij)   [j ranges over all
                                                            reactants except excluded]
    """
    bnd_mask = np.array([0.0 if m in excluded else 1.0 for m in boundary_metabolites])
    net_stoich = S_int.sum(axis=1) + (S_bnd * bnd_mask).sum(axis=1)
    shift = RT * math.log(ref_conc) * net_stoich
    return dg0_prime + shift


def compute_dg0_eff(dgm_prime, S_bnd, boundary_conc_M, boundary_metabolites, flux_mult,
                     ref_conc=REF_CONC_M, excluded=REF_EXCLUDED, RT=RT):
    """
    dG0_eff_i = dGm_i + RT * sum_j S_bnd[i,j] * ln(x_j_fixed / ref_conc)

    Boundary species are corrected RELATIVE to the dGm reference (ref_conc),
    so a boundary metabolite sitting exactly at ref_conc contributes zero
    correction. Excluded species (H2O, H+) use their real absolute value,
    since dgm_prime never shifted their baseline away from 1 M / unit activity.
    """
    ln_terms = []
    for m in boundary_metabolites:
        if m in excluded:
            ln_terms.append(math.log(boundary_conc_M[m]))
        else:
            ln_terms.append(math.log(boundary_conc_M[m] / ref_conc))
    ln_bnd = np.array(ln_terms)
    correction = RT * S_bnd @ ln_bnd
    dg0_eff = dgm_prime + correction
    dg_cat = np.sum(flux_mult * dg0_eff)
    return dg0_eff, dg_cat


def compute_omega(dg0_eff, flux_multipliers, v_nc):
    """Omega = -dG_CAT / sum(v_i / v_NC)."""
    dg_cat = np.sum(flux_multipliers * dg0_eff)
    sum_v_over_vnc = np.sum(flux_multipliers / v_nc)
    omega = -dg_cat / sum_v_over_vnc
    return omega, dg_cat, sum_v_over_vnc


def solve_mdf_gurobi(dg0_eff, S_int, RT=RT, conc_bounds_M=None,
                      internal_metabolites=None, met_idx_internal=None,
                      reaction_names=None, model_name="MDF", ref_conc=REF_CONC_M):
    """
    Generic MDF LP solver — works for any pathway given the right S_int etc.
    Internal ln_x[j] represents ln(C_j / ref_conc) — i.e. concentrations are
    relative to the dGm reference state (1 mM by default), consistent with
    dg0_eff being expressed on that same basis. conc_bounds_M is still given
    in absolute molar units; it's converted to the relative basis internally.
    """
    n_rxn, n_int = S_int.shape
    model = gp.Model(model_name)
    model.setParam("OutputFlag", 0)

    B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="B")

    if conc_bounds_M is None:
        ln_x = model.addVars(n_int, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="ln_x")
    else:
        lb_arr = np.full(n_int, -30.0)
        ub_arr = np.full(n_int,  30.0)
        for met, (lb_M, ub_M) in conc_bounds_M.items():
            if met in met_idx_internal:
                j = met_idx_internal[met]
                lb_arr[j] = math.log(lb_M / ref_conc)
                ub_arr[j] = math.log(ub_M / ref_conc)
        ln_x = model.addVars(n_int, lb=list(lb_arr), ub=list(ub_arr), name="ln_x")

    for i, rxn_name in enumerate(reaction_names):
        conc_term = gp.quicksum(
            RT * float(S_int[i, j]) * ln_x[j]
            for j in range(n_int) if S_int[i, j] != 0
        )
        model.addConstr(float(dg0_eff[i]) + conc_term + B <= 0, name=f"thermo_{rxn_name}")

    model.setObjective(B, GRB.MAXIMIZE)
    model.optimize()

    results = {"status": model.Status, "model": model}
    if model.Status == GRB.OPTIMAL:
        results["MDF"] = B.X
        ln_x_vals = {internal_metabolites[j]: ln_x[j].X for j in range(n_int)}
        results["ln_x"] = ln_x_vals
        dg_prime_eff = {}
        for i, rxn_name in enumerate(reaction_names):
            cc = RT * sum(S_int[i, j] * ln_x_vals[internal_metabolites[j]]
                          for j in range(n_int))
            dg_prime_eff[rxn_name] = dg0_eff[i] + cc
        results["dg_prime_eff"] = dg_prime_eff
    else:
        print(f"[WARNING] Gurobi status {model.Status} — infeasible or unbounded ({model_name}).")
    return results


def run_four_scenarios(dgm_prime, S_bnd, S_int, reaction_names, flux_mult,
                        boundary_metabolites, internal_metabolites, met_idx_internal,
                        boundary_std, boundary_phys, phys_int_bounds,
                        v_nc=1.0, model_prefix="MDF", ref_conc=REF_CONC_M):
    """
    Solves the 4 standard scenarios (std/phys boundary x free/bound internal
    concentrations) for any pathway, given its stoichiometry and dGm values
    (dg0' already converted to the 1 mM reference via dg0_to_dgm).
    Returns a dict keyed by 'std_free' / 'std_bound' / 'phys_free' / 'phys_bound'.
    """
    dg0_eff_std,  dg_cat_std  = compute_dg0_eff(dgm_prime, S_bnd, boundary_std,  boundary_metabolites, flux_mult, ref_conc=ref_conc)
    dg0_eff_phys, dg_cat_phys = compute_dg0_eff(dgm_prime, S_bnd, boundary_phys, boundary_metabolites, flux_mult, ref_conc=ref_conc)
    omega_std  = compute_omega(dg0_eff_std,  flux_mult, v_nc)[0]
    omega_phys = compute_omega(dg0_eff_phys, flux_mult, v_nc)[0]

    combos = [
        ("std_free",   dg0_eff_std,  dg_cat_std,  None,             omega_std),
        ("std_bound",  dg0_eff_std,  dg_cat_std,  phys_int_bounds,  omega_std),
        ("phys_free",  dg0_eff_phys, dg_cat_phys, None,             omega_phys),
        ("phys_bound", dg0_eff_phys, dg_cat_phys, phys_int_bounds,  omega_phys),
    ]
    results = {}
    for key, dg0_eff, dg_cat, bounds, omega in combos:
        r = solve_mdf_gurobi(dg0_eff, S_int, conc_bounds_M=bounds,
                              internal_metabolites=internal_metabolites,
                              met_idx_internal=met_idx_internal,
                              reaction_names=reaction_names,
                              model_name=f"{model_prefix}_{key}", ref_conc=ref_conc)
        r.update(dg0_eff=dg0_eff, dg_cat=dg_cat, omega=omega,
                  label=SCENARIO_LABELS[key], color=SCENARIO_COLORS[key],
                  bounds=(bounds is not None))
        results[key] = r
    return results


def cumulative_dg_profile(dg_prime_dict, rxn_names, flux_mult):
    """x,y for a waterfall line plot; segment width = flux_mult."""
    xs = [0.0]
    ys = [0.0]
    x = 0.0
    for name, v in zip(rxn_names, flux_mult):
        dg = dg_prime_dict[name]
        xs.append(x)
        xs.append(x + v)
        ys.append(ys[-1])
        ys.append(ys[-1] + dg)
        x += v
    return np.array(xs[1:]), np.array(ys[1:])


def print_scenario_summary(results, key, internal_metabolites, reaction_names):
    """Print Omega/MDF/concentrations/per-reaction dG' for one scenario."""
    r = results[key]
    print(f"{key} — {r['label'].splitlines()[0]} / {r['label'].splitlines()[1]}")
    print(f"  dG_CAT = {r['dg_cat']:+.3f} kJ/mol")
    print(f"  Omega  = {r['omega']:+.4f} kJ/mol")
    print(f"  MDF    = {r.get('MDF', float('nan')):+.4f} kJ/mol")
    if "ln_x" in r:
        print("  optimal internal concentrations (mM):")
        for m in internal_metabolites:
            conc_mM = math.exp(r['ln_x'][m]) * REF_CONC_M * 1e3
            print(f"    {m:8s} {conc_mM:10.4g} mM")
    if "dg_prime_eff" in r:
        print("  per-reaction dG' at optimum (kJ/mol):")
        for n in reaction_names:
            print(f"    {n:6s} {r['dg_prime_eff'][n]:+8.3f}")


def plot_pathway_figure(results, reaction_names, flux_mult,
                         internal_metabolites, met_labels, suptitle, path):
    """
    Shared 3-panel figure for a pathway's 4 scenarios:
      (a) grouped bar — 2 boundary groups x [Omega, MDF free, MDF bound]
      (b) waterfall — cumulative dG' profile per scenario + dGm reference
          (every metabolite at the 1 mM reference, i.e. results['std_free']'s
          dg0_eff before LP optimization — this guarantees the dashed line
          ends at the same point as the std_free colored line, since the
          cumulative profile's endpoint always equals dG_CAT regardless of
          how internal concentrations are distributed along the way)
      (c) grouped bar — optimal internal concentrations, 4 bars/metabolite
    """
    fig, (ax_bar, ax_profile, ax_conc) = plt.subplots(1, 3, figsize=(17, 9))
    # fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    # ── (a) Omega / MDF(free) / MDF(bound), grouped by boundary type ────────
    groups = ["1 mM boundary", "Physiological\nboundary"]
    group_keys = [("std_free", "std_bound"), ("phys_free", "phys_bound")]
    x = np.arange(len(groups))
    w = 0.25
    omega_vals = [results[free]["omega"] for free, _ in group_keys]
    mdf_free   = [results[free].get("MDF", np.nan) for free, _ in group_keys]
    mdf_bound  = [results[bound].get("MDF", np.nan) for _, bound in group_keys]

    ax_bar.bar(x - w, omega_vals, w, label="Omega", color="#555555", alpha=0.85)
    ax_bar.bar(x,     mdf_free,   w, label="MDF (no bounds)",
               color=[results[free]["color"] for free, _ in group_keys], alpha=0.85)
    ax_bar.bar(x + w, mdf_bound,  w, label="MDF (physiol. bounds)",
               color=[results[bound]["color"] for _, bound in group_keys], alpha=0.85,
               hatch="//")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(groups)
    ax_bar.set_ylabel("Driving force (kJ mol$^{-1}$)")
    ax_bar.set_title("(a) Omega and MDF")
    ax_bar.axhline(0, color="black", linewidth=0.5)
    ax_bar.legend(frameon=False, fontsize=12, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=1)

    # ── (b) waterfall: dGm reference (all mets at 1 mM) + dG' profile per scenario ──
    dg0_dict = results["std_free"]["dg0_eff"]
    if not isinstance(dg0_dict, dict):
        dg0_dict = {n: v for n, v in zip(reaction_names, dg0_dict)}
    xs0, ys0 = cumulative_dg_profile(dg0_dict, reaction_names, flux_mult)
    ax_profile.plot(xs0, ys0, "k--", linewidth=1.2,
                    label="$\\Delta_r G'^{m}$ (1 mM reference)", zorder=6)
    for key in ["std_free", "std_bound", "phys_free", "phys_bound"]:
        r = results[key]
        if "dg_prime_eff" not in r:
            continue
        xs, ys = cumulative_dg_profile(r["dg_prime_eff"], reaction_names, flux_mult)
        ax_profile.plot(xs, ys, color=r["color"], linewidth=1.8,
                        label=r["label"].replace("\n", " "))
    mids = []
    xx = 0
    for v in flux_mult:
        mids.append(xx + v/2)
        xx += v
    ax_profile.set_xticks(mids)
    ax_profile.set_xticklabels(reaction_names, rotation=45, ha="right", fontsize=11)
    ax_profile.set_ylabel("Cumulative $\\Delta G'$ (kJ mol$^{-1}$)")
    ax_profile.set_title("(b) Thermodynamic waterfall")
    ax_profile.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax_profile.legend(frameon=False, fontsize=11, loc="upper center",
                       bbox_to_anchor=(0.5, -0.2), ncol=1)

    # ── (c) optimal internal concentrations, 4 bars/metabolite ──────────────
    keys_with_conc = [k for k in ["std_free", "std_bound", "phys_free", "phys_bound"]
                       if "ln_x" in results[k]]
    n_met = len(internal_metabolites)
    x_met = np.arange(n_met)
    n_bars = len(keys_with_conc)
    bar_w = 0.8 / n_bars
    for idx, key in enumerate(keys_with_conc):
        r = results[key]
        ln_x_arr = np.array([r["ln_x"].get(m, 0.0) for m in internal_metabolites])
        conc_mM = np.exp(ln_x_arr) * REF_CONC_M * 1e3
        offset = (idx - n_bars/2 + 0.5) * bar_w
        ax_conc.bar(x_met + offset, conc_mM, bar_w, color=r["color"], alpha=0.85,
                    label=r["label"].replace("\n", " "))
    ax_conc.set_yscale("log")
    ax_conc.set_xticks(x_met)
    ax_conc.set_xticklabels(met_labels, rotation=45, ha="right", fontsize=11)
    ax_conc.set_ylabel("Concentration (mM)")
    ax_conc.set_title("(c) Optimal internal concentrations")
    ax_conc.axhline(1e-6 * 1e3, color="gray", linewidth=0.6, linestyle=":",
                    label="Physiol. bounds (1 µM – 10 mM)")
    ax_conc.axhline(10,         color="gray", linewidth=0.6, linestyle=":")
    ax_conc.legend(frameon=False, fontsize=11, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=1)

    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


# ── TCA cycle (acetyl-CoA -> 2 CO2, OAA internal) ───────────────────────────
# formulas use KEGG-qualified compound IDs (see METABOLITE_KEGG above)
TCA_REACTIONS = [
    ("CS",   to_kegg_formula("oaa + accoa = cit + coa"),                  1),
    ("ACO",  to_kegg_formula("cit = icit"),                               1),
    ("ICDH", to_kegg_formula("icit + nad = akg + co2 + nadh"),            1),
    ("OGDH", to_kegg_formula("akg + nad + coa = succoa + co2 + nadh"),    1),
    ("SCS",  to_kegg_formula("succoa + adp + pi = succ + atp + coa"),     1),
    ("SDH",  to_kegg_formula("succ + fad = fum + fadh2"),                 1),
    ("FUM",  to_kegg_formula("fum = mal-L"),                              1),
    ("MDH",  to_kegg_formula("mal-L + nad = oaa + nadh"),                 1),
]
TCA_NAMES = [r[0] for r in TCA_REACTIONS]
TCA_FLUX  = np.array([r[2] for r in TCA_REACTIONS], dtype=float)

TCA_INTERNAL = ["oaa", "cit", "icit", "akg", "succoa", "succ", "fum", "mal-L"]
TCA_BOUNDARY = ["accoa", "coa", "co2", "atp", "adp", "pi",
                "nad", "nadh", "fad", "fadh2", "h2o", "h"]
_tca_int_idx = {m: i for i, m in enumerate(TCA_INTERNAL)}
_tca_bnd_idx = {m: i for i, m in enumerate(TCA_BOUNDARY)}

TCA_DG0 = {
    "CS": -31.4, "ACO": +6.3, "ICDH": -21.0, "OGDH": -33.6,
    "SCS": -17.6, "SDH": +12.0, "FUM": -3.4, "MDH": +29.7,
}
TCA_DG0_ARR = np.array([TCA_DG0[n] for n in TCA_NAMES])

# "standard" boundary: all net-conversion metabolites at 1 mM, uniformly
TCA_STD_BOUNDARY = {
    "accoa": 1e-3, "coa": 1e-3, "co2": 1e-3, "atp": 1e-3, "adp": 1e-3, "pi": 1e-3,
    "nad": 1e-3, "nadh": 1e-3, "fad": 1e-3, "fadh2": 1e-3,
    "h2o": 1.0, "h": 1e-7,
}

# physiological CO2: dissolved [CO2(aq)] in equilibrium with atmospheric CO2,
# via Henry's law: [CO2(aq)] = kH * P_CO2
#   kH    = 0.034 mol L^-1 atm^-1   (Henry's law solubility constant for CO2
#                                     in water at 25C; CRC/NIST tables)
#   P_CO2 = x_CO2 * P_total         (x_CO2 = atmospheric mole fraction,
#                                     ~420-425 ppm as of the mid-2020s;
#                                     P_total = 1 atm)
KH_CO2 = 0.034          # mol/(L*atm), Henry's law constant for CO2 in water at 25C
X_CO2_ATM = 425e-6      # atmospheric CO2 mole fraction (~425 ppm, mid-2020s)
P_TOTAL_ATM = 1.0       # atm
CO2_PHYS_CONC_M = KH_CO2 * X_CO2_ATM * P_TOTAL_ATM   # ~1.45e-5 M = ~14.5 uM

# physiological boundary concentrations — adjust accoa/coa to your own
# literature values if these placeholders don't match your sources
TCA_PHYS_BOUNDARY = {
    "accoa": 0.6e-3, "coa": 0.1e-3, "co2": CO2_PHYS_CONC_M,
    "atp": 3e-3, "adp": 5e-4, "pi": 1e-2,
    "nad": 1e-2, "nadh": 1e-3, "fad": 5e-4, "fadh2": 1e-4,
    "h2o": 1.0, "h": 1e-7,
}
TCA_PHYS_INT_BOUNDS = {m: (1e-6, 10e-3) for m in TCA_INTERNAL}


def build_tca_stoich():
    n_rxn = len(TCA_REACTIONS)
    S_int = np.zeros((n_rxn, len(TCA_INTERNAL)))
    S_bnd = np.zeros((n_rxn, len(TCA_BOUNDARY)))
    stoich = [
        ("CS",   [("oaa",-1),("accoa",-1),("cit",1),("coa",1)]),
        ("ACO",  [("cit",-1),("icit",1)]),
        ("ICDH", [("icit",-1),("nad",-1),("akg",1),("co2",1),("nadh",1)]),
        ("OGDH", [("akg",-1),("nad",-1),("coa",-1),("succoa",1),("co2",1),("nadh",1)]),
        ("SCS",  [("succoa",-1),("adp",-1),("pi",-1),("succ",1),("atp",1),("coa",1)]),
        ("SDH",  [("succ",-1),("fad",-1),("fum",1),("fadh2",1)]),
        ("FUM",  [("fum",-1),("mal-L",1)]),
        ("MDH",  [("mal-L",-1),("nad",-1),("oaa",1),("nadh",1)]),
    ]
    for rxn, entries in stoich:
        i = TCA_NAMES.index(rxn)
        for met, coeff in entries:
            if met in _tca_int_idx:
                S_int[i, _tca_int_idx[met]] = coeff
            elif met in _tca_bnd_idx:
                S_bnd[i, _tca_bnd_idx[met]] = coeff
    return S_int, S_bnd


#%% Cell 2 — RUN: solve all 4 scenarios for EMP and for TCA
# Rerun this cell whenever you change a boundary/bound definition above, then
# re-run individual analysis/plot cells.

print("\n== Running EMP scenarios ==")
S_int, S_bnd = build_stoich_matrices()
dg0_raw, _ = get_dg0_prime_literature()
dgm_raw = dg0_to_dgm(dg0_raw, S_int, S_bnd, BOUNDARY_METABOLITES)
# To fetch dGm directly from eQuilibrator instead of converting the literature
# dG0' values, swap the line above for:
#   dgm_raw, _ = fetch_dgm_equilibrator(REACTIONS)

emp_results = run_four_scenarios(
    dgm_raw, S_bnd, S_int, REACTION_NAMES, FLUX_MULTIPLIERS,
    BOUNDARY_METABOLITES, INTERNAL_METABOLITES, _met_idx_internal,
    EMP_STD_BOUNDARY, EMP_PHYS_BOUNDARY, EMP_PHYS_INT_BOUNDS,
    v_nc=V_NC, model_prefix="EMP",
)
for k, r in emp_results.items():
    mdf_str = f"{r['MDF']:+.4f}" if "MDF" in r else "  N/A  "
    print(f"  {k:11s} Omega = {r['omega']:+.4f}  MDF = {mdf_str}  "
          f"Omega-MDF = {r['omega'] - r.get('MDF', r['omega']):+.2e}  kJ/mol")

print("\n== Running TCA scenarios ==")
tca_S_int, tca_S_bnd = build_tca_stoich()
dgm_tca = dg0_to_dgm(TCA_DG0_ARR, tca_S_int, tca_S_bnd, TCA_BOUNDARY)

tca_results = run_four_scenarios(
    dgm_tca, tca_S_bnd, tca_S_int, TCA_NAMES, TCA_FLUX,
    TCA_BOUNDARY, TCA_INTERNAL, _tca_int_idx,
    TCA_STD_BOUNDARY, TCA_PHYS_BOUNDARY, TCA_PHYS_INT_BOUNDS,
    v_nc=V_NC, model_prefix="TCA",
)
for k, r in tca_results.items():
    mdf_str = f"{r['MDF']:+.4f}" if "MDF" in r else "  N/A  "
    print(f"  {k:11s} Omega = {r['omega']:+.4f}  MDF = {mdf_str}  "
          f"Omega-MDF = {r['omega'] - r.get('MDF', r['omega']):+.2e}  kJ/mol")


#%% Cell 3 — analysis: EMP std_free (1 mM boundary, no internal bounds)
print_scenario_summary(emp_results, "std_free", INTERNAL_METABOLITES, REACTION_NAMES)

#%% Cell 4 — analysis: EMP std_bound (1 mM boundary, physiological internal bounds)
print_scenario_summary(emp_results, "std_bound", INTERNAL_METABOLITES, REACTION_NAMES)

#%% Cell 5 — analysis: EMP phys_free (physiological boundary, no internal bounds)
print_scenario_summary(emp_results, "phys_free", INTERNAL_METABOLITES, REACTION_NAMES)

#%% Cell 6 — analysis: EMP phys_bound (physiological boundary, physiological internal bounds)
print_scenario_summary(emp_results, "phys_bound", INTERNAL_METABOLITES, REACTION_NAMES)

#%% Cell 7 — analysis: TCA std_free (1 mM-ish boundary, no internal bounds)
print_scenario_summary(tca_results, "std_free", TCA_INTERNAL, TCA_NAMES)

#%% Cell 8 — analysis: TCA std_bound (1 mM-ish boundary, physiological internal bounds)
print_scenario_summary(tca_results, "std_bound", TCA_INTERNAL, TCA_NAMES)

#%% Cell 9 — analysis: TCA phys_free (physiological boundary, no internal bounds)
print_scenario_summary(tca_results, "phys_free", TCA_INTERNAL, TCA_NAMES)

#%% Cell 10 — analysis: TCA phys_bound (physiological boundary, physiological internal bounds)
print_scenario_summary(tca_results, "phys_bound", TCA_INTERNAL, TCA_NAMES)


#%% Cell 11 — plot: Figure 1, EMP (3 panels)



met_labels_emp = ["G6P","F6P","FBP","DHAP","G3P","1,3BPG","3PG","2PG","PEP","Pyr","NAD","NADH"]
fig1 = plot_pathway_figure(
    emp_results, REACTION_NAMES, FLUX_MULTIPLIERS,
    INTERNAL_METABOLITES, met_labels_emp,
    "EMP glycolysis -> lactate: MDF vs Omega across scenarios", 
    path=r"EMP_MDF.png"
)


#%% Cell 12 — plot: Figure 2, TCA cycle (3 panels)
met_labels_tca = ["OAA","Cit","ICit","aKG","Suc-CoA","Succ","Fum","Mal"]
fig2 = plot_pathway_figure(
    tca_results, TCA_NAMES, TCA_FLUX,
    TCA_INTERNAL, met_labels_tca,
    "TCA cycle (acetyl-CoA -> 2 CO2): MDF vs Omega across scenarios",
    path=r"TCA_MDF.png"
)


#%% Cell 13 — plot: Figure 3, Omega vs MDF scatter (all scenarios, both pathways)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.set_title("Omega vs MDF — all scenarios and pathways", fontsize=15)

all_omega, all_mdf, all_colors, all_labels, all_markers = [], [], [], [], []
for k, r in emp_results.items():
    if "MDF" in r:
        all_omega.append(r["omega"])
        all_mdf.append(r["MDF"])
        all_colors.append(r["color"])
        all_labels.append(f"EMP {r['label'].replace(chr(10), ' ')}")
        all_markers.append("o")
for k, r in tca_results.items():
    if "MDF" in r:
        all_omega.append(r["omega"])
        all_mdf.append(r["MDF"])
        all_colors.append(r["color"])
        all_labels.append(f"TCA {r['label'].replace(chr(10), ' ')}")
        all_markers.append("^")

for om, mdf, col, lab, mk in zip(all_omega, all_mdf, all_colors, all_labels, all_markers):
    ax.scatter(om, mdf, color=col, s=80, zorder=5, label=lab, marker=mk,
               edgecolor="black", linewidth=0.4)

lim_min = min(all_omega + all_mdf) * 1.1 if min(all_omega + all_mdf) < 0 else min(all_omega + all_mdf) * 0.9
lim_max = max(all_omega + all_mdf) * 1.1
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="Omega = MDF")
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel("Omega (kJ mol$^{-1}$)")
ax.set_ylabel("MDF (kJ mol$^{-1}$)")
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
ax.set_aspect("equal")

fig.tight_layout()

plt.savefig(r"MDF_omega_compare.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
