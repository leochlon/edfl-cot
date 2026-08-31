"""Pure-math core for EDFL accounting on autoregressive (CoT) generation.

Implements the machinery of "Thinking Doesn't Make It So: EDFL for
autoregressive generation" (the *child* paper) and deliberately does **not**
implement the parts of ``trace_budgets.py`` that the child paper retires.

Every public function here corresponds to a numbered result in the paper and is
checked in ``verify_cot_budget.py``:

  Lemma 1  (readout decomposition)   -> :func:`readout_split`, :func:`renormalize`
  Lemma 2  (drift/dispersion split)  -> :func:`drift_dispersion`
  Thm 3    (order-marginal decomp.)  -> :func:`omd_slack`, :func:`order_marginal`
  Thm 4    (rare-set decompression)  -> :func:`kl_rare_exact`, :func:`rare_floor`
  Thm 5    (commitment projection)   -> :func:`required_budget`, :func:`i_projection`
  Thm 6    (autoregressive decomp.)  -> :func:`prefix_updated_chain`, :func:`answer_pushforward`
  Thm 7    (twin-trace detection)    -> :func:`twin_trace_reject`
  App. B   (gate under ablated anchor) -> :func:`gate`, :func:`charge_headroom`

Conventions
-----------
* All divergences are in **nats**.
* The readout is always the *renormalized conditional* ``beta`` over the K
  candidate answer cells, never the raw next-token distribution: the raw
  distribution carries the answer-readiness coordinate ``d`` additively into
  the budget (Lemma 1).
* Renormalize per serialization, then average. Never average, then
  renormalize: ``E_G[P^G_a / d^G] != E_G[P^G_a] / E_G[d^G]`` unless ``d`` is
  a.s. constant.
* The decision variable is the **margin** ``M = A - R - C`` in nats. ``ISR``
  is retained for telemetry only; a ratio cannot distinguish 0.5/0.4 from
  5.0/4.9.
* No clipping constant appears anywhere. Under the ablated anchor the
  available budget is bounded by the measured quantity ``log(1/c)``, and
  clipping is not merely unnecessary: it destroys the Lemma 2 identity.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "EPS",
    "kl_bernoulli",
    "kl_categorical",
    "entropy",
    "logit",
    "renormalize",
    "readout_split",
    "drift_dispersion",
    "order_marginal",
    "omd_slack",
    "mutual_information_ag",
    "j_mu",
    "compensation_residual",
    "kl_rare_exact",
    "rare_floor",
    "required_budget",
    "i_projection",
    "available_budget",
    "charge_headroom",
    "p_star_effective",
    "p_star_effective_linear",
    "calibration_charge",
    "rho_penalty",
    "rho_saturating",
    "rho_charge_from_displacement",
    "n_max",
    "rho_feasible",
    "eb_lower",
    "eb_upper",
    "hoeffding_lower",
    "hoeffding_upper",
    "wsr_bounds",
    "conf_lower",
    "conf_upper",
    "split_half_displacement",
    "naive_squared_displacement",
    "renorm_order_gap",
    "per_step_headroom",
    "answer_pushforward",
    "prefix_updated_chain",
    "prior_weighted_chain",
    "twin_trace_reject",
    "paired_difference_ci",
    "AnchorReport",
    "anchor_from_family",
    "family_decision_invariant",
    "budget_capacity",
    "implied_serialization_sd",
    "evidence_gate",
    "tau_cat",
    "stability_cap",
    "decision_frozen",
    "monitor_charge",
    "anchor_pvalue",
    "wsr_sequence",
    "sequential_decision",
    "battery_plan",
    "Charges",
    "GateResult",
    "gate",
]

EPS = 1e-12


# --------------------------------------------------------------------------
# Divergences and elementary transforms
# --------------------------------------------------------------------------


def _clip(p: float, eps: float = EPS) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def kl_bernoulli(a: float, b: float) -> float:
    """KL(Ber(a) || Ber(b)) in nats."""
    a = _clip(a)
    b = _clip(b)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def kl_categorical(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in nats, over the option simplex."""
    if len(p) != len(q):
        raise ValueError("kl_categorical: length mismatch")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        total += pi * math.log(max(pi, EPS) / max(qi, EPS))
    return float(total)


def entropy(p: Sequence[float]) -> float:
    """Shannon entropy in nats."""
    return float(-sum(pi * math.log(max(pi, EPS)) for pi in p if pi > 0.0))


def logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


# --------------------------------------------------------------------------
# Lemma 1: readout decomposition.  The instrument reads beta, not P.
# --------------------------------------------------------------------------


def renormalize(full: Sequence[float], cells: Sequence[int]) -> Tuple[float, List[float]]:
    """Split a next-token distribution into ``(d, beta)``.

    ``d`` is the native candidate mass at the scored position: a validity
    condition on the instrument, never a quantity to charge. ``beta`` is the
    renormalized conditional on the K cells, the only sigma(W)-measurable
    coordinate and so the only thing the law may be applied to.
    """
    d = float(sum(full[i] for i in cells))
    if d <= 0.0:
        raise ValueError("renormalize: zero candidate mass at the scored position")
    return d, [float(full[i]) / d for i in cells]


def readout_split(
    full_p: Sequence[float], full_q: Sequence[float], cells: Sequence[int]
) -> Dict[str, float]:
    """Exact KL chain rule over {candidates, complement} (Lemma 1).

    ``KL_cat = KL(Ber(d)||Ber(d0)) + d * KL_cond(beta||beta0)``.
    Only ``belief`` is measurable in the answer variable; ``readiness`` is a
    frame quantity. Charging ``KL_cat`` is the full-simplex confound.
    """
    d, beta = renormalize(full_p, cells)
    d0, beta0 = renormalize(full_q, cells)
    readiness = kl_bernoulli(d, d0)
    belief = kl_categorical(beta, beta0)
    cell_set = set(cells)
    comp_p = [full_p[i] for i in range(len(full_p)) if i not in cell_set]
    comp_q = [full_q[i] for i in range(len(full_q)) if i not in cell_set]
    comp = 0.0
    if comp_p and sum(comp_p) > 0.0:
        sp, sq = sum(comp_p), sum(comp_q)
        comp = kl_categorical([x / sp for x in comp_p], [x / sq for x in comp_q])
    kl_cat = readiness + d * belief + (1.0 - d) * comp
    return {
        "d": d,
        "d0": d0,
        "readiness": readiness,
        "belief": belief,
        "kl_cat": kl_cat,
        "charged": d * belief,
    }


# --------------------------------------------------------------------------
# Lemma 2: drift/dispersion split.  Charge the mean; dispersion is free.
# --------------------------------------------------------------------------


def drift_dispersion(qs: Sequence[Sequence[float]], q0: Sequence[float]) -> Dict[str, float]:
    """Exact Bregman split of mean movement against a **frozen** anchor.

    ``mean_t KL(q_t||q0) = mean_t KL(q_t||qbar) + KL(qbar||q0)``.

    The identity is algebraic and presumes nothing, but it is *not preserved
    by clipping*, so the split must be taken on unclipped terms even where a
    downstream budget is clipped. Never accumulate ``sum_t KL(q_t||q_{t-1})``
    as a charge: it is unbounded for a wandering martingale on a path whose
    endpoint divergence stays bounded.
    """
    qbar = order_marginal(qs)
    mean_total = sum(kl_categorical(q, q0) for q in qs) / len(qs)
    dispersion = sum(kl_categorical(q, qbar) for q in qs) / len(qs)
    drift = kl_categorical(qbar, q0)
    return {
        "mean_total": mean_total,
        "dispersion": dispersion,
        "drift": drift,
        "residual": mean_total - dispersion - drift,
        "qbar": qbar,
    }


# --------------------------------------------------------------------------
# Theorem 3: the order-marginal is the reference.
# --------------------------------------------------------------------------


def order_marginal(
    qs: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None
) -> List[float]:
    """``b = E_G[q^G]``: the mean of *already renormalized* readouts."""
    if not qs:
        raise ValueError("order_marginal: empty readout family")
    k = len(qs[0])
    if weights is None:
        w = [1.0 / len(qs)] * len(qs)
    else:
        s = float(sum(weights))
        w = [float(x) / s for x in weights]
    return [float(sum(wi * q[a] for wi, q in zip(w, qs))) for a in range(k)]


def omd_slack(q_target: Sequence[float], qs: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Slack in ``E_G[KL(Q||q^G)] >= KL(Q||b)`` (Theorem 3), and its identity.

    The slack equals ``sum_a Q_a (log b_a - E_G[log q^G_a]) >= 0``: it is
    realization dispersion across serializations, not information about the
    answer. It sits on the **available** side, so computing a budget as a mean
    of per-serialization divergences inflates it and makes the gate permissive.
    """
    b = order_marginal(qs)
    mean_kl = sum(kl_categorical(q_target, q) for q in qs) / len(qs)
    at_mean = kl_categorical(q_target, b)
    identity = sum(
        q_target[a] * (math.log(max(b[a], EPS)) - sum(math.log(max(q[a], EPS)) for q in qs) / len(qs))
        for a in range(len(b))
    )
    return {"mean_kl": mean_kl, "at_mean": at_mean, "slack": mean_kl - at_mean, "identity": identity}


def mutual_information_ag(qs: Sequence[Sequence[float]]) -> float:
    """``I(A;G) = E_G[KL(q^G||b)] = H(b) - E_G[H(q^G)]``, bounded by ``log K``.

    The order term: how much a *single* serialization over-states its own
    resolution, independently of the anchor. Charged at both contexts, since an
    order-sensitive baseline otherwise leaves an uncharged residual.
    """
    b = order_marginal(qs)
    return float(entropy(b) - sum(entropy(q) for q in qs) / len(qs))


def j_mu(qs: Sequence[Sequence[float]]) -> float:
    """``J^mu = E_G[KL(b||q^G)]``: the *reverse* arm, and unbounded."""
    b = order_marginal(qs)
    return float(sum(kl_categorical(b, q) for q in qs) / len(qs))


def compensation_residual(qs: Sequence[Sequence[float]], r: Sequence[float]) -> float:
    """Residual of ``E_G[KL(q^G||r)] = KL(b||r) + I(A;G)``; zero for every r."""
    b = order_marginal(qs)
    lhs = sum(kl_categorical(q, r) for q in qs) / len(qs)
    return float(lhs - kl_categorical(b, r) - mutual_information_ag(qs))


# --------------------------------------------------------------------------
# Theorem 4: rare-set decompression, exactly.
# --------------------------------------------------------------------------


def kl_rare_exact(eps_mass: float, b_a: float) -> float:
    """``KL(Ber(1-eps) || Ber(b(A)))``: the exact commitment surprisal."""
    return kl_bernoulli(1.0 - eps_mass, b_a)


def rare_floor(eps_mass: float, b_a: float, *, valid: bool = True) -> float:
    """Lower bound on the commitment cost.

    ``valid=True`` returns the corrected floor ``(1-eps)log(1/b) - H(eps)``,
    tight to ``eps*b``. ``valid=False`` returns the parent's one-sided clause
    ``(1-eps)log(1/b)``, which does **not** hold.
    """
    b_a = _clip(b_a)
    e = _clip(eps_mass)
    base = (1.0 - e) * math.log(1.0 / b_a)
    if not valid:
        return float(base)
    h = -e * math.log(e) - (1.0 - e) * math.log(1.0 - e)
    return float(base - h)


# --------------------------------------------------------------------------
# Theorem 5: commitment is an I-projection, and its cost is exact.
# --------------------------------------------------------------------------


def required_budget(p_star: float, c: float) -> float:
    """``B2T = KL(Ber(p*) || Ber(c))``.

    Not a lossy binary reduction: it is the exact cost of the I-projection of
    the reference onto ``{Q : Q_a = p*}``. Both terms the Bernoulli step appears
    to discard vanish at the minimiser.
    """
    return kl_bernoulli(p_star, c)


def i_projection(b: Sequence[float], a: int, p_star: float) -> List[float]:
    """The minimiser ``Q_a = p*``, ``Q_c = (1-p*) b_c/(1-b_a)`` for ``c != a``."""
    b_a = _clip(b[a])
    rest = 1.0 - b_a
    return [
        p_star if i == a else (1.0 - p_star) * b[i] / max(rest, EPS) for i in range(len(b))
    ]


# --------------------------------------------------------------------------
# The gate (Appendix B).  Margin, hinge, headroom, effective target.
# --------------------------------------------------------------------------


def available_budget(b_lo_a: float, c: float) -> float:
    """``A = 1{b_lo >= c} * KL(Ber(b_lo)||Ber(c))``.

    The indicator is load-bearing. Without it, movement *away* from the
    committed cell is charged as budget *for* it, and the firing set gains a
    spurious component ``[0, x-]`` whenever ``log(1/(1-c)) >= R``.
    """
    if b_lo_a < c:
        return 0.0
    return kl_bernoulli(b_lo_a, c)


def charge_headroom(p_star: float, c: float) -> float:
    """``H0 = log(1/c) - KL(Ber(p*)||Ber(c)) = (1-p*) logit(1-c) + H(p*)``.

    The total nats available for any correction before the gate becomes
    unsatisfiable. The gate is satisfiable at all iff ``C < H0``.
    """
    return float(math.log(1.0 / _clip(c)) - required_budget(p_star, c))


def p_star_effective(p_star: float, c: float, charges: float, *, tol: float = 1e-14) -> float:
    """Exact effective target: the ``x`` in ``[c,1)`` with ``A(x) = R + C``.

    ``x -> KL(Ber(x)||Ber(c))`` is strictly increasing on ``[c,1)`` with
    supremum ``log(1/c)``, so the gate is exactly a threshold on ``b_lo``, and
    the threshold exists iff ``charges < charge_headroom(p*, c)``.
    """
    target = required_budget(p_star, c) + float(charges)
    if target >= math.log(1.0 / _clip(c)):
        return float("nan")  # infeasible: no belief can pay the charges
    lo, hi = _clip(c), 1.0 - 1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mid, c) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


def p_star_effective_linear(p_star: float, c: float, charges: float) -> float:
    """First-order form ``p* + C/(logit p* - logit c) + O(C^2)``."""
    return float(p_star + charges / (logit(p_star) - logit(c)))


def calibration_charge(p_star: float, c: float, err: float) -> float:
    """Price of a downward calibration error ``e``, which transfers 1:1.

    ``KL(Ber(p*+e)||Ber(c)) - KL(Ber(p*)||Ber(c))``; infeasible once
    ``e >= 1 - p*``.
    """
    if err <= 0.0:
        return 0.0
    if p_star + err >= 1.0 - 1e-12:
        return float("inf")
    return float(required_budget(p_star + err, c) - required_budget(p_star, c))


def rho_penalty(
    kappa: float, beta: float, n_tokens: int, *, ceiling: Optional[float] = None
) -> float:
    """Sub-orbit displacement penalty, capped power-law form.

    A causal probe at step k cannot permute what it has emitted, so it samples a
    strict sub-orbit and returns ``b_k + rho_k``.

    ``kappa`` and ``beta`` have no defaults and must be measured per deployment;
    the current battery fits 0.320/0.327/0.348 at 0.5B/3B/7B and the paper's
    model comparison rejects the power law in favour of a saturating form. Pass
    ``ceiling`` once a saturating amplitude is measured: an uncapped power law
    charges without limit for a bounded quantity.
    """
    if n_tokens <= 0:
        return 0.0
    v = float(kappa) * (float(n_tokens) ** float(beta))
    return float(v if ceiling is None else min(v, float(ceiling)))


def rho_saturating(amplitude: float, l_half: float, n_tokens: int) -> float:
    """``pen_rho(N) = A * N / (N + L0)``: the form the L-battery actually supports.

    Bounded by ``amplitude`` for every N, which is what makes length stop being
    the binding variable. Fit on the split-half first-moment displacement, never
    on a spread ratio: a spread is a second moment and Lemma 2 exempts it.
    """
    if n_tokens <= 0:
        return 0.0
    return float(amplitude) * float(n_tokens) / (float(n_tokens) + float(l_half))


def rho_charge_from_displacement(rho_a: float, b_a: float) -> float:
    """Exact nats a committed-cell displacement costs: ``log(b_a / (b_a - rho_a))``.

    The linear form ``rho_a / b_a`` under-states -- by 2.4% at ``b_a = .95`` and
    by 28% at ``b_a = .02`` -- and the gate is consulted on small-mass cells.
    """
    b_a = _clip(b_a)
    if rho_a <= 0.0:
        return 0.0
    if rho_a >= b_a:
        return float("inf")
    return float(math.log(b_a / (b_a - float(rho_a))))


def n_max(headroom: float, cal_pen: float, kappa: float, beta: float,
          *, ceiling: Optional[float] = None) -> float:
    """``N_max = ((H0 - pen_cal)/kappa)^(1/beta)`` (uncapped power law only).

    Deprecated as a deployment quantity. It is finite only because the power law
    is unbounded, and the measured displacement is not: under a saturating fit
    the answer is not a length but a yes/no, which is :func:`rho_feasible`. Use
    this only to reproduce the older sweep.
    """
    slack = headroom - cal_pen
    if slack <= 0.0 or kappa <= 0.0:
        return 0.0
    if ceiling is not None and float(ceiling) < slack:
        return float("inf")  # the charge saturates below the headroom: any N certifies
    return float((slack / kappa) ** (1.0 / beta))


def rho_feasible(headroom: float, other_charges: float, ceiling: float) -> Dict[str, object]:
    """Length is not the binding variable when the displacement charge is bounded.

    Returns the verdict and the slack. ``certifiable`` is a property of the
    (p*, anchor, charges) triple alone; if it holds, it holds at every N.
    """
    slack = float(headroom) - float(other_charges) - float(ceiling)
    return {
        "headroom": float(headroom),
        "other_charges": float(other_charges),
        "rho_ceiling": float(ceiling),
        "slack_nats": float(slack),
        "certifiable": bool(slack > 0.0),
        "n_max": (float("inf") if slack > 0.0 else 0.0),
    }


# --------------------------------------------------------------------------
# The ablation family: a conservative anchor plus a within-family validity test
# --------------------------------------------------------------------------


def paired_difference_ci(
    xs: Sequence[float], ys: Sequence[float], alpha: float = 0.05
) -> Tuple[float, float]:
    """Two-sided interval for ``E[x] - E[y]`` on paired draws in [0,1].

    Comparing two one-sided bounds is not a test of a difference: each carries
    its own width, and the comparison inherits both. When the two ablations are
    probed on the same serializations the draws are paired, the per-serialization
    difference is the statistic, and the interval is on that.
    """
    if len(xs) != len(ys):
        raise ValueError("paired_difference_ci: unpaired samples")
    d = [(float(a) - float(b) + 1.0) / 2.0 for a, b in zip(xs, ys)]  # map to [0,1]
    lo, hi = wsr_bounds(d, alpha)
    return float(2.0 * lo - 1.0), float(2.0 * hi - 1.0)


@dataclass
class AnchorReport:
    """Outcome of running a pre-registered family of content-destroying ablations."""

    anchor: float                       # the conservative (largest) member
    chosen: str
    per_member: Dict[str, float] = field(default_factory=dict)
    spread: float = 0.0                 # max - min across the family
    within_family_ok: bool = True
    disagreements: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "anchor": self.anchor,
            "chosen": self.chosen,
            "per_member": dict(self.per_member),
            "spread": self.spread,
            "within_family_ok": self.within_family_ok,
            "disagreements": list(self.disagreements),
        }


def anchor_from_family(
    members: Dict[str, Sequence[float]], *, alpha: float = 0.05, tau: float = 0.05,
    exact: bool = False,
) -> AnchorReport:
    """Anchor over a family of ablations that are all meant to destroy content.

    Two things happen here that a placeholder-vs-scramble agreement check does
    not do:

    * The anchor is the **largest** member, not one designated member. Where the
      gate would answer (``b_lo >= p*``) the margin is decreasing in ``c``, so
      the maximum is the conservative choice and the guarantee becomes "would
      have answered under every ablation in the family".
    * The agreement test runs **within** the family. Members that are all
      content-destroying should agree; if they do not, the anchor is responding
      to something other than evidence content and the accounting is void. A
      redaction and a word-scramble are *not* members of one family unless the
      scramble actually destroys the claim-relevant content, which for a
      lexically cued task it does not.
    """
    per = {
        name: (float(sum(v) / len(v)) if exact else conf_upper(list(v), alpha))
        for name, v in members.items()
    }
    if not per:
        raise ValueError("anchor_from_family: empty family")
    chosen = max(per, key=lambda k: per[k])
    lo_name = min(per, key=lambda k: per[k])
    spread = per[chosen] - per[lo_name]
    bad = [
        f"{a}-{b}={per[a] - per[b]:+.4f}"
        for a, b in itertools.combinations(sorted(per), 2)
        if abs(per[a] - per[b]) > tau
    ]
    return AnchorReport(anchor=per[chosen], chosen=chosen, per_member=per,
                        spread=float(spread), within_family_ok=not bad,
                        disagreements=bad)


def family_decision_invariant(
    members: Dict[str, Sequence[float]], *, b_lo_a: float, p_star: float,
    charges: "Charges | float" = 0.0, alpha: float = 0.05, exact: bool = False,
) -> Dict[str, object]:
    """Does the ablation family's disagreement change the decision?

    A tolerance in probability is the wrong statistic. The margin's sensitivity
    to the anchor is ``dM/dc = (p* - b_lo)/(c(1-c))``, which is *small* exactly
    where the gate is consulted -- ``b_lo`` close to ``p*``. A donor spread that
    looks alarming in probability can be worth a hundredth of a nat.

    Since ``M`` is monotone decreasing in ``c``, evaluating at the smallest and
    largest member brackets the whole family. Veto only when they disagree: the
    band then straddles the decision boundary and the anchor genuinely matters.
    Otherwise the decision is invariant across every member and the spread is
    not a reason to withhold one.
    """
    per = {
        name: (float(sum(v) / len(v)) if exact else conf_upper(list(v), alpha))
        for name, v in members.items()
    }
    c_lo, c_hi = min(per.values()), max(per.values())
    g_lo = gate(cell=0, b_lo_a=b_lo_a, anchor_c=c_lo, p_star=p_star, charges=charges)
    g_hi = gate(cell=0, b_lo_a=b_lo_a, anchor_c=c_hi, p_star=p_star, charges=charges)
    return {
        "per_member": per,
        "c_min": float(c_lo),
        "c_max": float(c_hi),
        "spread_probability": float(c_hi - c_lo),
        "spread_nats": float(abs(g_lo.margin - g_hi.margin)),
        "sensitivity": float((p_star - b_lo_a) / max(c_hi * (1.0 - c_hi), EPS)),
        "decision_invariant": bool(g_lo.answered == g_hi.answered),
        "answered": bool(g_hi.answered),          # the conservative member decides
        "margin": float(g_hi.margin),
    }


def budget_capacity(anchor_c: float, p_star: float, charges: float) -> Dict[str, float]:
    """What the ablation contrast can supply at all, before any evidence.

    ``A = KL(Ber(b)||Ber(c))`` is bounded above by ``log(1/c)`` at ``b = 1``, so
    the entire information capacity of the contrast is fixed by the anchor. Two
    ratios matter and neither depends on the evidence:

      ``charge_share``   = C / log(1/c)   -- fraction of capacity the method eats
      ``target_share``   = R / log(1/c)   -- fraction the target eats

    If they sum to 1 or more, no belief whatsoever certifies, and lowering
    ``p*`` cannot rescue it because ``A`` is capped independently of ``p*``.
    """
    cap = math.log(1.0 / _clip(anchor_c))
    req = required_budget(p_star, anchor_c)
    return {
        "capacity": float(cap),
        "required": float(req),
        "charges": float(charges),
        "headroom": float(cap - req),
        "charge_share": float(charges / cap) if cap > EPS else float("inf"),
        "target_share": float(req / cap) if cap > EPS else float("inf"),
        "max_achievable_margin": float(cap - req - charges),
        "certifiable_at_any_belief": bool(cap - req - charges > 0.0),
    }


def implied_serialization_sd(i_ag: float, anchor_c: float) -> float:
    """Back the readout spread out of an order term, for a binary cell.

    ``H`` is concave with ``H''(x) = -1/(x(1-x))``, so to second order
    ``I(A;G) = H(c) - E[H(q)] ~ Var(q) / (2 c (1-c))``. A large order charge is
    a statement about instrument stability: the remedy is a more stable scoring
    position, not a bigger payment.
    """
    return float(math.sqrt(max(0.0, 2.0 * i_ag * anchor_c * (1.0 - anchor_c))))


def evidence_gate(
    *, b_lo_a: float, anchor_c: float, charges: "Charges | float" = 0.0,
    delta_nats: float = 0.5,
) -> Dict[str, object]:
    """The guarantee that survives a high anchor -- and what it costs you.

    Certifies ``A - C >= delta``: the evidence supplied at least ``delta`` nats
    of decompression toward the committed cell, net of charges, at confidence
    ``1-alpha`` over serialization sampling. There is no ``R``, so there is no
    headroom constraint and no ``p*``.

    This is a *dependence* guarantee, not a *reliability* one. It
    does not say the answer is right with probability ``p*``, and it must not be
    reported as if it did. It is the right instrument when the anchor is high --
    when the model already believes the answer and the question is whether the
    retrieved evidence did any work -- and the wrong one when a calibrated error
    rate is what the deployment needs.
    """
    ch = charges if isinstance(charges, Charges) else Charges(other=float(charges))
    a = available_budget(b_lo_a, anchor_c)
    contribution = a - ch.total
    cap = math.log(1.0 / _clip(anchor_c))
    return {
        "available": float(a),
        "charges": float(ch.total),
        "contribution": float(contribution),
        "delta": float(delta_nats),
        "max_delta_at_this_anchor": float(cap - ch.total),
        "certified": bool(contribution >= delta_nats),
        "guarantee": "evidence-dependence, not calibrated reliability",
    }


# --------------------------------------------------------------------------
# The drift-monitor accounting (companion paper): anchored self-reference
# --------------------------------------------------------------------------


def tau_cat(q0: Sequence[float] | float, q_prior: Sequence[float] | float) -> float:
    """``tau_cat = sqrt(Var_{q0}[log(q0/q_prior)])``, the q0-centred dual norm.

    Displacements are simplex-tangent, so the log-ratio enters *centred*. On
    two cells this reduces exactly to ``|logit q0 - logit q_prior| *
    sqrt(q0(1-q0))``, so that ``tau_cat * sqrt(2 delta)`` is the familiar
    product. Collapsing a K-cell readout onto its modal predicate understates
    it whenever the log-ratio is not constant on each side of the partition.
    """
    if isinstance(q0, (int, float)):
        a, b = float(q0), float(q_prior)  # type: ignore[arg-type]
        return abs(logit(a) - logit(b)) * math.sqrt(a * (1.0 - a))
    lr = [math.log(max(x, EPS) / max(y, EPS)) for x, y in zip(q0, q_prior)]  # type: ignore[arg-type]
    mean = sum(p * l for p, l in zip(q0, lr))
    return float(math.sqrt(sum(p * (l - mean) ** 2 for p, l in zip(q0, lr))))


def stability_cap(
    q0: Sequence[float] | float, q_prior: Sequence[float] | float,
    delta_lic: float, rho_norm: float = 0.0,
) -> float:
    """``|Delta_t - Delta_0| <= delta + tau_cat (sqrt(2 delta) + sup ||rho||)``.

    This is where the sub-orbit displacement belongs: multiplied by the lever
    arm, inside a *cap on gate inflation*, not subtracted from the budget as an
    additive charge. ``q0`` is measured with nothing pinned and ``bar q_t`` with
    a trace pinned, so the displacements do not cancel. The naive cap that
    omits the cross term is false.
    """
    t = tau_cat(q0, q_prior)
    return float(delta_lic + t * (math.sqrt(2.0 * delta_lic) + float(rho_norm)))


def decision_frozen(
    *, q0_a: float, q_prior_a: float, p_star: float, delta_lic: float,
    rho: float = 0.0,
) -> Dict[str, object]:
    """Theorem 3(3): can any in-ball movement flip the answer/abstain decision?

    If the anchor's margin to the gate boundary exceeds the cap, no continuation
    can cross it, and at ``delta_lic = 0`` the decision is frozen outright. This
    is a usable verdict even when no absolute-reliability certificate is
    available: the answer is decided before the first token is generated.
    """
    b2t = required_budget(p_star, q_prior_a)
    observed = kl_bernoulli(q0_a, q_prior_a)
    margin = observed - b2t
    rho_norm = abs(rho) / math.sqrt(max(q0_a * (1.0 - q0_a), EPS))
    cap = stability_cap(q0_a, q_prior_a, delta_lic, rho_norm)
    return {
        "b2t": float(b2t),
        "observed": float(observed),
        "isr_0": float(observed / b2t) if b2t > EPS else float("inf"),
        "margin_nats": float(margin),
        "cap_nats": float(cap),
        "frozen": bool(abs(margin) > cap),
        "decision": "answer" if margin >= 0 else "abstain",
        "isr_inflation": float(cap / b2t) if b2t > EPS else float("inf"),
        "rho_norm": float(rho_norm),
        "original_cap_recovered": bool(rho_norm <= math.sqrt(2.0 * delta_lic)),
    }


def monitor_charge(
    *, d_t: float, b_bar: Sequence[float], b_0: Sequence[float], delta_lic: float,
    weight_by_d: bool = False,
) -> float:
    """``charge_t = max(0, [d_t *] KL_cond(b_bar || b_0) - delta_lic)``.

    ``b_0`` is the model's own pre-generation anchor, not an ablated context.
    ``weight_by_d`` defaults to False to match ``gate()``; the two conventions
    differ by 26-49% of the belief term at d-bar .51-.74. ``d_t`` is a validity
    condition under either. Order dispersion is exempt by the Lemma 2 identity
    and ``rho`` belongs in the stability cap, so neither appears here.
    """
    if not (0.0 < float(d_t) <= 1.0):
        raise ValueError("monitor_charge: d_t must be a candidate mass in (0,1]")
    scale = float(d_t) if weight_by_d else 1.0
    return float(max(0.0, scale * kl_categorical(b_bar, b_0) - float(delta_lic)))


def anchor_pvalue(q0_a: float, reached_a: float) -> float:
    """``P(sup_t q_{t,a} >= alpha) <= q_{0,a}/alpha``: anytime, per cell.

    A detector with no null to estimate, no control arm and no horizon. It is
    also a direct read on how much a high anchor costs you: at ``q_0 = 0.78``
    reaching 0.95 carries p <= 0.82 and certifies nothing.
    """
    if reached_a <= q0_a:
        return 1.0
    return float(min(1.0, q0_a / reached_a))


@dataclass
class Charges:
    """Additive charges ``C``, in nats, against the available side.

    The order terms are telemetry by default and are not summed into ``total``.

    Reason, and it does not depend on the companion paper: the compensation
    identity ``E_G[KL(q^G||r)] = KL(b||r) + I(A;G)`` says the mean-of-divergences
    convention over-states the at-the-mean convention by exactly ``I(A;G)``.
    :func:`available_budget` uses the at-the-mean convention -- it evaluates
    ``KL(Ber(b_lo)||Ber(c))`` at bounds on the means -- so the over-count has
    already been removed by construction and charging it again pays it twice.
    What the at-the-mean convention still owes is estimation error in ``b_hat``,
    and that is what the one-sided bound ``b_lo`` delivers, with coverage at
    every m.

    ``order_ablated`` must never be charged: the ablated context is an
    instrument, never deployed, and its order sensitivity enters through the
    width of the one-sided upper bound on ``c``.

    ``charge_order_evidence=True`` is available for the reading in which
    deployment answers under a single serialization rather than the marginal.
    Be aware that it does not buy that guarantee: the shift is an expected-KL
    correction and the guarantee wanted is a quantile, and the two do not track
    (measured insufficient at every dispersion level). If single-serialization
    deployment is the risk, bound the lower quantile of ``q^G_a`` directly.
    """

    order_evidence: float = 0.0   # telemetry unless charge_order_evidence
    order_ablated: float = 0.0    # telemetry ALWAYS; never charged
    rho: float = 0.0
    calibration: float = 0.0
    other: float = 0.0
    charge_order_evidence: bool = False

    @property
    def total(self) -> float:
        base = float(self.rho + self.calibration + self.other)
        if self.charge_order_evidence:
            base += float(self.order_evidence)
        return base

    def as_dict(self) -> Dict[str, float]:
        return {
            "order_evidence": self.order_evidence,
            "order_ablated": self.order_ablated,
            "order_evidence_charged": float(self.charge_order_evidence),
            "rho": self.rho,
            "calibration": self.calibration,
            "other": self.other,
            "total": self.total,
        }


@dataclass
class GateResult:
    cell: int
    p_star: float
    b_lo: float  # one-sided EB lower bound on b(A)
    anchor: float  # c = one-sided EB upper bound on b_ablated(A)
    available: float  # A
    required: float  # R
    charges: float  # C
    margin: float  # M = A - R - C   <- the decision variable
    headroom: float  # H0
    p_star_eff: float
    isr: Optional[float]  # telemetry only
    answered: bool
    feasible: bool
    reasons: List[str] = field(default_factory=list)
    charge_detail: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cell": self.cell,
            "p_star": self.p_star,
            "b_lo": self.b_lo,
            "anchor": self.anchor,
            "available": self.available,
            "required": self.required,
            "charges": self.charges,
            "margin": self.margin,
            "headroom": self.headroom,
            "p_star_eff": self.p_star_eff,
            "isr": self.isr,
            "answered": self.answered,
            "feasible": self.feasible,
            "reasons": list(self.reasons),
            "charge_detail": dict(self.charge_detail),
        }


def gate(
    *,
    cell: int,
    b_lo_a: float,
    anchor_c: float,
    p_star: float,
    charges: Charges | float = 0.0,
) -> GateResult:
    """The gate: answer iff the margin ``M = A - R - C >= 0``.

    There is no ``q_lo``, no clip, and no free parameter beyond ``p*`` and the
    confidence level ``alpha`` that produced ``b_lo`` and ``c``.
    """
    ch = charges if isinstance(charges, Charges) else Charges(other=float(charges))
    c_total = ch.total
    available = available_budget(b_lo_a, anchor_c)
    required = required_budget(p_star, anchor_c)
    h0 = charge_headroom(p_star, anchor_c)
    margin = available - required - c_total
    feasible = c_total < h0
    eff = p_star_effective(p_star, anchor_c, c_total)
    isr = (available / (required + c_total)) if (required + c_total) > EPS else None

    reasons: List[str] = []
    if anchor_c >= p_star:
        reasons.append("anchor_at_or_above_target")  # target met without evidence
    if b_lo_a < anchor_c:
        reasons.append("no_directional_movement")
    if not feasible:
        reasons.append("charges_exceed_headroom")
    if margin < 0.0 and not reasons:
        reasons.append("insufficient_margin")

    answered = bool(margin >= 0.0 and feasible and anchor_c < p_star)
    return GateResult(
        cell=int(cell),
        p_star=float(p_star),
        b_lo=float(b_lo_a),
        anchor=float(anchor_c),
        available=float(available),
        required=float(required),
        charges=float(c_total),
        margin=float(margin),
        headroom=float(h0),
        p_star_eff=float(eff),
        isr=(float(isr) if isr is not None else None),
        answered=answered,
        feasible=feasible,
        reasons=reasons,
        charge_detail=ch.as_dict(),
    )


# --------------------------------------------------------------------------
# Estimation: one-sided empirical-Bernstein, and the retired q_lo.
# --------------------------------------------------------------------------


def _mean_var(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v


def eb_upper(xs: Sequence[float], alpha: float = 0.05) -> float:
    """One-sided empirical-Bernstein upper bound (Maurer-Pontil) for x in [0,1].

    Width is ``O(m^{-1/2})`` with coverage at *every* m. This replaces
    ``q_lo = min_G q^G_a``, whose expected downward bias grows like
    ``sd_G(q^G_a) sqrt(2 log m)``, so that the inherited gate *tightens* as
    more serializations are collected.
    """
    n = len(xs)
    if n < 2:
        return 1.0
    m, v = _mean_var(xs)
    ln = math.log(2.0 / alpha)
    return float(min(1.0, m + math.sqrt(2.0 * v * ln / n) + 7.0 * ln / (3.0 * (n - 1))))


def eb_lower(xs: Sequence[float], alpha: float = 0.05) -> float:
    """One-sided empirical-Bernstein lower bound for x in [0,1]."""
    n = len(xs)
    if n < 2:
        return 0.0
    m, v = _mean_var(xs)
    ln = math.log(2.0 / alpha)
    return float(max(0.0, m - math.sqrt(2.0 * v * ln / n) - 7.0 * ln / (3.0 * (n - 1))))


def hoeffding_lower(xs: Sequence[float], alpha: float = 0.05) -> float:
    n = len(xs)
    if n < 1:
        return 0.0
    m, _ = _mean_var(xs)
    return float(max(0.0, m - math.sqrt(math.log(1.0 / alpha) / (2.0 * n))))


def hoeffding_upper(xs: Sequence[float], alpha: float = 0.05) -> float:
    n = len(xs)
    if n < 1:
        return 1.0
    m, _ = _mean_var(xs)
    return float(min(1.0, m + math.sqrt(math.log(1.0 / alpha) / (2.0 * n))))


def wsr_bounds(xs: Sequence[float], alpha: float = 0.05, lam_cap: float = 0.9) -> Tuple[float, float]:
    """Betting (predictable plug-in empirical-Bernstein) bounds for x in [0,1].

    Waudby-Smith & Ramdas. This is the estimator that makes the battery
    affordable (``lam_cap = 0.9`` is tuned below; 0.5 and 0.99 are both looser): the Maurer-Pontil bound carries a ``7 ln(2/alpha)/(3(m-1))``
    term that dominates at every m a serialization battery can reach, whereas
    the betting bound adapts to the (very small) variance of readouts near the
    boundary, which is exactly the regime the gate is consulted in.
    """
    n = len(xs)
    if n < 2:
        return 0.0, 1.0
    log_a = math.log(1.0 / alpha)
    s_sum = 0.0
    lam_sum = 0.0
    pad = 0.0
    run_sum, run_sq, mu_prev, var_prev = 0.0, 0.0, 0.5, 0.25
    for i, x in enumerate(xs, start=1):
        lam = math.sqrt(2.0 * log_a / max(var_prev * i * math.log(1.0 + i), EPS))
        lam = min(max(lam, 1e-6), lam_cap)
        v = 4.0 * (float(x) - mu_prev) ** 2
        psi = (-math.log(1.0 - lam) - lam) / 4.0
        s_sum += lam * float(x)
        lam_sum += lam
        pad += v * psi
        run_sum += float(x)
        run_sq += (float(x) - mu_prev) ** 2
        mu_prev = (0.5 + run_sum) / (1.0 + i)
        var_prev = (0.25 + run_sq) / (1.0 + i)
    bound = log_a + pad
    lo = (s_sum - bound) / lam_sum
    hi = (s_sum + bound) / lam_sum
    return float(max(0.0, lo)), float(min(1.0, hi))


def wsr_sequence(
    xs: Sequence[float], alpha: float = 0.05, lam_cap: float = 0.9
) -> List[Tuple[float, float]]:
    """The betting bound evaluated at every t: a confidence *sequence*.

    ``wsr_bounds`` returns only the final interval. The same capital process is
    anytime-valid by Ville, and the betting rate ``lam_t`` is predictable, so
    the interval holds simultaneously at all t. That is what licenses stopping
    the battery as soon as the decision is determined, with no correction: it is
    optional stopping on a confidence sequence, not on a fixed-sample bound.
    """
    n = len(xs)
    out: List[Tuple[float, float]] = []
    if n == 0:
        return out
    log_a = math.log(1.0 / alpha)
    s_sum = lam_sum = pad = 0.0
    run_sum, run_sq, mu_prev, var_prev = 0.0, 0.0, 0.5, 0.25
    for i, x in enumerate(xs, start=1):
        lam = math.sqrt(2.0 * log_a / max(var_prev * i * math.log(1.0 + i), EPS))
        lam = min(max(lam, 1e-6), lam_cap)
        pad += 4.0 * (float(x) - mu_prev) ** 2 * ((-math.log(1.0 - lam) - lam) / 4.0)
        s_sum += lam * float(x)
        lam_sum += lam
        run_sum += float(x)
        run_sq += (float(x) - mu_prev) ** 2
        mu_prev = (0.5 + run_sum) / (1.0 + i)
        var_prev = (0.25 + run_sq) / (1.0 + i)
        bound = log_a + pad
        out.append((max(0.0, (s_sum - bound) / lam_sum),
                    min(1.0, (s_sum + bound) / lam_sum)))
    return out


def sequential_decision(
    xs: Sequence[float], threshold: float, alpha: float = 0.05, *, min_draws: int = 8
) -> Dict[str, object]:
    """Stop as soon as the confidence sequence separates from ``threshold``.

    Returns the stopping time and the decision. Because the underlying object is
    a confidence sequence, both the coverage and the decision are valid at the
    data-dependent stopping time -- which a fixed-sample bound would not be.
    """
    seq = wsr_sequence(xs, alpha)
    for t, (lo, hi) in enumerate(seq, start=1):
        if t < min_draws:
            continue
        if lo >= threshold:
            return {"stopped_at": t, "decision": "above", "lo": lo, "hi": hi, "exhausted": False}
        if hi < threshold:
            return {"stopped_at": t, "decision": "below", "lo": lo, "hi": hi, "exhausted": False}
    lo, hi = seq[-1]
    return {"stopped_at": len(seq), "decision": "undetermined", "lo": lo, "hi": hi,
            "exhausted": True}


def battery_plan(n_spans: int, m_needed: int) -> Dict[str, object]:
    """Enumerate the orbit only when it is cheaper than the bound you need.

    Exhaustive enumeration makes ``b`` exact and removes ``alpha`` entirely, but
    it costs ``n!``, which crosses any fixed sampling budget between five and six
    spans. Enumerating because the orbit is small enough to enumerate, rather
    than because it is cheaper than the sample you would otherwise draw, makes a
    six-span context cost four times a nine-span one.
    """
    orbit = math.factorial(max(n_spans, 0))
    exhaustive = orbit <= max(1, int(m_needed))
    draws = orbit if exhaustive else int(m_needed)
    return {
        "orbit": orbit,
        "exhaustive": exhaustive,
        "draws_per_context": draws,
        "exact": exhaustive,
        "note": ("orbit enumerated: b is exact and alpha plays no role"
                 if exhaustive else "orbit sampled: confidence sequence required"),
    }


def conf_lower(xs: Sequence[float], alpha: float = 0.05, *, method: str = "auto") -> float:
    """One-sided lower confidence bound on ``b_a``, replacing ``q_lo``.

    ``method`` must be **pre-registered**, not chosen from the data: selecting
    the tighter of several valid bounds after seeing the sample costs a union
    bound. ``auto`` = the betting bound, which dominates the others at every
    battery size we can afford.
    """
    if method == "hoeffding":
        return hoeffding_lower(xs, alpha)
    if method == "eb":
        return eb_lower(xs, alpha)
    if method == "union":  # valid, but pays alpha/2 to both families
        return float(max(eb_lower(xs, alpha / 2.0), hoeffding_lower(xs, alpha / 2.0)))
    return wsr_bounds(xs, alpha)[0]


def conf_upper(xs: Sequence[float], alpha: float = 0.05, *, method: str = "auto") -> float:
    """One-sided upper confidence bound on the ablated anchor ``c``."""
    if method == "hoeffding":
        return hoeffding_upper(xs, alpha)
    if method == "eb":
        return eb_upper(xs, alpha)
    if method == "union":
        return float(min(eb_upper(xs, alpha / 2.0), hoeffding_upper(xs, alpha / 2.0)))
    return wsr_bounds(xs, alpha)[1]


# --------------------------------------------------------------------------
# The rho audit: split-half inner product, never a single squared norm.
# --------------------------------------------------------------------------


def split_half_displacement(
    half_a: Sequence[Sequence[float]],
    half_b: Sequence[Sequence[float]],
    full_orbit_mean: Sequence[float],
) -> float:
    """Unbiased per-cell displacement estimate ``<d1, d2>``; null exactly zero.

    ``rho`` is invisible to ``J^mu`` (second-order in the residual, where rho
    displaces the mean at first order) and it enters the budget where it hurts:
    ``log(1/(b_a - rho_a)) - log(1/b_a) ~ rho_a/b_a``, and the gate is
    consulted on small-mass cells.
    """
    m1 = order_marginal(half_a)
    m2 = order_marginal(half_b)
    return float(
        sum((m1[a] - full_orbit_mean[a]) * (m2[a] - full_orbit_mean[a]) for a in range(len(m1)))
    )


def naive_squared_displacement(
    draws: Sequence[Sequence[float]], full_orbit_mean: Sequence[float]
) -> float:
    """The biased alternative: ``||dhat||^2`` manufactures displacement from MC error."""
    m = order_marginal(draws)
    return float(sum((m[a] - full_orbit_mean[a]) ** 2 for a in range(len(m))))


# --------------------------------------------------------------------------
# The sequence layer: gate once, on the answer-measurable event.
# --------------------------------------------------------------------------


def renorm_order_gap(qs: Sequence[Sequence[float]], ds: Sequence[float], a: int) -> float:
    """``E_G[P_a/d] - E_G[P_a]/E_G[d] == -Cov_G(q_a, d) / E_G[d]``, exactly.

    The departure a renormalize-then-average implementation avoids is a
    A covariance, not a spread. It vanishes at zero correlation for any
    ``sd_G(d)``, and a small ``sd_G(d)`` with strong correlation can exceed a
    large one with none by an order of magnitude. Report this, not ``sd_G(d)``.
    """
    n = len(qs)
    if n != len(ds) or n == 0:
        raise ValueError("renorm_order_gap: mismatched or empty draws")
    qa = [float(q[a]) for q in qs]
    dd = [float(x) for x in ds]
    mq = sum(qa) / n
    md = sum(dd) / n
    cov = sum((x - mq) * (y - md) for x, y in zip(qa, dd)) / n
    return float(-cov / max(md, EPS))


def per_step_headroom(p_star: float, n_steps: int, c: float) -> float:
    """Headroom under the equal-allocation per-step target ``p*^(1/N)``.

    Any allocation meeting ``prod_k p_k >= p*`` drives the per-step targets
    toward 1 and ``H0`` to zero, so no continuation of useful length is
    certifiable step by step. This refutes a per-claim gate over a trace.
    """
    return charge_headroom(p_star ** (1.0 / max(1, int(n_steps))), c)


def answer_pushforward(
    law: Dict[Tuple[int, ...], float], in_answer: Callable[[Tuple[int, ...]], bool]
) -> float:
    """``B(A)``: the pushforward of a sequence law under the answer map.

    The answer map ``y -> 1{y in A}`` needs no exchangeability: at ``Q_dagger``
    the ratio ``dQ/dB`` takes two values and is A-measurable by construction, so
    the pushforward is exact. The count map is *not* sufficient on a
    non-exchangeable law and loses the log
    multiplicity.
    """
    return float(sum(w for y, w in law.items() if in_answer(y)))


def prefix_updated_chain(
    components: Sequence[Dict[Tuple[int, ...], float]], weights: Sequence[float]
) -> Dict[Tuple[int, ...], float]:
    """Chain **posterior**-weighted conditionals: reproduces the mixture exactly.

    B is defined by its conditionals ``b_k = E_G[S^G_k]`` and is *not* equal to
    ``E_G[S^G_{1:N}]``: the mixture of joints has prefix-updated weights,
    because emitted tokens are evidence about which serialization produced
    them. A probe that re-permutes at a fixed prefix cannot avoid this gap, no
    matter how many draws it takes.
    """
    out: Dict[Tuple[int, ...], float] = {}
    for y in components[0]:
        out[y] = float(sum(w * comp[y] for w, comp in zip(weights, components)))
    return out


def prior_weighted_chain(
    components: Sequence[Dict[Tuple[int, ...], float]],
    weights: Sequence[float],
    alphabet: Sequence[int],
    n_steps: int,
) -> Dict[Tuple[int, ...], float]:
    """Chain **prior**-weighted conditionals: the estimator a causal probe gets."""

    def cond(comp: Dict[Tuple[int, ...], float], prefix: Tuple[int, ...], tok: int) -> float:
        num = sum(w for y, w in comp.items() if y[: len(prefix) + 1] == prefix + (tok,))
        den = sum(w for y, w in comp.items() if y[: len(prefix)] == prefix)
        return num / den if den > 0 else 0.0

    out: Dict[Tuple[int, ...], float] = {}

    def walk(prefix: Tuple[int, ...], acc: float) -> None:
        if len(prefix) == n_steps:
            out[prefix] = acc
            return
        for tok in alphabet:
            p = sum(w * cond(comp, prefix, tok) for w, comp in zip(weights, components))
            if p > 0.0:
                walk(prefix + (tok,), acc * p)

    walk((), 1.0)
    return out


# --------------------------------------------------------------------------
# Theorem 7: twin-trace detection.
# --------------------------------------------------------------------------


def twin_trace_reject(
    stats: Sequence[float], rng: Optional[random.Random] = None, *, randomize_ties: bool = True
) -> bool:
    """Rank test on ``k+1`` exchangeable arms; arm 0 is the monitored trace.

    Level is exactly ``1/(k+1)`` for **any** statistic and **any**
    autocorrelation, provided the checkpoint schedule and ``k`` are fixed in
    advance and ties are broken at random. Deterministic tie-breaking inflates
    the level; stopping at the first favourable checkpoint inflates it further.
    """
    rng = rng or random.Random()
    best = max(stats)
    winners = [i for i, s in enumerate(stats) if s >= best - 0.0]
    if len(winners) == 1:
        return winners[0] == 0
    if not randomize_ties:
        return 0 in winners  # reject on any tie: inflates the level
    return rng.choice(winners) == 0
