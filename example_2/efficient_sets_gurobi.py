import numpy as np
import gurobipy as gp
from gurobipy import GRB
from functools import lru_cache

import config as c
import constants as C

MAX_EFFICIENT_SET_SOLVE_SECONDS = 60.0
MMNL_CONT_QUAD_POINTS = 25
SUPPORTED_MODELS = ("MNL", "MMNL_5PT", "MMNL_2PT", "MMNLcont")


def _action_int_from_binary(x_vals):
    """Convert a binary offer vector into integer bitmask action."""
    action_int = 0
    for i, val in enumerate(x_vals):
        if val > 0.5:
            action_int |= (1 << i)
    return action_int


def _decode_action_bits(action_int, n_products):
    return [(action_int >> i) & 1 for i in range(n_products)]


def _mmnl_cont_quadrature(mu_b, sigma_b, n_points=MMNL_CONT_QUAD_POINTS):
    """Approximate E[f(beta)] for beta=-exp(mu+sigma*Z), Z~N(0,1)."""
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be a positive integer")

    z_nodes, z_weights = np.polynomial.hermite.hermgauss(n_points)
    std_norm_nodes = np.sqrt(2.0) * z_nodes
    mix_weights = z_weights / np.sqrt(np.pi)
    draw_betas = -np.exp(float(mu_b) + float(sigma_b) * std_norm_nodes)
    return np.asarray(draw_betas, dtype=float), np.asarray(mix_weights, dtype=float)


def _get_segment_parameters(
    model,
    beta,
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
    mmnl_cont_points=MMNL_CONT_QUAD_POINTS,
):
    """Return segment betas and mixture weights for supported demand models."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model}' is not supported by efficient_sets_gurobi. Supported: {', '.join(SUPPORTED_MODELS)}"
        )

    if model == "MNL":
        if beta is None:
            raise ValueError("beta must be provided for MNL")
        return np.asarray([float(beta)], dtype=float), np.asarray([1.0], dtype=float)

    if model == "MMNLcont":
        if mu_b is None:
            if beta is None:
                raise ValueError("Either mu_b or beta must be provided for MMNLcont")
            if float(beta) >= 0:
                raise ValueError("beta must be negative when deriving mu_b for MMNLcont")
            mu_b = float(np.log(-float(beta)))
        else:
            mu_b = float(mu_b)

        sigma_b = 0.3 if sigma_b is None else float(sigma_b)
        if sigma_b < 0:
            raise ValueError("sigma_b must be non-negative for MMNLcont")

        return _mmnl_cont_quadrature(mu_b=mu_b, sigma_b=sigma_b, n_points=mmnl_cont_points)

    if segment_betas is None or segment_weights is None:
        raise ValueError(f"segment_betas and segment_weights must be provided for {model}")

    seg_betas = np.asarray(segment_betas, dtype=float)
    seg_weights = np.asarray(segment_weights, dtype=float)

    if seg_betas.size == 0:
        raise ValueError("segment_betas must contain at least one value")
    if seg_weights.shape != seg_betas.shape:
        raise ValueError("segment_weights must have the same shape as segment_betas")
    if np.any(seg_weights < 0):
        raise ValueError("segment_weights must be non-negative")

    weight_sum = float(np.sum(seg_weights))
    if weight_sum <= 0:
        raise ValueError("segment_weights must sum to a positive value")

    return seg_betas, seg_weights / weight_sum


def _build_offer_qr_model(prices, betas, weights):
    """Build a pure MILP model for aggregate (Q, R) of an offer set.

    The bilinear terms p_k * x_i are linearized via McCormick envelopes,
    avoiding the need for NonConvex=2 and making this a standard MILP.

    Let z_ki = p_k * x_i. Since x_i in {0,1} and 0 <= p_k <= 1:
        z_ki <= p_k
        z_ki <= x_i
        z_ki >= p_k + x_i - 1
        z_ki >= 0

    The logit purchase probability constraint p_k * D_k = purchase_num becomes:
        p_k + sum_i exp(beta_k*r_i) * z_ki = sum_i exp(beta_k*r_i) * x_i

    The logit revenue constraint R_k * D_k = reward_num becomes:
        R_k + sum_i r_i*exp(beta_k*r_i) * z_ki = sum_i r_i*exp(beta_k*r_i) * x_i
    """
    n_products = len(prices)
    n_segments = len(betas)

    exp_util = np.exp(np.outer(betas, prices))  # shape (n_segments, n_products)

    model = gp.Model("efficient_set_qr_milp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(MAX_EFFICIENT_SET_SOLVE_SECONDS)

    x = model.addVars(n_products, vtype=GRB.BINARY, name="x")

    purchase_prob = {}
    exp_revenue = {}

    for k in range(n_segments):
        # Auxiliary linearization vars: z_ki = p_k * x_i
        z = model.addVars(n_products, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_{k}")

        p_k = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"p_{k}")
        r_k = model.addVar(lb=0.0, ub=float(np.max(prices)), vtype=GRB.CONTINUOUS, name=f"r_{k}")

        purchase_prob[k] = p_k
        exp_revenue[k] = r_k

        # McCormick linearization of z_ki = p_k * x_i
        for i in range(n_products):
            model.addConstr(z[i] <= p_k,                 name=f"mcc_ub_p_{k}_{i}")
            model.addConstr(z[i] <= x[i],                name=f"mcc_ub_x_{k}_{i}")
            model.addConstr(z[i] >= p_k + x[i] - 1.0,   name=f"mcc_lb_{k}_{i}")

        # Linearized purchase-probability constraint:
        #   p_k + sum_i exp(b_k * r_i) * z_ki = sum_i exp(b_k * r_i) * x_i
        model.addConstr(
            p_k
            + gp.quicksum(float(exp_util[k, i]) * z[i] for i in range(n_products))
            == gp.quicksum(float(exp_util[k, i]) * x[i] for i in range(n_products)),
            name=f"p_def_{k}",
        )

        # Linearized expected-revenue constraint:
        #   r_k + sum_i r_i*exp(b_k * r_i) * z_ki = sum_i r_i*exp(b_k * r_i) * x_i
        model.addConstr(
            r_k
            + gp.quicksum(float(prices[i] * exp_util[k, i]) * z[i] for i in range(n_products))
            == gp.quicksum(float(prices[i] * exp_util[k, i]) * x[i] for i in range(n_products)),
            name=f"r_def_{k}",
        )

    q_total = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="q_total")
    r_total = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r_total")

    model.addConstr(
        q_total == gp.quicksum(float(weights[k]) * purchase_prob[k] for k in range(n_segments)),
        name="q_total_def",
    )
    model.addConstr(
        r_total == gp.quicksum(float(weights[k]) * exp_revenue[k] for k in range(n_segments)),
        name="r_total_def",
    )

    return model, x, q_total, r_total


def _exclude_action_constraint(model, x_vars, action_int):
    """Add a no-good constraint to exclude a previously selected action."""
    bits = _decode_action_bits(action_int, len(x_vars))
    lhs = gp.quicksum((1.0 - x_vars[i]) if bits[i] == 1 else x_vars[i] for i in range(len(x_vars)))
    return model.addConstr(lhs >= 1.0)


def _solve_best_marginal_ratio(model, x_vars, q_total, r_total, current_q, current_r, used_actions, tol):
    """Solve max (R-R0)/(Q-Q0) over feasible offers that weakly dominate current point."""
    eps_q = 1e-8
    dinkelbach_tol = 1e-8

    c_q = model.addConstr(q_total >= float(current_q) + eps_q, name="frontier_q_lb")
    c_r = model.addConstr(r_total >= float(current_r) - tol, name="frontier_r_lb")
    ng_constraints = [_exclude_action_constraint(model, x_vars, action) for action in used_actions]

    try:
        eta = 0.0
        best = None

        for _ in range(50):
            objective = (r_total - float(current_r)) - float(eta) * (q_total - float(current_q))
            model.setObjective(objective, GRB.MAXIMIZE)
            model.optimize()

            if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                return None
            if model.SolCount == 0:
                return None

            q_val = float(q_total.X)
            r_val = float(r_total.X)
            dq = q_val - float(current_q)
            dr = r_val - float(current_r)

            if dq <= eps_q:
                return None

            action_int = _action_int_from_binary([x_vars[i].X for i in range(len(x_vars))])
            best = (action_int, q_val, r_val)

            transformed_value = dr - float(eta) * dq
            if abs(transformed_value) <= dinkelbach_tol:
                return best

            new_eta = dr / dq
            if abs(new_eta - eta) <= dinkelbach_tol:
                return best
            eta = new_eta

        return best
    finally:
        model.remove(c_q)
        model.remove(c_r)
        for constr in ng_constraints:
            model.remove(constr)
        model.update()


@lru_cache(maxsize=8)
def _identify_efficient_sets(
    model,
    beta=None,
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
    mmnl_cont_points=MMNL_CONT_QUAD_POINTS,
):
    beta = None if beta is None else float(beta)
    segment_betas = None if segment_betas is None else tuple(float(b) for b in segment_betas)
    segment_weights = None if segment_weights is None else tuple(float(w) for w in segment_weights)
    mu_b = None if mu_b is None else float(mu_b)
    sigma_b = None if sigma_b is None else float(sigma_b)

    betas, weights = _get_segment_parameters(
        model=model,
        beta=beta,
        segment_betas=segment_betas,
        segment_weights=segment_weights,
        mu_b=mu_b,
        sigma_b=sigma_b,
        mmnl_cont_points=mmnl_cont_points,
    )

    prices = np.asarray(C.r, dtype=float)
    gp_model, x_vars, q_total, r_total = _build_offer_qr_model(prices=prices, betas=betas, weights=weights)

    tol = 1e-10
    efficient_sequence = [0]
    current_q = 0.0
    current_r = 0.0

    try:
        while True:
            candidate = _solve_best_marginal_ratio(
                model=gp_model,
                x_vars=x_vars,
                q_total=q_total,
                r_total=r_total,
                current_q=current_q,
                current_r=current_r,
                used_actions=tuple(efficient_sequence),
                tol=tol,
            )

            if candidate is None:
                break

            action_int, q_val, r_val = candidate
            if action_int in efficient_sequence:
                break

            if q_val <= current_q + tol:
                break

            efficient_sequence.append(action_int)
            current_q = q_val
            current_r = r_val

            if current_q >= 1.0 - 1e-10:
                break
    finally:
        gp_model.dispose()

    return tuple(efficient_sequence)


def compute_efficient_sets(model, beta=None, segment_betas=None, segment_weights=None, mu_b=None, sigma_b=None):
    """Compute efficient offer sets using Gurobi optimization over product binaries."""
    if beta is None and segment_betas is None and model != "MMNLcont":
        beta = C.SENSITIVITY_BETA_GT["high"] if c.HIGH_SENSITIVITY else C.SENSITIVITY_BETA_GT["low"]

    if segment_betas is not None and isinstance(segment_betas, list):
        segment_betas = tuple(segment_betas)
    if segment_weights is not None and isinstance(segment_weights, list):
        segment_weights = tuple(segment_weights)

    return _identify_efficient_sets(
        model=model,
        beta=beta,
        segment_betas=segment_betas,
        segment_weights=segment_weights,
        mu_b=mu_b,
        sigma_b=sigma_b,
        mmnl_cont_points=MMNL_CONT_QUAD_POINTS,
    )