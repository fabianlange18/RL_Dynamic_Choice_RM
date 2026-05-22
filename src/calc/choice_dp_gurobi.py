import numpy as np
import gurobipy as gp
from gurobipy import GRB
import warnings

import src.constants as C


def _get_segment_parameters(
    model,
    estimated_beta,
    segment_betas=None,
    segment_weights=None,
):
    """Return segment betas and weights used in the optimizer objective."""

    if model == "MNL":
        return np.asarray([float(estimated_beta)], dtype=float), np.asarray([1.0], dtype=float)

    return np.asarray(segment_betas, dtype=float), np.asarray(segment_weights, dtype=float)


def _action_int_from_binary(x_vals):
    """Convert a binary offer vector into integer bitmask action."""
    action_int = 0
    for i, val in enumerate(x_vals):
        if val > 0.5:
            action_int |= (1 << i)
    return action_int


def _build_offer_model(prices, betas, weights, env=None):
    """Build a reusable MIQCP model over product binaries."""
    n_products = len(prices)
    n_segments = len(betas)

    exp_util = np.exp(np.outer(betas, prices))

    model = gp.Model("dp_offer_selection_product_binary", env=env) if env is not None else gp.Model("dp_offer_selection_product_binary")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = float(C.MAX_STATE_SOLVE_SECONDS)
    x = model.addVars(n_products, vtype=GRB.BINARY, name="x")

    logit_denom = {}
    exp_revenue = {}
    purchase_prob = {}

    for k in range(n_segments):
        max_denom = 1.0 + float(np.sum(exp_util[k]))
        logit_denom[k] = model.addVar(lb=1.0, ub=max_denom, vtype=GRB.CONTINUOUS, name=f"logit_denom_{k}")
        exp_revenue[k] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"exp_revenue_{k}")
        purchase_prob[k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"purchase_prob_{k}")

        purchase_num = gp.quicksum(float(exp_util[k, i]) * x[i] for i in range(n_products))
        reward_num = gp.quicksum(float(prices[i] * exp_util[k, i]) * x[i] for i in range(n_products))

        model.addConstr(logit_denom[k] == 1.0 + purchase_num, name=f"logit_denom_def_{k}")
        model.addQConstr(exp_revenue[k] * logit_denom[k] == reward_num, name=f"exp_revenue_frac_{k}")
        model.addQConstr(purchase_prob[k] * logit_denom[k] == purchase_num, name=f"purchase_prob_frac_{k}")

    return model, x, exp_revenue, purchase_prob, np.asarray(weights, dtype=float)


def solve_by_dp(
    efficient_sets,
    estimated_beta,
    estimated_lambda,
    model="MNL",
    env=None,
    segment_betas=None,
    segment_weights=None,
):
    """DP solver using product-level binary optimization (no action enumeration)."""

    prices = np.asarray(C.r, dtype=float)
    betas, weights = _get_segment_parameters(
        model=model,
        estimated_beta=estimated_beta,
        segment_betas=segment_betas,
        segment_weights=segment_weights,
    )

    gp_model, x_vars, exp_revenue_vars, purchase_prob_vars, seg_weights = _build_offer_model(prices, betas, weights, env=env)

    v = np.zeros((C.T + 1, C.C + 1), dtype=np.float32)
    pi = np.zeros((C.T, C.C + 1), dtype=object)

    try:
        for t in range(C.T - 1, -1, -1):
            next_v = v[t + 1]

            for capacity in range(1, C.C + 1):
                stay_value = next_v[capacity]
                buy_value = next_v[capacity - 1]

                delta_value = buy_value - stay_value

                objective_expr = gp.quicksum(
                    float(seg_weights[k]) * (exp_revenue_vars[k] + float(delta_value) * purchase_prob_vars[k])
                    for k in range(len(betas))
                )
                gp_model.setObjective(objective_expr, GRB.MAXIMIZE)
                gp_model.optimize()

                if gp_model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                    raise RuntimeError(f"Gurobi failed with status {gp_model.Status} at t={t}, x={capacity}")
                
                if gp_model.SolCount == 0:
                    best_idx = 0
                    best_value = stay_value
                    warnings.warn(
                        f"Gurobi did not return a feasible solution at t={t}, x={capacity}; using empty-offer fallback",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    x_sol = [x_vars[i].X for i in range(len(prices))]
                    best_idx = int(_action_int_from_binary(x_sol))

                    best_increment = float(gp_model.ObjVal)
                    best_value = stay_value + estimated_lambda * best_increment

                v[t, capacity] = best_value
                pi[t, capacity] = best_idx
    finally:
        gp_model.dispose()

    return v, pi
