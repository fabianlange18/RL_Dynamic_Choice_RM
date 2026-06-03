import numpy as np
import gurobipy as gp
from gurobipy import GRB

import src.constants as C


def _action_int_from_binary(x_vals):
    """Convert a binary offer vector into integer bitmask action."""
    action_int = 0
    for i, val in enumerate(x_vals):
        if val > 0.5:
            action_int |= (1 << i)
    return action_int


def _decode_action_bits(action_int, n_products):
    return [(action_int >> i) & 1 for i in range(n_products)]


def _build_offer_qr_model(prices, betas, weights, env=None):
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

    model = gp.Model("efficient_set_qr_milp", env=env) if env is not None else gp.Model("efficient_set_qr_milp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(C.MAX_EFFICIENT_SET_SOLVE_SECONDS)

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
    """Solve max (R-R0)/(Q-Q0) in one nonconvex Gurobi solve."""
    eps_q = C.EFFICIENT_SET_FRONTIER_Q_EPS

    c_q = model.addConstr(q_total >= float(current_q) + eps_q, name="frontier_q_lb")
    c_r = model.addConstr(r_total >= float(current_r) - tol, name="frontier_r_lb")
    ng_constraints = [_exclude_action_constraint(model, x_vars, action) for action in used_actions]
    max_price = float(np.max(C.r))
    ratio_bound = max_price / max(float(eps_q), 1e-12)
    ratio = model.addVar(lb=-ratio_bound, ub=ratio_bound, vtype=GRB.CONTINUOUS, name="marginal_ratio")
    ratio_link = model.addConstr(
        (r_total - float(current_r)) >= ratio * (q_total - float(current_q)),
        name="marginal_ratio_link",
    )

    try:
        model.Params.NonConvex = 2
        model.setObjective(ratio, GRB.MAXIMIZE)
        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None
        if model.SolCount == 0:
            return None

        q_val = float(q_total.X)
        r_val = float(r_total.X)
        dq = q_val - float(current_q)

        if dq <= eps_q:
            return None

        action_int = _action_int_from_binary([x_vars[i].X for i in range(len(x_vars))])
        return (action_int, q_val, r_val)
    finally:
        model.remove(ratio_link)
        model.remove(ratio)
        model.remove(c_q)
        model.remove(c_r)
        for constr in ng_constraints:
            model.remove(constr)
        model.update()


def compute_efficient_sets(
    model,
    beta=None,
    segment_betas=None,
    segment_weights=None,
    env=None,
):
    
    if model == "MNL":
        betas, weights = [beta], [1.0]

    else:
        betas, weights = segment_betas, segment_weights

    prices = np.asarray(C.r, dtype=float)
    gp_model, x_vars, q_total, r_total = _build_offer_qr_model(prices=prices, betas=betas, weights=weights, env=env)

    tol = C.EFFICIENT_SET_NUMERIC_TOL
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

            if current_q >= 1.0 - C.EFFICIENT_SET_NUMERIC_TOL:
                break
    finally:
        gp_model.dispose()

    return tuple(efficient_sequence)