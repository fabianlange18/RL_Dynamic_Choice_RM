import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import warnings

import constants as C

MAX_STATE_SOLVE_SECONDS = 180.0
MMNL_CONT_QUAD_POINTS = 25
SUPPORTED_MODELS = ("MNL", "MMNL_5PT", "MMNL_2PT", "MMNLcont")


def _mmnl_cont_quadrature(mu_b, sigma_b, n_points=MMNL_CONT_QUAD_POINTS):
    """Approximate E[f(beta)] for beta=-exp(mu+sigma*Z), Z~N(0,1)."""
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be a positive integer")

    z_nodes, z_weights = np.polynomial.hermite.hermgauss(n_points)
    # Convert Gauss-Hermite nodes/weights for exp(-x^2) to standard normal expectation.
    std_norm_nodes = np.sqrt(2.0) * z_nodes
    mix_weights = z_weights / np.sqrt(np.pi)
    draw_betas = -np.exp(float(mu_b) + float(sigma_b) * std_norm_nodes)
    return np.asarray(draw_betas, dtype=float), np.asarray(mix_weights, dtype=float)


def _get_segment_parameters(
    model,
    estimated_beta,
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
    mmnl_cont_points=MMNL_CONT_QUAD_POINTS,
):
    """Return segment betas and weights used in the optimizer objective."""

    if model == "MNL":
        return np.asarray([float(estimated_beta)], dtype=float), np.asarray([1.0], dtype=float)

    if model == "MMNLcont":
        return _mmnl_cont_quadrature(mu_b=mu_b, sigma_b=sigma_b, n_points=mmnl_cont_points)

    return np.asarray(segment_betas, dtype=float), np.asarray(segment_weights, dtype=float)


def _action_int_from_binary(x_vals):
    """Convert a binary offer vector into integer bitmask action."""
    action_int = 0
    for i, val in enumerate(x_vals):
        if val > 0.5:
            action_int |= (1 << i)
    return action_int


def _build_offer_model(prices, betas, weights):
    """Build a reusable MIQCP model over product binaries."""
    n_products = len(prices)
    n_segments = len(betas)

    exp_util = np.exp(np.outer(betas, prices))

    model = gp.Model("dp_offer_selection_product_binary")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = float(MAX_STATE_SOLVE_SECONDS)
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


def solve_by_dp_gurobi(
    estimated_beta,
    estimated_lambda,
    model="MNL",
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
    mmnl_cont_points=MMNL_CONT_QUAD_POINTS,
):
    """DP solver using product-level binary optimization (no action enumeration)."""

    prices = np.asarray(C.r, dtype=float)
    betas, weights = _get_segment_parameters(
        model=model,
        estimated_beta=estimated_beta,
        segment_betas=segment_betas,
        segment_weights=segment_weights,
        mu_b=mu_b,
        sigma_b=sigma_b,
        mmnl_cont_points=mmnl_cont_points,
    )

    gp_model, x_vars, exp_revenue_vars, purchase_prob_vars, seg_weights = _build_offer_model(prices, betas, weights)

    v = np.zeros((C.T + 1, C.C + 1), dtype=np.float32)
    pi = np.zeros((C.T, C.C + 1), dtype=object)

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

    return v, pi


if __name__ == "__main__":
    test_failures = []

    def _record_check(condition, message):
        if condition:
            print(f"[PASS] {message}")
        else:
            print(f"[FAIL] {message}")
            test_failures.append(message)

    def _run_test(label, *, test_n, test_r, test_t, test_c, **solve_kwargs):
        old_n, old_r, old_t, old_c = C.n, C.r, C.T, C.C
        try:
            C.n = int(test_n)
            C.r = np.asarray(test_r, dtype=float)
            C.T = int(test_t)
            C.C = int(test_c)

            start = time.perf_counter()
            v_test, pi_test = solve_by_dp_gurobi(
                **solve_kwargs,
            )
            elapsed = time.perf_counter() - start
            v0c = float(v_test[0, C.C])
            print(f"[{label}] elapsed: {elapsed:.2f}s | V(0,C)={v0c:.4f} | pi shape={pi_test.shape}")
            _record_check(np.isfinite(v0c), f"[{label}] V(0,C) is finite")
            return v0c
        finally:
            C.n, C.r, C.T, C.C = old_n, old_r, old_t, old_c

    # Test 1: MNL baseline regression check.
    baseline_v0c = _run_test(
        "MNL-baseline",
        test_n=C.n,
        test_r=C.r,
        test_t=C.T,
        test_c=C.C,
        estimated_beta=-0.005044,
        estimated_lambda=0.501366,
        model="MNL",
    )
    expected_v0c = 36821.38
    _record_check(np.isclose(baseline_v0c, expected_v0c, atol=5), (
        f"MNL-baseline expected V(0,C)={expected_v0c:.2f}, got {baseline_v0c:.4f}"
    ))

    # Test 2: MMNL_5PT regression check with provided estimation outputs.
    mmnl_5pt_v0c = _run_test(
        "MMNL_5PT",
        test_n=C.n,
        test_r=C.r,
        test_t=C.T,
        test_c=C.C,
        estimated_beta=None,
        estimated_lambda=0.500732,
        model="MMNL_5PT",
        segment_betas=[-0.050000, -0.003484, -0.002575, -0.013784, -0.004943],
        segment_weights=[0.001068, 0.024394, 0.349788, 0.128167, 0.496584],
    )
    expected_mmnl_5pt_v0c = 28434.09
    _record_check(np.isclose(mmnl_5pt_v0c, expected_mmnl_5pt_v0c, atol=5), (
        f"MMNL_5PT expected V(0,C)={expected_mmnl_5pt_v0c:.2f}, got {mmnl_5pt_v0c:.4f}"
    ))

    # Test 3: MMNL_2PT regression check from current results logs.
    # Source: example_2/results/MMNL_2PT_high_all/00_exec.log
    mmnl_2pt_v0c = _run_test(
        "MMNL_2PT",
        test_n=C.n,
        test_r=C.r,
        test_t=C.T,
        test_c=C.C,
        estimated_beta=None,
        estimated_lambda=0.492829,
        model="MMNL_2PT",
        segment_betas=[-0.002351, -0.007523],
        segment_weights=[0.476964, 0.523036],
    )
    expected_mmnl_2pt_v0c = 37420.01
    _record_check(np.isclose(mmnl_2pt_v0c, expected_mmnl_2pt_v0c, atol=5), (
        f"MMNL_2PT expected V(0,C)={expected_mmnl_2pt_v0c:.2f}, got {mmnl_2pt_v0c:.4f}"
    ))

    # Test 4: MMNLcont regression check from current results logs.
    # Source: example_2/results/MMNL_2PT_high_all/00_exec.log
    mmnl_cont_v0c = _run_test(
        "MMNLcont",
        test_n=C.n,
        test_r=C.r,
        test_t=C.T,
        test_c=C.C,
        estimated_beta=None,
        estimated_lambda=0.492829,
        model="MMNLcont",
        mu_b=-5.429548,
        sigma_b=0.603061,
    )
    expected_mmnl_cont_v0c = 64144.36
    _record_check(np.isclose(mmnl_cont_v0c, expected_mmnl_cont_v0c, atol=15), (
        f"MMNLcont expected V(0,C)={expected_mmnl_cont_v0c:.2f}, got {mmnl_cont_v0c:.4f}"
    ))

    # Test 5: MNL with 100 products using temporary n and r.
    prices_100 = np.linspace(700.0, 100.0, 100, dtype=float)
    _run_test(
        "MNL-100-products",
        test_n=100,
        test_r=prices_100,
        test_t=C.T,
        test_c=C.C,
        estimated_beta=-0.0045,
        estimated_lambda=0.5,
        model="MNL",
    )

    if test_failures:
        print("\nTest run completed with failures:")
        for failure in test_failures:
            print(f" - {failure}")
    else:
        print("\nAll regression checks passed.")

    



    
