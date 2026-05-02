import argparse
import json
import os
import time

import config as c
from env_example_2 import TalluriExample2
from estimation_biogeme import BiogemeEstimator, collect_transaction_data as collect_biogeme_data
from estimation_scipy import ScipyEstimator


GT_MODELS = ["MNL", "MMNL_5PT", "MMNL_2PT", "MMNLcont"]


def _time_call(fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _print_comparison(comparison):
    print("\n=== Estimator Runtime and Fit Comparison ===")
    print("Model            | Backend | Time (s) | LL         | AIC        | BIC        | Success")
    print("-" * 86)
    for model_name, model_data in comparison.items():
        for backend in ("biogeme", "scipy"):
            row = model_data[backend]
            print(
                f"{model_name:16s} | {backend:7s} | {row['time_s']:8.4f} | "
                f"{row['final_log_likelihood']:10.4f} | {row['aic']:10.4f} | {row['bic']:10.4f} | {str(row['success']):7s}"
            )
        delta_t = model_data["scipy"]["time_s"] - model_data["biogeme"]["time_s"]
        delta_ll = model_data["scipy"]["final_log_likelihood"] - model_data["biogeme"]["final_log_likelihood"]
        print(f"{'':16s} | delta   | {delta_t:8.4f} | {delta_ll:10.4f}")
        print("-" * 86)


def _print_parameter_diffs(results):
    print("\n=== Parameter Comparison (SciPy - Biogeme) ===")
    fields_by_model = {
        "MNL": ["beta", "lambda"],
        "MMNL_5PT": ["betas", "mixing_weights", "lambda"],
        "MMNL_2PT": ["betas", "mixing_weights", "lambda"],
        "MMNL_CONT": ["mu_b", "sigma_b", "mean_beta", "lambda"],
    }

    for model_name, fields in fields_by_model.items():
        print(f"\n{model_name}")
        b = results[model_name]["biogeme"]
        s = results[model_name]["scipy"]
        for field in fields:
            print(f"  {field:14s} | biogeme={_fmt(b[field])} | scipy={_fmt(s[field])}")


def _run_one_gt_model(gt_model, args):
    # Keep env var and runtime config aligned for modules that already imported config.
    os.environ["GT_MODEL"] = gt_model
    c.GT_MODEL = gt_model

    env = TalluriExample2(efficient_sets=None)
    observations = collect_biogeme_data(env, n_episodes=args.episodes)
    env.close()

    biogeme_estimator = BiogemeEstimator(observations)
    scipy_estimator = ScipyEstimator(observations)

    model_fns = {
        "MNL": (biogeme_estimator.estimate_mnl, scipy_estimator.estimate_mnl),
        "MMNL_5PT": (
            lambda: biogeme_estimator.estimate_mmnl(
                n_starts=args.biogeme_starts,
                random_seed=args.biogeme_seed,
                weight_sep_lambda=args.weight_sep_lambda,
                weight_sep_alpha=args.weight_sep_alpha,
            ),
            lambda: scipy_estimator.estimate_mmnl(
                n_starts=args.scipy_starts,
                random_seed=args.scipy_seed,
                weight_sep_lambda=args.weight_sep_lambda,
                weight_sep_alpha=args.weight_sep_alpha,
            ),
        ),
        "MMNL_2PT": (
            lambda: biogeme_estimator.estimate_mmnl(
                K=2,
                n_starts=args.biogeme_starts,
                random_seed=args.biogeme_seed,
                weight_sep_lambda=args.weight_sep_lambda,
                weight_sep_alpha=args.weight_sep_alpha,
            ),
            lambda: scipy_estimator.estimate_mmnl(
                K=2,
                n_starts=args.scipy_starts,
                random_seed=args.scipy_seed,
                weight_sep_lambda=args.weight_sep_lambda,
                weight_sep_alpha=args.weight_sep_alpha,
            ),
        ),
        "MMNL_CONT": (
            lambda: biogeme_estimator.estimate_mmnl_continuous(n_starts=args.biogeme_starts, random_seed=args.biogeme_seed),
            lambda: scipy_estimator.estimate_mmnl_continuous(n_starts=args.scipy_starts, random_seed=args.scipy_seed),
        ),
    }

    comparison = {}
    raw_results = {}

    for model_name, (biogeme_fn, scipy_fn) in model_fns.items():
        b_result, b_time = _time_call(biogeme_fn)
        s_result, s_time = _time_call(scipy_fn)

        comparison[model_name] = {
            "biogeme": {
                "time_s": b_time,
                "final_log_likelihood": b_result["final_log_likelihood"],
                "aic": b_result["aic"],
                "bic": b_result["bic"],
                "success": b_result["success"],
            },
            "scipy": {
                "time_s": s_time,
                "final_log_likelihood": s_result["final_log_likelihood"],
                "aic": s_result["aic"],
                "bic": s_result["bic"],
                "success": s_result["success"],
            },
        }
        raw_results[model_name] = {"biogeme": b_result, "scipy": s_result}

    return {
        "gt_model": gt_model,
        "n_observations": len(observations),
        "comparison": comparison,
        "raw_results": raw_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Biogeme and SciPy estimators on shared sampled data.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for sampling observations.")
    parser.add_argument("--scipy-starts", type=int, default=2, help="Number of multistart runs for SciPy MMNL 5PT and MMNL CONT.")
    parser.add_argument("--scipy-seed", type=int, default=0, help="Random seed for SciPy multistart initialization.")
    parser.add_argument("--biogeme-starts", type=int, default=2, help="Number of multistart runs for Biogeme MMNL 5PT and MMNL 2PT.")
    parser.add_argument("--biogeme-seed", type=int, default=0, help="Random seed for Biogeme multistart initialization.")
    parser.add_argument("--weight-sep-lambda", type=float, default=1.0, help="Soft penalty strength encouraging distinct segment weights (0 disables).")
    parser.add_argument("--weight-sep-alpha", type=float, default=10.0, help="Sharpness of weight-separation penalty.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save raw comparison results as JSON.",
    )
    parser.add_argument(
        "--gt-models",
        type=str,
        nargs="+",
        default=GT_MODELS,
        help="Ground-truth models to benchmark (defaults: MNL MMNL_5PT MMNL_2PT MMNL_CONT).",
    )
    args = parser.parse_args()

    all_outputs = {}
    for gt_model in args.gt_models:
        print(f"\n\n################ GT_MODEL={gt_model} ################")
        output = _run_one_gt_model(gt_model, args)
        _print_comparison(output["comparison"])
        _print_parameter_diffs(output["raw_results"])
        all_outputs[gt_model] = output

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"\nSaved detailed results to: {args.output_json}")


if __name__ == "__main__":
    main()