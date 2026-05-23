import os
import gc
import time
import pickle
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

import src.config as c
import src.constants as C


from src.calc.estimation_scipy import (
    ScipyEstimator,
    collect_transaction_data,
)
from src.talluri_env import TalluriExample2
from src.calc.train_rl import train_rl
from src.eval.evaluation import evaluate_saved_models, print_evaluation_table
from src.eval.simulation import simulate
from src.calc.choice_adp import solve_by_adp, solve_by_adp_env_rollout
if c.LARGE_PRODUCT_SET and not c.TRAIN_ON_ALL_SETS:
    from src.calc.efficient_sets_gurobi import compute_efficient_sets
else:
    from src.calc.efficient_sets import compute_efficient_sets

_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "cxx=" not in _pytensor_flags:
    os.environ["PYTENSOR_FLAGS"] = f"{_pytensor_flags},cxx=".strip(",")
logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)

if c.LARGE_PRODUCT_SET and c.TRAIN_ON_ALL_SETS:
    from src.calc.choice_dp_gurobi import solve_by_dp
else:
    from src.calc.choice_dp import solve_by_dp

def log_message(message):
    """Log message to file."""
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def log_gurobi_message(message):
    """Log Gurobi license/slot messages to stdout (captured by SLURM logs)."""
    print(message, flush=True)


@contextmanager
def gurobi_slot_lock(phase_label):
    """Limit concurrent Gurobi phases across SLURM tasks.

    Uses file locks on Linux clusters when GUROBI_MAX_SLOTS and
    GUROBI_SLOT_LOCK_DIR are configured. Falls back to no-op locally.
    """
    max_slots_raw = os.environ.get("GUROBI_MAX_SLOTS", "")
    lock_dir = os.environ.get("GUROBI_SLOT_LOCK_DIR", "")

    # No lock configuration -> no-op
    if not max_slots_raw or not lock_dir:
        yield
        return

    try:
        max_slots = int(max_slots_raw)
    except ValueError:
        yield
        return

    if max_slots <= 0:
        yield
        return

    os.makedirs(lock_dir, exist_ok=True)
    poll_seconds = 60
    stale_seconds = int(os.environ.get("GUROBI_SLOT_STALE_SECONDS", "21600"))
    events_log_path = os.path.join(lock_dir, "gurobi_lock_events.log")
    lock_path = None
    lock_slot = None

    def _lock_metadata(slot):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "task_id": os.environ.get("TASK_ID", "N/A"),
            "phase": str(phase_label),
            "slot": str(slot),
        }

    def _write_lock_event(action, slot):
        meta = _lock_metadata(slot)
        line = (
            f"{meta['timestamp']} action={action}"
            f" job_id={meta['job_id']}"
            f" task_id={meta['task_id']}"
            f" phase={meta['phase']}"
            f" slot={meta['slot']}"
        )
        try:
            with open(events_log_path, "a", encoding="utf-8") as events_log:
                events_log.write(f"{line}\n")
        except OSError as exc:
            log_gurobi_message(f"[{phase_label}] Failed to write lock event log: {exc}")

    def _dispose_gurobi_default_env():
        """Best-effort teardown of default Gurobi environment before slot release."""
        try:
            import gurobipy as gp
        except Exception:
            return

        try:
            gc.collect()
            gp.disposeDefaultEnv()
            log_gurobi_message(f"[{phase_label}] Disposed Gurobi default environment")
        except Exception as exc:
            log_gurobi_message(f"[{phase_label}] Failed to dispose Gurobi default environment: {exc}")

    try:
        log_gurobi_message(
            f"[{phase_label}] Waiting for Gurobi slot (max concurrent: {max_slots})"
        )
        while True:
            for slot in range(1, max_slots + 1):
                path = os.path.join(lock_dir, f"slot_{slot}.lock")

                # Best-effort stale lock cleanup for crashed jobs.
                if os.path.exists(path):
                    try:
                        age = time.time() - os.path.getmtime(path)
                        if age > stale_seconds:
                            os.remove(path)
                            log_gurobi_message(
                                f"[{phase_label}] Removed stale lock file: {path} (age={age:.0f}s)"
                            )
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w", encoding="utf-8") as meta:
                        meta.write(
                            f"pid={os.getpid()} task_id={os.environ.get('TASK_ID', 'N/A')} phase={phase_label}\n"
                        )
                    lock_path = path
                    lock_slot = slot
                    _write_lock_event("acquire", slot)
                    log_gurobi_message(f"[{phase_label}] Acquired Gurobi slot {slot}")
                    yield
                    return
                except FileExistsError:
                    pass
            time.sleep(poll_seconds)
    finally:
        if lock_path is not None:
            _dispose_gurobi_default_env()
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
            _write_lock_event("release", lock_slot)
            log_gurobi_message(f"[{phase_label}] Released Gurobi slot {lock_slot}")


def _effsets_pending_path():
    """Return the pending-marker path for this task, or None if not configured."""
    lock_dir = os.environ.get("GUROBI_SLOT_LOCK_DIR", "")
    task_id = os.environ.get("TASK_ID", "")
    if not lock_dir or not task_id:
        return None
    # Separate pending markers into their own subdirectory to avoid interference with slot locks
    pending_dir = os.path.join(lock_dir, "effsets_pending")
    os.makedirs(pending_dir, exist_ok=True)
    return os.path.join(pending_dir, f"task_{task_id}")


def _wait_for_effsets_priority(log_fn, poll_seconds=60):
    """Block until no other task's effsets pending-marker file remains."""
    lock_dir = os.environ.get("GUROBI_SLOT_LOCK_DIR", "")
    if not lock_dir:
        return
    import glob
    # Check the dedicated pending directory, not the main lock dir
    pending_dir = os.path.join(lock_dir, "effsets_pending")
    if not os.path.exists(pending_dir):
        return
    while True:
        pending = glob.glob(os.path.join(pending_dir, "task_*"))
        if not pending:
            return
        log_fn(f"[dp] Yielding to {len(pending)} active efficient-set task(s); waiting {poll_seconds}s")
        time.sleep(poll_seconds)


def main():
    """Main execution function."""
    
    # Log task configuration at start
    task_id = os.environ.get("TASK_ID", "N/A")
    log_message(f"\n{'='*60}")
    log_message(f"TASK_ID: {task_id}")
    log_message(f"LARGE_PRODUCT_SET: {c.LARGE_PRODUCT_SET}")
    log_message(f"TRAIN_ON_ALL_SETS: {c.TRAIN_ON_ALL_SETS}")
    log_message(f"HIGH_SENSITIVITY: {c.HIGH_SENSITIVITY}")
    log_message(f"GT_MODEL: {c.GT_MODEL}")
    log_message(f"{'='*60}")
    log_message(f"RL Training Seeds: {C.N_EVAL_EPISODES}, RL Training Steps: {C.TOTAL_TIMESTEPS}, Estimation Episodes: {C.N_ESTIMATION_EPISODES}\n")

    uses_gurobi_efficient_sets = c.LARGE_PRODUCT_SET and not c.TRAIN_ON_ALL_SETS
    _pending_marker = _effsets_pending_path() if uses_gurobi_efficient_sets else None
    if _pending_marker:
        open(_pending_marker, "w").close()
        log_gurobi_message(f"[efficient_sets] Registered pending marker: {_pending_marker}")

    # -- Shared observations: collect once, estimate all models ----------
    t0 = time.perf_counter()
    _env = TalluriExample2(efficient_sets=None)
    observations = collect_transaction_data(_env)
    _env.close()
    sampling_time = time.perf_counter() - t0
    log_message(f"Observation sampling time: {sampling_time:.4f} seconds")

    estimator = ScipyEstimator(observations)

    t0 = time.perf_counter()
    estimation_mnl_result = estimator.estimate_mnl()
    estimation_mnl_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    estimation_mmnl_5pt_result = estimator.estimate_mmnl()
    estimation_mmnl_5pt_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    estimation_mmnl_2pt_result = estimator.estimate_mmnl(K=2)
    estimation_mmnl_2pt_time = time.perf_counter() - t0

    beta_mnl   = estimation_mnl_result["beta"]
    lambda_mnl = estimation_mnl_result["lambda"]
    ll_mnl = estimation_mnl_result["final_log_likelihood"]
    aic_mnl = estimation_mnl_result["aic"]
    bic_mnl = estimation_mnl_result["bic"]

    betas_mmnl_5pt = estimation_mmnl_5pt_result["betas"]
    weights_mmnl_5pt = estimation_mmnl_5pt_result["mixing_weights"]
    lambda_mmnl_5pt = estimation_mmnl_5pt_result["lambda"]
    ll_mmnl_5pt = estimation_mmnl_5pt_result["final_log_likelihood"]
    aic_mmnl_5pt = estimation_mmnl_5pt_result["aic"]
    bic_mmnl_5pt = estimation_mmnl_5pt_result["bic"]

    betas_mmnl_2pt = estimation_mmnl_2pt_result["betas"]
    weights_mmnl_2pt = estimation_mmnl_2pt_result["mixing_weights"]
    lambda_mmnl_2pt = estimation_mmnl_2pt_result["lambda"]
    ll_mmnl_2pt = estimation_mmnl_2pt_result["final_log_likelihood"]
    aic_mmnl_2pt = estimation_mmnl_2pt_result["aic"]
    bic_mmnl_2pt = estimation_mmnl_2pt_result["bic"]

    log_message(f"Estimation_MNL  time: {estimation_mnl_time:.4f}s | beta: {beta_mnl:.6f}, lambda: {lambda_mnl:.6f}")
    log_message(f"  LL: {ll_mnl:.6f}, AIC: {aic_mnl:.3f}, BIC: {bic_mnl:.3f}\n")

    log_message(f"Estimation_MMNL_5PT time: {estimation_mmnl_5pt_time:.4f}s | lambda: {lambda_mmnl_5pt:.6f}")
    log_message(f"  LL: {ll_mmnl_5pt:.6f}, AIC: {aic_mmnl_5pt:.3f}, BIC: {bic_mmnl_5pt:.3f}")
    log_message(f"Betas: {[f'{b:.6f}' for b in betas_mmnl_5pt]}")
    log_message(f"Weights: {[f'{w:.6f}' for w in weights_mmnl_5pt]}\n")

    log_message(f"Estimation_MMNL_2PT time: {estimation_mmnl_2pt_time:.4f}s | lambda: {lambda_mmnl_2pt:.6f}")
    log_message(f"  LL: {ll_mmnl_2pt:.6f}, AIC: {aic_mmnl_2pt:.3f}, BIC: {bic_mmnl_2pt:.3f}")
    log_message(f"Betas: {[f'{b:.6f}' for b in betas_mmnl_2pt]}")
    log_message(f"Weights: {[f'{w:.6f}' for w in weights_mmnl_2pt]}\n")

    del observations
    del estimator
    gc.collect()

    # -- Efficient sets ---------------------------------------------------
    if c.TRAIN_ON_ALL_SETS:
        efficient_sets_mnl  = None
        efficient_sets_mmnl_5pt = None
        efficient_sets_mmnl_2pt = None
        efficient_sets_rl = None
        efficient_sets_time_mnl  = 0.0
        efficient_sets_time_mmnl_5pt = 0.0
        efficient_sets_time_mmnl_2pt = 0.0
    else:
        if uses_gurobi_efficient_sets:
            with gurobi_slot_lock("efficient_sets"):
                import gurobipy as gp

                with gp.Env() as gurobi_env:
                    t0 = time.perf_counter()
                    efficient_sets_mnl = compute_efficient_sets(model="MNL", beta=beta_mnl, env=gurobi_env)
                    efficient_sets_time_mnl = time.perf_counter() - t0
                    log_message(f"MNL  efficient sets time: {efficient_sets_time_mnl:.4f}s | sets: {efficient_sets_mnl}")

                    t0 = time.perf_counter()
                    efficient_sets_mmnl_5pt = compute_efficient_sets(
                        model="MMNL_5PT",
                        env=gurobi_env,
                        segment_betas=betas_mmnl_5pt,
                        segment_weights=weights_mmnl_5pt,
                    )
                    efficient_sets_time_mmnl_5pt = time.perf_counter() - t0
                    log_message(f"MMNL 5PT efficient sets time: {efficient_sets_time_mmnl_5pt:.4f}s | sets: {efficient_sets_mmnl_5pt}")

                    t0 = time.perf_counter()
                    efficient_sets_mmnl_2pt = compute_efficient_sets(
                        model="MMNL_2PT",
                        env=gurobi_env,
                        segment_betas=betas_mmnl_2pt,
                        segment_weights=weights_mmnl_2pt,
                    )
                    efficient_sets_time_mmnl_2pt = time.perf_counter() - t0
                    log_message(f"MMNL 2PT efficient sets time: {efficient_sets_time_mmnl_2pt:.4f}s | sets: {efficient_sets_mmnl_2pt}")

                # Flush lingering Gurobi references before the slot lock releases.
                gc.collect()
        else:
            # Non-Gurobi path: compute efficient sets without env
            t0 = time.perf_counter()
            efficient_sets_mnl = compute_efficient_sets(model="MNL", beta=beta_mnl)
            efficient_sets_time_mnl = time.perf_counter() - t0
            log_message(f"MNL  efficient sets time: {efficient_sets_time_mnl:.4f}s | sets: {efficient_sets_mnl}")

            t0 = time.perf_counter()
            efficient_sets_mmnl_5pt = compute_efficient_sets(
                model="MMNL_5PT",
                segment_betas=betas_mmnl_5pt,
                segment_weights=weights_mmnl_5pt,
            )
            efficient_sets_time_mmnl_5pt = time.perf_counter() - t0
            log_message(f"MMNL 5PT efficient sets time: {efficient_sets_time_mmnl_5pt:.4f}s | sets: {efficient_sets_mmnl_5pt}")

            t0 = time.perf_counter()
            efficient_sets_mmnl_2pt = compute_efficient_sets(
                model="MMNL_2PT",
                segment_betas=betas_mmnl_2pt,
                segment_weights=weights_mmnl_2pt,
            )
            efficient_sets_time_mmnl_2pt = time.perf_counter() - t0
            log_message(f"MMNL 2PT efficient sets time: {efficient_sets_time_mmnl_2pt:.4f}s | sets: {efficient_sets_mmnl_2pt}")

    if _pending_marker and os.path.exists(_pending_marker):
        os.remove(_pending_marker)
        log_gurobi_message(f"[efficient_sets] Removed pending marker: {_pending_marker}")

    efficient_sets_rl_candidates = [
        efficient_sets_mnl,
        efficient_sets_mmnl_5pt,
        efficient_sets_mmnl_2pt,
    ]
    efficient_sets_rl = None
    for _sets in efficient_sets_rl_candidates:
        if _sets:
            efficient_sets_rl = set(_sets) if efficient_sets_rl is None else (efficient_sets_rl | set(_sets))
    

    # -- DP solutions (with memory optimization: delete after use) -------
    uses_gurobi_dp = c.LARGE_PRODUCT_SET and c.TRAIN_ON_ALL_SETS
    if uses_gurobi_dp:
        _wait_for_effsets_priority(log_gurobi_message)
    
    if uses_gurobi_dp:
        with gurobi_slot_lock("dp"):
            import gurobipy as gp

            with gp.Env() as gurobi_env:
                t0 = time.perf_counter()
                v_mnl, pi_mnl = solve_by_dp(
                    efficient_sets=efficient_sets_mnl,
                    estimated_beta=beta_mnl,
                    estimated_lambda=lambda_mnl,
                    model="MNL",
                    env=gurobi_env,
                )
                dp_mnl_time = time.perf_counter() - t0
                _avg_mnl = sum(simulate(efficient_sets_mnl, pi_mnl, seed=i)[0] for i in range(1000)) / 1000
                log_message(f"DP_MNL  time: {dp_mnl_time:.4f}s | V(0,C): {v_mnl[0, C.C]:.2f} | avg reward: {_avg_mnl:.2f}")

                t0 = time.perf_counter()
                v_mmnl_5pt, pi_mmnl_5pt = solve_by_dp(
                    efficient_sets=efficient_sets_mmnl_5pt,
                    estimated_beta=None,
                    estimated_lambda=lambda_mmnl_5pt,
                    model="MMNL_5PT",
                    env=gurobi_env,
                    segment_betas=betas_mmnl_5pt,
                    segment_weights=weights_mmnl_5pt,
                )
                dp_mmnl_5pt_time = time.perf_counter() - t0
                _avg_mmnl_5pt = sum(simulate(efficient_sets_mmnl_5pt, pi_mmnl_5pt, seed=i)[0] for i in range(1000)) / 1000
                log_message(f"DP_MMNL_5PT time: {dp_mmnl_5pt_time:.4f}s | V(0,C): {v_mmnl_5pt[0, C.C]:.2f} | avg reward: {_avg_mmnl_5pt:.2f}")

                t0 = time.perf_counter()
                v_mmnl_2pt, pi_mmnl_2pt = solve_by_dp(
                    efficient_sets=efficient_sets_mmnl_2pt,
                    estimated_beta=None,
                    estimated_lambda=lambda_mmnl_2pt,
                    model="MMNL_2PT",
                    env=gurobi_env,
                    segment_betas=betas_mmnl_2pt,
                    segment_weights=weights_mmnl_2pt,
                )
                dp_mmnl_2pt_time = time.perf_counter() - t0
                _avg_mmnl_2pt = sum(simulate(efficient_sets_mmnl_2pt, pi_mmnl_2pt, seed=i)[0] for i in range(1000)) / 1000
                log_message(f"DP_MMNL_2PT time: {dp_mmnl_2pt_time:.4f}s | V(0,C): {v_mmnl_2pt[0, C.C]:.2f} | avg reward: {_avg_mmnl_2pt:.2f}")

            # Flush lingering Gurobi references before the slot lock releases.
            gc.collect()
    else:
        # Non-Gurobi path: solve DP without env
        t0 = time.perf_counter()
        v_mnl, pi_mnl = solve_by_dp(
            efficient_sets=efficient_sets_mnl,
            estimated_beta=beta_mnl,
            estimated_lambda=lambda_mnl,
            model="MNL",
        )
        dp_mnl_time = time.perf_counter() - t0
        _avg_mnl = sum(simulate(efficient_sets_mnl, pi_mnl, seed=i)[0] for i in range(1000)) / 1000
        log_message(f"DP_MNL  time: {dp_mnl_time:.4f}s | V(0,C): {v_mnl[0, C.C]:.2f} | avg reward: {_avg_mnl:.2f}")

        t0 = time.perf_counter()
        v_mmnl_5pt, pi_mmnl_5pt = solve_by_dp(
            efficient_sets=efficient_sets_mmnl_5pt,
            estimated_beta=None,
            estimated_lambda=lambda_mmnl_5pt,
            model="MMNL_5PT",
            segment_betas=betas_mmnl_5pt,
            segment_weights=weights_mmnl_5pt,
        )
        dp_mmnl_5pt_time = time.perf_counter() - t0
        _avg_mmnl_5pt = sum(simulate(efficient_sets_mmnl_5pt, pi_mmnl_5pt, seed=i)[0] for i in range(1000)) / 1000
        log_message(f"DP_MMNL_5PT time: {dp_mmnl_5pt_time:.4f}s | V(0,C): {v_mmnl_5pt[0, C.C]:.2f} | avg reward: {_avg_mmnl_5pt:.2f}")

        t0 = time.perf_counter()
        v_mmnl_2pt, pi_mmnl_2pt = solve_by_dp(
            efficient_sets=efficient_sets_mmnl_2pt,
            estimated_beta=None,
            estimated_lambda=lambda_mmnl_2pt,
            model="MMNL_2PT",
            segment_betas=betas_mmnl_2pt,
            segment_weights=weights_mmnl_2pt,
        )
        dp_mmnl_2pt_time = time.perf_counter() - t0
        _avg_mmnl_2pt = sum(simulate(efficient_sets_mmnl_2pt, pi_mmnl_2pt, seed=i)[0] for i in range(1000)) / 1000
        log_message(f"DP_MMNL_2PT time: {dp_mmnl_2pt_time:.4f}s | V(0,C): {v_mmnl_2pt[0, C.C]:.2f} | avg reward: {_avg_mmnl_2pt:.2f}")

    adp_mnl_time = 0.0
    adp_mmnl_5pt_time = 0.0
    adp_mmnl_2pt_time = 0.0
    adp_env_time = 0.0
    t0 = time.perf_counter()
    v_adp_mnl, pi_adp_mnl = solve_by_adp(
        efficient_sets=efficient_sets_mnl,
        estimated_beta=beta_mnl,
        estimated_lambda=lambda_mnl,
        model="MNL",
    )
    adp_mnl_time = time.perf_counter() - t0
    _avg_adp_mnl = sum(simulate(efficient_sets_mnl, pi_adp_mnl, seed=i)[0] for i in range(1000)) / 1000
    log_message(f"ADP_MNL  time: {adp_mnl_time:.4f}s | V(0,C): {v_adp_mnl[0, C.C]:.2f} | avg reward: {_avg_adp_mnl:.2f}")

    t0 = time.perf_counter()
    v_adp_mmnl_5pt, pi_adp_mmnl_5pt = solve_by_adp(
        efficient_sets=efficient_sets_mmnl_5pt,
        estimated_beta=None,
        estimated_lambda=lambda_mmnl_5pt,
        model="MMNL_5PT",
        segment_betas=betas_mmnl_5pt,
        segment_weights=weights_mmnl_5pt,
    )
    adp_mmnl_5pt_time = time.perf_counter() - t0
    _avg_adp_mmnl_5pt = sum(simulate(efficient_sets_mmnl_5pt, pi_adp_mmnl_5pt, seed=i)[0] for i in range(1000)) / 1000
    log_message(f"ADP_MMNL_5PT time: {adp_mmnl_5pt_time:.4f}s | V(0,C): {v_adp_mmnl_5pt[0, C.C]:.2f} | avg reward: {_avg_adp_mmnl_5pt:.2f}")

    t0 = time.perf_counter()
    v_adp_mmnl_2pt, pi_adp_mmnl_2pt = solve_by_adp(
        efficient_sets=efficient_sets_mmnl_2pt,
        estimated_beta=None,
        estimated_lambda=lambda_mmnl_2pt,
        model="MMNL_2PT",
        segment_betas=betas_mmnl_2pt,
        segment_weights=weights_mmnl_2pt,
    )
    adp_mmnl_2pt_time = time.perf_counter() - t0
    _avg_adp_mmnl_2pt = sum(simulate(efficient_sets_mmnl_2pt, pi_adp_mmnl_2pt, seed=i)[0] for i in range(1000)) / 1000
    log_message(f"ADP_MMNL_2PT time: {adp_mmnl_2pt_time:.4f}s | V(0,C): {v_adp_mmnl_2pt[0, C.C]:.2f} | avg reward: {_avg_adp_mmnl_2pt:.2f}")

    t0 = time.perf_counter()
    v_adp_env, pi_adp_env = solve_by_adp_env_rollout(
        efficient_sets=efficient_sets_rl,
    )
    adp_env_time = time.perf_counter() - t0
    _avg_adp_env = sum(simulate(efficient_sets_rl, pi_adp_env, seed=i)[0] for i in range(1000)) / 1000
    log_message(f"ADP_ENV time: {adp_env_time:.4f}s | V(0,C): {v_adp_env[0, C.C]:.2f} | avg reward: {_avg_adp_env:.2f}")

    time_results = {
        "Sampling":                sampling_time,
        "Estimation_MNL":          estimation_mnl_time,
        "Estimation_MMNL_5PT":     estimation_mmnl_5pt_time,
        "Estimation_MMNL_2PT":     estimation_mmnl_2pt_time,
        "EfficientSets_MNL":       efficient_sets_time_mnl,
        "EfficientSets_MMNL_5PT":  efficient_sets_time_mmnl_5pt,
        "EfficientSets_MMNL_2PT":  efficient_sets_time_mmnl_2pt,
        "DP_MNL":                  dp_mnl_time,
        "DP_MMNL_5PT":             dp_mmnl_5pt_time,
        "DP_MMNL_2PT":             dp_mmnl_2pt_time,
        "ADP_MNL":                 adp_mnl_time,
        "ADP_MMNL_5PT":            adp_mmnl_5pt_time,
        "ADP_MMNL_2PT":            adp_mmnl_2pt_time,
        "ADP_ENV":                 adp_env_time,
    }

    # RL always trains against MNL policy using MNL efficient sets
    training_times_by_step = train_rl(dp_pi=pi_mnl, efficient_sets=efficient_sets_rl)

    for step in C.TOTAL_TIMESTEPS:
        training_times_by_step[step].append(
            {
                "DP_MNL": dp_mnl_time,
                "DP_MMNL_5PT": dp_mmnl_5pt_time,
                "DP_MMNL_2PT": dp_mmnl_2pt_time,
                "ADP_MNL": adp_mnl_time,
                "ADP_MMNL_5PT": adp_mmnl_5pt_time,
                "ADP_MMNL_2PT": adp_mmnl_2pt_time,
                "ADP_ENV": adp_env_time,
            }
        )

    # Delete large DP arrays early to free memory before evaluation
    del v_mnl, v_mmnl_5pt, v_mmnl_2pt
    gc.collect()

    dp_policy_configs = {
        "DP_MNL": {
            "pi": pi_mnl,
            "efficient_sets": efficient_sets_mnl,
        },
        "DP_MMNL_5PT": {
            "pi": pi_mmnl_5pt,
            "efficient_sets": efficient_sets_mmnl_5pt,
        },
        "DP_MMNL_2PT": {
            "pi": pi_mmnl_2pt,
            "efficient_sets": efficient_sets_mmnl_2pt,
        },
    }

    dp_policy_configs.update(
        {
            "ADP_MNL": {
                "pi": pi_adp_mnl,
                "efficient_sets": efficient_sets_mnl,
            },
            "ADP_MMNL_5PT": {
                "pi": pi_adp_mmnl_5pt,
                "efficient_sets": efficient_sets_mmnl_5pt,
            },
            "ADP_MMNL_2PT": {
                "pi": pi_adp_mmnl_2pt,
                "efficient_sets": efficient_sets_mmnl_2pt,
            },
            "ADP_ENV": {
                "pi": pi_adp_env,
                "efficient_sets": efficient_sets_rl,
            },
        }
    )

    evaluation_results_by_step = evaluate_saved_models(
        dp_policy_configs=dp_policy_configs,
        rl_efficient_sets=efficient_sets_rl,
    )

    for step in C.TOTAL_TIMESTEPS:
        table_str = print_evaluation_table(training_times_by_step[step], evaluation_results_by_step[step])
        log_message(f"\n=== {step:,} training timesteps ===\n{table_str}")

    data = {
        "beta_mnl": beta_mnl,
        "lambda_mnl": lambda_mnl,
        "betas_mmnl_5pt": betas_mmnl_5pt,
        "weights_mmnl_5pt": weights_mmnl_5pt,
        "betas_mmnl_2pt": betas_mmnl_2pt,
        "weights_mmnl_2pt": weights_mmnl_2pt,
        "lambda_mmnl_5pt": lambda_mmnl_5pt,
        "lambda_mmnl_2pt": lambda_mmnl_2pt,
        "efficient_sets_mnl": efficient_sets_mnl,
        "efficient_sets_mmnl_5pt": efficient_sets_mmnl_5pt,
        "efficient_sets_mmnl_2pt": efficient_sets_mmnl_2pt,
        "efficient_sets_rl": efficient_sets_rl,
        "pi_mnl": pi_mnl,
        "pi_mmnl_5pt": pi_mmnl_5pt,
        "pi_mmnl_2pt": pi_mmnl_2pt,
        "pi_adp_mnl": pi_adp_mnl,
        "pi_adp_mmnl_5pt": pi_adp_mmnl_5pt,
        "pi_adp_mmnl_2pt": pi_adp_mmnl_2pt,
        "pi_adp_env": pi_adp_env,
        "time_results": time_results,
        "training_times_by_step": training_times_by_step,
        "evaluation_results_by_step": evaluation_results_by_step,
    }

    with open(f"{C.OUTPUT_DIR}/results.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(C.OUTPUT_DIR, "00_exec.log")
    main()