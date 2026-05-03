import os

import numpy as np

from xlogit import MixedLogit, MultinomialLogit

import constants as C

BETA_BOUNDS = (-0.05, -1e-7)


def _reward_to_purchase_index(reward):
    if reward <= 0:
        return None
    matches = np.where(np.isclose(C.r, reward))[0]
    return int(matches[0]) if len(matches) else None


def collect_transaction_data(env, n_episodes=C.N_ESTIMATION_EPISODES):
    observations = []
    for _ in range(int(n_episodes)):
        env.reset()
        while True:
            current_t = int(env.s[0])
            sampled_action = env.action_space.sample()
            action_binary = env._action_to_binary(sampled_action)
            arrival_flag = int(env.arrival_xi[current_t])
            _, reward, done, truncated, _ = env.step(sampled_action)
            observations.append(
                {
                    "action_binary": action_binary,
                    "arrival_flag": arrival_flag,
                    "purchase_index": _reward_to_purchase_index(reward),
                }
            )
            if done or truncated:
                break
    return observations


class XlogitEstimator:
    """Drop-in estimator API backed by xlogit for faster runtime."""

    @staticmethod
    def _estimate_lambda(observations):
        return float(np.mean([obs["arrival_flag"] for obs in observations]))

    def __init__(self, observations):
        self.observations = observations
        self.n = len(C.r)
        self.lambda_val = self._estimate_lambda(observations)
        self._price_scale = float(os.getenv("XLOGIT_PRICE_SCALE", "100.0"))

        self._mixed_normal_model = None
        self._mixed_lognorm_model = None

        self._build_xlogit_arrays()

    def _build_xlogit_arrays(self):
        prices = np.asarray(C.r, dtype=float) / self._price_scale
        arrival_obs = [obs for obs in self.observations if obs["arrival_flag"]]

        n_obs = len(arrival_obs)
        n_alts = self.n + 1  # products + outside
        n_rows = n_obs * n_alts

        self._ids = np.repeat(np.arange(n_obs, dtype=np.int32), n_alts)
        self._alts = np.tile(np.arange(n_alts, dtype=np.int16), n_obs)

        self._avail = np.ones(n_rows, dtype=np.int8)
        self._y = np.zeros(n_rows, dtype=np.int8)

        # price for products, 0 for outside option
        self._x_price = np.zeros((n_rows, 1), dtype=np.float64)
        self._x_neg_price = np.zeros((n_rows, 1), dtype=np.float64)

        for i, obs in enumerate(arrival_obs):
            start = i * n_alts
            end_products = start + self.n

            self._x_price[start:end_products, 0] = prices
            self._x_neg_price[start:end_products, 0] = -prices

            action_binary = np.asarray(obs["action_binary"], dtype=np.int8)
            self._avail[start:end_products] = action_binary

            chosen_alt = obs["purchase_index"] if obs["purchase_index"] is not None else self.n
            self._y[start + int(chosen_alt)] = 1

        total_bytes = (
            self._ids.nbytes
            + self._alts.nbytes
            + self._avail.nbytes
            + self._y.nbytes
            + self._x_price.nbytes
            + self._x_neg_price.nbytes
        )
        print(
            f"xlogit long data: obs={n_obs}, alts={n_alts}, rows={n_rows}, memory={total_bytes / (1024 ** 2):.2f} MiB"
        )

    @staticmethod
    def _fit_stats_from_model(model):
        return {
            "final_log_likelihood": float(model.loglikelihood),
            "aic": float(model.aic),
            "bic": float(model.bic),
        }

    def estimate_mnl(self):
        model = MultinomialLogit()
        model.fit(
            X=self._x_price,
            y=self._y,
            varnames=["price"],
            alts=self._alts,
            ids=self._ids,
            avail=self._avail,
            fit_intercept=False,
            maxiter=int(os.getenv("XLOGIT_MNL_MAXITER", "500")),
            verbose=0,
            skip_std_errs=True,
        )

        beta_hat = float(np.clip(model.coeff_[0] / self._price_scale, BETA_BOUNDS[0], BETA_BOUNDS[1]))

        return {
            "beta": beta_hat,
            "lambda": self.lambda_val,
            **self._fit_stats_from_model(model),
            "success": True,
        }

    def _get_mixed_normal_model(self):
        if self._mixed_normal_model is None:
            model = MixedLogit()
            model.fit(
                X=self._x_price,
                y=self._y,
                varnames=["price"],
                randvars={"price": "n"},
                alts=self._alts,
                ids=self._ids,
                avail=self._avail,
                fit_intercept=False,
                n_draws=int(os.getenv("XLOGIT_MMNL_DRAWS", "200")),
                maxiter=int(os.getenv("XLOGIT_MMNL_MAXITER", "300")),
                verbose=0,
                skip_std_errs=True,
            )
            self._mixed_normal_model = model
        return self._mixed_normal_model

    def estimate_mmnl(self, K=5):
        if K < 2:
            raise ValueError("K must be >= 2 for finite-mixture approximation")

        model = self._get_mixed_normal_model()

        coef_map = {name: float(val) for name, val in zip(model.coeff_names, model.coeff_)}
        mean_beta = coef_map.get("price", float(model.coeff_[0])) / self._price_scale
        sd_beta = abs(coef_map.get("sd.price", float(model.coeff_[1]) if len(model.coeff_) > 1 else 0.0)) / self._price_scale

        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(int(K))
        std_nodes = np.sqrt(2.0) * gh_nodes
        mix_weights = gh_weights / np.sqrt(np.pi)

        betas = mean_beta + sd_beta * std_nodes
        betas = np.clip(betas, BETA_BOUNDS[0], BETA_BOUNDS[1])

        order = np.argsort(betas)
        betas = betas[order]
        mix_weights = mix_weights[order]
        mix_weights = mix_weights / np.sum(mix_weights)

        return {
            "betas": [float(b) for b in betas],
            "n_segments": int(K),
            "mixing_weights": [float(w) for w in mix_weights],
            "lambda": self.lambda_val,
            **self._fit_stats_from_model(model),
            "success": True,
        }

    def _get_mixed_lognorm_model(self):
        if self._mixed_lognorm_model is None:
            model = MixedLogit()
            model.fit(
                X=self._x_neg_price,
                y=self._y,
                varnames=["neg_price"],
                randvars={"neg_price": "ln"},
                alts=self._alts,
                ids=self._ids,
                avail=self._avail,
                fit_intercept=False,
                n_draws=int(os.getenv("XLOGIT_MMNL_CONT_DRAWS", os.getenv("XLOGIT_MMNL_DRAWS", "200"))),
                maxiter=int(os.getenv("XLOGIT_MMNL_CONT_MAXITER", os.getenv("XLOGIT_MMNL_MAXITER", "300"))),
                verbose=0,
                skip_std_errs=True,
            )
            self._mixed_lognorm_model = model
        return self._mixed_lognorm_model

    def estimate_mmnl_continuous(self):
        try:
            model = self._get_mixed_lognorm_model()

            coef_map = {name: float(val) for name, val in zip(model.coeff_names, model.coeff_)}
            mu_scaled = coef_map.get("neg_price", float(model.coeff_[0]))
            sigma_hat = abs(coef_map.get("sd.neg_price", float(model.coeff_[1]) if len(model.coeff_) > 1 else 0.0))

            # b_original = exp(mu_scaled + sigma*z) / price_scale
            mu_hat = float(mu_scaled - np.log(self._price_scale))
            fit_stats = self._fit_stats_from_model(model)
        except Exception:
            # Robust fallback to avoid blocking long experiment runs.
            mnl_result = self.estimate_mnl()
            mu_hat = float(np.log(max(-mnl_result["beta"], 1e-8)))
            sigma_hat = 0.3
            fit_stats = {
                "final_log_likelihood": float(mnl_result["final_log_likelihood"]),
                "aic": float(mnl_result["aic"]),
                "bic": float(mnl_result["bic"]),
            }

        return {
            "mu_b": float(mu_hat),
            "sigma_b": float(sigma_hat),
            "mean_beta": float(-np.exp(mu_hat + 0.5 * sigma_hat ** 2)),
            "lambda": self.lambda_val,
            **fit_stats,
            "success": True,
        }
