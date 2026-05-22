import os

import numpy as np
from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalLogit
from choice_learn.models.latent_class_mnl import LatentClassConditionalLogit

import src.constants as C
from src.calc.estimation_xlogit import collect_transaction_data


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)


def _flatten_scalar_weight(weight_tensor):
    weight_array = np.asarray(weight_tensor.numpy(), dtype=float).reshape(-1)
    if weight_array.size != 1:
        raise ValueError(f"Expected a single shared coefficient, got shape {weight_tensor.shape}")
    return float(weight_array[0])


class ChoiceLearnEstimator:
    """Direct finite-mixture estimator API backed by choice-learn latent classes."""

    @staticmethod
    def _estimate_lambda(observations):
        return float(np.mean([obs["arrival_flag"] for obs in observations]))

    def __init__(self, observations):
        self.observations = observations
        self.n = len(C.r)
        self.lambda_val = self._estimate_lambda(observations)
        self._price_scale = float(os.getenv("CHOICE_LEARN_PRICE_SCALE", "100.0"))
        self._lbfgs_tolerance = float(os.getenv("CHOICE_LEARN_LBFGS_TOLERANCE", "1e-8"))
        self._mnl_optimizer = os.getenv("CHOICE_LEARN_MNL_OPTIMIZER", "lbfgs")
        self._mnl_epochs = int(os.getenv("CHOICE_LEARN_MNL_EPOCHS", "200"))
        self._mmnl_optimizer = os.getenv("CHOICE_LEARN_MMNL_OPTIMIZER", "lbfgs")
        self._mmnl_epochs = int(os.getenv("CHOICE_LEARN_MMNL_EPOCHS", "500"))
        self._mmnl_fit_method = os.getenv("CHOICE_LEARN_MMNL_FIT_METHOD", "mle")
        self._verbose = int(os.getenv("CHOICE_LEARN_VERBOSE", "0"))

        self._dataset = None
        self._build_choice_dataset()

    def _build_choice_dataset(self):
        prices = np.asarray(C.r, dtype=np.float32) / self._price_scale
        arrival_obs = [obs for obs in self.observations if obs["arrival_flag"]]

        if not arrival_obs:
            raise ValueError("No arrival observations available for choice-learn estimation.")

        n_obs = len(arrival_obs)
        n_alts = self.n + 1

        choices = np.full(n_obs, self.n, dtype=np.int32)
        available = np.ones((n_obs, n_alts), dtype=np.float32)
        items_features = np.zeros((n_obs, n_alts, 1), dtype=np.float32)
        items_features[:, : self.n, 0] = prices

        for obs_index, obs in enumerate(arrival_obs):
            action_binary = np.asarray(obs["action_binary"], dtype=np.float32)
            available[obs_index, : self.n] = action_binary
            chosen_alt = obs["purchase_index"]
            if chosen_alt is not None:
                choices[obs_index] = int(chosen_alt)

        self._dataset = ChoiceDataset(
            choices=choices,
            items_features_by_choice=items_features,
            available_items_by_choice=available,
            items_features_by_choice_names=(["price"],),
        )

        total_bytes = choices.nbytes + available.nbytes + items_features.nbytes
        print(
            "choice-learn data: "
            f"obs={n_obs}, alts={n_alts}, memory={total_bytes / (1024 ** 2):.2f} MiB"
        )

    @staticmethod
    def _fit_stats_from_nll(avg_negative_log_likelihood, n_obs, n_parameters):
        final_log_likelihood = -float(avg_negative_log_likelihood) * int(n_obs)
        aic = 2.0 * float(n_parameters) - 2.0 * final_log_likelihood
        bic = float(n_parameters) * np.log(int(n_obs)) - 2.0 * final_log_likelihood
        return {
            "final_log_likelihood": final_log_likelihood,
            "aic": float(aic),
            "bic": float(bic),
        }

    def estimate_mnl(self):
        model = ConditionalLogit(
            coefficients={"price": "constant"},
            optimizer=self._mnl_optimizer,
            epochs=self._mnl_epochs,
            add_exit_choice=False,
            lbfgs_tolerance=self._lbfgs_tolerance,
        )
        model.fit(self._dataset, verbose=self._verbose)

        avg_nll = float(model.evaluate(self._dataset).numpy())
        beta_hat = _flatten_scalar_weight(model.trainable_weights[0]) / self._price_scale
        beta_hat = float(np.clip(beta_hat, C.ESTIMATION_BETA_BOUNDS[0], C.ESTIMATION_BETA_BOUNDS[1]))

        return {
            "beta": beta_hat,
            "lambda": self.lambda_val,
            **self._fit_stats_from_nll(avg_nll, len(self._dataset), 1),
            "success": True,
        }

    def estimate_mmnl(self, K=5):
        if int(K) < 2:
            raise ValueError("ChoiceLearnEstimator.estimate_mmnl requires at least 2 latent classes.")

        model = LatentClassConditionalLogit(
            n_latent_classes=int(K),
            fit_method=self._mmnl_fit_method,
            coefficients={"price": "constant"},
            optimizer=self._mmnl_optimizer,
            epochs=self._mmnl_epochs,
            add_exit_choice=False,
            lbfgs_tolerance=self._lbfgs_tolerance,
        )
        model.fit(self._dataset, verbose=self._verbose)

        avg_nll = float(model.evaluate(self._dataset).numpy())

        latent_logits = np.asarray(model.latent_logits.numpy(), dtype=float).reshape(-1)
        mixing_weights = _softmax(np.concatenate([latent_logits, np.zeros(1, dtype=float)]))

        betas = np.asarray(
            [
                np.clip(
                    _flatten_scalar_weight(latent_model.trainable_weights[0]) / self._price_scale,
                    C.ESTIMATION_BETA_BOUNDS[0],
                    C.ESTIMATION_BETA_BOUNDS[1],
                )
                for latent_model in model.models
            ],
            dtype=float,
        )

        order = np.argsort(betas)
        betas = betas[order]
        mixing_weights = mixing_weights[order]
        mixing_weights = mixing_weights / np.sum(mixing_weights)

        n_parameters = int(K) + (int(K) - 1)

        return {
            "betas": [float(beta) for beta in betas],
            "n_segments": int(K),
            "mixing_weights": [float(weight) for weight in mixing_weights],
            "lambda": self.lambda_val,
            **self._fit_stats_from_nll(avg_nll, len(self._dataset), n_parameters),
            "success": True,
        }


if __name__ == "__main__":
    from src.talluri_env import TalluriExample2

    test_episodes = int(os.getenv("CHOICE_LEARN_TEST_EPISODES", "5"))
    environment = TalluriExample2(efficient_sets=None)

    try:
        test_observations = collect_transaction_data(environment, n_episodes=test_episodes)
    finally:
        environment.close()

    test_estimator = ChoiceLearnEstimator(test_observations)

    print("Running choice-learn smoke test")
    print(f"Episodes: {test_episodes}")
    print("MNL:")
    print(test_estimator.estimate_mnl())
    print("MMNL 2PT:")
    print(test_estimator.estimate_mmnl(K=2))
    print("MMNL 5PT:")
    print(test_estimator.estimate_mmnl(K=5))