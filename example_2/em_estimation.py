import numpy as np

from buying_probabilities import get_buying_probabilities_by_model


ALL_METHODS = (
	"MNL",
	"MMNL",
	"Probit",
	"MNLrefPrice",
	"MNLConsidSet",
	"NLogit",
)

SENSITIVITY_BETA_TARGETS = {
	"low": -0.0015,
	"high": -0.005,
}


def _safe_log(probability, eps=1e-12):
	return np.log(np.clip(float(probability), eps, 1.0))


def _maximize_scalar_bounded(objective_fn, lower, upper, iterations=70):
	phi = (1.0 + np.sqrt(5.0)) / 2.0
	resphi = 2.0 - phi

	x1 = lower + resphi * (upper - lower)
	x2 = upper - resphi * (upper - lower)
	f1 = objective_fn(x1)
	f2 = objective_fn(x2)

	for _ in range(iterations):
		if f1 < f2:
			lower = x1
			x1 = x2
			f1 = f2
			x2 = upper - resphi * (upper - lower)
			f2 = objective_fn(x2)
		else:
			upper = x2
			x2 = x1
			f2 = f1
			x1 = lower + resphi * (upper - lower)
			f1 = objective_fn(x1)

	if f1 >= f2:
		return x1, f1
	return x2, f2


def estimate_beta_and_arrival_em(
	observations,
	prices,
	model="MNL",
	beta_init=-0.002,
	lambda_init=0.5,
	beta_bounds=(-0.05, 0.0),
	reference_price=None,
	max_iter=500,
	tol=1e-7,
):
	"""Estimate beta and arrival probability via EM for incomplete transaction data.

	Each observation must be a dict with:
	- action_binary: iterable of 0/1 open-product flags
	- purchase_index: product index in [0, n-1] if a purchase occurred, else None

	The EM updates follow equations (14)-(16):
	- E-step: estimate latent arrivals in no-purchase periods
	- M-step: update lambda in closed form and beta by maximizing expected log-likelihood
	"""
	if len(observations) == 0:
		raise ValueError("observations must not be empty")

	prices = np.asarray(prices, dtype=float)
	n_products = len(prices)
	beta_low, beta_high = float(beta_bounds[0]), float(beta_bounds[1])

	if beta_low >= beta_high:
		raise ValueError("beta_bounds must satisfy lower < upper")

	processed = []
	for obs in observations:
		action_binary = np.asarray(obs["action_binary"], dtype=int)
		if len(action_binary) != n_products:
			raise ValueError("Each action_binary must have same length as prices")

		purchase_index = obs.get("purchase_index", None)
		if purchase_index is not None:
			purchase_index = int(purchase_index)
			if purchase_index < 0 or purchase_index >= n_products:
				raise ValueError("purchase_index must be in [0, n_products-1] or None")
			if action_binary[purchase_index] != 1:
				raise ValueError("purchase_index must correspond to an active product")

		processed.append(
			{
				"action_binary": action_binary,
				"purchase_index": purchase_index,
			}
		)

	D = len(processed)
	P = [obs for obs in processed if obs["purchase_index"] is not None]
	Pbar = [obs for obs in processed if obs["purchase_index"] is None]

	lambda_hat = float(np.clip(lambda_init, 1e-6, 1.0 - 1e-6))
	beta_hat = float(np.clip(beta_init, beta_low, beta_high))

	history = []

	action_groups = {}
	for obs in processed:
		key = tuple(int(value) for value in obs["action_binary"])
		group = action_groups.get(key)
		if group is None:
			group = {
				"action_binary": obs["action_binary"],
				"purchase_counts": np.zeros(n_products, dtype=float),
				"no_purchase_count": 0.0,
			}
			action_groups[key] = group

		if obs["purchase_index"] is None:
			group["no_purchase_count"] += 1.0
		else:
			group["purchase_counts"][obs["purchase_index"]] += 1.0

	group_list = list(action_groups.values())

	def expected_log_likelihood(beta_value, lambda_value, a_hats):
		result = 0.0
		for idx, group in enumerate(group_list):
			probs = get_buying_probabilities_by_model(
				group["action_binary"],
				prices,
				beta_value,
				model=model,
				reference_price=reference_price,
				include_outside=True,
			)

			purchase_count_sum = float(np.sum(group["purchase_counts"]))
			if purchase_count_sum > 0:
				result += purchase_count_sum * _safe_log(lambda_value)
				for product_idx in range(n_products):
					count_ij = group["purchase_counts"][product_idx]
					if count_ij > 0:
						result += count_ij * _safe_log(probs[product_idx])

			no_purchase_count = group["no_purchase_count"]
			if no_purchase_count > 0:
				a_hat = a_hats[idx]
				result += no_purchase_count * a_hat * (_safe_log(lambda_value) + _safe_log(probs[-1]))
				result += no_purchase_count * (1.0 - a_hat) * _safe_log(1.0 - lambda_value)

		return float(result)

	for iteration in range(max_iter):
		a_hats = np.zeros(len(group_list), dtype=float)
		for idx, group in enumerate(group_list):
			if group["no_purchase_count"] <= 0:
				continue

			p0 = get_buying_probabilities_by_model(
				group["action_binary"],
				prices,
				beta_hat,
				model=model,
				reference_price=reference_price,
				include_outside=True,
			)[-1]
			numerator = lambda_hat * p0
			denominator = numerator + (1.0 - lambda_hat)
			a_hats[idx] = numerator / max(denominator, 1e-12)

		expected_arrivals_in_no_purchase = 0.0
		for idx, group in enumerate(group_list):
			expected_arrivals_in_no_purchase += group["no_purchase_count"] * a_hats[idx]

		lambda_new = (len(P) + expected_arrivals_in_no_purchase) / float(D)
		lambda_new = float(np.clip(lambda_new, 1e-6, 1.0 - 1e-6))

		def objective_for_beta(beta_candidate):
			score = 0.0
			for idx, group in enumerate(group_list):
				probs = get_buying_probabilities_by_model(
					group["action_binary"],
					prices,
					beta_candidate,
					model=model,
					reference_price=reference_price,
					include_outside=True,
				)

				for product_idx in range(n_products):
					count_ij = group["purchase_counts"][product_idx]
					if count_ij > 0:
						score += count_ij * _safe_log(probs[product_idx])

				if group["no_purchase_count"] > 0:
					score += group["no_purchase_count"] * a_hats[idx] * _safe_log(probs[-1])

			return float(score)

		beta_new, _ = _maximize_scalar_bounded(objective_for_beta, beta_low, beta_high)

		ll_old = expected_log_likelihood(beta_hat, lambda_hat, a_hats)
		ll_new = expected_log_likelihood(beta_new, lambda_new, a_hats)

		history.append(
			{
				"iteration": iteration + 1,
				"beta": beta_new,
				"lambda": lambda_new,
				"expected_log_likelihood": ll_new,
			}
		)

		delta = np.sqrt((beta_new - beta_hat) ** 2 + (lambda_new - lambda_hat) ** 2)
		beta_hat, lambda_hat = float(beta_new), float(lambda_new)

		if delta < tol or abs(ll_new - ll_old) < tol:
			break

	return {
		"beta": beta_hat,
		"lambda": lambda_hat,
		"iterations": len(history),
		"history": history,
		"n_periods": D,
		"n_purchase_periods": len(P),
		"n_no_purchase_periods": len(Pbar),
	}


def estimate_betas_for_all_methods(
	observations,
	prices,
	methods=None,
	beta_init=-0.002,
	lambda_init=0.5,
	beta_bounds=(-0.05, 0.0),
	reference_price=None,
	max_iter=500,
	tol=1e-7,
):
	"""Estimate beta/lambda with EM for every requested choice method.

	Returns a dictionary keyed by method name.
	"""
	if methods is None:
		methods = ALL_METHODS

	results = {}
	for method in methods:
		results[method] = estimate_beta_and_arrival_em(
			observations=observations,
			prices=prices,
			model=method,
			beta_init=beta_init,
			lambda_init=lambda_init,
			beta_bounds=beta_bounds,
			reference_price=reference_price,
			max_iter=max_iter,
			tol=tol,
		)

	return results


def estimate_betas_for_both_sensitivities_all_methods(
	observations_by_sensitivity,
	prices,
	methods=None,
	beta_init_by_sensitivity=None,
	lambda_init=0.5,
	beta_bounds=(-0.05, 0.0),
	reference_price=None,
	max_iter=500,
	tol=1e-7,
):
	"""Estimate beta/lambda for low/high sensitivities across all methods.

	Parameters
	----------
	observations_by_sensitivity : dict
		Expected keys are "low" and "high", each mapping to a list of observations
		compatible with ``estimate_beta_and_arrival_em``.

	Returns
	-------
	dict
		{
		  "low":  {method: em_result, ...},
		  "high": {method: em_result, ...}
		}
		Each ``em_result`` also includes ``target_beta`` for quick comparison.
	"""
	if methods is None:
		methods = ALL_METHODS

	if beta_init_by_sensitivity is None:
		beta_init_by_sensitivity = {
			"low": -0.0015,
			"high": -0.005,
		}

	missing = [key for key in ("low", "high") if key not in observations_by_sensitivity]
	if missing:
		raise ValueError(f"observations_by_sensitivity is missing keys: {missing}")

	results = {}
	for sensitivity in ("low", "high"):
		observations = observations_by_sensitivity[sensitivity]
		beta_init = float(beta_init_by_sensitivity.get(sensitivity, -0.002))

		sensitivity_results = estimate_betas_for_all_methods(
			observations=observations,
			prices=prices,
			methods=methods,
			beta_init=beta_init,
			lambda_init=lambda_init,
			beta_bounds=beta_bounds,
			reference_price=reference_price,
			max_iter=max_iter,
			tol=tol,
		)

		target_beta = SENSITIVITY_BETA_TARGETS[sensitivity]
		for method in sensitivity_results:
			sensitivity_results[method]["target_beta"] = target_beta

		results[sensitivity] = sensitivity_results

	return results
