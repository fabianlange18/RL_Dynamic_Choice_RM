import numpy as np
from buying_probabilities import get_buying_probabilities_by_model
import config as c
import constants as C
from env_example_2 import TalluriExample2


EM_BETA_INIT = -0.002
EM_LAMBDA_INIT = 0.5
EM_BETA_BOUNDS = (-0.05, -1e-6)
EM_MAX_ITER = 200
EM_TOL = 1e-7


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


def estimate_beta_and_arrival_em(observations):
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

	n_products = len(C.r)
	beta_low, beta_high = float(EM_BETA_BOUNDS[0]), float(EM_BETA_BOUNDS[1])

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

	lambda_hat = float(np.clip(EM_LAMBDA_INIT, 1e-6, 1.0 - 1e-6))
	beta_hat = float(np.clip(EM_BETA_INIT, beta_low, beta_high))

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
				C.r,
				beta_value,
				model=c.OPT_MODEL,
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

	for iteration in range(EM_MAX_ITER):
		a_hats = np.zeros(len(group_list), dtype=float)
		for idx, group in enumerate(group_list):
			if group["no_purchase_count"] <= 0:
				continue

			p0 = get_buying_probabilities_by_model(
				group["action_binary"],
				C.r,
				beta_hat,
				model=c.OPT_MODEL,
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
					C.r,
					beta_candidate,
					model=c.OPT_MODEL,
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

		if delta < EM_TOL or abs(ll_new - ll_old) < EM_TOL:
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


def _reward_to_purchase_index(reward):
	if reward <= 0:
		return None

	matching = np.where(np.isclose(C.r, reward))[0]
	if len(matching) == 0:
		return None
	return int(matching[0])


def collect_incomplete_transaction_data(env, n_episodes=C.N_ESTIMATION_EPISODES):
	"""Collect incomplete transaction observations from random interaction."""
	observations = []

	for _ in range(int(n_episodes)):
		env.reset()

		while True:
			sampled_action = env.action_space.sample()
			action_binary = env._action_to_binary(sampled_action)

			_, reward, done, truncated, _ = env.step(sampled_action)

			observations.append(
				{
					"action_binary": action_binary,
					"purchase_index": _reward_to_purchase_index(reward),
				}
			)

			if done or truncated:
				break

	return observations


def run_em():
	"""Collect N episodes of incomplete transactions and run EM estimation.

	Returns a dictionary with both collected observations and EM output.
	"""

	env = TalluriExample2(efficient_sets=None)

	try:
		observations = collect_incomplete_transaction_data(env)
		em_result = estimate_beta_and_arrival_em(observations)
		return {
			"observations": observations,
			"em_result": em_result,
			"n_estimation_episodes": C.N_ESTIMATION_EPISODES,
		}
	finally:
		env.close()