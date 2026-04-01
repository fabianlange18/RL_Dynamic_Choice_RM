import numpy as np
from buying_probabilities import get_buying_probabilities_by_model
import constants as C
from env_example_2 import TalluriExample2


EM_BETA_INIT = -0.002
EM_LAMBDA_INIT = 0.5
EM_BETA_BOUNDS = (-0.05, -1e-6)
EM_MAX_ITER = 200
EM_TOL = 1e-7

MMNL_N_SEGMENTS = 5


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


def _validate_and_process_observations(observations, n_products):
	if len(observations) == 0:
		raise ValueError("observations must not be empty")

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

	return processed


def _build_action_groups(processed, n_products):
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

	return list(action_groups.values())


def _stable_log_mnl_denom(beta_value, active_prices):
	if len(active_prices) == 0:
		return 0.0

	utilities = beta_value * active_prices
	shift = max(0.0, float(np.max(utilities)))
	sum_exp_shifted = np.exp(-shift) + float(np.sum(np.exp(utilities - shift)))
	return float(shift + np.log(sum_exp_shifted))


def estimate_mnl_em(observations, keep_history=False):
	"""Estimate beta and arrival probability via EM for incomplete transaction data.

	Each observation must be a dict with:
	- action_binary: iterable of 0/1 open-product flags
	- purchase_index: product index in [0, n-1] if a purchase occurred, else None

	The EM updates follow equations (14)-(16):
	- E-step: estimate latent arrivals in no-purchase periods
	- M-step: update lambda in closed form and beta by maximizing expected log-likelihood
	"""
	n_products = len(C.r)
	beta_low, beta_high = float(EM_BETA_BOUNDS[0]), float(EM_BETA_BOUNDS[1])

	if beta_low >= beta_high:
		raise ValueError("beta_bounds must satisfy lower < upper")

	processed = _validate_and_process_observations(observations, n_products)

	D = len(processed)
	n_purchase_periods = int(sum(1 for obs in processed if obs["purchase_index"] is not None))
	n_no_purchase_periods = int(D - n_purchase_periods)

	lambda_hat = float(np.clip(EM_LAMBDA_INIT, 1e-6, 1.0 - 1e-6))
	beta_hat = float(np.clip(EM_BETA_INIT, beta_low, beta_high))

	history = []
	group_list = _build_action_groups(processed, n_products)

	def expected_log_likelihood(beta_value, lambda_value, a_hats):
		result = 0.0
		for idx, group in enumerate(group_list):
			probs = get_buying_probabilities_by_model(
				group["action_binary"],
				C.r,
				beta_value,
				model="MNL",
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
				model="MNL",
				include_outside=True,
			)[-1]
			numerator = lambda_hat * p0
			denominator = numerator + (1.0 - lambda_hat)
			a_hats[idx] = numerator / max(denominator, 1e-12)

		expected_arrivals_in_no_purchase = 0.0
		for idx, group in enumerate(group_list):
			expected_arrivals_in_no_purchase += group["no_purchase_count"] * a_hats[idx]

		lambda_new = (n_purchase_periods + expected_arrivals_in_no_purchase) / float(D)
		lambda_new = float(np.clip(lambda_new, 1e-6, 1.0 - 1e-6))

		def objective_for_beta(beta_candidate):
			score = 0.0
			for idx, group in enumerate(group_list):
				probs = get_buying_probabilities_by_model(
					group["action_binary"],
					C.r,
					beta_candidate,
					model="MNL",
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

		if keep_history:
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

	n_iterations = iteration + 1

	return {
		"beta": beta_hat,
		"lambda": lambda_hat,
		"iterations": n_iterations,
		"history": history,
		"n_periods": D,
		"n_purchase_periods": n_purchase_periods,
		"n_no_purchase_periods": n_no_purchase_periods,
	}



def estimate_mmnl_em(observations, keep_history=False):
	"""Estimate finite-mixture MMNL parameters via EM for incomplete transaction data.

	Fits MMNL_N_SEGMENTS MNL components with equal mixing weights (1/K each).
	Parameters estimated: beta_1, ..., beta_K, lambda.

	Two layers of latent variables:
	- a_t in {0,1}: customer arrival for no-purchase periods (same as scalar EM)
	- z_t in {1,...,K}: segment membership for each observation

	E-step:
	  For each group and segment k, compute posterior segment responsibilities:
	    w_{g,j,k}  = P_MNL(j | A_g, beta_k) / sum_k' P_MNL(j | A_g, beta_k')  (purchase)
	    w_{g,0,k}  = P_MNL_outside(A_g, beta_k) / sum_k' P_MNL_outside(A_g, beta_k') (no-purch)
	  Arrival estimate for no-purchase groups (same formula as scalar EM):
	    a_hat_g = lambda * mixture_p0_g / (lambda * mixture_p0_g + (1 - lambda))

	M-step:
	  lambda* = (|P| + sum_g no_purch_g * a_hat_g) / |D|  (closed form)
	  beta_k*  = argmax_beta sum_g [ sum_j eff_purch_{g,j,k} * log P_MNL(j|A_g,beta)
	                                  + eff_nopur_{g,k} * log P_MNL_outside(A_g,beta) ]
	             solved independently per k via golden-section search.
	"""
	n_products = len(C.r)
	prices = np.asarray(C.r, dtype=float)
	beta_low, beta_high = float(EM_BETA_BOUNDS[0]), float(EM_BETA_BOUNDS[1])
	K = int(MMNL_N_SEGMENTS)

	if beta_low >= beta_high:
		raise ValueError("beta_bounds must satisfy lower < upper")

	processed = _validate_and_process_observations(observations, n_products)

	D = len(processed)
	n_purchase_periods = int(sum(1 for obs in processed if obs["purchase_index"] is not None))

	group_list = _build_action_groups(processed, n_products)
	n_groups = len(group_list)
	group_active_indices = [np.where(group["action_binary"] == 1)[0] for group in group_list]
	group_active_prices = [prices[active_indices] for active_indices in group_active_indices]

	# Initialise: spread betas evenly across feasible interval, lambda from constant
	betas = np.linspace(beta_low, beta_high, K)
	lambda_hat = float(np.clip(EM_LAMBDA_INIT, 1e-6, 1.0 - 1e-6))

	history = []

	for iteration in range(EM_MAX_ITER):
		# -- E-step --------------------------------------------------
		# Precompute per-group MMNL probabilities:
		# - segment_probs[g] has shape (K, n_products + 1)
		# - mixture_probs[g] has shape (n_products + 1)
		group_segment_probs = []
		group_mixture_probs = []
		for group in group_list:
			segment_probs, mixture_probs = get_buying_probabilities_by_model(
				group["action_binary"],
				C.r,
				beta=None,
				model="MMNL",
				segment_betas=betas,
				return_components=True,
				include_outside=True,
			)
			group_segment_probs.append(segment_probs)
			group_mixture_probs.append(mixture_probs)

		a_hats = np.zeros(n_groups, dtype=float)
		# eff_purchase_counts[g] shape (K, n_products): purchase_count * responsibility
		eff_purchase_counts = [np.zeros((K, n_products), dtype=float) for _ in group_list]
		# eff_nopur_counts[g, k]: no_purchase_count * a_hat * segment_responsibility
		eff_nopur_counts = np.zeros((n_groups, K), dtype=float)
		expected_arrivals_in_no_purchase = 0.0

		for idx, (group, probs_k, probs_mix) in enumerate(zip(group_list, group_segment_probs, group_mixture_probs)):
			# No-purchase: arrival estimate using mixture outside probability
			if group["no_purchase_count"] > 0:
				mixture_p0 = float(probs_mix[-1])
				numerator = lambda_hat * mixture_p0
				denominator = numerator + (1.0 - lambda_hat)
				a_hats[idx] = numerator / max(denominator, 1e-12)
				expected_arrivals_in_no_purchase += group["no_purchase_count"] * a_hats[idx]

				# Segment responsibilities for arrivals in no-purchase periods
				p0_per_seg = probs_k[:, -1]  # shape (K,)
				seg_sum = float(np.sum(p0_per_seg))
				w_nopur = p0_per_seg / seg_sum if seg_sum > 1e-12 else np.full(K, 1.0 / K)
				eff_nopur_counts[idx] = group["no_purchase_count"] * a_hats[idx] * w_nopur

			# Purchase: segment responsibilities only over active products.
			active_indices = group_active_indices[idx]
			if len(active_indices) > 0:
				active_counts = group["purchase_counts"][active_indices]
				positive_mask = active_counts > 0
				if np.any(positive_mask):
					active_purchase_indices = active_indices[positive_mask]
					positive_counts = active_counts[positive_mask]

					pj_matrix = probs_k[:, active_purchase_indices]  # shape (K, m)
					seg_sums = np.sum(pj_matrix, axis=0)

					weights = np.empty_like(pj_matrix)
					valid = seg_sums > 1e-12
					if np.any(valid):
						weights[:, valid] = pj_matrix[:, valid] / seg_sums[valid]
					if np.any(~valid):
						weights[:, ~valid] = 1.0 / K

					eff_purchase_counts[idx][:, active_purchase_indices] = weights * positive_counts[None, :]

		# Build weighted sufficient statistics for the beta updates.
		eff_purchase_sums = np.array(
			[np.sum(eff_purchase_counts[idx], axis=1) for idx in range(n_groups)],
			dtype=float,
		)
		eff_purchase_weighted_prices = np.array(
			[eff_purchase_counts[idx].dot(prices) for idx in range(n_groups)],
			dtype=float,
		)

		# -- M-step --------------------------------------------------
		# Update lambda (closed form, same as scalar EM)
		total_purchase_count = float(sum(np.sum(g["purchase_counts"]) for g in group_list))
		lambda_new = (total_purchase_count + expected_arrivals_in_no_purchase) / float(D)
		lambda_new = float(np.clip(lambda_new, 1e-6, 1.0 - 1e-6))

		# Update each beta_k independently via golden-section search
		betas_new = np.empty(K, dtype=float)
		for k in range(K):
			def objective_for_beta_k(beta_candidate, k=k):
				score = 0.0
				beta_value = float(beta_candidate)
				for idx, active_prices in enumerate(group_active_prices):
					log_denom = _stable_log_mnl_denom(beta_value, active_prices)

					purchase_weight_sum = eff_purchase_sums[idx, k]
					if purchase_weight_sum > 0:
						score += beta_value * eff_purchase_weighted_prices[idx, k]
						score -= purchase_weight_sum * log_denom

					if eff_nopur_counts[idx, k] > 0:
						score -= eff_nopur_counts[idx, k] * log_denom
				return float(score)

			betas_new[k], _ = _maximize_scalar_bounded(objective_for_beta_k, beta_low, beta_high)

		# -- Convergence check ----------------------------------------
		delta = float(np.sqrt(np.sum((betas_new - betas) ** 2) + (lambda_new - lambda_hat) ** 2))
		betas = betas_new.copy()
		lambda_hat = lambda_new

		if keep_history:
			history.append({
				"iteration": iteration + 1,
				"betas": betas.tolist(),
				"lambda": lambda_hat,
			})

		if delta < EM_TOL:
			break

	n_iterations = iteration + 1

	return {
		"betas": sorted(float(b) for b in betas),
		"lambda": lambda_hat,
		"n_segments": K,
		"mixing_weights": [1.0 / K] * K,
		"iterations": n_iterations,
		"history": history,
		"n_periods": D,
		"n_purchase_periods": n_purchase_periods,
		"n_no_purchase_periods": int(D - n_purchase_periods),
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


def run_em(model="MNL", keep_history=False):
	"""Collect N episodes of incomplete transactions and run EM estimation.

	Returns a dictionary with both collected observations and EM output.
	"""

	env = TalluriExample2(efficient_sets=None)

	try:
		observations = collect_incomplete_transaction_data(env)
		em_result_mnl = (
			estimate_mnl_em(observations, keep_history=keep_history)
			if model != "MMNL"
			else estimate_mmnl_em(observations, keep_history=keep_history)
		)

		return {
			"observations": observations,
			"em_result": em_result_mnl,
			"n_estimation_episodes": C.N_ESTIMATION_EPISODES,
		}
	finally:
		env.close()