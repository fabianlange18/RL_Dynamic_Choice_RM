import numpy as np
from functools import lru_cache

import constants as C


@lru_cache(maxsize=32)
def _subset_membership_matrix(k):
	"""Return a boolean matrix with all subset memberships for k items.

	Row s corresponds to subset bitmask s, column i indicates whether item i
	is present in subset s.
	"""
	masks = np.arange(1 << k, dtype=np.uint32)
	bits = np.arange(k, dtype=np.uint32)
	return ((masks[:, None] >> bits[None, :]) & 1).astype(bool)

def _mnl_probabilities(action_binary, beta):
	"""Compute MNL choice probabilities via softmax over active products plus outside option.

	For each offered product i, utility contribution is exp(beta * price_i); inactive
	products contribute 0. Probabilities are normalized by:
		denominator = 1 + sum_i exp(beta * price_i),
	where the leading 1 is the outside (no-purchase) option.
	"""
	prices = np.asarray(C.r, dtype=float)
	action_binary = np.asarray(action_binary, dtype=bool)
	utilities = np.where(action_binary, np.exp(beta * prices), 0.0)

	denominator = 1.0 + np.sum(utilities)
	probabilities = np.zeros(len(prices) + 1, dtype=float)
	probabilities[:-1] = utilities / denominator
	probabilities[-1] = 1.0 / denominator
	return probabilities


def _mmnl_probabilities(action_binary, beta=None, segment_betas=None, segment_weights=None, _2pt=False):
	"""Compute mixed-logit probabilities as a weighted mixture of MNL segments.

	If segment_betas is provided, those segment-specific betas are used directly.
	Otherwise, beta must be provided and the default five scaled segments are used.
	"""
	if segment_betas is None:
		if beta is None:
			raise ValueError("Either beta or segment_betas must be provided for MMNL")
		segment_betas = [0.5 * beta, 1.5 * beta] if _2pt else [0.6 * beta, 0.8 * beta, 1.0 * beta, 1.2 * beta, 1.4 * beta]

	segment_betas = np.asarray(segment_betas, dtype=float)
	if segment_betas.size == 0:
		raise ValueError("segment_betas must contain at least one value")

	if segment_weights is None:
		segment_weights = np.full(segment_betas.size, 1.0 / segment_betas.size, dtype=float)
	else:
		segment_weights = np.asarray(segment_weights, dtype=float)
		if segment_weights.shape != segment_betas.shape:
			raise ValueError("segment_weights must have the same shape as segment_betas")
		if np.any(segment_weights < 0):
			raise ValueError("segment_weights must be non-negative")
		weight_sum = float(np.sum(segment_weights))
		if weight_sum <= 0:
			raise ValueError("segment_weights must sum to a positive value")
		segment_weights = segment_weights / weight_sum

	component_probabilities = np.array(
		[_mnl_probabilities(action_binary, b) for b in segment_betas],
		dtype=float,
	)
	mixture_probabilities = np.average(component_probabilities, axis=0, weights=segment_weights)
	return mixture_probabilities


def _mmnl_continuous_probabilities(
	action_binary,
	beta=None,
	mu_b=None,
	sigma_b=None,
	seed=None,
	n_draws=1,
):
	"""Compute continuous-MMNL probabilities via Monte Carlo integration.

	The random coefficient is beta_draw ~ -exp(mu_b + sigma_b * N(0,1)).
	"""
	if mu_b is None:
		if beta is None:
			raise ValueError("Either mu_b or beta must be provided for MMNLcont")
		mu_b = float(np.log(-beta))
	else:
		mu_b = float(mu_b)
	sigma_b = float(sigma_b) if sigma_b is not None else 0.3
	rng = np.random.default_rng(seed)
	draw_betas = -np.exp(mu_b + sigma_b * rng.standard_normal(int(n_draws)))

	component_probabilities = np.array(
		[_mnl_probabilities(action_binary, draw_beta) for draw_beta in draw_betas],
		dtype=float,
	)
	mixture_probabilities = np.mean(component_probabilities, axis=0)
	return mixture_probabilities


def _probit_probabilities(action_binary, beta, seed=None):
	"""Return one-draw Probit outcome probabilities.

	This intentionally performs exactly one utility draw.
	- seed is None: stochastic draw (training)
	- seed is int: deterministic draw for repeatable evaluation
	"""
	prices = np.asarray(C.r, dtype=float)
	active_indices = np.where(action_binary == 1)[0]

	counts = np.zeros(len(prices) + 1, dtype=float)
	if len(active_indices) == 0:
		counts[-1] = 1.0
		return counts

	rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
	outside_utility = float(rng.normal(0.0, 1.0))
	utilities = np.full(len(prices), -np.inf, dtype=float)
	utilities[active_indices] = beta * prices[active_indices] + rng.normal(0.0, 1.0, size=len(active_indices))

	chosen = int(np.argmax(utilities))
	if utilities[chosen] > outside_utility:
		counts[chosen] = 1.0
	else:
		counts[-1] = 1.0

	return counts


def _mnl_ref_price_probabilities(action_binary, beta=None, reference_price=None):
	"""Compute MNL probabilities with a reference-price adjustment.

	Active-product utility exponent is beta*price_i + beta_ref*(reference_price-price_i).
	This shifts utility up when price is below the reference and down when above it.
	Probabilities are then the standard MNL normalization with an outside option.
	"""
	prices = np.asarray(C.r, dtype=float)
	action_binary = np.asarray(action_binary, dtype=bool)

	if reference_price is None:
		reference_price = float(np.mean(prices))

	beta_ref = 0.0025

	ref_adjustment = beta_ref * (reference_price - prices)
	utilities = np.where(action_binary, np.exp(beta * prices + ref_adjustment), 0.0)

	denominator = 1.0 + np.sum(utilities)
	probabilities = np.zeros(len(prices) + 1, dtype=float)
	probabilities[:-1] = utilities / denominator
	probabilities[-1] = 1.0 / denominator
	return probabilities


def _mnl_consideration_probabilities(action_binary, beta=None):
	"""Compute MNL consideration-set probabilities with latent consideration sets.

	`action_binary` encodes the offered set A, not the realized consideration set C.
	For each offered product i, consideration is Bernoulli with probability
	q_i = 1 / (1 + exp((price_i - 300)/80)). We then integrate the piecewise
	MNL probabilities P(j|C) over all C subseteq A.
	"""
	prices = np.asarray(C.r, dtype=float)
	action_binary = np.asarray(action_binary, dtype=int)
	n_products = len(prices)

	offered_indices = np.where(action_binary == 1)[0]
	probabilities = np.zeros(n_products + 1, dtype=float)

	if len(offered_indices) == 0:
		probabilities[-1] = 1.0
		return probabilities

	consideration_prob = 1.0 / (1.0 + np.exp(prices * -0.0125))
	k = len(offered_indices)
	q = consideration_prob[offered_indices]
	one_minus_q = 1.0 - q
	exp_utility_offered = np.exp(beta * prices[offered_indices])

	subset_matrix = _subset_membership_matrix(k)
	subset_matrix_float = subset_matrix.astype(float)

	set_probabilities = np.prod(
		np.where(subset_matrix, q[None, :], one_minus_q[None, :]),
		axis=1,
	)

	denominators = 1.0 + subset_matrix_float.dot(exp_utility_offered)
	weights = set_probabilities / denominators

	offered_probabilities = exp_utility_offered * subset_matrix_float.T.dot(weights)
	probabilities[offered_indices] = offered_probabilities
	probabilities[-1] = float(np.sum(weights))

	return probabilities


def _tmnl_probabilities(action_binary, beta, delta=0.5):
	"""Compute Threshold MNL choice probabilities (Wang 2022).

	Consideration set Psi(S+; delta) = {i in S+ : u_i >= max_j u_j - delta},
	where u_i = beta * price_i and u_0 = 0 for the outside option.
	Choice probabilities are then standard MNL over the consideration set.
	"""
	prices = np.asarray(C.r, dtype=float)
	action_binary = np.asarray(action_binary, dtype=bool)

	offered_indices = np.where(action_binary)[0]
	probabilities = np.zeros(len(prices) + 1, dtype=float)

	if len(offered_indices) == 0:
		probabilities[-1] = 1.0
		return probabilities

	u_products = beta * prices  # shape (n,)
	u_outside = -1

	# max utility over offered products and outside option
	u_max = max(float(np.max(u_products[offered_indices])), u_outside)
	threshold = u_max - delta

	# consideration set mask over offered products
	in_consideration = action_binary & (u_products >= threshold)
	outside_considered = u_outside >= threshold

	exp_u_considered = np.where(in_consideration, np.exp(u_products), 0.0)
	denom = np.sum(exp_u_considered)
	if outside_considered:
		denom += np.exp(u_outside)

	if denom == 0.0:
		probabilities[-1] = 1.0
		return probabilities

	probabilities[:-1] = exp_u_considered / denom
	probabilities[-1] = np.exp(u_outside) / denom if outside_considered else 0.0
	return probabilities


def _nested_logit_probabilities(action_binary, beta=None):
	"""Compute nested-logit probabilities with two nests and nest-specific scales.

	Products are split into nest A (first half) and nest B (second half). Within each
	nest, scaled utilities are exp((beta*price_i)/mu_nest). Nest attractiveness is
	G_nest=(sum scaled utilities)^mu_nest. Overall denominator is 1 + G_A + G_B.
	Product probability = P(nest) * P(product|nest), and outside is 1/denominator.
	"""
	prices = np.asarray(C.r, dtype=float)
	action_binary = np.asarray(action_binary, dtype=int)
	n_products = len(prices)

	nest_a = np.arange(n_products // 2)
	nest_b = np.arange(n_products // 2, n_products)
	mu_a = 0.7
	mu_b = 0.8

	def _nest_terms(indices, mu):
		active = indices[action_binary[indices] == 1]
		if len(active) == 0:
			return 0.0, active, np.array([], dtype=float)

		scaled_utilities = np.exp((beta * prices[active]) / mu)
		return float(np.sum(scaled_utilities)), active, scaled_utilities

	d_a, active_a, scaled_a = _nest_terms(nest_a, mu_a)
	d_b, active_b, scaled_b = _nest_terms(nest_b, mu_b)

	g_a = d_a ** mu_a
	g_b = d_b ** mu_b
	denominator = 1.0 + g_a + g_b

	probabilities = np.zeros(n_products + 1, dtype=float)

	if d_a > 0:
		p_a = g_a / denominator
		probabilities[active_a] = p_a * (scaled_a / d_a)

	if d_b > 0:
		p_b = g_b / denominator
		probabilities[active_b] = p_b * (scaled_b / d_b)

	probabilities[-1] = 1.0 / denominator
	return probabilities


def get_buying_probabilities_by_model(
	action_binary,
	beta,
	model,
	segment_betas=None,
	segment_weights=None,
	mu_b=None,
	sigma_b=None,
	reference_price=None,
	seed=None,
	include_outside=False,
):
	"""Dispatch to the selected demand model and return product-only probabilities.

	Computes full probabilities including outside option for the chosen model, then
	returns only product probabilities (drops the last outside-option entry).

	Additional MMNL variants:
	- `MMNL_5PT`: five-point discrete MMNL
	- `MMNL_2PT`: `segment_betas=[beta_1, beta_2]`, `segment_weights=[w_1, 1-w_1]`
	- `MMNLcont`: `mu_b=mu_b`, `sigma_b=sigma_b`
	"""
	
	match model:
		case "MNL":
			probabilities = _mnl_probabilities(action_binary, beta=beta)
		case "MMNL_5PT":
			probabilities = _mmnl_probabilities(
				action_binary,
				beta=beta,
				segment_betas=segment_betas,
				segment_weights=segment_weights,
			)
		case "Probit":
			probabilities = _probit_probabilities(action_binary, beta=beta, seed=seed)
		case "MNLrefPrice":
			probabilities = _mnl_ref_price_probabilities(action_binary, beta=beta, reference_price=reference_price)
		case "MNLConsidSet":
			probabilities = _mnl_consideration_probabilities(action_binary, beta=beta)
		case "NLogit":
			probabilities = _nested_logit_probabilities(action_binary, beta=beta)
		case "TMNL":
			probabilities = _tmnl_probabilities(action_binary, beta=beta)
		case "MMNL_2PT":
			probabilities = _mmnl_probabilities(
				action_binary,
				beta=beta,
				segment_betas=segment_betas,
				segment_weights=segment_weights,
				_2pt=True,
			)
		case "MMNLcont":
			probabilities = _mmnl_continuous_probabilities(
				action_binary,
				beta=beta,
				mu_b=mu_b,
				sigma_b=sigma_b,
				seed=seed,
			)
		case _:
			raise ValueError(f"Unsupported model '{model}'")

	return probabilities if include_outside else probabilities[:-1]

