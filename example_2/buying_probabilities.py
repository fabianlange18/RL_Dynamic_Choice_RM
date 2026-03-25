import numpy as np
from functools import lru_cache


@lru_cache(maxsize=32)
def _subset_membership_matrix(k):
	"""Return a boolean matrix with all subset memberships for k items.

	Row s corresponds to subset bitmask s, column i indicates whether item i
	is present in subset s.
	"""
	masks = np.arange(1 << k, dtype=np.uint32)
	bits = np.arange(k, dtype=np.uint32)
	return ((masks[:, None] >> bits[None, :]) & 1).astype(bool)

def _mnl_probabilities(action_binary, prices, beta):
	"""Compute MNL choice probabilities via softmax over active products plus outside option.

	For each offered product i, utility contribution is exp(beta * price_i); inactive
	products contribute 0. Probabilities are normalized by:
		denominator = 1 + sum_i exp(beta * price_i),
	where the leading 1 is the outside (no-purchase) option.
	"""
	prices = np.asarray(prices, dtype=float)
	action_binary = np.asarray(action_binary, dtype=bool)
	utilities = np.where(action_binary, np.exp(beta * prices), 0.0)

	denominator = 1.0 + np.sum(utilities)
	probabilities = np.zeros(len(prices) + 1, dtype=float)
	probabilities[:-1] = utilities / denominator
	probabilities[-1] = 1.0 / denominator
	return probabilities


def _mmnl_probabilities(action_binary, prices, beta):
	"""Compute mixed-logit probabilities as an average of two MNL segments.

	Uses five taste draws (0.6*beta, 0.8*beta, 1.0*beta, 1.2*beta, 1.4*beta),
	computes MNL probabilities for each, then returns their equal-weight mixture.
	"""
	beta_candidates = [0.6 * beta, 0.8 * beta, 1.0 * beta, 1.2 * beta, 1.4 * beta]
	return np.mean(
		[_mnl_probabilities(action_binary, prices, b) for b in beta_candidates],
		axis=0,
	)


def _probit_probabilities(action_binary, prices, beta, seed=None):
	"""Return one-draw Probit outcome probabilities.

	This intentionally performs exactly one utility draw.
	- seed is None: stochastic draw (training)
	- seed is int: deterministic draw for repeatable evaluation
	"""
	prices = np.asarray(prices, dtype=float)
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


def _mnl_ref_price_probabilities(
	action_binary,
	prices,
	beta=None,
	reference_price=None,
):
	"""Compute MNL probabilities with a reference-price adjustment.

	Active-product utility exponent is beta*price_i + beta_ref*(reference_price-price_i).
	This shifts utility up when price is below the reference and down when above it.
	Probabilities are then the standard MNL normalization with an outside option.
	"""
	prices = np.asarray(prices, dtype=float)
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


def _mnl_consideration_probabilities(action_binary, prices, beta=None):
	"""Compute MNL consideration-set probabilities with latent consideration sets.

	`action_binary` encodes the offered set A, not the realized consideration set C.
	For each offered product i, consideration is Bernoulli with probability
	q_i = 1 / (1 + exp((price_i - 300)/80)). We then integrate the piecewise
	MNL probabilities P(j|C) over all C subseteq A.
	"""
	prices = np.asarray(prices, dtype=float)
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


def _nested_logit_probabilities(action_binary, prices, beta=None):
	"""Compute nested-logit probabilities with two nests and nest-specific scales.

	Products are split into nest A (first half) and nest B (second half). Within each
	nest, scaled utilities are exp((beta*price_i)/mu_nest). Nest attractiveness is
	G_nest=(sum scaled utilities)^mu_nest. Overall denominator is 1 + G_A + G_B.
	Product probability = P(nest) * P(product|nest), and outside is 1/denominator.
	"""
	prices = np.asarray(prices, dtype=float)
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
	prices,
	beta,
	model,
	reference_price=None,
	include_outside=False,
	probit_seed=None,
):
	"""Dispatch to the selected demand model and return product-only probabilities.

	Computes full probabilities including outside option for the chosen model, then
	returns only product probabilities (drops the last outside-option entry).
	"""
	
	match model:
		case "MNL":
			probabilities = _mnl_probabilities(action_binary, prices, beta=beta)
		case "MMNL":
			probabilities = _mmnl_probabilities(action_binary, prices, beta=beta)
		case "Probit":
			probabilities = _probit_probabilities(action_binary, prices, beta=beta, seed=probit_seed)
		case "MNLrefPrice":
			probabilities = _mnl_ref_price_probabilities(action_binary, prices, beta=beta, reference_price=reference_price)
		case "MNLConsidSet":
			probabilities = _mnl_consideration_probabilities(action_binary, prices, beta=beta)
		case "NLogit":
			probabilities = _nested_logit_probabilities(action_binary, prices, beta=beta)
		case _:
			raise ValueError(f"Unsupported model '{model}'")

	return probabilities if include_outside else probabilities[:-1]

