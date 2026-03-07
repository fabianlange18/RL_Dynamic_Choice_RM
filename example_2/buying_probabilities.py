import numpy as np

def _mnl_probabilities(action_binary, prices, beta):
	"""Compute MNL choice probabilities via softmax over active products plus outside option.

	For each offered product i, utility contribution is exp(beta * price_i); inactive
	products contribute 0. Probabilities are normalized by:
		denominator = 1 + sum_i exp(beta * price_i),
	where the leading 1 is the outside (no-purchase) option.
	"""
	prices = np.asarray(prices, dtype=float)
	utilities = np.zeros(len(prices), dtype=float)

	for idx in range(len(prices)):
		if action_binary[idx] == 1:
			utilities[idx] = np.exp(beta * prices[idx])

	denominator = 1.0 + np.sum(utilities)
	probabilities = np.zeros(len(prices) + 1, dtype=float)
	probabilities[:-1] = utilities / denominator
	probabilities[-1] = 1.0 / denominator
	return probabilities


def _mmnl_probabilities(action_binary, prices, beta):
	"""Compute mixed-logit probabilities as an average of two MNL segments.

	Uses two taste draws (0.7*beta and 1.3*beta), computes MNL probabilities for each,
	then returns their equal-weight (0.5/0.5) mixture.
	"""
	beta_candidates = np.array([0.7 * beta, 1.3 * beta], dtype=float)

	mix_probabilities = np.zeros(len(prices) + 1, dtype=float)
	for beta_draw in beta_candidates:
		mix_probabilities += 0.5 * _mnl_probabilities(action_binary, prices, beta_draw)

	return mix_probabilities


def _probit_probabilities(action_binary, prices, beta):
	"""Estimate Probit probabilities by Monte Carlo simulation.

	Across 100 draws, product utility is beta*price_i + Normal(0,1) for active items,
	and outside utility is Normal(0,1). The chosen alternative is the max utility; if
	the best product beats outside utility it gets the count, otherwise outside gets it.
	Final probabilities are normalized choice counts.
	"""
	prices = np.asarray(prices, dtype=float)
	rng = np.random.default_rng(12345)

	counts = np.zeros(len(prices) + 1, dtype=float)
	active_indices = np.where(action_binary == 1)[0]

	if len(active_indices) == 0:
		counts[-1] = 1.0
		return counts

	for _ in range(int(100)):
		outside_utility = rng.normal(0.0, 1.0)

		utilities = np.full(len(prices), -np.inf, dtype=float)
		utilities[active_indices] = beta * prices[active_indices] + rng.normal(0.0, 1.0, size=len(active_indices))

		chosen = int(np.argmax(utilities))
		if utilities[chosen] > outside_utility:
			counts[chosen] += 1.0
		else:
			counts[-1] += 1.0

	return counts / max(1.0, float(np.sum(counts)))


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

	if reference_price is None:
		reference_price = float(np.mean(prices))

	beta_ref = 0.0025

	utilities = np.zeros(len(prices), dtype=float)
	for idx in range(len(prices)):
		if action_binary[idx] == 1:
			ref_adjustment = beta_ref * (reference_price - prices[idx])
			utilities[idx] = np.exp(beta * prices[idx] + ref_adjustment)

	denominator = 1.0 + np.sum(utilities)
	probabilities = np.zeros(len(prices) + 1, dtype=float)
	probabilities[:-1] = utilities / denominator
	probabilities[-1] = 1.0 / denominator
	return probabilities


def _mnl_consideration_probabilities(action_binary, prices, beta=None):
	"""Compute MNL probabilities weighted by consideration-set likelihood.

	Each product gets a consideration weight sigmoid((300-price_i)/80), equivalently
	1/(1+exp((price_i-300)/80)). Active utility is this weight times exp(beta*price_i),
	then probabilities are normalized with the outside option as in MNL.
	"""
	prices = np.asarray(prices, dtype=float)

	consideration_prob = 1.0 / (1.0 + np.exp((prices - 300.0) / 80.0))

	utilities = np.zeros(len(prices), dtype=float)
	for idx in range(len(prices)):
		if action_binary[idx] == 1:
			utilities[idx] = consideration_prob[idx] * np.exp(beta * prices[idx])

	denominator = 1.0 + np.sum(utilities)
	probabilities = np.zeros(len(prices) + 1, dtype=float)
	probabilities[:-1] = utilities / denominator
	probabilities[-1] = 1.0 / denominator
	return probabilities


def _nested_logit_probabilities(action_binary, prices, beta=None):
	"""Compute nested-logit probabilities with two nests and nest-specific scales.

	Products are split into nest A (first half) and nest B (second half). Within each
	nest, scaled utilities are exp((beta*price_i)/mu_nest). Nest attractiveness is
	G_nest=(sum scaled utilities)^mu_nest. Overall denominator is 1 + G_A + G_B.
	Product probability = P(nest) * P(product|nest), and outside is 1/denominator.
	"""
	prices = np.asarray(prices, dtype=float)
	n_products = len(prices)

	nest_a = np.array([idx for idx in range(n_products) if idx < n_products // 2], dtype=int)
	nest_b = np.array([idx for idx in range(n_products) if idx >= n_products // 2], dtype=int)
	mu_a = 0.7
	mu_b = 0.8

	def _nest_terms(indices, mu):
		active = np.array([idx for idx in indices if action_binary[idx] == 1], dtype=int)
		if len(active) == 0:
			return 0.0, active, np.array([], dtype=float)

		scaled_utilities = np.exp((beta * prices[active]) / mu)
		return float(np.sum(scaled_utilities)), active, scaled_utilities

	d_a, active_a, scaled_a = _nest_terms(nest_a, mu_a)
	d_b, active_b, scaled_b = _nest_terms(nest_b, mu_b)

	g_a = d_a ** mu_a if d_a > 0 else 0.0
	g_b = d_b ** mu_b if d_b > 0 else 0.0
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
			probabilities = _probit_probabilities(action_binary, prices, beta=beta)
		case "MNLrefPrice":
			probabilities = _mnl_ref_price_probabilities(action_binary, prices, beta=beta, reference_price=reference_price)
		case "MNLConsidSet":
			probabilities = _mnl_consideration_probabilities(action_binary, prices, beta=beta)
		case "NLogit":
			probabilities = _nested_logit_probabilities(action_binary, prices, beta=beta)
		case _:
			raise ValueError(f"Unsupported model '{model}'")

	return probabilities if include_outside else probabilities[:-1]

