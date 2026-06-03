from src.demand.mnl import mnl_probabilities
from src.demand.mmnl import mmnl_probabilities
from src.demand.probit import probit_probabilities, multivariate_probit_probabilities
from src.demand.mnl_reference_price import mnl_reference_price_probabilities
from src.demand.mnl_consideration_set import mnl_consideration_set_probabilities
from src.demand.tmnl import tmnl_probabilities
from src.demand.nested_logit import nested_logit_probabilities
import src.constants as C


def get_buying_probabilities_by_model(
	action_binary,
	beta,
	model,
	segment_betas=None,
	segment_weights=None,
	reference_price=None,
	seed=None,
	include_outside=False,
):
	"""Dispatch to the selected demand model and return product-only probabilities."""

	match model:
		case "MNL":
			probabilities = mnl_probabilities(action_binary, beta=beta)
		case "MMNL_2PT":
			if segment_betas is None:
				segment_betas = C.MMNL_2PT_BETA_MULTIPLIERS * beta
			probabilities = mmnl_probabilities(
				action_binary,
				segment_betas=segment_betas,
				segment_weights=segment_weights,
			)
		case "MMNL_5PT":
			if segment_betas is None:
				segment_betas = C.MMNL_5PT_BETA_MULTIPLIERS * beta
			probabilities = mmnl_probabilities(
				action_binary,
				segment_betas=segment_betas,
				segment_weights=segment_weights,
			)
		case "Probit":
			probabilities = probit_probabilities(action_binary, beta=beta, seed=seed)
		case "MProbit":
			probabilities = multivariate_probit_probabilities(action_binary, beta=beta, seed=seed)
		case "MNLrefPrice":
			probabilities = mnl_reference_price_probabilities(
				action_binary,
				beta=beta,
				reference_price=reference_price,
			)
		case "MNLConsidSet":
			probabilities = mnl_consideration_set_probabilities(
				action_binary,
				beta=beta,
				seed=seed,
			)
		case "NLogit":
			probabilities = nested_logit_probabilities(action_binary, beta=beta)
		case "TMNL":
			probabilities = tmnl_probabilities(action_binary, beta=beta)
		case _:
			raise ValueError(f"Unsupported model '{model}'")

	return probabilities if include_outside else probabilities[:-1]

