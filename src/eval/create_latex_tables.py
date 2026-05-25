"""
Convert execution logs to LaTeX tables.
This script parses the text-formatted tables in exec.log files and 
generates LaTeX table code with metadata headers.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

import src.constants as C


MODEL_ORDER_MAP = {name.lower(): idx for idx, name in enumerate(C.DEMAND_MODELS)}
def _resolve_results_dir() -> Path:
    """Resolve the results directory regardless of invocation working directory."""
    script_src_dir = Path(__file__).resolve().parents[1]
    candidates = [
        script_src_dir / 'results',
        Path(C.BASE_DIR) / 'results',
        Path.cwd() / 'results',
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Default to canonical location under src for clearer error reporting.
    return script_src_dir / 'results'


def _parse_result_folder_name(folder_name: str) -> Tuple[str, str, str]:
    """Parse supported results folder naming conventions.

    Returns: (model, sensitivity, regime_kind)
    where regime_kind is one of: classical/model_informed/all/effsets.
    """
    # Current naming: <small|large>_<classical|model_informed>_<high|low>_<model>
    current_match = re.match(
        r'^(small|large)_(classical|model_informed)_(high|low)_(.+)$',
        folder_name,
        flags=re.IGNORECASE,
    )
    if current_match:
        regime_kind = current_match.group(2)
        sensitivity = current_match.group(3)
        model = current_match.group(4)
        return model, sensitivity, regime_kind

    # Legacy naming: <model>_<high|low>_<all|effsets>
    legacy_match = re.match(r'^(.+)_(high|low)_(all|effsets)$', folder_name, flags=re.IGNORECASE)
    if legacy_match:
        model = legacy_match.group(1)
        sensitivity = legacy_match.group(2)
        regime_kind = legacy_match.group(3)
        return model, sensitivity, regime_kind

    # Fallback for unexpected names.
    return folder_name, '', ''


def parse_exec_log(log_path: str) -> Dict:
    """
    Parse an exec.log file and extract tables by timestep, plus metadata.
    
    Returns a dictionary with structure:
    {
        'metadata': {...},
        'timesteps': {
            '20000': [...rows...],
            '100000': [...rows...],
            ...
        }
    }
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = {
        'metadata': {},
        'timesteps': {}
    }
    
    # Extract observation sampling time
    sampling_match = re.search(r'Observation sampling time: ([\d.]+) seconds', content)
    if sampling_match:
        results['metadata']['sampling_time'] = sampling_match.group(1)
    
    # Extract Estimation_MNL info
    estimation_mnl_match = re.search(r'Estimation_MNL\s+time: ([\d.]+)s \| beta: ([-\d.]+), lambda: ([\d.]+)', content)
    if estimation_mnl_match:
        results['metadata']['estimation_mnl_time'] = estimation_mnl_match.group(1)
        results['metadata']['estimation_mnl_beta'] = estimation_mnl_match.group(2)
        results['metadata']['estimation_mnl_lambda'] = estimation_mnl_match.group(3)
        # Extract LL, AIC, BIC - try format with AIC/BIC first
        ll_aic_bic_same_line = re.search(r'Estimation_MNL.*?final log likelihood: ([-\d.]+), AIC: ([\d.]+), BIC: ([\d.]+)', content, re.DOTALL)
        if ll_aic_bic_same_line:
            results['metadata']['estimation_mnl_ll'] = ll_aic_bic_same_line.group(1)
            results['metadata']['estimation_mnl_aic'] = ll_aic_bic_same_line.group(2)
            results['metadata']['estimation_mnl_bic'] = ll_aic_bic_same_line.group(3)
        else:
            # Try next-line format for backward compatibility
            ll_aic_bic_match = re.search(r'Estimation_MNL.*?\n\s+LL: ([-\d.]+), AIC: ([\d.]+), BIC: ([\d.]+)', content, re.DOTALL)
            if ll_aic_bic_match:
                results['metadata']['estimation_mnl_ll'] = ll_aic_bic_match.group(1)
                results['metadata']['estimation_mnl_aic'] = ll_aic_bic_match.group(2)
                results['metadata']['estimation_mnl_bic'] = ll_aic_bic_match.group(3)
    
    # Extract MMNL 5PT estimation info.
    estimation_mmnl_5pt_match = re.search(r'Estimation_MMNL_5PT time: ([\d.]+)s \| lambda: ([\d.]+)', content)
    if estimation_mmnl_5pt_match:
        results['metadata']['estimation_mmnl_5pt_time'] = estimation_mmnl_5pt_match.group(1)
        results['metadata']['estimation_mmnl_5pt_lambda'] = estimation_mmnl_5pt_match.group(2)
        # Extract betas and weights for MMNL_5PT from the next lines
        betas_match = re.search(r'Estimation_MMNL_5PT.*?\nBetas: (\[.*?\])', content, re.DOTALL)
        if betas_match:
            results['metadata']['estimation_mmnl_5pt_betas'] = betas_match.group(1)
        weights_match = re.search(r'Estimation_MMNL_5PT.*?\nWeights: (\[.*?\])', content, re.DOTALL)
        if weights_match:
            results['metadata']['estimation_mmnl_5pt_weights'] = weights_match.group(1)
        # Extract LL, AIC, BIC - try format with AIC/BIC first
        ll_aic_bic_same_line = re.search(r'Estimation_MMNL_5PT.*?final log likelihood: ([-\d.]+), AIC: ([\d.]+), BIC: ([\d.]+)', content, re.DOTALL)
        if ll_aic_bic_same_line:
            results['metadata']['estimation_mmnl_5pt_ll'] = ll_aic_bic_same_line.group(1)
            results['metadata']['estimation_mmnl_5pt_aic'] = ll_aic_bic_same_line.group(2)
            results['metadata']['estimation_mmnl_5pt_bic'] = ll_aic_bic_same_line.group(3)
        else:
            # Try next-line format for backward compatibility
            ll_aic_bic_match = re.search(r'Estimation_MMNL_5PT.*?\n\s+LL: ([-\d.]+), AIC: ([\d.]+), BIC: ([\d.]+)', content, re.DOTALL)
            if ll_aic_bic_match:
                results['metadata']['estimation_mmnl_5pt_ll'] = ll_aic_bic_match.group(1)
                results['metadata']['estimation_mmnl_5pt_aic'] = ll_aic_bic_match.group(2)
                results['metadata']['estimation_mmnl_5pt_bic'] = ll_aic_bic_match.group(3)
    
    # Extract MMNL 2PT estimation info.
    estimation_mmnl_2pt_match = re.search(r'Estimation_MMNL_2PT time: ([\d.]+)s \| lambda: ([\d.]+)', content)
    if estimation_mmnl_2pt_match:
        results['metadata']['estimation_mmnl_2pt_time'] = estimation_mmnl_2pt_match.group(1)
        results['metadata']['estimation_mmnl_2pt_lambda'] = estimation_mmnl_2pt_match.group(2)
        # Extract betas and weights for MMNL_2PT from the next lines
        betas_match = re.search(r'Estimation_MMNL_2PT.*?\nBetas: (\[.*?\])', content, re.DOTALL)
        if betas_match:
            results['metadata']['estimation_mmnl_2pt_betas'] = betas_match.group(1)
        weights_match = re.search(r'Estimation_MMNL_2PT.*?\nWeights: (\[.*?\])', content, re.DOTALL)
        if weights_match:
            results['metadata']['estimation_mmnl_2pt_weights'] = weights_match.group(1)
        # Extract LL, AIC, BIC - try format with AIC/BIC first
        ll_aic_bic_same_line = re.search(r'Estimation_MMNL_2PT.*?final log likelihood: ([-\d.]+), AIC: ([\d.]+), BIC: ([\d.]+)', content, re.DOTALL)
        if ll_aic_bic_same_line:
            results['metadata']['estimation_mmnl_2pt_ll'] = ll_aic_bic_same_line.group(1)
            results['metadata']['estimation_mmnl_2pt_aic'] = ll_aic_bic_same_line.group(2)
            results['metadata']['estimation_mmnl_2pt_bic'] = ll_aic_bic_same_line.group(3)
        else:
            # Try next-line format for backward compatibility
            ll_aic_bic_match = re.search(r'Estimation_MMNL_2PT.*?\n\s+LL: ([-\d.]+), AIC: ([\d.]+), BIC: ([\d.]+)', content, re.DOTALL)
            if ll_aic_bic_match:
                results['metadata']['estimation_mmnl_2pt_ll'] = ll_aic_bic_match.group(1)
                results['metadata']['estimation_mmnl_2pt_aic'] = ll_aic_bic_match.group(2)
                results['metadata']['estimation_mmnl_2pt_bic'] = ll_aic_bic_match.group(3)
    
    # Extract efficient sets info for all variants
    mnl_eff_match = re.search(r'MNL\s+efficient sets time: ([\d.]+)s \| sets: (\([^)]+\))', content)
    if mnl_eff_match:
        results['metadata']['mnl_effsets_time'] = mnl_eff_match.group(1)
        results['metadata']['mnl_effsets'] = mnl_eff_match.group(2)
    
    mmnl_5pt_eff_match = re.search(r'MMNL 5PT efficient sets time: ([\d.]+)s \| sets: (\([^)]+\))', content)
    if mmnl_5pt_eff_match:
        results['metadata']['mmnl_5pt_effsets_time'] = mmnl_5pt_eff_match.group(1)
        results['metadata']['mmnl_5pt_effsets'] = mmnl_5pt_eff_match.group(2)
    
    mmnl_2pt_eff_match = re.search(r'MMNL 2PT efficient sets time: ([\d.]+)s \| sets: (\([^)]+\))', content)
    if mmnl_2pt_eff_match:
        results['metadata']['mmnl_2pt_effsets_time'] = mmnl_2pt_eff_match.group(1)
        results['metadata']['mmnl_2pt_effsets'] = mmnl_2pt_eff_match.group(2)
    
    # Extract DP info for all variants
    dp_mnl_match = re.search(r'DP_MNL\s+time: ([\d.]+)s \| V\(0,C\): ([\d.]+)(?: \| avg reward: ([\d.]+))?', content)
    if dp_mnl_match:
        results['metadata']['dp_mnl_time'] = dp_mnl_match.group(1)
        results['metadata']['dp_mnl_value'] = dp_mnl_match.group(2)
        if dp_mnl_match.group(3):
            results['metadata']['dp_mnl_sim_reward'] = dp_mnl_match.group(3)
    
    dp_mmnl_5pt_match = re.search(r'DP_MMNL_5PT time: ([\d.]+)s \| V\(0,C\): ([\d.]+)(?: \| avg reward: ([\d.]+))?', content)
    if dp_mmnl_5pt_match:
        results['metadata']['dp_mmnl_5pt_time'] = dp_mmnl_5pt_match.group(1)
        results['metadata']['dp_mmnl_5pt_value'] = dp_mmnl_5pt_match.group(2)
        if dp_mmnl_5pt_match.group(3):
            results['metadata']['dp_mmnl_5pt_sim_reward'] = dp_mmnl_5pt_match.group(3)
    
    dp_mmnl_2pt_match = re.search(r'DP_MMNL_2PT time: ([\d.]+)s \| V\(0,C\): ([\d.]+)(?: \| avg reward: ([\d.]+))?', content)
    if dp_mmnl_2pt_match:
        results['metadata']['dp_mmnl_2pt_time'] = dp_mmnl_2pt_match.group(1)
        results['metadata']['dp_mmnl_2pt_value'] = dp_mmnl_2pt_match.group(2)
        if dp_mmnl_2pt_match.group(3):
            results['metadata']['dp_mmnl_2pt_sim_reward'] = dp_mmnl_2pt_match.group(3)

    # Extract ADP info for all variants
    adp_mnl_match = re.search(r'ADP_MNL\s+time: ([\d.]+)s \| V\(0,C\): ([\d.]+)(?: \| avg reward: ([\d.]+))?', content)
    if adp_mnl_match:
        results['metadata']['adp_mnl_time'] = adp_mnl_match.group(1)
        results['metadata']['adp_mnl_value'] = adp_mnl_match.group(2)
        if adp_mnl_match.group(3):
            results['metadata']['adp_mnl_sim_reward'] = adp_mnl_match.group(3)

    adp_mmnl_5pt_match = re.search(r'ADP_MMNL_5PT\s+time: ([\d.]+)s \| V\(0,C\): ([\d.]+)(?: \| avg reward: ([\d.]+))?', content)
    if adp_mmnl_5pt_match:
        results['metadata']['adp_mmnl_5pt_time'] = adp_mmnl_5pt_match.group(1)
        results['metadata']['adp_mmnl_5pt_value'] = adp_mmnl_5pt_match.group(2)
        if adp_mmnl_5pt_match.group(3):
            results['metadata']['adp_mmnl_5pt_sim_reward'] = adp_mmnl_5pt_match.group(3)

    adp_mmnl_2pt_match = re.search(r'ADP_MMNL_2PT\s+time: ([\d.]+)s \| V\(0,C\): ([\d.]+)(?: \| avg reward: ([\d.]+))?', content)
    if adp_mmnl_2pt_match:
        results['metadata']['adp_mmnl_2pt_time'] = adp_mmnl_2pt_match.group(1)
        results['metadata']['adp_mmnl_2pt_value'] = adp_mmnl_2pt_match.group(2)
        if adp_mmnl_2pt_match.group(3):
            results['metadata']['adp_mmnl_2pt_sim_reward'] = adp_mmnl_2pt_match.group(3)
    
    # Extract experiment description
    config_match = re.search(r'Run (.+?): (.+)', content)
    if config_match:
        results['metadata']['experiment'] = config_match.group(1).strip()
    
    # Extract training on all or efficient sets
    train_all_match = re.search(r'Training on all available sets \(if false only on efficient sets\): (.+)', content)
    if train_all_match:
        results['metadata']['train_all_sets'] = train_all_match.group(1).strip() == 'True'
    
    # Extract tables for each timestep
    pattern = r'=== ([\d,]+) training timesteps ===\n(Method.*?)(?==== |$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        timestep = match.group(1).replace(',', '')
        table_text = match.group(2)
        
        # Parse table rows
        rows = []
        lines = table_text.strip().split('\n')
        
        # Skip header and separator lines
        data_lines = [line for line in lines[2:] if line.strip() and '--------' not in line]
        
        for line in data_lines:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 1:
                method = parts[0]
                values = parts[1:]
                
                try:
                    row = {
                        'Method': method,
                        'TrainTimeMean': float(values[0]),
                        'TrainTimeStd': float(values[1]),
                        'RewardMean': float(values[2]),
                        'RewardStdAcrossRuns': float(values[3]),
                        'PctOfDP': float(values[4]),
                        'PctOfDPStd': float(values[5]),
                        'LoadFactorMean': float(values[6]),
                        'LoadFactorStdAcrossRuns': float(values[7]),
                    }
                    rows.append(row)
                except (ValueError, IndexError):
                    pass
        
        if rows:
            results['timesteps'][timestep] = rows
    
    return results


def extract_model_display_name(folder_name: str) -> str:
    """Extract and map model code in folder name to requested display name."""
    model_code, _, _ = _parse_result_folder_name(folder_name)

    model_display_map = {
        'MNL': 'MNL',
        'MMNL': 'MMNL',
        'MMNL_5PT': 'MMNL 5PT',
        'MMNL_2PT': 'MMNL 2PT',
        'Probit': 'Probit',
        'MNLrefPrice': 'MNL with Reference Price',
        'MNLConsidSet': 'MNL with Consideration Set',
        'TMNL': 'TMNL',
        'NLogit': 'Nested Logit',
    }

    return model_display_map.get(model_code, model_code)


def _parse_mmnl_betas(betas_raw: str) -> List[str]:
    """Parse MMNL betas string from log into a list of values."""
    if not betas_raw:
        return []

    try:
        parsed = ast.literal_eval(betas_raw)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass

    # Fallback: return raw string as a single value if parsing fails.
    return [str(betas_raw)]


def _count_sets(sets_raw: str) -> str:
    """Return number of efficient sets from raw tuple/list text."""
    if not sets_raw:
        return 'N/A'

    try:
        parsed = ast.literal_eval(sets_raw)
        if isinstance(parsed, (list, tuple, set)):
            return str(len(parsed))
    except (ValueError, SyntaxError):
        pass

    return 'N/A'


def _format_steps_with_dots(step_count: int) -> str:
    """Format integer with dot thousands separators (e.g., 100000 -> 100.000)."""
    return f"{step_count:,}".replace(',', '.')


def _format_decimal(value: str, digits: int) -> str:
    """Format numeric string to a fixed number of decimal places."""
    if value in (None, '', 'N/A'):
        return 'N/A'

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_multiline_values(values_list: List[str], max_per_line: int = 3) -> str:
    """
    Format a list of values across multiple lines using LaTeX makecell.
    Distributes values more evenly to avoid wide cells.
    """
    if not values_list or len(values_list) == 0:
        return 'N/A'
    
    # For short lists, keep on one line
    if len(values_list) <= max_per_line:
        return ', '.join(values_list)
    
    # Split into multiple lines
    lines = []
    for i in range(0, len(values_list), max_per_line):
        chunk = values_list[i:i+max_per_line]
        lines.append(', '.join(chunk))
    
    # Use makecell for multi-line support with proper LaTeX line breaks and top-right alignment
    return r'\makecell[tr]{' + r' \\ '.join(lines) + '}'


def _result_folder_sort_key(folder: Path) -> Tuple[int, int, int, str]:
    """Sort folders by model order, then regime, then sensitivity."""
    name = folder.name
    model, sensitivity, regime_kind = _parse_result_folder_name(name)

    model_rank = MODEL_ORDER_MAP.get(model.lower(), len(MODEL_ORDER_MAP))
    regime_map = {
        'classical': 0,
        'all': 0,
        'model_informed': 1,
        'effsets': 1,
    }
    regime_rank = regime_map.get(regime_kind.lower(), 2)
    sensitivity_rank = 0 if sensitivity.lower() == 'low' else 1

    return (model_rank, regime_rank, sensitivity_rank, name.lower())


def create_metadata_section(metadata: Dict, folder_name: str) -> str:
    """Create LaTeX section with metadata in a more beautiful format."""
    latex = []

    mmnl_5pt_betas = _parse_mmnl_betas(metadata.get('estimation_mmnl_5pt_betas', ''))
    mmnl_5pt_beta_text = _format_multiline_values([_format_decimal(beta, 4) for beta in mmnl_5pt_betas], max_per_line=3) if mmnl_5pt_betas else 'N/A'
    mmnl_5pt_weights = _parse_mmnl_betas(metadata.get('estimation_mmnl_5pt_weights', ''))
    mmnl_5pt_weights_text = _format_multiline_values([_format_decimal(w, 4) for w in mmnl_5pt_weights], max_per_line=3) if mmnl_5pt_weights else 'N/A'

    mmnl_2pt_betas = _parse_mmnl_betas(metadata.get('estimation_mmnl_2pt_betas', ''))
    mmnl_2pt_beta_text = ', '.join(_format_decimal(beta, 4) for beta in mmnl_2pt_betas) if mmnl_2pt_betas else 'N/A'
    mmnl_2pt_weights = _parse_mmnl_betas(metadata.get('estimation_mmnl_2pt_weights', ''))
    mmnl_2pt_weights_text = ', '.join(_format_decimal(w, 4) for w in mmnl_2pt_weights) if mmnl_2pt_weights else 'N/A'

    mnl_effsets_count = _count_sets(metadata.get('mnl_effsets', ''))
    mmnl_5pt_effsets_count = _count_sets(metadata.get('mmnl_5pt_effsets', ''))
    mmnl_2pt_effsets_count = _count_sets(metadata.get('mmnl_2pt_effsets', ''))

    def fmt_seconds(value: str) -> str:
        rounded = _format_decimal(value, 2)
        return f"{rounded}s" if rounded != 'N/A' else 'N/A'

    latex.append(r'\begin{center}')
    latex.append(r'  \small')
    latex.append(r'  \begin{tabular}{l|r|r|r}')
    latex.append(r'    \toprule')
    latex.append(r'    Metric & \textbf{MNL} & \textbf{MMNL 5PT} & \textbf{MMNL 2PT} \\')
    latex.append(r'    \midrule')

    sampling_time = _format_decimal(metadata.get('sampling_time', 'N/A'), 2)
    latex.append(f"    Obs. Sampling Time & \\multicolumn{{3}}{{c}}{{{sampling_time} s}} \\\\")

    est_mnl_lambda = _format_decimal(metadata.get('estimation_mnl_lambda', 'N/A'), 6)
    latex.append(f"    Estim. $\\lambda$ & \\multicolumn{{3}}{{c}}{{{est_mnl_lambda}}} \\\\")

    est_mnl_time = fmt_seconds(metadata.get('estimation_mnl_time', 'N/A'))
    est_5pt_time = fmt_seconds(metadata.get('estimation_mmnl_5pt_time', 'N/A'))
    est_2pt_time = fmt_seconds(metadata.get('estimation_mmnl_2pt_time', 'N/A'))
    latex.append(f"    Estim. Time & {est_mnl_time} & {est_5pt_time} & {est_2pt_time} \\\\")

    est_mnl_beta = _format_decimal(metadata.get('estimation_mnl_beta', 'N/A'), 6)
    latex.append(f"    Estim. $\\beta$ & {est_mnl_beta} & {mmnl_5pt_beta_text} & {mmnl_2pt_beta_text} \\\\")
    latex.append(f"    Estim. Weights & - & {mmnl_5pt_weights_text} & {mmnl_2pt_weights_text} \\\\")

    est_mnl_ll = _format_decimal(metadata.get('estimation_mnl_ll', 'N/A'), 2)
    est_5pt_ll = _format_decimal(metadata.get('estimation_mmnl_5pt_ll', 'N/A'), 2)
    est_2pt_ll = _format_decimal(metadata.get('estimation_mmnl_2pt_ll', 'N/A'), 2)
    latex.append(f"    Log Likelihood & {est_mnl_ll} & {est_5pt_ll} & {est_2pt_ll} \\\\")

    est_mnl_aic = _format_decimal(metadata.get('estimation_mnl_aic', 'N/A'), 2)
    est_5pt_aic = _format_decimal(metadata.get('estimation_mmnl_5pt_aic', 'N/A'), 2)
    est_2pt_aic = _format_decimal(metadata.get('estimation_mmnl_2pt_aic', 'N/A'), 2)
    latex.append(f"    AIC & {est_mnl_aic} & {est_5pt_aic} & {est_2pt_aic} \\\\")

    est_mnl_bic = _format_decimal(metadata.get('estimation_mnl_bic', 'N/A'), 2)
    est_5pt_bic = _format_decimal(metadata.get('estimation_mmnl_5pt_bic', 'N/A'), 2)
    est_2pt_bic = _format_decimal(metadata.get('estimation_mmnl_2pt_bic', 'N/A'), 2)
    latex.append(f"    BIC & {est_mnl_bic} & {est_5pt_bic} & {est_2pt_bic} \\\\")

    if 'effsets' in folder_name.lower():
        eff_mnl_time = fmt_seconds(metadata.get('mnl_effsets_time', 'N/A'))
        eff_5pt_time = fmt_seconds(metadata.get('mmnl_5pt_effsets_time', 'N/A'))
        eff_2pt_time = fmt_seconds(metadata.get('mmnl_2pt_effsets_time', 'N/A'))
        latex.append(f"    Eff. Sets Time & {eff_mnl_time} & {eff_5pt_time} & {eff_2pt_time} \\\\")
        latex.append(f"    \\# Eff. Sets & {mnl_effsets_count} & {mmnl_5pt_effsets_count} & {mmnl_2pt_effsets_count} \\\\")

    dp_mnl_value = metadata.get('dp_mnl_value', 'N/A')
    dp_5pt_value = metadata.get('dp_mmnl_5pt_value', 'N/A')
    dp_2pt_value = metadata.get('dp_mmnl_2pt_value', 'N/A')
    latex.append(f"    DP $V(0,C)$ & {dp_mnl_value} & {dp_5pt_value} & {dp_2pt_value} \\\\")

    dp_mnl_sim_reward = metadata.get('dp_mnl_sim_reward', 'N/A')
    dp_5pt_sim_reward = metadata.get('dp_mmnl_5pt_sim_reward', 'N/A')
    dp_2pt_sim_reward = metadata.get('dp_mmnl_2pt_sim_reward', 'N/A')
    latex.append(
        f"    Sim. Rew. (n=1000) & {dp_mnl_sim_reward} & {dp_5pt_sim_reward} & {dp_2pt_sim_reward} \\\\")

    adp_mnl_value = metadata.get('adp_mnl_value', 'N/A')
    adp_5pt_value = metadata.get('adp_mmnl_5pt_value', 'N/A')
    adp_2pt_value = metadata.get('adp_mmnl_2pt_value', 'N/A')
    latex.append(f"    ADP $V(0,C)$ & {adp_mnl_value} & {adp_5pt_value} & {adp_2pt_value} \\\\")

    adp_mnl_sim_reward = metadata.get('adp_mnl_sim_reward', 'N/A')
    adp_5pt_sim_reward = metadata.get('adp_mmnl_5pt_sim_reward', 'N/A')
    adp_2pt_sim_reward = metadata.get('adp_mmnl_2pt_sim_reward', 'N/A')
    latex.append(
        f"    ADP Sim. Rew. (n=1000) & {adp_mnl_sim_reward} & {adp_5pt_sim_reward} & {adp_2pt_sim_reward} \\\\")

    latex.append(r'    \bottomrule')
    latex.append(r'  \end{tabular}')
    latex.append(r'\end{center}')
    latex.append(r'\vspace{0.5em}')
    latex.append('')

    return '\n'.join(latex)


def create_latex_table(data: List[Dict], timestep: str) -> str:
    """
    Create a LaTeX table from parsed data for a single timestep.
    """
    latex = []
    
    # Reorder rows: DP first, then ADP, then remaining methods.
    dp_method_order = ['DP_MNL', 'DP_MMNL_5PT', 'DP_MMNL_2PT']
    adp_method_order = ['ADP_MNL', 'ADP_MMNL_5PT', 'ADP_MMNL_2PT', 'ADP_ENV']
    dp_rows = [r for r in data if r['Method'] in dp_method_order]
    adp_rows = [r for r in data if r['Method'] in adp_method_order]
    other_rows = [
        r
        for r in data
        if r['Method'] not in dp_method_order and r['Method'] not in adp_method_order
    ]
    
    # Sort DP/ADP rows to ensure a stable canonical order.
    dp_rows_sorted = sorted(dp_rows, key=lambda x: dp_method_order.index(x['Method']))
    adp_rows_sorted = sorted(adp_rows, key=lambda x: adp_method_order.index(x['Method']))
    
    # Keep groups separate so we can insert visual separators in the table.
    grouped_data = [dp_rows_sorted, adp_rows_sorted, other_rows]
    
    # Subsection for this timestep (convert to int for formatting)
    timestep_int = int(timestep)
    timestep_label = _format_steps_with_dots(timestep_int)
    
    # Non-floating table block to avoid float-queue overflows when including many result tables.
    latex.append(r'\begin{center}')
    latex.append(r'  \small')
    latex.append(r'  \begin{tabular}{l|rr|rr|rr|rr}')
    latex.append(r'    \toprule')
    latex.append(rf'    \multicolumn{{9}}{{c}}{{\textbf{{Training Results over {timestep_label} Steps (Sample Size: 15)}}}} \\')
    latex.append(r'    \midrule')
    
    # Header
    latex.append(r'    Method & \multicolumn{2}{c|}{Training Time (s)} & '
                r'\multicolumn{2}{c|}{Reward} & \multicolumn{2}{c|}{\% of DP} & '
                r'\multicolumn{2}{c}{Load Factor (\%)} \\')
    latex.append(r'    & Mean & Std & Mean & Std & Mean & Std & Mean & Std \\')
    latex.append(r'    \midrule')
    
    # Data rows with proper underscore escaping and separators between DP/ADP/RL blocks.
    for group_index, group_rows in enumerate(grouped_data):
        if not group_rows:
            continue

        for row in group_rows:
            method = row['Method'].replace('_', '\\_')  # Escape underscores
            train_time_mean = row['TrainTimeMean']
            train_time_std = row['TrainTimeStd']
            reward_mean = row['RewardMean']
            reward_std = row['RewardStdAcrossRuns']
            pct_dp = row['PctOfDP']
            pct_dp_std = row['PctOfDPStd']
            load_mean = row['LoadFactorMean']
            load_std = row['LoadFactorStdAcrossRuns']

            # Mark DP methods with a star (reference percentage), but not for MMNL_2PT and MMNL_5PT.
            if row['Method'].startswith('DP_') and row['Method'] not in ['DP_MMNL_2PT', 'DP_MMNL_5PT']:
                pct_dp_display = f'{pct_dp:.2f}*'
            else:
                pct_dp_display = f'{pct_dp:.2f}'

            # For DP and ADP methods, display -- for training time std.
            if row['Method'].startswith('DP_') or row['Method'].startswith('ADP_'):
                train_time_std_display = '   --'
            else:
                train_time_std_display = f'{train_time_std:8.2f}'

            line = (f'    {method:12s} & {train_time_mean:8.2f} & {train_time_std_display} & '
                 f'{reward_mean:10.1f} & {reward_std:10.2f} & {pct_dp_display:>7s} & {pct_dp_std:7.2f} & '
                   f'{load_mean:7.2f} & {load_std:7.2f} \\\\')
            latex.append(line)

        has_next_nonempty_group = any(next_group for next_group in grouped_data[group_index + 1:])
        if has_next_nonempty_group:
            latex.append(r'    \midrule')
    
    latex.append(r'    \bottomrule')
    latex.append(r'  \end{tabular}')
    latex.append(r'\end{center}')
    latex.append(r'\vspace{0.5em}')
    latex.append('')
    
    return '\n'.join(latex)


def create_master_latex_file(result_folders: List[Path], output_path: Path) -> None:
    """Create a paper-ready include snippet grouped by demand model and regime."""
    model_display_map = {
        'MNL': 'MNL',
        'MMNL_2PT': 'MMNL 2PT',
        'MMNL_5PT': 'MMNL 5PT',
        'Probit': 'Probit',
        'MNLrefPrice': 'MNL with Reference Price',
        'MNLConsidSet': 'MNL with Consideration Set',
        'TMNL': 'TMNL',
        'NLogit': 'Nested Logit',
    }

    parsed = []
    for folder in result_folders:
        size_match = re.match(
            r'^(small|large)_(classical|model_informed)_(high|low)_(.+)$',
            folder.name,
            flags=re.IGNORECASE,
        )
        if size_match:
            size_key = size_match.group(1).lower()
            regime_kind = size_match.group(2)
            sensitivity = size_match.group(3)
            model = size_match.group(4)
        else:
            size_key = ''
            model, sensitivity, regime_kind = _parse_result_folder_name(folder.name)

        normalized_regime = regime_kind.lower()
        if normalized_regime == 'all':
            normalized_regime = 'classical'
        elif normalized_regime == 'effsets':
            normalized_regime = 'model_informed'
        parsed.append((folder, size_key, model, sensitivity.lower(), normalized_regime))

    section_titles = {
        'small': 'Experimental Results - Small number of products (n=10)',
        'large': 'Experimental Results - Large number of products (n=100)',
    }

    regime_titles_default = {
        'classical': 'Classical approach over all possible product assortments',
        'model_informed': 'Model-Informed approach over efficient product assortments',
    }

    lines = []
    for size_key in ['small', 'large']:
        lines.append(rf'\section{{{section_titles.get(size_key, size_key)}}}')
        lines.append('')

        for model_code in C.DEMAND_MODELS:
            model_folders = [
                p
                for p in parsed
                if p[1] == size_key and p[2].lower() == model_code.lower()
            ]
            if not model_folders:
                continue

            lines.append(rf'\subsection{{{model_display_map.get(model_code, model_code)}}}')
            lines.append('')

            for regime_key in ['classical', 'model_informed']:
                regime_label = regime_titles_default[regime_key]
                lines.append(rf'\subsubsection{{{regime_label}}}')
                lines.append('')

                for sensitivity_label, sensitivity_key in [('Low Price Sensitivity', 'low'), ('High Price Sensitivity', 'high')]:
                    selected = [
                        folder
                        for folder, _, _, sensitivity, regime in model_folders
                        if regime == regime_key and sensitivity == sensitivity_key
                    ]
                    selected = sorted(selected, key=lambda f: f.name.lower())
                    if not selected:
                        continue

                    lines.append(rf'\paragraph{{{sensitivity_label}}}')
                    lines.append('')

                    for folder in selected:
                        lines.append(rf'\input{{results/{folder.name}/results_table.tex}}')
                        lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    """Generate LaTeX tables for all result folders."""
    results_dir = _resolve_results_dir()
    
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        return
    
    # Get all result folders
    result_folders = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=_result_folder_sort_key,
    )
    
    print(f'Processing {len(result_folders)} result folders...\n')
    
    for folder in result_folders:
        log_path = folder / '00_exec.log'
        
        if not log_path.exists():
            print(f"⚠ Skipped {folder.name}: no 00_exec.log found")
            continue
        
        print(f"Processing: {folder.name}")
        
        # Parse the log
        parsed = parse_exec_log(str(log_path))
        
        # Create output directory within the results folder
        output_file = folder / 'results_table.tex'
        
        with open(output_file, 'w') as f:
            # Write metadata section
            f.write(create_metadata_section(parsed['metadata'], folder.name))
            
            # Write tables for each timestep
            for timestep in sorted(parsed['timesteps'].keys(), key=int):
                data = parsed['timesteps'][timestep]
                latex_code = create_latex_table(data, timestep)
                f.write(latex_code)
        
        print(f'  ✓ Created: {folder.name}/results_table.tex\n')

    master_file = results_dir / 'all_results_tables.tex'
    create_master_latex_file(result_folders, master_file)
    print(f'✓ Created master file: {master_file}\n')
    
    print('Done!')


if __name__ == '__main__':
    main()
