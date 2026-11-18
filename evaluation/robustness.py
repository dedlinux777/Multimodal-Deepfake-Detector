"""Domain-stratified evaluation helpers for generator/dataset shift."""
from __future__ import annotations
import pandas as pd
from .evaluate import BRANCH_TARGET, _branch_metrics

def evaluate_domains(predictions: pd.DataFrame, calibrated_thresholds: dict,
                     in_distribution_domains: set[str], domain_column: str) -> dict:
    """Report per-domain branch metrics using fixed validation-calibrated thresholds.

    This function never fits thresholds or other parameters. Domains absent
    from the validation domain set are marked out-of-distribution. Results are
    descriptive and do not by themselves establish broad generalization.
    """
    if domain_column not in predictions:
        raise ValueError(f"Missing domain metadata column: {domain_column}")
    result={"domain_column":domain_column,"in_distribution":{},"out_of_distribution":{}}
    for domain,group in predictions.groupby(domain_column,dropna=False):
        name=str(domain); destination="in_distribution" if name in in_distribution_domains else "out_of_distribution"
        result[destination][name]={"sample_count":len(group),"branches":{}}
        for branch in BRANCH_TARGET:
            usable=group.dropna(subset=[f"{branch}_p_fake"])
            if len(usable):result[destination][name]["branches"][branch]=_branch_metrics(usable,branch,calibrated_thresholds[branch]["threshold"])
            else:result[destination][name]["branches"][branch]={"sample_count":0,"error":"no valid predictions"}
    return result
