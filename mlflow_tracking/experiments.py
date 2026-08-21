"""Experiment naming conventions for MLflow tracking."""

PROBLEM_EXPERIMENTS = {
    "p1": "p1_role_classification",
    "p2": "p2_clan_rank",
    "p3": "p3_war_performance",
    "p4": "p4_clan_performance_classification",
    "p5": "p5_player_clustering",
}

def get_experiment_name(problem: str) -> str:
    """Return the standard MLflow experiment name for a given problem code."""
    if problem not in PROBLEM_EXPERIMENTS:
        raise ValueError(f"Unknown problem code: {problem}. Valid codes: {list(PROBLEM_EXPERIMENTS)}")
    return PROBLEM_EXPERIMENTS[problem]
