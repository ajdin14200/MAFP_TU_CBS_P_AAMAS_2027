"""Simple supplementary-material runner for all methods compared in the paper.

Methods:
  CBS_TU, SAT, CBS_P, CBS_k, EECBS_TU, EECBS_P, EECBS_k

Example:
python test_paper_methods_all.py --map maps/benchmark/empty-16-16.map \
    --picat-dir path/to/MAPF-TU/picat --instances 10 --agents 20 \
    --uncertainty 1 --k 0.7 --time-limit 300 --seed 0
"""
import argparse
import csv
import random

from pathfinding.planners.cbstu import CBSTUPlanner
from pathfinding.planners.utils.tu_problem import TimeUncertaintyProblem
from pathfinding.planners.picat_sat_runner import PicatSATPlanner


def make_instance(map_path, agents, uncertainty, seed):
    # TimeUncertaintyProblem's generator uses Python's random module.
    random.seed(seed)
    problem = TimeUncertaintyProblem(map_path)
    problem.generate_problem_instance(uncertainty=uncertainty)
    problem.generate_agents(agents)
    problem.fill_heuristic_table()
    return problem


def solve_cbs_tu(problem, limit):
    return CBSTUPlanner(problem).find_solution(
        min_best_case=False, time_lim=limit, soc=False,
        use_cat=True, use_pc=False, use_bp=True)


def solve_cbs_p(problem, k, limit):
    return CBSTUPlanner(problem).find_solution_propagation_upper_bound_minimal_split(
        min_best_case=False, time_lim=limit, soc=False,
        use_cat=True, use_pc=False, use_bp=True, k_safe=k)


def solve_eecbs_tu(problem, limit):
    return CBSTUPlanner(problem).find_solution_eecbs_tu_strong(
        min_best_case=False, time_lim=limit, soc=False,
        use_cat=True, use_pc=False, use_bp=True,
        focal_w=1.2, use_low_level_focal=True, low_level_focal_w=1.2,
        conflict_weight=1.0, use_wdg=False,
        wdg_weight=0.1, wdg_overlap_weight=0.05)


def solve_eecbs_p(problem, k, limit):
    return CBSTUPlanner(problem).find_solution_eecbs_p_strong(
        min_best_case=False, time_lim=limit, soc=False,
        use_cat=True, use_pc=False, use_bp=True,
        focal_w=1.2, use_low_level_focal=True, low_level_focal_w=1.2,
        conflict_weight=1.0, use_wdg=False,
        wdg_weight=0.1, wdg_overlap_weight=0.05,
        compute_optimal_policy=True, k_safe=k)


def solve_sat(problem, picat_dir, limit, instance_name):
    return PicatSATPlanner(
        problem, picat_dir=picat_dir, objective="mks"
    ).find_solution(time_lim=limit, instance_name=instance_name)


def normal_row(instance_name, method, s):
    # SAT exposes cost as a scalar; CBS-family solutions expose [lower, upper].
    cost = s.cost if method == "SAT" else s.cost[1]
    return [
        instance_name, method, s.is_solved, s.success_rate, cost,
        s.time_to_solve, s.iteration, s.iteration_with_edge_conflicts
    ]


def main():
    ap = argparse.ArgumentParser(
        description="Generate random MAPF-TU instances and run all paper methods.")
    ap.add_argument("--map", required=True, help="MovingAI .map file.")
    ap.add_argument("--picat-dir", required=True,
                    help="Directory containing picat, mks.pi and aux.pi.")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--uncertainty", type=int, default=1)
    ap.add_argument("--k", type=float, default=0.7)
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="test_results.csv")
    args = ap.parse_args()

    if not 0.0 <= args.k <= 1.0:
        ap.error("--k must be in [0,1].")

    rows = []

    for i in range(args.instances):
        name = f"instance_{i}"
        seed = args.seed + i
        print(f"\n{name} (seed={seed})")

        # Generate exactly one problem. All methods operate on the same
        # starts, goals and uncertain edge weights.
        problem = make_instance(
            args.map, args.agents, args.uncertainty, seed)

        methods = [
            ("CBS_TU", lambda: solve_cbs_tu(problem, args.time_limit)),
            ("SAT", lambda: solve_sat(
                problem, args.picat_dir, args.time_limit, name)),
            ("CBS_P", lambda: solve_cbs_p(problem, 1.0, args.time_limit)),
            ("CBS_k", lambda: solve_cbs_p(problem, args.k, args.time_limit)),
            ("EECBS_TU", lambda: solve_eecbs_tu(problem, args.time_limit)),
            ("EECBS_P", lambda: solve_eecbs_p(problem, 1.0, args.time_limit)),
            ("EECBS_k", lambda: solve_eecbs_p(
                problem, args.k, args.time_limit)),
        ]

        for method, run in methods:
            print(f"  {method}: ", end="", flush=True)
            try:
                sol = run()
                rows.append(normal_row(name, method, sol))
                cost = sol.cost if method == "SAT" else sol.cost[1]
                print(f"solved={sol.is_solved}, cost={cost}, "
                      f"time={sol.time_to_solve:.3f}s")
            except Exception as exc:
                rows.append([name, method, False, 0, float("inf"), 0, None, None])
                print(f"ERROR: {exc}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "method", "solved", "success_rate", "cost", "time",
            "nb_iter", "nb_iter_edge_conflicts"
        ])
        writer.writerows(rows)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
