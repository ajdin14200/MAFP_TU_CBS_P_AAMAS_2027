"""
This class represents the solutions that are stored in each constraint node. They consist of conformant plans for each
agent, the total cost range of the solution and the length (i.e the max length between the different paths).
"""
import math
import copy
import json
import os
from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
from collections import defaultdict

STAY_STILL_COST = 1


class TimeUncertaintySolution:

    def __init__(self):
        self.cost = math.inf, math.inf
        self.paths = {}
        self.success_rate = 0
        self.iteration = 0
        self.iteration_with_edge_conflicts = 0
        self.is_solved = False
        self.tuple_solution = {}
        self.nodes_generated = -1
        self.constraints = defaultdict(list)
        self.execution_policy = None
        self.time_to_solve = -1
        self.sic = -1, -1

    @staticmethod
    def empty_solution(nodes_generated=-1):
        empty_sol = TimeUncertaintySolution()
        empty_sol.nodes_generated = nodes_generated
        return empty_sol

    @staticmethod
    def void_solution(time_to_solve):
        empty_sol = TimeUncertaintySolution()
        empty_sol.time_to_solve = time_to_solve
        return empty_sol

    def copy_solution(self, other_sol):
        """Copy a CT-node solution without sharing mutable plan objects.

        Constraint-tree children must never share TimeUncertaintyPlan instances
        with their parent or siblings.  add_stationary_moves() mutates paths; a
        shallow copy therefore makes padding performed in one CT node silently
        modify all related CT nodes and can inflate costs by orders of magnitude.
        """
        self.paths = {
            agent: copy.deepcopy(plan)
            for agent, plan in other_sol.paths.items()
        }

        self.tuple_solution = (
            copy.deepcopy(other_sol.tuple_solution)
            if other_sol.tuple_solution else {}
        )
        self.time_to_solve = other_sol.time_to_solve
        self.cost = tuple(other_sol.cost)
        self.success_rate = other_sol.success_rate
        self.iteration = other_sol.iteration
        self.iteration_with_edge_conflicts = other_sol.iteration_with_edge_conflicts
        self.is_solved = other_sol.is_solved
        self.nodes_generated = other_sol.nodes_generated
        self.constraints = copy.deepcopy(other_sol.constraints)
        self.execution_policy = copy.deepcopy(other_sol.execution_policy)
        self.sic = tuple(other_sol.sic)

    @staticmethod
    def _goal_arrival_cost(plan):
        """Return the first occurrence in the final run of the goal vertex.

        The value is recomputed from the current path instead of trusting a
        cached value.  This prevents a cache created before replanning or path
        padding from becoming stale.
        """
        if not getattr(plan, 'path', None):
            return math.inf, math.inf

        goal_vertex = plan.path[-1][1]
        first_goal_index = len(plan.path) - 1
        while (
            first_goal_index > 0
            and plan.path[first_goal_index - 1][1] == goal_vertex
        ):
            first_goal_index -= 1

        return tuple(plan.path[first_goal_index][0])

    def compute_solution_cost(self, sum_of_costs=True):
        """Compute SOC or makespan without charging terminal goal waits."""
        if not self.paths:
            self.cost = math.inf, math.inf
            return self.cost

        goal_costs = []
        for plan in self.paths.values():
            if not getattr(plan, 'path', None):
                self.cost = math.inf, math.inf
                return self.cost
            cost = self._goal_arrival_cost(plan)
            goal_costs.append(cost)
            # Keep the public plan cost consistent with the actual arrival at
            # the goal rather than the artificial terminal padding.
            plan.cost = cost

        if sum_of_costs:
            self.cost = (
                sum(cost[0] for cost in goal_costs),
                sum(cost[1] for cost in goal_costs),
            )
        else:
            self.cost = (
                max(cost[0] for cost in goal_costs),
                max(cost[1] for cost in goal_costs),
            )
        return self.cost

    def get_max_of_min_path_time(self):
        """Maximum best-case goal-arrival time, excluding terminal waits."""
        if not self.paths:
            return math.inf
        return max(self._goal_arrival_cost(plan)[0] for plan in self.paths.values())

    def get_max_path_time(self):
        """Maximum worst-case goal-arrival time, excluding terminal waits."""
        if not self.paths:
            return math.inf
        return max(self._goal_arrival_cost(plan)[1] for plan in self.paths.values())

    def create_movement_tuples(self, agents=None):
        """
        converts each path in solution to a tuple of ( (t1,t2), (u,v) ) where (u,v) is an edge and t1 is when the agent
        began the movement across it and t2 is when the agent completed the movement.

        For easier comparison, 'u' and 'v' will be sorted. We still maintain 'f' or 'b' to signify what was the original
        direction. This is necessary in a few very specific conflicts.
        """
        agents = self.paths.keys() if not agents else agents
        for agent in agents:
            path = self.paths[agent].path
            new_path = []
            for move in range(0, len(path) - 1):
                start_vertex = min(path[move][1], path[move + 1][1])
                start_time = path[move][0][0]
                next_vertex = max(path[move][1], path[move + 1][1])
                finish_time = path[move + 1][0][1]
                if start_vertex == path[move][1]:
                    direction = 'f'  # The original beginning vertex was 'start_vertex'
                else:
                    direction = 'b'  # The original beginning vertex was 'next_vertex'

                new_path.append(((start_time, finish_time), (start_vertex, next_vertex), direction))

            self.tuple_solution[agent] = new_path

    def add_stationary_moves(self, agents_to_update=None):
        """Normalize terminal padding without changing arrival costs.

        Before adding padding, every path is reduced to exactly one occurrence
        of its final goal vertex.  This makes the operation idempotent and
        prevents repeated CT-node updates from appending longer and longer
        terminal waits.
        """
        if not self.paths:
            return set()

        # Remove terminal padding already present.  The first occurrence in the
        # trailing run of the final vertex is the true goal arrival.
        for plan in self.paths.values():
            if not getattr(plan, 'path', None):
                continue
            goal_vertex = plan.path[-1][1]
            first_goal_index = len(plan.path) - 1
            while (
                first_goal_index > 0
                and plan.path[first_goal_index - 1][1] == goal_vertex
            ):
                first_goal_index -= 1
            if first_goal_index + 1 < len(plan.path):
                plan.path = plan.path[:first_goal_index + 1]
            plan.cost = tuple(plan.path[-1][0])

        max_min_time = max(
            plan.path[-1][0][0]
            for plan in self.paths.values()
            if getattr(plan, 'path', None)
        )

        if agents_to_update is None:
            selected_agents = set(self.paths.keys())
        elif hasattr(agents_to_update, "keys"):
            selected_agents = set(agents_to_update.keys())
        else:
            selected_agents = set(agents_to_update)

        new_moves = set()
        for agent, plan in self.paths.items():
            if agent not in selected_agents or not getattr(plan, 'path', None):
                continue

            arrival_interval, goal_vertex = plan.path[-1]
            arrival_min = arrival_interval[0]
            if arrival_min < max_min_time:
                # Padding exists only for conflict detection/CAT.  Its lower
                # endpoint starts one tick after the real goal arrival.
                wait_interval = (arrival_min + 1, max_min_time)
                if wait_interval[0] <= wait_interval[1]:
                    plan.path.append((wait_interval, goal_vertex))
                    new_moves.add((agent, wait_interval, goal_vertex))

        # Recompute from the normalized paths; padding is ignored by
        # _goal_arrival_cost because it belongs to the trailing goal run.
        self.compute_solution_cost(sum_of_costs=True)
        return new_moves

    def save(self, agent_num, uncertainty, map_type, agent_seed, map_seed, min_best_case, use_pc, use_bp, folder):
        solution_path = os.path.join(folder, map_type, str(agent_num) + ' agents')
        if not os.path.exists(solution_path):
            os.makedirs(solution_path)
        objective = 'min best case' if min_best_case else 'min worst case'
        file_name = f'map seed {map_seed}_{agent_num} agents_agent seed {agent_seed}_{uncertainty} uncertainty_' \
            f'{objective}_using pc {use_pc}_using bypass {use_bp}, {map_type}.sol'
        path = os.path.join(solution_path, file_name)

        with open(path, 'w+') as sol_file:
            json_sol = {'paths': {}, 'constraints': None, 'time_to_solve': self.time_to_solve, 'sic': self.sic,
                        'nodes_generated': self.nodes_generated}
            for agent, path in self.paths.items():
                json_sol['paths'][agent] = path.path
            json_sol['constraints'] = list(self.constraints.items())
            json.dump(json_sol, sol_file)

    @staticmethod
    def load(agent_num, uncertainty, map_type, agent_seed, map_seed, min_best_case, use_pc, use_bp, folder):
        """
        Loads a previously computed solution.
        """
        objective = 'min best case' if min_best_case else 'min worst case'
        file_name = f'map seed {map_seed}_{agent_num} agents_agent seed {agent_seed}_{uncertainty} uncertainty_' \
            f'{objective}_using pc {use_pc}_using bypass {use_bp}, {map_type}.sol'
        path = os.path.join(folder, map_type, f'{agent_num} agents', file_name)

        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as sol_file:
                json_sol = json.load(sol_file)
                tu_sol = TimeUncertaintySolution()
                for agent, path in json_sol['paths'].items():
                    tuple_path = []
                    for presence in path:
                        tuple_path.append((tuple(presence[0]), tuple(presence[1])))
                    tu_plan = TimeUncertaintyPlan(int(agent), tuple_path, math.inf)
                    tu_sol.paths[int(agent)] = tu_plan
                for con in json_sol['constraints']:
                    if type(con[0][0]) == int:  # it's a vertex constraint
                        loc = tuple(con[0])
                    elif type(con[0][0]) == list:  # it's an edge constraint
                        loc = tuple(con[0][0]), tuple(con[0][1])
                    else:
                        raise TypeError
                    tuple_con = con[1][0][0], tuple(con[1][0][1])
                    tu_sol.constraints[loc].append(tuple_con)

                tu_sol.time_to_solve = json_sol['time_to_solve']
                tu_sol.nodes_generated = json_sol['nodes_generated']
                tu_sol.compute_solution_cost()
                tu_sol.create_movement_tuples()
                return tu_sol
        except:
            os.remove(path)  # Delete the messed up file
            return None