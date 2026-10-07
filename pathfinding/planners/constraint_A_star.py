"""
A class that represents an algorithm for solving the path for multiple agents / single agent in accordance with
a group of constraints.

Currently a naive implementation of A*

Author: Ajdin Sumic (2026)

The MIT License (MIT)

Copyright (c) 2026 ONERA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pathfinding.planners.utils.custom_heap import OpenListHeap
from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
from pathfinding.planners.utils.time_error import OutOfTimeError

import math
import heapq
import itertools
import networkx
import time

# The positions of each parameter in the tuple receives from map.edges
VERTEX_ID = 0
STAY_STILL_COST = 1


def _stable_key(value):
    """Deterministic ordering key for agents, vertices, intervals, constraints, etc."""
    if isinstance(value, dict):
        return tuple(sorted((_stable_key(k), _stable_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_stable_key(v) for v in value)
    return repr(value)


class ConstraintAstar:

    def __init__(self, tu_problem):
        self.tu_problem = tu_problem
        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = -1

    """
    Computes the path for a particular agent in a given node. The node should contain the new constraint for this
    agents. Note that the agent is simply an int (the agent's id).

    returns a tuple containing (agent_id, cost, [ path list])

    *** Currently naive implementation (basic A* with constraints)****
    """

    def compute_agent_path(self, constraints, agent, start_pos, goal_pos, conf_table, mbc=True, time_limit=5,
                           curr_time=(0, 0), pos_cons=None, suboptimal=False):

        self.open_list = OpenListHeap()
        self.open_dict = {}  # A dictionary mapping node tuple to the actual node. Used for faster access..
        self.closed_list = set()
        self.agent = agent
        start_time = time.time()
        start_node = SingleAgentNode(start_pos, None, curr_time, self.tu_problem, goal_pos, confs_created=0)
        self.__add_node_to_open(start_node, mbc)

        while len(self.open_list.internal_heap) > 0:
            if time.time() - start_time > time_limit:
                #raise OutOfTimeError('Ran out of time in low level solver')  # no solution
                return TimeUncertaintyPlan.get_empty_plan(agent)
            best_node = self.open_list.pop()  # removes from open_list

            if best_node.current_position == goal_pos and \
                    self.__can_stay(agent, best_node, constraints, suboptimal):
                #if self.tu_problem.calc_heuristic(start_pos, goal_pos) > 0:
                #    print(f'Ratio: {len(self.closed_list) / self.tu_problem.calc_heuristic(start_pos, goal_pos)}')
                return best_node.calc_path(agent)

            successors = best_node.expand(agent, constraints, conf_table, self.tu_problem, pos_cons, suboptimal)

            self.__remove_node_from_open(best_node)

            for neighbor in sorted(successors, key=_stable_key):
                if neighbor in self.closed_list:
                    continue
                g_val = neighbor[1]
                if neighbor not in self.open_dict:
                    neighbor_node = SingleAgentNode(neighbor[0], best_node, neighbor[1], self.tu_problem, goal_pos,
                                                    neighbor[2])
                    self.__add_node_to_open(neighbor_node, mbc)
                else:  # We've already reached this node but haven't expanded it.
                    neighbor_node = self.open_dict[neighbor]

                    if mbc and g_val[0] >= neighbor_node.g_val[0]:
                        continue  # No need to update node. Continue iterating successors
                    elif not mbc and g_val[1] >= neighbor_node.g_val[1]:
                        continue  # No need to update node. Continue iterating successors
                    print("****** Found a faster path for node *******" + neighbor_node.current_position)
                    self.__update_node(neighbor_node, best_node, g_val, goal_pos, self.tu_problem)
        return TimeUncertaintyPlan.get_empty_plan(agent)  # no solution



    def compute_agent_path_focal(self, constraints, agent, start_pos, goal_pos, conf_table,
                                 mbc=True, time_limit=5, curr_time=(0, 0), pos_cons=None,
                                 suboptimal=False, focal_w=1.1):
        """Safe bounded-suboptimal low-level focal search.

        A classical constrained A* run first computes an exact anchor path and
        therefore an exact cost bound C*.  The focal phase may prefer paths with
        fewer CAT conflicts, but it is never allowed to return a path whose
        primary cost exceeds ``focal_w * C*``.  If focal search times out or
        cannot improve the CAT ordering within the remaining budget, the anchor
        path is returned.  This prevents the very large path costs observed in
        the previous dynamic-threshold implementation.
        """
        if focal_w < 1:
            raise ValueError('focal_w must be >= 1')
        if time_limit <= 0:
            return TimeUncertaintyPlan.get_empty_plan(agent)

        overall_start = time.time()

        # Compute a safe optimal anchor with the original constrained A*.
        anchor = self.compute_agent_path(
            constraints, agent, start_pos, goal_pos, conf_table,
            mbc=mbc, time_limit=time_limit, curr_time=curr_time,
            pos_cons=pos_cons, suboptimal=suboptimal
        )
        if not getattr(anchor, 'path', None):
            return anchor

        anchor_cost = anchor.path[-1][0][0] if mbc else anchor.path[-1][0][1]
        try:
            anchor.low_level_f_min = float(anchor_cost)
        except Exception:
            pass

        # With no useful secondary signal, the anchor is already the right result.
        if focal_w == 1 or not conf_table:
            return anchor

        remaining = time_limit - (time.time() - overall_start)
        if remaining <= 0:
            return anchor
        deadline = time.time() + remaining
        absolute_bound = float(focal_w) * float(anchor_cost)

        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = agent

        counter = itertools.count()
        records = {}          # state key -> (version, node)
        open_heap = []        # (f, order, key, version)
        promote_heap = []     # nodes not yet promoted to focal
        focal_heap = []       # secondary ordering
        focal_members = set()
        last_bound = -math.inf

        def state_key_from_values(position, g_val):
            return position, g_val

        def state_key(node):
            return state_key_from_values(node.current_position, node.g_val)

        def primary_g(g_val):
            return g_val[0] if mbc else g_val[1]

        def f_value(node):
            return primary_g(node.g_val) + node.h_val

        def alive(key, version):
            rec = records.get(key)
            return rec is not None and rec[0] == version

        def push_focal(key, version, node):
            token = (key, version)
            if token in focal_members or not alive(key, version):
                return
            heapq.heappush(
                focal_heap,
                (
                    node.confs_created,
                    f_value(node),
                    -primary_g(node.g_val),
                    _stable_key(node.current_position),
                    _stable_key(node.g_val),
                    next(counter), key, version,
                )
            )
            focal_members.add(token)

        def insert_or_replace(node):
            key = state_key(node)
            old = records.get(key)
            version = 0 if old is None else old[0] + 1
            records[key] = (version, node)
            self.open_dict[key] = node
            item = (f_value(node), next(counter), key, version)
            heapq.heappush(open_heap, item)
            heapq.heappush(promote_heap, item)
            if f_value(node) <= last_bound + 1e-12:
                push_focal(key, version, node)

        def clean_open():
            while open_heap:
                f, _order, key, version = open_heap[0]
                if alive(key, version):
                    node = records[key][1]
                    if abs(f - f_value(node)) <= 1e-12:
                        return
                heapq.heappop(open_heap)

        def current_f_min():
            clean_open()
            return open_heap[0][0] if open_heap else math.inf

        def rebuild_focal(bound):
            nonlocal promote_heap
            focal_heap.clear()
            focal_members.clear()
            promote_heap = []
            for key, (version, node) in records.items():
                item = (f_value(node), next(counter), key, version)
                heapq.heappush(promote_heap, item)
                if item[0] <= bound + 1e-12:
                    push_focal(key, version, node)

        def promote(bound):
            nonlocal last_bound
            if bound + 1e-12 < last_bound:
                rebuild_focal(bound)
            last_bound = bound
            while promote_heap and promote_heap[0][0] <= bound + 1e-12:
                _f, _order, key, version = heapq.heappop(promote_heap)
                if alive(key, version):
                    push_focal(key, version, records[key][1])

        def pop_focal(bound):
            while focal_heap:
                (_conf, _f, _neg_g, _pos_key, _g_key, _order,
                 key, version) = heapq.heappop(focal_heap)
                focal_members.discard((key, version))
                if not alive(key, version):
                    continue
                node = records[key][1]
                if f_value(node) > bound + 1e-12:
                    continue
                records.pop(key, None)
                self.open_dict.pop(key, None)
                self.closed_list.add(key)
                return node
            return None

        start_node = SingleAgentNode(
            start_pos, None, curr_time, self.tu_problem, goal_pos,
            confs_created=0
        )
        insert_or_replace(start_node)

        while records:
            if time.time() > deadline:
                return anchor

            f_min = current_f_min()
            if f_min == math.inf or f_min > absolute_bound + 1e-12:
                return anchor

            # Dynamic ECBS threshold, additionally capped by the proven anchor bound.
            bound = min(float(focal_w) * float(f_min), absolute_bound)
            promote(bound)

            best_node = pop_focal(bound)
            if best_node is None:
                rebuild_focal(bound)
                best_node = pop_focal(bound)
                if best_node is None:
                    return anchor

            if best_node.current_position == goal_pos and \
                    self.__can_stay(agent, best_node, constraints, suboptimal):
                candidate_cost = primary_g(best_node.g_val)
                if candidate_cost <= absolute_bound + 1e-12:
                    plan = best_node.calc_path(agent)
                    try:
                        plan.low_level_f_min = float(anchor_cost)
                    except Exception:
                        pass
                    return plan
                # Defensive only: an over-bound goal is never accepted.

            successors = best_node.expand(
                agent, constraints, conf_table, self.tu_problem,
                pos_cons, suboptimal
            )

            for position, g_val, confs_created in sorted(successors, key=_stable_key):
                if time.time() > deadline:
                    return anchor

                # States that can no longer lead to a bounded solution are pruned.
                node_f = primary_g(g_val) + self.tu_problem.calc_heuristic(position, goal_pos)
                if node_f > absolute_bound + 1e-12:
                    continue

                key = state_key_from_values(position, g_val)
                if key in self.closed_list:
                    continue

                old = records.get(key)
                if old is None:
                    insert_or_replace(SingleAgentNode(
                        position, best_node, g_val, self.tu_problem,
                        goal_pos, confs_created
                    ))
                    continue

                old_node = old[1]
                new_primary = primary_g(g_val)
                old_primary = primary_g(old_node.g_val)
                improves = new_primary < old_primary
                tie_improves = (
                    new_primary == old_primary
                    and confs_created < old_node.confs_created
                )
                if not improves and not tie_improves:
                    continue

                insert_or_replace(SingleAgentNode(
                    position, best_node, g_val, self.tu_problem,
                    goal_pos, confs_created
                ))

        return anchor

    def __remove_node_from_open(self, node):
        node_tuple = node.create_tuple()
        self.open_dict.pop(node_tuple, None)
        self.closed_list.add(node_tuple)

    def __add_node_to_open(self, node, min_best_case):
        if min_best_case:  # minimize the lower time bound
            self.open_list.push(node, node.g_val[0] + node.h_val, node.confs_created, -node.g_val[0])
        else:  # minimize the upper time bound ToDo: What is a better tie breaker? g or conflicts?
            self.open_list.push(node, node.g_val[1] + node.h_val, node.confs_created, -node.g_val[1])
        key_tuple = node.create_tuple()
        self.open_dict[key_tuple] = node

        return node

    @staticmethod
    def __update_node(neighbor_node, prev_node, g_val, goal_pos, grid_map):
        neighbor_node.prev_node = prev_node
        neighbor_node.g_val = g_val
        neighbor_node.h_val = grid_map.calc_heuristic(neighbor_node.current_position, goal_pos)
        neighbor_node.f_val = g_val[0] + neighbor_node.h_val, g_val[1] + neighbor_node.h_val

    def __can_stay(self, agent, best_node, constraints, suboptimal):
        """
        A function that verifies that the agent has reached the goal at an appropriate time. Useful when an agent
        reaches the goal in, for example, 5-8 time units however in the solution it takes another agent 10-15 time
        units.
        Therefor we must verify what happens if this agent stands still all this time (there might be a conflict!)
        :param agent: The agent that's searching
        :param best_node: The current path
        :param constraints: A set of constraints
        :param suboptimal: If true, constraints apply to all agents and not the typical (agent, vertex, time) setting.
        this means that constraints are just (vertex, time) and apply to all agents. Used in prioritized planning since
        there is no need to iterate over all of the constraints.
        :return: True if the agent can stay in the current place, False otherwise.
        """
        if suboptimal:
            if best_node.current_position in constraints:
                # max_t = max(t[1] for t in constraints[best_node.current_position])
                for tick in sorted(constraints[best_node.current_position], key=lambda k: k[1]):
                    if best_node.g_val[0] <= tick[1]:
                        self.__create_wait_node(best_node, tick[1])
                        self.open_dict = {}
                        self.open_list = OpenListHeap()
                        return False
            return True
        if best_node.current_position in constraints:
            for con in sorted(constraints[best_node.current_position], key=_stable_key):
                if con[0] == agent and best_node.g_val[0] <= con[1][1]:
                    #print('found goal constraint: agent {} is at goal at time {}, constraint at time {}'.format(agent, best_node.g_val[0], con[1][1]))
                    return False  # A constraint was found

        return True

    def dijkstra_solution(self, source_vertex, min_best_case=False):
        """
        A function that calculates, using Dijkstra's algorithm, the distance from each point in the map to the given
        goal. This will be used as a perfect heuristic. It receives as input the goal position and returns a dictionary
        mapping each coordinate to the distance from it to the goal.

        For now we will use the minimum time taken to pass an edge as the weight, in order to keep the heuristic
        admissible.
        """

        graph = networkx.Graph()
        for vertex, edges in sorted(self.tu_problem.edges_and_weights.items(), key=lambda item: _stable_key(item[0])):
            for edge in sorted(edges, key=_stable_key):
                if min_best_case:
                    graph.add_edge(vertex, edge[0], weight=edge[1][0])
                else:
                    graph.add_edge(vertex, edge[0], weight=edge[1][1])
        try:
            return networkx.single_source_dijkstra_path_length(graph, source_vertex)
        except ValueError:
            print("neg value wut")

    @staticmethod
    def overlapping(time_1, time_2):
        """ Returns true if the time intervals in 'time_1' and 'time_2' overlap.

        1: A<-- a<-->b -->B
        2: a<-- A -->b<---B>
        3: A<-- a<--- B --->b
        4: a<-- A<--->B --->b
        5: A<-->B
           a<-->b
        ===> A <= a <= B or a <= A <= b
        """
        return (time_1[0] <= time_2[0] <= time_1[1]) or (time_2[0] <= time_1[0] <= time_2[1])

    def __create_wait_node(self, best_node, con_time):
        """
        Function called when an agent has a constraint on their goal vertex in the future. We change the current state
        to be one where the agent simply waits at the goal and then continue the search from there.
        :param best_node: The current state
        :param con_time: The time of the constraint on the goal
        :return: Changes the current state.
        """
        curr_pos = best_node.current_position
        temp_best = SingleAgentNode(curr_pos, best_node.prev_node, best_node.g_val, self.tu_problem, curr_pos,
                                    best_node.confs_created)
        delta = best_node.g_val[1] - best_node.g_val[0]
        g_val = (best_node.g_val[0] + 1, best_node.g_val[0] + 1 + delta)
        prev_node = SingleAgentNode(curr_pos, temp_best, g_val, self.tu_problem, curr_pos, best_node.confs_created)
        for g in range(prev_node.g_val[0] + 1, con_time - 1):
            g_val = g, g + delta
            curr_node = SingleAgentNode(curr_pos, prev_node, g_val, self.tu_problem, curr_pos, best_node.confs_created)
            prev_node = curr_node

        best_node.prev_node = prev_node
        best_node.g_val = g_val[0] + 1, g_val[1] + 1


class SingleAgentNode:
    """
    The class that represents the nodes being created during the search for a single agent.
    """

    def __init__(self, current_position, prev_node, g, grid_map, goal, confs_created):
        self.current_position = current_position
        self.prev_node = prev_node
        self.h_val = grid_map.calc_heuristic(current_position, goal)
        self.g_val = g  # The time to reach the node.
        self.confs_created = confs_created
        self.f_val = self.g_val[0] + self.h_val, self.g_val[1] + self.h_val

    """
    The function that creates all the possible vertices an agent can go to from the current node. For example,
    if a tuple from the map would be ((3,4), 1, 3) and the current time vector is t0, the agent can be at vertex (3,4)
    at times t0+1, t0+2 or t0+3.
    """

    def expand(self, agent, constraints, conflict_table, search_map, pos_cons, suboptimal):
        neighbors = []

        still_time = (self.g_val[0] + STAY_STILL_COST, self.g_val[1] + STAY_STILL_COST)  # Add the option of not moving.
        if self.legal_move(agent, self.current_position, still_time, constraints, pos_cons, suboptimal):
            if len(conflict_table) > 0:
                confs_created = self.confs_created + self.count_conflicts(agent, conflict_table, (still_time[1], still_time[1]),
                                                                          (self.current_position, self.current_position))
                stay_still = (self.current_position, still_time, confs_created)
            else:
                stay_still = (self.current_position, still_time, 0)
            neighbors.append(stay_still)

        for edge_tuple in sorted(search_map.edges_and_weights[self.current_position], key=_stable_key):
            successor_time = (self.g_val[0] + edge_tuple[1][0], self.g_val[1] + edge_tuple[1][1])
            vertex = edge_tuple[VERTEX_ID]
            if self.legal_move(agent, vertex, successor_time, constraints, pos_cons, suboptimal):
                if len(conflict_table) > 0:
                    confs_created = self.confs_created + self.count_conflicts(agent, conflict_table, successor_time,
                                                                              (self.current_position, vertex))
                    successor = (vertex, successor_time, confs_created)
                else:
                    successor = (vertex, successor_time, 0)

                neighbors.append(successor)

        return neighbors

    @staticmethod
    def count_conflicts(agent, conflict_table, succ_time, edge):
        """
        Checks how many new conflict will arise from traversing this edge, whether it leads to an edge conflict or a
        vertex conflict.
        :param agent: The agent we're currently planning for.
        :param conflict_table: The Conflict Avoidance Table.
        :param succ_time: The time we'll be occupying the vertex
        :param edge: The edge we're traversing. If we are staying in the same spot then it'll be (v, v) and not (v, u)
        :return: Number of overlapping time ticks.
        """
        new_confs = 0
        for other, locations in sorted(conflict_table.items(), key=lambda item: _stable_key(item[0])):
            if other == agent:
                continue
            if edge[1] in locations:
                for pres in sorted(locations[edge[1]], key=_stable_key):
                    if (pres[0] <= succ_time[0] <= pres[1]) or (succ_time[0] <= pres[0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[1]) - max(succ_time[0], pres[0]) + 1
            if edge in locations:
                for pres in sorted(locations[edge], key=_stable_key):
                    if (pres[0][0] <= succ_time[0] <= pres[0][1]) or (succ_time[0] <= pres[0][0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[0][1]) - max(succ_time[0], pres[0][0]) + 1

        return new_confs

    def calc_path(self, agent):
        """
        returns the path calculated from the node by traversing the previous nodes.
        """
        path = []
        curr_node = self
        while curr_node:
            move = (curr_node.g_val, curr_node.current_position)
            path.insert(0, move)
            curr_node = curr_node.prev_node
        return TimeUncertaintyPlan(agent, path, self.g_val)

    def create_tuple(self):
        """
        Converts a node object into a tuple.
        (vertex, time)
        """
        return self.current_position, self.g_val, self.confs_created

    def legal_move(self, agent, vertex, succ_time, constraints, pos_cons, suboptimal):  # ToDo: reduce this bottleneck

        """
        A function that checks if a certain movement is legal. First we check for vertex constraints and then edge
        constraints. The loop checks for the time range that the agent might be at the next vertex.
        vertex - The node the agent is traveling to
        time - The arrival time at vertex
        constraints - A set of constraints.

        Returns false if move is illegal. Returns true if move is fine.
        """
        edge = min(self.current_position, vertex), max(self.current_position, vertex)
        edge_time = self.calc_edge_time(succ_time)
        if suboptimal:
            if edge in constraints:
                for tick in sorted(constraints[edge], key=_stable_key):
                    if (tick[0] <= edge_time[0] <= tick[1]) or (edge_time[0] <= tick[0] <= edge_time[1]):
                        return False
            if vertex in constraints:
                for tick in sorted(constraints[vertex], key=_stable_key):
                    if (tick[0] <= succ_time[0] <= tick[1]) or (succ_time[0] <= tick[0] <= succ_time[1]):
                        return False
            return True
        if pos_cons:  # Only check positive constraints if it exists
            for tick in range(succ_time[0], succ_time[1] + 1):
                if (agent, vertex, tick) not in pos_cons:
                    return False

        if edge in constraints:
            for con in sorted(constraints[edge], key=_stable_key):
                if con[0] == agent and (
                        (con[1][0] <= edge_time[0] <= con[1][1]) or (edge_time[0] <= con[1][1] <= edge_time[1])):
                    return False
        if vertex in constraints:
            for con in sorted(constraints[vertex], key=_stable_key):
                if con[0] == agent and (
                        (con[1][0] <= succ_time[0] <= con[1][1]) or (succ_time[0] <= con[1][1] <= succ_time[1])):
                    return False
        return True

    def calc_edge_time(self, succ_time):
        """
        Calculates how much time an agent will occupy the edge it is traversing
        :param succ_time: The time at successor node
        :return:
        """
        if (succ_time[0] - self.g_val[0], succ_time[1] - self.g_val[1]) == (1, 1):
            return self.g_val[0], succ_time[1]
        return self.g_val[0], succ_time[1]