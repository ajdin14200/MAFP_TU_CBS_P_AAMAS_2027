"""Implementation of a conformant variation of the Conlift-Based Search and Meta-agent Conflict based
search algorithms from

Meta-agent Conflict-Based Search for Optimal Multi-Agent Path Finding by
Guni Sharon, Roni Stern, Ariel Felner and Nathan Sturtevant

This implementation uses maze generation code from https://github.com/boppreh/maze

Basic idea is independent planning, then check for pairwise collisions.  Branch
into two separate searches, which each require that one robot avoids that
particular space-time position.  The search over the conflict tree continues
until a set of collision free paths are found.

constraints will be of the form:
(agent_ID, conflict_node, time) <--- vertex constraint
(agent_ID, (prev_node, conflict_node), time) <--- edge constraint

where each disallowed state will have the form (agent, node/edge, time)
The time for an edge constraint refers to the time at which the agent occupies the edge. Not using objects so I can
use constraints in dictionary keys.

This project adds two aspects to the planner:
1) The edges are weighted.
2) The time that it takes to pass an edge is not static, meaning that it won't necessarily be completed in 1 time-step,
but it can be anywhere between (for example) 2-5.

*Note that a weight of 1 for all edges and a time range of (1,1) is simply the regular CBS.

The goal is to find an optimal solution that guarantees that no two agents will collide.

Written by: Tomer Shahar, AI search lab, department of Software & Information Systems Engineering.
-- August 2018
"""
import math
import time
import copy
import heapq
import itertools

from pathfinding.planners.constraint_A_star import ConstraintAstar as Cas
from pathfinding.planners.utils.constraint_node import ConstraintNode as Cn, ConstraintNode
from pathfinding.planners.utils.time_error import OutOfTimeError
from pathfinding.planners.utils.time_uncertainty_solution import TimeUncertaintySolution
from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
from pathfinding.planners.utils.custom_heap import OpenListHeap
from collections import defaultdict

STAY_STILL_COST = 1


def _stable_key(value):
    """Return a deterministic, homogeneous string key for sorting.

    Important: this function must always return a string. Returning tuples can
    fail when nested values mix types such as tuple and str.
    """
    if isinstance(value, dict):
        items = sorted(
            (_stable_key(k), _stable_key(v)) for k, v in value.items()
        )
        return "dict(" + ",".join(f"{k}:{v}" for k, v in items) + ")"

    if isinstance(value, (set, frozenset)):
        items = sorted(_stable_key(v) for v in value)
        return type(value).__name__ + "(" + ",".join(items) + ")"

    if isinstance(value, (list, tuple)):
        items = [_stable_key(v) for v in value]
        return type(value).__name__ + "(" + ",".join(items) + ")"

    return type(value).__name__ + ":" + repr(value)


class _EECBSOpen:
    """Heap-backed OPEN/CLEANUP/FOCAL manager with lazy deletion.

    Node keys are computed once at insertion. Nodes are immutable while they
    remain open; bypassed nodes are removed and reinserted after their paths
    have changed. This avoids the repeated full-list scans that dominated the
    previous Python implementation.
    """

    def __init__(self, planner, focal_w, conflict_weight, pair_weight, edge_weight,
                 use_wdg, wdg_weight, wdg_overlap_weight, wdg_cap_pair_weight):
        self.planner = planner
        self.focal_w = focal_w
        self.params = dict(
            conflict_weight=conflict_weight, pair_weight=pair_weight,
            edge_weight=edge_weight, use_wdg=use_wdg, wdg_weight=wdg_weight,
            wdg_overlap_weight=wdg_overlap_weight,
            wdg_cap_pair_weight=wdg_cap_pair_weight,
        )
        self.nodes = {}
        self.cleanup_heap = []
        self.open_heap = []
        # Heap of admissible lower bounds, used only to compute the
        # global EECBS reference bound w * min_LB.
        self.lb_heap = []
        # Heap of actual current CT-node costs, used to decide FOCAL
        # eligibility. A low lower bound alone must never make an arbitrarily
        # expensive node eligible.
        self.eligibility_heap = []
        self.focal_heap = []
        self.focal_members = set()
        self.counter = itertools.count()
        self.last_bound = -math.inf

    def __len__(self):
        return len(self.nodes)

    def _alive(self, sig, node):
        return self.nodes.get(sig) is node

    def _clean(self, heap):
        while heap:
            _key, _order, sig, node = heap[0]
            if self._alive(sig, node):
                return
            heapq.heappop(heap)

    def _push_focal(self, sig, node):
        if sig in self.focal_members or not self._alive(sig, node):
            return
        key = self.planner._eecbs_focal_key(node, **self.params)
        heapq.heappush(self.focal_heap, (key, next(self.counter), sig, node))
        self.focal_members.add(sig)

    def insert(self, node, count_generated=True):
        if not self.planner._is_valid_ct_node(node):
            return False
        sig = self.planner._eecbs_signature(node)
        if sig in self.planner.closed_nodes or sig in self.nodes:
            return False
        self.nodes[sig] = node
        order = next(self.counter)
        cleanup_key = self.planner._eecbs_cleanup_key(node)
        open_key = self.planner._eecbs_open_key(node, **self.params)
        lb = self.planner._node_lower_bound(node)
        actual_cost = self.planner._node_primary_cost(node)
        heapq.heappush(self.cleanup_heap, (cleanup_key, order, sig, node))
        heapq.heappush(self.open_heap, (open_key, order, sig, node))
        heapq.heappush(self.lb_heap, (lb, order, sig, node))
        heapq.heappush(self.eligibility_heap, (actual_cost, order, sig, node))
        if actual_cost <= self.last_bound:
            self._push_focal(sig, node)
        if count_generated and hasattr(self.planner.open_nodes, 'entry_count'):
            self.planner.open_nodes.entry_count += 1
        return True

    def _rebuild_focal(self, bound):
        """Rebuild FOCAL after a decreasing bound.

        FOCAL membership is based on the *actual current node cost*.  The
        admissible lower-bound heap is deliberately left untouched because it
        serves a different purpose: computing min_LB for the global bound.
        """
        self.focal_heap = []
        self.focal_members = set()
        self.eligibility_heap = []
        for sig, node in self.nodes.items():
            order = next(self.counter)
            actual_cost = self.planner._node_primary_cost(node)
            heapq.heappush(
                self.eligibility_heap,
                (actual_cost, order, sig, node)
            )
            if actual_cost <= bound:
                self._push_focal(sig, node)

    def _promote(self, bound):
        """Promote nodes whose actual current cost enters FOCAL."""
        if bound + 1e-12 < self.last_bound:
            self._rebuild_focal(bound)
        self.last_bound = bound
        self._clean(self.eligibility_heap)
        while self.eligibility_heap and self.eligibility_heap[0][0] <= bound:
            _cost, _order, sig, node = heapq.heappop(self.eligibility_heap)
            if self._alive(sig, node):
                self._push_focal(sig, node)
            self._clean(self.eligibility_heap)

    def _peek(self, heap):
        self._clean(heap)
        return heap[0] if heap else None

    def best_cleanup_node(self):
        item = self._peek(self.cleanup_heap)
        return item[3] if item else None

    def pop_best(self):
        cleanup_item = self._peek(self.cleanup_heap)
        if cleanup_item is None:
            return None
        best_cleanup = cleanup_item[3]

        # The reference bound is computed from the smallest admissible
        # low-level lower bound in OPEN.  This is different from FOCAL
        # eligibility, which uses each node's actual current path cost.
        lb_item = self._peek(self.lb_heap)
        if lb_item is None:
            return None
        f_min = lb_item[0]
        bound = self.focal_w * f_min
        self._promote(bound)

        focal_item = self._peek(self.focal_heap)
        open_item = self._peek(self.open_heap)
        chosen = best_cleanup

        if focal_item is not None:
            focal_node = focal_item[3]
            _f, _h, _d, focal_f_hat = self.planner._eecbs_estimates(
                focal_node, **self.params
            )
            if (
                self.planner._node_primary_cost(focal_node) <= bound
                and focal_f_hat <= bound
            ):
                chosen = focal_node
            elif open_item is not None:
                open_node = open_item[3]
                _f, _h, _d, open_f_hat = self.planner._eecbs_estimates(
                    open_node, **self.params
                )
                if (
                    self.planner._node_primary_cost(open_node) <= bound
                    and open_f_hat <= bound
                ):
                    chosen = open_node
        elif open_item is not None:
            open_node = open_item[3]
            _f, _h, _d, open_f_hat = self.planner._eecbs_estimates(
                open_node, **self.params
            )
            if (
                self.planner._node_primary_cost(open_node) <= bound
                and open_f_hat <= bound
            ):
                chosen = open_node

        sig = self.planner._eecbs_signature(chosen)
        self.nodes.pop(sig, None)
        self.focal_members.discard(sig)
        return chosen


class CBSTUPlanner:
    """
    The class that represents the cbs solver. The input is a conformedMap - a map that also contains the weights
    of the edges and the time step range to complete the transfer.

    constraints - a set of constraints.

    """

    def __init__(self, tu_problem):

        self.tu_problem = tu_problem
        self.final_constraints = None
        self.start_time = 0
        self.planner = Cas(tu_problem)
        self.curr_time = (0, 0)
        self.root = None
        self.open_nodes = OpenListHeap()
        self.closed_nodes = set()
        self.min_best_case = False
        self.soc = True
        self.use_cat = True
        self.computed_c_nodes = {}  # Dictionary mapping constraints, conf_num -> Constraint Node

    def potential_presence(self, plans, weights):

        potential_presence = {}

        for agent, plan in sorted(plans.items(), key=lambda item: _stable_key(item[0])):

            agent_pp = [[[0, 0]]]

            for t in range (1, len(plan.path)) :
                previous_node = plan.path[t - 1][1]
                current_node = plan.path[t][1]

                previous_pp = agent_pp[t - 1]
                durations = weights[(previous_node, current_node)]
                l = durations[0]
                u = durations[1]

                node_pp = [[previous_pp[0][0] + l, previous_pp[0][1] + u]]

                agent_pp.append(node_pp)
            potential_presence[agent] = agent_pp

        return potential_presence

    def enforce_upper_bounds_global(self, I, J, time_limit, p=0.001):
        """
        Make the interval sets I and J pairwise disjoint while preserving
        their upper-bound semantics.

        Returns:
            (I, J)       successful reduction
            ([], [])     infeasible reduction
            (None, None) timeout
        """
        I = [list(interval) for interval in I]
        J = [list(interval) for interval in J]

        I.sort()
        J.sort()

        changed = True

        while changed:
            changed = False
            i = 0
            j = 0

            while i < len(I) and j < len(J):
                if time.time() - self.start_time > time_limit:
                    return None, None

                a_start, a_end = I[i]
                b_start, b_end = J[j]

                if a_end + p <= b_start:
                    i += 1
                    continue

                if b_end + p <= a_start:
                    j += 1
                    continue

                changed = True

                if a_end <= b_end:
                    new_b_start = a_end + p

                    # [t, t] is a valid instantaneous interval.
                    if new_b_start <= b_end:
                        J[j][0] = new_b_start
                        i += 1
                    else:
                        J.pop(j)

                else:
                    new_a_start = b_end + p

                    # [t, t] is a valid instantaneous interval.
                    if new_a_start <= a_end:
                        I[i][0] = new_a_start
                        j += 1
                    else:
                        I.pop(i)

        return I, J

    def enforce_upper_bounds_with_split(self, I, J, time_limit, p=0.001):
        """
        Enforce:
        - All upper bounds preserved
        - Lower bounds adjusted
        - No overlaps between I and J
        - During-case handled by splitting the outer interval
        """

        I = [list(x) for x in I]
        J = [list(x) for x in J]

        I.sort()
        J.sort()

        # -------------------------------------------------
        # PASS 1 — handle DURING cases via splitting
        # -------------------------------------------------

        changed = True
        while changed:
            changed = False

            if time.time() - self.start_time > time_limit:
                return [], []


            new_J = []
            for b_start, b_end in J:

                split_done = False

                for a_start, a_end in I:
                    # A during B  (A in I, B in J)
                    if b_start < a_start and a_end < b_end:

                        # left piece
                        if b_start < a_start - p:
                            new_J.append([b_start, a_start - p])

                        # right piece
                        if a_end + p < b_end:
                            new_J.append([a_end + p, b_end])

                        split_done = True
                        changed = True
                        break

                if not split_done:
                    new_J.append([b_start, b_end])

            J = new_J

            new_I = []
            for a_start, a_end in I:

                split_done = False

                for b_start, b_end in J:
                    # A during B (A in J, B in I)
                    if a_start < b_start and b_end < a_end:

                        if a_start < b_start - p:
                            new_I.append([a_start, b_start - p])

                        if b_end + p < a_end:
                            new_I.append([b_end + p, a_end])

                        split_done = True
                        changed = True
                        break

                if not split_done:
                    new_I.append([a_start, a_end])

            I = new_I
        # -------------------------------------------------
        # PASS 2 — global sweep with fallback shrinking
        # -------------------------------------------------

        I.sort()
        J.sort()

        changed = True

        while changed:
            changed = False

            i = 0
            j = 0

            while i < len(I) and j < len(J):

                if time.time() - self.start_time > time_limit:
                    return [], []

                a_start, a_end = I[i]
                b_start, b_end = J[j]

                # Already separated
                if a_end + p <= b_start:
                    i += 1
                    continue

                if b_end + p <= a_start:
                    j += 1
                    continue

                # Overlap or touching
                changed = True

                # Preserve both upper bounds
                if a_end <= b_end:
                    # shift J lower bound
                    new_b_start = a_end + p

                    if new_b_start < b_end:
                        J[j][0] = new_b_start
                        i += 1
                    else:
                        # remove invalid J interval
                        J.pop(j)

                else:
                    # shift I lower bound
                    new_a_start = b_end + p

                    if new_a_start < a_end:
                        I[i][0] = new_a_start
                        j += 1
                    else:
                        # remove invalid I interval
                        I.pop(i)

        return I, J


    def propagates_new_pp(self, plan, pp, t, edge_weights):

        for i in range(t, len(plan.path)):
            previous_node = plan.path[i - 1][1]
            current_node = plan.path[i][1]

            bounds = edge_weights[(previous_node, current_node)]
            #print("bounds : ", bounds)
            new_current_node_pp = []
            # print("pp : ", pp)
            for interval in pp[i - 1]:
                new_interval = [interval[0] + bounds[0], interval[1] + bounds[1]]
                new_current_node_pp.append(new_interval)

            # Keep interval sets canonical after propagation.
            pp[i] = self._merge_intervals(new_current_node_pp)

    def substract_intervals(self, interval_1, interval_2, p=0.001):

        l = interval_1[0]
        u = interval_1[1]
        u_2 = interval_2[1]


        case = 2

        if u > u_2 > l:
            interval_1 = [u_2 + p, u]
            case = 0

        elif u == u_2:
            interval_1 = []
            case = 1


        return interval_1, case

    @staticmethod
    def intervals_overlap(interval_1, interval_2):
        """Returns True iff two closed time intervals overlap.

        This is used before trying to reduce a potential-presence
        interval. If two agents visit the same vertex but their
        propagated time intervals are already disjoint, no repair is
        needed.
        """
        return interval_1 and interval_2 and Cas.overlapping(interval_1, interval_2)



    @staticmethod
    def _separated(interval_a, interval_b, p=0.001):
        """True iff two intervals are separated by at least p."""
        return interval_a[1] + p <= interval_b[0] or interval_b[1] + p <= interval_a[0]

    @staticmethod
    def _valid_interval(interval):
        """Arrival intervals may be instantaneous, so [t,t] is valid."""
        return interval is not None and len(interval) == 2 and interval[0] <= interval[1]

    def _canonical_intervals(self, intervals, p=0.001):
        """Remove empty intervals, sort, and merge overlapping intervals.

        Intervals that overlap are merged. Intervals separated by a positive gap
        are kept distinct.
        """
        clean = []
        for interval in intervals:
            if self._valid_interval(interval):
                clean.append([interval[0], interval[1]])

        if not clean:
            return []

        clean.sort(key=lambda x: (x[0], x[1]))
        result = [clean[0]]

        for l, u in clean[1:]:
            last = result[-1]
            # Merge only if they overlap. Do not merge merely because the gap is
            # smaller than p: the p-margin is for separating conflicts between
            # different agents, not for erasing subintervals of one policy.
            if l <= last[1]:
                last[1] = max(last[1], u)
            else:
                result.append([l, u])

        return result

    def _reduce_pair_preserve_upper(self, A, B, p=0.001):
        """Resolve overlap between one interval A and one interval B.

        Returns (new_A_intervals, new_B_intervals), or (None, None) if the
        overlap cannot be removed while preserving the upper-bound semantic.

        The invariant is:
        - if A survives, at least one returned A subinterval ends at A[1];
        - if B survives, at least one returned B subinterval ends at B[1].

        Examples:
            A=[5,10], B=[6,8]
            -> A=[[5,6-p],[8+p,10]], B=[[6,8]]

            A=[5,10], B=[8,12]
            -> A=[[5,10]], B=[[10+p,12]]
        """
        a_l, a_u = A
        b_l, b_u = B

        if self._separated(A, B, p=p):
            return [A], [B]

        # If both intervals have the same upper bound, both cannot keep an
        # upper-preserving right part while becoming disjoint.
        if abs(a_u - b_u) <= p / 10:
            return None, None

        # B is strictly inside A: split A around B.
        if a_l < b_l and b_u < a_u:
            new_A = []
            left = [a_l, b_l - p]
            right = [b_u + p, a_u]
            if self._valid_interval(left):
                new_A.append(left)
            if self._valid_interval(right):
                new_A.append(right)
            if not new_A or not any(abs(x[1] - a_u) <= p / 10 for x in new_A):
                return None, None
            return new_A, [B]

        # A is strictly inside B: split B around A.
        if b_l < a_l and a_u < b_u:
            new_B = []
            left = [b_l, a_l - p]
            right = [a_u + p, b_u]
            if self._valid_interval(left):
                new_B.append(left)
            if self._valid_interval(right):
                new_B.append(right)
            if not new_B or not any(abs(x[1] - b_u) <= p / 10 for x in new_B):
                return None, None
            return [A], new_B

        # Partial overlap. Preserve the interval with the earlier upper bound,
        # and shift the lower bound of the interval with the later upper bound.
        if a_u < b_u:
            new_B = [a_u + p, b_u]
            if not self._valid_interval(new_B):
                return None, None
            return [A], [new_B]

        if b_u < a_u:
            new_A = [b_u + p, a_u]
            if not self._valid_interval(new_A):
                return None, None
            return [new_A], [B]

        return None, None

    def enforce_upper_bounds_with_minimal_split(self, I, J, time_limit,
                                                p=0.001, max_iterations=1000,
                                                max_intervals=256):
        """Remove overlaps between two interval sets with minimal splitting.

        This function preserves the upper-bound semantic: if an original
        interval [l, u] is reduced and remains feasible, the returned set keeps
        at least one subinterval ending at u.

        Example:
            I = [[5, 10]], J = [[6, 8]]
            returns [[5, 5.999], [8.001, 10]], [[6, 8]]

        It works both inside CBS and when called directly in a unit test. It
        returns (None, None) only on timeout, and ([], []) if no valid reduction
        exists.
        """
        # Use a local deadline when the method is tested directly before CBS
        # initializes self.start_time.
        start_time = self.start_time if getattr(self, "start_time", 0) else time.time()
        deadline = start_time + time_limit

        def out_of_time():
            return time.time() > deadline

        def valid(interval):
            return interval is not None and len(interval) == 2 and interval[0] <= interval[1]

        def canonical(intervals):
            clean = [list(x) for x in intervals if valid(x)]
            clean.sort(key=lambda x: (x[0], x[1]))
            merged = []
            for l, u in clean:
                if not merged or l > merged[-1][1]:
                    merged.append([l, u])
                else:
                    merged[-1][1] = max(merged[-1][1], u)
            return merged

        def separated(A, B):
            return A[1] + p <= B[0] or B[1] + p <= A[0]

        def reduce_pair(A, B):
            a_l, a_u = A
            b_l, b_u = B

            if separated(A, B):
                return [A], [B]

            # Same upper bound: both intervals cannot keep a surviving piece
            # ending at their original upper bound while being made disjoint.
            if abs(a_u - b_u) <= p / 10:
                return None, None

            # B is strictly inside A: split A around B.
            if a_l < b_l and b_u < a_u:
                new_A = []
                left = [a_l, b_l - p]
                right = [b_u + p, a_u]
                if valid(left):
                    new_A.append(left)
                if valid(right):
                    new_A.append(right)
                return new_A, [B]

            # A is strictly inside B: split B around A.
            if b_l < a_l and a_u < b_u:
                new_B = []
                left = [b_l, a_l - p]
                right = [a_u + p, b_u]
                if valid(left):
                    new_B.append(left)
                if valid(right):
                    new_B.append(right)
                return [A], new_B

            # Partial overlap. Keep the interval with the earlier upper bound
            # unchanged and shift the lower bound of the one with the later
            # upper bound.
            if a_u < b_u:
                new_B = [a_u + p, b_u]
                if not valid(new_B):
                    return None, None
                return [A], [new_B]

            if b_u < a_u:
                new_A = [b_u + p, a_u]
                if not valid(new_A):
                    return None, None
                return [new_A], [B]

            return None, None

        I_curr = canonical(I)
        J_curr = canonical(J)

        if not I_curr or not J_curr:
            return [], []

        for _ in range(max_iterations):
            if out_of_time():
                return None, None

            changed = False

            for i, A in enumerate(list(I_curr)):
                for j, B in enumerate(list(J_curr)):
                    if separated(A, B):
                        continue

                    new_A, new_B = reduce_pair(A, B)
                    if new_A is None or new_B is None:
                        # Equal-upper-bound conflict for this component.  A
                        # component may be discarded when the same potential
                        # presence has another surviving component.  If neither
                        # side has an alternative component, the whole conflict
                        # is genuinely unreducible.
                        if len(I_curr) > 1:
                            I_curr = canonical(I_curr[:i] + I_curr[i + 1:])
                            changed = True
                            break
                        if len(J_curr) > 1:
                            J_curr = canonical(J_curr[:j] + J_curr[j + 1:])
                            changed = True
                            break
                        return [], []

                    I_curr = canonical(I_curr[:i] + new_A + I_curr[i + 1:])
                    J_curr = canonical(J_curr[:j] + new_B + J_curr[j + 1:])

                    if not I_curr or not J_curr:
                        return [], []
                    if len(I_curr) > max_intervals or len(J_curr) > max_intervals:
                        return [], []

                    changed = True
                    break
                if changed:
                    break

            if not changed:
                return I_curr, J_curr

        return None, None


    def _merge_intervals(self, intervals, p=0.001):
        return self._canonical_intervals(intervals, p=p)

    def _normalize_interval_list(self, intervals, p=0.001):
        return self._canonical_intervals(intervals, p=p)


    def _canonicalize_policy_or_fail(self, potential_presence, plans=None):
        """Canonicalize all interval sets and reject infeasible policies.

        Empty subintervals are removed by _canonical_intervals. If a whole
        vertex occurrence has no admissible interval left, [] is returned.
        """
        if potential_presence is None:
            return None
        if potential_presence == []:
            return []

        if plans is not None:
            for agent, plan in plans.items():
                if agent not in potential_presence:
                    return []
                if len(potential_presence[agent]) != len(plan.path):
                    return []

        for agent, pp_agent in potential_presence.items():
            if pp_agent is None or pp_agent == []:
                return []
            for idx, intervals in enumerate(pp_agent):
                pp_agent[idx] = self._canonical_intervals(intervals)
                if not pp_agent[idx]:
                    return []

        return potential_presence

    @staticmethod
    def _policy_has_empty_interval_set(potential_presence):
        """True iff some vertex occurrence has no admissible interval."""
        if potential_presence is None or potential_presence == []:
            return True
        for _agent, pp_agent in potential_presence.items():
            if pp_agent is None or pp_agent == []:
                return True
            for intervals in pp_agent:
                if not intervals:
                    return True
        return False

    def compute_valid_schedule_UB_minimal_split(self, plans, potential_presence,
                                                edge_weights, time_limit,
                                                max_outer_iterations=10000,
                                                max_intervals=256):
        """Split-based version of compute_valid_schedule_UB.

        This keeps the same execution-policy semantics: the policy remains a
        list of admissible arrival intervals for every vertex occurrence.

        Difference with compute_valid_schedule_UB:
          - a vertex occurrence may now have several admissible subintervals;
          - when an overlap is found, the procedure removes only the conflicting
            part of the interval that must be delayed, instead of replacing it
            by a single right-side interval.

        The procedure is run to a fixpoint because splitting and propagation can
        expose conflicts that were not present in the current scan order.
        It is bounded by time_limit, max_outer_iterations, and max_intervals.
        """
        outer_iteration = 0
        changed = True

        while changed:
            if time.time() - self.start_time > time_limit:
                return None

            outer_iteration += 1
            if outer_iteration > max_outer_iterations:
                return []

            changed = False

            for agent, plan in sorted(plans.items(), key=lambda item: _stable_key(item[0])):

                for t in range(1, len(plan.path)):

                    if time.time() - self.start_time > time_limit:
                        return None

                    agent_node = plan.path[t][1]
                    previous_agent_node = plan.path[t - 1][1]

                    for other_agent, other_agent_plan in sorted(plans.items(), key=lambda item: _stable_key(item[0])):

                        if other_agent == agent:
                            continue

                        # Deterministically avoid processing the same pair twice
                        # in the same outer scan.
                        if _stable_key(agent) > _stable_key(other_agent):
                            continue

                        for t2 in range(1, len(other_agent_plan.path)):

                            if time.time() - self.start_time > time_limit:
                                return None

                            other_agent_node = other_agent_plan.path[t2][1]
                            previous_other_agent_node = other_agent_plan.path[t2 - 1][1]

                            if agent_node != other_agent_node:
                                continue

                            I = potential_presence[agent][t]
                            J = potential_presence[other_agent][t2]

                            if len(I) > max_intervals or len(J) > max_intervals:
                                return []

                            if not self._interval_lists_overlap(I, J):
                                continue

                            agent_is_waiting = previous_agent_node == agent_node
                            other_agent_is_waiting = previous_other_agent_node == other_agent_node
                            if agent_is_waiting or other_agent_is_waiting:
                                return []

                            new_I, new_J = self.enforce_upper_bounds_with_minimal_split(
                                I, J, time_limit,
                                max_intervals=max_intervals
                            )

                            if not new_I and not new_J:
                                return []

                            if new_I != I:
                                potential_presence[agent][t] = new_I
                                self._propagate_agent_from(
                                    plans, potential_presence,
                                    edge_weights, agent, t + 1
                                )
                                changed = True

                            if new_J != J:
                                potential_presence[other_agent][t2] = new_J
                                self._propagate_agent_from(
                                    plans, potential_presence,
                                    edge_weights, other_agent, t2 + 1
                                )
                                changed = True

        return potential_presence


    def _apply_edge_order_candidate(self, plans, potential_presence, edge_weights,
                                    first_agent, first_move,
                                    second_agent, second_move,
                                    p=0.001):
        """Single-interval edge-order candidate.

        Enforce that first_agent crosses the opposite edge before second_agent
        by reducing the source vertex interval of second_agent.

        This keeps the execution-policy semantics: only vertex occurrence
        intervals are reduced.
        """
        pp = copy.deepcopy(potential_presence)

        first_arrival_interval = pp[first_agent][first_move + 1][0]
        second_source_interval = pp[second_agent][second_move][0]

        required_lower = first_arrival_interval[1] + p
        new_lower = max(second_source_interval[0], required_lower)

        if new_lower > second_source_interval[1]:
            return None

        pp[second_agent][second_move][0] = [new_lower, second_source_interval[1]]

        self.propagates_new_pp(
            plans[second_agent],
            pp[second_agent],
            second_move + 1,
            edge_weights
        )

        return pp

    def _apply_edge_order_candidate_minimal_split(self, plans, potential_presence,
                                                  edge_weights,
                                                  first_agent, first_move,
                                                  second_agent, second_move,
                                                  p=0.001):
        """Split-compatible version of isolated edge ordering.

        It still reduces a vertex occurrence, not an edge interval. The second
        agent is only allowed to arrive at the source vertex of the conflicting
        edge after first_agent has certainly reached the opposite endpoint.
        """
        pp = copy.deepcopy(potential_presence)

        first_arrival_upper = max(
            interval[1] for interval in pp[first_agent][first_move + 1]
        )
        required_lower = first_arrival_upper + p

        old_source_intervals = pp[second_agent][second_move]
        new_source_intervals = []

        for l, u in old_source_intervals:
            if u <= required_lower:
                continue
            new_source_intervals.append([max(l, required_lower), u])

        new_source_intervals = self._normalize_interval_list(new_source_intervals, p)
        if not new_source_intervals:
            return None

        pp[second_agent][second_move] = new_source_intervals
        self.propagates_new_pp(
            plans[second_agent],
            pp[second_agent],
            second_move + 1,
            edge_weights
        )

        return pp

    def compute_valid_schedule_UB_with_isolated_edges_minimal_split(
            self, plans, potential_presence, edge_weights, time_limit,
            max_edge_iterations=10000, max_intervals=256):
        """Isolated-edge repair plus split-based vertex repair.

        Edge conflicts are still handled conservatively:
          - reversed triplets are not repaired and fall back to CBS;
          - isolated conflicts choose the ordering with the smallest immediate
            interval-measure loss.
        """
        edge_iteration = 0

        while True:
            if time.time() - self.start_time > time_limit:
                return None

            edge_iteration += 1
            if edge_iteration > max_edge_iterations:
                return []

            conflict = self._first_edge_conflict_in_pp(plans, potential_presence)
            if conflict is None:
                break

            agent_i, move_i, agent_j, move_j = conflict
            plan_i = plans[agent_i]
            plan_j = plans[agent_j]

            if self._belongs_to_reversed_triplet(plan_i, move_i, plan_j, move_j):
                return []

            candidates = []

            before_j_measure = self._total_interval_measure(
                potential_presence[agent_j][move_j]
            )
            pp_ij = self._apply_edge_order_candidate_minimal_split(
                plans, potential_presence, edge_weights,
                agent_i, move_i, agent_j, move_j
            )
            if pp_ij is not None:
                after_j_measure = self._total_interval_measure(pp_ij[agent_j][move_j])
                loss = before_j_measure - after_j_measure
                candidates.append((loss, _stable_key((agent_i, agent_j)), pp_ij))

            before_i_measure = self._total_interval_measure(
                potential_presence[agent_i][move_i]
            )
            pp_ji = self._apply_edge_order_candidate_minimal_split(
                plans, potential_presence, edge_weights,
                agent_j, move_j, agent_i, move_i
            )
            if pp_ji is not None:
                after_i_measure = self._total_interval_measure(pp_ji[agent_i][move_i])
                loss = before_i_measure - after_i_measure
                candidates.append((loss, _stable_key((agent_j, agent_i)), pp_ji))

            if not candidates:
                return []

            candidates.sort(key=lambda x: (x[0], x[1]))
            potential_presence = candidates[0][2]

            # Avoid interval explosion.
            for pp_agent in potential_presence.values():
                for intervals in pp_agent:
                    if len(intervals) > max_intervals:
                        return []

        potential_presence = self.compute_valid_schedule_UB_minimal_split(
            plans, potential_presence, edge_weights, time_limit,
            max_intervals=max_intervals
        )
        if potential_presence is None or potential_presence == []:
            return potential_presence

        if self._has_any_conflict_in_pp(plans, potential_presence):
            return []

        return potential_presence




    @staticmethod
    def _is_edge_conflict_key(loc):
        """True iff a conflict-dict key denotes an edge conflict."""
        return isinstance(loc, tuple) and len(loc) > 0 and isinstance(loc[0], tuple)

    @staticmethod
    def _interval_lists_overlap(intervals_i, intervals_j):
        """True iff any subinterval in intervals_i overlaps one in intervals_j."""
        if not intervals_i or not intervals_j:
            return False
        for int_i in intervals_i:
            for int_j in intervals_j:
                if int_i and int_j and Cas.overlapping(int_i, int_j):
                    return True
        return False

    def _find_vertex_occurrence_index(self, plan, vertex, conflict_interval):
        """Find the path index corresponding to a CT vertex conflict.

        We first try exact interval equality with plan.path[k][0], then overlap,
        then any visit to the same vertex. This uses the CT node's original
        path, not the reduced policy, so it stays consistent with
        node.conflicts.
        """
        exact = []
        overlap = []
        same_vertex = []

        c_int = tuple(conflict_interval)

        for k, move in enumerate(plan.path):
            if move[1] != vertex:
                continue

            same_vertex.append(k)
            move_interval = tuple(move[0])

            if move_interval == c_int:
                exact.append(k)
            elif Cas.overlapping(move_interval, c_int):
                overlap.append(k)

        if exact:
            return exact[0]
        if overlap:
            return overlap[0]
        if same_vertex:
            return same_vertex[0]
        return None

    @staticmethod
    def _movement_tuple_for_plan_move(plan, move_index):
        """Return the same tuple representation used by create_movement_tuples."""
        start_vertex = min(plan.path[move_index][1], plan.path[move_index + 1][1])
        next_vertex = max(plan.path[move_index][1], plan.path[move_index + 1][1])
        interval = (plan.path[move_index][0][0], plan.path[move_index + 1][0][1])
        direction = 'f' if start_vertex == plan.path[move_index][1] else 'b'
        return interval, (start_vertex, next_vertex), direction

    def _find_edge_move_index(self, plan, conflict_edge, conflict_interval, direction):
        """Find the move index corresponding to a CT edge conflict."""
        exact = []
        overlap = []
        same_edge_dir = []

        c_int = tuple(conflict_interval)

        for k in range(len(plan.path) - 1):
            move_interval, move_edge, move_dir = self._movement_tuple_for_plan_move(plan, k)

            if move_edge != conflict_edge or move_dir != direction:
                continue

            same_edge_dir.append(k)

            if tuple(move_interval) == c_int:
                exact.append(k)
            elif Cas.overlapping(move_interval, c_int):
                overlap.append(k)

        if exact:
            return exact[0]
        if overlap:
            return overlap[0]
        if same_edge_dir:
            return same_edge_dir[0]
        return None

    def _current_edge_conflict_overlaps(self, potential_presence, agent_i, move_i, agent_j, move_j):
        """Check the current reduced policy for one CT edge conflict."""
        intervals_i = self._edge_presence_intervals(potential_presence[agent_i], move_i)
        intervals_j = self._edge_presence_intervals(potential_presence[agent_j], move_j)
        return self._interval_lists_overlap(intervals_i, intervals_j)

    def _reduce_one_vertex_conflict_from_ct(self, plans, potential_presence, edge_weights,
                                            vertex, conflict, time_limit,
                                            split=False, max_intervals=256):
        """Try to reduce one CT vertex conflict.

        Returns:
          dict  -> updated policy
          []    -> conflict cannot be reduced; CBS must branch on vertex
          None  -> timeout
        """
        agent_i, agent_j, interval_i, interval_j = conflict

        idx_i = self._find_vertex_occurrence_index(plans[agent_i], vertex, interval_i)
        idx_j = self._find_vertex_occurrence_index(plans[agent_j], vertex, interval_j)

        if idx_i is None or idx_j is None:
            return []

        intervals_i = potential_presence[agent_i][idx_i]
        intervals_j = potential_presence[agent_j][idx_j]

        if not self._interval_lists_overlap(intervals_i, intervals_j):
            return potential_presence

        # Wait-induced vertex conflict: not reducible by policy interval shrinking.
        if idx_i > 0 and plans[agent_i].path[idx_i - 1][1] == plans[agent_i].path[idx_i][1]:
            return []
        if idx_j > 0 and plans[agent_j].path[idx_j - 1][1] == plans[agent_j].path[idx_j][1]:
            return []

        if split:
            new_i, new_j = self.enforce_upper_bounds_with_minimal_split(
                intervals_i,
                intervals_j,
                time_limit,
                max_intervals=max_intervals
            )
            if new_i is None or new_j is None:
                return None
        else:
            new_i, new_j = self.enforce_upper_bounds_global(
                intervals_i,
                intervals_j,
                time_limit
            )

        if new_i is None or new_j is None:
            return None
        if not new_i or not new_j:
            return []

        pp = copy.deepcopy(potential_presence)
        pp[agent_i][idx_i] = self._merge_intervals(new_i)
        pp[agent_j][idx_j] = self._merge_intervals(new_j)

        if not pp[agent_i][idx_i] or not pp[agent_j][idx_j]:
            return []

        self.propagates_new_pp(plans[agent_i], pp[agent_i], idx_i + 1, edge_weights)
        self.propagates_new_pp(plans[agent_j], pp[agent_j], idx_j + 1, edge_weights)

        pp = self._canonicalize_policy_or_fail(pp, plans=plans)
        return pp


    @staticmethod
    def _is_opposite_move(plan_i, i, plan_j, j):
        """True iff move i of plan_i and move j of plan_j traverse the same
        graph edge in opposite directions.
        """
        u_i = plan_i.path[i][1]
        v_i = plan_i.path[i + 1][1]
        u_j = plan_j.path[j][1]
        v_j = plan_j.path[j + 1][1]
        return u_i == v_j and v_i == u_j and u_i != v_i


    @staticmethod
    def _belongs_to_reversed_triplet(plan_i, i, plan_j, j):
        """Detect whether an opposite-edge conflict belongs to a reversed
        length-2 chain.

        Example:
            agent 1: A -> B -> C
            agent 2: C -> B -> A

        If the conflict is on B->C / C->B, this returns True, so the edge
        conflict is not reduced and CBS branches instead.
        """
        path_i = [x[1] for x in plan_i.path]
        path_j = [x[1] for x in plan_j.path]

        # ... A, B, C for i and C, B, A ... for j
        if i > 0 and j + 2 < len(path_j):
            if path_i[i - 1] == path_j[j + 2]:
                return True

        # B, C, D ... for i and ... D, C, B for j
        if i + 2 < len(path_i) and j > 0:
            if path_i[i + 2] == path_j[j - 1]:
                return True

        return False


    @staticmethod
    def _edge_presence_intervals(pp_agent, move_index):
        """Return conservative edge-presence intervals for split policies."""
        result = []
        for src_interval in pp_agent[move_index]:
            for dst_interval in pp_agent[move_index + 1]:
                result.append([src_interval[0], dst_interval[1]])
        return result


    @staticmethod
    def _total_interval_measure(intervals):
        """Total length of a list of intervals."""
        if not intervals:
            return 0
        return sum(max(0, u - l) for l, u in intervals)

    def _reduce_one_edge_conflict_from_ct(self, plans, potential_presence, edge_weights,
                                          conflict_edge, conflict, time_limit,
                                          split=False, max_intervals=256):
        """Try to reduce one CT edge conflict.

        Edge reduction is attempted only for isolated opposite-edge conflicts.
        Structural reversed-triplet/corridor conflicts return [] so CBS branches
        on the edge conflict.
        """
        agent_i, agent_j, interval_i, interval_j, dir_i, dir_j = conflict

        move_i = self._find_edge_move_index(plans[agent_i], conflict_edge, interval_i, dir_i)
        move_j = self._find_edge_move_index(plans[agent_j], conflict_edge, interval_j, dir_j)

        if move_i is None or move_j is None:
            return []

        if not self._is_opposite_move(plans[agent_i], move_i, plans[agent_j], move_j):
            # Same-direction edge conflicts are not handled by this edge-reduction rule.
            return []

        if not self._current_edge_conflict_overlaps(potential_presence, agent_i, move_i, agent_j, move_j):
            return potential_presence

        if self._belongs_to_reversed_triplet(plans[agent_i], move_i, plans[agent_j], move_j):
            return []

        candidates = []

        if split:
            before_j = self._total_interval_measure(potential_presence[agent_j][move_j])
            pp_ij = self._apply_edge_order_candidate_minimal_split(
                plans, potential_presence, edge_weights,
                agent_i, move_i, agent_j, move_j
            )
            if pp_ij is not None:
                pp_ij = self._canonicalize_policy_or_fail(pp_ij, plans=plans)
                if pp_ij not in (None, []):
                    loss = before_j - self._total_interval_measure(pp_ij[agent_j][move_j])
                    candidates.append((loss, _stable_key((agent_i, agent_j)), pp_ij))

            before_i = self._total_interval_measure(potential_presence[agent_i][move_i])
            pp_ji = self._apply_edge_order_candidate_minimal_split(
                plans, potential_presence, edge_weights,
                agent_j, move_j, agent_i, move_i
            )
            if pp_ji is not None:
                pp_ji = self._canonicalize_policy_or_fail(pp_ji, plans=plans)
                if pp_ji not in (None, []):
                    loss = before_i - self._total_interval_measure(pp_ji[agent_i][move_i])
                    candidates.append((loss, _stable_key((agent_j, agent_i)), pp_ji))
        else:
            pp_ij = self._apply_edge_order_candidate(
                plans, potential_presence, edge_weights,
                agent_i, move_i, agent_j, move_j
            )
            if pp_ij is not None:
                pp_ij = self._canonicalize_policy_or_fail(pp_ij, plans=plans)
                if pp_ij not in (None, []):
                    delta = pp_ij[agent_j][move_j][0][0] - potential_presence[agent_j][move_j][0][0]
                    candidates.append((delta, _stable_key((agent_i, agent_j)), pp_ij))

            pp_ji = self._apply_edge_order_candidate(
                plans, potential_presence, edge_weights,
                agent_j, move_j, agent_i, move_i
            )
            if pp_ji is not None:
                pp_ji = self._canonicalize_policy_or_fail(pp_ji, plans=plans)
                if pp_ji not in (None, []):
                    delta = pp_ji[agent_i][move_i][0][0] - potential_presence[agent_i][move_i][0][0]
                    candidates.append((delta, _stable_key((agent_j, agent_i)), pp_ji))

        if not candidates:
            return []

        candidates.sort(key=lambda x: (x[0], x[1]))
        pp = candidates[0][2]

        if split:
            for pp_agent in pp.values():
                for intervals in pp_agent:
                    if len(intervals) > max_intervals:
                        return []

        return pp

    @staticmethod
    def _interval_intersection_measure(intervals_a, intervals_b):
        """Lebesgue measure of the overlap of two canonical interval sets."""
        total = 0.0
        for a_l, a_u in intervals_a:
            for b_l, b_u in intervals_b:
                total += max(0.0, min(a_u, b_u) - max(a_l, b_l))
        return total

    def _k_safe_score(self, intervals_a, intervals_b, p=0.001):
        """Local k-safety for an unreducible equal-upper-bound conflict.

        The uniform model used in the paper estimates the conflict probability
        as overlap_measure / max(measure(A), measure(B)).  The larger support is
        used as the reference potential presence, matching examples such as
        [5,10] vs [9,10] -> k=0.8 and [[2,5],[7,10]] vs [9,10]
        -> k=5/6.  Returns None when the conflict is not an equal-upper-bound
        case and therefore must still be handled by ordinary CBS branching.
        """
        A = self._canonical_intervals(intervals_a, p=p)
        B = self._canonical_intervals(intervals_b, p=p)
        if not A or not B or not self._interval_lists_overlap(A, B):
            return 1.0
        upper_a = max(x[1] for x in A)
        upper_b = max(x[1] for x in B)
        if abs(upper_a - upper_b) > p / 10:
            return None
        denom = max(self._total_interval_measure(A), self._total_interval_measure(B))
        if denom <= 0:
            return 0.0
        overlap = self._interval_intersection_measure(A, B)
        return max(0.0, min(1.0, 1.0 - overlap / denom))

    def _k_safe_vertex_score_from_ct(self, plans, potential_presence, vertex, conflict):
        agent_i, agent_j, interval_i, interval_j = conflict
        idx_i = self._find_vertex_occurrence_index(plans[agent_i], vertex, interval_i)
        idx_j = self._find_vertex_occurrence_index(plans[agent_j], vertex, interval_j)
        if idx_i is None or idx_j is None:
            return None
        # Wait conflicts remain hard conflicts: k-safety only relaxes failures
        # caused by equal upper bounds in the interval-reduction operator.
        if idx_i > 0 and plans[agent_i].path[idx_i - 1][1] == plans[agent_i].path[idx_i][1]:
            return None
        if idx_j > 0 and plans[agent_j].path[idx_j - 1][1] == plans[agent_j].path[idx_j][1]:
            return None
        return self._k_safe_score(potential_presence[agent_i][idx_i],
                                  potential_presence[agent_j][idx_j])

    def _k_safe_edge_score_from_ct(self, plans, potential_presence, conflict_edge, conflict):
        agent_i, agent_j, interval_i, interval_j, dir_i, dir_j = conflict
        move_i = self._find_edge_move_index(plans[agent_i], conflict_edge, interval_i, dir_i)
        move_j = self._find_edge_move_index(plans[agent_j], conflict_edge, interval_j, dir_j)
        if move_i is None or move_j is None:
            return None
        if not self._is_opposite_move(plans[agent_i], move_i, plans[agent_j], move_j):
            return None
        if self._belongs_to_reversed_triplet(plans[agent_i], move_i, plans[agent_j], move_j):
            return None
        A = self._edge_presence_intervals(potential_presence[agent_i], move_i)
        B = self._edge_presence_intervals(potential_presence[agent_j], move_j)
        return self._k_safe_score(A, B)

    def _reduce_ct_conflicts_k_safe_status(self, plans, potential_presence, edge_weights,
                                           time_limit, node_conflicts, k_safe=1.0,
                                           split=False, max_intervals=256):
        """CBS_P reduction with optional k-safe acceptance.

        Normal reductions are always attempted first.  Only a reduction failure
        caused by an equal upper bound may be tolerated.  Wait conflicts and
        reversed edge chains remain hard conflicts.  Returns
        (policy, is_node_conflict, achieved_k).
        """
        if not 0.0 <= k_safe <= 1.0:
            raise ValueError('k_safe must be in [0, 1]')
        if time.time() - self.start_time > time_limit:
            return None, False, 0.0
        potential_presence = self._canonicalize_policy_or_fail(potential_presence, plans=plans)
        if potential_presence is None:
            return None, False, 0.0
        if potential_presence == []:
            return [], True, 0.0

        achieved_k = 1.0
        edge_items = [(loc, cs) for loc, cs in node_conflicts.items() if self._is_edge_conflict_key(loc)]
        vertex_items = [(loc, cs) for loc, cs in node_conflicts.items() if not self._is_edge_conflict_key(loc)]

        for loc, conflicts in sorted(edge_items, key=lambda item: _stable_key(item[0])):
            for conflict in sorted(conflicts, key=_stable_key):
                pp_new = self._reduce_one_edge_conflict_from_ct(
                    plans, potential_presence, edge_weights, loc, conflict, time_limit,
                    split=split, max_intervals=max_intervals)
                if pp_new is None:
                    return None, False, achieved_k
                if pp_new == []:
                    score = self._k_safe_edge_score_from_ct(plans, potential_presence, loc, conflict)
                    if score is None or score + 1e-12 < k_safe:
                        return [], False, min(achieved_k, score if score is not None else 0.0)
                    achieved_k = min(achieved_k, score)
                    continue
                potential_presence = pp_new

        for loc, conflicts in sorted(vertex_items, key=lambda item: _stable_key(item[0])):
            for conflict in sorted(conflicts, key=_stable_key):
                pp_new = self._reduce_one_vertex_conflict_from_ct(
                    plans, potential_presence, edge_weights, loc, conflict, time_limit,
                    split=split, max_intervals=max_intervals)
                if pp_new is None:
                    return None, True, achieved_k
                if pp_new == []:
                    score = self._k_safe_vertex_score_from_ct(plans, potential_presence, loc, conflict)
                    if score is None or score + 1e-12 < k_safe:
                        return [], True, min(achieved_k, score if score is not None else 0.0)
                    achieved_k = min(achieved_k, score)
                    continue
                potential_presence = pp_new

        potential_presence = self._canonicalize_policy_or_fail(potential_presence, plans=plans)
        if potential_presence is None:
            return None, True, achieved_k
        if potential_presence == []:
            return [], True, achieved_k
        return potential_presence, True, achieved_k

    def _reduce_ct_conflicts_status(self, plans, potential_presence, edge_weights,
                                    time_limit, node_conflicts,
                                    split=False, max_intervals=256):
        """Reduce only conflicts stored in the CT node.

        This keeps the reduction pipeline consistent with the CBS conflict
        detector. It does not rescan the reduced policy looking for new conflicts
        with a different detector.

        Returns (policy, is_node_conflict).
        """
        if time.time() - self.start_time > time_limit:
            return None, False

        potential_presence = self._canonicalize_policy_or_fail(potential_presence, plans=plans)
        if potential_presence is None:
            return None, False
        if potential_presence == []:
            return [], True

        edge_items = [
            (loc, conflicts) for loc, conflicts in node_conflicts.items()
            if self._is_edge_conflict_key(loc)
        ]
        vertex_items = [
            (loc, conflicts) for loc, conflicts in node_conflicts.items()
            if not self._is_edge_conflict_key(loc)
        ]

        # Edge reductions first. If one fails, branch on edge.
        for loc, conflicts in sorted(edge_items, key=lambda item: _stable_key(item[0])):
            conflict_edge = loc
            for conflict in sorted(conflicts, key=_stable_key):
                if time.time() - self.start_time > time_limit:
                    return None, False

                pp_new = self._reduce_one_edge_conflict_from_ct(
                    plans,
                    potential_presence,
                    edge_weights,
                    conflict_edge,
                    conflict,
                    time_limit,
                    split=split,
                    max_intervals=max_intervals
                )

                if pp_new is None:
                    return None, False
                if pp_new == []:
                    return [], False

                potential_presence = pp_new

        # Vertex reductions second. If one fails, branch on vertex.
        for loc, conflicts in sorted(vertex_items, key=lambda item: _stable_key(item[0])):
            vertex = loc
            for conflict in sorted(conflicts, key=_stable_key):
                if time.time() - self.start_time > time_limit:
                    return None, True

                pp_new = self._reduce_one_vertex_conflict_from_ct(
                    plans,
                    potential_presence,
                    edge_weights,
                    vertex,
                    conflict,
                    time_limit,
                    split=split,
                    max_intervals=max_intervals
                )

                if pp_new is None:
                    return None, True
                if pp_new == []:
                    return [], True

                potential_presence = pp_new

        potential_presence = self._canonicalize_policy_or_fail(potential_presence, plans=plans)
        if potential_presence is None:
            return None, True
        if potential_presence == []:
            return [], True

        return potential_presence, True

    def _reduce_edges_then_vertices_status(self, plans, potential_presence, edge_weights,
                                           time_limit, vertex_reducer):
        """Try to build an execution policy and return the correct CBS branch type.

        The priority is:

        1. Try to reduce edge conflicts first.
           - If an edge conflict forms a reversed chain/corridor, do not reduce it.
           - If an isolated edge conflict cannot be reduced due to intervals, do not
             reduce it.
           - In both cases, return ([], False), meaning CBS must branch on an edge
             conflict.

        2. If all edge conflicts were reduced, reduce vertex conflicts using
           vertex_reducer.
           - If a vertex conflict cannot be reduced, return ([], True), meaning CBS
             must branch on a vertex conflict.

        3. Validate the resulting policy.
           - Remaining edge conflict -> branch on edge.
           - Remaining vertex conflict -> branch on vertex.
           - No remaining conflicts -> return (policy, True).

        Returns
        -------
        (policy, is_node_conflict)

        policy:
            dict  -> success, execution policy intervals
            []    -> reduction failed, CBS should branch
            None  -> timeout

        is_node_conflict:
            False -> branch on an edge conflict
            True  -> branch on a vertex conflict
        """
        # Phase 1: edge reductions first.
        while True:
            if time.time() - self.start_time > time_limit:
                return None, False

            edge_conflict = self._first_edge_conflict_in_pp(plans, potential_presence)
            if edge_conflict is None:
                break

            agent_i, move_i, agent_j, move_j = edge_conflict
            plan_i = plans[agent_i]
            plan_j = plans[agent_j]

            # Reversed triplet/corridor pattern:
            # e.g. a_i: A -> B -> C and a_j: C -> B -> A.
            # This is structural, so it must be resolved by CBS edge branching.
            if self._belongs_to_reversed_triplet(plan_i, move_i, plan_j, move_j):
                return [], False

            candidates = []

            # Candidate 1: i crosses first, j is delayed before reaching the
            # source vertex of its opposite edge traversal.
            pp_ij = self._apply_edge_order_candidate(
                plans, potential_presence, edge_weights,
                agent_i, move_i, agent_j, move_j
            )
            if pp_ij is not None:
                delta = pp_ij[agent_j][move_j][0][0] - potential_presence[agent_j][move_j][0][0]
                candidates.append((delta, _stable_key((agent_i, agent_j)), pp_ij))

            # Candidate 2: j crosses first, i is delayed.
            pp_ji = self._apply_edge_order_candidate(
                plans, potential_presence, edge_weights,
                agent_j, move_j, agent_i, move_i
            )
            if pp_ji is not None:
                delta = pp_ji[agent_i][move_i][0][0] - potential_presence[agent_i][move_i][0][0]
                candidates.append((delta, _stable_key((agent_j, agent_i)), pp_ji))

            # Isolated edge conflict, but neither ordering is feasible by
            # interval reduction. CBS must branch on this edge conflict.
            if not candidates:
                return [], False

            # Greedy deterministic choice: smallest immediate reduction.
            candidates.sort(key=lambda x: (x[0], x[1]))
            potential_presence = candidates[0][2]

        # Phase 2: vertex reductions.
        potential_presence = vertex_reducer(
            plans,
            potential_presence,
            edge_weights,
            time_limit
        )

        if potential_presence is None:
            return None, True

        # A vertex conflict could not be reduced, e.g., because it is wait-induced
        # or because the admissible interval becomes empty.
        if potential_presence == []:
            return [], True

        # Phase 3: final validation with priority-consistent failure reason.
        if self._first_edge_conflict_in_pp(plans, potential_presence) is not None:
            return [], False

        if self._first_vertex_conflict_in_pp(plans, potential_presence) is not None:
            return [], True

        return potential_presence, True

    def compute_valid_schedule_UB_with_isolated_edges_status(self, plans, potential_presence,
                                                             edge_weights, time_limit,
                                                             node_conflicts=None):
        """CT-consistent single-interval edge-then-vertex reduction.

        Only conflicts already stored in the CT node are considered. This avoids
        false failures caused by rescanning reduced interval sets with a
        different conflict detector.
        """
        if node_conflicts is None:
            node_conflicts = {}
        if not node_conflicts:
            potential_presence = self._canonicalize_policy_or_fail(potential_presence, plans=plans)
            if potential_presence is None:
                return None, True
            if potential_presence == []:
                return [], True
            return potential_presence, True
        return self._reduce_ct_conflicts_status(
            plans,
            potential_presence,
            edge_weights,
            time_limit,
            node_conflicts,
            split=False
        )



    def compute_valid_schedule_UB_with_isolated_edges_minimal_split_status(self, plans, potential_presence,
                                                                           edge_weights, time_limit,
                                                                           node_conflicts=None,
                                                                           max_intervals=256):
        """CT-consistent minimal-split edge-then-vertex reduction.

        Edge conflicts are attempted first. If an edge conflict cannot be
        reduced, CBS branches on that edge conflict. If all edge conflicts are
        reduced, vertex conflicts are reduced using upper-bound-preserving
        splitting.
        """
        if node_conflicts is None:
            node_conflicts = {}
        if not node_conflicts:
            potential_presence = self._canonicalize_policy_or_fail(potential_presence, plans=plans)
            if potential_presence is None:
                return None, True
            if potential_presence == []:
                return [], True
            return potential_presence, True
        return self._reduce_ct_conflicts_status(
            plans,
            potential_presence,
            edge_weights,
            time_limit,
            node_conflicts,
            split=True,
            max_intervals=max_intervals
        )



    def compute_plan_success_rate_UB(self, pp, propagation_pp):

        is_success_guarentee = True

        for agent, pp_agent in propagation_pp.items():

            for t, intervals in enumerate(pp_agent):

                original_size = pp[agent][t][0][1] - pp[agent][t][0][0]
                new_size = 0
                is_upper_bound = False
                for interval in intervals:
                    new_size += interval[1] - interval[0]
                    is_upper_bound = is_upper_bound or interval[1] == pp[agent][t][0][1]

                if not is_upper_bound:
                    is_success_guarentee = False

        return 1 if is_success_guarentee else 0



    def is_edge_conflict(self, node):
        for key, value in node.conflicts.items():
            if isinstance(key[0], tuple):
                return True
        return False

    def find_solution_propagation_upper_bound(self, min_best_case=False, time_lim=60, soc=True, use_cat=True, existing_cons=None,
                      curr_time=(0, 0), use_pc=False, use_bp=False, k_safe=1.0):


        self.__initialize_class_variables(curr_time, min_best_case, soc, use_cat)
        if not self.create_root(existing_cons, time_limit=time_lim):
            #raise OutOfTimeError("Couldn't find initial solution")
            return self.create_solution(ConstraintNode())

        try:
            while self.open_nodes:
                if time.time() - self.start_time > time_lim:
                    #raise OutOfTimeError('Ran out of time :-(')
                    return self.create_solution(best_node)

                best_node = self.open_nodes.pop()
                self.__insert_into_closed_list(best_node)
                pp = self.potential_presence(best_node.sol.paths, self.tu_problem.weights)

                # Invariant: if the CT node has no CBS conflicts, no interval
                # reduction is needed and CBS must not branch.
                if best_node.conf_num == 0 or not best_node.conflicts:
                    return self.create_solution(
                        best_node,
                        success_rate=1,
                        is_solved=True,
                        execution_policy=pp
                    )

                # Run a single reduction/validation pipeline.
                #
                # Important: even if edge conflicts are present, resolving them
                # is not sufficient. The candidate execution policy must also
                # satisfy all vertex constraints after propagation.
                #
                # compute_valid_schedule_UB_with_isolated_edges() performs:
                #   1) isolated opposite-edge repairs when possible;
                #   2) vertex interval reduction;
                #   3) final validation of both vertex and edge conflicts.
                #
                # If an edge conflict is structural or cannot be repaired, the
                # function returns [] and CBS branches on an edge conflict.
                has_edge_conflict = self.is_edge_conflict(best_node)
                if has_edge_conflict:
                    best_node.iteration_with_edge_conflicts += 1

                new_pp, is_node_conflict, achieved_k = self._reduce_ct_conflicts_k_safe_status(
                    best_node.sol.paths, copy.deepcopy(pp), self.tu_problem.weights,
                    time_lim, best_node.conflicts, k_safe=k_safe, split=False
                )

                success_rate = achieved_k if (new_pp is not None and new_pp != [] and not self._policy_has_empty_interval_set(new_pp)) else 0

                if new_pp is not None and new_pp != [] and not self._policy_has_empty_interval_set(new_pp):
                    return self.create_solution(best_node, success_rate, is_solved=True, execution_policy=new_pp)

                best_node.iteration+= 1
                new_constraints, c1, c2, is_cardinal = self.find_best_conflict(best_node, time_lim, use_pc, is_node_conflict)
                if use_bp and not is_cardinal and self.can_bypass(best_node, new_constraints, c1, c2):
                    continue
                self.__insert_open_node(c1)
                self.__insert_open_node(c2)
            print("Empty open list - No Solution")
            return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)

        except OutOfTimeError:  # Ran out of time. Return an empty solution.
            return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)



    def find_solution_propagation_upper_bound_minimal_split(
            self, min_best_case=False, time_lim=60, soc=True, use_cat=True,
            existing_cons=None, curr_time=(0, 0), use_pc=False, use_bp=False,
            max_intervals=256, k_safe=1.0):
        """CBS_P variant using minimal split-based interval reduction.

        It has the same high-level behavior as find_solution_propagation_upper_bound,
        but calls the split-based schedule checker. The returned solution remains
        a normal TimeUncertaintySolution; the reduced execution envelope is used
        only to decide whether the current CT node is acceptable.
        """
        self.__initialize_class_variables(curr_time, min_best_case, soc, use_cat)
        if not self.create_root(existing_cons, time_limit=time_lim):
            return self._empty_solution_now(iteration=0)

        try:
            while self.open_nodes:
                if time.time() - self.start_time > time_lim:
                    return self.create_solution(best_node)

                best_node = self.open_nodes.pop()
                self.__insert_into_closed_list(best_node)

                pp = self.potential_presence(best_node.sol.paths, self.tu_problem.weights)

                # Invariant: if the CT node has no CBS conflicts, no interval
                # reduction is needed and CBS must not branch.
                if best_node.conf_num == 0 or not best_node.conflicts:
                    return self.create_solution(
                        best_node,
                        success_rate=1,
                        is_solved=True,
                        execution_policy=pp
                    )

                # Unified minimal-split reduction pipeline:
                #   1) reduce eligible isolated edge conflicts first;
                #   2) then reduce vertex conflicts using minimal splitting;
                #   3) if edge reduction fails, branch on edge;
                #   4) if vertex reduction fails, branch on vertex.
                if self.is_edge_conflict(best_node):
                    best_node.iteration_with_edge_conflicts += 1

                new_pp, is_node_conflict, achieved_k = self._reduce_ct_conflicts_k_safe_status(
                    best_node.sol.paths, copy.deepcopy(pp), self.tu_problem.weights,
                    time_lim, best_node.conflicts, k_safe=k_safe, split=True,
                    max_intervals=max_intervals
                )

                success_rate = achieved_k if (new_pp is not None and new_pp != [] and not self._policy_has_empty_interval_set(new_pp)) else 0

                if new_pp is not None and new_pp != [] and not self._policy_has_empty_interval_set(new_pp):
                    return self.create_solution(best_node, success_rate, is_solved=True, execution_policy=new_pp)

                best_node.iteration += 1
                remaining_time = time_lim - (time.time() - self.start_time)
                if remaining_time <= 0:
                    return self._empty_solution_now(iteration=best_node.iteration)

                new_constraints, c1, c2, is_cardinal = self.find_best_conflict(
                    best_node, time_lim, use_pc, is_node_conflict
                )
                if use_bp and not is_cardinal and self.can_bypass(
                        best_node, new_constraints, c1, c2):
                    continue

                self.__insert_open_node(c1)
                self.__insert_open_node(c2)

            print("Empty open list - No Solution")
            return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)

        except OutOfTimeError:
            return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)



    # ------------------------------------------------------------------
    # Strong EECBS-TU high-level search
    # ------------------------------------------------------------------
    def _ct_signature(self, node):
        """Constraint-only signature used by the classical CBS implementation."""
        return frozenset(
            (k, tuple(sorted(val, key=_stable_key)))
            for k, val in node.constraints.items()
        )

    def _path_signature(self, node):
        """Deterministic signature of the current path combination.

        With CAT and focal low-level search, identical constraints may yield
        different paths. EECBS must therefore not collapse all such nodes into
        one constraint-only duplicate.
        """
        result = []
        for agent, plan in sorted(node.sol.paths.items(), key=lambda x: _stable_key(x[0])):
            moves = tuple(
                (tuple(move[0]), _stable_key(move[1]))
                for move in plan.path
            )
            result.append((_stable_key(agent), moves))
        return tuple(result)

    def _eecbs_signature(self, node):
        return self._ct_signature(node), self._path_signature(node)

    def _plan_primary_cost(self, plan):
        if not getattr(plan, 'path', None):
            return math.inf
        interval = plan.path[-1][0]
        return interval[0] if self.min_best_case else interval[1]

    def _node_lower_bound(self, node):
        """Aggregate low-level lower bounds for the high-level EECBS bound.

        Focal low-level search stores ``low_level_f_min`` on each returned plan.
        For classical A*, the returned path cost itself is exact and is used as
        the fallback.
        """
        values = []
        for plan in node.sol.paths.values():
            path_cost = self._plan_primary_cost(plan)
            value = getattr(plan, 'low_level_f_min', None)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                value = path_cost
            else:
                # A lower bound can never exceed the returned path cost. Some
                # low-level implementations record the next OPEN minimum after
                # removing the goal; cap it to preserve admissibility.
                value = min(float(value), float(path_cost))
            values.append(float(value))
        if not values:
            return math.inf
        return sum(values) if self.soc else max(values)

    def _node_primary_cost(self, node):
        """Actual current CT-node solution cost (not a lower bound)."""
        if self.min_best_case:
            return node.sol.cost[0]
        return node.sol.cost[1]

    def _node_secondary_cost(self, node):
        """Secondary true cost used only for deterministic tie-breaking."""
        if self.min_best_case:
            return node.sol.cost[1]
        return node.sol.cost[0]

    @staticmethod
    def _is_valid_ct_node(node):
        return (
            node is not None
            and getattr(node, 'sol', None) is not None
            and node.sol.cost[0] != math.inf
            and node.sol.cost[1] != math.inf
            and len(getattr(node.sol, 'paths', {})) > 0
        )

    def _conflict_agent_pairs(self, node):
        """Return deterministic set of agent pairs involved in conflicts."""
        pairs = set()
        for loc, conflicts in sorted(node.conflicts.items(), key=lambda item: _stable_key(item[0])):
            for conflict in sorted(conflicts, key=_stable_key):
                if not conflict or len(conflict) < 2:
                    continue
                a1, a2 = conflict[0], conflict[1]
                if _stable_key(a2) < _stable_key(a1):
                    a1, a2 = a2, a1
                pairs.add((a1, a2))
        return pairs

    def _greedy_matching_size(self, pairs):
        """Cheap admissible-ish lower bound proxy from disjoint conflicting pairs.

        True EECBS often uses stronger CG/DG/WDG heuristics. This code avoids
        expensive MDD construction and gives a stable estimate that is useful
        for MAPF-TU experiments. It is only used for node ordering, not for
        pruning or correctness.
        """
        used = set()
        size = 0
        for a1, a2 in sorted(pairs, key=_stable_key):
            if a1 in used or a2 in used:
                continue
            used.add(a1)
            used.add(a2)
            size += 1
        return size


    @staticmethod
    def _interval_overlap_measure(interval_a, interval_b):
        """Continuous overlap length between two time intervals.

        This is used only for a TU-WDG-style ordering heuristic. It is not a
        correctness/pruning condition.
        """
        try:
            lo = max(interval_a[0], interval_b[0])
            hi = min(interval_a[1], interval_b[1])
            return max(0.0, float(hi - lo))
        except Exception:
            return 0.0

    def _tu_wdg_pair_features(self, node):
        """Build pair-level conflict features for a cheap TU-WDG proxy.

        Classical WDG computes, for each pair of agents, the minimum unavoidable
        extra cost needed to resolve their dependency. Computing that exactly in
        MAPF-TU would require solving many two-agent CBS-TU subproblems per CT
        node, which is often too expensive in Python.

        This function instead extracts a lightweight dependency graph from the
        current CT conflicts. Each edge (i,j) stores:
          - number of vertex conflicts,
          - number of edge conflicts,
          - total temporal overlap measure.

        The resulting heuristic is used only for EECBS node ordering, never for
        pruning. Therefore it does not affect correctness; it only changes which
        CT node is expanded first.
        """
        features = {}

        def norm_pair(a1, a2):
            if _stable_key(a2) < _stable_key(a1):
                return a2, a1
            return a1, a2

        if getattr(node, 'conf_num', math.inf) == math.inf:
            node.conflicts, node.edge_conflicts = node.find_all_conflicts()

        edge_conflict_locs = set()
        try:
            edge_conflict_locs = set(getattr(node, 'edge_conflicts', {}).keys())
        except Exception:
            edge_conflict_locs = set()

        for loc, conflicts in sorted(node.conflicts.items(), key=lambda item: _stable_key(item[0])):
            is_edge = loc in edge_conflict_locs or self._is_edge_conflict_key(loc)
            for conflict in sorted(conflicts, key=_stable_key):
                if not conflict or len(conflict) < 4:
                    continue
                a1, a2 = norm_pair(conflict[0], conflict[1])
                data = features.setdefault(
                    (a1, a2),
                    {'vertex': 0, 'edge': 0, 'overlap': 0.0, 'count': 0}
                )
                data['count'] += 1
                if is_edge:
                    data['edge'] += 1
                else:
                    data['vertex'] += 1
                data['overlap'] += self._interval_overlap_measure(conflict[2], conflict[3])

        return features

    @staticmethod
    def _greedy_weighted_matching_value(edge_weights):
        """Greedy disjoint-edge lower-bound proxy for a weighted dependency graph.

        For true WDG, the graph heuristic is derived from pairwise unavoidable
        cost increases and a vertex-cover computation. Here we use a robust and
        cheap proxy: sort dependency edges by decreasing weight and sum weights
        of disjoint edges. This captures independent groups of coupled agents
        and is far more informative than a raw conflict count, while remaining
        cheap enough to compute at every EECBS node.
        """
        used = set()
        value = 0.0
        for (a1, a2), weight in sorted(
                edge_weights.items(),
                key=lambda item: (-item[1], _stable_key(item[0]))):
            if weight <= 0:
                continue
            if a1 in used or a2 in used:
                continue
            used.add(a1)
            used.add(a2)
            value += weight
        return value

    def _tu_wdg_proxy(self, node, edge_conflict_weight=2.0, overlap_weight=0.05,
                      cap_pair_weight=10.0):
        """Return a lightweight TU-WDG-style dependency estimate.

        The edge weight for a pair is:
            vertex_conflicts
          + edge_conflict_weight * edge_conflicts
          + overlap_weight * total_overlap_measure

        The final graph value is a greedy weighted matching over these pair
        weights. This is intentionally a proxy, not a formal admissible WDG
        heuristic. Use it inside EECBS/EES as an inadmissible estimate/ranking
        term, not as an optimal CBS pruning heuristic.
        """
        pair_features = self._tu_wdg_pair_features(node)
        edge_weights = {}
        for pair, data in pair_features.items():
            weight = (
                data['vertex']
                + edge_conflict_weight * data['edge']
                + overlap_weight * data['overlap']
            )
            if cap_pair_weight is not None:
                weight = min(float(cap_pair_weight), float(weight))
            edge_weights[pair] = float(weight)
        return self._greedy_weighted_matching_value(edge_weights)

    def _eecbs_estimates(self, node, conflict_weight=1.0, pair_weight=1.0, edge_weight=2.0,
                         use_wdg=False, wdg_weight=1.0,
                         wdg_overlap_weight=0.05, wdg_cap_pair_weight=10.0):
        """Compute EECBS-style estimates for a CT node.

        Returns:
            f      : cleanup/admissible node cost
            h_hat  : inadmissible estimate of remaining cost increase
            d_hat  : estimated distance-to-go, used inside FOCAL
            f_hat  : explicit estimated solution cost f + h_hat

        For MAPF-TU we bias edge conflicts more strongly because they are often
        harder to repair than vertex conflicts and cannot be fixed by simple
        safe-interval shrinking in CBS_P.
        """
        f = self._node_lower_bound(node)
        if f == math.inf:
            return math.inf, math.inf, math.inf, math.inf

        # Make sure conflict counts exist. Most generated nodes already have
        # them, but root/children from cached paths may occasionally need this.
        if getattr(node, 'conf_num', math.inf) == math.inf:
            node.conflicts, node.edge_conflicts = node.find_all_conflicts()

        edge_conflicts = 0
        try:
            edge_conflicts = sum(len(v) for v in getattr(node, 'edge_conflicts', {}).values())
        except Exception:
            edge_conflicts = 0

        total_conflicts = node.conf_num if node.conf_num != math.inf else 0
        vertex_conflicts = max(0, total_conflicts - edge_conflicts)
        pairs = self._conflict_agent_pairs(node)
        pair_lb = self._greedy_matching_size(pairs)

        d_hat = edge_weight * edge_conflicts + vertex_conflicts

        # Optional TU-WDG-style dependency estimate. This is a stronger
        # pair-level signal than raw conflict count: it groups conflicts by
        # agent pair, gives edge conflicts extra weight, and accounts for the
        # amount of temporal overlap. It is used only for EECBS ordering.
        wdg_proxy = 0.0
        # Important: when wdg_weight == 0, the WDG proxy has no effect on the
        # EECBS keys. Do not compute it anyway: it can be expensive enough to
        # change timeout-limited experiments even though the ordering is
        # mathematically unchanged.
        if use_wdg and abs(float(wdg_weight)) > 1e-12:
            wdg_proxy = self._tu_wdg_proxy(
                node,
                edge_conflict_weight=edge_weight,
                overlap_weight=wdg_overlap_weight,
                cap_pair_weight=wdg_cap_pair_weight
            )

        h_hat = pair_weight * pair_lb + conflict_weight * d_hat + wdg_weight * wdg_proxy
        f_hat = f + h_hat
        return f, h_hat, d_hat, f_hat

    def _eecbs_cleanup_key(self, node):
        """CLEANUP ordering: classical CBS-TU ordering, plus deterministic tie-breakers."""
        return (
            self._node_lower_bound(node),
            node.conf_num,
            self._node_secondary_cost(node),
            _stable_key(self._ct_signature(node)),
        )

    def _eecbs_open_key(self, node, conflict_weight=1.0, pair_weight=1.0, edge_weight=2.0,
                         use_wdg=False, wdg_weight=1.0,
                         wdg_overlap_weight=0.05, wdg_cap_pair_weight=10.0):
        """OPEN ordering by explicit estimated solution cost f_hat."""
        f, h_hat, d_hat, f_hat = self._eecbs_estimates(
            node, conflict_weight=conflict_weight,
            pair_weight=pair_weight, edge_weight=edge_weight,
            use_wdg=use_wdg, wdg_weight=wdg_weight,
            wdg_overlap_weight=wdg_overlap_weight,
            wdg_cap_pair_weight=wdg_cap_pair_weight
        )
        return (
            f_hat,
            d_hat,
            f,
            node.conf_num,
            self._node_secondary_cost(node),
            _stable_key(self._ct_signature(node)),
        )

    def _eecbs_focal_key(self, node, conflict_weight=1.0, pair_weight=1.0, edge_weight=2.0,
                          use_wdg=False, wdg_weight=1.0,
                          wdg_overlap_weight=0.05, wdg_cap_pair_weight=10.0):
        """FOCAL ordering by distance-to-go d_hat, then f_hat and cost."""
        f, h_hat, d_hat, f_hat = self._eecbs_estimates(
            node, conflict_weight=conflict_weight,
            pair_weight=pair_weight, edge_weight=edge_weight,
            use_wdg=use_wdg, wdg_weight=wdg_weight,
            wdg_overlap_weight=wdg_overlap_weight,
            wdg_cap_pair_weight=wdg_cap_pair_weight
        )
        edge_conflicts = 0
        try:
            edge_conflicts = sum(len(v) for v in getattr(node, 'edge_conflicts', {}).values())
        except Exception:
            edge_conflicts = 0
        return (
            d_hat,
            edge_conflicts,
            node.conf_num,
            f_hat,
            f,
            self._node_secondary_cost(node),
            _stable_key(self._ct_signature(node)),
        )

    def _eecbs_select_node(self, open_nodes, focal_w=1.1,
                           conflict_weight=1.0, pair_weight=1.0, edge_weight=2.0,
                           use_wdg=False, wdg_weight=1.0,
                           wdg_overlap_weight=0.05, wdg_cap_pair_weight=10.0):
        """Select a CT node using Explicit Estimation Search.

        The decision rule mirrors the EES/EECBS idea:
          1. Let f_min be the best cleanup cost in OPEN.
          2. Let best_focal be the node with smallest d_hat among nodes whose
             f <= w * f_min.
          3. Let best_open be the node with smallest f_hat in OPEN.
          4. If best_focal.f_hat <= w * f_min, expand best_focal.
             Else if best_open.f_hat <= w * f_min, expand best_open.
             Else expand best_cleanup.

        This keeps the ECBS-style bound through CLEANUP while using explicit
        estimates whenever they look safe with respect to the current bound.
        """
        if not open_nodes:
            return None

        # If the WDG coefficient is zero, WDG must be a true no-op.  This avoids
        # paying the proxy-computation overhead and guarantees that
        # use_wdg=True, wdg_weight=0 behaves like use_wdg=False.
        if abs(float(wdg_weight)) <= 1e-12:
            use_wdg = False

        best_cleanup = min(open_nodes, key=self._eecbs_cleanup_key)
        f_min = self._node_primary_cost(best_cleanup)
        bound = focal_w * f_min

        focal = [node for node in open_nodes if self._node_primary_cost(node) <= bound]
        best_focal = min(
            focal,
            key=lambda n: self._eecbs_focal_key(
                n, conflict_weight=conflict_weight,
                pair_weight=pair_weight, edge_weight=edge_weight,
                use_wdg=use_wdg, wdg_weight=wdg_weight,
                wdg_overlap_weight=wdg_overlap_weight,
                wdg_cap_pair_weight=wdg_cap_pair_weight
            )
        ) if focal else None

        best_open = min(
            open_nodes,
            key=lambda n: self._eecbs_open_key(
                n, conflict_weight=conflict_weight,
                pair_weight=pair_weight, edge_weight=edge_weight,
                use_wdg=use_wdg, wdg_weight=wdg_weight,
                wdg_overlap_weight=wdg_overlap_weight,
                wdg_cap_pair_weight=wdg_cap_pair_weight
            )
        )

        if best_focal is not None:
            _, _, _, focal_f_hat = self._eecbs_estimates(
                best_focal, conflict_weight=conflict_weight,
                pair_weight=pair_weight, edge_weight=edge_weight,
                use_wdg=use_wdg, wdg_weight=wdg_weight,
                wdg_overlap_weight=wdg_overlap_weight,
                wdg_cap_pair_weight=wdg_cap_pair_weight
            )
            if focal_f_hat <= bound:
                return best_focal

        _, _, _, open_f_hat = self._eecbs_estimates(
            best_open, conflict_weight=conflict_weight,
            pair_weight=pair_weight, edge_weight=edge_weight,
            use_wdg=use_wdg, wdg_weight=wdg_weight,
            wdg_overlap_weight=wdg_overlap_weight,
            wdg_cap_pair_weight=wdg_cap_pair_weight
        )
        if open_f_hat <= bound:
            return best_open

        return best_cleanup

    def _eecbs_insert_node(self, open_nodes, open_signatures, node, count_generated=True):
        """Insert a valid CT node into EECBS OPEN if not duplicate/closed."""
        if not self._is_valid_ct_node(node):
            return False

        sig = self._eecbs_signature(node)
        if sig in self.closed_nodes or sig in open_signatures:
            return False

        open_nodes.append(node)
        open_signatures.add(sig)

        # Original OpenListHeap.push() increments this counter. Because EECBS
        # keeps an explicit OPEN container, update the same statistic manually.
        if count_generated and hasattr(self.open_nodes, 'entry_count'):
            self.open_nodes.entry_count += 1
        return True

    def _empty_solution_now(self, iteration=0):
        """Return an empty solution with useful runtime/stat metadata."""
        sol = TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)
        sol.time_to_solve = time.time() - self.start_time
        sol.iteration = iteration
        sol.is_solved = False
        sol.success_rate = 0
        return sol

    def find_solution_eecbs_tu_strong(self, min_best_case=False, time_lim=60, soc=True,
                                      use_cat=True, existing_cons=None, curr_time=(0, 0),
                                      use_pc=True, use_bp=True, focal_w=1.1,
                                      conflict_weight=0.25, pair_weight=1.0,
                                      edge_weight=2.0, low_level_focal_w=None,
                                      use_low_level_focal=False, use_wdg=False,
                                      wdg_weight=1.0, wdg_overlap_weight=0.05,
                                      wdg_cap_pair_weight=10.0, verbose=False):
        if focal_w < 1:
            raise ValueError('focal_w must be >= 1')
        if low_level_focal_w is None:
            low_level_focal_w = focal_w if use_low_level_focal else 1.0
        if low_level_focal_w < 1:
            raise ValueError('low_level_focal_w must be >= 1')

        self.__initialize_class_variables(curr_time, min_best_case, soc, use_cat)
        self.use_low_level_focal = bool(use_low_level_focal)
        self.low_level_focal_w = low_level_focal_w
        self._eecbs_path_aware_duplicates = True
        if not self.create_root(existing_cons, time_limit=time_lim):
            return self._empty_solution_now(iteration=0)

        open_nodes = _EECBSOpen(
            self, focal_w, conflict_weight, pair_weight, edge_weight,
            use_wdg, wdg_weight, wdg_overlap_weight, wdg_cap_pair_weight
        )
        open_nodes.insert(self.root, count_generated=False)
        iteration = 0

        try:
            while len(open_nodes):
                if time.time() - self.start_time > time_lim:
                    return self._empty_solution_now(iteration=iteration)
                iteration += 1
                best_node = open_nodes.pop_best()
                if best_node is None:
                    break
                expanded_sig = self._eecbs_signature(best_node)
                self.__insert_into_closed_list(best_node)
                best_node.iteration = iteration

                if self.is_edge_conflict(best_node):
                    best_node.iteration_with_edge_conflicts += 1
                if verbose and iteration % 100 == 0:
                    cleanup = open_nodes.best_cleanup_node() or best_node
                    print('[EECBS-TU] iter=', iteration, 'open=', len(open_nodes),
                          'generated=', self.open_nodes.entry_count,
                          'best_lb=', self._node_lower_bound(cleanup),
                          'expanded_cost=', self._node_primary_cost(best_node),
                          'conf=', best_node.conf_num)
                if best_node.conf_num == 0:
                    sol = self.create_solution(best_node, is_solved=True)
                    sol.iteration = iteration
                    return sol

                new_constraints, c1, c2, is_cardinal = self.find_best_conflict(
                    best_node, time_lim, use_pc, True
                )
                if use_bp and not is_cardinal and self.can_bypass(best_node, new_constraints, c1, c2):
                    self.closed_nodes.discard(expanded_sig)
                    open_nodes.insert(best_node, count_generated=False)
                    continue
                open_nodes.insert(c1)
                open_nodes.insert(c2)

            if verbose:
                print('Empty EECBS OPEN list - No Solution')
            return self._empty_solution_now(iteration=iteration)
        except OutOfTimeError:
            return self._empty_solution_now(iteration=iteration)

    def find_solution_eecbs_p_strong(self, min_best_case=False, time_lim=60, soc=True,
                                     use_cat=True, existing_cons=None, curr_time=(0, 0),
                                     use_pc=True, use_bp=True, focal_w=1.1,
                                     conflict_weight=0.25, pair_weight=1.0,
                                     edge_weight=2.0, low_level_focal_w=None,
                                     use_low_level_focal=False, use_wdg=False,
                                     wdg_weight=1.0, wdg_overlap_weight=0.05,
                                     wdg_cap_pair_weight=10.0,
                                     compute_optimal_policy=False, max_intervals=256,
                                     verbose=False, k_safe=1.0):
        if focal_w < 1:
            raise ValueError('focal_w must be >= 1')
        if low_level_focal_w is None:
            low_level_focal_w = focal_w if use_low_level_focal else 1.0
        if low_level_focal_w < 1:
            raise ValueError('low_level_focal_w must be >= 1')

        self.__initialize_class_variables(curr_time, min_best_case, soc, use_cat)
        self.use_low_level_focal = bool(use_low_level_focal)
        self.low_level_focal_w = low_level_focal_w
        self._eecbs_path_aware_duplicates = True
        if not self.create_root(existing_cons, time_limit=time_lim):
            return self._empty_solution_now(iteration=0)

        open_nodes = _EECBSOpen(
            self, focal_w, conflict_weight, pair_weight, edge_weight,
            use_wdg, wdg_weight, wdg_overlap_weight, wdg_cap_pair_weight
        )
        open_nodes.insert(self.root, count_generated=False)
        iteration = 0

        try:
            while len(open_nodes):
                if time.time() - self.start_time > time_lim:
                    return self._empty_solution_now(iteration=iteration)
                iteration += 1
                best_node = open_nodes.pop_best()
                if best_node is None:
                    break
                expanded_sig = self._eecbs_signature(best_node)
                self.__insert_into_closed_list(best_node)
                best_node.iteration = iteration
                pp = self.potential_presence(best_node.sol.paths, self.tu_problem.weights)

                if verbose and iteration % 100 == 0:
                    cleanup = open_nodes.best_cleanup_node() or best_node
                    print('[EECBS-P] iter=', iteration, 'open=', len(open_nodes),
                          'generated=', self.open_nodes.entry_count,
                          'best_lb=', self._node_lower_bound(cleanup),
                          'expanded_cost=', self._node_primary_cost(best_node),
                          'conf=', best_node.conf_num,
                          'optimal_policy=', compute_optimal_policy)

                if best_node.conf_num == 0 or not best_node.conflicts:
                    sol = self.create_solution(best_node, success_rate=1, is_solved=True, execution_policy=pp)
                    sol.iteration = iteration
                    return sol
                if self.is_edge_conflict(best_node):
                    best_node.iteration_with_edge_conflicts += 1

                remaining_time = time_lim - (time.time() - self.start_time)
                if remaining_time <= 0:
                    return self._empty_solution_now(iteration=iteration)

                new_pp, is_node_conflict, achieved_k = self._reduce_ct_conflicts_k_safe_status(
                    best_node.sol.paths, copy.deepcopy(pp), self.tu_problem.weights,
                    time_lim, best_node.conflicts, k_safe=k_safe,
                    split=compute_optimal_policy, max_intervals=max_intervals)
                if new_pp is None:
                    return self._empty_solution_now(iteration=iteration)
                if new_pp != [] and not self._policy_has_empty_interval_set(new_pp):
                    sol = self.create_solution(best_node, success_rate=achieved_k, is_solved=True, execution_policy=new_pp)
                    sol.iteration = iteration
                    return sol

                new_constraints, c1, c2, is_cardinal = self.find_best_conflict(
                    best_node, time_lim, use_pc, is_node_conflict
                )
                if use_bp and not is_cardinal and self.can_bypass(best_node, new_constraints, c1, c2):
                    self.closed_nodes.discard(expanded_sig)
                    open_nodes.insert(best_node, count_generated=False)
                    continue
                open_nodes.insert(c1)
                open_nodes.insert(c2)

            if verbose:
                print('Empty EECBS-P OPEN list - No Solution')
            return self._empty_solution_now(iteration=iteration)
        except OutOfTimeError:
            return self._empty_solution_now(iteration=iteration)

    def find_solution(self, min_best_case=False, time_lim=60, soc=True, use_cat=True, existing_cons=None,
                      curr_time=(0, 0), use_pc=False, use_bp=False):
        """
        The main function - returns a solution consisting of a path for each agent and the total cost of the solution.
        This is an implementation of CBS' basic pseudo code. The main difference comes in the creation of constraints
        in the low-level search.

        root - the root node. Contains no constraints.
        open - the open list.
        sum_of_costs - How to count the solution cost. If it's true, the solution cost is the  sum of all individual
        paths. Otherwise the cost is the longest path in the solution.
        time_limit - Unsurprisingly, the maximum time for this function to run (in seconds)
        """
        self.__initialize_class_variables(curr_time, min_best_case, soc, use_cat)
        self.use_low_level_focal = False
        if not self.create_root(existing_cons, time_limit=time_lim):
            #raise OutOfTimeError("Couldn't find initial solution")
            return self.create_solution(ConstraintNode())

        try:
            iteration = 0
            while self.open_nodes:
                iteration += 1

                best_node = self.open_nodes.pop()
                self.__insert_into_closed_list(best_node)

                best_node.iteration = iteration

                if time.time() - self.start_time > time_lim:
                    #raise OutOfTimeError('Ran out of time :-(')
                    return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)

                if self.is_edge_conflict(best_node): #len(best_node.edge_conflicts)>1:
                    #print("edge conflicts at iteration : ", best_node.iteration)
                    best_node.iteration_with_edge_conflicts +=1
                if best_node.conf_num == 0:  # Meaning that there are no conflicts.
                    return self.create_solution(best_node, is_solved=True)
                #print("yes")
                new_constraints, c1, c2, is_cardinal = self.find_best_conflict(best_node, time_lim, use_pc, True)
                if use_bp and not is_cardinal and self.can_bypass(best_node, new_constraints, c1, c2):
                    continue
                self.__insert_open_node(c1)
                self.__insert_open_node(c2)
            print("Empty open list - No Solution")
            #return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)
            return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)

        except OutOfTimeError:  # Ran out of time. Return an empty solution.
            return TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)

    def __initialize_class_variables(self, curr_time, min_best_case, soc, use_cat):
        """
        Resets all the variables and initializes them with default values
        """
        self.start_time = time.time()
        self.curr_time = curr_time
        self.min_best_case = min_best_case
        self.soc = soc
        self.use_cat = use_cat
        self.computed_c_nodes = {}  # Dictionary mapping constraints, conf_num -> Constraint Node
        self.final_constraints = None
        self.open_nodes = OpenListHeap()
        self.closed_nodes = set()
        # Low-level focal search is opt-in. Classical CBS-TU and CBS_P calls
        # keep the original low-level A* unless a solver explicitly enables it
        # after initialization.
        self.use_low_level_focal = False
        self.low_level_focal_w = 1.1
        self._eecbs_path_aware_duplicates = False
        self._eecbs_path_aware_duplicates = False

    def create_solution(self, best_node, success_rate=1, is_solved=False, execution_policy=None):
        """
        Compiles together all necessary parameters for the final solution that will be returned.
        :param best_node: The node containing the solution
        :return: The proper Time Uncertain Solution.
        """

        best_node.sol.time_to_solve = time.time() - self.start_time
        best_node.sol.compute_solution_cost(self.soc)
        best_node.sol.nodes_generated = self.open_nodes.entry_count
        best_node.sol.iteration = best_node.iteration
        best_node.sol.is_solved = is_solved
        best_node.sol.iteration_with_edge_conflicts = best_node.iteration_with_edge_conflicts
        best_node.sol.success_rate = success_rate
        # Execution policy/envelope: maps each agent to the list of admissible
        # arrival intervals for each vertex occurrence in its plan.
        #
        # Format:
        #   solution.execution_policy[agent][k] = [[l1, u1], [l2, u2], ...]
        #
        # For the original UB version, each occurrence usually has one interval.
        # For the minimal-split version, an occurrence may have several
        # subintervals.
        if execution_policy is not None:
            canonical_policy = copy.deepcopy(execution_policy)
            canonical_policy = self._canonicalize_policy_or_fail(
                canonical_policy,
                plans=best_node.sol.paths
            )
            best_node.sol.execution_policy = canonical_policy
        else:
            best_node.sol.execution_policy = None
        self.final_constraints = best_node.constraints
        best_node.sol.constraints = best_node.constraints
        best_node.sol.sic = self.root.sol.cost

        return best_node.sol


    def _compute_low_level_path(self, constraints, agent, start_pos, goal_pos, conf_table,
                                mbc=True, time_limit=5, curr_time=(0, 0),
                                pos_cons=None, suboptimal=False):
        """Dispatch to the classical or focal low-level search.

        Classical find_solution() leaves self.use_low_level_focal = False and
        therefore still calls ConstraintAstar.compute_agent_path().

        find_solution_eecbs_tu_strong() sets self.use_low_level_focal = True
        and therefore calls ConstraintAstar.compute_agent_path_focal() when the
        method exists. This keeps the two solvers cleanly separated.
        """
        if getattr(self, 'use_low_level_focal', False):
            if not hasattr(self.planner, 'compute_agent_path_focal'):
                raise AttributeError(
                    'ConstraintAstar.compute_agent_path_focal() is missing. '
                    'Use the modified constraint_A_star.py with low-level focal search.'
                )
            return self.planner.compute_agent_path_focal(
                constraints, agent, start_pos, goal_pos, conf_table,
                mbc=mbc, time_limit=time_limit, curr_time=curr_time,
                pos_cons=pos_cons, suboptimal=suboptimal,
                focal_w=getattr(self, 'low_level_focal_w', 1.1)
            )

        return self.planner.compute_agent_path(
            constraints, agent, start_pos, goal_pos, conf_table,
            mbc=mbc, time_limit=time_limit, curr_time=curr_time,
            pos_cons=pos_cons, suboptimal=suboptimal
        )

    def create_root(self, existing_cons=None, time_limit=60):
        self.root = Cn()
        self.compute_all_paths_and_conflicts(self.root, time_limit = time_limit)
        if not len(self.root.sol.paths):
            print("No solution for cbs root node")
            return False
        self.root.sol.compute_solution_cost(self.soc)
        if existing_cons:
            self.root.constraints = Cn.append_constraints(self.root.constraints, existing_cons)
        self.__insert_open_node(self.root)  # initialize the list with root
        return self.root

    def find_best_conflict(self, node, time_lim, use_pc, is_node_conflict):
        """
        Tries to find cardinal, semi-cardinal or non-cardinal conflicts in the given node, in that order of priority.
        If the conflict found is cardinal, the function will return a tuple of <constraint, True> otherwise it will
        return <constraint, False> in order to assess later on if it's cardinal or not.
        :param use_pc: Whether to use prioritizing conflicts or not.
        :param time_lim: The time limit to pass over. This function can be quite long
        :param node: The node who we're searching a solution for.
        :return: A tuple of the constraint and a boolean indicating if it's cardinal or not (for the bypass)
        """

        if not use_pc:
            new_con, c1, c2 = self.generate_children_nodes(node, time_lim, is_node_conflict)
            return new_con, c1, c2, True
        semi_cardinals, non_cardinals = [], []
        sorted_constraints = self.get_sorted_constraints(node.conflicts)
        for constraint in sorted_constraints:
            c1, c2, conf_type = self.get_conflict_type(constraint, node, time_lim)
            if conf_type == 'cardinal':
                return constraint, c1, c2, True
            elif conf_type == 'semi-cardinal':
                semi_cardinals.append((constraint, c1, c2, False))
            else:  # Only update a non-cardinal conf if we haven't found even a semi one
                non_cardinals.append((constraint, c1, c2, False))

        return semi_cardinals[0] if len(semi_cardinals) > 0 else non_cardinals[0]

    def generate_children_nodes(self, node, time_lim, is_node_conflict):
        """Generate the two CBS children for the requested conflict type.

        This function is intentionally strict: if reduction says that an
        unreducible edge/vertex conflict remains, that conflict type must exist
        in the CT node. Otherwise, the reduction/validation code is inconsistent
        and should be fixed rather than masked.
        """
        new_constraints = self.find_single_conflict(node, is_node_conflict)

        if not new_constraints:
            empty_1 = ConstraintNode()
            empty_1.sol = TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)
            empty_2 = ConstraintNode()
            empty_2.sol = TimeUncertaintySolution.empty_solution(self.open_nodes.entry_count)
            return set(), empty_1, empty_2

        res = [new_constraints]
        for new_conf in sorted(new_constraints, key=_stable_key):
            c = self.generate_constraint_node(new_conf, node, time_lim)
            res.append(c)

        if len(res) != 3:
            raise RuntimeError(
                f"CBS branching expected exactly two child constraints, got {len(res) - 1}: {new_constraints}"
            )

        return tuple(res)

    def get_conflict_type(self, constraints, node, time_lim):
        """
        given a constraint and father node, we check what type of constraint this is. If it's cardinal, then we
        :param time_lim:
        :param constraints: The constraints we are checking
        :param node: The node whose children we are generating
        :return: a tuple <constraints, conflict type>
        """

        c1 = self.generate_constraint_node(constraints[0], node, time_lim)
        c2 = self.generate_constraint_node(constraints[1], node, time_lim)

        if self.min_best_case:
            node_cost, c1_cost, c2_cost = node.sol.cost[0], c1.sol.cost[0], c2.sol.cost[0]
        else:
            node_cost, c1_cost, c2_cost = node.sol.cost[1], c1.sol.cost[1], c2.sol.cost[1]
        if node_cost < c1_cost and node_cost < c2_cost:
            return c1, c2, 'cardinal'
        elif (c1_cost > node_cost and c2_cost == node_cost) or (c2_cost > node_cost and c1_cost == node_cost):
            return c1, c2, 'semi-cardinal'
        else:
            return c1, c2, 'non-cardinal'

    def find_single_conflict(self, node, is_node_conflict):
        """
        Given a solution, this function will validate it.
        i.e checking if any conflicts arise. If there are, returns two constraints: For a conflict (a1,a2,v1,v2,t1,t2),
        meaning agents a1 and a2 cannot both be at vertex v1 or v2 or the edge (v1,v2) between t1 and t2, the function
        will return ((a1,v1,v2,t1,t2), (a2,v1,v2,t1,t2))
        :param node: The node whose solution we are validating.
        :returns: A tuple where the first item is the new constraints and the second is a Boolean indicating whether
        the conflict found is cardinal or not (True for cardinal, False otherwise)
        """
        node.sol.create_movement_tuples()

        if is_node_conflict:
            #print("yes")
            new_vertex_constraints, count = node.find_all_vertex_conflicts()
            for vertex, conflicts in sorted(new_vertex_constraints.items(), key=lambda item: _stable_key(item[0])):
                for conflict in sorted(conflicts, key=_stable_key):
                    node.add_conflicting_agents(conflict[0], conflict[1])
                    return self.extract_vertex_constraints(*conflict, vertex)

        # Check for edge conflicts
        new_edge_swap_constraints, count = node.find_all_edge_conflicts()
        for edge, conflicts in sorted(new_edge_swap_constraints.items(), key=lambda item: _stable_key(item[0])):
            for conflict in sorted(conflicts, key=_stable_key):
                node.add_conflicting_agents(conflict[0], conflict[1])
                return self.extract_edge_cons(*conflict, edge)

        return set()

    def __check_conf_in_visited_node(self, visited_nodes, move_i, interval_i, agent_i):
        """
        Checks for a conflict in a node that at least two agents have been at, possible in overlapping times.
        :param visited_nodes: A dictionary of nodes that agents have been at
        :param move_i: The node location
        :param interval_i: The interval of the visiting agent
        :param agent_i: The agent's ID
        :return: Returns a constraint if there is a conflict, otherwise None
        """
        for occupancy in sorted(visited_nodes[move_i[1]], key=_stable_key):  # Iterate over the times agents have been at this node
            if occupancy[0] != agent_i and Cas.overlapping(interval_i, occupancy[1]):  # There is a conflict.
                return self.extract_vertex_constraints(agent_i, occupancy[0], interval_i, occupancy[1], move_i[1])
        return None

    def extract_vertex_constraints(self, agent_i, agent_j, interval_i, interval_j, vertex):
        """
        Helper function. We know that at that time interval there is some conflict between two given agents on the
        given vertex. This function creates the proper tuple of constraints.

        Interval example:     ______________
                        _____|_\_\_\_\_|<---- The time we want to isolate - Maximal time of overlap.

            Constraints will be of the form (agent, conflict_node, time)
        """
        t = self.__pick_times_to_constrain(interval_i, interval_j)
        return {(agent_i, vertex, t)}, {(agent_j, vertex, t)}

    def extract_edge_cons(self, agent_i, agent_j, interval_i, interval_j, dir_i, dir_j, conflict_edge):
        """
        We know there's a conflict at some edge, and agent1 cannot BEGIN traversing it at time 1 and agent 2 cannot
        begin traversing it at time 2. Time 1 and time 2 must be computed through the given intervals and the time to
        traverse the conflicting edge.

        returns the appropriate set of constraints: (agent_ID, (prev_node, conflict_node), time)

        Note: In the case of an edge with a traversal time of 1, theoretically, the agent does not spend any time on it
        and arrives at the end of the edge instantly at 1 time tick. We must address these kind of edges differently.
        :param agent_i: First agent
        :param agent_j: Second agent
        :param interval_i: The interval between BEGINNING of movement until ARRIVAL at next vertex for first agent
        :param interval_j: Same, but for second agent
        :param dir_j: The direction agent i is going ('f' for forwards, 'b' for backwards)
        :param dir_i: The direction agent j is going ('f' for forwards, 'b' for backwards)
        :param conflict_edge: The edge where the conflict occurs
        :return: The appropriate constraints
        """

        agent_i_constraints = set()
        agent_j_constraints = set()

        if dir_i == dir_j and interval_i[1] != interval_j[1]:  # same direction
            if interval_i[1] < interval_j[1]:  # i arrives first, his last tick is not a problem.
                interval_i = interval_i[0], interval_i[1] - 1
            else:  # j arrives first, his last tick is not a problem.
                interval_j = interval_j[0], interval_j[1] - 1

        t = self.__pick_times_to_constrain(interval_i, interval_j)
        agent_i_constraints.add((agent_i, conflict_edge, t))
        agent_j_constraints.add((agent_j, conflict_edge, t))

        return agent_i_constraints, agent_j_constraints

    def compute_all_paths_and_conflicts(self, root, time_limit = 60):
        """
        A function that computes the paths for all agents, i.e a solution. Used for the root node before any constraints
        are added.
        :param root:
        :param use_cat:
        :return:
        """

        for agent_id, agent_start in sorted(self.tu_problem.start_positions.items(), key=lambda item: _stable_key(item[0])):
            remaining_time = time_limit - (time.time() - self.start_time)
            if remaining_time <= 0:
                print("Root low-level planning timed out before agent " + str(agent_id))
                root.solution = None
                return

            agent_plan = self._compute_low_level_path(
                root.constraints, agent_id, agent_start, self.tu_problem.goal_positions[agent_id], root.conflict_table,
                mbc=self.min_best_case, curr_time=self.curr_time, time_limit=remaining_time)

            if len(agent_plan.path):  # Solution found
                root.sol.paths[agent_id] = agent_plan
            else:  # No solution for a particular agent
                print("No solution for agent number " + str(agent_id))
                root.solution = None
                return
        root.sol.add_stationary_moves()  # inserts missing time steps
        root.conflicts, root.edge_conflicts  = root.find_all_conflicts()

        if self.use_cat:
            root.update_conflict_avoidance_table()

    @staticmethod
    def __get_best_node(open_list):
        """
        this function returns the best node in the open list. I used a function since the best node might be something
        else depending on the search type (lowest cost vs shortest time etc.)
        """
        return open_list.pop()

    def __insert_open_node(self, new_node):
        """
        Simply inserts new_node into the list open_nodes and sorts the list. I used a function for modularity's
        sake - sometimes what is considered to be the "best node" is the lowest cost, while sometimes least conflicts
        or shortest time.
        """
        if self.min_best_case:
            self.open_nodes.push(new_node, new_node.sol.cost[0], new_node.conf_num, new_node.sol.cost[1])
        else:
            self.open_nodes.push(new_node, new_node.sol.cost[1], new_node.conf_num, new_node.sol.cost[0])

    def __insert_into_closed_list(self, new_node):
        """
        Inserts the new node into the closed list. We use a function since inserting the node into the closed
        list demands that we create frozen set out of all constraints for hashing.
        :param new_node: The node to insert.
        """
        sig = self._eecbs_signature(new_node) if getattr(self, '_eecbs_path_aware_duplicates', False) else self._ct_signature(new_node)
        self.closed_nodes.add(sig)

    @staticmethod
    def __pick_times_to_constrain(interval_i, interval_j, t='max'):
        """
        Returns the time ticks to constrain in the conflict interval. For example if the intervals are (4,17) and (8,22)
        , the conflicting interval will be (8,17). Note that the format for constraining e.g. time tick 4 is (4, 4).
        This is so we can introduce range constraints with ease later.
        :param interval_i: Agent i's interval for occupying a resource
        :param interval_j: Agent j's interval for occupying a resource
        :param t: what time ticks to constrain. Can be max or min. Any other value will choose the average time.
        :return: The time tick to constraints. Currently returns the maximum time of the CONFLICT interval.
        """
        # Choose the minimum between the max times in the conflict interval
        conf_interval = max(interval_i[0], interval_j[0]), min(interval_i[1], interval_j[1])
        if t == 'max':
            return conf_interval[1], conf_interval[1]
        elif t == 'min':
            return conf_interval[0], conf_interval[0]
        else:
            return int((conf_interval[0] + conf_interval[1]) / 2), int((conf_interval[0] + conf_interval[1]) / 2)

    def __check_previously_conflicting_agents(self, solution, prev_agents):
        """
        We check if the agents that previously collided still collide. Empirically, this is usually the case
        and is one of the drawbacks of CBS. This attempts to speed it up.
        :param solution: The given solution of the node
        :param prev_agents: The previously colliding agents
        :return: The proper constraint if one exists, otherwise "None"
        """
        if not prev_agents:
            return None

        agent_i = prev_agents[0]
        agent_j = prev_agents[1]

        visited_nodes = defaultdict(set)  # A dictionary containing all the nodes visited
        for agent in [agent_i, agent_j]:
            for move in solution.paths[agent].path:
                interval = move[0]
                if not move[1] in visited_nodes:  # First time an agent has visited this node
                    visited_nodes[move[1]].add((agent, interval))  # Add the interval to the set.
                    continue
                else:  # Some other agent HAS been to this node..
                    conf = self.__check_conf_in_visited_node(visited_nodes, move, interval, agent)
                    if conf:
                        return conf
                    visited_nodes[move[1]].add((agent, interval))  # No conflict, add to vertex set.

        positions = {}

        for agent in [agent_i, agent_j]:
            path = solution.tuple_solution[agent]
            for move in path:
                edge = move[1]
                if edge not in positions:
                    positions[edge] = set()
                if move[0][1] - move[0][0] > 1:  # Edge weight is more than 1
                    occ_time = move[0][0], move[0][1] - 1
                    positions[edge].add((agent, (move[0][0], move[0][1] - 1), move[2]))
                    for pres in sorted(positions[edge], key=_stable_key):
                        if pres[0] != agent:
                            if pres[2] == move[2] and Cn.strong_overlapping(occ_time, pres[1]):
                                return self.extract_edge_cons(pres[0], agent, pres[1], move[0], pres[2], occ_time, edge)
                            elif pres[2] != move[2] and Cas.overlapping(occ_time, pres[1]):
                                return self.extract_edge_cons(pres[0], agent, pres[1], move[0], pres[2], occ_time, edge)
                else:
                    # Agent begins to travel at move[0][0] and arrives at move[0][1]
                    positions[edge].add((agent, move[0], move[2]))
                    for pres in sorted(positions[edge], key=_stable_key):
                        if pres[0] != agent and Cn.strong_overlapping(move[0], pres[1]):
                            return self.extract_edge_cons(pres[0], agent, pres[1], move[0], pres[2], move[2], edge)
        return None

    def generate_constraint_node(self, new_cons, best_node, time_limit):
        merged_constraints = Cn.append_constraints(best_node.constraints, new_cons).items()
        key_cons = frozenset((k, tuple(sorted(val, key=_stable_key))) for k, val in merged_constraints)
        # Classical CBS can cache solely by constraints because its low-level
        # search is deterministic. EECBS with CAT/focal search may produce a
        # different path combination for the same constraints, so constraint-
        # only pruning would discard useful nodes.
        if not getattr(self, '_eecbs_path_aware_duplicates', False):
            if key_cons in self.closed_nodes:
                empty_node = Cn()
                empty_node.sol = TimeUncertaintySolution.empty_solution()
                return empty_node
            if key_cons in self.computed_c_nodes:
                return self.computed_c_nodes[key_cons]
        new_node = Cn(new_constraints=new_cons, parent=best_node)
        agent = sorted(new_cons, key=_stable_key)[0][0]  # Deterministically extract agent index.
        time_passed = time.time() - self.start_time
        remaining_time = time_limit - time_passed
        if remaining_time <= 0:
            new_plan = TimeUncertaintyPlan.get_empty_plan(agent)
        else:
            new_plan = self._compute_low_level_path(new_node.constraints, agent,
                self.tu_problem.start_positions[agent],
                self.tu_problem.goal_positions[agent],
                best_node.conflict_table,
                mbc=self.min_best_case, time_limit=remaining_time,
                curr_time=self.curr_time)  # compute the path for a single agent.
        #if time.time() - self.start_time > time_limit:
            #raise OutOfTimeError('Ran out of time :-(')
            #return self.create_solution(best_node)
        new_node.update_solution(new_plan, self.use_cat, self.soc)
        new_node.update_conflicts(agent)
        if not getattr(self, '_eecbs_path_aware_duplicates', False):
            self.computed_c_nodes[key_cons] = new_node
        return new_node

    def can_bypass(self, best_node, new_constraints, c1, c2):
        """
        Performs the bypass maneuver of ICBS. We look at the two immediate children of the best_node, and if one of
        them offers a helpful bypass, we replace best_node's solution with that child's solution. There is no need to
        add a new node to the CT, just update the cost of best_node, insert it into open and continue.
        Note that best_node should be chosen again for expansion since it previously had the best cost.s
        :param c1: First child, generated from the first new constraint
        :param c2: Second child, generated from the second new constraint
        :param best_node: The current best node in open
        :param new_constraints: The new constraints found for the path of best_node
        :return: If a bypass was found, updates best_node and returns true. Otherwise False.
        """
        children = [c1, c2]
        for idx, new_con_set in enumerate(new_constraints):
            child = children[idx]  # There are only 2 new constraints, we will insert each one into "open"
            if ((self.min_best_case and best_node.sol.cost[0] == child.sol.cost[0]) or
                (not self.min_best_case and best_node.sol.cost[1] == child.sol.cost[1])) and \
                    child.conf_num < best_node.conf_num:
                agent = sorted(new_con_set, key=_stable_key)[0][0]  # Deterministically extract agent index.
                best_node.update_solution(child.sol.paths[agent], self.use_cat, self.soc)
                best_node.conf_num = child.conf_num
                best_node.conflicts = child.conflicts
                self.__insert_open_node(best_node)
                return True

        return False  # No bypass found, just

    def get_sorted_constraints(self, all_conflicts):
        """
        Receives a dictionary of conflicts creates constraints out of them. Then sorts the constraints based on the
        time that is constrained. This is because we'd rather sort earlier conflicts first.
        :param all_conflicts: A dictionary of location - > [list of conflicts]
        :return:
        """
        result = []
        for loc, conflicts in sorted(all_conflicts.items(), key=lambda item: _stable_key(item[0])):
            for conf in sorted(conflicts, key=_stable_key):
                conflict = conf + (loc,)
                if type(conflict[-1][0]) == int:  # Vertex constraint
                    constraints = self.extract_vertex_constraints(*conflict)
                else:  # Edge constraint
                    constraints = self.extract_edge_cons(*conflict)
                result.append(constraints)

        def _constraint_pair_key(pair):
            left = sorted(pair[0], key=_stable_key)[0]
            right = sorted(pair[1], key=_stable_key)[0]
            return (left[2][0], left[2][1], _stable_key(left), _stable_key(right))

        result.sort(key=_constraint_pair_key, reverse=False)  # Deterministic conflict ordering.
        return result