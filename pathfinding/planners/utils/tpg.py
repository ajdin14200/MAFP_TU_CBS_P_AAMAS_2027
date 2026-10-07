from dataclasses import dataclass, field


from dataclasses import dataclass, field


@dataclass
class TPGNode:
    node_id: str
    agent: int
    index: int
    time_interval: tuple
    location: object


@dataclass
class TPGEdge:
    source: str
    target: str
    edge_type: int   # 1 = intra-agent, 2 = inter-agent
    resource: object = None


@dataclass
class TemporalPlanGraph:
    nodes: dict = field(default_factory=dict)   # node_id -> TPGNode
    edges: list = field(default_factory=list)

    def add_node(self, node: TPGNode):
        self.nodes[node.node_id] = node

    def add_edge(self, source, target, edge_type, resource=None):
        self.edges.append(TPGEdge(source, target, edge_type, resource))

    def print_graph(self):
        print("\n=== TPG NODES ===")
        for node_id in sorted(self.nodes):
            n = self.nodes[node_id]
            print(f"{node_id}: agent={n.agent}, loc={n.location}, interval={n.time_interval}")

        print("\n=== TPG EDGES ===")
        for e in self.edges:
            print(f"{e.source} -> {e.target} [type-{e.edge_type}] resource={e.resource}")

def make_node_id(agent, index, vertex):
    return f"a{agent}_{vertex}_{index}"

def intervals_precedence(i1, i2):
    """
    Returns:
        -1 if interval i1 strictly precedes i2
         1 if interval i2 strictly precedes i1
         0 otherwise
    """
    if i1[1] < i2[0]:
        return -1
    if i2[1] < i1[0]:
        return 1
    return 0

def same_edge(m1_from, m1_to, m2_from, m2_to):
    """
    True if the two traversals use the same undirected edge.
    """
    return {m1_from, m1_to} == {m2_from, m2_to}




def build_tpg_from_solution(solution):
    tpg = TemporalPlanGraph()

    # --------------------------------------------------
    # 1) Nodes + type-1 edges (same agent)
    # --------------------------------------------------
    node_map = {}  # (agent, index) -> node_id

    for agent, plan in solution.paths.items():
        for i, movement in enumerate(plan.path):
            interval, vertex = movement

            node_id = make_node_id(agent, i, vertex)
            node_map[(agent, i)] = node_id

            tpg.add_node(TPGNode(
                node_id=node_id,
                agent=agent,
                index=i,
                time_interval=interval,
                location=vertex
            ))

            if i > 0:
                prev_id = node_map[(agent, i - 1)]
                tpg.add_edge(prev_id, node_id, edge_type=1)

    agents = list(solution.paths.keys())

    # --------------------------------------------------
    # 2) Vertex conflicts → type-2 edges
    # --------------------------------------------------
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a1 = agents[i]
            a2 = agents[j]

            path1 = solution.paths[a1].path
            path2 = solution.paths[a2].path

            for idx1, (int1, v1) in enumerate(path1):
                for idx2, (int2, v2) in enumerate(path2):

                    if v1 != v2:
                        continue

                    rel = intervals_precedence(int1, int2)

                    n1 = node_map[(a1, idx1)]
                    n2 = node_map[(a2, idx2)]

                    if rel == -1:
                        tpg.add_edge(n1, n2, 2, ("vertex", v1))
                    elif rel == 1:
                        tpg.add_edge(n2, n1, 2, ("vertex", v1))

    return tpg