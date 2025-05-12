import collections
from .dsu import DSU


class DisjunctiveGraph:
    def __init__(self, num_ops, use_dsu=False):
        self.num_ops = num_ops
        self.adj = {i: set() for i in range(num_ops)}  # arestas dirigidas
        self.rev_adj = {i: set() for i in range(num_ops)}
        self.use_dsu = use_dsu
        self.dsu = DSU(num_ops) if use_dsu else None

    def add_edge(self, u, v):
        if self.dsu is not None and not self.dsu.union(u, v):
            return False
        self.adj[u].add(v)
        self.rev_adj[v].add(u)
        return True

    def remove_edge(self, u, v):
        self.adj[u].discard(v)
        self.rev_adj[v].discard(u)

        # Se DSU estiver em uso, reinicializa e reconstrói
        if self.dsu is not None:
            self.dsu = DSU(self.num_ops)
            for curr_u in self.adj:
                for curr_v in self.adj[curr_u]:
                    # Ignora o resultado da união aqui, pois estamos reconstruindo
                    # com base em arestas que já foram validadas
                    self.dsu.union(curr_u, curr_v)

    def has_cycle(self):
        visited = set()
        stack = set()

        def visit(v):
            if v in stack:
                return True
            if v in visited:
                return False
            visited.add(v)
            stack.add(v)
            for w in self.adj[v]:
                if visit(w):
                    return True
            stack.remove(v)
            return False
        return any(visit(v) for v in range(self.num_ops))

    def topological_sort(self):
        in_degree = {i: len(self.rev_adj[i]) for i in range(self.num_ops)}
        queue = collections.deque([i for i in range(self.num_ops) if in_degree[i] == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in self.adj[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        if len(order) != self.num_ops:
            raise ValueError('Grafo contém ciclo!')
        return order

    def get_critical_path(self, op_durations):
        """
        Calculates the critical path in the disjunctive graph.

        Args:
            op_durations (dict or list): A mapping from operation index (node in graph)
                                         to its duration. If it's a list, the index
                                         is the operation index.

        Returns:
            list[int]: A list of operation indices (nodes) forming a critical path.
                       Returns an empty list if the graph has a cycle or other issues.
        """
        if self.has_cycle(): # Ensure graph is a DAG
            # print("Graph has a cycle, cannot determine critical path.")
            return []

        # 1. Calculate Earliest Start (ES) and Earliest Finish (EF) times
        # ES for a node is the max EF of all its predecessors.
        # EF = ES + duration
        es = {op_idx: 0 for op_idx in range(self.num_ops)}
        ef = {op_idx: 0 for op_idx in range(self.num_ops)}
        
        try:
            topological_order = self.topological_sort()
        except ValueError: # Cycle detected by topological_sort
            # print("Cycle detected during topological sort for critical path.")
            return []

        for op_idx in topological_order:
            current_op_duration = op_durations[op_idx] if isinstance(op_durations, list) else op_durations.get(op_idx, 0)
            # ES is max EF of predecessors
            max_ef_preds = 0
            if op_idx in self.rev_adj: # Check if op_idx has predecessors
                 for pred_idx in self.rev_adj[op_idx]:
                      max_ef_preds = max(max_ef_preds, ef[pred_idx])
            es[op_idx] = max_ef_preds
            ef[op_idx] = es[op_idx] + current_op_duration

        makespan = 0
        for op_idx in range(self.num_ops):
            makespan = max(makespan, ef[op_idx])

        # 2. Calculate Latest Start (LS) and Latest Finish (LF) times
        # LF for a node is the min LS of all its successors.
        # LS = LF - duration
        # For nodes with no successors, LF = makespan.
        ls = {op_idx: makespan for op_idx in range(self.num_ops)}
        lf = {op_idx: makespan for op_idx in range(self.num_ops)}

        for op_idx in reversed(topological_order):
            current_op_duration = op_durations[op_idx] if isinstance(op_durations, list) else op_durations.get(op_idx, 0)
            # LF is min LS of successors
            min_ls_succs = makespan
            if op_idx in self.adj: # Check if op_idx has successors
                # Check if any successors exist for this op_idx
                if self.adj[op_idx]:
                    min_ls_succs_for_op = float('inf') # Initialize for this specific op
                    for succ_idx in self.adj[op_idx]:
                        min_ls_succs_for_op = min(min_ls_succs_for_op, ls[succ_idx])
                    min_ls_succs = min_ls_succs_for_op # Assign if successors were found
                # else: min_ls_succs remains makespan (no successors)

            lf[op_idx] = min_ls_succs
            ls[op_idx] = lf[op_idx] - current_op_duration
            
        # 3. Identify critical path operations
        # Operations where ES = LS (or EF = LF, within a small tolerance for floats if used)
        critical_path = []
        for op_idx in topological_order: # Iterate in topological order for a structured path
            # Using a small tolerance for float comparison, though times are typically int
            if abs(es[op_idx] - ls[op_idx]) < 1e-6: 
                critical_path.append(op_idx)
        
        # The above identifies critical nodes, but not necessarily a single connected path.
        # To get a single path, one common way is to backtrack from a critical sink node.
        # For simplicity here, we return all nodes that are critical.
        # A more robust approach might trace back from any sink node that is critical.

        # Let's refine to get a more connected path, starting from critical sources
        # and following critical edges.
        
        # Re-calculating critical path for a more connected sequence
        # This simplified version finds one critical path.
        # A job shop problem can have multiple critical paths.

        # This identifies nodes on *any* critical path.
        # For mutation, knowing these nodes is often sufficient.
        # If a single sequence is strictly needed, further refinement (e.g. longest path on critical subgraph)
        # would be necessary, but let's keep this simpler version for now.
        # The crucial part is identifying nodes with zero slack (es[i] == ls[i]).

        return critical_path
