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

    def get_makespan(self, op_durations):
        """
        Calculates the makespan (length of the longest path) of the graph.

        Args:
            op_durations (dict or list): A mapping from operation index (node) 
                                         to its duration.

        Returns:
            float: The makespan, or float('inf') if the graph contains a cycle.
        """
        if self.has_cycle(): 
            return float('inf')

        try:
            topological_order = self.topological_sort()
        except ValueError: # Cycle detected by topological_sort
            return float('inf')

        ef = {op_idx: 0 for op_idx in range(self.num_ops)} # Earliest Finish times

        for op_idx in topological_order:
            # Determine current operation's duration
            current_op_duration = 0
            if isinstance(op_durations, dict):
                current_op_duration = op_durations.get(op_idx, 0)
            elif isinstance(op_durations, list):
                 # Ensure index is within bounds
                 if 0 <= op_idx < len(op_durations):
                     current_op_duration = op_durations[op_idx]
            
            # Calculate Earliest Start (max EF of predecessors)
            max_ef_preds = 0
            if op_idx in self.rev_adj:
                 for pred_idx in self.rev_adj[op_idx]:
                      max_ef_preds = max(max_ef_preds, ef.get(pred_idx, 0))
            
            # Calculate Earliest Finish
            ef[op_idx] = max_ef_preds + current_op_duration

        # Makespan is the maximum EF of all operations
        makespan = 0.0
        if ef:
            makespan = max(ef.values())
        
        return makespan

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
            return []

        es = {op_idx: 0 for op_idx in range(self.num_ops)}
        ef = {op_idx: 0 for op_idx in range(self.num_ops)}
        
        try:
            topological_order = self.topological_sort()
        except ValueError: # Cycle detected by topological_sort
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

        ls = {op_idx: makespan for op_idx in range(self.num_ops)}
        lf = {op_idx: makespan for op_idx in range(self.num_ops)}

        for op_idx in reversed(topological_order):
            current_op_duration = op_durations[op_idx] if isinstance(op_durations, list) else op_durations.get(op_idx, 0)
            # LF is min LS of successors
            min_ls_succs = makespan
            if op_idx in self.adj: # Check if op_idx has successors
                if self.adj[op_idx]:
                    min_ls_succs_for_op = float('inf') # Initialize for this specific op
                    for succ_idx in self.adj[op_idx]:
                        min_ls_succs_for_op = min(min_ls_succs_for_op, ls[succ_idx])
                    min_ls_succs = min_ls_succs_for_op # Assign if successors were found

            lf[op_idx] = min_ls_succs
            ls[op_idx] = lf[op_idx] - current_op_duration
            
        critical_path = []
        for op_idx in topological_order: # Iterate in topological order for a structured path
            if abs(es[op_idx] - ls[op_idx]) < 1e-6: 
                critical_path.append(op_idx)
        
        return critical_path
