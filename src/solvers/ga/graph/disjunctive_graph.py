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
        queue = [i for i in range(self.num_ops) if in_degree[i] == 0]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in self.adj[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        if len(order) != self.num_ops:
            raise ValueError('Grafo contém ciclo!')
        return order
