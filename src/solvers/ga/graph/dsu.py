class DSU:
    """
    Estrutura Disjoint Set Union (Union-Find) com path compression e union by rank.
    Útil para detecção rápida de ciclos não direcionados e agrupamento dinâmico de componentes.
    Agora com suporte a reset parcial de componentes (DSU parcimonioso).
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False  # Já estão conectados, ciclo detectado
        # Union by rank
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def reset_partial(self, indices):
        """
        Reinicializa apenas os índices fornecidos (raízes e ranks), para uso parcimonioso.
        """
        for i in indices:
            self.parent[i] = i
            self.rank[i] = 0
