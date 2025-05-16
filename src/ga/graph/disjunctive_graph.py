import collections
from .dsu import DSU


class DisjunctiveGraph:
    def __init__(self, num_ops, use_dsu=False):
        self.num_ops = num_ops
        self.adj = {i: set() for i in range(num_ops)}  # Arestas direcionadas
        self.rev_adj = {i: set() for i in range(num_ops)} # Arestas reversas para facilitar o cálculo do grau de entrada
        self.use_dsu = use_dsu
        self.dsu = DSU(num_ops) if use_dsu else None

    def add_edge(self, u, v):
        # Se DSU estiver em uso, verifica se a adição da aresta cria um ciclo.
        # A união retorna False se u e v já estão no mesmo conjunto (ciclo).
        if self.dsu is not None and not self.dsu.union(u, v):
            return False  # Aresta formaria um ciclo
        self.adj[u].add(v)
        self.rev_adj[v].add(u)
        return True

    def remove_edge(self, u, v):
        self.adj[u].discard(v)
        self.rev_adj[v].discard(u)

        # Se DSU estiver em uso, precisa ser reinicializado e reconstruído,
        # pois a remoção de uma aresta pode separar componentes.
        if self.dsu is not None:
            self.dsu = DSU(self.num_ops)
            for curr_u in self.adj:
                for curr_v in self.adj[curr_u]:
                    # Ignora o resultado da união aqui, pois estamos reconstruindo
                    # com base em arestas que já foram validadas como não formando ciclo
                    # no momento de sua adição original (ou o ciclo foi permitido e tratado por has_cycle).
                    self.dsu.union(curr_u, curr_v) # Apenas para reconstruir os conjuntos

    def has_cycle(self):
        visited = set()
        stack = set() # Pilha de recursão para nós atualmente no caminho de visita

        def visit(v):
            if v in stack: # Se o nó já está na pilha de recursão, encontramos um ciclo
                return True
            if v in visited: # Se já visitado e não na pilha, não há ciclo a partir daqui
                return False
            
            visited.add(v)
            stack.add(v)
            
            for w in self.adj.get(v, []): # Itera sobre vizinhos
                if visit(w):
                    return True
            
            stack.remove(v) # Remove da pilha de recursão ao retroceder
            return False

        # Verifica ciclos a partir de todos os nós não visitados
        for v_node in range(self.num_ops):
            if v_node not in visited:
                if visit(v_node):
                    return True
        return False

    def topological_sort(self):
        in_degree = {i: len(self.rev_adj[i]) for i in range(self.num_ops)}
        queue = collections.deque([i for i in range(self.num_ops) if in_degree[i] == 0])
        order = []
        
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in self.adj.get(v, []):
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        
        if len(order) != self.num_ops:
            # Se a ordenação topológica não inclui todos os nós, há um ciclo.
            raise ValueError("Grafo contém um ciclo, ordenação topológica impossível.")
        return order

    def get_makespan(self, op_durations):
        """
        Calcula o makespan (comprimento do caminho mais longo) do grafo.

        Args:
            op_durations (dict ou list): Um mapeamento do índice da operação (nó)
                                         para sua duração.

        Returns:
            float: O makespan, ou float('inf') se o grafo contiver um ciclo.
        """
        # Primeiro, verificamos explicitamente se há ciclo, pois a ordenação topológica
        # também pode lançar um erro, mas esta verificação é mais direta para o makespan.
        if self.has_cycle(): 
            return float('inf')

        try:
            topological_order = self.topological_sort()
        except ValueError: # Ciclo detectado pela ordenação topológica
            return float('inf')

        # ef (Earliest Finish times) - Tempo mais cedo de finalização
        ef = {op_idx: 0 for op_idx in range(self.num_ops)} 

        for op_idx in topological_order:
            current_op_duration = 0
            if isinstance(op_durations, dict):
                current_op_duration = op_durations.get(op_idx, 0)
            elif isinstance(op_durations, list):
                 if 0 <= op_idx < len(op_durations): # Garante que o índice está dentro dos limites
                     current_op_duration = op_durations[op_idx]
            
            # Calcula o Tempo Mais Cedo de Início (ES) como o máximo EF dos predecessores
            max_ef_preds = 0
            # Verifica se op_idx existe em rev_adj e tem predecessores
            if op_idx in self.rev_adj:
                 for pred_idx in self.rev_adj[op_idx]:
                      max_ef_preds = max(max_ef_preds, ef.get(pred_idx, 0)) # Usa ef.get para segurança
            
            # Calcula o Tempo Mais Cedo de Finalização (EF)
            ef[op_idx] = max_ef_preds + current_op_duration

        # O Makespan é o máximo EF de todas as operações
        makespan = 0.0
        if ef: # Garante que ef não está vazio
            makespan = max(ef.values()) if ef else 0.0 # Adicionado if ef else 0.0 para robustez
        
        return makespan

    def get_critical_path(self, op_durations):
        """
        Calcula o caminho crítico no grafo disjuntivo.

        Args:
            op_durations (dict ou list): Mapeamento do índice da operação (nó no grafo)
                                         para sua duração. Se for uma lista, o índice
                                         é o índice da operação.

        Returns:
            list[int]: Uma lista de índices de operações (nós) formando um caminho crítico.
                       Retorna uma lista vazia se o grafo tiver um ciclo ou outros problemas.
        """
        if self.has_cycle(): # Garante que o grafo é um DAG
            return []

        es = {op_idx: 0 for op_idx in range(self.num_ops)} # Earliest Start
        ef = {op_idx: 0 for op_idx in range(self.num_ops)} # Earliest Finish
        
        try:
            topological_order = self.topological_sort()
        except ValueError: # Ciclo detectado
            return []

        # Calcula ES e EF
        for op_idx in topological_order:
            # Determina a duração da operação atual
            current_op_duration = 0
            if isinstance(op_durations, dict):
                current_op_duration = op_durations.get(op_idx, 0)
            elif isinstance(op_durations, list) and 0 <= op_idx < len(op_durations):
                current_op_duration = op_durations[op_idx]
            
            # ES é o máximo EF dos predecessores
            max_ef_preds = 0
            if op_idx in self.rev_adj: 
                 for pred_idx in self.rev_adj[op_idx]:
                      max_ef_preds = max(max_ef_preds, ef.get(pred_idx,0)) # ef[pred_idx] já calculado
            es[op_idx] = max_ef_preds
            ef[op_idx] = es[op_idx] + current_op_duration

        # Calcula o makespan (EF máximo)
        makespan = 0.0
        if ef: # Garante que ef não está vazio
            makespan = max(ef.values()) if ef else 0.0

        ls = {op_idx: makespan for op_idx in range(self.num_ops)} # Latest Start
        lf = {op_idx: makespan for op_idx in range(self.num_ops)} # Latest Finish

        # Calcula LS e LF na ordem topológica reversa
        for op_idx in reversed(topological_order):
            current_op_duration = 0
            if isinstance(op_durations, dict):
                current_op_duration = op_durations.get(op_idx, 0)
            elif isinstance(op_durations, list) and 0 <= op_idx < len(op_durations):
                current_op_duration = op_durations[op_idx]

            # LF é o mínimo LS dos sucessores
            min_ls_succs = makespan # Default se não houver sucessores
            # Verifica se op_idx tem sucessores em self.adj
            if op_idx in self.adj and self.adj[op_idx]:
                min_ls_succs_for_op = float('inf') 
                for succ_idx in self.adj[op_idx]:
                    min_ls_succs_for_op = min(min_ls_succs_for_op, ls.get(succ_idx, makespan)) # ls[succ_idx] já calculado
                min_ls_succs = min_ls_succs_for_op
            
            lf[op_idx] = min_ls_succs
            ls[op_idx] = lf[op_idx] - current_op_duration
            
        critical_path = []
        # Operações no caminho crítico têm ES == LS (ou EF == LF)
        # Iterar na ordem topológica para um caminho estruturado
        for op_idx in topological_order: 
            # Usar uma pequena tolerância para comparações de float
            if abs(es[op_idx] - ls[op_idx]) < 1e-6: 
                critical_path.append(op_idx)

        return critical_path
