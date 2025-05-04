import random
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Any

# Importar das classes base que ainda estão em genetic_operators.py (temporariamente)
# Assumindo que genetic_operators está um nível acima
from ..genetic_operators import CrossoverStrategy, LocalSearchStrategy
# Importar DSU e graph_builder se necessário para DisjunctiveCrossover
# from ..graph.dsu import DSU # Descomentar se DSU for usado diretamente aqui
# from ..graph import GraphBuilder # Ajustar o caminho conforme necessário


class OrderCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    def crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None]*size
        child[a:b] = parent1[a:b]
        fill = [gene for gene in parent2 if gene not in child[a:b]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        if self.local_search_strategy:
            # Precisamos garantir que local_search_strategy seja chamado corretamente
            if hasattr(self.local_search_strategy, 'local_search'):
                return self.local_search_strategy.local_search(child)
            else:
                # Lidar com caso onde a estratégia não tem o método esperado (ou é None)
                # print("Aviso: local_search_strategy inválida ou ausente em OrderCrossover.")
                pass  # Ou retorna child diretamente
        return child


class PMXCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    def crossover(self, parent1, parent2):
        size = len(parent1)
        # Garante que os pais sejam válidos antes de prosseguir (opcional, mas bom)
        if len(set(parent1)) != size or len(set(parent2)) != size:
            # print("Warning: PMX recebeu pais inválidos (duplicatas).")
            # Retorna um dos pais como fallback
            return parent1

        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        # Dicionário para mapeamento reverso (valor -> índice) para performance
        p1_val_to_idx = {val: i for i, val in enumerate(parent1)}
        p2_val_to_idx = {val: i for i, val in enumerate(parent2)}

        # 1. Copia o segmento do primeiro pai
        child[a:b] = parent1[a:b]
        child_segment_set = set(child[a:b])

        # 2. Mapeamento PMX para elementos do segmento de parent2
        for i in range(a, b):
            p2_val = parent2[i]
            if p2_val not in child_segment_set:
                # Segue a cadeia de mapeamento até encontrar um slot vazio fora do segmento
                current_val_from_p1 = parent1[i]
                # Verifica se current_val_from_p1 existe em p2_val_to_idx antes de acessar
                if current_val_from_p1 not in p2_val_to_idx:
                    # Situação inesperada, talvez um dos pais não seja uma permutação válida
                    # print(f"Erro PMX: Valor {current_val_from_p1} de parent1 não encontrado em parent2.")
                    return parent1  # Retorna fallback
                target_idx = p2_val_to_idx[current_val_from_p1]
                while a <= target_idx < b:
                    current_val_from_p1 = parent1[target_idx]
                    # Verifica novamente antes de acessar
                    if current_val_from_p1 not in p2_val_to_idx:
                        # print(f"Erro PMX: Valor {current_val_from_p1} de parent1 não encontrado em parent2.")
                        return parent1  # Retorna fallback
                    target_idx = p2_val_to_idx[current_val_from_p1]
                # Coloca o valor de p2 no slot encontrado
                child[target_idx] = p2_val

        # 3. Preenche os slots restantes (None) com elementos de parent2 que não foram usados
        child_current_elements = set(filter(None, child))
        for i in range(size):
            if child[i] is None:
                if parent2[i] not in child_current_elements:
                    child[i] = parent2[i]
                    child_current_elements.add(parent2[i])
                else:
                    # Se parent2[i] já está no filho, precisamos encontrar um valor de p2 que ainda não está
                    # Esta parte é um pouco mais complexa na implementação padrão PMX,
                    # mas uma alternativa simples é preencher com os restantes não usados.
                    # Adiaremos a lógica de preenchimento mais complexa se necessário.
                    pass

        # 3. (Alternativa mais segura e simples para preencher Nones): Preenche com elementos de P2 que faltam
        p2_elements_not_in_child = [
            elem for elem in parent2 if elem not in child_current_elements]
        idx_fill = 0
        for i in range(size):
            if child[i] is None:
                if idx_fill < len(p2_elements_not_in_child):
                    child[i] = p2_elements_not_in_child[idx_fill]
                    idx_fill += 1
                else:
                    # Isso não deveria acontecer se os pais são permutações válidas
                    # print("Erro no preenchimento PMX - Faltando elementos?")
                    # Como fallback, poderia colocar um placeholder ou retornar um pai
                    # Por ora, vamos assumir que não acontece com pais válidos.
                    pass

        # Verificação final (opcional)
        if len(set(filter(None, child))) != len(list(filter(None, child))) or None in child:
            # print(f"Warning: PMX gerou filho inválido: {child}")
            return parent1  # Retorna pai como fallback

        if self.local_search_strategy:
            if hasattr(self.local_search_strategy, 'local_search'):
                return self.local_search_strategy.local_search(child)
            else:
                # print("Aviso: local_search_strategy inválida ou ausente em PMXCrossover.")
                pass
        return child


class CycleCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    def crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        # Dicionários para mapeamento rápido (valor -> índice) podem otimizar `parent1.index`
        p1_val_to_idx = {val: i for i, val in enumerate(parent1)}

        # Rastreia a qual ciclo cada índice pertence (opcional, mas útil para debug)
        cycles = [0] * size
        cycle = 1
        idx = 0
        while None in child:
            # Encontra o próximo índice não processado
            try:
                idx = child.index(None)
            except ValueError:
                break  # Todos os slots preenchidos

            start_idx = idx
            current_cycle_indices = set()  # Conjunto para rastrear índices no ciclo atual

            while idx not in current_cycle_indices:
                # Segurança extra, não deveria acontecer na lógica principal
                if child[idx] is not None:
                    break
                current_cycle_indices.add(idx)
                child[idx] = parent1[idx] if cycle % 2 == 1 else parent2[idx]
                cycles[idx] = cycle  # Marca o ciclo

                p2_val_at_idx = parent2[idx]
                # Encontra onde o valor de p2 está em p1
                try:
                    # Usa o dicionário para busca rápida
                    idx = p1_val_to_idx[p2_val_at_idx]
                except KeyError:
                    # print(f"Erro Cycle Crossover: Valor {p2_val_at_idx} de parent2 não encontrado em parent1.")
                    return parent1  # Fallback se os pais não forem permutações um do outro

                if idx == start_idx:
                    break  # Fechou o ciclo

            cycle += 1  # Incrementa para o próximo ciclo

        # Verificação final (opcional)
        if None in child:
            # print(f"Erro Cycle Crossover: Filho incompleto: {child}")
            return parent1  # Retorna fallback

        if self.local_search_strategy:
            if hasattr(self.local_search_strategy, 'local_search'):
                return self.local_search_strategy.local_search(child)
            else:
                # print("Aviso: local_search_strategy inválida ou ausente em CycleCrossover.")
                pass
        return child


class PositionBasedCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    def crossover(self, parent1, parent2):
        size = len(parent1)
        if size == 0:
            return []  # Lida com caso de cromossomo vazio
        # Garante que k seja pelo menos 1 e no máximo size // 2 (ou size se size < 2)
        max_k = size // 2 if size > 1 else size
        if max_k < 1:
            k = 1
        else:
            k = random.randint(1, max_k)

        positions = sorted(random.sample(range(size), k))
        child = [None] * size
        child_genes_from_p1 = set()

        # Copia genes das posições escolhidas do primeiro pai
        for pos in positions:
            gene = parent1[pos]
            child[pos] = gene
            child_genes_from_p1.add(gene)

        # Preenche o restante na ordem do segundo pai
        fill = [gene for gene in parent2 if gene not in child_genes_from_p1]
        idx = 0
        for i in range(size):
            if child[i] is None:
                if idx < len(fill):
                    child[i] = fill[idx]
                    idx += 1
                else:
                    # Isso pode acontecer se houver duplicatas ou problemas nos pais
                    # print(f"Erro Position Based Crossover: Faltando elementos para preencher.")
                    # Tentativa de preencher com o que falta de parent1
                    remaining_p1 = [
                        g for g in parent1 if g not in child_genes_from_p1 and g not in filter(None, child)]
                    if remaining_p1:
                        child[i] = remaining_p1.pop(0)
                    else:
                        # Fallback final: talvez retornar um dos pais
                        return parent1

        # Verificação final (opcional)
        if None in child or len(set(child)) != size:
            # print(f"Warning: Position Based Crossover gerou filho inválido: {child}")
            return parent1  # Retorna fallback

        if self.local_search_strategy:
            if hasattr(self.local_search_strategy, 'local_search'):
                return self.local_search_strategy.local_search(child)
            else:
                # print("Aviso: local_search_strategy inválida ou ausente em PositionBasedCrossover.")
                pass
        return child


class DisjunctiveCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    # A assinatura precisa incluir machine_ops_builder e graph_builder
    # Adicionamos **kwargs para flexibilidade, mas idealmente deveriam ser explícitos
    def crossover(self, parent1, parent2, **kwargs):
        # Extrai os argumentos necessários de kwargs
        machine_ops_builder: Optional[Callable[[Any], dict[Any, list]]] = kwargs.get(
            'machine_ops_builder')
        graph_builder: Optional[Callable[[Any], Any]
                                ] = kwargs.get('graph_builder')
        use_dsu = kwargs.get('use_dsu', False)
        dsu = kwargs.get('dsu')  # Pode ser None

        if not machine_ops_builder or not graph_builder:
            raise ValueError(
                "DisjunctiveCrossover requer machine_ops_builder e graph_builder.")

        # Obtém machine_ops para ambos os pais
        # É crucial que machine_ops_builder retorne uma cópia ou não modifique estado compartilhado
        try:
            ops_dict_p1 = machine_ops_builder(parent1)
            ops_dict_p2 = machine_ops_builder(parent2)
        except Exception as e:
            # print(f"Erro ao obter machine_ops em DisjunctiveCrossover: {e}")
            return parent1  # Fallback

        # Verifica se as máquinas são consistentes entre os pais
        if ops_dict_p1.keys() != ops_dict_p2.keys():
            # print("Erro: Conjunto de máquinas inconsistente entre pais em DisjunctiveCrossover.")
            return parent1  # Fallback

        child_machine_ops = {}
        all_machines = sorted(ops_dict_p1.keys())  # Garante ordem consistente

        for m in all_machines:
            ops1 = ops_dict_p1.get(m, [])
            ops2 = ops_dict_p2.get(m, [])

            # Verifica se as listas de operações para a máquina m são válidas
            if len(ops1) != len(ops2):
                # print(f"Erro: Tamanhos de ops inconsistentes para máquina {m} entre pais.")
                return parent1  # Fallback
            if not ops1:  # Se a lista de operações estiver vazia, pula
                child_machine_ops[m] = []
                continue

            size = len(ops1)
            # Order Crossover (OX1) dentro de cada máquina
            if size < 2:  # Crossover não faz sentido com menos de 2 elementos
                child_ops = ops1[:]  # Simplesmente copia
            else:
                a, b = sorted(random.sample(range(size), 2))
                child_ops = [None]*size
                # Copia segmento de ops1
                child_ops[a:b] = ops1[a:b]
                segment_genes = set(child_ops[a:b])
                # Preenche o restante com genes de ops2 na ordem, pulando os já presentes
                fill = [op for op in ops2 if op not in segment_genes]
                idx = 0
                for i in list(range(b, size)) + list(range(a)):  # Preenche fora do segmento
                    if child_ops[i] is None:  # Segurança, embora não devesse ser necessário aqui
                        if idx < len(fill):
                            child_ops[i] = fill[idx]
                            idx += 1
                        else:
                            # print(f"Erro no preenchimento OX para máquina {m}.")
                            return parent1  # Fallback

            child_machine_ops[m] = child_ops

        # Reconstrói cromossomo a partir das operações das máquinas ordenadas
        new_chrom = []
        for m in all_machines:
            new_chrom.extend(child_machine_ops[m])

        # Validação de Ciclo (Simplificada - Assume que graph_builder lida com isso)
        # A validação DSU original era complexa e potencialmente incorreta.
        # Vamos confiar na verificação de ciclo do graph_builder.
        try:
            # Passamos use_dsu=False aqui para a verificação final dirigida.
            # A lógica DSU original parecia tentar uma verificação não-dirigida prematura.
            graph = graph_builder(new_chrom, use_dsu=False)
            if graph.has_cycle():
                # print("DisjunctiveCrossover rejeitado: ciclo detectado.")
                return parent1  # Rejeita crossover se criar ciclo
        except Exception as e:
            # print(f"Erro durante a construção/verificação do grafo em DisjunctiveCrossover: {e}")
            return parent1  # Fallback

        # Aplica busca local se configurado
        if self.local_search_strategy:
            if hasattr(self.local_search_strategy, 'local_search'):
                # Passa argumentos extras se a busca local precisar deles (improvável aqui)
                return self.local_search_strategy.local_search(new_chrom)
            else:
                # print("Aviso: local_search_strategy inválida ou ausente em DisjunctiveCrossover.")
                pass

        return new_chrom
