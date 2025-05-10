# Algoritmo Genético para o Job Shop Scheduling Problem

Este módulo implementa um algoritmo genético para resolver o Job Shop Scheduling Problem (JSSP).

## Estrutura de Arquivos

```
/ga
  ├── __init__.py                 # Exporta todos os componentes
  ├── solver.py                   # Implementação principal do solver genético
  ├── fitness.py                  # Funções para avaliação de fitness
  ├── initialization.py           # Inicialização da população
  ├── selection.py                # Métodos de seleção
  ├── operators/                  # Classes base e operadores genéricos
  │   ├── __init__.py
  │   ├── base.py                 # Classes abstratas para os operadores
  │   ├── ucb.py                  # Implementação do algoritmo UCB para seleção adaptativa
  ├── crossover/                  # Operadores de crossover
  │   ├── __init__.py
  │   ├── base.py                 # Classe abstrata CrossoverStrategy
  │   ├── strategies.py           # Implementações específicas de crossover
  ├── mutation/                   # Operadores de mutação
  │   ├── __init__.py
  │   ├── base.py                 # Classe abstrata MutationStrategy
  │   ├── strategies.py           # Implementações específicas de mutação
  ├── local_search/               # Estratégias de busca local
  │   ├── __init__.py
  │   ├── base.py                 # Classe abstrata LocalSearchStrategy
  │   ├── strategies.py           # Implementações específicas de busca local
  ├── graph/                      # Componentes relacionados a grafos
  │   ├── __init__.py
  │   ├── disjunctive_graph.py    # Implementação do grafo disjuntivo
  │   ├── dsu.py                  # Implementação de DSU (Disjoint Set Union)
  ├── population/                 # Componentes relacionados à população
  │   ├── __init__.py
  │   ├── diversity.py            # Funções para medir diversidade da população
```

## Componentes Principais

### Solver

O arquivo `solver.py` contém a implementação principal do algoritmo genético.

### Fitness

O arquivo `fitness.py` contém a classe `FitnessEvaluator` responsável por calcular o fitness (makespan) dos cromossomos.

### Inicialização

O arquivo `initialization.py` contém a classe `PopulationInitializer` responsável por inicializar a população.

### Seleção

O arquivo `selection.py` contém a classe `SelectionOperator` com diferentes métodos de seleção:
- Seleção por torneio
- Seleção por roleta
- Seleção por ranking
- Seleção com elitismo

### Crossover

A pasta `crossover` contém diferentes estratégias de crossover:
- OrderCrossover (OX)
- PMXCrossover (Partially Mapped Crossover)
- CycleCrossover (CX)
- PositionBasedCrossover
- DisjunctiveCrossover (específico para JSSP)

### Mutação

A pasta `mutation` contém diferentes estratégias de mutação:
- StandardMutation
- DisjunctiveMutation (específico para JSSP)

### Busca Local

A pasta `local_search` contém estratégias de busca local para melhorar as soluções:
- VNDLocalSearch (Variable Neighborhood Descent)

### Grafo

A pasta `graph` contém implementações relacionadas a grafos:
- DisjunctiveGraph: Grafo disjuntivo para representar o JSSP
- DSU: Disjoint Set Union para verificação eficiente de ciclos
