# Job-Shop Scheduling Problem (JSSP) com Memetic GA

## Sumário
- [Objetivos](#objetivos)
- [Descrição do Problema](#descrição-do-problema)
- [Visão Geral da Solução](#visão-geral-da-solução)
- [Componentes Implementados](#componentes-implementados)
- [Resumo do Estado Atual](#resumo-do-estado-atual)

---

## Objetivos
- [x] Desenvolver um algoritmo Memetic GA para resolver o problema de otimização multiobjetivo do Job-Shop Scheduling Problem (JSSP).
- [x] Implementar um algoritmo de busca local para melhorar as soluções geradas pelo Memetic GA.
- [ ] Escolher qual modelo DRL utilizar para o problema de JSSP para melhorar as soluções geradas pelo Memetic GA.
- [ ] Implementar o modelo DRL escolhido junto ao Memetic GA.

---

## Descrição do Problema

O **Job-Shop Scheduling Problem (JSSP)** é um problema clássico de otimização combinatória. O objetivo é escalonar um conjunto de *jobs* (tarefas), cada um composto por uma sequência de *operações*, em um conjunto de *máquinas*, respeitando:
1. **Precedência:** Operações de um mesmo job devem ser executadas na ordem especificada.
2. **Capacidade:** Cada máquina processa no máximo uma operação por vez.

O objetivo típico é minimizar critérios como o *makespan* (tempo total para completar todos os jobs) ou o *total tardiness* (atraso total).

- **Representação:**
  - Os jobs são listas de operações, cada uma como `(machine_id, duration)`.
  - A solução (cronograma) é representada pela classe `Schedule` (`src/models/schedule.py`), que permite adicionar operações, ordenar por início, calcular makespan, etc.

---

## Visão Geral da Solução

O projeto utiliza um **Memetic Genetic Algorithm (Memetic GA)**, que combina operadores genéticos tradicionais com busca local intensiva (memética), permitindo explorar globalmente o espaço de busca e refinar soluções localmente.

Principais características:
- Inicialização híbrida (aleatória, heurísticas e solução exata via CP-SAT)
- Diversos operadores de crossover e mutação
- Busca local avançada (VND) após operadores genéticos
- Avaliação multiobjetivo (ex: makespan, diversidade)
- Estruturas robustas de grafo para manipulação de restrições
- Otimização automática de hiperparâmetros (Bayesian Optimization, PBT)
- Validação e promoção de diversidade na população

---

## Componentes Implementados

### 1. Representação do Problema e Soluções
- **Estrutura dos Jobs e Operações:**
  - Jobs como listas de operações `(machine_id, duration)`
- **Classe Schedule:**
  - Adição, ordenação, cálculo de makespan, impressão de cronogramas

### 2. Algoritmo Memetic GA
- **Classe Principal:**
  - `GeneticSolver` (`src/solvers/genetic_solver.py`)
  - Inicialização híbrida da população
  - Seleção: torneio, roleta, ranking, elitismo (`src/ga/selection.py`)
  - Diversos operadores de crossover e mutação
  - Busca local avançada (VND)
  - Avaliação multiobjetivo e diversidade

### 3. Operadores Genéticos
- **Crossover (`src/ga/genetic_operators/crossover.py`):**
  - Order Crossover (OX), PMX, Cycle, Position-Based, Disjunctive
- **Mutação (`src/ga/genetic_operators/mutation.py`):**
  - Standard, Disjunctive, Critical Path Swap, base para Insert/2-Opt

### 4. Busca Local
- **Variable Neighborhood Descent (VND) (`src/local_search/strategies.py`):**
  - Vizinhanças adaptativas, operadores de bloco, operadores de caminho crítico, shaking LNS
  - Avaliação paralela, orquestração UCB1, aprendizado reativo
  - Operadores: swap, inversion, scramble, 2-opt, 3-opt, block move, block swap, critical insert, critical block swap, critical 2-opt, critical LNS, etc.

### 5. Estruturas de Grafo
- **Grafo Disjuntivo (`src/ga/graph/disjunctive_graph.py`):**
  - Precedência de job e máquina, detecção de ciclos, ordenação topológica, makespan, caminho crítico
  - Suporte a DSU para verificação eficiente de ciclos
- **Utilitários (`src/ga/graph/graph_utils.py`):**
  - Verificação de precedência entre operações

### 6. Solvers e Integrações
- **Solver Exato (CP-SAT):**
  - Google OR-Tools CP-SAT (`src/solvers/ortools_cpsat_solver.py`)
- **Otimização Bayesiana:**
  - Ajuste automático de hiperparâmetros (`src/solvers/bayesopt_genetic.py`)
- **Population Based Training (PBT):**
  - Evolução simultânea de múltiplos GAs (`src/solvers/pbt_genetic.py`)

### 7. Outros Utilitários
- **Diversidade de População:**
  - Medidas para promover diversidade (`src/ga/population/diversity.py`)
- **Validação de Cronogramas:**
  - Garante viabilidade das soluções

---

## Resumo do Estado Atual

- [x] **Memetic GA funcional** com operadores genéticos variados, seleção, elitismo e busca local avançada
- [x] **Busca local VND** com múltiplas vizinhanças, shaking LNS e orquestração adaptativa
- [x] **Solver exato CP-SAT** para soluções de referência e inicialização
- [x] **Otimização Bayesiana** e **PBT** para ajuste de hiperparâmetros
- [x] **Estruturas robustas de grafo** para manipulação de caminhos críticos e restrições
- [x] **Validação e diversidade** de soluções
- [ ] **Integração com DRL** (Deep Reinforcement Learning) ainda não implementada