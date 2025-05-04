## Objetivos
- [x] Desenvolver um NSGA-II para resolver o problema de otimização multiobjetivo do Job-Shop Scheduling Problem (JSSP).
- [ ] Implementar um algoritmo de busca local para melhorar as soluções geradas pelo NSGA-II.
- [ ] Escolher qual modelo DRL utilizar para o problema de JSSP para melhorar as soluções geradas pelo NSGA-II.
- [ ] Implementar o modelo DRL escolhido junto ao NSGA-II.

## Detalhes da Implementação

Esta seção descreve os componentes implementados neste projeto para resolver o Job-Shop Scheduling Problem (JSSP) utilizando o algoritmo NSGA-II.

### Job-Shop Scheduling Problem (JSSP)

**Conceito:**

O JSSP é um problema clássico de otimização combinatória na área de pesquisa operacional e ciência da computação. O objetivo é escalonar um conjunto de *jobs* (tarefas) em um conjunto de *machines* (máquinas), sujeito a restrições. Cada *job* consiste em uma sequência de *operations* (operações), cada uma devendo ser processada em uma máquina específica por um determinado tempo. As restrições principais são:
1.  **Precedência:** As operações de um mesmo *job* devem ser executadas na ordem especificada.
2.  **Capacidade:** Cada máquina pode processar no máximo uma operação por vez.

O objetivo típico é encontrar um cronograma (schedule) que minimize um ou mais critérios, como o *makespan* (tempo total para completar todos os *jobs*) ou o *total tardiness* (atraso total em relação aos prazos de entrega).

**Implementação no Código:**

*   **Representação:** A estrutura de dados para representar o problema e as soluções (cronogramas) pode ser encontrada em `src/models/`. O arquivo `schedule.py` provavelmente contém classes ou funções para manipular os cronogramas, calcular tempos de início e fim das operações, etc. (Necessário verificar o conteúdo exato para detalhes).
*   **Objetivos:** Os objetivos de otimização considerados (ex: makespan, tardiness) são calculados com base no cronograma gerado.

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**Conceito:**

O NSGA-II é um algoritmo genético multiobjetivo popular, projetado para encontrar um conjunto de soluções Pareto-ótimas. Ele funciona mantendo uma população de soluções candidatas e aplicando operadores genéticos (seleção, cruzamento e mutação) ao longo de gerações para evoluir a população em direção à fronteira de Pareto. Seus principais componentes são:
1.  **Non-Dominated Sorting:** Classifica a população em diferentes frentes de não dominância. Soluções na primeira frente são as melhores (não dominadas por nenhuma outra).
2.  **Crowding Distance:** Usada como critério de desempate dentro de uma mesma frente, promovendo a diversidade das soluções na fronteira de Pareto. Soluções em regiões menos povoadas são preferidas.
3.  **Seleção:** Geralmente utiliza seleção por torneio binário, considerando tanto o *rank* de não dominância quanto a *crowding distance*.
4.  **Operadores Genéticos:** Cruzamento e mutação são aplicados para gerar novas soluções (filhos) a partir das soluções selecionadas (pais).

**Implementação no Código:**

*   **Estrutura:** A lógica principal do NSGA-II e seus componentes estão implementados no diretório `src/solvers/ga/`.
*   **Operadores Genéticos (`genetic_operators.py`):** Este arquivo contém as implementações dos operadores de cruzamento (crossover) e mutação adaptados para a representação do JSSP utilizada. Podem existir diferentes tipos de operadores implementados.
*   **Codificação:** A forma como uma solução (cronograma) é codificada como um cromossomo para o algoritmo genético é um aspecto crucial. (Necessário verificar detalhes da codificação usada).
*   **Avaliação:** A função de avaliação calcula os valores dos objetivos (makespan, etc.) para cada indivíduo (solução) na população.
*   **Diversidade (`diversity.py`):** Contém a implementação do cálculo da *crowding distance*.
*   **Grafo Disjuntivo (`disjunctive_graph.py`):** Pode ser usado para modelar as restrições do JSSP e auxiliar no cálculo do makespan ou na validação dos cronogramas. (Necessário verificar o uso específico).

*(Esta seção será atualizada conforme mais detalhes da implementação forem analisados ou novos componentes forem adicionados.)*
