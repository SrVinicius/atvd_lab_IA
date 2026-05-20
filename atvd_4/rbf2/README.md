# RBF 2 – Aproximação Funcional: Sistema de Injeção Eletrônica de Combustível

**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 20/05/2026

---

## Descrição do Problema

Mapear a quantidade de gasolina `y` a ser injetada em função de três variáveis do motor `{x₁, x₂, x₃}` usando uma Rede de Função de Base Radial (RBF).

**Topologia geral:** 3 entradas → N1 neurônios RBF → 1 saída

| Rede   | N1 (neurônios escondidos) |
|--------|--------------------------|
| Rede 1 | 5                        |
| Rede 2 | 10                       |
| Rede 3 | 15                       |

**Treinamento da camada escondida:** K-Means sobre os 150 padrões (semente fixa)  
**Treinamento da camada de saída:** Regra Delta Generalizada (η = 0,01 · ε = 10⁻⁷)  
**Inicialização dos pesos de saída:** Uniforme(0, 1) com sementes distintas por treinamento

---

## Como Executar

```bash
python3 rbf2.py
```

Gera também o arquivo `eqm_por_epoca.png` com os gráficos de convergência.  
Requer **NumPy** e **Matplotlib** (sem scikit-learn).

---

## Atividade 1 & 2 – Treinamentos e Resultados de EQM

Cada topologia recebeu **3 treinamentos independentes**, diferenciados pela semente de inicialização dos pesos da camada de saída.

| Treinamento | Rede 1 (N1=5) EQM | Épocas | Rede 2 (N1=10) EQM | Épocas | Rede 3 (N1=15) EQM | Épocas |
|-------------|-------------------|--------|--------------------|--------|--------------------|--------|
| T1          | 8.314992e−03      | 119    | 5.021818e−03       | 498    | 4.962031e−03       | 690    |
| T2          | 8.314987e−03      | 139    | 5.021843e−03       | 484    | 4.962048e−03       | 709    |
| T3          | 8.315013e−03      | 141    | 5.021834e−03       | 506    | 4.962052e−03       | 694    |

**Observações:**
- Os três treinamentos de cada topologia convergem para valores de EQM muito próximos, indicando que a camada escondida (K-Means, fixo) domina a representação e pequenas variações nos pesos iniciais de saída não afetam significativamente o resultado final.
- O maior N1 converge para menor EQM de **treinamento**, porém exige mais épocas.

---

## Atividade 3 – Validação no Conjunto de Teste

### Saídas da rede por amostra

| # | d      | N5 T1  | N5 T2  | N5 T3  | N10 T1 | N10 T2 | N10 T3 | N15 T1 | N15 T2 | N15 T3 |
|---|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 01| 0.5965 | 0.6351 | 0.6351 | 0.6351 | 0.6251 | 0.6251 | 0.6251 | 0.5804 | 0.5804 | 0.5804 |
| 02| 0.6790 | 0.7513 | 0.7513 | 0.7513 | 0.6592 | 0.6592 | 0.6592 | 0.6637 | 0.6637 | 0.6637 |
| 03| 0.4662 | 0.5593 | 0.5593 | 0.5593 | 0.5143 | 0.5143 | 0.5143 | 0.5223 | 0.5223 | 0.5223 |
| 04| 0.5012 | 0.5641 | 0.5640 | 0.5640 | 0.5081 | 0.5081 | 0.5081 | 0.5435 | 0.5435 | 0.5434 |
| 05| 0.6810 | 0.6880 | 0.6880 | 0.6880 | 0.6802 | 0.6802 | 0.6802 | 0.6198 | 0.6198 | 0.6198 |
| 06| 0.5643 | 0.5677 | 0.5680 | 0.5679 | 0.5367 | 0.5367 | 0.5367 | 0.5895 | 0.5895 | 0.5895 |
| 07| 0.5875 | 0.5731 | 0.5732 | 0.5732 | 0.5692 | 0.5692 | 0.5692 | 0.5939 | 0.5939 | 0.5939 |
| 08| 0.7853 | 0.8318 | 0.8316 | 0.8317 | 0.7847 | 0.7847 | 0.7847 | 0.6839 | 0.6839 | 0.6839 |
| 09| 0.8506 | 0.8964 | 0.8962 | 0.8962 | 0.8965 | 0.8965 | 0.8965 | 0.8380 | 0.8380 | 0.8380 |
| 10| 0.6165 | 0.5712 | 0.5715 | 0.5714 | 0.6441 | 0.6441 | 0.6441 | 0.7288 | 0.7288 | 0.7288 |
| 11| 0.4957 | 0.4911 | 0.4912 | 0.4912 | 0.4843 | 0.4843 | 0.4843 | 0.5288 | 0.5288 | 0.5288 |
| 12| 0.6625 | 0.6338 | 0.6340 | 0.6340 | 0.6168 | 0.6168 | 0.6168 | 0.6590 | 0.6590 | 0.6590 |
| 13| 0.4402 | 0.4518 | 0.4517 | 0.4517 | 0.4428 | 0.4428 | 0.4428 | 0.4683 | 0.4683 | 0.4683 |
| 14| 0.7663 | 0.6989 | 0.6990 | 0.6990 | 0.7500 | 0.7500 | 0.7500 | 0.6872 | 0.6872 | 0.6872 |
| 15| 0.7893 | 0.8549 | 0.8548 | 0.8548 | 0.8730 | 0.8730 | 0.8730 | 0.7029 | 0.7029 | 0.7029 |

### Erro Relativo Médio (%) e Variância dos Erros (%)

| Rede   | T1 – Erro Rel. Médio | T1 – Variância | T2 – Erro Rel. Médio | T2 – Variância | T3 – Erro Rel. Médio | T3 – Variância |
|--------|----------------------|----------------|----------------------|----------------|----------------------|----------------|
| N1=5   | 6.4917%              | 25.1999%       | 6.4784%              | 25.1814%       | 6.4813%              | 25.1811%       |
| N1=10  | 4.0002%              | 10.2057%       | 4.0000%              | 10.2065%       | 4.0001%              | 10.2062%       |
| N1=15  | 7.1597%              | 24.9163%       | 7.1598%              | 24.9167%       | 7.1598%              | 24.9154%       |

---

## Atividade 4 – Gráficos de EQM por Época

Os gráficos de convergência (EQM × Época) para o melhor treinamento de cada topologia foram salvos em:

```
eqm_por_epoca.png
```

| Topologia | Melhor Treinamento | EQM Final    | Épocas |
|-----------|-------------------|--------------|--------|
| N1=5      | T2                | 8.314987e−03 | 139    |
| N1=10     | T1                | 5.021818e−03 | 498    |
| N1=15     | T1                | 4.962031e−03 | 690    |

---

## Atividade 5 – Melhor Topologia e Configuração

### Comparativo final

| Rede   | EQM Treino (melhor T) | Erro Rel. Médio Teste | Variância Teste |
|--------|-----------------------|-----------------------|-----------------|
| N1=5   | 8.314987e−03          | 6.4784%               | 25.1814%        |
| N1=10  | **5.021818e−03**      | **4.0002%**           | **10.2057%**    |
| N1=15  | 4.962031e−03          | 7.1597%               | 24.9163%        |

### Rede selecionada: **Rede 2 (N1=10), Treinamento T1**

**Justificativa:**

Embora a Rede 3 (N1=15) apresente o menor EQM de treinamento (4.962031e−03 vs. 5.021818e−03 da Rede 2), ela obtém o **pior desempenho no conjunto de teste**, com erro relativo médio de 7.16% e variância de 24.92%. Esse comportamento indica **overfitting**: a rede com mais neurônios ajustou-se demais aos dados de treinamento e perdeu capacidade de generalização.

A Rede 2 (N1=10, T1) apresenta o **melhor equilíbrio entre ajuste e generalização**:
- Erro relativo médio no teste = **4.00%** (menor entre todas as redes)
- Variância dos erros no teste = **10.21%** (menor entre todas as redes)
- Convergência em 498 épocas, razoavelmente eficiente

A Rede 1 (N1=5) é insuficiente: com apenas 5 funções de base, não consegue representar adequadamente a não-linearidade do processo.

**Conclusão:** a topologia N1=10 com treinamento T1 é a mais adequada para este problema de injeção eletrônica de combustível.
