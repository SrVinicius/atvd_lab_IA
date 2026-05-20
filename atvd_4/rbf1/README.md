# RBF 1 – Classificação de Radiação Nuclear

**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 20/05/2026

---

## Descrição do Problema

Classificar a presença ou ausência de radiação em compostos nucleares a partir de duas variáveis medidas (x₁, x₂).

| Status de Radiação | Saída (y) |
|--------------------|-----------|
| Presença           | +1        |
| Ausência           | −1        |

**Topologia da rede:** 2 entradas → 2 neurônios RBF → 1 saída  
**Treinamento da camada escondida:** K-Means (apenas padrões com d = 1)  
**Treinamento da camada de saída:** Regra Delta Generalizada (η = 0,01 · ε = 10⁻⁷)

---

## Como Executar

```bash
python3 rbf1.py
```

Requer apenas **NumPy** (sem scikit-learn).

---

## Atividade 1 – Treinamento da Camada Escondida (K-Means)

O K-Means foi aplicado **exclusivamente** aos 19 padrões com presença de radiação (d = 1), gerando 2 clusters.

| Cluster | Centro (x₁, x₂)            | Variância  |
|---------|----------------------------|------------|
| 1       | (0.398969,  0.157131)      | 0.038460   |
| 2       | (0.164833,  0.612117)      | 0.029806   |

A variância de cada cluster é calculada como a distância quadrática média de todos os pontos do cluster ao seu respectivo centro:

$$\sigma_j^2 = \frac{1}{n_j} \sum_{x \in C_j} \|x - c_j\|^2$$

---

## Atividade 2 – Treinamento da Camada de Saída (Regra Delta Generalizada)

Após o cálculo das ativações gaussianas da camada escondida para todos os 40 padrões de treinamento, os pesos da camada de saída foram treinados com a regra delta estocástica até convergência.

**Parâmetros:** η = 0,01 · ε = 10⁻⁷ · Convergência: época 355 · MSE final ≈ 2,37 × 10⁻¹

| Peso   | Valor     |
|--------|-----------|
| W₂₁,₀  | −1.002536 |
| W₂₁,₁  |  2.697508 |
| W₂₁,₂  |  2.377562 |

> W₂₁,₀ é o bias; W₂₁,₁ e W₂₁,₂ são os pesos associados às saídas dos neurônios RBF 1 e 2, respectivamente.

---

## Atividade 3 – Pós-processamento (Função Sinal)

A saída contínua da rede é convertida para ±1 pela função:

$$y_{pós} = \begin{cases} +1, & y \geq 0 \\ -1, & y < 0 \end{cases}$$

---

## Atividade 4 – Validação no Conjunto de Teste

| # | x₁     | x₂     | d  | y (rede)  | y_pós | Correto? |
|---|--------|--------|----|-----------|-------|----------|
| 1 | 0.8705 | 0.9329 | −1 | −1.002376 | −1    | ✓        |
| 2 | 0.0388 | 0.2703 | +1 | −0.323085 | −1    | ✗        |
| 3 | 0.8236 | 0.4458 | −1 | −0.913922 | −1    | ✓        |
| 4 | 0.7075 | 0.1502 | +1 | −0.220011 | −1    | ✗        |
| 5 | 0.9587 | 0.8663 | −1 | −1.002449 | −1    | ✓        |
| 6 | 0.6115 | 0.9365 | −1 | −0.987657 | −1    | ✓        |
| 7 | 0.3534 | 0.3646 | +1 |  0.966411 | +1    | ✓        |
| 8 | 0.3268 | 0.2766 | +1 |  1.323104 | +1    | ✓        |
| 9 | 0.6129 | 0.4518 | −1 | −0.468110 | −1    | ✓        |
|10 | 0.9948 | 0.4962 | −1 | −0.996528 | −1    | ✓        |

**Taxa de Acerto: 80,0%** (8 de 10 corretos)

---

## Atividade 5 – Estratégias para Aumentar a Taxa de Acerto

As amostras 2 e 4 (ambas com d = +1 e x₂ pequeno) foram classificadas erroneamente como −1. A rede produziu saídas negativas para esses pontos, sugerindo que os centros dos clusters positivos estão mal posicionados para cobrir casos com baixo x₂ combinado a x₁ também baixo.

Estratégias possíveis para melhorar o desempenho:

1. **Aumentar o número de clusters (k > 2):** Com mais neurônios na camada escondida, a fronteira de decisão pode se tornar mais precisa, capturando sub-regiões distintas dentro da classe positiva.

2. **Usar todos os dados para o K-Means:** Inicializar os centros a partir de todos os padrões (não apenas d = 1) permite uma cobertura melhor do espaço de entrada, reduzindo zonas cegas na classificação.

3. **Aumentar o conjunto de treinamento:** A base atual tem apenas 40 amostras. Mais dados aumentam a cobertura do espaço de entrada e reduzem a variância do estimador.

4. **Ajustar a taxa de aprendizado ou usar decaimento:** Uma taxa adaptativa pode evitar que a convergência precoce fique presa em soluções subótimas.

5. **Normalizar as variâncias pelo método heurístico d_max / √(2k):** Essa técnica distribui as variâncias de forma mais uniforme, melhorando a cobertura das funções de base radial.
