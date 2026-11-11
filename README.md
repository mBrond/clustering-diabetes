# Clustering — Perfis de estilo de vida e saúde (BRFSS 2015)

Trabalho prático de **Algoritmos de Clustering** (UFSM). Aplica clustering não supervisionado aos indicadores de saúde do CDC/BRFSS 2015 e compara o **k-means** clássico (Lloyd) com o **e-means**, algoritmo publicado pela Meta FAIR em 2026.

## Problema

> É possível identificar diferentes perfis de estilo de vida e saúde na população a partir de hábitos como tabagismo, alimentação e atividade física?

## Dataset

**Diabetes Health Indicators Dataset**, extraído do BRFSS 2015 (CDC):
- Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- Arquivo usado: `diabetes_012_health_indicators_BRFSS2015.csv`
- 253.680 registros × 22 colunas, todas numéricas e **sem valores ausentes**
- Artigos que usam o dataset:
  - https://arxiv.org/abs/2503.04137
  - https://ieeexplore.ieee.org/document/10743921

O rótulo `Diabetes_012` (0 = sem diabetes, 1 = pré-diabetes, 2 = diabetes) é **removido antes da clusterização**. Ele pode ser usado depois, só para interpretar os clusters.

### Variáveis usadas (14)

| Grupo | Variáveis |
|---|---|
| Hábitos | `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump` |
| Saúde | `BMI`, `GenHlth`, `PhysHlth`, `MentHlth`, `HighBP`, `HighChol` |
| Sociodemográficas | `Age`, `Income`, `Education` |

## Metodologia

1. **Análise descritiva:** `df.info()`, verificação de NaN e matriz de correlação (heatmap).
2. **Pré-processamento:** seleção das 14 variáveis e padronização com `StandardScaler`.
3. **PCA:** variância explicada acumulada e projeção em 2 componentes para visualização.
4. **Clustering** para k = 2 a 9, avaliado com:
   - Inércia (soma das distâncias² ao centróide)
   - Silhouette (calculado numa amostra de 10.000 pontos, porque o cálculo completo é O(n²))
   - Davies-Bouldin
   - Calinski-Harabasz
5. **Comparação k-means × e-means** com divisão treino/teste (80/20):
   - Inércia no treino e no teste, e tempo de treino
   - Métricas internas
   - ARI (Adjusted Rand Index) entre as partições dos dois algoritmos

### Sobre o e-means

O e-means (Bessa et al., 2026, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)) otimiza o mesmo objetivo do k-means, mas atualiza os centróides em **mini-batches com médias móveis exponenciais**. O tamanho do batch e o passo vêm de leis de escala, então o único hiperparâmetro é `k`.

Neste projeto foi usada a implementação oficial `NumpyEmeans`, de `faiss.contrib.e_means` (FAISS ≥ 1.16).

## Estrutura

```
.
├── clustering.ipynb          # análise descritiva, PCA e k-means
├── clustering_emeans.ipynb   # mesmo pipeline com e-means + comparação com k-means
├── infos.txt                 # links do dataset e definição do problema
└── Clustering - Material/    # material da disciplina (enunciado, apresentação, exemplos)
```

## Como executar

Requisitos: Python 3.10+ (testado em 3.14) e Jupyter.

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn faiss-cpu
jupyter notebook
```

O dataset é baixado automaticamente pelo `kagglehub` na primeira execução.

## Resultados

### Análise exploratória
- As correlações entre as variáveis são em geral fracas (|r| < 0,3). Há dois blocos mais fortes:
  - saúde física: `GenHlth`, `PhysHlth` e `DiffWalk`
  - socioeconômico: `Education` e `Income`
- Os hábitos (`Smoker`, `Fruits`, `Veggies`, `HvyAlcoholConsump`) quase não se correlacionam com as demais variáveis.
- **PCA:** PC1 explica 20,7% da variância e PC1+PC2 explicam 31%. São precisos 10 componentes para chegar a 85%. A projeção 2D mostra uma única nuvem contínua, sem grupos visivelmente separados.

### k-means × e-means (inércia no conjunto de teste)

| k | k-means (`n_init=10`) | e-means | Diferença | ARI |
|---|---|---|---|---|
| 2 | 612.039 | 612.044 | +0,00% | 0,99 |
| 3 | 564.671 | 564.683 | +0,00% | 0,99 |
| 4 | 529.973 | 529.976 | +0,00% | 0,99 |
| 5 | 486.639 | 508.826 | +4,56% | 0,57 |
| 6 | 466.669 | 466.690 | +0,01% | 0,98 |
| 7 | 447.243 | 459.169 | +2,67% | 0,60 |
| 8 | 430.858 | 439.290 | +1,96% | 0,43 |
| 9 | 417.997 | 421.106 | +0,74% | 0,56 |

Tempo de treino (~200 mil × 14):

| Algoritmo | Tempo por k |
|---|---|
| e-means | ~0,5–0,6 s |
| k-means `n_init=1` | ~0,1–0,3 s |
| k-means `n_init=10` | ~0,6–2 s |

### Conclusões
- **Estrutura fraca.** O silhouette fica abaixo de 0,18 para todo k, e o Davies-Bouldin fica acima de 1,9. Os dados não formam grupos naturais bem separados: o clustering divide um contínuo em perfis.
- **Escolha de k.** k = 5 é o candidato mais defensável:
  - a inércia tem um cotovelo em k = 5;
  - o silhouette em k = 5 é o segundo melhor;
  - o Davies-Bouldin cai em k = 5.

  O k = 2 tem silhouette maior, mas separa apenas "mais saudável × menos saudável".
- **e-means.** Em k = 2, 3, 4 e 6, uma única execução chega à mesma solução que o melhor de 10 execuções do k-means (ARI ≥ 0,98). Nos outros valores de k, ele para em mínimos locais 1% a 5% piores, porque usa uma única inicialização.
  - Neste tamanho de dados ele **não é mais rápido** que o k-means do scikit-learn. Os ganhos de 10× relatados no artigo valem para milhões ou bilhões de vetores de alta dimensão.
  - Que dois algoritmos produzam partições diferentes (ARI ≈ 0,5) com inércias parecidas reforça que a estrutura de clusters é fraca.

### Limitações
- 7 das 14 variáveis são binárias. Padronizar e usar distância euclidiana trata 0/1 como contínuo. Alternativas: k-prototypes ou distância de Gower.
- O BRFSS é autodeclarado (questionário telefônico).


## Referências
- Bessa, S. et al. *e-means: outperforming k-means (in one-tenth the epochs)*. arXiv:2611.09501, 2026. https://e-means.github.io
- FAISS — `contrib/e_means`: https://github.com/facebookresearch/faiss/tree/main/contrib/e_means
- CDC — Behavioral Risk Factor Surveillance System (BRFSS) 2015.
