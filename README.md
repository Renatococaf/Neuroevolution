# Neuroevolution

Jupyter Notebook explorando **Neuroevolução** — a aplicação de Estratégias Evolutivas para otimizar os parâmetros de redes neurais — usando PyTorch e o ambiente **LunarLander-v3** (Gymnasium).

## Conteúdo

### Rede Neural com Encoding Direto

A política é representada por uma rede neural com 2 camadas ocultas de 32 neurônios:

```
Input(8) → Linear(8, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 4)
```

O **encoding direto** mapeia o genoma do indivíduo diretamente nos pesos da rede (1476 parâmetros no total). A aptidão é a recompensa acumulada em um episódio do LunarLander.

### Algoritmos Aplicados

| Algoritmo | Descrição |
|-----------|-----------|
| $(\mu, \lambda)$ ES | Aproximação de gradiente via população, atualização por média ponderada |
| $(1+\lambda)$ ES | Estratégia simples com seleção do melhor offspring |
| **CMA-ES** | Adapta a matriz de covariância para aprender correlações entre parâmetros |

### Ambiente

- **LunarLander-v3** (Gymnasium / Box2D)
- Espaço de observação: 8 variáveis contínuas
- Espaço de ações: 4 ações discretas

## Exercícios e Respostas

### Exercício 1 — Escalabilidade do CMA-ES com o tamanho da rede

> A rede usada tem 2 camadas de 32 neurônios. Tente alterar isso e observe o impacto no número total de parâmetros para o CMA-ES. Quão grande pode ser a rede que o CMA-ES consegue otimizar?

O CMA-ES mantém uma matriz de covariância de tamanho $O(n^2)$, onde $n$ é o número de parâmetros. Com a rede padrão de 32 neurônios, há 1476 parâmetros e a matriz tem ~2,2 milhões de entradas. Ao aumentar para 64 neurônios por camada, o número de parâmetros salta para ~4996, resultando em uma matriz de ~25 milhões de entradas. Com 128 neurônios, isso passa de 350 mil parâmetros e a matriz torna-se inviável em memória e tempo de computação.

Na prática, o CMA-ES consegue otimizar redes de até ~1000–2000 parâmetros de forma eficiente. Para redes maiores, alternativas como **sep-CMA-ES** (que usa apenas a diagonal da covariância, $O(n)$) ou o **OpenAI ES** são necessárias.

---

### Exercício 2 — Comparação entre $(1+\lambda)$ ES, $(\mu,\lambda)$ ES e CMA-ES no LunarLander

> Compare os três algoritmos no LunarLander. Um deles é consistentemente melhor que os outros em diferentes inicializações?

Sim, o **CMA-ES é consistentemente superior** aos outros dois para este problema. As razões são:

- **$(1+\lambda)$ ES**: Explora o espaço de forma aleatória ao redor do melhor indivíduo, sem usar informação coletiva da população. Converge muito lentamente em 1476 dimensões, sendo o menos adequado para este problema.

- **$(\mu,\lambda)$ ES**: Usa o gradiente aproximado por toda a população para guiar a busca. Melhor que o $(1+\lambda)$, mas usa uma distribuição isotrópica (mesma variância em todas as direções), o que limita sua eficiência em espaços com correlações complexas entre parâmetros.

- **CMA-ES**: Aprende a estrutura do espaço de busca adaptando a matriz de covariância a cada geração, o que permite explorar correlações entre pesos da rede. No experimento do notebook, o CMA-ES alcançou recompensa ~75 em apenas 20 gerações (375 avaliações), desempenho que os outros ES não atingem no mesmo número de avaliações. Essa vantagem se mantém em diferentes seeds de inicialização.

## Dependências

```bash
pip install torch numpy gymnasium[box2d] cma pyvirtualdisplay moviepy
apt-get install xvfb
```

## Como executar

```bash
jupyter notebook neuroevolution/Neuroevolution.ipynb
```

## Estrutura do repositório

```
Neuroevolution/
└── neuroevolution/
    └── Neuroevolution.ipynb
```

## Referências

- Wilson, D. G., Lavinas, Y., Templier, P. — [Evolutionary Algorithms Course](https://d9w.github.io/evolution/)
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial.* arXiv:1604.00772
- Gymnasium: [gymnasium.farama.org](https://gymnasium.farama.org)
