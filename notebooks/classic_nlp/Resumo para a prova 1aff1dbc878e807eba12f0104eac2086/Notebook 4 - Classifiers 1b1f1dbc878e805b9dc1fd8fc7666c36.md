# Notebook 4 - Classifiers

# Bag-Of-Words

O modelo Bag of Words (BoW) é uma técnica de processamento de linguagem natural que transforma texto em representações numéricas. Ele funciona criando um "saco de palavras" onde:

- Cada palavra única do texto se torna uma característica (feature)
- A ordem das palavras é ignorada
- A frequência de cada palavra é contada

Exemplo prático de classificação de livros:

Considere dois textos de sinopses:

> Comédia: "Uma história divertida e engraçada sobre uma família maluca em férias"
Drama: "Uma narrativa profunda e emocionante sobre perda e superação"
> 

O BoW criaria um vocabulário como:

```python
vocabulario = {
    'historia': 1, 'divertida': 2, 'engracada': 3, 'familia': 4, 
    'maluca': 5, 'ferias': 6, 'narrativa': 7, 'profunda': 8,
    'emocionante': 9, 'perda': 10, 'superacao': 11
}

```

Cada texto seria representado como um vetor:

```python
comedia = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # presença das palavras
drama =   [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

```

Um classificador poderia então **aprender padrões nestes vetores para categorizar novos textos em "comédia" ou "drama" baseado nas palavras presentes.**

<aside>
💡 Esta representação, embora simples, é efetiva para muitas tarefas de classificação de texto, especialmente quando combinada com técnicas de normalização e seleção de características.

</aside>

# Distribuição de Bernoulli

A distribuição de Bernoulli é uma variação do modelo Bag-of-Words que considera apenas a presença (1) ou ausência (0) de palavras, ignorando suas frequências.

Principais características:

- Cada palavra é representada por um valor binário (0 ou 1)
- Não importa quantas vezes a palavra aparece no texto
- Útil para textos curtos ou quando a frequência não é relevante

Exemplo usando o caso anterior:

```python
# Mesmo com múltiplas ocorrências, só marcamos presença (1) ou ausência (0)
comedia = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # presença das palavras
drama =   [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Se "divertida" aparecesse 3 vezes, ainda seria representada como 1

```

<aside>
💡 A distribuição de Bernoulli pode ser mais eficiente computacionalmente e menos sensível a outliers, já que não precisamos lidar com contagens.

</aside>

# Regressão Logística

A Regressão Logística é um algoritmo de classificação que estima a probabilidade de um evento pertencer a uma determinada classe.

Principais características:

- Modelo linear que usa uma função sigmóide para mapear valores para probabilidades entre 0 e 1
- Eficiente para problemas de classificação binária
- Fornece probabilidades interpretáveis como output

Exemplo para classificação de texto:

```python
# Usando vetores BoW como input para regressão logística
from sklearn.linear_model import LogisticRegression

# X contém os vetores BoW, y contém as classes (0: drama, 1: comédia)
modelo = LogisticRegression()
modelo.fit(X, y)

# Obtém probabilidades para um novo texto
probabilidades = modelo.predict_proba(novo_texto_vetorizado) # Probabilidades
predict = modelo.predict(novo_texto_vetorizado) # Classificação
```

<aside>
💡 A Regressão Logística é particularmente útil quando precisamos não só da classificação, mas também da probabilidade/confiança da predição.

</aside>

Aqui está um resumo das características da Regressão Logística:

| **Característica** | **Descrição** |
| --- | --- |
| Tipo de classificação | Binária (naturalmente) ou multiclasse (one-vs-rest) |
| Output | Probabilidades entre 0 e 1 |
| Complexidade | Baixa (modelo linear) |
| Interpretabilidade | Alta (coeficientes indicam importância das features) |
| Requisitos de dados | Features numéricas, preferencialmente escaladas |
| Uso em NLP | Muito comum com BoW ou TF-IDF como input |

## Comparação dos Modelos

Aqui está uma análise comparativa detalhada dos três modelos discutidos:

| **Característica** | **Bag-of-Words** | **Distribuição de Bernoulli** | **Regressão Logística** |
| --- | --- | --- | --- |
| Representação | Frequência das palavras | Presença/ausência (binário) | Probabilidades (0-1) |
| Complexidade | Média | Baixa | Baixa |
| Uso de memória | Alto (armazena frequências) | Baixo (apenas binário) | Baixo |
| Melhor cenário | Textos longos com frequências importantes | Textos curtos, documentos binários | Classificação binária com features numéricas |
| Limitações | Ignora ordem das palavras | Perde informação de frequência | Assume relação linear entre features |
| Interpretabilidade | Média | Alta | Alta |
| Tipo de output | Vetores numéricos | Vetores binários | Probabilidades |

<aside>
💡 Cada modelo tem seus pontos fortes específicos:
- BoW é excelente para análise detalhada de frequência de palavras
- Bernoulli é eficiente para classificação simples de textos curtos
- Regressão Logística oferece probabilidades interpretáveis para tomada de decisão

</aside>

# TFIDF

- Term-Frequency-Inverse-Document-Frequency
    - TF: frequência de cada termo em cada documento (dá uma ideia da importância de cada termo em cada documento)
    - DF: importância de um termo para a coleção inteira
        - Um termo com um DF pequeno, tende a ser mais relevante em um documento

## Resumo dos Conceitos

Neste notebook, foram abordados importantes conceitos de classificação de texto e processamento de linguagem natural:

- **Bag-of-Words (BoW):**
    - Técnica que transforma texto em representação numérica
    - Conta a frequência de cada palavra no texto
    - Ignora a ordem das palavras
- **Distribuição de Bernoulli:**
    - Variação do BoW que considera apenas presença (1) ou ausência (0)
    - Mais eficiente computacionalmente
    - Ideal para textos curtos
- **Regressão Logística:**
    - Algoritmo de classificação que estima probabilidades
    - Usa função sigmóide para mapear valores entre 0 e 1
    - Alta interpretabilidade e baixa complexidade
- **TFIDF (Term Frequency-Inverse Document Frequency):**
    - Combina frequência do termo no documento (TF)
    - Considera a relevância do termo na coleção inteira (IDF)
    - Permite identificar termos mais distintivos em documentos

| **Modelo** | **Melhor Cenário de Uso** | **Exemplo de Aplicação** |
| --- | --- | --- |
| Bag-of-Words | - Textos longos
- Quando frequência é importante
- Análise de conteúdo detalhada | - Análise de artigos científicos
- Categorização de documentos longos
- Análise de reviews detalhados |
| Distribuição de Bernoulli | - Textos curtos
- Classificação simples
- Recursos computacionais limitados | - Classificação de tweets
- Análise de mensagens curtas
- Filtro de spam |
| Regressão Logística | - Necessidade de probabilidades
- Classificação binária
- Dados bem estruturados | - Previsão de sentimentos
- Detecção de fraude
- Classificação de clientes |
| TFIDF | - Identificação de termos relevantes
- Análise de coleções de documentos
- Busca por termos distintivos | - Sistemas de busca
- Recomendação de conteúdo
- Análise de relevância de documentos |