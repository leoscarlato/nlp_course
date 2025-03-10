# Notebook 2 - Document Frequency

## **Document Frequency**

- Número de **documentos** em que uma palavra aparece
- Não é a mesma coisa que contar quantas vezes uma palavra aparece
    - Se uma palavra aparece em 2 documentos, sua DF deve ser 2, mas se aparecer 2 vezes em 1 documento sua DF é 1

## CountVectorizer - Sklearn

- Cria uma matriz X de dimensões N x V, onde N é o número de documentos da coleção e V é o tamanho do vocabulário
- O elemento x(n,v) pode ser:
    - 1 → caso a palavra v aparecer no documento n
    - 0 → caso a palavra v não aparecer no documento n
- Cria uma propriedade `vocabulary_` contendo um dicionário que mapeia as palavras para seus respectivos índices

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df_filt['description'])

# Does the word 'the' appear in document 3?
#X[3, vectorizer.vocabulary_['the']]

# Continue with your solution
doc_freq_matrix = X.mean(axis=0)
doc_freq = {
    word: doc_freq_matrix[0, vectorizer.vocabulary_[word]]
    for word in vectorizer.vocabulary_
}
```

## Lematização e Stemming

São técnicas de normalização de texto usadas para reduzir palavras à sua forma base:

- **Stemming:**
    - Remove prefixos e sufixos das palavras de forma mecânica
    - Mais rápido e simples, mas menos preciso
    - Exemplo: "correndo" → "corr"
- **Lematização:**
    - Converte palavras para sua forma base (lema) usando análise morfológica
    - Mais preciso, mas computacionalmente mais custoso
    - Considera o contexto e significado da palavra
    - Exemplo: "correndo" → "correr"
- **Aplicação no Contexto:**
    - Útil para reduzir a dimensionalidade do vocabulário
    - Ajuda a melhorar a eficiência do CountVectorizer
    - Agrupa variações da mesma palavra, melhorando o cálculo da frequência dos documentos

## LDA (Latent Dirichlet Allocation)

LDA é um modelo probabilístico generativo usado para descobrir tópicos latentes em uma coleção de documentos.

- **Diferenças na Matriz:**
    - No CountVectorizer: matriz binária N x V (documentos x vocabulário)
    - No LDA: duas matrizes principais são geradas:
        - Matriz documento-tópico (N x K): distribuição de tópicos para cada documento
        - Matriz tópico-palavra (K x V): distribuição de palavras para cada tópico
- **Características do LDA:**
    - Cada documento é uma mistura de tópicos
    - Cada tópico é uma distribuição sobre o vocabulário
    - O número K de tópicos é definido previamente
- **Vantagens:**
    - Permite descoberta de estruturas temáticas latentes
    - Reduz dimensionalidade dos dados
    - Facilita interpretação semântica dos documentos

Exemplo de implementação do LDA usando sklearn:

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Exemplo de documentos
documentos = [
    "machine learning is fascinating and powerful",
    "deep learning networks process data efficiently",
    "neural networks learn patterns automatically",
    "artificial intelligence transforms industries"
]

# Criar o CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(documentos)

# Configurar e treinar o modelo LDA
n_topics = 2
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='batch'
)
lda_output = lda.fit_transform(X)

# Visualizar os tópicos principais
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
    print(f"Tópico {topic_idx + 1}: {', '.join(top_words)}")

# Visualizar a distribuição de tópicos para cada documento
for doc_idx, doc_topics in enumerate(lda_output):
    print(f"\nDocumento {doc_idx + 1}:")
    for topic_idx, topic_prob in enumerate(doc_topics):
        print(f"Tópico {topic_idx + 1}: {topic_prob:.2f}")
```

O código acima gera dois tipos principais de resultados:

- **1. Palavras principais por tópico:**
    - Para cada tópico, mostra as 4 palavras mais relevantes
    - Exemplo: "Tópico 1: learning, networks, neural, deep"
- **2. Distribuição de tópicos por documento:**
    - Para cada documento, mostra a probabilidade dele pertencer a cada tópico
    - Exemplo: "Documento 1: Tópico 1 (0.60), Tópico 2 (0.40)"
    - Quanto maior o número, mais aquele documento está relacionado ao tópico
    

Em outras palavras, o LDA:

- Agrupa palavras similares em tópicos
- Indica quanto cada documento "fala sobre" cada tópico
- Ajuda a entender os principais assuntos presentes nos textos