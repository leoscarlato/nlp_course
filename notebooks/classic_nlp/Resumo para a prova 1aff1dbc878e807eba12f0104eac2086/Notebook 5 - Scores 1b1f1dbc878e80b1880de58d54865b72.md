# Notebook 5 - Scores

## Recall (Sensibilidade)

O Recall mede a proporção de casos positivos reais que foram corretamente identificados.

```python
Recall = TP / (TP + FN)
```

Onde:

- TP = True Positives (Verdadeiros Positivos)
- FN = False Negatives (Falsos Negativos)

## Precision (Precisão)

A Precision mede a proporção de identificações positivas que foram realmente corretas.

```python
Precision = TP / (TP + FP)
```

Onde:

- TP = True Positives (Verdadeiros Positivos)
- FP = False Positives (Falsos Positivos)

## F1-Score

O F1-Score é a média harmônica entre Precision e Recall, fornecendo um único valor que equilibra ambas as métricas.

```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## Balanced Accuracy Score

O Balanced Accuracy Score é particularmente útil para conjuntos de dados desbalanceados, calculando a média entre a sensibilidade e a especificidade.

```python
Balanced Accuracy = (Sensitivity + Specificity) / 2
Onde:
Sensitivity = Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

Onde:

- TN = True Negatives (Verdadeiros Negativos)
- FP = False Positives (Falsos Positivos)

## Quando usar cada métrica?

- **Recall:** Quando falsos negativos são mais custosos que falsos positivos (ex: diagnóstico de doenças)
- **Precision:** Quando falsos positivos são mais custosos que falsos negativos (ex: spam detection)
- **F1-Score:** Quando você precisa de um equilíbrio entre precision e recall
- **Balanced Accuracy:** Quando suas classes são desbalanceadas e você quer dar igual importância para ambas

Exemplo:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Saida:

precision    recall  f1-score   support

      comedy       0.80      0.63      0.71       874
       drama       0.77      0.89      0.82      1195

    accuracy                           0.78      2069
   macro avg       0.79      0.76      0.76      2069
weighted avg       0.78      0.78      0.77      2069

# Matriz de confusão

O elemento c nas coordenadas (i, j) indica o número de vezes que um item cuja classe verdadeira é i, e a classe predita é j.

```python
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay

c = confusion_matrix(y_test, y_pred)
print(c)
_ = ConfusionMatrixDisplay(c, display_labels=['comedy', 'drama']).plot()
```

Saída:

[[ 549  325]
[ 133 1062]]

![image.png](Notebook%205%20-%20Scores%201b1f1dbc878e80b1880de58d54865b72/image.png)

Para mostrar as proporções de cada classificação, utilizar o parâmetro `normalize=’true’`.