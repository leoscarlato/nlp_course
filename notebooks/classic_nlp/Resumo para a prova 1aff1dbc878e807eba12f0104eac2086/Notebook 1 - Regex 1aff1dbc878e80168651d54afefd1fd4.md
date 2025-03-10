# Notebook 1 - Regex

| **Regex** | **Definição** |
| --- | --- |
| ? | Caractere opcional (Ex.: `aa?bb?cc?` - poderia ser `“abc”`, `“aabc”`, `“abbc”`,…) |
| + | Uma ou mais ocorrências do símbolo anterior (Ex.: `[aA][bB]c+` - poderia ser `“ABcccc”` , `“aBc”` … |
| * | Zero ou mais ocorrências do símbolo anterior, basicamente pode estar ou não (Ex.: `[aA][bB]c*` - poderia ser `“ab”` , `“abc”`, … |
| [A-Z] | É um range dos caracteres permitidos (neste caso por exemplo são todas as letras maiúsculas - poderia ser `[A-Za-z]` para incluir as minúsculas)  |
| [0-9] | É um range dos números permitidos |
| \w | Qualquer caractere que pode formar uma palavra |
| \d | Qualquer digito |