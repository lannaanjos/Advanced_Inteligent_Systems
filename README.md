# Sistemas Inteligentes Avançados

Repositório dedicado para estudo da disciplina. Partes (até o dado momento):
1. Normalização dos dados.
2. Hiperparametrização, treinamento, descrição de Clusters + inferência de nova instância
3. Classificadores

>**Anotações em Seguida**

## Pré-Processamento
Algoritmos de ML só entendem números e a qualidade dos dados define o limite superior do modelo (acurácia máxima teórica possível/melhor desempenho que se pode esperar).

**Dados Escalares**
- Representam quantidades numéricas (idade, preço, altura).
- Subtipos:
    - Variáveis Discretas: variáveis que podem assumir apenas valores específicos e contáveis, geralmente números inteiros (número de filhos).
    - Variáveis Contínuas: variáveis que podem assumir qualquer valor dentro de um intervalo. Podem ser medidas comprecisão teoricamente infinita (altura).

**Dados Categóricos**
- Representam rótulos ou categorias (cores, tipo de animal),
- Subtipos:
    - Variáveis Nominais: rótulos sem ordem ou hierarquia, sendo apenas "nomes" para diferenciar grupos (ex: estado civil que pode ser solteiro ou casado).
    - Variáveis Ordinais: rótulos com ordem
