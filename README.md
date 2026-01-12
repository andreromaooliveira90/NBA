
# Modelo para Otimização na Formação de Times da NBA utilizando Machine Learning

Este projeto utiliza técnicas de **Machine Learning** para analisar dados históricos da NBA e prever o desempenho das equipes em duas frentes: a classificação para os playoffs e a identificação do potencial campeão da liga. Inspirado pela filosofia *Moneyball*, o estudo busca transformar a abundância de dados estatísticos do basquete em inteligência competitiva.

## 🏀 Contexto e Objetivos

O projeto analisa mais de 60 anos de história da NBA, focando no período quantitativo de 1990 a 2022.

Os principais objetivos são:

* **Prever times classificados aos playoffs**.
* **Identificar o potencial campeão** da temporada.
* **Determinar fatores estatísticos relevantes** que ditam o sucesso de uma equipe.

## 📊 Metodologia e Dados

A análise foi estruturada em duas etapas sequenciais, utilizando um dataset dividido em:

* **Treino:** Dados de 1990 a 2015.
* **Teste:** Dados de 2016 a 2022.

### Divisão da Análise:

1. **Modelo 1 (Classificação aos Playoffs):** Baseado em 966 registros, focado no status "Sim/Não" para a vaga na pós-temporada.
2. **Modelo 2 (Identificação do Campeão):** Focado apenas nas 529 equipes que chegaram aos playoffs, buscando prever o campeão final.

---

## 🤖 Algoritmos e Performance

Foram testados diferentes algoritmos para cada fase do projeto, com métricas de avaliação como Acurácia, Precision, Recall e ROC AUC.

### Etapa 1: Previsão de Playoffs

| Modelo | Acurácia | ROC AUC | Recall (Classificados) |
| --- | --- | --- | --- |
| **Random Forest** | 69% | 80% | 98% |
| **LGBMClassifier** | 68% | 81% | 98% |

> 
> **Nota:** Ambos os modelos demonstraram uma capacidade elevada (98% recall) de identificar corretamente as equipes que avançam para os playoffs.
> 
> 

### Etapa 2: Previsão do Campeão

| Modelo | Acurácia | ROC AUC | Precision (Não-Campeões) |
| --- | --- | --- | --- |
| **Gradient Boosting** | 87% | 70% | 95% |
| **LGBMClassifier** | 86% | 67% | 94% |

> **Nota:** Embora a acurácia global seja alta, prever o campeão real é um desafio significativamente maior. O modelo brilha em identificar quem **não** será campeão (95% de precisão).
> 
> 

---

## 📈 Fatores Determinantes para o Sucesso

O projeto identificou variáveis chave que influenciam os resultados:

* **Para os Playoffs:** Eficiência ofensiva (FG%), assistências, rebotes defensivos e segurança de posse (baixos turnovers).
* **Para o Título:** Eficácia em lances livres, agressividade ofensiva, rebotes totais e bloqueios.

## 💡 Conclusões e Implicações

* As estatísticas da temporada regular são preditivas para a classificação, mas o título depende de fatores críticos adicionais, como execução sob pressão e lances livres.
* O trabalho oferece suporte prático para **dirigentes e analistas** na tomada de decisão baseada em dados e na interpretação de padrões de sucesso no esporte profissional.

