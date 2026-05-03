# Previsão de Churn em Telecomunicações

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=mlflow&logoColor=blue)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)
![Ruff](https://img.shields.io/badge/Ruff-Linting-black?style=for-the-badge)

Projeto de entrega final da **Fase 01** da Pós-Graduação em Machine Learning Engineering da **FIAP**, focado no desenvolvimento ponta a ponta de um modelo de Machine Learning para prever o churn (cancelamento) de clientes em uma operadora de telecomunicações. 

O projeto abrange desde a exploração inicial de dados (EDA) até o deploy de uma API utilizando redes neurais (PyTorch) e baselines no Scikit-Learn, com tracking de experimentos via MLflow.

## 📖 Contexto do Negócio

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa de um **modelo preditivo de churn** que classifique os clientes atuais com risco iminente de cancelamento.

O objetivo do projeto é atuar em todo o ciclo de vida dos dados: construir a solução do zero, partindo da modelagem analítica e terminando com o modelo servido via API, aplicando as melhores práticas de Engenharia de Machine Learning.

## 📊 O Dataset (Telco Customer Churn: IBM)

O dataset utilizado é o **Telco customer churn: IBM dataset** (base samples dataset do IBM Cognos Analytics 11.1.3+). Ele traz dados de uma empresa fictícia de telecomunicações que forneceu serviços de telefonia fixa e internet para **7.043 clientes na Califórnia** durante o terceiro trimestre.

Os dados contemplam:
- **Demografia:** Gênero, idade (Senior Citizen), dependentes e parceiros.
- **Localização:** País, Estado, Cidade, CEP e Coordenadas (embora algumas destas possam ser removidas no pré-processamento por falta de generalização).
- **Serviços:** Telefone fixo, múltiplas linhas, tipo de internet, segurança online, suporte técnico, streaming, entre outros.
- **Faturamento e Contrato:** Tipo de contrato, método de pagamento, cobrança mensal e total acumulado.
- **Variável Alvo (Target):** `Churn Label` e `Churn Value`, indicando se o cliente cancelou o serviço no trimestre vigente.

## 📁 Estrutura do Projeto

```text
├── data/               # Conjuntos de dados brutos e processados
├── docs/               # Documentação técnica (Model Card, Monitoramento, Arquitetura)
├── notebooks/          # Notebooks de EDA e desenvolvimento de modelos iniciais
├── models/             # Modelos serializados, scalers e artefatos (outputs do MLflow)
├── src/                # Código fonte principal
│   ├── api/            # API de inferência usando FastAPI
│   ├── data/           # Scripts de ingestão e pré-processamento
│   ├── models/         # Arquiteturas de modelos (PyTorch)
│   └── training/       # Pipeline de treinamento e tracking
├── tests/              # Testes automatizados (pytest, pandera)
├── Makefile            # Comandos úteis de build e execução
├── pyproject.toml      # Configuração de dependências e linting
└── README.md           # Documentação principal do projeto
```

## 🚀 Setup e Instalação

Pré-requisitos: Python 3.10 ou superior. Recomendado uso de ambiente virtual (`venv` ou `conda`).

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/MateusSouza74/Tech-Challenge-01.git
   cd Tech-Challenge-01
   ```

2. **Criar e ativar um ambiente virtual:**
   ```bash
   python -m venv .venv
   
   # Windows (PowerShell)
   .\.venv\Scripts\activate
   
   # Linux/MacOS
   source .venv/bin/activate
   ```

3. **Instalar as dependências:**
   ```bash
   # Opção com Make:
   make install
   
   # Opção com Python/Pip:
   pip install -e ".[dev]"
   ```

## 💻 Execução

É possível executar os comandos do projeto de duas formas: utilizando o utilitário **`make`** (padrão em ambientes Linux/Mac) ou executando os **comandos nativos do Python** (recomendado para ambientes Windows, evitando a instalação de dependências extras).

> **Antes de executar qualquer comando abaixo**, é necessário certificar-se de estar dentro do diretório `Tech-Challenge-01` e com o ambiente virtual ativado:
> ```bash
> cd Tech-Challenge-01
> .\.venv\Scripts\activate   # No Windows
> ```

### 1. Treinar os modelos
Executar o treinamento e registrar os resultados no MLflow local. Este passo gera os artefatos necessários para a API e os testes.
- **Opção com Make:** `make train`
- **Opção com Python:** `python -m src.training.train`

Para visualizar a interface do MLflow e acessar `http://127.0.0.1:5000`:
- **Opção com Make:** `make mlflow`
- **Opção com Python:** `mlflow ui`

### 2. Rodar os testes
Garantir o funcionamento da pipeline, esquemas de dados, modelo e API (26 testes cobrindo 4 categorias: smoke, schema, preprocessing e API).
- **Opção com Make:** `make test`
- **Opção com Python:** `pytest tests/ -v`

### 3. Rodar o linting
Garantir a qualidade do código com `ruff`.
- **Opção com Make:** `make lint`
- **Opção com Python:** `ruff check src/ tests/`

### 4. Rodar a API FastAPI localmente
Iniciar o servidor de inferência.
- **Opção com Make:** `make run`
- **Opção com Python:** `uvicorn src.api.api:app --reload`
> **Acessar a documentação da API em:** `http://127.0.0.1:8000/docs`

## 📚 Documentação Adicional

Documentações detalhadas do projeto na pasta `docs/`:

- [ML Canvas (Business)](docs/ml_canvas.md) - Contexto de negócio, métricas e SLOs.
- [Model Card](docs/model_card.md) - Detalhes da rede neural, limitações, vieses e performance.
- [Arquitetura de Deploy](docs/deployment_architecture.md) - Estratégia de serviço (Real-time vs Batch).
- [Plano de Monitoramento](docs/monitoring_plan.md) - Alertas e mitigação de data drift/concept drift.

## 👥 Autores

| Nome                                | RM       | Função no Projeto                               |
| :---------------------------------- | :------- | :---------------------------------------------- |
| **Mateus de Souza Nascimento**      | RM373134 | Analyst / DevOps / Data Scientist / ML Engineer |
| **Raphael Dyorgenes Vitor**         | RM371314 | Analyst / DevOps / Data Scientist / ML Engineer |
