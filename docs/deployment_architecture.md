# Arquitetura de Deploy

Para operacionalizar o modelo de previsão de churn, foi definida a seguinte arquitetura focada no ecossistema AWS, utilizando um balanceamento entre infraestrutura gerenciada (Serverless/PaaS) e flexibilidade de containers.

## 1. Justificativa: Real-Time vs Batch Deploy

Foi definido o uso de um modelo híbrido com foco predominante no **Real-Time (Online Inference)** através da API FastAPI.
- **Motivo para o Real-Time**: As equipes de call center (Retenção) precisam de uma pontuação de churn em tempo real assim que o cliente entra em contato (inbound call) ou quando o perfil do cliente sofre uma alteração brusca. Ter a API permite que o CRM consuma o score dinamicamente.
- **Batch Processing Opcional**: Para ações massivas mensais de e-mail marketing, uma DAG (ex: Apache Airflow) pode bater no endpoint da API ou carregar o objeto do modelo do bucket S3 diretamente para gerar scores em lote para toda a base.

## 2. Desenho Arquitetural Sugerido (AWS)

```mermaid
graph TD
    A[Sistemas de Negócio / CRM] -->|HTTP POST /predict| B(API Gateway / Load Balancer)
    B --> C(Amazon ECS com AWS Fargate)
    
    subgraph Container da API (FastAPI)
        C --> D(Validação de Dados - Pandera/Pydantic)
        D --> E(Carregamento do Pipeline Sklearn & Modelo PyTorch)
    end
    
    E --> F[Amazon S3 - Model Registry]
    C -.->|Métricas e Logs| G(Amazon CloudWatch)
    C -.->|Monitoramento de ML| H(MLflow / Prometheus)
```

## 3. Componentes Principais

- **Docker & Amazon ECS (Elastic Container Service)**: A API desenvolvida com FastAPI será embutida em um container Docker, garantindo portabilidade. O Fargate escalará os containers automaticamente dependendo do volume de predições.
- **Model Registry via Amazon S3**: O MLflow grava os artefatos de treinamento em um bucket S3. O container FastAPI busca o modelo em produção diretamente deste bucket ao inicializar (ou sob demanda).
- **AWS API Gateway / ALB**: Fornece um endpoint HTTPs seguro, lida com rate-limiting e garante autenticação antes de passar o payload ao FastAPI.
- **Observabilidade (Amazon CloudWatch)**: Os logs estruturados configurados no FastAPI são exportados para o CloudWatch. O Middleware de Latência reportará tempos de resposta e falhas HTTP 500 para alarmes (SNS).

## 4. CI/CD e Pipeline

1. **Commit na branch `main`**.
2. **GitHub Actions**: 
   - Dispara os testes unitários (`pytest`), validações de estilo (`ruff`) e esquema.
   - Se os testes passarem, constrói a nova imagem Docker.
   - Publica a imagem no Amazon ECR (Elastic Container Registry).
3. **Deploy Contínuo**:
   - O ECS atualiza os serviços de inferência (Rolling Update) sem causar inatividade.
