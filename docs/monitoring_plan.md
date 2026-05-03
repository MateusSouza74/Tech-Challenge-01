# Plano de Monitoramento do Modelo

Colocar um modelo de previsão de churn em produção é apenas o primeiro passo. Precisamos monitorar continuamente a saúde da API (Serviço) e a saúde preditiva do modelo (Machine Learning).

## 1. Métricas de Serviço (Operational Monitoring)
Monitoradas no CloudWatch ou Prometheus/Grafana:
- **Latência (P50, P90, P99)**: Quanto tempo a API demora para responder. Alerta se P90 > 500ms.
- **Taxa de Erro HTTP**: Percentual de códigos 4xx (problema no input do cliente) e 5xx (problema no servidor/código).
- **Throughput (RPS)**: Requisições por segundo recebidas.

## 2. Métricas de Machine Learning (ML Monitoring)
Como o ground truth (o cliente de fato cancelou?) demora a chegar, precisamos atuar com proxies e observabilidade dos dados em tempo real.

### Data Drift (Desvio dos Dados de Entrada)
Mudanças drásticas nas características demográficas ou econômicas dos clientes em relação aos dados de treino.
- **Métricas**: Divergência de Kullback-Leibler (KL) e Teste de Kolmogorov-Smirnov (KS).
- **Alvo Principal**: `MonthlyCharges` e proporção de contratos `Month-to-month`. Se passarmos a receber um fluxo massivo de clientes com faturamento zero (erro sistêmico) ou com um novo tipo de contrato dominante, o modelo ficará cego.

### Concept Drift (Desvio de Conceito)
A relação entre as variáveis e o target (Churn) mudou. Exemplo: A introdução de uma nova taxa faz com que clientes antigos e de longo prazo comecem a cancelar repentinamente.
- **Métricas**: Queda de ROC-AUC e F1-Score ao longo de janelas temporais de 30 dias (comparando a predição feita com a resposta real do CRM após o fechamento do mês).

### Prediction Drift
A distribuição das probabilidades cuspidas pelo modelo mudou, mesmo sem o ground truth.
- **Exemplo**: A média de clientes classificados como "Risco de Churn" saltou de 26% (histórico) para 60% em uma semana.

## 3. Alertas e Limiares
| Alarme | Limiar | Severidade | Ação Esperada |
|--------|--------|------------|---------------|
| **API Down (500s)** | > 1% das req/min | Crítica (P1) | Rollback para versão anterior; PagerDuty. |
| **Data Quality Drop** | Schema rejections > 5% | Alta (P2) | Inspecionar a ingestão do CRM/Frontend. |
| **Prediction Drift** | Aumento de +15% de "Churners" previstos/semana | Média (P3) | Notificar time de Ciência de Dados para análise no notebook. |
| **Performance Drop** | ROC-AUC cai de 0.84 para < 0.80 no mês | Alta (P2) | Retreinar o modelo com dados recentes. |

## 4. Playbook de Resposta (Retreinamento Contínuo)
1. **Diagnóstico**: Quando o alerta de *Performance Drop* ou *Data Drift* soar, a equipe deve extrair os dados rejeitados ou degradados.
2. **Retreinamento Sombra (Shadow Deployment)**: Executar o pipeline de treinamento (`make train`) com os dados dos últimos 3 meses usando a infraestrutura do MLflow.
3. **Avaliação Campeão vs Desafiante (Champion/Challenger)**: Comparar o modelo em produção (Campeão) com o recém-treinado (Desafiante). 
4. **Promoção**: Se o desafiante vencer nas métricas sem comprometer outros segmentos (verificado no Model Card), ele é promovido ao S3 Registry e o serviço o puxa na próxima reinicialização.
