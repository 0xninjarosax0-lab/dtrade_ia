# Pipeline Profissional — Agente de IA para WIN (5m)

Este repositório foi ajustado para um pipeline mais aderente ao plano técnico:

- dados com esquema mínimo `bars_5m`;
- separação explícita entre **features** e **labels**;
- treino supervisionado com split temporal;
- contrato de sinal (`action/confidence/size_multiplier/reason_code`);
- execução de backtest com **backtesting.py**.

## Estrutura

- `data.py`: carrega CSV ou gera mock e normaliza para o esquema mínimo de barras.
- `features.py`: engenharia de features anti-leakage e geração de labels (`label_bin`, `label_tri`).
- `model.py`: treino com `LogisticRegression` + `TimeSeriesSplit` + decisão por probabilidade.
- `signal_engine.py`: contrato de decisão e guardrails de risco intradiário.
- `backtest.py`: adaptador para executar o backtest no `backtesting.py`.
- `main.py`: pipeline ponta a ponta.

## Requisitos

- Python 3.10+

Instalação mínima:

```bash
pip install pandas numpy scikit-learn backtesting
```

## CSV esperado (mínimo)

Se quiser usar dados reais locais:

Colunas obrigatórias:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Exemplo:

```csv
timestamp,open,high,low,close,volume
2024-01-02 09:00:00,129000,129120,128950,129080,350
```
Colunas opcionais (se ausentes, o pipeline preenche):

- `symbol`
- `session_id`
- `is_roll_day`

## Como rodar

### 1) Dados mock

```bash
python main.py
```

### 2) CSV real

```bash
python main.py --csv /caminho/para/win_5m.csv
```
## Parâmetros úteis

```bash

python main.py \
  --horizon-bars 3 \
  --cost-buffer-bps 2.0 \
  --threshold-buy 0.55 \
  --threshold-sell 0.45 \
  --max-trades-day 8 \
  --commission 0.0002
```

## Backtesting.py

O pipeline já executa via `backtesting.py`. Se quiser testar regras diferentes no site/projeto oficial,
reaproveite o dataframe no formato:

- `Open`, `High`, `Low`, `Close`, `Volume`, `signal`
onde `signal` = `1` (long), `-1` (short), `0` (flat).