# MVP — Agente de IA para WIN (5m)

Este repositório contém a primeira versão funcional (MVP) para pesquisa de um agente de IA de day trade no mini índice WIN (timeframe 5 minutos), sem integração com corretora.

## Estrutura

- `data.py`: carrega CSV real ou gera dados mock realistas de candles 5m.
- `features.py`: engenharia de features técnicas (SMA, RSI, ATR, MACD).
- `model.py`: treino com `LogisticRegression` e validação temporal com `TimeSeriesSplit`.
- `backtest.py`: backtest simples de sinais buy/sell com custo de transação.
- `main.py`: pipeline ponta a ponta (dados -> features -> treino -> backtest -> métricas).

## Requisitos

- Python 3.10+

Instalação:

```bash
pip install pandas numpy scikit-learn
```

## Formato do CSV (opcional)

Se quiser usar dados reais locais:

Colunas obrigatórias (case-insensitive):

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

## Como rodar

### 1) Com dados mock

```bash
python main.py
```

### 2) Com CSV real

```bash
python main.py --csv /caminho/para/win_5m.csv
```

Parâmetros úteis:

```bash
python main.py --splits 6 --cost-bps 1.5 --start 2023-01-02 --end 2024-12-31
```

## Métricas do MVP

- Acurácia em teste temporal (último fold do `TimeSeriesSplit`).
- Retorno cumulativo do backtest.
- Drawdown máximo do backtest.

## Observações

- Este MVP é para pesquisa/validação inicial.
- Não inclui execução em corretora.
- Próximos passos naturais: meta-labeling, walk-forward mais robusto, e risk engine mais detalhado.
