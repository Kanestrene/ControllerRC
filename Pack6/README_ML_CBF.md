# ML-CBF no Pack6

Esta pasta tem uma adaptacao do exemplo ML-CBF para o controlador atual em
`v,w`. O QP continua a ser o filtro final de seguranca; a camada ML aprende
uma fronteira de viabilidade com SVM polinomial e depois usa uma procura FGO
para escolher parametros continuos da CBF/QP.

## Treinar

```powershell
.\env\Scripts\python.exe Pack6\train_ml_cbf.py
```

Isto cria:

- `Pack6/ml_cbf_model.npz`
- `Pack6/ml_cbf_training_report.csv`
- `Pack6/ml_cbf_fgo_params.csv`

Para um teste rapido:

```powershell
.\env\Scripts\python.exe Pack6\train_ml_cbf.py --samples 6 --episodes-per-sample 1 --horizon 8 --feasible-min-progress 0.15
```

Para uma volta completa, usa cerca de 50 s. Foi medido que com `horizon=20`
o baseline faz so cerca de 58% da pista, enquanto `horizon=50` completa a
volta.

Para um treino de volta completa:

```powershell
.\env\Scripts\python.exe Pack6\train_ml_cbf.py --samples 36 --episodes-per-sample 1 --horizon 50
```

Para um treino mais robusto, mas bem mais lento:

```powershell
.\env\Scripts\python.exe Pack6\train_ml_cbf.py --samples 60 --episodes-per-sample 2 --horizon 50
```

## Comparar baseline vs ML-CBF

```powershell
.\env\Scripts\python.exe Pack6\simulate_ml_cbf.py
```

Isto cria `Pack6/ml_cbf_comparison.pdf` e imprime metricas como progresso,
falhas do QP, erro lateral medio e margens minimas a obstaculos/barreiras.

## Onde entra no controlo

- `qp.cbf_clf_qp_filter(..., ml_cbf_selector=model)` ativa a selecao aprendida.
- Sem `ml_cbf_selector`, o Pack6 usa exatamente os parametros passados na chamada.
- A formulacao continua em `u=[v,w]`, com CBF de primeira ordem e ponto lookahead.
- O SVM usa kernel polinomial de grau 7, como no exemplo do paper.
- A FGO procura valores continuos dentro de intervalos, nao apenas candidatos fixos.
- O `eps_clf` tambem e treinado, juntamente com `alpha`, `margin`,
  `lookahead_l`, `Wv`, `Ww` e `p_slack`.
