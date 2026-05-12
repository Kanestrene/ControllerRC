# ML-CBF no Pack5

Esta pasta tem uma adaptacao do exemplo ML-CBF para o controlador atual em
`v,w`. O QP continua a ser o filtro final de seguranca; a camada ML escolhe
parametros da CBF/QP que foram avaliados antes em simulacao.

## Treinar

```powershell
.\env\Scripts\python.exe Pack5\train_ml_cbf.py
```

Isto cria:

- `Pack5/ml_cbf_model.npz`
- `Pack5/ml_cbf_training_report.csv`

Para um teste rapido:

```powershell
.\env\Scripts\python.exe Pack5\train_ml_cbf.py --max-candidates 3 --episodes-per-candidate 1 --horizon 4
```

Para um treino mais util:

```powershell
.\env\Scripts\python.exe Pack5\train_ml_cbf.py --episodes-per-candidate 3 --horizon 20
```

## Comparar baseline vs ML-CBF

```powershell
.\env\Scripts\python.exe Pack5\simulate_ml_cbf.py
```

Isto cria `Pack5/ml_cbf_comparison.pdf` e imprime metricas como progresso,
falhas do QP, erro lateral medio e margens minimas a obstaculos/barreiras.

## Onde entra no controlo

- `qp.cbf_clf_qp_filter(..., ml_cbf_selector=model)` ativa a selecao aprendida.
- Sem `ml_cbf_selector`, o Pack5 usa exatamente os parametros passados na chamada.
- A formulacao continua em `u=[v,w]`, com CBF de primeira ordem e ponto lookahead.
