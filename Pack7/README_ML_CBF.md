# ML-CBF p,q no Pack7

Esta pasta adapta o Pack6 para ficar mais perto da formulacao dos PDFs. O QP
continua a ser o filtro final de seguranca em `u=[v,w]`, mas a camada ML passa
a aprender os parametros `p,q` da funcao de classe K:

```text
alpha_1(h) = p_1 sign(h)|h|^q_1
```

Como a CBF usada aqui e de primeira ordem, temos `m=1`, logo o vetor aprendido
e simplesmente `(p_1, q_1)`.

## Treinar

```powershell
.\env\Scripts\python.exe Pack7\train_ml_cbf.py
```

Isto cria:

- `Pack7/ml_cbf_pq_model.npz`
- `Pack7/ml_cbf_training_report.csv`
- `Pack7/ml_cbf_fgo_params.csv`

Para um teste rapido:

```powershell
.\env\Scripts\python.exe Pack7\train_ml_cbf.py --samples 6 --episodes-per-sample 1 --horizon 8 --feasible-min-progress 0.15
```

Para uma volta completa:

```powershell
.\env\Scripts\python.exe Pack7\train_ml_cbf.py --samples 36 --episodes-per-sample 1 --horizon 50
```

## Comparar baseline vs ML-CBF

```powershell
.\env\Scripts\python.exe Pack7\simulate_ml_cbf.py
```

## Correspondencia com o artigo

- No artigo, `p=(p_1,...,p_m)` e `q=(q_1,...,q_m)` parametrizam as funcoes de classe K da HOCBF.
- No Pack7, `m=1`, portanto aprende-se `class_k_p = p_1` e `class_k_q = q_1`.
- A restricao de seguranca implementada e `h_dot + alpha_1(h) >= 0`.
- A SVM usa kernel polinomial de grau 7, `k(y,z)=(c1+c2 y^T z)^7`, com `y=(p,q)`.
- A procura FGO segue a forma da Eq. 5.12: usa gradientes numericos de `D` e `H`,
  resolve um LP em `nu`, e atualiza a dinamica auxiliar dos parametros `(p,q)`.
