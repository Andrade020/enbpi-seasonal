"""
garch_robustness.py  —  EXPLORATÓRIO (não entra no manuscrito principal)

Pergunta: GARCH captura heteroscedasticidade sazonal tão bem quanto EnbPI-S?

DGP idêntico ao simulation.py:
  Y_t = 0.6*Y_{t-12} + 0.3*Y_{t-1} + sigma_{s(t)} * eta_t,  eta_t ~ t_3
  sigma_s = 2  para s in {Jan, Feb, Aug, Sep},  sigma_s = 1  caso contrário

Métodos comparados:
  1. GARCH(1,1)-Normal   : Ridge AR(13) p/ média + GARCH c/ inovações Gaussianas
  2. GARCH(1,1)-t        : idem, mas inovações t_nu estimado
  3. Pooled EnbPI        : sem estratificação sazonal
  4. EnbPI-S             : estratificado por mês

Observação prévia: GARCH captura CLUSTERING de volatilidade (efeito ARCH),
não variância sazonal determinística (calendar-driven). A variância do DGP
muda de mês para mês por construção — não por causa de choques recentes.
GARCH tenderá a "atrasar" o padrão sazonal: super-estima variância em
meses de baixa volatilidade que seguem meses de alta, e vice-versa.

Requisito: pip install arch scikit-learn scipy statsmodels
"""

import sys, os, time, argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
except ImportError:
    sys.exit("Instale a biblioteca arch:  pip install arch")

# -- Paths --------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)
os.makedirs(os.path.join(PROJ, 'tables'), exist_ok=True)

# -- Parâmetros (idênticos ao simulation.py) ----------------------------------
S          = 12
SIGMA_HIGH = 2.0
HIGH_SET   = {0, 1, 7, 8}   # 0-indexed: Jan(0),Feb(1),Aug(7),Sep(8)
HIGH_SET_1 = {1, 2, 8, 9}   # 1-indexed (para impressão)
ALPHA      = 0.10
T_TRAIN    = 480             # 40 anos mensais
T_TEST     = 120             # 10 anos de teste
P_LAGS     = 13
B_BOOT     = 50
BURN       = 60
N_REP_DEF  = 30              # reps default; use 200 para resultado final

# -- DGP ----------------------------------------------------------------------
def sigma_t(t0_zero):
    return SIGMA_HIGH if (t0_zero % S) in HIGH_SET else 1.0

def generate(T_total, seed):
    rng = np.random.default_rng(seed)
    n   = T_total + BURN
    y   = np.zeros(n)
    for t in range(max(S, P_LAGS), n):
        y[t] = 0.6*y[t-S] + 0.3*y[t-1] + sigma_t(t)*float(rng.standard_t(3))
    return y[BURN:]

# -- Auxiliares ---------------------------------------------------------------
def make_X(y, P):
    """Matriz de features: P lags para cada observação a partir da posição P."""
    return np.column_stack([y[P-k-1 : len(y)-k-1] for k in range(P)])

def eq(x, p):
    """Quantil empírico no nível p."""
    sx = np.sort(x)
    n  = len(sx)
    return float(sx[max(0, min(n-1, int(np.ceil(n*p)) - 1))])

def ls(buf, alpha, m=200):
    """Line search para beta ótimo (mínima largura com cobertura 1-alpha)."""
    best, bw = 0., np.inf
    for beta in np.linspace(0, alpha, m+1):
        w = eq(buf, 1-alpha+beta) - eq(buf, beta)
        if w < bw:
            bw, best = w, beta
    return best

def make_interval(pred, buf, alpha):
    b = ls(buf, alpha)
    return pred + eq(buf, b), pred + eq(buf, 1-alpha+b)

# -- LOO bootstrap ensemble (Ridge) -------------------------------------------
def fit_loo(X_tr, y_tr, B, seed=0):
    n   = len(y_tr)
    rng = np.random.default_rng(seed)
    models   = []
    loo_sum  = np.zeros(n)
    loo_cnt  = np.zeros(n)

    for _ in range(B):
        idx     = rng.integers(0, n, size=n)
        uniq    = np.unique(idx)
        oob     = np.setdiff1d(np.arange(n), uniq)
        m       = Ridge(alpha=1.0).fit(X_tr[idx], y_tr[idx])
        models.append(m)
        if len(oob):
            loo_sum[oob] += m.predict(X_tr[oob])
            loo_cnt[oob] += 1

    # Observações sem nenhum OOB: usar média de todos os modelos
    for i in np.where(loo_cnt == 0)[0]:
        loo_sum[i] = np.mean([m.predict(X_tr[i:i+1])[0] for m in models])
        loo_cnt[i] = 1

    loo_resid = y_tr - loo_sum / loo_cnt
    return models, loo_resid

def ens_pred(models, x):
    return float(np.mean([m.predict(x.reshape(1, -1))[0] for m in models]))

# -- EnbPI (pooled e estratificado) -------------------------------------------
def run_enbpi(y, alpha=ALPHA, stratified=True, rep_seed=0):
    P    = P_LAGS
    X    = make_X(y, P)
    Y    = y[P:]
    n_tr = T_TRAIN - P

    X_tr, y_tr = X[:n_tr], Y[:n_tr]
    X_te, y_te = X[n_tr:n_tr+T_TEST], Y[n_tr:n_tr+T_TEST]

    models, loo_resid = fit_loo(X_tr, y_tr, B_BOOT, seed=rep_seed)

    tr_months   = np.array([(P + i) % S for i in range(n_tr)])
    test_months = np.array([(T_TRAIN + t) % S for t in range(T_TEST)])

    if stratified:
        buf = {s: list(loo_resid[tr_months == s]) for s in range(S)}
    else:
        buf_pool = list(loo_resid)

    lowers, uppers = np.zeros(T_TEST), np.zeros(T_TEST)
    for t in range(T_TEST):
        s    = test_months[t]
        yhat = ens_pred(models, X_te[t])
        b_use = buf[s] if stratified else buf_pool
        if len(b_use) >= 2:
            lo, hi = make_interval(yhat, b_use, alpha)
        else:
            lo, hi = yhat - 5., yhat + 5.

        lowers[t], uppers[t] = lo, hi

        resid = y_te[t] - yhat
        if stratified:
            buf[s].append(resid); buf[s].pop(0)
        else:
            buf_pool.append(resid); buf_pool.pop(0)

    return lowers, uppers, test_months + 1, y_te   # 1-indexed months

# -- GARCH (Ridge p/ média + GARCH(1,1) p/ variância) -------------------------
def run_garch(y, alpha=ALPHA, dist='normal'):
    """
    Abordagem two-stage:
      1. Ridge AR(P_LAGS) para a média — treinado uma vez, estático no teste.
         (Nota: usar mean model estático é uma simplificação intencional para
          velocidade. Para produção, usaria rolling SARIMA. Essa escolha
          penaliza GARCH na margem, não no foco do teste: a variância.)
      2. GARCH(1,1) nos resíduos de treinamento — rolando 1-step-ahead.

    Parâmetro dist: 'normal' ou 't'
    """
    P    = P_LAGS
    X    = make_X(y, P)
    Y    = y[P:]
    n_tr = T_TRAIN - P

    X_tr, y_tr = X[:n_tr], Y[:n_tr]
    X_te, y_te = X[n_tr:n_tr+T_TEST], Y[n_tr:n_tr+T_TEST]

    test_months = np.array([(T_TRAIN + t) % S + 1 for t in range(T_TEST)])

    # --- Estágio 1: Ridge ---
    ridge       = Ridge(alpha=1.0).fit(X_tr, y_tr)
    train_resid = y_tr - ridge.predict(X_tr)

    # --- Estágio 2: GARCH nos resíduos ---
    am = arch_model(train_resid, mean='Zero', vol='Garch', p=1, q=1, dist=dist)
    try:
        gr = am.fit(disp='off', options={'maxiter': 500})
    except Exception as e:
        return np.full(T_TEST, np.nan), np.full(T_TEST, np.nan), test_months, y_te

    omega = float(gr.params['omega'])
    a1    = float(gr.params['alpha[1]'])
    b1    = float(gr.params['beta[1]'])

    if dist == 't':
        nu    = max(float(gr.params.get('nu', 5.0)), 2.05)
        # arch t: z_t ~ t_nu / sqrt(nu/(nu-2))  =>  E[z_t^2]=1
        # Quantis de epsilon_t = sigma_t * z_t:
        #   q_p(epsilon_t) = sigma_t * t.ppf(p, nu) / sqrt(nu/(nu-2))
        scale = np.sqrt((nu - 2.0) / nu)
        qlo   = float(stats.t.ppf(alpha / 2,       nu)) * scale
        qhi   = float(stats.t.ppf(1.0 - alpha / 2, nu)) * scale
    else:
        qlo = float(stats.norm.ppf(alpha / 2))
        qhi = float(stats.norm.ppf(1.0 - alpha / 2))

    # Estado inicial do GARCH (converte para numpy para evitar problemas de versão)
    h_prev   = float(np.asarray(gr.conditional_volatility)[-1]) ** 2
    eps_prev = float(np.asarray(train_resid)[-1])

    lowers, uppers = np.zeros(T_TEST), np.zeros(T_TEST)
    for t in range(T_TEST):
        # Previsão da variância 1-step-ahead
        h_t  = max(omega + a1 * eps_prev**2 + b1 * h_prev, 1e-10)
        std  = np.sqrt(h_t)

        # Previsão da média (Ridge estático)
        yhat = float(ridge.predict(X_te[t:t+1])[0])

        lowers[t] = yhat + qlo * std
        uppers[t]  = yhat + qhi * std

        # Atualiza estado GARCH com observação real
        eps_prev = y_te[t] - yhat
        h_prev   = h_t

    return lowers, uppers, test_months, y_te

# -- Uma replicação MC ---------------------------------------------------------
def one_rep(seed):
    y = generate(T_TRAIN + T_TEST, seed)

    rows = []

    def record(method, lo, hi, months, y_true):
        covered = (y_true >= lo) & (y_true <= hi)
        width   = hi - lo
        for t in range(T_TEST):
            rows.append(dict(method=method,
                             month=int(months[t]),
                             covered=int(covered[t]),
                             width=float(width[t])))

    lo, hi, mnths, yte = run_enbpi(y, stratified=False, rep_seed=seed)
    record('pooled',    lo, hi, mnths, yte)

    lo, hi, mnths, yte = run_enbpi(y, stratified=True, rep_seed=seed)
    record('enbpis',    lo, hi, mnths, yte)

    lo, hi, mnths, yte = run_garch(y, dist='normal')
    record('garch_norm', lo, hi, mnths, yte)

    lo, hi, mnths, yte = run_garch(y, dist='t')
    record('garch_t',   lo, hi, mnths, yte)

    return pd.DataFrame(rows)

# -- Impressão de resultados ---------------------------------------------------
MONTH_ABB = ['Jan','Feb','Mar','Apr','May','Jun',
             'Jul','Aug','Sep','Oct','Nov','Dec']

def print_results(summary):
    methods = ['garch_norm', 'garch_t', 'pooled', 'enbpis']
    labels  = ['GARCH-Normal', 'GARCH-t', 'Pooled', 'EnbPI-S']

    print(f"\n{'Mês':>6}  {'':2}" +
          "".join(f"{l:>14}" for l in labels) +
          "   (cobertura, nominal = 90%)")
    print("-" * (10 + 14*len(methods)))

    for m in range(1, S+1):
        star = "*" if m in HIGH_SET_1 else " "
        line = f"  {MONTH_ABB[m-1]:<3}{star}  "
        for meth in methods:
            v = summary.loc[(summary.method==meth) & (summary.month==m), 'coverage']
            line += f"{float(v.values[0]):>14.3f}" if len(v) else f"{'---':>14}"
        print(line)

    print("\n  * = meses de alta volatilidade (sigma=2)")
    print()
    print(f"{'Método':<14}  {'Overall':>8}  {'HighVol':>8}  "
          f"{'LowVol':>8}  {'Spread':>8}  {'Width(avg)':>10}")
    print("-" * 62)

    for meth, lab in zip(methods, labels):
        sub = summary[summary.method == meth]
        by_m = sub.set_index('month')['coverage']
        ov   = by_m.mean()
        hv   = by_m[[m for m in HIGH_SET_1 if m in by_m.index]].mean()
        lv   = by_m[[m for m in range(1,13) if m not in HIGH_SET_1 and m in by_m.index]].mean()
        spr  = by_m.max() - by_m.min()
        wd   = sub['width'].mean()
        print(f"  {lab:<12}  {ov:>8.3f}  {hv:>8.3f}  {lv:>8.3f}  {spr:>8.3f}  {wd:>10.3f}")

# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='5 replicações (teste rápido)')
    parser.add_argument('--reps', type=int, default=N_REP_DEF)
    args = parser.parse_args()

    N_REP = 5 if args.quick else args.reps

    print("=" * 60)
    print("GARCH Robustness Test  (EXPLORATORY — não entra no paper)")
    print(f"N_REP={N_REP}, T_TRAIN={T_TRAIN}, T_TEST={T_TEST}, alpha={ALPHA}")
    print("=" * 60)

    all_dfs = []
    t_start = time.time()
    for rep in range(N_REP):
        t0 = time.time()
        try:
            df = one_rep(seed=rep * 31 + 7)
            all_dfs.append(df)
        except Exception as exc:
            print(f"  Rep {rep+1}: FALHOU  ({exc})")
            continue
        dt = time.time() - t0
        if (rep+1) % max(1, N_REP//6) == 0 or rep == 0:
            print(f"  Rep {rep+1}/{N_REP}  ({dt:.1f}s)")

    if not all_dfs:
        print("Todas as replicações falharam.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    summary  = (combined
                .groupby(['method', 'month'])
                .agg(coverage=('covered', 'mean'),
                     width   =('width',   'mean'))
                .reset_index())

    print_results(summary)

    out = os.path.join(PROJ, 'tables', 'garch_robustness.csv')
    summary.to_csv(out, index=False)
    print(f"\nResultados salvos em {out}")
    print(f"Tempo total: {(time.time()-t_start)/60:.1f} min")
