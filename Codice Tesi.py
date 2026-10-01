# =============================================================================
#  REPLICA DI GIRMA & PAULSON (1999) SU DATI EIA
#  Versione snella: finestre dell'articolo (5, 10) per il confronto,
#  finestre aggiuntive (3, 15) in un'unica tabella di estensione.
#  PEZZO 1 - DATI
# =============================================================================
import os
import re
import glob
import numpy as np
import pandas as pd
from arch.unitroot import ADF, PhillipsPerron
import statsmodels.api as sm
from arch.unitroot.cointegration import engle_granger, phillips_ouliaris
from scipy import stats
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CARTELLA = os.environ.get("CART_TESI", r"C:\Users\aless\OneDrive\Desktop")
FILE_DATI = os.path.join(CARTELLA, "WTI-HEATING OIL-RBOB GASOLINE (EIA CONTRATTI 1-4, 2005-2023).xlsx")
CARTELLA_OUT = os.path.join(CARTELLA, "Tabelle tesi")
INIZIO = pd.Timestamp("2007-01-01")
FINE = pd.Timestamp("2023-12-31")
GALLONI_PER_BARILE = 42
SPREAD = ["CS", "GCS", "HOCS"]

k = pd.read_excel(FILE_DATI, sheet_name="contratti")
k = k[(k["Timestamp"] >= INIZIO) & (k["Timestamp"] <= FINE)].copy()
k["CL"] = k["SETTLE_WTI"]
k["HU"] = k["SETTLE_RBOB"] * GALLONI_PER_BARILE
k["HO"] = k["SETTLE_HO"] * GALLONI_PER_BARILE
k["CS"] = 2 / 3 * k["HU"] + 1 / 3 * k["HO"] - k["CL"]
k["GCS"] = k["HU"] - k["CL"]
k["HOCS"] = k["HO"] - k["CL"]

consegna_target = (k["Timestamp"].dt.to_period("M") + 2).dt.to_timestamp()
CONT = k[k["CONSEGNA"] == consegna_target].set_index("Timestamp").sort_index()

MATRICE = {s: k.pivot(index="Timestamp", columns="CONSEGNA", values=s) for s in SPREAD}
NEGOZIABILE = k.pivot(index="Timestamp", columns="CONSEGNA", values="NEGOZIABILE").fillna(0).astype(bool)
GIORNI = MATRICE["CS"].index

print(f"Periodo: {GIORNI[0].date()} -> {GIORNI[-1].date()}, {len(GIORNI)} giorni di borsa")
print(f"A) serie continua: {len(CONT)} giorni, un solo contratto per giorno: {CONT.index.is_unique}, "
      f"giorni mancanti: {len(GIORNI.difference(CONT.index))}")
n_contr = MATRICE["CS"].notna().sum(axis=1)
n_neg = NEGOZIABILE.sum(axis=1)
print(f"B) matrice: {MATRICE['CS'].shape[1]} mesi di consegna")
print(f"   contratti osservati per giorno: {n_contr.value_counts().sort_index().to_dict()}")
print(f"   contratti negoziabili per giorno: {n_neg.value_counts().sort_index().to_dict()}")
print("\nSpread della serie continua (USD/barile)")
print(CONT[SPREAD].describe().T[["mean", "std", "min", "max"]].round(2).to_string())

# =============================================================================
#  PEZZO 2 - TABELLA I: RADICE UNITARIA SUI PREZZI (eq. 2 e 2a)
# =============================================================================
RITARDI_GP = {"CL": [4, 6, 8], "HU": [1, 4], "HO": [1, 4]}
NOMI = {"CL": "Crude Oil", "HU": "Gasoline (RBOB)", "HO": "Heating Oil"}
ALFA = 0.05


def stelle(p):
    """Significativita' di un p-value: *** all'1%, ** al 5%, * al 10%."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


print("\n" + "=" * 70)
print("TABELLA I - Unit Root Test for Petroleum Futures (livelli)")
print("=" * 70)
print(f"{'':18s}{'Lag (p)':>8}{'ADF':>12}{'':2s}{'Phillips-Perron':>18}")
for s in ["CL", "HU", "HO"]:
    serie = CONT[s]
    p_bic = ADF(serie, trend="c", method="bic").lags
    ritardi = sorted(set(RITARDI_GP[s] + [p_bic]))
    for j, p in enumerate(ritardi):
        adf = ADF(serie, trend="c", lags=p)
        pp = PhillipsPerron(serie, trend="c", lags=p)
        etichetta = NOMI[s] if j == 0 else ""
        nota = "  <- BIC" if p == p_bic else ""
        print(f"{etichetta:18s}{p:8d}{adf.stat:12.3f}{stelle(adf.pvalue):4s}"
              f"{pp.stat:16.3f}{stelle(pp.pvalue):4s}{nota}")
cv = ADF(CONT["CL"], trend="c", lags=1).critical_values
print(f"Valori critici (costante): 1% {cv['1%']:.2f}   5% {cv['5%']:.2f}   10% {cv['10%']:.2f}")

print("\nTest sulle differenze prime (due radici unitarie), ritardi con BIC")
ordine = {}
for s in ["CL", "HU", "HO"]:
    d = CONT[s].diff().dropna()
    adf_d = ADF(d, trend="c", method="bic")
    pp_d = PhillipsPerron(d, trend="c")
    print(f"{NOMI[s]:18s}{adf_d.lags:8d}{adf_d.stat:12.3f}{stelle(adf_d.pvalue):4s}"
          f"{pp_d.stat:16.3f}{stelle(pp_d.pvalue):4s}")
    livello = ADF(CONT[s], trend="c", method="bic")
    livelli_nonstaz = livello.pvalue > ALFA and PhillipsPerron(CONT[s], trend="c").pvalue > ALFA
    diff_staz = adf_d.pvalue < ALFA and pp_d.pvalue < ALFA
    ordine[s] = "I(1)" if (livelli_nonstaz and diff_staz) else "da discutere"
print("Ordine di integrazione: " + "   ".join(f"{NOMI[s]}: {o}" for s, o in ordine.items()))

# =============================================================================
#  PEZZO 3 - TABELLA II: COINTEGRAZIONE (eq. 3-5, 3a-5a)
# =============================================================================
MODELLI = [
    ("Modello 1", "CL", ["HU", "HO"]),
    ("Modello 2", "CL", ["HU"]),
    ("Modello 3", "CL", ["HO"]),
    ("Modello 1, normalizzato su HU", "HU", ["CL", "HO"]),
    ("Modello 1, normalizzato su HO", "HO", ["CL", "HU"]),
]


def cointegrazione(y, x):
    """Coefficienti OLS, Engle-Granger (BIC e AIC) e Phillips-Ouliaris."""
    ols = sm.OLS(CONT[y], sm.add_constant(CONT[x])).fit()
    eg_bic = engle_granger(CONT[y], CONT[x], trend="c", method="bic")
    eg_aic = engle_granger(CONT[y], CONT[x], trend="c", method="aic")
    po = phillips_ouliaris(CONT[y], CONT[x], trend="c", test_type="Za")
    return ols, eg_bic, eg_aic, po


print("\n" + "=" * 94)
print("TABELLA II - Test for Cointegration in Models 1, 2 and 3")
print("=" * 94)
RIS_COINT = {}
RHO_RESIDUI = {}
for nome, y, x in MODELLI:
    ols, eb, ea, po = cointegrazione(y, x)
    RIS_COINT[nome] = (ols, eb, ea, po)
    e = ols.resid.to_numpy()
    RHO_RESIDUI[nome] = np.polyfit(e[:-1], e[1:], 1)[0]          # coefficiente AR(1) dei residui
    termini = " ".join(f"{ols.params[v]:+.3f}*{v}" for v in x)
    print(f"{nome}: {y} = {ols.params['const']:.2f} {termini}")
    print(f"   ADF(BIC) {eb.stat:.3f}{stelle(eb.pvalue)} [{eb.lags}]   ADF(AIC) {ea.stat:.3f}{stelle(ea.pvalue)} "
          f"[{ea.lags}]   Za {po.stat:.2f}{stelle(po.pvalue)}   R2 = {ols.rsquared:.3f}   "
          f"rho residui = {RHO_RESIDUI[nome]:.4f} (emivita {np.log(0.5) / np.log(RHO_RESIDUI[nome]):.0f} gg)")
print("Valori critici al 5%: ADF 3 var. {:.2f}, 2 var. {:.2f}; Za 3 var. {:.2f}, 2 var. {:.2f}".format(
    RIS_COINT["Modello 1"][1].critical_values[5], RIS_COINT["Modello 2"][1].critical_values[5],
    RIS_COINT["Modello 1"][3].critical_values[5], RIS_COINT["Modello 2"][3].critical_values[5]))
print("Girma & Paulson (1983-94), ADF: -6.564 / -4.682 / -5.049;  Za: -100.98 / -43.16 / -69.72")
a1, a2 = RIS_COINT["Modello 1"][0].params[["HU", "HO"]]
print(f"Modello 1: alfa1 = {a1:.3f}, alfa2 = {a2:.3f}; rapporto {a1 / a2:.2f} (tecnologia 2), "
      f"somma {a1 + a2:.3f} (tecnologia 1)")

# =============================================================================
#  PEZZO 4 - TABELLA III: RADICE UNITARIA SUGLI SPREAD A PESI FISSI
# =============================================================================
NOMI_SPREAD = {"CS": "3:2:1 Crack Spread", "GCS": "1:1:0 GCS", "HOCS": "1:0:1 HOCS"}
CV_EY = {"CS": RIS_COINT["Modello 1"][1].critical_values[5],
         "GCS": RIS_COINT["Modello 2"][1].critical_values[5],
         "HOCS": RIS_COINT["Modello 3"][1].critical_values[5]}
CV_DF = ADF(CONT["CS"], trend="c", lags=1).critical_values["5%"]

print("\n" + "=" * 100)
print("TABELLA III - Results of Unit Root Test for Crack Spreads")
print("=" * 100)
EMIVITA, PHI = {}, {}
for s in SPREAD:
    x = CONT[s]
    ab = ADF(x, trend="c", method="bic")
    aa = ADF(x, trend="c", method="aic")
    pp = PhillipsPerron(x, trend="c")
    staz_df = ab.stat < CV_DF and pp.stat < CV_DF
    staz_ey = ab.stat < CV_EY[s] and pp.stat < CV_EY[s]
    ar1 = sm.OLS(x.values[1:], sm.add_constant(x.values[:-1])).fit()
    PHI[s] = ar1.params[1]
    EMIVITA[s] = np.log(0.5) / np.log(PHI[s])
    print(f"{NOMI_SPREAD[s]:20s} ADF(BIC) {ab.stat:.3f}{stelle(ab.pvalue)} [{ab.lags}]  ADF(AIC) {aa.stat:.3f}"
          f"{stelle(aa.pvalue)} [{aa.lags}]  PP {pp.stat:.3f}{stelle(pp.pvalue)}  staz.DF {'si' if staz_df else 'no'}  "
          f"staz.EY {'si' if staz_ey else 'no'}  phi {PHI[s]:.4f}  emivita {EMIVITA[s]:.0f} gg")
print(f"Valori critici al 5%: Dickey-Fuller {CV_DF:.2f}; Engle-Yoo {CV_EY['CS']:.2f} (3 var.), {CV_EY['GCS']:.2f} (2 var.)")
print("Girma & Paulson (1983-94), ADF: -5.515 / -4.705 / -5.376;  PP: -5.700 / -4.638 / -5.818")

# --- controllo descrittivo: crack 5:3:2 (non negoziato) ---
k["CS532"] = 3 / 5 * k["HU"] + 2 / 5 * k["HO"] - k["CL"]
CONT["CS532"] = 3 / 5 * CONT["HU"] + 2 / 5 * CONT["HO"] - CONT["CL"]
assert np.allclose(CONT["CS532"] - CONT["CS"], (CONT["HO"] - CONT["HU"]) / 15)
M532 = k.pivot(index="Timestamp", columns="CONSEGNA", values="CS532")
scost = lambda M: (M.sub(M.mean(axis=1), axis=0)).to_numpy().ravel()
a, b = scost(MATRICE["CS"]), scost(M532.reindex_like(MATRICE["CS"]))
ok = ~np.isnan(a) & ~np.isnan(b)
DIFF_532 = CONT["CS532"] - CONT["CS"]
CORR_532 = (CONT["CS"].corr(CONT["CS532"]), CONT["CS"].diff().corr(CONT["CS532"].diff()),
            np.corrcoef(a[ok], b[ok])[0, 1])
print(f"\nControllo 5:3:2: differenza (HO-HU)/15 media {DIFF_532.mean():.2f}, dev.st {DIFF_532.std():.2f}; "
      f"correlazioni con il 3:2:1: livelli {CORR_532[0]:.3f}, variazioni {CORR_532[1]:.3f}, "
      f"scostamenti fra contratti {CORR_532[2]:.3f}")

# =============================================================================
#  PEZZO 5 - MEDIE MOBILI (eq. 6, 6a, 9, 9a)
# =============================================================================
FINESTRE_ART = (5, 10)          # finestre dell'articolo: confronto con Girma e Paulson
FINESTRE_EXT = (3, 15)          # finestre aggiuntive: estensione
FINESTRE = FINESTRE_ART + FINESTRE_EXT


def media_trasversale(M, n):
    """eq. (6) e (6a) sulla matrice M (giorni x contratti)."""
    somma = M.sum(axis=1, min_count=1).fillna(0).rolling(n, min_periods=n).sum()
    quadrati = (M ** 2).sum(axis=1, min_count=1).fillna(0).rolling(n, min_periods=n).sum()
    conteggio = M.notna().sum(axis=1).rolling(n, min_periods=n).sum()
    Xn = somma / conteggio
    Sn = np.sqrt(quadrati / conteggio - Xn ** 2)
    return Xn, Sn


MA = {}
for s in SPREAD:
    for n in FINESTRE:
        Xn, Sn = media_trasversale(MATRICE[s], n)
        MA[(s, n)] = {"Xn": Xn, "Sn": Sn,
                      "Xkn": MATRICE[s].rolling(n, min_periods=n).mean(),        # eq. (9)
                      "Skn": MATRICE[s].rolling(n, min_periods=n).std(ddof=1)}   # eq. (9a)

# verifica dell'eq. (6)-(6a) scritta come nell'articolo, su alcune date
for t in [GIORNI[100], GIORNI[2000], GIORNI[-1]]:
    for n in FINESTRE:
        pos = GIORNI.get_loc(t)
        valori = MATRICE["CS"].iloc[pos - n + 1:pos + 1].to_numpy().ravel()
        valori = valori[~np.isnan(valori)]
        xbar = valori.sum() / len(valori)
        sn = np.sqrt(((valori - xbar) ** 2).sum() / len(valori))
        assert np.isclose(xbar, MA[("CS", n)]["Xn"][t]) and np.isclose(sn, MA[("CS", n)]["Sn"][t])

print("\n" + "=" * 78)
print("PEZZO 5 - MEDIE MOBILI (verifica dell'eq. 6-6a superata)")
print("=" * 78)
print(f"{'spread':20s}{'n':>3}{'media Xn':>11}{'media Sn':>11}{'mediana Sn':>12}{'Sn 20/04/2020':>15}{'gg con Xkn':>13}")
t_2020 = pd.Timestamp("2020-04-20")
QUOTA_XKN = {}
for s in SPREAD:
    for n in FINESTRE:
        m = MA[(s, n)]
        QUOTA_XKN[(s, n)] = m["Xkn"].notna().sum().sum() / MATRICE[s].notna().sum().sum()
        print(f"{NOMI_SPREAD[s]:20s}{n:3d}{m['Xn'].mean():11.2f}{m['Sn'].mean():11.2f}"
              f"{m['Sn'].median():12.2f}{m['Sn'][t_2020]:15.2f}{QUOTA_XKN[(s, n)]:13.1%}")

distanza = pd.DataFrame(
    {c: (c.to_period("M") - GIORNI.to_period("M")).map(lambda d: d.n) for c in MATRICE["CS"].columns},
    index=GIORNI)
print("\nScostamento medio standardizzato (x_tk - Xn) / Sn per distanza dalla consegna (n = 5)")
for s in SPREAD:
    m = MA[(s, 5)]
    z = MATRICE[s].sub(m["Xn"], axis=0).div(m["Sn"], axis=0)
    riga = ""
    for d in range(1, 5):
        valori = z.where(distanza == d).to_numpy().ravel()
        valori = valori[~np.isnan(valori)]
        riga += f"{valori.mean():12.2f} ({len(valori):4d})" if len(valori) else f"{'-':>18}"
    print(f"{NOMI_SPREAD[s]:20s}" + riga)

# =============================================================================
#  PEZZO 6 - REGOLE DI NEGOZIAZIONE (eq. 7, 7a, 8; pp. 943-945)
# =============================================================================
SOGLIE_C = (1.50, 1.75, 2.00, 2.25, 2.50)
CONTRATTI = MATRICE["CS"].columns
ULTIMO_GIORNO = {j: np.flatnonzero(MATRICE["CS"][c].notna().to_numpy())[-1] for j, c in enumerate(CONTRATTI)}
MESI_ALLA_CONSEGNA = distanza.reindex(columns=CONTRATTI).to_numpy()
NEG = NEGOZIABILE.reindex(columns=CONTRATTI).to_numpy()


def negozia(s, n, c):
    """Restituisce la lista delle operazioni per una strategia."""
    X = MATRICE[s].to_numpy()
    XKN = MA[(s, n)]["Xkn"].to_numpy()
    XN = MA[(s, n)]["Xn"].to_numpy()
    SN = MA[(s, n)]["Sn"].to_numpy()
    T = len(GIORNI)
    aperte = {}
    operazioni = []

    def chiudi(j, t, motivo):
        d, t0 = aperte.pop(j)
        operazioni.append({"spread": s, "n": n, "c": c, "consegna": CONTRATTI[j],
                           "direzione": d, "apertura": GIORNI[t0], "chiusura": GIORNI[t],
                           "x_apertura": X[t0, j], "x_chiusura": X[t, j],
                           "giorni": t - t0, "motivo": motivo,
                           "mesi_alla_consegna": MESI_ALLA_CONSEGNA[t0, j]})

    for t in range(T):
        chiusi_oggi = set()
        for j in list(aperte):                       # 1) uscite
            d, t0 = aperte[j]
            if t == t0:
                continue
            x, xk = X[t, j], XKN[t, j]
            if not np.isnan(x) and not np.isnan(xk) and ((d == 1 and x >= xk) or (d == -1 and x <= xk)):
                chiudi(j, t, "media mobile")
            elif t == ULTIMO_GIORNO[j]:
                chiudi(j, t, "scadenza" if t < T - 1 else "fine campione")
            elif t == T - 1:
                chiudi(j, t, "fine campione")
            else:
                continue
            chiusi_oggi.add(j)
        if t == T - 1 or np.isnan(XN[t]) or np.isnan(SN[t]):   # 2) ingresso
            continue
        cand = np.flatnonzero(NEG[t] & ~np.isnan(X[t]) & ~np.isnan(XKN[t]))
        cand = [j for j in cand if j not in aperte and j not in chiusi_oggi]
        if not cand:
            continue
        valori = X[t, cand]
        bassi = [(v, j) for v, j in zip(valori, cand) if v <= XN[t] - c * SN[t]]
        alti = [(v, j) for v, j in zip(valori, cand) if v >= XN[t] + c * SN[t]]
        scelta = []
        if bassi:
            v, j = min(bassi)
            scelta.append((abs(XN[t] - v), j, 1))
        if alti:
            v, j = max(alti)
            scelta.append((abs(XN[t] - v), j, -1))
        if scelta:
            _, j, d = max(scelta)
            aperte[j] = (d, t)
    return operazioni


OPERAZIONI = pd.DataFrame([op for s in SPREAD for n in FINESTRE for c in SOGLIE_C
                           for op in negozia(s, n, c)])


def sel(s, n, c, direzione=None):
    """Operazioni di una strategia (eventualmente di una sola direzione)."""
    o = OPERAZIONI[(OPERAZIONI.spread == s) & (OPERAZIONI.n == n) & (OPERAZIONI.c == c)]
    return o if direzione is None else o[o.direzione == direzione]


GP_OPER = {("CS", 5): [552, 319, 176, 87, 36], ("CS", 10): [498, 302, 161, 90, 53],
           ("GCS", 5): [453, 215, 103, 33, 13], ("GCS", 10): [407, 215, 102, 48, 20],
           ("HOCS", 5): [743, 304, 93, 29, 6], ("HOCS", 10): [633, 277, 108, 40, 15]}
print("\n" + "=" * 96)
print("PEZZO 6 - OPERAZIONI GENERATE")
print("=" * 96)
for s in SPREAD:
    for n in FINESTRE:
        for i, c in enumerate(SOGLIE_C):
            o = sel(s, n, c)
            gp = GP_OPER.get((s, n))
            print(f"{NOMI_SPREAD[s] if i == 0 else '':20s}{n:3d}{c:6.2f}{len(o):7d}"
                  f"{(o.direzione == 1).sum():8d}{(o.direzione == -1).sum():7d}{o.giorni.mean():13.1f}"
                  f"{(o.motivo == 'media mobile').mean():13.1%}{(o.motivo == 'scadenza').mean():12.1%}"
                  f"{(str(gp[i]) if gp else '-'):>11}")

print("\nDistanza dalla consegna all'apertura (c = 1.50): quota di operazioni")
for n in FINESTRE:
    for s in SPREAD:
        for d, nome in [(1, "lunghe"), (-1, "corte")]:
            sotto = sel(s, n, 1.5, d)
            if len(sotto):
                q = sotto.mesi_alla_consegna.value_counts(normalize=True).sort_index()
                print(f"   n={n:2d} {NOMI_SPREAD[s]:20s}{nome:8s}N={len(sotto):4d}  "
                      + "  ".join(f"+{int(m)} mesi: {v:5.1%}" for m, v in q.items()))

# =============================================================================
#  PEZZO 7 - PROFITTI E TABELLE IV, IVA, IVB (eq. 10; pp. 945-952)
# =============================================================================
BARILI = 3000
COSTO_GP = 100.0
COSTO_TICK = 2 * (3 * 10.0 + 3 * 4.20)
CAPITALE = 75_000
GP_TAB4 = {
    ("CS", 5): ([281, 453, 552, 589, 557], [7.176, 9.783, 8.417, 5.914, 3.533]),
    ("CS", 10): ([372, 475, 644, 641, 474], [7.952, 6.978, 6.057, 4.607, 2.539]),
    ("GCS", 5): ([307, 546, 662, 757, 997], [5.427, 6.926, 6.807, 5.900, 5.789]),
    ("GCS", 10): ([402, 550, 826, 703, 1012], [6.686, 5.628, 6.054, 4.210, 3.558]),
    ("HOCS", 5): ([305, 409, 802, 934, 767], [7.204, 4.861, 4.201, 3.136, 1.649]),
    ("HOCS", 10): ([429, 569, 906, 1485, 1447], [9.812, 6.595, 6.223, 3.917, 2.350]),
}
OPERAZIONI["profitto"] = OPERAZIONI["direzione"] * BARILI * (OPERAZIONI["x_chiusura"] - OPERAZIONI["x_apertura"])


def statistiche(o):
    p = o["profitto"].to_numpy()
    N = len(p)
    if N < 2:
        return None
    media, sd = p.mean(), p.std(ddof=1)
    se = sd / np.sqrt(N)
    t = media / se
    g = o["giorni"].mean()
    return {"totale": p.sum(), "media": media, "sd": sd, "N": N, "giorni": g,
            "vinc": (p > 0).mean(), "se": se, "t": t, "p": stats.t.sf(t, N - 1),
            "t_netto": (media - COSTO_GP) / se, "t_tick": (media - COSTO_TICK) / se,
            "r_lordo": (1 + media / (g * CAPITALE)) ** 252 - 1,
            "r_netto": (1 + (media - COSTO_GP) / (g * CAPITALE)) ** 252 - 1}


def tabella(titolo_tab, filtro):
    print("\n" + "=" * 120)
    print(titolo_tab)
    print("=" * 120)
    print(f"{'':20s}{'c':>5}{'totale':>10}{'medio':>8}{'dev.st':>8}{'oper.':>6}{'giorni':>7}{'% vinc':>8}"
          f"{'t':>7}{'t netto':>9}{'t tick':>8}{'r lordo':>9}{'r netto':>9}{'G&P medio':>10}{'G&P t':>7}")
    for n in FINESTRE:
        print(f"Media mobile a {n} giorni")
        for s in SPREAD:
            for i, c in enumerate(SOGLIE_C):
                r = statistiche(filtro(sel(s, n, c)))
                nome = NOMI_SPREAD[s] if i == 0 else ""
                gp = GP_TAB4.get((s, n))
                rif = f"{gp[0][i]:10d}{gp[1][i]:7.2f}" if (gp and titolo_tab.startswith("TABELLA IV ")) else ""
                if r is None:
                    print(f"{nome:20s}{c:5.2f}{'meno di 2 operazioni':>40}")
                    continue
                print(f"{nome:20s}{c:5.2f}{r['totale']:10,.0f}{r['media']:8.0f}{r['sd']:8.0f}{r['N']:6d}"
                      f"{r['giorni']:7.1f}{r['vinc']:8.1%}{r['t']:7.2f}{r['t_netto']:9.2f}{r['t_tick']:8.2f}"
                      f"{r['r_lordo']:9.1%}{r['r_netto']:9.1%}" + rif)


tabella("TABELLA IV  - Risultati (operazioni lunghe e corte insieme)", lambda o: o)
tabella("TABELLA IVA - Solo operazioni lunghe", lambda o: o[o.direzione == 1])
tabella("TABELLA IVB - Solo operazioni corte", lambda o: o[o.direzione == -1])


def drawdown(s, n, c):
    """Massimo drawdown sui flussi giornalieri e massime posizioni aperte."""
    o = sel(s, n, c)
    flussi = pd.Series(0.0, index=GIORNI)
    aperte_gg = pd.Series(0, index=GIORNI)
    for r in o.itertuples():
        x = MATRICE[s].loc[r.apertura:r.chiusura, r.consegna]
        flussi = flussi.add(r.direzione * BARILI * x.diff().fillna(0), fill_value=0)
        aperte_gg.loc[r.apertura:r.chiusura] += 1
    cum = flussi.cumsum()
    return (cum - cum.cummax()).min(), int(aperte_gg.max())


DD = {(s, n, c): drawdown(s, n, c) for s in SPREAD for n in FINESTRE for c in SOGLIE_C}
print("\nCAPITALE: massimo drawdown giornaliero (lordo, USD) e massime posizioni aperte")
for s in SPREAD:
    for n in FINESTRE:
        print(f"{NOMI_SPREAD[s]:20s}{n:3d}" + "".join(f"{DD[(s, n, c)][0]:11,.0f} ({DD[(s, n, c)][1]})" for c in SOGLIE_C))

# =============================================================================
#  PEZZO 8 - VERIFICHE: ESECUZIONE +1, SOTTOPERIODI, STRATEGIE LUNGHE SIGNIFICATIVE
# =============================================================================
POS_GIORNO = {g: i for i, g in enumerate(GIORNI)}
COL = {c: j for j, c in enumerate(CONTRATTI)}
XMAT = {s: MATRICE[s].to_numpy() for s in SPREAD}


def valore_successivo(s, t, j, limite):
    X = XMAT[s]
    for u in range(t + 1, limite + 1):
        if not np.isnan(X[u, j]):
            return X[u, j]
    return np.nan


righe = []
for r in OPERAZIONI.itertuples():
    t0, t1, j = POS_GIORNO[r.apertura], POS_GIORNO[r.chiusura], COL[r.consegna]
    ultimo = ULTIMO_GIORNO[j]
    x0 = valore_successivo(r.spread, t0, j, ultimo)
    x1 = valore_successivo(r.spread, t1, j, ultimo) if (r.motivo == "media mobile" and t1 < ultimo) else r.x_chiusura
    righe.append(r.direzione * BARILI * (x1 - x0))
OPERAZIONI["profitto_t1"] = righe


def media_t(p):
    p = np.asarray(p, float)
    p = p[~np.isnan(p)]
    if len(p) < 2:
        return np.nan, np.nan, len(p)
    return p.mean(), p.mean() / (p.std(ddof=1) / np.sqrt(len(p))), len(p)


PERIODI = [("2007-2014", "2007-01-01", "2014-12-31"), ("2015-2023", "2015-01-01", "2023-12-31")]


def per_periodo(o, a, b):
    return o[(o.apertura >= a) & (o.apertura <= b)]


print("\n" + "=" * 100)
print("8.1 ESECUZIONE +1 e 8.2 SOTTOPERIODI: profitto medio (t) - tutte / lunghe")
print("=" * 100)
for s in SPREAD:
    for n in FINESTRE:
        for c in SOGLIE_C:
            o = sel(s, n, c)
            riga = ""
            for sotto in [o, o[o.direzione == 1]]:
                m0, t0_, N = media_t(sotto.profitto)
                m1, t1_, _ = media_t(sotto.profitto_t1)
                riga += f"  oggi {m0:7.0f} ({t0_:5.2f})  +1 {m1:7.0f} ({t1_:5.2f})" if N > 1 else f"{'-':>44}"
                for _, a_, b_ in PERIODI:
                    mp, tp, Np = media_t(per_periodo(sotto, a_, b_).profitto)
                    riga += f"  {mp:6.0f} ({tp:5.2f},{Np:4d})" if Np > 1 else f"{'-':>22}"
            print(f"{NOMI_SPREAD[s]:20s}{n:3d}{c:6.2f}" + riga)

# strategie lunghe significative al 5% nelle finestre dell'articolo (le "otto")
significative = [(s, n, c) for s in SPREAD for n in FINESTRE_ART for c in SOGLIE_C
                 if (r := statistiche(sel(s, n, c, 1))) is not None and r["p"] < 0.05]
ANNI = range(INIZIO.year, FINE.year + 1)
codici = [chr(65 + i) for i in range(len(significative))]
print(f"\nStrategie lunghe significative (finestre dell'articolo): {len(significative)}")
for (s, n, c), cod in zip(significative, codici):
    o = sel(s, n, c, 1)
    per_anno = o.groupby(o.apertura.dt.year).profitto.sum()
    migliore = per_anno.idxmax()
    _, t, _ = media_t(o.profitto)
    _, t_senza, _ = media_t(o[o.apertura.dt.year != migliore].profitto)
    _, t_dopo, _ = media_t(o.profitto_t1)
    print(f"   {cod} {NOMI_SPREAD[s]:20s} n={n:2d} c={c:.2f}  totale {o.profitto.sum() / 1000:7.1f}k  "
          f"anno migliore {migliore} ({per_anno.max() / o.profitto.sum():.0%})  t {t:.2f}  "
          f"t senza {t_senza:.2f}  t +1 {t_dopo:.2f}  anni+ {(per_anno > 0).sum()}/{len(per_anno)}")

# =============================================================================
#  PEZZO 9 - DIPENDENZA FRA LE OPERAZIONI: TEST SUI PROFITTI MENSILI
# =============================================================================
MESI = pd.period_range(INIZIO, FINE, freq="M")
RITARDI_NW = 3


def t_mensile(o, colonna):
    if len(o) < 2:
        return np.nan, np.nan
    serie = o.groupby(o["chiusura"].dt.to_period("M"))[colonna].sum().reindex(MESI, fill_value=0.0)
    y = serie.to_numpy()
    if np.allclose(y, 0):
        return np.nan, np.nan
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": RITARDI_NW})
    return y.mean(), fit.tvalues[0]


def t_operazioni(p):
    p = np.asarray(p, float)
    return p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 else np.nan


print("\n" + "=" * 100)
print("9. t per operazione contro t sui profitti mensili (Newey-West) - tutte / lunghe")
print("=" * 100)
for s in SPREAD:
    for n in FINESTRE:
        for c in SOGLIE_C:
            o = sel(s, n, c)
            riga = ""
            for sotto in [o, o[o.direzione == 1]]:
                if len(sotto) < 2:
                    riga += f"{'-':>30}"
                    continue
                _, t_m = t_mensile(sotto, "profitto")
                _, t_m1 = t_mensile(sotto.dropna(subset=["profitto_t1"]), "profitto_t1")
                riga += f"   oper {t_operazioni(sotto.profitto):6.2f} mens {t_m:6.2f} mens+1 {t_m1:6.2f}"
            print(f"{NOMI_SPREAD[s]:20s}{n:3d}{c:6.2f}" + riga)

# =============================================================================
#  PEZZO 10 - TABELLE E GRAFICI PER LA TESI
# =============================================================================
os.makedirs(CARTELLA_OUT, exist_ok=True)
GP_TAB1 = {"CL": {4: (-2.387, -2.397), 6: (-2.645, -2.338), 8: (-2.154, -2.296)},
           "HU": {1: (-2.977, -2.895), 4: (-2.811, -2.933)},
           "HO": {1: (-2.400, -2.320), 4: (-2.419, -2.324)}}
GP_TAB2 = {"Modello 1": (-6.564, -100.975), "Modello 2": (-4.682, -43.158), "Modello 3": (-5.049, -69.721)}
GP_TAB3 = {"CS": (-5.515, -5.700), "GCS": (-4.705, -4.638), "HOCS": (-5.376, -5.818)}
NOMI_BREVI = {"CS": "3:2:1", "GCS": "benzina 1:1", "HOCS": "gasolio 1:1"}
NOMI_SERIE = {"CL": "Greggio", "HU": "Benzina (RBOB)", "HO": "Gasolio"}
ASTERISCHI = "$^{*}$: 10\\%; $^{**}$: 5\\%; $^{***}$: 1\\%."


def num(x, dec=2, mille=False):
    """Numero in formato italiano: virgola decimale, punto per le migliaia."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{abs(x):,.{dec}f}" if mille else f"{abs(x):.{dec}f}"
    s = s.replace(",", "@").replace(".", ",").replace("@", ".")
    return ("$-$" + s) if x < 0 else s


def ast(p):
    a = stelle(p)
    return f"$^{{{a}}}$" if a else ""


def pct(x, dec=1):
    return "--" if x is None or np.isnan(x) else num(100 * x, dec) + r"\%"


def cella(m, t, N=None):
    """'media (t)' oppure 'media (t; N)'."""
    return f"{num(m, 0, True)} ({num(t)})" if N is None else f"{num(m, 0, True)} ({num(t)}; {num(N, 0, True)})"


def tex_escape(s):
    s = str(s)
    if "$" in s or "\\" in s:
        return s
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def pulisci(t):
    t = str(t).replace("$-$", "-").replace("\\hat Z_\\alpha", "Za").replace("\\%", "%")
    for a_, b_ in [("@@", ""), ("{=}", "="), ("\\,", " "), ("\\&", "&"), ("$", ""),
                   ("\\textit", ""), ("\\", ""), ("{", ""), ("}", "")]:
        t = t.replace(a_, b_)
    return t.strip()


TABELLE = {}


def aggiungi(nome, df, didascalia, etichetta, fonte="elaborazione propria su dati EIA.", nota=None,
             dim="footnotesize", sep=6, colspec=None):
    TABELLE[nome] = (df, didascalia, etichetta, fonte, nota, dim, sep, colspec)


def scrivi_tex(nome, df, didascalia, etichetta, fonte, nota, dim="footnotesize", sep=6, colspec=None):
    col = colspec if colspec else "l" + "r" * (df.shape[1] - 1)
    righe_ = [" & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        v = list(r)
        if str(v[0]).startswith("@@"):
            righe_.append(f"\\multicolumn{{{df.shape[1]}}}{{@{{}}l@{{}}}}{{\\textit{{{tex_escape(str(v[0])[2:])}}}}} \\\\")
            continue
        righe_.append(" & ".join(tex_escape(x) for x in v) + r" \\")
    corpo = "\n    ".join(righe_)
    testo = (f"\\begin{{table}}[htbp]\n  \\centering\n  \\caption{{{didascalia}}}\n"
             f"  \\label{{{etichetta}}}\n  \\{dim}\n  \\setlength{{\\tabcolsep}}{{{sep}pt}}\n"
             f"  \\begin{{tabular}}{{@{{}}{col}@{{}}}}\n    \\toprule\n    {corpo}\n    \\bottomrule\n"
             f"  \\end{{tabular}}\n"
             + (f"  \\par\\vspace{{0.2em}}{{\\footnotesize {nota}}}\n" if nota else "")
             + f"  \\fonte{{{fonte}}}\n\\end{{table}}\n")
    with open(os.path.join(CARTELLA_OUT, nome + ".tex"), "w", encoding="utf-8") as f:
        f.write(testo)


# --- TABELLA 1 - struttura dei dati ------------------------------------------
oss = int(MATRICE["CS"].notna().sum().sum())
righe = [("Periodo", f"{GIORNI[0].strftime('%d/%m/%Y')} -- {GIORNI[-1].strftime('%d/%m/%Y')}"),
         ("Giorni di borsa", num(len(GIORNI), 0, True)),
         ("Mesi di consegna (contratti)", num(len(CONTRATTI), 0, True)),
         ("Osservazioni (giorno, contratto)", num(oss, 0, True)),
         ("Giorni con 4 contratti osservati", num(int((n_contr == 4).sum()), 0, True)),
         ("Giorni con 3 contratti osservati", num(int((n_contr == 3).sum()), 0, True)),
         ("Contratti negoziabili per giorno (mediana)", num(float(NEGOZIABILE.sum(axis=1).median()), 0)),
         ("Giorni di borsa negoziabili per contratto (mediana)", num(float(NEGOZIABILE.sum().replace(0, np.nan).median()), 0))]
aggiungi("tabella_01_struttura_dati", pd.DataFrame(righe, columns=["", "Valore"]),
         "Struttura del campione", "tab:struttura", "elaborazione propria su dati EIA (contratti 1-4).")

# --- TABELLA 2 - descrittive dei crack spread --------------------------------
righe = []
for s, etichetta in [("CS", "3:2:1"), ("GCS", "Benzina 1:1"), ("HOCS", "Gasolio 1:1"), ("CS532", "5:3:2")]:
    x = CONT[s]
    ar1 = sm.OLS(x.values[1:], sm.add_constant(x.values[:-1])).fit()
    righe.append([etichetta, num(len(x), 0, True), num(x.mean()), num(x.std()), num(x.min()),
                  num(x.max()), num(x.diff().std()), num(np.log(0.5) / np.log(ar1.params[1]), 0)])
aggiungi("tabella_02_descrittive",
         pd.DataFrame(righe, columns=["Crack spread", "Oss.", "Media", "Dev. st.", "Minimo",
                                      "Massimo", "Dev. st. variazioni", "Emivita (gg)"]),
         "Statistiche descrittive dei crack spread (USD per barile), serie continua a due mesi",
         "tab:descrittive", "elaborazione propria su dati EIA.",
         "L'emivita, in giorni di borsa, è calcolata come $\\ln 0{,}5/\\ln\\hat\\phi$ da un modello AR(1) "
         f"sullo spread. Il 5:3:2 differisce dal 3:2:1 per $(HO_t - HU_t)/15$, con media {num(DIFF_532.mean())} "
         f"e deviazione standard {num(DIFF_532.std())}; correlazione con il 3:2:1: {num(CORR_532[0], 3)} nei "
         f"livelli, {num(CORR_532[1], 3)} nelle variazioni giornaliere, {num(CORR_532[2], 3)} negli scostamenti "
         "fra contratti dello stesso giorno.")

# --- TABELLA 3 - radice unitaria sui prezzi ----------------------------------
righe = []
for s in ["CL", "HU", "HO"]:
    serie = CONT[s]
    p_bic = ADF(serie, trend="c", method="bic").lags
    for j, p in enumerate(sorted(set(list(GP_TAB1[s]) + [p_bic]))):
        adf = ADF(serie, trend="c", lags=p)
        pp = PhillipsPerron(serie, trend="c", lags=p)
        gp = GP_TAB1[s].get(p)
        righe.append([NOMI_SERIE[s] if j == 0 else "", str(p) + (" (BIC)" if p == p_bic else ""),
                      num(adf.stat, 3) + ast(adf.pvalue), num(pp.stat, 3) + ast(pp.pvalue),
                      num(gp[0], 3) if gp else "--", num(gp[1], 3) if gp else "--"])
d = {s: ADF(CONT[s].diff().dropna(), trend="c", method="bic").stat for s in ["CL", "HU", "HO"]}
aggiungi("tabella_03_radice_unitaria",
         pd.DataFrame(righe, columns=["Serie", "Ritardi", "ADF", "Phillips-Perron", "G&P: ADF", "G&P: PP"]),
         "Test di radice unitaria sui prezzi e confronto con Girma e Paulson (1999)", "tab:radiceunitaria",
         "elaborazione propria su dati EIA; Girma e Paulson (1999), Tabella I.",
         "Serie continua a due mesi; regressioni con la sola costante. Valori critici: $-3{,}43$ (1\\%), "
         "$-2{,}86$ (5\\%), $-2{,}57$ (10\\%). Sulle differenze prime i test rifiutano sempre all'1\\% "
         f"(ADF fra {num(min(d.values()), 1)} e {num(max(d.values()), 1)}). Gli asterischi si riferiscono "
         "alle sole stime proprie; nel campione degli autori la benzina rifiuta al 5\\% con un ritardo e al "
         "10\\% (ADF) con quattro. " + ASTERISCHI,
         dim="footnotesize", sep=5)

# --- TABELLA 4 - cointegrazione ----------------------------------------------
righe = []
for nome, y, x in MODELLI:
    ols, eb, ea, po = RIS_COINT[nome]
    coef = f"{num(ols.params['const'])} " + " ".join(
        f"{'+' if ols.params[v] >= 0 else '-'} {num(abs(ols.params[v]), 3)} {v}" for v in x)
    gp = GP_TAB2.get(nome)
    righe.append([nome.replace("Modello ", "Mod. ").replace(", normalizzato su ", " norm. "),
                  f"{y} = {coef}".replace(" + ", "+").replace(" HU", "HU").replace(" HO", "HO").replace(" CL", "CL"),
                  f"{num(eb.stat, 3)}{ast(eb.pvalue)} [{eb.lags}]", f"{num(ea.stat, 3)}{ast(ea.pvalue)} [{ea.lags}]",
                  num(po.stat, 2) + ast(po.pvalue), num(RHO_RESIDUI[nome], 3),
                  num(gp[0], 3) if gp else "--", num(gp[1], 2) if gp else "--"])
aggiungi("tabella_04_cointegrazione",
         pd.DataFrame(righe, columns=["Modello", "Regressione stimata", "ADF (BIC)", "ADF (AIC)",
                                      "$\\hat Z_\\alpha$", "$\\hat\\rho$", "G&P: ADF", "G&P: PO"]),
         "Test di cointegrazione fra greggio e prodotti raffinati", "tab:cointegrazione",
         "elaborazione propria su dati EIA; Girma e Paulson (1999), Tabella II.",
         f"Valori critici al 5\\%: ADF {num(RIS_COINT['Modello 1'][1].critical_values[5])} (3 variabili) e "
         f"{num(RIS_COINT['Modello 2'][1].critical_values[5])} (2 variabili); $\\hat Z_\\alpha$ "
         f"{num(RIS_COINT['Modello 1'][3].critical_values[5])} e {num(RIS_COINT['Modello 2'][3].critical_values[5])}. "
         "Fra parentesi quadre i ritardi scelti. $\\hat\\rho$: coefficiente autoregressivo del primo ordine "
         "dei residui. " + ASTERISCHI,
         dim="scriptsize", sep=2, colspec="@{}llrrrrrr@{}")

# --- TABELLA 5 - stazionarieta' degli spread ---------------------------------
righe = []
for s, etichetta in [("CS", "3:2:1"), ("GCS", "Benzina 1:1"), ("HOCS", "Gasolio 1:1"), ("CS532", "5:3:2")]:
    x = CONT[s]
    ab, aa = ADF(x, trend="c", method="bic"), ADF(x, trend="c", method="aic")
    pp = PhillipsPerron(x, trend="c")
    cv_ey = CV_EY.get(s, CV_EY["CS"])
    gp = GP_TAB3.get(s)
    righe.append([etichetta, f"{num(ab.stat, 3)}{ast(ab.pvalue)} [{ab.lags}]",
                  f"{num(aa.stat, 3)}{ast(aa.pvalue)} [{aa.lags}]", num(pp.stat, 3) + ast(pp.pvalue),
                  "si" if (ab.stat < CV_DF and pp.stat < CV_DF) else "no",
                  "si" if (ab.stat < cv_ey and pp.stat < cv_ey) else "no",
                  num(gp[0], 3) if gp else "--", num(gp[1], 3) if gp else "--"])
aggiungi("tabella_05_stazionarieta_spread",
         pd.DataFrame(righe, columns=["Crack spread", "ADF (BIC)", "ADF (AIC)", "Phillips-Perron",
                                      "Staz. (DF)", "Staz. (EY)", "G&P: ADF", "G&P: PP"]),
         "Test di radice unitaria sui crack spread a pesi fissi", "tab:stazionarieta",
         "elaborazione propria su dati EIA; Girma e Paulson (1999), Tabella III.",
         f"Valori critici al 5\\%: Dickey-Fuller {num(CV_DF)} (corretti, pesi non stimati); Engle-Yoo "
         f"{num(CV_EY['CS'])} (3 variabili) e {num(CV_EY['GCS'])} (2 variabili). ``Staz.'': ADF (BIC) e "
         "Phillips-Perron rifiutano entrambi al 5\\%. " + ASTERISCHI,
         dim="scriptsize", sep=4)

# --- TABELLA 6 - scostamento medio per distanza dalla consegna --------------
righe = []
for s in SPREAD:
    m = MA[(s, 5)]
    z = MATRICE[s].sub(m["Xn"], axis=0).div(m["Sn"], axis=0)
    riga = [NOMI_BREVI[s]]
    for dmesi in (1, 2, 3, 4):
        v = z.where(distanza == dmesi).to_numpy().ravel()
        v = v[~np.isnan(v)]
        riga.append(f"{num(v.mean())} ({num(len(v), 0, True)})" if len(v) else "--")
    righe.append(riga)
aggiungi("tabella_06_scostamenti",
         pd.DataFrame(righe, columns=["Crack spread", "+1 mese", "+2 mesi", "+3 mesi", "+4 mesi"]),
         "Scostamento medio standardizzato dalla media trasversale per distanza dalla consegna ($n = 5$)",
         "tab:scostamenti", "elaborazione propria su dati EIA.",
         "Scostamento $(x_{t,k} - \\bar X_n)/S_n$; fra parentesi il numero di osservazioni. Il contratto a un "
         "mese è osservato solo nei giorni con quattro contratti disponibili; i primi quattro giorni del "
         "campione, privi di finestra completa, sono esclusi.")

# --- TABELLA 6-bis - distanza dalla consegna all'apertura -------------------
righe = []
for s in SPREAD:
    for n in FINESTRE_ART:
        for i, c in enumerate(SOGLIE_C):
            if c > 2.0:
                continue
            riga = [NOMI_BREVI[s] if (n == FINESTRE_ART[0] and i == 0) else "", str(n), num(c)]
            for dz in (1, -1):
                o = sel(s, n, c, dz)
                riga.append(num(len(o), 0, True))
                for mesi in (1, 2, 3, 4):
                    riga.append(pct((o.mesi_alla_consegna == mesi).mean(), 0) if len(o) else "--")
            righe.append(riga)
aggiungi("tabella_06bis_apertura_scadenza",
         pd.DataFrame(righe, columns=["Crack spread", "$n$", "$c$", "Lunghe: N", "+1", "+2", "+3", "+4",
                                      "Corte: N", "+1", "+2", "+3", "+4"]),
         "Distanza dalla consegna dei contratti all'apertura, per finestra e soglia", "tab:aperturascadenza",
         "elaborazione propria su dati EIA.",
         "Quote delle operazioni aperte su contratti a uno, due, tre e quattro mesi dalla consegna; "
         "``N'': numero di operazioni. Soglie $c = 2{,}25$ e $2{,}50$ omesse per il numero esiguo di "
         "operazioni. Con tre contratti negoziabili nella maggior parte dei giorni, una selezione "
         "indipendente dalla scadenza darebbe circa un terzo per scadenza.",
         dim="scriptsize", sep=3)

# --- TABELLA 7 - operazioni generate -----------------------------------------
righe = []
for s in SPREAD:
    for n in FINESTRE_ART:
        for i, c in enumerate(SOGLIE_C):
            o = sel(s, n, c)
            righe.append([NOMI_BREVI[s] if (n == FINESTRE_ART[0] and i == 0) else "", str(n), num(c),
                          num(len(o), 0, True), num(int((o.direzione == 1).sum()), 0, True),
                          num(int((o.direzione == -1).sum()), 0, True), num(o.giorni.mean(), 1),
                          pct((o.motivo == "media mobile").mean()), num(GP_OPER[(s, n)][i], 0, True)])
aggiungi("tabella_07_operazioni",
         pd.DataFrame(righe, columns=["Crack spread", "$n$", "$c$", "Oper.", "Lunghe", "Corte",
                                      "Giorni medi", "Uscite su MA", "G&P: oper."]),
         "Operazioni generate dalla strategia e confronto con Girma e Paulson (1999)", "tab:operazioni",
         "elaborazione propria su dati EIA; Girma e Paulson (1999), Tabella IV.",
         "``Uscite su MA'': quota di operazioni chiuse per incrocio della media mobile del contratto. "
         "Campioni di diversa lunghezza: 17 anni per questa tesi; 10 anni (dicembre 1984--novembre 1994) "
         "per il 3:2:1 e la benzina e 11,7 anni (aprile 1983--novembre 1994) per il gasolio in Girma e "
         "Paulson (1999).",
         dim="scriptsize", sep=4)

# --- TABELLA 7-bis - media e deviazione standard trasversale ----------------
righe = []
for s in SPREAD:
    for n in FINESTRE_ART:
        m = MA[(s, n)]
        righe.append([NOMI_BREVI[s] if n == FINESTRE_ART[0] else "", str(n), num(m["Xn"].mean()),
                      num(m["Sn"].mean()), num(m["Sn"].median()), num(m["Sn"][t_2020]), pct(QUOTA_XKN[(s, n)])])
aggiungi("tabella_07bis_soglie",
         pd.DataFrame(righe, columns=["Crack spread", "$n$", "Media di $\\bar X_n$", "Media di $S_n$",
                                      "Mediana di $S_n$", "$S_n$ il 20/04/2020", "Quota con $\\bar X_{kn}$"]),
         "Media e deviazione standard trasversale degli spread, per finestra", "tab:soglie",
         "elaborazione propria su dati EIA.",
         "Valori in USD per barile. $\\bar X_n$ e $S_n$ sono la media e la deviazione standard di tutti gli "
         "spread osservati negli ultimi $n$ giorni, su tutte le scadenze (equazioni 6 e 6a di Girma e "
         "Paulson, 1999); la soglia di ingresso è $\\bar X_n \\pm c\\,S_n$. ``Quota con $\\bar X_{kn}$'': "
         "quota di osservazioni (giorno, contratto) per cui la media mobile del singolo contratto, usata per "
         "l'uscita, è disponibile.")


# --- TABELLA 8 - profitti per operazione, quattro pannelli ------------------
def pannello_profitti(filtro, etichetta, usa_gp=False):
    righe_ = []
    for s in SPREAD:
        for n in FINESTRE_ART:
            riga = [f"{NOMI_BREVI[s]}, $n{{=}}{n}$"]
            for i, c in enumerate(SOGLIE_C):
                if usa_gp:
                    gp = GP_TAB4[(s, n)]
                    riga.append(cella(gp[0][i], gp[1][i]))
                    continue
                r = statistiche(filtro(sel(s, n, c)))
                riga.append("--" if r is None else cella(r["media"], r["t"]))
            righe_.append(riga)
    return pd.DataFrame([["@@" + etichetta] + [""] * len(SOGLIE_C)] + righe_,
                        columns=["Strategia"] + [f"$c={num(c)}$" for c in SOGLIE_C])


OP_ART = OPERAZIONI[OPERAZIONI.n.isin(FINESTRE_ART)]
LU_ART, CO_ART = OP_ART[OP_ART.direzione == 1], OP_ART[OP_ART.direzione == -1]
df8 = pd.concat([pannello_profitti(lambda o: o, "A. Tutte"),
                 pannello_profitti(lambda o: o[o.direzione == 1], "B. Lunghe"),
                 pannello_profitti(lambda o: o[o.direzione == -1], "C. Corte"),
                 pannello_profitti(lambda o: o, "D. G&P (tutte)", usa_gp=True)], ignore_index=True)
aggiungi("tabella_08_profitti", df8,
         "Profitto medio per operazione (USD) e statistica $t$, per spread, finestra e soglia", "tab:profitti",
         "elaborazione propria su dati EIA; pannello D: Girma e Paulson (1999), Tabella IV.",
         "Profitti su tre contratti di greggio (3.000 barili); fra parentesi la statistica $t$ sul profitto "
         f"medio per operazione. Sull'insieme delle {num(len(OP_ART), 0, True)} operazioni, le "
         f"{num(len(LU_ART), 0, True)} lunghe guadagnano in media {num(LU_ART.profitto.mean(), 0)} USD e "
         f"chiudono in utile nel {pct((LU_ART.profitto > 0).mean())} dei casi; le {num(len(CO_ART), 0, True)} "
         f"corte perdono in media {num(abs(CO_ART.profitto.mean()), 0)} USD e chiudono in utile nel "
         f"{pct((CO_ART.profitto > 0).mean())} dei casi. Deviazioni standard, quote di operazioni in utile e "
         "statistiche al netto dei costi nella Tabella~\\ref{tab:profittidettaglio}. "
         "Soglie del $t$ unilaterale: 1,65 (5\\%) e 2,33 (1\\%).",
         dim="scriptsize", sep=3)

# --- TABELLA 8-bis - dispersione, costi e rendimento annuo -------------------
righe = []
for s in SPREAD:
    for n in FINESTRE_ART:
        for i, c in enumerate(SOGLIE_C):
            o = sel(s, n, c)
            r = statistiche(o)
            etichetta = NOMI_BREVI[s] if (n == FINESTRE_ART[0] and i == 0) else ""
            if r is None:
                righe.append([etichetta, str(n), num(c), num(len(o), 0, True)] + ["--"] * 6)
                continue
            righe.append([etichetta, str(n), num(c), num(r["N"], 0, True), num(r["sd"], 0, True),
                          pct(r["vinc"]), num(r["t_netto"]), num(r["t_tick"]),
                          num(100 * r["r_lordo"], 1), num(100 * r["r_netto"], 1)])
aggiungi("tabella_08bis_profitti_dettaglio",
         pd.DataFrame(righe, columns=["Crack spread", "$n$", "$c$", "Oper.", "Dev. st.", "\\% vinc.",
                                      "$t$ (100 USD)", "$t$ (85,20 USD)", "$r$ lordo (\\%)", "$r$ netto (\\%)"]),
         "Dispersione, costi di transazione e rendimento annuo per strategia (tutte le operazioni)",
         "tab:profittidettaglio", "elaborazione propria su dati EIA.",
         "Dev. st.: deviazione standard del profitto per operazione (USD); \\% vinc.: quota di operazioni in "
         "utile; $t$ (100 USD) e $t$ (85,20 USD): statistica $t$ sul profitto medio al netto di 100 USD per "
         "operazione completa (Girma e Paulson, 1999) e di 85,20 USD (un tick per gamba in apertura e in "
         "chiusura); $r$: rendimento annuo composto secondo l'equazione~\\eqref{eq:rendimento}, su "
         f"{num(CAPITALE, 0, True)} USD.",
         dim="scriptsize", sep=4)

# --- TABELLA 9 - capitale: drawdown e posizioni aperte -----------------------
righe = []
for s in SPREAD:
    for n in FINESTRE_ART:
        righe.append([f"{NOMI_BREVI[s]}, $n{{=}}{n}$"] +
                     [f"{num(DD[(s, n, c)][0], 0, True)} ({DD[(s, n, c)][1]})" for c in SOGLIE_C])
aggiungi("tabella_09_capitale",
         pd.DataFrame(righe, columns=["Strategia"] + [f"$c = {num(c)}$" for c in SOGLIE_C]),
         "Massimo drawdown sui flussi giornalieri (USD) e numero massimo di posizioni aperte", "tab:capitale",
         "elaborazione propria su dati EIA.",
         "Il drawdown è la massima perdita cumulata dei flussi giornalieri di tutte le posizioni aperte "
         "rispetto al precedente massimo del profitto cumulato; fra parentesi le posizioni aperte "
         f"contemporaneamente al massimo. Capitale ipotizzato da Girma e Paulson (1999): {num(CAPITALE, 0, True)} USD.",
         dim="scriptsize", sep=4)

# --- TABELLA 10 - profitto per distanza dalla consegna -----------------------
righe = []
for s in SPREAD:
    for n in FINESTRE_ART:
        for dz, nome_d in [(1, "lunghe"), (-1, "corte")]:
            o = sel(s, n, 1.5, dz)
            riga = [f"{NOMI_BREVI[s]}, $n{{=}}{n}$, {nome_d}"]
            for mesi in (2, 3, 4):
                m, t, N = media_t(o[o.mesi_alla_consegna == mesi].profitto)
                riga.append(cella(m, t, N) if N > 1 else "--")
            righe.append(riga)
aggiungi("tabella_10_distanza_profitti",
         pd.DataFrame(righe, columns=["Strategia", "+2 mesi", "+3 mesi", "+4 mesi"]),
         "Profitto medio per operazione secondo la distanza dalla consegna ($c = 1{,}50$)", "tab:distanzaprofitti",
         "elaborazione propria su dati EIA.",
         "Profitto medio in USD; fra parentesi la statistica $t$ e il numero di operazioni. Escluse le poche "
         "operazioni aperte a un mese dalla consegna, per cui i totali per riga sono inferiori a quelli della "
         "Tabella~\\ref{tab:operazioni}. Soglie del $t$ unilaterale: 1,65 (5\\%) e 2,33 (1\\%).",
         dim="scriptsize", sep=4)

# --- TABELLE 11, 12, 14, 16: le strategie lunghe significative --------------
OTTO = list(zip(significative, codici))
ETICHETTA_OTTO = {cod: f"{NOMI_BREVI[s]}, {n}, {num(c)}" for (s, n, c), cod in OTTO}
legenda = "; ".join(f"{cod}: {NOMI_BREVI[s]}, $n{{=}}{n}$, $c{{=}}{num(c)}$" for (s, n, c), cod in OTTO)

# eccezioni sull'insieme delle operazioni (finestre dell'articolo), per le note
ecc_t1_tutte = [(s, n, c, media_t(sel(s, n, c).profitto_t1)) for s in SPREAD for n in FINESTRE_ART for c in SOGLIE_C
                if media_t(sel(s, n, c).profitto_t1)[2] > 1 and media_t(sel(s, n, c).profitto_t1)[1] >= 1.645]
ecc_t1_lunghe = [(s, n, c, media_t(sel(s, n, c, 1).profitto_t1)) for s in SPREAD for n in FINESTRE_ART for c in SOGLIE_C
                 if (s, n, c) not in significative and media_t(sel(s, n, c, 1).profitto_t1)[2] > 1
                 and media_t(sel(s, n, c, 1).profitto_t1)[1] >= 1.645]


def descrivi(lista):
    return "; ".join(f"{NOMI_BREVI[s]} con $n = {n}$ e $c = {num(c)}$ ($t = {num(r[1])}$, {num(r[2], 0, True)} operazioni)"
                     for s, n, c, r in lista) or "nessuna"


righe = []
for (s, n, c), cod in OTTO:
    o = sel(s, n, c, 1)
    m0, t0_, N = media_t(o.profitto)
    m1, t1_, _ = media_t(o.profitto_t1)
    righe.append([cod, ETICHETTA_OTTO[cod], num(N, 0, True), cella(m0, t0_), cella(m1, t1_)])
aggiungi("tabella_11_esecuzione",
         pd.DataFrame(righe, columns=["Strategia", "Spread, $n$, $c$", "Oper.", "Stesso giorno", "Giorno successivo"]),
         "Profitto medio per operazione delle strategie lunghe significative con esecuzione nello stesso "
         "giorno del segnale e il giorno successivo", "tab:esecuzione", "elaborazione propria su dati EIA.",
         "Sole operazioni lunghe delle strategie significative al 5\\% nel test per operazione "
         "(Tabella~\\ref{tab:profitti}, pannello B). Profitto medio in USD; fra parentesi la statistica $t$. "
         "Le operazioni sono le stesse nelle due colonne; cambia solo il prezzo di apertura e di chiusura, che "
         "nella seconda è quello del giorno successivo al segnale. Sull'insieme delle trenta strategie "
         "complete (lunghe e corte) nessuna è significativa con l'esecuzione nello stesso giorno; con quella "
         f"ritardata lo diventa: {descrivi(ecc_t1_tutte)}. Fra le lunghe non comprese nelle strategie "
         f"elencate, con esecuzione ritardata diventa significativa: {descrivi(ecc_t1_lunghe)}. " + legenda + ".",
         dim="footnotesize", sep=6, colspec="@{}llrrr@{}")

n_sig = {p[0]: [0, 0] for p in PERIODI}
for s in SPREAD:
    for n in FINESTRE_ART:
        for c in SOGLIE_C:
            for nome_p, a_, b_ in PERIODI:
                _, tp, Np = media_t(per_periodo(sel(s, n, c), a_, b_).profitto)
                if Np > 1 and tp >= 1.645:
                    n_sig[nome_p][0] += 1
                    n_sig[nome_p][1] += Np <= 19
righe = []
for (s, n, c), cod in OTTO:
    o = sel(s, n, c, 1)
    riga = [cod, ETICHETTA_OTTO[cod]]
    for _, a_, b_ in PERIODI:
        m, t, N = media_t(per_periodo(o, a_, b_).profitto)
        riga.append(cella(m, t, N) if N > 1 else "--")
    righe.append(riga)
aggiungi("tabella_12_sottoperiodi",
         pd.DataFrame(righe, columns=["Strategia", "Spread, $n$, $c$", "2007--2014", "2015--2023"]),
         "Profitto medio per operazione delle strategie lunghe significative nei due sottoperiodi",
         "tab:sottoperiodi", "elaborazione propria su dati EIA.",
         "Sole operazioni lunghe delle strategie significative al 5\\% nel test per operazione "
         "(Tabella~\\ref{tab:profitti}, pannello B), assegnate ai sottoperiodi in base alla data di apertura. "
         "Profitto medio in USD; fra parentesi la statistica $t$ e il numero di operazioni. Sull'insieme delle "
         f"trenta strategie complete, {n_sig['2007-2014'][0]} sono significative al 5\\% nel 2007--2014 "
         f"({n_sig['2007-2014'][1]} delle quali con al più 19 operazioni) e {n_sig['2015-2023'][0]} nel "
         "2015--2023. " + legenda + ".",
         dim="footnotesize", sep=6, colspec="@{}llrr@{}")

# --- TABELLA 13 - profitto annuo ---------------------------------------------
righe = []
for a_ in ANNI:
    riga = [str(a_)]
    for (s, n, c), cod in OTTO:
        o = sel(s, n, c, 1)
        riga.append(num(o[o.apertura.dt.year == a_].profitto.sum() / 1000, 1))
    righe.append(riga)
aggiungi("tabella_13_profitto_annuo", pd.DataFrame(righe, columns=["Anno"] + codici),
         "Profitto annuo delle strategie lunghe significative (migliaia di USD)", "tab:profittoannuo",
         "elaborazione propria su dati EIA.",
         "Operazioni assegnate all'anno di apertura; esecuzione nello stesso giorno del segnale. " + legenda + ".",
         dim="scriptsize", sep=3)

# --- TABELLA 14 - concentrazione e robustezza --------------------------------
righe = []
for (s, n, c), cod in OTTO:
    o = sel(s, n, c, 1)
    r = statistiche(o)
    per_anno = o.groupby(o.apertura.dt.year).profitto.sum()
    migliore = per_anno.idxmax()
    _, t_senza, _ = media_t(o[o.apertura.dt.year != migliore].profitto)
    _, t_dopo, _ = media_t(o.profitto_t1)
    righe.append([cod, num(o.profitto.sum() / 1000, 1), str(migliore), pct(per_anno.max() / o.profitto.sum(), 0),
                  num(r["t"]), num(r["t_netto"]), num(t_senza), num(t_dopo),
                  f"{int((per_anno > 0).sum())}/{len(per_anno)}"])
aggiungi("tabella_14_concentrazione",
         pd.DataFrame(righe, columns=["Strategia", "Totale", "Anno migl.", "Quota", "$t$", "$t$ netto",
                                      "$t$ senza", "$t$ +1", "Anni +"]),
         "Concentrazione temporale e robustezza delle strategie lunghe significative", "tab:concentrazione",
         "elaborazione propria su dati EIA.",
         "Totale in migliaia di USD; ``$t$ netto'': al netto di 100 USD per operazione; ``$t$ senza'': "
         "escludendo l'anno migliore; ``$t$ +1'': esecuzione il giorno successivo; ``Anni +'': anni con "
         "profitto positivo sugli anni con operazioni. " + legenda + ".",
         dim="scriptsize", sep=4)

# --- TABELLA 16 - test sui profitti mensili ---------------------------------
t_max_T, t_max_T1 = -np.inf, -np.inf
for s in SPREAD:
    for n in FINESTRE_ART:
        for c in SOGLIE_C:
            o = sel(s, n, c)
            _, tm = t_mensile(o, "profitto")
            _, tm1 = t_mensile(o.dropna(subset=["profitto_t1"]), "profitto_t1")
            t_max_T, t_max_T1 = np.nanmax([t_max_T, tm]), np.nanmax([t_max_T1, tm1])
righe = []
for (s, n, c), cod in OTTO:
    o = sel(s, n, c, 1)
    _, tm = t_mensile(o, "profitto")
    _, tm1 = t_mensile(o.dropna(subset=["profitto_t1"]), "profitto_t1")
    righe.append([cod, ETICHETTA_OTTO[cod], num(t_operazioni(o.profitto)), num(tm), num(tm1),
                  str(o.chiusura.dt.to_period("M").nunique())])
aggiungi("tabella_16_mensili",
         pd.DataFrame(righe, columns=["Strategia", "Spread, $n$, $c$", "$t$ oper.", "$t$ mens.",
                                      "$t$ mens. +1", "Mesi con oper."]),
         "Statistica $t$ per operazione e sui profitti mensili delle strategie lunghe significative "
         "(errori standard HAC)", "tab:mensili", "elaborazione propria su dati EIA.",
         "Sole operazioni lunghe delle strategie significative al 5\\% nel test per operazione "
         "(Tabella~\\ref{tab:profitti}, pannello B). ``$t$ oper.'': test per operazione, come nell'articolo; "
         f"``$t$ mens.'': test sui profitti sommati per mese di chiusura su {len(MESI)} mesi, con errori standard "
         f"di Newey-West a {RITARDI_NW} ritardi; ``+1'': esecuzione il giorno successivo al segnale; ``Mesi con "
         f"oper.'': mesi su {len(MESI)} in cui la strategia ha chiuso almeno un'operazione. Sull'insieme delle "
         f"trenta strategie complete la statistica $t$ mensile massima è {num(t_max_T)} ({num(t_max_T1)} con "
         "esecuzione ritardata). " + legenda + ".",
         dim="footnotesize", sep=6, colspec="@{}llrrrr@{}")

# --- TABELLA 16-bis - finestre aggiuntive: 3 e 15 giorni --------------------
righe = []
for n in FINESTRE_EXT:
    for s in SPREAD:
        for i, c in enumerate((1.50, 1.75, 2.00)):
            o, lu = sel(s, n, c), sel(s, n, c, 1)
            r_t, r_l = statistiche(o), statistiche(lu)
            _, td, _ = media_t(lu.profitto_t1)
            _, tm1 = t_mensile(lu.dropna(subset=["profitto_t1"]), "profitto_t1")
            dd_, _ = DD[(s, n, c)]
            righe.append([str(n) if (s == SPREAD[0] and i == 0) else "", NOMI_BREVI[s] if i == 0 else "", num(c),
                          num(len(o), 0, True), num(o.giorni.mean(), 1),
                          "--" if r_t is None else num(r_t["t"]),
                          "--" if r_l is None else cella(r_l["media"], r_l["t"]),
                          "--" if r_l is None else num(r_l["t_netto"]), num(td), num(tm1),
                          num(dd_, 0, True)])
aggiungi("tabella_16bis_finestre_aggiuntive",
         pd.DataFrame(righe, columns=["$n$", "Crack spread", "$c$", "Oper.", "Giorni", "$t$ tutte",
                                      "Lunghe: media ($t$)", "$t$ netto", "$t$ +1", "$t$ mens. +1", "Drawdown"]),
         "Finestre aggiuntive di 3 e 15 giorni: operazioni, durata, statistiche delle sole lunghe e drawdown",
         "tab:finestreaggiuntive", "elaborazione propria su dati EIA.",
         "Soglie $c \\le 2{,}00$ (le altre hanno poche operazioni). ``$t$ tutte'': statistica sul profitto medio "
         "di lunghe e corte insieme; ``Lunghe'': profitto medio in USD e statistica $t$ delle sole lunghe; "
         "``$t$ netto'': al netto di 100 USD; ``$t$ +1'': con esecuzione il giorno successivo; ``$t$ mens. +1'': "
         f"sui profitti mensili con esecuzione il giorno successivo (Newey-West, {RITARDI_NW} ritardi); "
         "``Drawdown'': massima perdita cumulata di tutte le operazioni (USD). Soglie del $t$ unilaterale: "
         "1,65 (5\\%) e 2,33 (1\\%).",
         dim="scriptsize", sep=3)

# --- TABELLA 17 - sintesi ----------------------------------------------------
stat_art = [statistiche(sel(s, n, c)) for s in SPREAD for n in FINESTRE_ART for c in SOGLIE_C]
stat_art = [r for r in stat_art if r is not None]
righe = [
    ["Periodo", "1983/84--1994 (10--12 anni)", f"{GIORNI[0].year}--{GIORNI[-1].year} (17 anni)"],
    ["Contratti per giorno", "tutti i mesi negoziati", "3--4 (contratti 1--4 EIA)"],
    ["Prezzi I(1)", "si", "si"],
    ["Cointegrazione (Modelli 1--3)", "si, all'1\\%", "si con il BIC; sensibile al criterio dei ritardi"],
    ["Spread stazionari", "tutti e tre, all'1\\%", "3:2:1 e benzina si; gasolio no"],
    ["Operazioni per anno (3:2:1, $c = 1{,}50$)", num((552 + 498) / 10, 0),
     num(sum(len(sel('CS', n, 1.5)) for n in FINESTRE_ART) / 17, 0)],
    ["Profitti significativi (tutte le operazioni)", "29 strategie su 30",
     f"nessuna (statistica $t$ massima: {num(max(r['t'] for r in stat_art))})"],
    ["Operazioni lunghe più redditizie delle corte", "si", "si"],
    ["Dev. st. del profitto per operazione", "621--2.398 USD",
     f"{num(min(r['sd'] for r in stat_art), 0, True)}--{num(max(r['sd'] for r in stat_art), 0, True)} USD"],
    ["Strategie lunghe significative dopo tutte le verifiche", "--",
     "tre al 5\\%, nessuna all'1\\%"],
    ["Massimo drawdown", "entro 75.000 USD",
     f"fino a {num(abs(min(DD[(s, n, c)][0] for s in SPREAD for n in FINESTRE_ART for c in SOGLIE_C)), 0, True)} USD"],
    ["Finestre aggiuntive (3 e 15 giorni)", "--",
     "nessun miglioramento a 15 giorni; sole lunghe sul 3:2:1 a 3 giorni significative"],
]
aggiungi("tabella_17_sintesi", pd.DataFrame(righe, columns=["Aspetto", "Girma e Paulson (1999)", "Questa tesi"]),
         "Sintesi del confronto con Girma e Paulson (1999)", "tab:sintesi",
         "elaborazione propria su dati EIA; Girma e Paulson (1999).",
         "La riga sulle strategie lunghe dopo tutte le verifiche va aggiornata a mano se i risultati cambiano.",
         colspec="@{}p{4.1cm}p{4.2cm}p{6.0cm}@{}")

# --- scrittura ---------------------------------------------------------------
for nome, (df, did, et, fonte, nota, dim, sep, colspec) in TABELLE.items():
    scrivi_tex(nome, df, did, et, fonte, nota, dim, sep, colspec)
with pd.ExcelWriter(os.path.join(CARTELLA_OUT, "Tabelle tesi.xlsx")) as writer:
    for nome, (df, did, et, fonte, nota, dim, sep, colspec) in TABELLE.items():
        pulito = df.apply(lambda col: col.map(pulisci))
        pulito.columns = [pulisci(c) for c in df.columns]
        pulito.to_excel(writer, sheet_name=nome[:31], index=False)

# --- grafici -----------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(6.1, 3.4))
for s, col in zip(SPREAD, ["#1f77b4", "#d62728", "#2ca02c"]):
    ax.plot(CONT.index, CONT[s], lw=0.7, color=col, label=NOMI_BREVI[s])
ax.set_ylabel("USD per barile")
ax.set_xlabel("anno")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(CARTELLA_OUT, "grafico_1_crack_spread.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.1, 3.4))
for (s, n, c), cod in OTTO:
    o = sel(s, n, c, 1).sort_values("chiusura")
    serie = o.groupby("chiusura").profitto.sum().reindex(GIORNI, fill_value=0).cumsum() / 1000
    ax.plot(GIORNI, serie, lw=0.9, label=f"{cod}: {NOMI_BREVI[s]}, n={n}, c={num(c)}")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("profitto cumulato (migliaia di USD)")
ax.set_xlabel("anno")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(CARTELLA_OUT, "grafico_2_profitto_cumulato.png"))
plt.close(fig)

print("\nPEZZO 10 - file generati in", CARTELLA_OUT)
for nome in TABELLE:
    print("  ", nome + ".tex")

# =============================================================================
#  PEZZO 11 - EFFETTO DELLA GUERRA FRA RUSSIA E UCRAINA
# =============================================================================
INIZIO_EV = pd.Timestamp("2016-01-01")
INVASIONE = pd.Timestamp("2022-02-24")
TRIM_GH = 0.15
RITARDI_HAC = 21
MAX_LAG_GH = 12
ALFA_EV = 0.05
CV_GH_C = {1: (-5.13, -4.61, -4.34), 2: (-5.44, -4.92, -4.69)}   # Gregory e Hansen (1996), Tab. 1, mod. C: 1%, 5%, 10%

EV = CONT[CONT.index >= INIZIO_EV].copy()
GG_EV = EV.index
T_EV = len(GG_EV)
I_INV = int(GG_EV.searchsorted(INVASIONE))
MODELLO_DI = {"CS": "Modello 1", "GCS": "Modello 2", "HOCS": "Modello 3"}
ETICHETTA_EV = {"CS": "3:2:1", "GCS": "Benzina 1:1", "HOCS": "Gasolio 1:1"}

print("\n" + "=" * 96)
print("PEZZO 11 - EFFETTO DELLA GUERRA FRA RUSSIA E UCRAINA")
print("=" * 96)
print(f"Campione: {GG_EV[0].date()} -> {GG_EV[-1].date()}, {T_EV} giorni di borsa; invasione al "
      f"{I_INV / T_EV:.1%} del campione, {T_EV - I_INV} giorni successivi")


def adf_residui(e, p):
    de = np.diff(e)
    n = len(de)
    if n - p - 1 < 10:
        return np.nan
    y = de[p:]
    X = np.column_stack([e[p:-1]] + [de[p - i:n - i] for i in range(1, p + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (len(y) - X.shape[1])
    return beta[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])


def gregory_hansen_C(y, Xreg, trim=TRIM_GH):
    """Modello C (spostamento del livello): la statistica e' il minimo dell'ADF sui residui."""
    y = np.asarray(y, float)
    Xreg = np.asarray(Xreg, float)
    T = len(y)
    base = np.column_stack([np.ones(T), Xreg])
    b0, *_ = np.linalg.lstsq(base, y, rcond=None)
    p = ADF(pd.Series(y - base @ b0), trend="n", method="bic", max_lags=MAX_LAG_GH).lags
    a_, b_ = int(trim * T), int((1 - trim) * T)
    stat = np.full(T, np.nan)
    for tau in range(a_, b_):
        D = np.zeros(T)
        D[tau:] = 1.0
        Z = np.column_stack([np.ones(T), D, Xreg])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        stat[tau] = adf_residui(y - Z @ beta, p)
    tau_min = int(np.nanargmin(stat))
    return {"adf": float(stat[tau_min]), "tau": tau_min, "p": p, "m": Xreg.shape[1],
            "serie": stat, "finestra": (a_, b_ - 1)}


GH = {}
for nome, y, x in MODELLI[:3]:
    r = gregory_hansen_C(EV[y], EV[x])
    r["data"] = GG_EV[r["tau"]]
    r["cv"] = CV_GH_C[r["m"]][1]
    r["rifiuta"] = r["adf"] < r["cv"]
    GH[nome] = r
    print(f"Gregory-Hansen {nome}: ADF* {r['adf']:.3f} (ritardi {r['p']}, cv 5% {r['cv']:.2f}), rottura "
          f"{r['data'].date()} ({(r['data'] - INVASIONE).days} giorni dall'invasione)")
FIN_GH = GH["Modello 1"]["finestra"]

ORIZZONTI_EV = [("0-1 mese", 0, 1), ("1-3 mesi", 1, 3), ("3-6 mesi", 3, 6), ("6-12 mesi", 6, 12), ("oltre 12 mesi", 12, None)]


def finestre_evento(indice):
    out = {}
    for nome, m0, m1 in ORIZZONTI_EV:
        da = INVASIONE + pd.DateOffset(months=m0)
        a_ = indice[-1] + pd.Timedelta(days=1) if m1 is None else INVASIONE + pd.DateOffset(months=m1)
        out[nome] = ((indice >= da) & (indice < a_)).astype(float)
    return out


def chow_e_orizzonti(x):
    y = x.to_numpy()
    D = (x.index >= INVASIONE).astype(float)
    rss_r = ((y - y.mean()) ** 2).sum()
    y1, y2 = y[D == 0], y[D == 1]
    rss_u = ((y1 - y1.mean()) ** 2).sum() + ((y2 - y2.mean()) ** 2).sum()
    kk, N = 1, len(y)
    F = ((rss_r - rss_u) / kk) / (rss_u / (N - 2 * kk))
    fit = sm.OLS(y, sm.add_constant(D)).fit(cov_type="HAC", cov_kwds={"maxlags": RITARDI_HAC})
    W = finestre_evento(x.index)
    fo = sm.OLS(y, sm.add_constant(np.column_stack([W[n_] for n_, _, _ in ORIZZONTI_EV]))).fit(
        cov_type="HAC", cov_kwds={"maxlags": RITARDI_HAC})
    return {"media_pre": y1.mean(), "media_post": y2.mean(), "F": F, "p_F": stats.f.sf(F, kk, N - 2 * kk),
            "delta": fit.params[1], "t": fit.tvalues[1], "p": fit.pvalues[1], "mu": fo.params[0],
            "oriz": {n_: {"delta": fo.params[i + 1], "t": fo.tvalues[i + 1], "p": fo.pvalues[i + 1],
                          "se": abs(fo.params[i + 1] / fo.tvalues[i + 1]), "N": int(W[n_].sum())}
                     for i, (n_, _, _) in enumerate(ORIZZONTI_EV)}}


CHOW = {s: chow_e_orizzonti(EV[s]) for s in SPREAD}
for s in SPREAD:
    r = CHOW[s]
    print(f"Chow {ETICHETTA_EV[s]:12s}: prima {r['media_pre']:.2f} dopo {r['media_post']:.2f} spostamento "
          f"{r['delta']:.2f} (t HAC {r['t']:.2f}, F {r['F']:.1f})")
print(f"Errori standard HAC di Newey-West con {RITARDI_HAC} ritardi.")

TABELLE_EV = {}


def aggiungi_ev(nome, df, didascalia, etichetta, fonte="elaborazione propria su dati EIA.", nota=None,
                dim="footnotesize", sep=6, colspec=None):
    TABELLE_EV[nome] = (df, didascalia, etichetta, fonte, nota, dim, sep, colspec)


righe = []
for nome, y, x in MODELLI[:3]:
    r = GH[nome]
    spread_corr = [s for s, mm in MODELLO_DI.items() if mm == nome][0]
    righe.append([nome.replace("Modello", "Mod."), y + " su " + " e ".join(x), ETICHETTA_EV[spread_corr],
                  str(r["p"]), num(r["adf"], 3), r["data"].strftime("%d/%m/%Y"), num(r["cv"]),
                  "si" if r["rifiuta"] else "no", num((r["data"] - INVASIONE).days, 0, True)])
aggiungi_ev("tabella_18_gregory_hansen",
            pd.DataFrame(righe, columns=["Modello", "Regressione", "Spread", "Rit.", "$ADF^*$", "Data stimata",
                                         "c.v. 5\\%", "Rifiuta", "Giorni dall'inv."]),
            "Test di Gregory e Hansen (modello C) sulle regressioni di cointegrazione", "tab:gregoryhansen",
            "elaborazione propria su dati EIA; valori critici: Gregory e Hansen (1996), Tabella 1.",
            f"Campione {GG_EV[0].strftime('%d/%m/%Y')}--{GG_EV[-1].strftime('%d/%m/%Y')}; finestra di ricerca "
            f"{num(100 * TRIM_GH, 0)}\\%--{num(100 * (1 - TRIM_GH), 0)}\\% ({GG_EV[FIN_GH[0]].strftime('%d/%m/%Y')}--"
            f"{GG_EV[FIN_GH[1]].strftime('%d/%m/%Y')}). Ipotesi nulla: assenza di cointegrazione; ``Rifiuta'': la "
            "statistica è inferiore al valore critico al 5\\%. ``Spread'': crack spread corrispondente alla "
            "regressione. Ritardi scelti con il criterio di Schwarz sui residui senza rottura. ``Giorni "
            "dall'inv.'': giorni di calendario.",
            dim="scriptsize", sep=3)

righe = []
for s in SPREAD:
    r = CHOW[s]
    righe.append([ETICHETTA_EV[s], num(r["media_pre"]), num(r["media_post"]), num(r["delta"]), num(r["t"]),
                  "si" if r["p"] < ALFA_EV else "no", num(r["F"], 1), "si" if r["p_F"] < ALFA_EV else "no"])
aggiungi_ev("tabella_19_chow",
            pd.DataFrame(righe, columns=["Crack spread", "Media prima", "Media dopo", "Spostamento", "$t$ (HAC)",
                                         "Signif. 5\\%", "$F$", "Signif. 5\\% ($F$)"]),
            "Test di Chow sui crack spread, con data di rottura imposta al 24 febbraio 2022", "tab:chow",
            "elaborazione propria su dati EIA.",
            f"Valori in USD per barile; campione {GG_EV[0].strftime('%d/%m/%Y')}--{GG_EV[-1].strftime('%d/%m/%Y')}. "
            "Lo spostamento è il coefficiente della variabile indicatrice, con errori standard HAC di Newey-West "
            f"({RITARDI_HAC} ritardi); $F$ è il test di Chow classico. Per costruzione lo spostamento del 3:2:1 è "
            "la media ponderata di quelli dei due crack 1:1, con pesi 2/3 e 1/3.",
            dim="scriptsize", sep=4)

righe = []
for s in SPREAD:
    for i, (nome, _, _) in enumerate(ORIZZONTI_EV):
        o = CHOW[s]["oriz"][nome]
        righe.append([ETICHETTA_EV[s] if i == 0 else "", nome, num(o["N"], 0, True), num(o["delta"]),
                      num(o["t"]), "si" if o["p"] < ALFA_EV else "no"])
aggiungi_ev("tabella_20_orizzonti",
            pd.DataFrame(righe, columns=["Crack spread", "Finestra", "Giorni", "Scostamento", "$t$ (HAC)", "Signif. 5\\%"]),
            "Scostamento del crack spread dal livello precedente l'invasione, per orizzonte", "tab:orizzonti",
            "elaborazione propria su dati EIA.",
            "Scostamenti in USD per barile rispetto al livello medio del periodo precedente il 24 febbraio 2022, "
            "stimati in un'unica regressione con cinque variabili indicatrici; errori standard HAC di Newey-West "
            f"({RITARDI_HAC} ritardi). L'ultima finestra arriva alla fine del campione ({GG_EV[-1].strftime('%d/%m/%Y')}).",
            dim="footnotesize", sep=5)

for nome, (df, did, et, fonte, nota, dim, sep, colspec) in TABELLE_EV.items():
    scrivi_tex(nome, df, did, et, fonte, nota, dim, sep, colspec)
with pd.ExcelWriter(os.path.join(CARTELLA_OUT, "Tabelle evento.xlsx")) as writer:
    for nome, (df, did, et, fonte, nota, dim, sep, colspec) in TABELLE_EV.items():
        pulito = df.apply(lambda col: col.map(pulisci))
        pulito.columns = [pulisci(c) for c in df.columns]
        pulito.to_excel(writer, sheet_name=nome[:31], index=False)

W_EV = finestre_evento(GG_EV)
fig, assi = plt.subplots(3, 1, figsize=(6.1, 7.2), sharex=True)
for ax, s in zip(assi, SPREAD):
    ax.plot(GG_EV, EV[s], lw=0.7, color="#1f77b4")
    ax.axvline(INVASIONE, color="black", lw=1.0, ls="--")
    data_gh = GH[MODELLO_DI[s]]["data"]
    ax.axvline(data_gh, color="#d62728", lw=1.0, ls=":")
    pre = GG_EV[GG_EV < INVASIONE]
    ax.hlines(CHOW[s]["mu"], pre[0], INVASIONE, color="#2ca02c", lw=1.3)
    for nome, _, _ in ORIZZONTI_EV:
        gg = GG_EV[W_EV[nome].astype(bool)]
        if len(gg):
            ax.hlines(CHOW[s]["mu"] + CHOW[s]["oriz"][nome]["delta"], gg[0], gg[-1], color="#2ca02c", lw=1.3)
    ax.set_ylabel("USD per barile")
    ax.set_title(f"{ETICHETTA_EV[s]}: spostamento {CHOW[s]['delta']:+.2f} USD, rottura stimata "
                 f"{data_gh.strftime('%d/%m/%Y')}", fontsize=8)
assi[0].plot([], [], color="black", lw=1.0, ls="--", label="24 febbraio 2022")
assi[0].plot([], [], color="#d62728", lw=1.0, ls=":", label="rottura stimata (Gregory-Hansen)")
assi[0].plot([], [], color="#2ca02c", lw=1.3, label="livello medio per finestra")
assi[0].legend(loc="upper left", frameon=False, fontsize=7)
assi[-1].set_xlabel("anno")
assi[-1].xaxis.set_major_locator(mdates.YearLocator())
assi[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(os.path.join(CARTELLA_OUT, "grafico_3_evento.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.1, 3.4))
colori = {"Modello 1": "#1f77b4", "Modello 2": "#d62728", "Modello 3": "#2ca02c"}
for nome, y, x in MODELLI[:3]:
    r = GH[nome]
    a_, b_ = r["finestra"]
    ax.plot(GG_EV[a_:b_ + 1], r["serie"][a_:b_ + 1], lw=0.9, color=colori[nome], label=f"{nome} ({y} su {' e '.join(x)})")
    ax.plot(r["data"], r["adf"], "o", ms=4, color=colori[nome])
for m_, stile in [(1, ":"), (2, "--")]:
    ax.axhline(CV_GH_C[m_][1], color="#7f7f7f", lw=0.8, ls=stile,
               label=f"valore critico 5% ({m_} regressor" + ("e" if m_ == 1 else "i") + ")")
ax.axvline(INVASIONE, color="black", lw=1.0, ls="--", label="24 febbraio 2022")
ax.set_ylabel("statistica ADF sui residui")
ax.set_xlabel("data di rottura ipotizzata")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="lower left", frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(CARTELLA_OUT, "grafico_4_gregory_hansen.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.1, 3.4))
nomi_or = [n_ for n_, _, _ in ORIZZONTI_EV]
xpos = np.arange(len(nomi_or))
for i, (s, colore) in enumerate(zip(SPREAD, ["#1f77b4", "#d62728", "#2ca02c"])):
    d_ = [CHOW[s]["oriz"][n_]["delta"] for n_ in nomi_or]
    se_ = [1.96 * CHOW[s]["oriz"][n_]["se"] for n_ in nomi_or]
    ax.bar(xpos + (i - 1) * 0.26, d_, 0.26, yerr=se_, capsize=2, color=colore, label=ETICHETTA_EV[s])
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(nomi_or)
ax.set_ylabel("scostamento (USD per barile)")
ax.set_xlabel("finestra dopo l'invasione")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(CARTELLA_OUT, "grafico_5_orizzonti.png"))
plt.close(fig)
print("File dell'evento generati:", ", ".join(sorted(TABELLE_EV)))

# =============================================================================
#  PEZZO 12 - ASTERISCHI DI SIGNIFICATIVITA' NELLE TABELLE .TEX
#  Tabelle sui profitti: asterischi solo sui t positivi (H1: profitto > 0).
#  Tabelle sull'evento: asterischi su |t|. Gregory-Hansen: valori critici propri.
# =============================================================================
SOGLIE_T = {liv: norm.ppf(1 - liv) for liv in (0.10, 0.05, 0.01)}
LEG_PROFITTI = (r"$^{*}$, $^{**}$, $^{***}$: profitto medio significativamente positivo al 10\%, "
                r"5\% e 1\% (test unilaterale, soglie 1,28, 1,65 e 2,33).")
LEG_EVENTO = (r"$^{*}$, $^{**}$, $^{***}$: significativo al 10\%, 5\% e 1\% "
              r"(soglie 1,28, 1,65 e 2,33 sul valore assoluto della statistica).")
LEG_GH = (r"$^{*}$, $^{**}$, $^{***}$: rifiuto al 10\%, 5\% e 1\% "
          r"(valori critici di Gregory e Hansen, 1996, Tabella 1, modello C).")
VECCHIA_NOTA = re.compile(r"Soglie del \$t\$ unilaterale: 1,65 \(5\\%\) e 2,33 \(1\\%\)\.?")
PARENTESI = re.compile(r"\((\$-\$)?(\d+,\d+)(;[^)]*)?\)")
NUMERO = re.compile(r"(\$-\$)?\d+,\d+")
TABELLE_AST = {   # file -> (modalita', colonne con t, solo_positivi)
    "tabella_08_profitti.tex":              ("parentesi", None, True),
    "tabella_10_distanza_profitti.tex":     ("parentesi", None, True),
    "tabella_11_esecuzione.tex":            ("parentesi", None, True),
    "tabella_12_sottoperiodi.tex":          ("parentesi", None, True),
    "tabella_16bis_finestre_aggiuntive.tex": ("parentesi", None, True),
    "tabella_08bis_profitti_dettaglio.tex": ("colonne", [6, 7], True),
    "tabella_14_concentrazione.tex":        ("colonne", [4, 5, 6, 7], True),
    "tabella_16_mensili.tex":               ("colonne", [2, 3, 4], True),
    "tabella_19_chow.tex":                  ("colonne", [4], False),
    "tabella_20_orizzonti.tex":             ("colonne", [4], False),
    "tabella_18_gregory_hansen.tex":        ("gregory_hansen", None, False),
}


def stelle_t(t, solo_positivi):
    if solo_positivi and t <= 0:
        return ""
    a_ = abs(t)
    if a_ >= SOGLIE_T[0.01]:
        return "$^{***}$"
    if a_ >= SOGLIE_T[0.05]:
        return "$^{**}$"
    if a_ >= SOGLIE_T[0.10]:
        return "$^{*}$"
    return ""


def stelle_gh(v, m):
    c1, c5, c10 = CV_GH_C[m]
    return "$^{***}$" if v < c1 else "$^{**}$" if v < c5 else "$^{*}$" if v < c10 else ""


def a_numero(s):
    return float(s.replace("$-$", "-").replace(".", "").replace(",", "."))


def aggiungi_asterischi(cartella):
    for nome, (modo, colonne, solo_pos) in TABELLE_AST.items():
        percorso = os.path.join(cartella, nome)
        if not os.path.exists(percorso):
            print("   non trovata, saltata:", nome)
            continue
        testo = open(percorso, encoding="utf-8").read().replace("\r\n", "\n")
        if "$^{*" in testo.split(r"\bottomrule")[0]:
            print("   asterischi gia' presenti, saltata:", nome)
            continue
        out, nel_corpo = [], False
        for riga in testo.split("\n"):
            if r"\midrule" in riga:
                nel_corpo = True
                out.append(riga)
                continue
            if r"\bottomrule" in riga:
                nel_corpo = False
            if nel_corpo and "&" in riga and riga.strip().endswith(r"\\"):
                if modo == "parentesi":
                    riga = PARENTESI.sub(lambda mt: mt.group(0) +
                                         stelle_t(a_numero((mt.group(1) or "") + mt.group(2)), solo_pos), riga)
                else:
                    celle = [c.strip() for c in riga.rstrip()[:-2].split("&")]
                    if modo == "colonne":
                        for i in colonne:
                            if NUMERO.fullmatch(celle[i]):
                                celle[i] += stelle_t(a_numero(celle[i]), solo_pos)
                    else:
                        celle[4] += stelle_gh(a_numero(celle[4]), 2 if "HU e HO" in celle[1] else 1)
                    riga = "    " + " & ".join(celle) + r" \\"
            out.append(riga)
        testo = "\n".join(out)
        legenda_ = LEG_GH if modo == "gregory_hansen" else (LEG_PROFITTI if solo_pos else LEG_EVENTO)
        if VECCHIA_NOTA.search(testo):
            testo = VECCHIA_NOTA.sub(legenda_, testo)
        elif r"\par\vspace{0.2em}{\footnotesize" in testo:
            testo = re.sub(r"(\\par\\vspace\{0\.2em\}\{\\footnotesize .*?)\}\n",
                           lambda mt: mt.group(1) + " " + legenda_ + "}\n", testo, count=1, flags=re.S)
        else:
            testo = testo.replace(r"  \fonte{", "  \\par\\vspace{0.2em}{\\footnotesize " + legenda_ + "}\n  \\fonte{", 1)
        open(percorso, "w", encoding="utf-8").write(testo)
        print("   asterischi aggiunti:", nome)


def ricostruisci_tutte_le_tabelle(cartella):
    files = sorted(glob.glob(os.path.join(cartella, "tabella_*.tex")))
    with open(os.path.join(cartella, "tutte le tabelle.tex"), "w", encoding="utf-8") as f:
        for p in files:
            f.write("% " + os.path.basename(p) + "\n")
            f.write(open(p, encoding="utf-8").read().rstrip() + "\n\n")
    print("   ricostruito: tutte le tabelle.tex (%d tabelle)" % len(files))


print("\nPEZZO 12 - ASTERISCHI")
aggiungi_asterischi(CARTELLA_OUT)
ricostruisci_tutte_le_tabelle(CARTELLA_OUT)