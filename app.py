import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Bot de trading (simulation)", layout="wide")
st.title("Bot de trading crypto - mode simulation")
st.warning("Simulation uniquement : aucun ordre réel n'est passé. Les performances passées ne garantissent pas les résultats futurs.")


@st.cache_data(show_spinner=False)
def get_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    # Source : Yahoo Finance (gratuit, sans cle). Binance bloque les serveurs Streamlit.
    yahoo_symbol = symbol.replace("USDT", "-USD")
    yahoo_interval = {"1d": "1d", "4h": "1h", "1h": "1h"}[interval]
    range_ = "5y" if interval == "1d" else "730d"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, params={"interval": yahoo_interval, "range": range_},
                     headers=headers, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s"),
        "close": res["indicators"]["quote"][0]["close"],
    }).dropna()
    if interval == "4h":
        df = df.set_index("date").resample("4h").last().dropna().reset_index()
    return df.tail(limit).reset_index(drop=True)


def backtest(df, fast, slow, trend, fee, stop, trail, risk_frac, vol_cap, prudent, capital):
    p = df["close"].to_numpy()
    s_p = pd.Series(p)
    f = s_p.rolling(fast).mean().to_numpy()
    s = s_p.rolling(slow).mean().to_numpy()
    t = s_p.rolling(trend).mean().to_numpy()
    v = s_p.pct_change().rolling(20).std().to_numpy()
    start_i = max(trend, slow, 21)
    cash, qty, entry, peak = capital, 0.0, 0.0, 0.0
    equity = np.full(len(p), capital, dtype=float)
    trades = []
    markers_buy, markers_sell = [], []
    for i in range(len(p)):
        if i < start_i:
            equity[i] = cash
            continue
        if qty > 0:
            peak = max(peak, p[i])
            exit_sig = f[i - 1] >= s[i - 1] and f[i] < s[i]
            if prudent:
                exit_sig = exit_sig or p[i] <= entry * (1 - stop) or p[i] <= peak * (1 - trail)
            else:
                exit_sig = exit_sig or p[i] <= entry * (1 - 0.08)
            if exit_sig:
                cash += qty * p[i] * (1 - fee)
                trades.append({"entrée": df["date"].iloc[entry_i], "sortie": df["date"].iloc[i],
                               "prix entrée": entry, "prix sortie": p[i],
                               "résultat %": round((p[i] / entry - 1 - 2 * fee) * 100, 2)})
                markers_sell.append(i)
                qty = 0.0
        else:
            cross_up = f[i - 1] <= s[i - 1] and f[i] > s[i]
            ok = cross_up and ((p[i] > t[i] and v[i] < vol_cap) if prudent else True)
            if ok:
                invest = cash * (risk_frac if prudent else 1.0)
                qty = invest * (1 - fee) / p[i]
                cash -= invest
                entry = peak = p[i]
                entry_i = i
                markers_buy.append(i)
        equity[i] = cash + qty * p[i]
    eq = pd.Series(equity, index=df["date"])
    return eq, pd.DataFrame(trades), markers_buy, markers_sell, start_i


with st.sidebar:
    st.header("Réglages")
    pair = st.selectbox("Paire", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
    interval = st.selectbox("Bougies", ["1d", "4h", "1h"], index=0)
    limit = st.slider("Nombre de bougies", 300, 1000, 1000, step=50)
    capital = st.number_input("Capital de départ (simulé)", 100.0, 1_000_000.0, 1000.0, step=100.0)
    profil = st.radio("Profil", ["Prudent (V2)", "Offensif (V1)"])
    prudent = profil.startswith("Prudent")
    st.subheader("Stratégie")
    fast = st.slider("Moyenne mobile rapide", 5, 50, 20)
    slow = st.slider("Moyenne mobile lente", 20, 150, 50)
    trend = st.slider("Filtre de tendance (V2)", 100, 300, 200)
    fee = st.slider("Frais par ordre (%)", 0.0, 0.5, 0.1, step=0.01) / 100
    if prudent:
        stop = st.slider("Stop-loss (%)", 2, 15, 5) / 100
        trail = st.slider("Stop suiveur (%)", 3, 25, 8) / 100
        risk_frac = st.slider("Part du capital investie (%)", 10, 100, 50) / 100
        vol_cap = st.slider("Volatilité max pour entrer (%)", 1, 10, 5) / 100
    else:
        stop, trail, risk_frac, vol_cap = 0.08, 0.08, 1.0, 1.0
    run = st.button("Lancer le backtest", type="primary")

if fast >= slow:
    st.error("La moyenne rapide doit être plus courte que la lente.")
    st.stop()

if run:
    try:
        df = get_data(pair, interval, limit)
    except Exception as e:
        st.error(f"Impossible de télécharger les données : {e}")
        st.stop()
    st.caption(f"{len(df)} bougies reçues, du {df['date'].iloc[0].date()} au {df['date'].iloc[-1].date()}" if len(df) else "Aucune bougie reçue")
    if len(df) < max(trend, slow, 21) + 30:
        st.error(f"Pas assez de données : {len(df)} bougies reçues, il en faut au moins {max(trend, slow, 21) + 30}. Essaie une autre paire ou un autre intervalle.")
        st.stop()
    eq, trades, mb, ms, start_i = backtest(df, fast, slow, trend, fee, stop, trail, risk_frac, vol_cap, prudent, capital)
    bh = df["close"].iloc[-1] / df["close"].iloc[start_i] - 1
    ret = eq.iloc[-1] / capital - 1
    dd = (eq / eq.cummax() - 1).min()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement du bot", f"{ret * 100:.1f}%")
    c2.metric("Acheter et garder", f"{bh * 100:.1f}%")
    c3.metric("Perte max en cours de route", f"{dd * 100:.1f}%")
    wr = (trades["résultat %"] > 0).mean() * 100 if len(trades) else 0
    c4.metric("Trades gagnants", f"{wr:.0f}% sur {len(trades)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Capital du bot"))
    bh_curve = capital * df["close"] / df["close"].iloc[start_i]
    fig.add_trace(go.Scatter(x=df["date"], y=bh_curve, name="Acheter et garder", line=dict(dash="dot")))
    fig.update_layout(height=420, yaxis_title="Valeur", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Prix"))
    fig2.add_trace(go.Scatter(x=df["date"].iloc[mb], y=df["close"].iloc[mb], mode="markers",
                              name="Achat", marker=dict(symbol="triangle-up", size=11, color="green")))
    fig2.add_trace(go.Scatter(x=df["date"].iloc[ms], y=df["close"].iloc[ms], mode="markers",
                              name="Vente", marker=dict(symbol="triangle-down", size=11, color="red")))
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Historique des trades")
    st.dataframe(trades, use_container_width=True)
    st.caption("Conseil : teste plusieurs paires et plusieurs périodes. Si le bot ne marche que sur une seule, méfie-toi, c'est probablement du hasard.")
else:
    st.info("Choisis tes réglages à gauche puis clique sur « Lancer le backtest ».")
