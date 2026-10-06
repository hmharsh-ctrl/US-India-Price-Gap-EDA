"""ADR Spread Monitor: how far US-listed ADRs drift from their NSE share prices (Streamlit).

The only AI feature is the optional written market briefing from Gemini 3.6 Flash.
Everything else (charts, flags, explanations) is plain calculation.

Run:
    pip install streamlit pandas numpy altair google-genai
    py -m streamlit run streamlit_dashboard.py
"""

import json
import os
import random
import time
from datetime import datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ================================ CONFIG ====================================

STOCKS = {
    "HDFC_BANK":   {"label": "HDFC Bank",   "us": "HDB",  "nse": "HDFC.NS",       "ratio": 3},
    "INFOSYS":     {"label": "Infosys",     "us": "INFY", "nse": "INFY.NS",       "ratio": 1},
    "WIPRO":       {"label": "Wipro",       "us": "WIT",  "nse": "WIPRO.NS",      "ratio": 1},
    "ICICI_BANK":  {"label": "ICICI Bank",  "us": "IBN",  "nse": "ICICIBANK.NS",  "ratio": 2},
    "TATA_MOTORS": {"label": "Tata Motors", "us": "TTM",  "nse": "TATAMOTORS.NS", "ratio": 5},
}
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(HERE, "us_india_stock_gap_ai.csv")   # the file from your EDA folder
SIM_FILE = os.path.join(HERE, "adr_spread_data_v3.csv")      # realistic simulation (auto-created)
HISTORY_DAYS = 1650

MODEL_NAME = "gemini-3.6-flash"      # the only model this app uses
API_CALL_DELAY_SEC = 1.0
MAX_CONSECUTIVE_FAILURES = 4

REQUIRED = {"date", "stock_name", "us_adr_price_usd", "nse_close_inr", "usdinr",
            "adr_ratio", "indian_eq_price", "price_gap_inr", "price_gap_pct"}

# per data source: label, default flag level, slider max, slider step
SOURCES = {
    "csv": {"name": "Your CSV (EDA folder)", "thr": 50.0, "max": 150.0, "step": 1.0},
    "sim": {"name": "Realistic simulation", "thr": 1.5, "max": 5.0, "step": 0.1},
}

INK, SLATE, RULE = "#1A2150", "#5E688A", "#E3E8F2"
INDIGO, GREEN, RED, AMBER = "#4F46E5", "#10A36B", "#E5484D", "#F59E0B"

st.set_page_config(page_title="ADR Spread Monitor", page_icon="📈", layout="wide")

# ================================== STYLE ===================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Figtree:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Figtree', sans-serif; color:#1E2447; }}
.stApp {{ background:#F4F6FB; }}
h1, h2, h3 {{ font-family:'Bricolage Grotesque', sans-serif !important; color:{INK}; letter-spacing:-0.01em; }}
h3 {{ margin-bottom:.2rem; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:2.2rem; max-width:1250px; }}

section[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#141B4D,#1F2A6B); }}
section[data-testid="stSidebar"] * {{ color:#E9ECFA !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="tag"],
section[data-testid="stSidebar"] input {{ background:#2C3A82 !important; border-color:#4658B0 !important; }}
section[data-testid="stSidebar"] .stButton > button {{ background:transparent; border:1px solid #7C8BD6; color:#fff;
  border-radius:10px; transition:all .2s; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background:#2C3A82; transform:translateY(-1px); }}

[class*="st-key-panel"] {{ background:#fff; border:1px solid {RULE}; border-radius:16px; padding:1.1rem 1.25rem;
  box-shadow:0 1px 2px rgba(26,33,80,.04); transition:box-shadow .25s ease; }}
[class*="st-key-panel"]:hover {{ box-shadow:0 8px 24px rgba(26,33,80,.08); }}
[class*="st-key-panel"] div[data-testid="stMetric"] {{ background:transparent; border:none;
  border-left:3px solid {INDIGO}; padding:.1rem .9rem; }}
[class*="st-key-panel"] div[data-testid="stMetricValue"] {{ font-family:'Bricolage Grotesque',sans-serif; color:{INK}; }}

.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; color:{SLATE}; padding:.65rem 1.1rem; border-radius:10px 10px 0 0;
  transition:background .2s, color .2s; }}
.stTabs [data-baseweb="tab"]:hover {{ background:#E9ECFA; color:{INK}; }}
.stTabs [aria-selected="true"] {{ color:{INDIGO}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{INDIGO}; height:3px; }}
.stTabs [data-baseweb="tab-panel"] {{ animation:swap .35s ease; }}
@keyframes swap {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:none; }} }}

.stButton > button, .stDownloadButton > button {{ border-radius:10px; font-weight:600; transition:all .2s; }}
.stDownloadButton > button:hover {{ transform:translateY(-1px); box-shadow:0 4px 12px rgba(79,70,229,.2); }}

.hint {{ color:{SLATE}; font-size:.88rem; margin:.1rem 0 .6rem; }}
.pill {{ display:inline-block; padding:.12rem .6rem; border-radius:999px; font-weight:600; font-size:.85rem; }}
.pill.up {{ background:#DDF6EC; color:#0B7A50; }}  .pill.down {{ background:#FDE6E7; color:#B52A30; }}
.row {{ display:flex; justify-content:space-between; align-items:center; padding:.55rem 0; border-bottom:1px solid #EEF1F7; }}
.row:last-child {{ border-bottom:none; }}
.row small {{ color:{SLATE}; display:block; }}
.flag {{ font-size:.75rem; color:#B45309; background:#FEF3C7; padding:.05rem .45rem; border-radius:6px; margin-left:.4rem; }}
.callout {{ background:#EEF0FF; border-left:4px solid {INDIGO}; border-radius:10px; padding:.9rem 1.1rem; }}
.brief {{ background:#fff; border:1px solid #D9DCFB; border-left:4px solid {INDIGO}; border-radius:12px;
  padding:1rem 1.2rem; margin-top:.6rem; animation:rise .4s ease; }}
.brief .tag {{ display:inline-block; font-size:.78rem; font-weight:600; color:{INDIGO}; background:#EEF0FF;
  padding:.1rem .6rem; border-radius:999px; margin-bottom:.5rem; }}
.notice {{ background:#FFF7E6; border:1px solid #FBD38D; border-radius:12px; padding:.8rem 1.1rem; margin:.8rem 0 .2rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>
""", unsafe_allow_html=True)


def panel(name: str):
    """A white card. Falls back to a bordered container on older Streamlit versions."""
    try:
        return st.container(key=f"panel_{name}")
    except TypeError:
        return st.container(border=True)

# ============================== DATA LOADING ================================

@st.cache_data(show_spinner="Generating realistic simulation...")
def generate_simulation() -> pd.DataFrame:
    dates = pd.bdate_range(end=datetime.today(), start=datetime.today() - timedelta(days=HISTORY_DAYS))
    n = len(dates)
    rng = np.random.default_rng(42)
    fx = 82.0 + np.cumsum(rng.normal(0.001, 0.04, n))
    frames = []
    for key, meta in STOCKS.items():
        us = rng.uniform(25, 120) * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
        spread = np.zeros(n)                      # mean-reverting gap, like real ADRs
        for i in range(1, n):
            spread[i] = 0.6 * spread[i - 1] + rng.normal(0, 0.007)
        implied = us * fx / meta["ratio"]
        nse = implied / (1 + spread)
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "stock_name": key,
            "us_adr_price_usd": us.round(2), "nse_close_inr": nse.round(2),
            "usdinr": fx.round(2), "adr_ratio": meta["ratio"],
            "indian_eq_price": implied.round(2),
            "price_gap_inr": (implied - nse).round(2),
            "price_gap_pct": ((implied - nse) / nse * 100).round(2)}))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(SIM_FILE, index=False)
    return df


@st.cache_data
def load_source(src: str):
    """Returns (dataframe, message). message is set when we had to fall back."""
    if src == "csv":
        if not os.path.exists(CSV_FILE):
            return generate_simulation(), "us_india_stock_gap_ai.csv was not found next to this script, so the simulation is shown."
        df = pd.read_csv(CSV_FILE)
        missing = REQUIRED - set(df.columns)
        if missing:
            return generate_simulation(), f"The CSV is missing columns {sorted(missing)}, so the simulation is shown."
        return df, None
    if os.path.exists(SIM_FILE):
        df = pd.read_csv(SIM_FILE)
        if REQUIRED.issubset(df.columns):
            return df, None
    return generate_simulation(), None


def explain(gap: float, thr: float) -> str:
    """Plain rule-based explanation. No AI involved."""
    mult = abs(gap) / thr if thr else 1
    size = "just over" if mult < 1.25 else "well over" if mult < 2 else "more than double"
    where = "above its Indian share price" if gap > 0 else "below its Indian share price"
    text = (f"After converting currency and share ratio, the US ADR is priced {abs(gap):.2f}% {where}. "
            f"That is {size} your {thr:g}% flag level. ")
    if abs(gap) > 10:
        text += ("Real ADRs rarely drift this far from their home shares, so a gap this wide may point to a "
                 "data issue rather than a market event.")
    else:
        text += ("Gaps like this usually reflect overnight US trading, exchange-rate moves or thin liquidity. "
                 "On paper it looks like an opportunity, but costs, FX risk and settlement rules mean it is "
                 "not guaranteed profit.")
    return text

# ================================= GEMINI ===================================
# The only AI feature in this app: a written briefing for one flagged day.

def get_client(api_key: str):
    if not HAS_GENAI:
        return None, "The google-genai package is not installed. Run: pip install google-genai"
    if not api_key:
        return None, "No API key entered."
    try:
        return genai.Client(api_key=api_key), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def build_prompt(stock, date, gap_pct, us_price, nse_price, fx, ratio) -> str:
    implied = us_price * fx / ratio
    diff = implied - nse_price
    return (
        f"You are a financial market analyst reviewing one flagged ADR-NSE price gap for {stock} on {date}.\n\n"
        f"DATA:\n- US ADR price: ${us_price:.2f}\n- USD/INR: {fx:.2f}\n"
        f"- ADR ratio: 1 ADR = {ratio} NSE shares\n"
        f"- ADR-implied NSE price: INR {implied:.2f}\n- NSE close: INR {nse_price:.2f}\n"
        f"- Difference: INR {diff:+.2f}\n- Gap: {gap_pct:+.2f}%\n\n"
        f"RULES: Do not invent numbers. Separate observed data from possible explanations. "
        f"Do not claim any cause as certain. A large theoretical spread is not guaranteed profit. "
        f"If the gap is implausibly large for a liquid ADR (above roughly 10%), say plainly that it may "
        f"indicate a data problem rather than a market event.\n\n"
        f"Answer in exactly this structure:\n\n"
        f"Numerical Analysis:\nState the implied price, NSE price, INR difference and % gap.\n\n"
        f"Market Interpretation:\n2-3 concise sentences on plausible reasons (overnight US trading, FX, "
        f"liquidity, timing, news, regulation, settlement, borrowing limits).\n\n"
        f"Arbitrage View:\nSay whether this is a theoretical arbitrage signal, and why costs, borrowing, "
        f"FX risk, settlement, liquidity and regulation may prevent capturing it.")


def gemini_briefing(client, prompt: str, max_retries: int = 5) -> str:
    """Calls gemini-3.6-flash. Retries busy-server and rate-limit errors with growing waits."""
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = (resp.text or "").strip()
            if not text:
                raise ValueError("Gemini returned an empty answer.")
            return text
        except Exception as e:
            last, msg = e, str(e)
            if getattr(e, "code", None) == 404 or "NOT_FOUND" in msg:
                raise RuntimeError(f"Model '{MODEL_NAME}' was not found for this API key.") from e
            retryable = (getattr(e, "code", None) in (429, 500, 503, 504)
                         or any(s in msg for s in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "503")))
            if not retryable:
                raise
            time.sleep(min(2 ** attempt + random.random(), 30))
    raise last


def render_brief(text: str) -> str:
    html = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    for h in ("Numerical Analysis", "Market Interpretation", "Arbitrage View"):
        html = html.replace(f"{h}:", f"<b>{h}</b><br>")
    html = html.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return f"<div class='brief'><span class='tag'>Gemini 3.6 Flash</span><br>{html}</div>"

# ================================ SIDEBAR ===================================

st.sidebar.markdown("### Data")
src_choice = st.sidebar.radio("Data source", [SOURCES["csv"]["name"], SOURCES["sim"]["name"]],
                              help="The CSV from your EDA folder, or a simulation where ADR and NSE prices "
                                   "stay realistically close.")
src = "csv" if src_choice == SOURCES["csv"]["name"] else "sim"
S = SOURCES[src]
weekdays_only = st.sidebar.toggle("Weekdays only", value=True, help="Hides Saturdays and Sundays.")

raw, load_msg = load_source(src)
df = raw.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().any():
    st.error("The dataset has invalid dates. Check the CSV or switch to the simulation.")
    st.stop()
df["label"] = df["stock_name"].map(lambda k: STOCKS.get(k, {}).get("label", str(k).replace("_", " ").title()))
df["abs_gap"] = df["price_gap_pct"].abs()
df["side"] = np.where(df["price_gap_pct"] > 0, "ADR premium", "ADR discount")
all_labels = sorted(df["label"].unique())
dmin, dmax = df["date"].min().date(), df["date"].max().date()
if weekdays_only:
    df = df[df["date"].dt.dayofweek < 5]


def K(name):                          # per-source widget keys, so ranges never clash
    return f"{name}_{src}"


DEFAULTS = {K("thr"): S["thr"], K("picked"): all_labels, K("rng"): (dmin, dmax), K("dir"): "Both"}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


st.sidebar.markdown("### Filters")
st.sidebar.slider("Flag gaps at or above (%)", 0.5, S["max"], step=S["step"], key=K("thr"),
                  help="A day is flagged when the gap is at least this big.")
st.sidebar.multiselect("Stocks", all_labels, key=K("picked"))
st.sidebar.slider("Date range", min_value=dmin, max_value=dmax, key=K("rng"), format="MMM YYYY")
st.sidebar.radio("Direction", ["Both", "ADR premium", "ADR discount"], key=K("dir"), horizontal=True,
                 help="Premium: ADR costs more than the NSE share. Discount: it costs less.")
st.sidebar.button("Reset filters", on_click=reset_filters)

st.sidebar.markdown("### Market briefings")
brief_mode = st.sidebar.radio("Briefing type", ["Standard (no AI)", "Gemini 3.6 Flash"],
                              help="Standard uses fixed rules. Gemini writes a short analysis for one day "
                                   "at a time, only when you press a button.")
client = None
if brief_mode.startswith("Gemini"):
    env_key = os.environ.get("GEMINI_API_KEY", "")
    key_in = st.sidebar.text_input("Gemini API key", type="password",
                                   placeholder="Using GEMINI_API_KEY from environment" if env_key else "",
                                   help="Kept in this session only. Never saved to disk.")
    api_key = key_in or env_key
    if api_key:
        client, err = get_client(api_key)
        if err:
            st.sidebar.error(err)
        else:
            st.sidebar.success(f"Connected to {MODEL_NAME}.")
    else:
        st.sidebar.info("Enter an API key to turn on Gemini briefings.")

st.sidebar.markdown("---")
if st.sidebar.button("Regenerate simulation"):
    if os.path.exists(SIM_FILE):
        os.remove(SIM_FILE)
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Showing: {S['name']}")

thr = st.session_state[K("thr")]
picked = st.session_state[K("picked")]
rng_dates = st.session_state[K("rng")]
if not isinstance(rng_dates, (tuple, list)) or len(rng_dates) != 2:
    rng_dates = (dmin, dmax)

if not picked:
    st.warning("Pick at least one stock in the sidebar to see results.")
    st.stop()

view = df[df["label"].isin(picked)
          & (df["date"] >= pd.Timestamp(rng_dates[0])) & (df["date"] <= pd.Timestamp(rng_dates[1]))]
if view.empty:
    st.warning("No data in this date range. Widen the range in the sidebar.")
    st.stop()

over = view[view["abs_gap"] >= thr]
flagged = over if st.session_state[K("dir")] == "Both" else over[over["side"] == st.session_state[K("dir")]]

# ================================ HERO ======================================

HERO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Figtree:wght@400;500;600&display=swap');
*{box-sizing:border-box} body{margin:0;font-family:'Figtree',sans-serif}
.hero{background:linear-gradient(120deg,#17205C 0%,#3B2FBF 55%,#6D3FE0 100%);border-radius:20px;padding:26px 30px 24px;color:#fff}
.chip{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.14);border-radius:999px;padding:4px 12px;font-size:13px;font-weight:500}
.dot{width:8px;height:8px;border-radius:50%;background:#5EF2B0;animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(94,242,176,.7)}70%{box-shadow:0 0 0 9px rgba(94,242,176,0)}100%{box-shadow:0 0 0 0 rgba(94,242,176,0)}}
h1{font-family:'Bricolage Grotesque',sans-serif;font-size:36px;margin:12px 0 4px;letter-spacing:-.02em}
p{margin:0;opacity:.82;font-size:15px;max-width:640px;line-height:1.45}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-top:22px}
.tile{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-radius:14px;padding:14px 16px;
 backdrop-filter:blur(6px);transition:transform .2s,background .2s}
.tile:hover{transform:translateY(-3px);background:rgba(255,255,255,.16)}
.lab{font-size:13px;opacity:.8}
.val{font-family:'Bricolage Grotesque',sans-serif;font-size:34px;font-weight:700;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.sub{font-size:12.5px;opacity:.85}
.up{color:#7CF2C0}.down{color:#FF9AA8}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="hero">
  <span class="chip"><span class="dot"></span>__CHIP__</span>
  <h1>ADR Spread Monitor</h1>
  <p>How far US-listed ADRs drift from their NSE share prices, and the days the gap becomes unusual.</p>
  <div class="tiles" id="tiles"></div>
</div>
<script>
const tiles = __TILES__;
const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = (n,t)=> (t.sign&&n>0?'+':'') + n.toLocaleString('en-US',{minimumFractionDigits:t.dec,maximumFractionDigits:t.dec}) + (t.suffix||'');
const root = document.getElementById('tiles');
tiles.forEach(t=>{
  const d=document.createElement('div'); d.className='tile';
  d.innerHTML=`<div class="lab">${t.label}</div><div class="val">0</div><div class="sub ${t.tone||''}">${t.sub}</div>`;
  root.appendChild(d);
  const el=d.querySelector('.val');
  if(calm){el.textContent=fmt(t.value,t);return;}
  const start=performance.now(), dur=1000;
  const step=now=>{const p=Math.min((now-start)/dur,1), e=1-Math.pow(1-p,3);
    el.textContent=fmt(t.value*e,t); if(p<1)requestAnimationFrame(step);};
  requestAnimationFrame(step);
});
</script>
"""

widest = view.loc[view["abs_gap"].idxmax()]
share = len(flagged) / max(len(view), 1) * 100
tiles = [
    {"label": "Trading days", "value": int(view["date"].nunique()), "dec": 0,
     "sub": f"{rng_dates[0]:%b %Y} to {rng_dates[1]:%b %Y}"},
    {"label": "Flagged days", "value": int(len(flagged)), "dec": 0,
     "sub": f"{share:.1f}% of stock-days at {thr:g}% or more"},
    {"label": "Average gap size", "value": round(float(view["abs_gap"].mean()), 2), "dec": 2, "suffix": "%",
     "sub": "typical distance from NSE price"},
    {"label": "Widest gap", "value": float(widest["price_gap_pct"]), "dec": 2, "suffix": "%", "sign": True,
     "sub": f"{widest['label']}, {widest['date']:%d %b %Y}",
     "tone": "up" if widest["price_gap_pct"] > 0 else "down"},
]
chip = "Synthetic data, not live prices"
components.html(HERO.replace("__TILES__", json.dumps(tiles)).replace("__CHIP__", chip), height=335)

if load_msg:
    st.warning(load_msg)
median_gap = float(view["abs_gap"].median())
if median_gap > 10:
    st.markdown(
        f"<div class='notice'><b>Heads up about this data.</b> The typical gap here is {median_gap:.0f}%, and the "
        f"ADR and NSE prices in this file move independently of each other, so most days look flagged. Real ADRs "
        f"usually stay within a few percent. Switch to <b>Realistic simulation</b> in the sidebar for "
        f"meaningful flags, or raise the flag level.</div>", unsafe_allow_html=True)
    st.write("")

# ============================ CHART HELPERS =================================


def finish(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#EEF1F7", domain=False, tickColor="#EEF1F7",
                            labelColor=SLATE, titleColor=SLATE, labelFont="Figtree", titleFont="Figtree")
            .configure_legend(labelColor=SLATE, titleColor=SLATE, labelFont="Figtree")
            .configure(background="#FFFFFF"))


SIDE_COLORS = alt.Scale(domain=["ADR premium", "ADR discount"], range=[GREEN, RED])

# ================================= TABS =====================================

tab_over, tab_explore, tab_days, tab_brief, tab_data = st.tabs(
    ["Overview", "Stock explorer", "Flagged days", "Briefings", "Data and export"])

# ------------------------------- OVERVIEW -----------------------------------
with tab_over:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        with panel("bars"):
            st.markdown("### Which stocks get flagged most?")
            st.markdown("<div class='hint'>Hover a bar for the exact count.</div>", unsafe_allow_html=True)
            counts = flagged.groupby(["label", "side"]).size().reset_index(name="days")
            if counts.empty:
                st.info("No days cross the flag level. Lower it in the sidebar.")
            else:
                bars = alt.Chart(counts).mark_bar(cornerRadiusEnd=6, height=26).encode(
                    y=alt.Y("label:N", title=None, sort="-x"),
                    x=alt.X("days:Q", title="Flagged days"),
                    color=alt.Color("side:N", scale=SIDE_COLORS, title=None),
                    tooltip=[alt.Tooltip("label:N", title="Stock"), alt.Tooltip("side:N", title="Side"),
                             alt.Tooltip("days:Q", title="Days")]).properties(height=250)
                st.altair_chart(finish(bars), use_container_width=True, theme=None)
    with right:
        with panel("latest"):
            st.markdown("### Latest reading")
            st.markdown("<div class='hint'>Most recent day in your range.</div>", unsafe_allow_html=True)
            last = view.sort_values("date").groupby("label").tail(1).sort_values("abs_gap", ascending=False)
            html = ""
            for _, r in last.iterrows():
                cls = "up" if r["price_gap_pct"] > 0 else "down"
                tag = "<span class='flag'>flagged</span>" if r["abs_gap"] >= thr else ""
                html += (f"<div class='row'><div><b>{r['label']}</b>{tag}<small>{r['date']:%d %b %Y}</small></div>"
                         f"<span class='pill {cls}'>{r['price_gap_pct']:+.2f}%</span></div>")
            st.markdown(html, unsafe_allow_html=True)

    with panel("heat"):
        st.markdown("### When did gaps get wide?")
        st.markdown("<div class='hint'>Each square is one month. Darker means a bigger average gap.</div>",
                    unsafe_allow_html=True)
        heat = (view.assign(month=view["date"].dt.to_period("M").dt.to_timestamp())
                .groupby(["month", "label"], as_index=False)["abs_gap"].mean())
        hm = alt.Chart(heat).mark_rect(cornerRadius=3, stroke="#fff", strokeWidth=1.5).encode(
            x=alt.X("month:T", title=None, timeUnit="yearmonth"),
            y=alt.Y("label:N", title=None),
            color=alt.Color("abs_gap:Q", scale=alt.Scale(scheme="purples"), title="Avg gap (%)"),
            tooltip=[alt.Tooltip("label:N", title="Stock"),
                     alt.Tooltip("month:T", title="Month", format="%b %Y"),
                     alt.Tooltip("abs_gap:Q", title="Avg gap %", format=".2f")]).properties(height=220)
        st.altair_chart(finish(hm), use_container_width=True, theme=None)

# --------------------------- STOCK EXPLORER ---------------------------------
with tab_explore:
    c1, c2, c3 = st.columns([3, 2, 2])
    stock_label = c1.radio("Stock", picked, horizontal=True)
    freq_name = c2.radio("Detail level", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True,
                         help="Weekly and monthly smooth out day-to-day noise.")
    show_marks = c3.toggle("Mark flagged points", value=True)

    sdf = view[view["label"] == stock_label].sort_values("date")
    key = sdf["stock_name"].iloc[0]
    meta = STOCKS.get(key, {})
    freq = {"Daily": None, "Weekly": "W", "Monthly": "MS"}[freq_name]
    series = (sdf[["date", "price_gap_pct"]].copy() if freq is None else
              sdf.set_index("date")["price_gap_pct"].resample(freq).mean().dropna().reset_index())
    series["price_gap_pct"] = series["price_gap_pct"].round(2)

    with panel("explore"):
        st.markdown(f"### {stock_label}: {freq_name.lower()} gap")
        st.markdown(f"<div class='hint'>{meta.get('us', '?')} on NYSE vs {meta.get('nse', '?')} on NSE "
                    f"(1 ADR = {meta.get('ratio', '?')} NSE shares). Drag to pan, scroll to zoom, "
                    f"double-click to reset.</div>", unsafe_allow_html=True)

        lo = min(series["price_gap_pct"].min(), -thr) * 1.15
        hi = max(series["price_gap_pct"].max(), thr) * 1.15
        ysc = alt.Scale(domain=[lo, hi])
        yenc = alt.Y("price_gap_pct:Q", title="Gap (%)", scale=ysc)
        base = alt.Chart(series).encode(x=alt.X("date:T", title=None))
        area = base.mark_area(
            interpolate="monotone",
            color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                               stops=[alt.GradientStop(color="rgba(79,70,229,0.02)", offset=0),
                                      alt.GradientStop(color="rgba(79,70,229,0.28)", offset=1)])
        ).encode(y=yenc)
        line = base.mark_line(color=INDIGO, strokeWidth=2.2, interpolate="monotone").encode(y=yenc)

        nearest = alt.selection_point(nearest=True, on="mouseover", fields=["date"], empty=False)
        hover = base.mark_rule(color=SLATE, strokeWidth=1).encode(
            opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("price_gap_pct:Q", title="Gap %", format="+.2f")]).add_params(nearest)
        dot = base.mark_circle(size=110, color=INDIGO, stroke="#fff", strokeWidth=2).encode(
            y=yenc, opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

        guides = alt.Chart(pd.DataFrame({"y": [thr, -thr]})).mark_rule(
            color=AMBER, strokeDash=[6, 4], strokeWidth=1.5).encode(y=alt.Y("y:Q", scale=ysc))
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#C9D0E4").encode(y=alt.Y("y:Q", scale=ysc))

        layers = [area, zero, guides, line]
        if show_marks and len(series) <= 600:        # markers get noisy on very long daily series
            layers.append(base.transform_filter(
                (alt.datum.price_gap_pct >= thr) | (alt.datum.price_gap_pct <= -thr)
            ).mark_circle(size=70, stroke="#fff", strokeWidth=1.5).encode(
                y=yenc, color=alt.condition(alt.datum.price_gap_pct > 0, alt.value(GREEN), alt.value(RED))))
        layers += [hover, dot]
        chart = alt.layer(*layers).properties(height=340).interactive(bind_y=False)
        st.altair_chart(finish(chart), use_container_width=True, theme=None)
        st.caption("Amber dashed lines are your flag level. Green dots: ADR premium. Red dots: ADR discount. "
                   "Flag dots are hidden on long daily views; choose Weekly or Monthly to see them.")

    with panel("stats"):
        a, b, c, d = st.columns(4)
        a.metric("Days tracked", f"{len(sdf):,}")
        b.metric("Days flagged", f"{int((sdf['abs_gap'] >= thr).sum()):,}")
        c.metric("Average gap size", f"{sdf['abs_gap'].mean():.2f}%")
        d.metric("Time at a premium", f"{(sdf['price_gap_pct'] > 0).mean() * 100:.0f}%")

    with panel("hist"):
        st.markdown("### How common is each gap size?")
        st.markdown("<div class='hint'>Bars outside the amber flag level are the days you would flag.</div>",
                    unsafe_allow_html=True)
        span = float(sdf["price_gap_pct"].max() - sdf["price_gap_pct"].min())
        step = max(0.25, round(span / 60, 2))
        hist = alt.Chart(sdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("price_gap_pct:Q", bin=alt.Bin(step=step), title="Gap (%)"),
            y=alt.Y("count():Q", title="Days"),
            color=alt.condition((alt.datum.price_gap_pct >= thr) | (alt.datum.price_gap_pct <= -thr),
                                alt.value(AMBER), alt.value("#A5ABF5")),
            tooltip=[alt.Tooltip("count():Q", title="Days")]).properties(height=200)
        st.altair_chart(finish(hist), use_container_width=True, theme=None)

# ----------------------------- FLAGGED DAYS ---------------------------------
with tab_days:
    if flagged.empty:
        st.info("No days cross the flag level with these filters. Lower it or widen the date range.")
    else:
        f1, f2 = st.columns([2, 3])
        order = f1.radio("Sort by", ["Largest gap", "Most recent"], horizontal=True)
        top_n = (f2.slider("How many days to show", 5, min(100, len(flagged)), min(25, len(flagged)))
                 if len(flagged) > 5 else len(flagged))

        ev = (flagged.sort_values("abs_gap", ascending=False) if order == "Largest gap"
              else flagged.sort_values("date", ascending=False)).head(top_n).reset_index(drop=True)

        with panel("table"):
            st.markdown(f"### {len(ev)} flagged days")
            show = pd.DataFrame({
                "Date": ev["date"].dt.strftime("%d %b %Y"), "Stock": ev["label"],
                "Gap (%)": ev["price_gap_pct"], "Gap size": ev["abs_gap"], "Side": ev["side"]})
            st.dataframe(show, hide_index=True, use_container_width=True, column_config={
                "Gap (%)": st.column_config.NumberColumn(format="%+.2f"),
                "Gap size": st.column_config.ProgressColumn(
                    min_value=0, max_value=float(ev["abs_gap"].max()), format="%.2f"),
            })

        st.caption("To read or write a market briefing for a day, open the Briefings tab.")

# ------------------------------- BRIEFINGS ----------------------------------
with tab_brief:
    if flagged.empty:
        st.info("No days cross the flag level with these filters. Lower it or widen the date range.")
    else:
        if brief_mode.startswith("Gemini"):
            st.markdown("<div class='hint'>Gemini briefings are on. Pick a day, then press the button. "
                        "Use the box at the bottom to brief several days at once.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='hint'>Showing standard briefings. Choose Gemini 3.6 Flash under "
                        "Market briefings in the sidebar for AI-written ones.</div>",
                        unsafe_allow_html=True)
        g1, g2 = st.columns([2, 3])
        b_order = g1.radio("Sort by", ["Largest gap", "Most recent"], horizontal=True, key="brief_order")
        b_top = (g2.slider("How many days to choose from", 5, min(100, len(flagged)),
                           min(25, len(flagged)), key="brief_top")
                 if len(flagged) > 5 else len(flagged))

        ev = (flagged.sort_values("abs_gap", ascending=False) if b_order == "Largest gap"
              else flagged.sort_values("date", ascending=False)).head(b_top).reset_index(drop=True)

        with panel("inspect"):
            st.markdown("### Inspect one day")
            pick = st.selectbox(
                "Choose a day", list(ev.index),
                format_func=lambda i: f"{ev.loc[i, 'date']:%d %b %Y}  |  {ev.loc[i, 'label']}  |  "
                                      f"{ev.loc[i, 'price_gap_pct']:+.2f}%")
            r = ev.loc[pick]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ADR price", f"${r['us_adr_price_usd']:,.2f}")
            m2.metric("USD/INR", f"{r['usdinr']:.2f}")
            m3.metric("ADR in rupees", f"₹{r['indian_eq_price']:,.2f}",
                      help=f"ADR price × USD/INR ÷ {int(r['adr_ratio'])} shares per ADR")
            m4.metric("NSE close", f"₹{r['nse_close_inr']:,.2f}",
                      delta=f"{r['price_gap_inr']:+,.2f} gap", delta_color="off")

            st.markdown("**Market briefing**")
            day_key = f"{r['stock_name']}|{r['date']:%Y-%m-%d}|{r['price_gap_pct']:.2f}"
            briefs = st.session_state.setdefault("briefs", {})

            if not brief_mode.startswith("Gemini"):
                st.markdown(f"<div class='callout'>{explain(r['price_gap_pct'], thr)}</div>",
                            unsafe_allow_html=True)
                st.caption("Standard briefing, written from fixed rules. Choose Gemini 3.6 Flash under "
                           "Market briefings in the sidebar for a written analysis.")
            else:
                if st.button("Write Gemini briefing", type="primary", key="gen_one"):
                    if not client:
                        st.warning("Gemini is not connected. Add your API key in the sidebar.")
                    else:
                        with st.spinner("Asking Gemini 3.6 Flash. Busy servers are retried automatically..."):
                            try:
                                briefs[day_key] = gemini_briefing(client, build_prompt(
                                    r["label"], f"{r['date']:%Y-%m-%d}", r["price_gap_pct"],
                                    r["us_adr_price_usd"], r["nse_close_inr"], r["usdinr"], r["adr_ratio"]))
                            except Exception as e:
                                st.error(f"Gemini could not write this briefing ({type(e).__name__}): {e}")
                if day_key in briefs:
                    st.markdown(render_brief(briefs[day_key]), unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='callout'>{explain(r['price_gap_pct'], thr)}</div>",
                                unsafe_allow_html=True)
                    st.caption("This is the standard explanation. Press the button for a Gemini briefing.")

                with st.expander("Write briefings for several days at once"):
                    st.caption(f"Makes one Gemini call per day, pausing {API_CALL_DELAY_SEC:g} second between "
                               "calls. Days you already briefed are reused, not called again.")
                    n_bulk = st.number_input("How many of the days listed above", 1, min(30, len(ev)),
                                             min(5, len(ev)))
                    if st.button("Run bulk briefings", key="gen_bulk"):
                        if not client:
                            st.error("Gemini is not connected. Add your API key in the sidebar.")
                        else:
                            bar, status, fails, rows = st.progress(0.0), st.empty(), 0, []
                            batch = ev.head(int(n_bulk))
                            for i, (_, b) in enumerate(batch.iterrows()):
                                k2 = f"{b['stock_name']}|{b['date']:%Y-%m-%d}|{b['price_gap_pct']:.2f}"
                                status.text(f"Day {i + 1} of {len(batch)}: {b['label']}, {b['date']:%d %b %Y}")
                                if k2 not in briefs:
                                    try:
                                        briefs[k2] = gemini_briefing(client, build_prompt(
                                            b["label"], f"{b['date']:%Y-%m-%d}", b["price_gap_pct"],
                                            b["us_adr_price_usd"], b["nse_close_inr"], b["usdinr"],
                                            b["adr_ratio"]))
                                        fails = 0
                                        time.sleep(API_CALL_DELAY_SEC)
                                    except Exception:
                                        fails += 1
                                        if fails >= MAX_CONSECUTIVE_FAILURES:
                                            status.warning(f"{fails} failures in a row. Stopped early.")
                                            break
                                if k2 in briefs:
                                    rows.append({"Date": f"{b['date']:%d %b %Y}", "Stock": b["label"],
                                                 "Gap (%)": b["price_gap_pct"], "Briefing": briefs[k2]})
                                bar.progress((i + 1) / len(batch))
                            if rows:
                                status.text(f"Finished. {len(rows)} briefings ready.")
                                out = pd.DataFrame(rows)
                                st.dataframe(out, hide_index=True, use_container_width=True)
                                st.download_button("Download briefings (CSV)",
                                                   out.to_csv(index=False).encode("utf-8"),
                                                   file_name="gemini_briefings.csv", mime="text/csv")

# ------------------------------ DATA / EXPORT -------------------------------
with tab_data:
    with panel("data"):
        st.markdown("### Your filtered data")
        st.markdown(f"<div class='hint'>{len(view):,} rows after filters.</div>", unsafe_allow_html=True)
        cols = ["date", "label", "us_adr_price_usd", "usdinr", "indian_eq_price", "nse_close_inr",
                "price_gap_inr", "price_gap_pct"]
        names = {"date": "Date", "label": "Stock", "us_adr_price_usd": "ADR (USD)", "usdinr": "USD/INR",
                 "indian_eq_price": "ADR in INR", "nse_close_inr": "NSE close (INR)",
                 "price_gap_inr": "Gap (INR)", "price_gap_pct": "Gap (%)"}
        table = view[cols].rename(columns=names)
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(table, hide_index=True, use_container_width=True, height=380)
        ftable = flagged[cols].rename(columns=names)
        ftable["Date"] = ftable["Date"].dt.strftime("%Y-%m-%d")
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Download filtered data (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="adr_gap_data.csv", mime="text/csv")
        d2.download_button("Download flagged days (CSV)", ftable.to_csv(index=False).encode("utf-8"),
                           file_name="adr_flagged_days.csv", mime="text/csv")

st.caption("Prices are synthetic, not live market data. A day is flagged when the FX-adjusted "
           "ADR-implied price differs from the NSE close by at least your flag level. "
           "Gemini briefings are written by an AI model and are not investment advice.")