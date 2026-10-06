"""
ADR Spread Monitor — Streamlit dashboard
------------------------------------------
Visualizes the US-India cross-border ADR/NSE price gap dataset and generates
AI market briefings for flagged events, either offline (deterministic, no
network) or online via a live Gemini API call.

Run:
    pip install streamlit pandas numpy google-genai
    streamlit run streamlit_dashboard.py


"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ============================== CONFIG =====================================

STOCKS = {
    "HDFC_BANK":   {"us": "HDB",  "nse": "HDFC.NS",      "ratio": 3},
    "INFOSYS":     {"us": "INFY", "nse": "INFY.NS",      "ratio": 1},
    "WIPRO":       {"us": "WIT",  "nse": "WIPRO.NS",     "ratio": 1},
    "ICICI_BANK":  {"us": "IBN",  "nse": "ICICIBANK.NS", "ratio": 2},
    "TATA_MOTORS": {"us": "TTM",  "nse": "TATAMOTORS.NS", "ratio": 5},
}

HISTORY_DAYS = 1650
DEFAULT_THRESHOLD = 1.5
DATA_FILE = "us_india_stock_gap_ai.csv"

API_CALL_DELAY_SEC = 1.0
MAX_CONSECUTIVE_FAILURES = 5

st.set_page_config(
    page_title="ADR Spread Monitor",
    layout="wide",
    page_icon="📊"
)


# ============================== DATA LOADING / GENERATION ==================

def _offline_briefing(stock_name: str, gap_pct: float) -> str:

    direction = "premium" if gap_pct > 0 else "discount"

    return (
        f"Offline Insight: {stock_name} ADR-implied price shows a "
        f"{abs(gap_pct):.2f}% {direction} versus its NSE close, "
        f"consistent with overnight US trading flow and FX movement."
    )


@st.cache_data(show_spinner="Generating synthetic dataset...")
def generate_dataset() -> pd.DataFrame:

    end_date = datetime.today()

    start_date = (
        end_date -
        timedelta(days=HISTORY_DAYS)
    )

    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    np.random.seed(42)

    records = []

    for key, meta in STOCKS.items():

        base_us = np.random.uniform(25, 120)

        base_nse = (
            base_us *
            85.0 /
            meta["ratio"]
        )

        us_prices = (
            base_us *
            np.cumprod(
                1 +
                np.random.normal(
                    0.0003,
                    0.015,
                    len(date_range)
                )
            )
        )

        nse_prices = (
            base_nse *
            np.cumprod(
                1 +
                np.random.normal(
                    0.0003,
                    0.014,
                    len(date_range)
                )
            )
        )

        usdinr_series = (
            82.0 +
            np.cumsum(
                np.random.normal(
                    0.001,
                    0.04,
                    len(date_range)
                )
            )
        )

        for i, d in enumerate(date_range):

            us_p = us_prices[i]
            nse_p = nse_prices[i]
            fx = usdinr_series[i]

            # Convert ADR price to equivalent Indian-share price
            eq_price = (
                us_p *
                fx
            ) / meta["ratio"]

            gap_inr = (
                eq_price -
                nse_p
            )

            gap_pct = (
                gap_inr /
                nse_p
            ) * 100

            briefing = "Normal market alignment."

            sentiment = 0.0

            source = "n/a"

            if abs(gap_pct) >= DEFAULT_THRESHOLD:

                sentiment = round(
                    np.random.uniform(
                        -7.0,
                        7.0
                    ),
                    1
                )

                briefing = _offline_briefing(
                    key,
                    gap_pct
                )

                source = "offline"

            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "stock_name": key,
                "us_ticker": meta["us"],
                "nse_ticker": meta["nse"],
                "us_adr_price_usd": round(us_p, 2),
                "nse_close_inr": round(nse_p, 2),
                "usdinr": round(fx, 2),
                "adr_ratio": meta["ratio"],
                "indian_eq_price": round(eq_price, 2),
                "price_gap_inr": round(gap_inr, 2),
                "price_gap_pct": round(gap_pct, 2),
                "ai_sentiment_score": sentiment,
                "ai_briefing_source": source,
                "ai_morning_briefing": briefing,
            })

    df = pd.DataFrame(records)

    df.to_csv(
        DATA_FILE,
        index=False
    )

    return df


@st.cache_data
def load_data() -> pd.DataFrame:

    if os.path.exists(DATA_FILE):

        df = pd.read_csv(DATA_FILE)

        # Required columns
        required_columns = [
            "date",
            "stock_name",
            "us_ticker",
            "nse_ticker",
            "us_adr_price_usd",
            "nse_close_inr",
            "usdinr",
            "adr_ratio",
            "indian_eq_price",
            "price_gap_inr",
            "price_gap_pct",
            "ai_sentiment_score",
            "ai_briefing_source",
            "ai_morning_briefing",
        ]

        # Check for missing columns
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            st.warning(
                "Existing CSV is missing required columns: "
                f"{missing_columns}. Generating a fresh dataset."
            )

            return generate_dataset()

        return df

    return generate_dataset()


# ============================== GEMINI ======================================

def get_gemini_client(api_key: str):

    if not HAS_GENAI:

        return (
            None,
            "google-genai package not installed. "
            "Run: pip install google-genai"
        )

    if not api_key:

        return (
            None,
            "No API key provided."
        )

    try:

        client = genai.Client(
            api_key=api_key
        )

        return client, None

    except Exception as e:

        return (
            None,
            f"{type(e).__name__}: {e}"
        )


def live_briefing(
    client,
    model: str,
    stock_name: str,
    gap_pct: float,
    date: str,
    us_price: float,
    nse_price: float,
    fx_rate: float,
    adr_ratio: int
) -> str:

    # Calculate the ADR-implied Indian-share price
    fx_adjusted_us_price = (
        us_price *
        fx_rate
    ) / adr_ratio

    # Calculate absolute INR difference
    difference_inr = (
        fx_adjusted_us_price -
        nse_price
    )

    # Calculate percentage difference independently
    calculated_gap_pct = (
        difference_inr /
        nse_price
    ) * 100

    prompt = (
        f"You are an AI financial market analyst analyzing one "
        f"flagged ADR-NSE price anomaly for {stock_name} on {date}.\n\n"

        f"NUMERICAL DATA:\n"
        f"- US ADR price: ${us_price:.2f}\n"
        f"- USD/INR exchange rate: {fx_rate:.2f}\n"
        f"- ADR ratio: 1 ADR = {adr_ratio} NSE shares\n"
        f"- ADR-implied NSE price: INR {fx_adjusted_us_price:.2f}\n"
        f"- Actual NSE closing price: INR {nse_price:.2f}\n"
        f"- Absolute price difference: INR {difference_inr:+.2f}\n"
        f"- Price gap: {gap_pct:+.2f}%\n"
        f"- Independently calculated gap: {calculated_gap_pct:+.2f}%\n\n"

        f"Your task is to analyze this specific event.\n\n"

        f"First, explain the numerical difference between the "
        f"ADR-implied price and the actual NSE price.\n\n"

        f"Then provide a plausible market-based explanation for "
        f"why this spread could exist. Consider factors such as:\n"
        f"- overnight US trading\n"
        f"- USD/INR exchange-rate movement\n"
        f"- liquidity differences\n"
        f"- market timing\n"
        f"- company-specific news\n"
        f"- regulatory restrictions\n"
        f"- settlement differences\n"
        f"- short-selling or borrowing constraints\n"
        f"- cross-border arbitrage frictions\n\n"

        f"IMPORTANT RULES:\n"
        f"1. Do not invent additional numerical data.\n"
        f"2. Use the supplied numbers in your numerical analysis.\n"
        f"3. Clearly distinguish observed data from possible explanations.\n"
        f"4. Do not claim that a regulatory factor definitely caused "
        f"the gap unless the supplied data proves it.\n"
        f"5. Treat market explanations as plausible interpretations, "
        f"not proof of causation.\n"
        f"6. Do not assume that a large theoretical spread means "
        f"guaranteed profit.\n\n"

        f"Return the answer in exactly this structure:\n\n"

        f"Numerical Analysis:\n"
        f"State the ADR-implied price, NSE price, absolute INR "
        f"difference, and percentage gap.\n\n"

        f"Market Interpretation:\n"
        f"Give 2-3 concise sentences explaining plausible reasons "
        f"for the observed divergence.\n\n"

        f"Arbitrage View:\n"
        f"State whether the gap represents a theoretical arbitrage "
        f"signal, while explaining that transaction costs, borrowing, "
        f"FX risk, settlement, liquidity, and regulatory constraints "
        f"may prevent the theoretical spread from being captured."
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    text = (
        response.text or ""
    ).strip()

    if not text:

        raise ValueError(
            "Empty response from Gemini API"
        )

    return (
        f"Gemini Live ({model}):\n\n"
        f"{text}"
    )


# ============================== SIDEBAR =====================================

st.sidebar.title("Controls")


mode = st.sidebar.radio(
    "Briefing mode",
    [
        "Offline (simulated)",
        "Online (live Gemini)"
    ],
    index=0
)

online = mode.startswith("Online")


api_key = ""

model = "gemini-2.5-flash"

client = None

client_error = None


if online:

    api_key = st.sidebar.text_input(
        "Gemini API key",
        type="password",
        help=(
            "Kept only in this session's memory, "
            "never written to disk."
        )
    )

    model = st.sidebar.selectbox(
        "Model",
        [
            "gemini-3.6-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash"
        ],
        index=0,
        help=(
            "Google periodically retires older models. "
            "If a call fails with a 404 NOT_FOUND error, "
            "use a currently available model."
        )
    )

    custom_model = st.sidebar.text_input(
        "Or type a model name manually",
        value="",
        placeholder="e.g. gemini-3.6-flash"
    )

    if custom_model.strip():

        model = custom_model.strip()

    # Create client BEFORE trying to list models
    if api_key:

        client, client_error = get_gemini_client(
            api_key
        )

        if client_error:

            st.sidebar.error(
                f"Client error: {client_error}"
            )

        else:

            st.sidebar.success(
                "Gemini client ready."
            )

    else:

        st.sidebar.info(
            "Enter a key to enable live calls."
        )

    # Model listing button
    if st.sidebar.button(
        "List models available to my key"
    ):

        if client:

            try:

                names = [
                    m.name.replace(
                        "models/",
                        ""
                    )
                    for m in client.models.list()
                ]

                st.sidebar.success(
                    f"{len(names)} models available:"
                )

                st.sidebar.code(
                    "\n".join(
                        sorted(names)
                    )
                )

            except Exception as e:

                st.sidebar.error(
                    f"Couldn't list models: "
                    f"{type(e).__name__}: {e}"
                )

        else:

            st.sidebar.warning(
                "Enter a valid API key first."
            )


threshold = st.sidebar.slider(
    "Gap threshold for flagging (%)",
    0.5,
    5.0,
    DEFAULT_THRESHOLD,
    0.1
)


if st.sidebar.button(
    "🔄 Regenerate synthetic dataset"
):

    st.cache_data.clear()

    st.rerun()


st.sidebar.caption(
    f"Data source: `{DATA_FILE}`"
    if os.path.exists(DATA_FILE)
    else
    "Freshly generated dataset."
)


# ============================== LOAD DATA ===================================

df = load_data()


# Convert date column to datetime
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# Check for invalid dates
if df["date"].isna().any():

    st.error(
        "Invalid date values found in the dataset."
    )

    st.stop()


stocks = sorted(
    df["stock_name"].unique()
)


# ============================== HEADER / KPIs ===============================

st.title(
    "ADR Spread Monitor"
)

st.caption(
    "US–India cross-border ADR/NSE price gap tracker "
    "with AI-generated market briefings."
)


flagged = df[
    df["price_gap_pct"].abs() >= threshold
]


widest = df.loc[
    df["price_gap_pct"].abs().idxmax()
]


k1, k2, k3, k4 = st.columns(4)


k1.metric(
    "Rows in dataset",
    f"{len(df):,}"
)


k2.metric(
    "Threshold breaches",
    f"{len(flagged):,}"
)


k3.metric(
    "Avg. absolute gap",
    f"{df['price_gap_pct'].abs().mean():.2f}%"
)


k4.metric(
    "Widest single gap",
    f"{widest['price_gap_pct']:+.2f}%",
    delta=(
        f"{widest['stock_name']} · "
        f"{widest['date'].strftime('%Y-%m-%d')}"
    ),
    delta_color="off"
)


st.divider()


# ============================== TABS PER STOCK ==============================

tabs = st.tabs(
    [
        s.replace("_", " ")
        for s in stocks
    ]
)


for tab, stock in zip(
    tabs,
    stocks
):

    with tab:

        sdf = df[
            df["stock_name"] == stock
        ].copy()

        sdf["date"] = pd.to_datetime(
            sdf["date"],
            errors="coerce"
        )

        sdf = sdf.sort_values(
            "date"
        )


        meta = STOCKS.get(
            stock,
            {}
        )


        st.caption(
            f"{meta.get('us', '?')} (US) / "
            f"{meta.get('nse', '?')} (NSE) · "
            f"ADR ratio {meta.get('ratio', '?')}:1 · "
            f"{len(sdf):,} trading days"
        )


        # ================= WEEKLY CHART =================

        weekly = (
            sdf
            .set_index("date")[
                "price_gap_pct"
            ]
            .resample("W")
            .mean()
        )


        st.line_chart(
            weekly,
            height=280,
            use_container_width=True
        )


        # ================= FLAGGED EVENTS =================

        events = sdf[
            sdf["price_gap_pct"].abs() >= threshold
        ].copy()


        events = (
            events
            .sort_values(
                "price_gap_pct",
                key=lambda s: s.abs(),
                ascending=False
            )
            .head(15)
        )


        st.subheader(
            f"Largest gaps "
            f"(top {len(events)}, "
            f"threshold ≥ {threshold}%)"
        )


        if events.empty:

            st.info(
                "No events crossed this threshold "
                "for this stock."
            )

        else:

            display_cols = [
                "date",
                "price_gap_pct",
                "ai_sentiment_score",
                "ai_briefing_source",
                "ai_morning_briefing"
            ]


            styled = (
                events[display_cols]
                .rename(
                    columns={
                        "date": "Date",
                        "price_gap_pct": "Gap %",
                        "ai_sentiment_score": "Sentiment",
                        "ai_briefing_source": "Source",
                        "ai_morning_briefing": "Briefing"
                    }
                )
            )


            def color_gap(val):

                if isinstance(
                    val,
                    (int, float)
                ):

                    color = (
                        "#3FD68C"
                        if val > 0
                        else "#F2545B"
                    )

                    return f"color: {color}"

                return ""


            st.dataframe(
                styled.style
                .applymap(
                    color_gap,
                    subset=["Gap %"]
                )
                .format(
                    {
                        "Gap %": "{:+.2f}%",
                        "Date": lambda d:
                            d.strftime("%Y-%m-%d")
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            # ================= LIVE BRIEFING =================

            st.markdown(
                "**Get a live AI analysis of one event**"
            )


            event_dates = (
                events["date"]
                .dt
                .strftime("%Y-%m-%d")
                .tolist()
            )


            chosen_date = st.selectbox(
                "Event date",
                event_dates,
                key=f"date_{stock}"
            )


            chosen_row = events[
                events["date"]
                .dt
                .strftime("%Y-%m-%d")
                == chosen_date
            ].iloc[0]


            col_a, col_b = st.columns(
                [1, 3]
            )


            with col_a:

                generate = st.button(
                    "Generate briefing",
                    key=f"gen_{stock}"
                )


            with col_b:

                result_box = st.empty()


            if generate:

                if online and client:

                    with st.spinner(
                        "Calling Gemini..."
                    ):

                        try:

                            text = live_briefing(
                                client,
                                model,
                                stock,
                                chosen_row[
                                    "price_gap_pct"
                                ],
                                chosen_date,

                                us_price=chosen_row[
                                    "us_adr_price_usd"
                                ],

                                nse_price=chosen_row[
                                    "nse_close_inr"
                                ],

                                fx_rate=chosen_row[
                                    "usdinr"
                                ],

                                adr_ratio=STOCKS[
                                    stock
                                ]["ratio"]
                            )


                            result_box.success(
                                text
                            )


                        except Exception as e:

                            result_box.error(
                                f"Live call failed: "
                                f"{type(e).__name__}: {e}\n\n"
                                f"Falling back to offline briefing: "
                                f"{chosen_row['ai_morning_briefing']}"
                            )


                elif online and not client:

                    result_box.warning(
                        "No working Gemini client. "
                        "Check your API key and model."
                    )


                else:

                    result_box.info(
                        chosen_row[
                            "ai_morning_briefing"
                        ]
                    )


            # ================= BULK GEMINI =================

            with st.expander(
                "Bulk-regenerate live briefings for all events above"
            ):

                st.caption(
                    "Calls the Gemini API once per row shown "
                    "in the table, with a 1-second delay between "
                    "calls to stay under free-tier rate limits."
                )


                if st.button(
                    f"Run bulk regeneration for {stock}",
                    key=f"bulk_{stock}"
                ):

                    if not (online and client):

                        st.error(
                            "Switch to Online mode and enter "
                            "a valid API key first."
                        )

                    else:

                        progress = st.progress(
                            0.0
                        )

                        status = st.empty()

                        results = []

                        failures = 0


                        for i, (_, row) in enumerate(
                            events.iterrows()
                        ):

                            date_str = (
                                row["date"]
                                .strftime("%Y-%m-%d")
                            )


                            status.text(
                                f"Processing {date_str} "
                                f"({i + 1}/{len(events)})..."
                            )


                            try:

                                text = live_briefing(
                                    client,
                                    model,
                                    stock,
                                    row[
                                        "price_gap_pct"
                                    ],
                                    date_str,

                                    us_price=row[
                                        "us_adr_price_usd"
                                    ],

                                    nse_price=row[
                                        "nse_close_inr"
                                    ],

                                    fx_rate=row[
                                        "usdinr"
                                    ],

                                    adr_ratio=STOCKS[
                                        stock
                                    ]["ratio"]
                                )


                                results.append(
                                    (
                                        date_str,
                                        "live",
                                        text
                                    )
                                )


                                failures = 0


                            except Exception:

                                failures += 1


                                results.append(
                                    (
                                        date_str,
                                        "offline (call failed)",
                                        row[
                                            "ai_morning_briefing"
                                        ]
                                    )
                                )


                                if (
                                    failures
                                    >= MAX_CONSECUTIVE_FAILURES
                                ):

                                    status.warning(
                                        f"{failures} consecutive "
                                        f"failures — stopping early."
                                    )

                                    break


                            progress.progress(
                                (i + 1) /
                                len(events)
                            )


                            time.sleep(
                                API_CALL_DELAY_SEC
                            )


                        status.text(
                            "Done."
                        )


                        st.dataframe(
                            pd.DataFrame(
                                results,
                                columns=[
                                    "Date",
                                    "Source",
                                    "Briefing"
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True
                        )


        # ================= CSV EXPORT =================

        csv_bytes = (
            sdf
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            f"Download {stock} data as CSV",
            csv_bytes,
            file_name=(
                f"{stock.lower()}_gap_data.csv"
            ),
            mime="text/csv"
        )


# ============================== FOOTER ======================================

st.divider()


st.caption(
    "Prices are a synthetic simulation, not live market data. "
    "A gap is flagged when the FX-adjusted ADR-implied price "
    "diverges from the NSE close by the threshold set in the sidebar."
)