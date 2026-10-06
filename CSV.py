import os
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. API Key Initialization
# ---------------------------------------------------------------------------
# Set your key via environment variable or pass it directly as a string:
# client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------------------
# 2. Company Mapping & Conversion Ratios (7 Cross-Sector ADRs)
# ---------------------------------------------------------------------------
COMPANY_MAP = {
    "INFY": {
        "name": "Infosys",
        "nse_symbol": "INFY.NS",
        "adr_symbol": "INFY",
        "ratio": 1.0,
    },
    "HDB": {
        "name": "HDFC Bank",
        "nse_symbol": "HDFCBANK.NS",
        "adr_symbol": "HDB",
        "ratio": 3.0,
    },
    "WIT": {
        "name": "Wipro",
        "nse_symbol": "WIPRO.NS",
        "adr_symbol": "WIT",
        "ratio": 1.0,
    },
    "IBN": {
        "name": "ICICI Bank",
        "nse_symbol": "ICICIBANK.NS",
        "adr_symbol": "IBN",
        "ratio": 2.0,
    },
    "RDY": {
        "name": "Dr Reddy's Labs",
        "nse_symbol": "DRREDDY.NS",
        "adr_symbol": "RDY",
        "ratio": 1.0,
    },
}

FOREX_SYMBOL = "INR=X"


def extract_close_series(df, ticker_symbol):
    """Safely extracts a 1D Pandas Series for 'Close' regardless of yfinance multi-indexing."""
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            return df["Close"][ticker_symbol]
        elif ticker_symbol in df.columns.levels[0]:
            return df[ticker_symbol]["Close"]
    elif "Close" in df.columns:
        return df["Close"]
    return df.squeeze()


# ---------------------------------------------------------------------------
# 3. Data Processing Engine
# ---------------------------------------------------------------------------
def fetch_and_process_adr_gaps(start_date="2026-01-01", end_date="2026-09-12"):
    all_records = []

    print("Fetching USD/INR Forex Rates...")
    raw_forex = yf.download(
        FOREX_SYMBOL, start=start_date, end=end_date, progress=False
    )
    forex_series = extract_close_series(raw_forex, FOREX_SYMBOL)

    for key, info in COMPANY_MAP.items():
        print(f"Processing {info['name']} ({info['adr_symbol']})...")

        raw_nse = yf.download(
            info["nse_symbol"], start=start_date, end=end_date, progress=False
        )
        raw_adr = yf.download(
            info["adr_symbol"], start=start_date, end=end_date, progress=False
        )

        nse_close = extract_close_series(raw_nse, info["nse_symbol"])
        adr_close = extract_close_series(raw_adr, info["adr_symbol"])

        # Merge series into clean single-level DataFrame
        df = pd.DataFrame(
            {
                "nse_close_inr": nse_close,
                "adr_close_usd": adr_close,
                "usd_inr": forex_series,
            }
        ).dropna()

        # Ratio and Currency Conversion
        df["company_name"] = info["name"]
        df["adr_inr_equiv"] = (df["adr_close_usd"] * df["usd_inr"]) / info[
            "ratio"
        ]

        # Gap Computation
        df["price_gap_inr"] = df["adr_inr_equiv"] - df["nse_close_inr"]
        df["price_gap_pct"] = (
            df["price_gap_inr"] / df["nse_close_inr"]
        ) * 100

        # Regime Categorization
        df["regime"] = df["price_gap_pct"].apply(
            lambda x: "High Arbitrage Spread"
            if abs(x) >= 2.0
            else "Normal Alignment"
        )

        all_records.append(df.reset_index())

    return pd.concat(all_records, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Gemini 2.5 Flash GenAI Briefing Function
# ---------------------------------------------------------------------------
def generate_gap_briefing(
    company, date, adr_inr_equiv, nse_close_inr, gap_pct
):
    prompt = f"""
    You are a quantitative stock market sentiment analyst reviewing an overnight US ADR price gap event.
    
    Data Point:
    - Company: {company}
    - Date: {date}
    - US ADR Converted Price (INR): ₹{adr_inr_equiv:.2f}
    - Previous Indian NSE Close (INR): ₹{nse_close_inr:.2f}
    - Overnight Price Gap: {gap_pct:+.2f}%
    
    Task:
    Provide a concise 2-sentence morning market briefing explaining the sentiment behind this gap and what Indian investors should watch for at the 9:15 AM IST market open.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text.strip()
    except Exception as e:
        return f"Briefing unavailable: {str(e)}"


# ---------------------------------------------------------------------------
# 5. Execution Pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_market = fetch_and_process_adr_gaps()

    # AI Enrichment for top gap instances (|Gap| >= 1.5%)
    high_gaps = df_market[df_market["price_gap_pct"].abs() >= 1.5].head(3)

    print("\n--- SAMPLE GENERATIVE AI BRIEFING ---")
    for idx, row in high_gaps.iterrows():
        briefing = generate_gap_briefing(
            row["company_name"],
            pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
            row["adr_inr_equiv"],
            row["nse_close_inr"],
            row["price_gap_pct"],
        )
        print(
            f"\n[{pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}] {row['company_name']} ({row['price_gap_pct']:+.2f}%):"
        )
        print(briefing)

    # Save to CSV
    output_path = "us_india_stock_gap_ai.csv"
    df_market.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated and exported dataset to '{output_path}'.")