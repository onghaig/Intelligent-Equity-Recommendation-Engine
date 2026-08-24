import os
import json
import pickle
import argparse
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from yahooquery import Ticker as YQTicker
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import logging

# Set up basic logging for warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Suppress specific noisy warnings from pandas or sklearn, instead of a sweeping global ignore
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# The universe of 85 stocks we will analyze and train the ML model on, 
# categorized by their respective GICS economic sectors.
UNIVERSE = [
# --- Technology ---
"AAPL","MSFT","NVDA","GOOGL","META","ORCL","ADBE","CRM","INTC","AMD",
"CSCO","TXN","AVGO","QCOM","IBM",
# --- Communication Services ---
"DIS","NFLX","CMCSA","T","VZ",
# --- Consumer Discretionary ---
"AMZN","TSLA","HD","MCD","SBUX","NKE","LOW","BKNG","F",
# --- Consumer Staples ---
"PG","KO","PEP","WMT","COST","PM","MO","MDLZ",
# --- Financials ---
"JPM","BAC","WFC","C","GS","MS","BLK","AXP","SCHW",
# --- Healthcare ---
"JNJ","PFE","MRK","ABBV","LLY","UNH","TMO","DHR","BMY",
# --- Industrials ---
"CAT","BA","GE","HON","UPS","LMT","RTX","DE",
# --- Energy ---
"XOM","CVX","COP","SLB","EOG",
# --- Materials ---
"LIN","APD","SHW","FCX","NEM",
# --- Utilities ---
"NEE","DUK","SO","EXC",
# --- Real Estate ---
"PLD","AMT","O","SPG",
# --- Payments / FinTech ---
"V","MA","PYPL","SQ"
]

# Benchmark index (S&P 500) used for relative returns and calculating Beta
BENCH = "^GSPC"
# Local directory used for storing historical network fetches to speed up reruns
CACHE_DIR = "./cache"
ARTIFACTS_PATH = os.path.join(CACHE_DIR, "artifacts.pkl")

# --- Model Gating Constants ---
BUY_SCORE = 67
SELL_SCORE = 42
YHAT_BUY_FLOOR = -0.005
YHAT_SELL_CEILING = 0.0
SIGMOID_SCALE = 0.03
MISSINGNESS_HIGH = 0.35

def ensure_cache():
    """Ensures the local caching directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_prices(ticker, years=10.5):
    """
    Downloads daily historical 'Adjusted Close' prices using yfinance.
    Always looks first in CACHE_DIR before making a network request.
    """
    ensure_cache()
    # Define cache location based on the ticker name
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_px.csv")
    end = datetime.now()
    # Calculate start date by subtracting the required number of years
    start = end - timedelta(days=int(years*365.25))
    
    # 1. Attempt to load from cache
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df['Adj close'] if 'Adj close' in df.columns else df['Adj Close']
    
    # 2. If not cached, download from yfinance and save to cache
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        # Fix multi-index column formatting sometimes returned by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            df.to_csv(cache_path)
            # Return specifically the Adjusted Close price series
            return df['Adj close'] if 'Adj close' in df.columns else df['Adj Close']
    except ImportError:
        logging.error("yfinance is required but not installed.")
        raise
    except Exception as e:
        logging.warning(f"[{ticker}] Failed to fetch prices: {e}")
        
    return pd.Series(dtype=float)

# Standardized Dictionary of ESG Risk Scores (Lower is Better)
ESG_VALUES = {
    "AAPL": 17.2, "MSFT": 15.1, "NVDA": 13.6, "GOOGL": 24.2, "META": 34.1,
    "ORCL": 13.9, "ADBE": 13.1, "CRM": 14.9, "INTC": 18.9, "AMD": np.nan,
    "CSCO": 13.9, "TXN": 20.6, "AVGO": 20.0, "QCOM": 15.4, "IBM": 14.1,
    "DIS": 15.7, "NFLX": 16.4, "CMCSA": 23.1, "T": 23.9, "VZ": 18.7,
    "AMZN": 30.6, "TSLA": 25.2, "HD": 12.6, "MCD": 26.0, "SBUX": 24.7,
    "NKE": 19.6, "LOW": 11.8, "BKNG": 19.2, "F": 22.4, "PG": 28.6,
    "KO": 21.6, "PEP": 22.1, "WMT": 25.3, "COST": 23.3, "PM": 28.5,
    "MO": 31.3, "MDLZ": 22.0, "JPM": 29.3, "BAC": 28.3, "WFC": 36.2,
    "C": 29.2, "GS": 25.5, "MS": 24.6, "BLK": 18.3, "AXP": 18.6,
    "SCHW": 23.3, "JNJ": 24.0, "PFE": 24.6, "MRK": 21.4, "ABBV": 29.9,
    "LLY": 24.3, "UNH": 15.3, "TMO": 12.8, "DHR": 11.9, "BMY": 22.7,
    "CAT": 36.2, "BA": 39.6, "GE": 40.5, "HON": 28.6, "UPS": 18.6,
    "LMT": 30.2, "RTX": np.nan, "DE": 20.0, "XOM": 41.6, "CVX": 36.6,
    "COP": 33.9, "SLB": 20.3, "EOG": 34.2, "LIN": np.nan, "APD": 12.3,
    "SHW": 29.4, "FCX": 31.6, "NEM": 20.5, "NEE": 23.3, "DUK": 27.0,
    "SO": 33.0, "EXC": 22.9, "PLD": 10.3, "AMT": 10.9, "O": 15.5,
    "SPG": 14.0, "V": 16.7, "MA": 17.1, "PYPL": 17.8, "SQ": np.nan
}

def get_local_esg(ticker):
    """Retrieves the ESG score for a given ticker, handling alias routing."""
    # Map dual-class shares to their primary listing score where data exists
    aliases = {'GOOG': 'GOOGL', 'BRK.B': 'BRK-B', 'BRK-B': 'BRK.B', 'BF.B': 'BF-B'}
    search_ticker = aliases.get(ticker.upper(), ticker.upper())
    
    score =  ESG_VALUES.get(search_ticker, np.nan)
    if not pd.isna(score):
        # Format identical to what Yahoo's API used to return so downstream code doesn't break
        return {"totalEsg": float(score)}
    return {}

def fetch_fundamentals(ticker):
    """
    Downloads foundational fundamental data, accounting info, and ESG.
    Uses yahooquery for financial data and get_local_esg for ESG data.
    """
    ensure_cache()
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_fun.json")
    
    # 1. Attempt to load from JSON cache
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            data = json.load(f)
            # Patch in local ESG score retroactively if old cache is missing it
            if not data.get('esg_scores'):
                data['esg_scores'] = get_local_esg(ticker)
            return data
            
    # 2. On cache miss, fetch via yahooquery endpoints
    try:
        yq = YQTicker(ticker)
        
        # We need historical earnings to calculate 'earnings surprise'
        try:
            eh_df = yq.earning_history
            # Convert Pandas DF returned by yahooquery to list of dicts for JSON serialization
            eh = eh_df.to_dict('records') if hasattr(eh_df, 'to_dict') else []
        except:
            eh = []
            
        # Compile all underlying fundamental profiles
        data = {
            "asset_profile": yq.asset_profile.get(ticker, {}),
            "summary_detail": yq.summary_detail.get(ticker, {}),
            "key_statistics": yq.key_stats.get(ticker, {}),
            "financial_data": yq.financial_data.get(ticker, {}),
            "earning_history": eh,
            "esg_scores": get_local_esg(ticker)
        }
        
        # Clean string fallbacks from yahooquery to prevent downstream dictionary errors
        for k in data:
            if isinstance(data[k], str):
                data[k] = {}
        
        # Save payload to JSON cache
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return data
    except ImportError:
        logging.error("yahooquery is required but not installed.")
        raise
    except Exception as e:
        logging.warning(f"[{ticker}] Failed to fetch fundamentals: {e}")
        return {} 

def compute_features(ticker, fundamentals, px, spx):
    """
    Engineers 16 numerical factors some qualitative factors and ratios from the raw price/fundamental 
    payloads to be used in ML training and scorecard rank evaluation.
    """
    # Extract underlying data dictionaries
    asset_prof = fundamentals.get("asset_profile", {})
    summary = fundamentals.get("summary_detail", {})
    key_stats = fundamentals.get("key_statistics", {})
    financials = fundamentals.get("financial_data", {})

    # Extract ESG Score. Since we treat missing keys dynamically later, log if it exists.
    esg_data = fundamentals.get("esg_scores", {})
    esg_score = esg_data.get("totalEsg", np.nan)
    esg_available = 1 if not pd.isna(esg_score) else 0

    sector = asset_prof.get("sector", "Unknown")
    
    # Try fetching Beta from Yahoo. If missing, manually calculate it using 2y rolling daily covariance (industry standard).
    beta = key_stats.get("beta", summary.get("beta", np.nan))
    if pd.isna(beta) and len(px) > 500: # Ensure we have at least ~2 years of trading days
        cutoff = px.index[-1] - timedelta(days=2*365.25)
        px_2y = px.loc[cutoff:] # Slice stock price to last 2y
        spx_2y = spx.reindex(px_2y.index).ffill() # Align S&P500 dates
        df_merged = pd.DataFrame({'p': px_2y, 's': spx_2y}).dropna()
        if len(df_merged) > 0:
            r_px = df_merged['p'].pct_change().dropna() # Daily stock returns
            r_spx = df_merged['s'].pct_change().dropna() # Daily market returns
            var = r_spx.var()
            if var > 0:
                # Formula for Beta: Cov(Stock, Market) / Var(Market)
                beta = r_px.cov(r_spx) / var

    # Categorize market capitalization into buckets
    mc = summary.get("marketCap", np.nan)
    cb = "Mid"
    if not pd.isna(mc):
        if mc > 1e10: 
            cb = "Large"
        elif mc < 2e9: 
            cb = "Small"
        
    # Standard Valuation & Income Metrics
    div_yield = summary.get("dividendYield", np.nan)
    tpe = summary.get("trailingPE", np.nan)
    fpe = summary.get("forwardPE", np.nan)
    pb = key_stats.get("priceToBook", np.nan)
    rg = financials.get("revenueGrowth", np.nan)
    payout_ratio = summary.get("payoutRatio", np.nan)
    
    # Initialize Price-based metrics
    vol = np.nan
    mdd = np.nan
    mom = np.nan
    rel_mom = np.nan
    
    # Calculate 1-Year risk and momentum features
    if len(px) >= 252: # Requires at least 1 trading year of history
        px_1y = px.tail(252)
        spx_1y = spx.reindex(px_1y.index).ffill()
        rets = px_1y.pct_change().dropna()
        if len(rets) > 0:
            # Annualized volatility: daily standard deviation * sqrt(trading days)
            vol = rets.std() * np.sqrt(252)
        
        # Max Drawdown: Identify the peak-to-trough drop over the 1-year window
        roll_max = px_1y.cummax()
        drawdowns = (roll_max - px_1y) / roll_max
        if not drawdowns.empty:
            mdd = drawdowns.max() # Kept as a positive magnitude
        
        # 1Y Momentum (Stock Return minus Benchmark Return)
        mom = (px_1y.iloc[-1] / px_1y.iloc[0]) - 1
        spx_mom = (spx_1y.iloc[-1] / spx_1y.iloc[0]) - 1
        rel_mom = mom - spx_mom

    # Process Historical Earnings Surprises
    es = np.nan
    eh = fundamentals.get("earning_history", [])
    if isinstance(eh, list) and len(eh) > 0:
        # Collect recent surprise percentages
        surprises = [r.get('surprisePercent') for r in eh if r.get('surprisePercent') is not None and not pd.isna(r.get('surprisePercent'))]
        if surprises:
            # Output the mean surprise
            es = float(np.mean(surprises))

    # Basic combination of Value and Growth (Synthetic metric)
    gvv = np.nan
    if not pd.isna(div_yield) and not pd.isna(tpe) and not pd.isna(rg) and not pd.isna(vol):
        gvv = rg + vol - (1.0 / (tpe + 1e-5)) - div_yield

    # Compile the final parsed feature dictionary
    return {
        "sector": sector, "beta": beta, "market_cap": mc, "cap_bucket": cb,
        "esg_score": esg_score, "esg_available": esg_available,
        "growth_vs_value": gvv,
        "earnings_surprise": es,
        "trailing_pe": tpe, "forward_pe": fpe, "price_to_book": pb,
        "profit_margin": financials.get("profitMargins", np.nan),
        "operating_margin": financials.get("operatingMargins", np.nan),
        "revenue_growth": rg, "dividend_yield": div_yield,
        "payout_ratio": payout_ratio,
        "vol_1y": vol, "max_drawdown_1y": mdd, "momentum_12m": mom,
        "relative_momentum_12m": rel_mom
    }

def build_dataset(universe):
    """
    Loops through the universe list to compute features for every stock.
    Calculates the actual 10-year excess CAGR to use as the true Y target label.
    """
    spx_px = fetch_prices(BENCH, years=10.5)
    X_list, y_list, valid = [], [], []
    
    for t in universe:
        try:
            px = fetch_prices(t, years=10.5)
            fun = fetch_fundamentals(t)
            feats = compute_features(t, fun, px, spx_px)
            
            # Label calculations require a robust amount of history (at least 5 years)
            if len(px) > 252 * 5:
                # Find number of years recorded in series
                yrs = (px.index[-1] - px.index[0]).days / 365.25
                # Asset's compound annual growth
                cagr = (px.iloc[-1] / px.iloc[0])**(1/yrs) - 1
                
                spx_sub = spx_px.reindex(px.index).ffill()
                if len(spx_sub) > 0:
                    # Benchmark's compound annual growth
                    cagr_spx = (spx_sub.iloc[-1] / spx_sub.iloc[0])**(1/yrs) - 1
                    
                    # Store Excess CAGR as the "y" variable
                    y_list.append(cagr - cagr_spx)
                    feats['ticker'] = t
                    X_list.append(feats)
                    valid.append(t)
        except Exception as e:
            logging.warning(f"[{t}] Skipped building dataset: {e}")
            
    if not X_list:
        raise ValueError("No valid data found for building the dataset.")
        
    X_df = pd.DataFrame(X_list).set_index("ticker")
    y_series = pd.Series(y_list, index=valid)
    return X_df, y_series, {"dataset_date": str(datetime.now())}

def train_model(X_df, y):
    """
    Builds an ML Pipeline to predict Excess CAGR from Features `X_df`.
    Imputes NaN values, scales features, applies Ridge Regression, and pickles it.
    """
    # Grab only mathematical columns to pass to ML
    num_cols = X_df.select_dtypes(include=[float, int]).columns.tolist()
    
    # Drop categorical or meta variables that shouldn't impact ML
    for col in ['cap_bucket', 'esg_available']:
        if col in num_cols: num_cols.remove(col)
        
    # Create Imputation, Scaling, and Ridge penalty Machine Learning pipeline
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Treat NaN dynamically based on universe median
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    pipe.fit(X_df[num_cols], y)
    
    # Store validation names and dump serialized pipeline for live inference later
    valid_cols = pipe.named_steps['imputer'].get_feature_names_out(num_cols)
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump({'model': pipe, 'X_df': X_df, 'num_cols': num_cols, 'valid_cols': valid_cols}, f)
    return pipe

def load_model():
    """Loads `artifacts.pkl`. Auto-runs the dataset generation step if the artifacts do not exist."""
    if not os.path.exists(ARTIFACTS_PATH):
        X, y, m = build_dataset(UNIVERSE)
        train_model(X, y)
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)

def scorecard(features, client, sector):
    """
    Foundational heuristic rule-engine that assigns a 0-100 grade to a given stock.
    Sub-scores the stock into PM, RK, VL, QG, IN, ES categories based on percentiles.
    Applies custom index multipliers depending on target client or GICS sector.
    """
    try: X_df = load_model()['X_df']
    except: X_df = pd.DataFrame([features])
        
    def get_pct(val, col, invert=False):
        """
        Takes a specific feature scalar value, compares it across the training 
        universe distribution for that column, and returns its bounded 0-100 Percentile rank.
        Invert=True assigns higher scores to lower numeric values (like low valuation).
        """
        if pd.isna(val) or col not in X_df.columns: 
            return 50.0 # Return perfectly neutral 50/100 to avoid punishing missing dimensions
        ser = X_df[col].dropna()
        if len(ser) == 0: 
            return 50.0
            
        # Calculate exactly what fraction of the universe is lower/higher than this asset
        pct = (ser <= val).mean() * 100
        if invert:
            pct = (ser >= val).mean() * 100
        return max(0.0, min(100.0, float(pct)))

    # Price Momentum Subscore
    PM = np.mean([
        get_pct(features.get('momentum_12m', np.nan), 'momentum_12m'),
        get_pct(features.get('relative_momentum_12m', np.nan), 'relative_momentum_12m'),
        # Note: we invert drawdown here because a low positive magnitude equals a mild drawdown (GOOD)
        get_pct(features.get('max_drawdown_1y', np.nan), 'max_drawdown_1y', invert=True) 
    ])
    
    # Risk Profile Subscore
    RK = np.mean([
        # We invert Beta and Volatility since low risk/volatility equals a high score
        get_pct(features.get('beta', np.nan), 'beta', invert=True),
        get_pct(features.get('vol_1y', np.nan), 'vol_1y', invert=True),
        get_pct(features.get('max_drawdown_1y', np.nan), 'max_drawdown_1y', invert=True) 
    ])
    
    # Valuation Level Subscore (Invert all so low Multiples = high score)
    VL = np.mean([
        get_pct(features.get('trailing_pe', np.nan), 'trailing_pe', invert=True),
        get_pct(features.get('forward_pe', np.nan), 'forward_pe', invert=True),
        get_pct(features.get('price_to_book', np.nan), 'price_to_book', invert=True)
    ])
    
    # Quality & Growth Profile (Higher is better)
    QG = np.mean([
        get_pct(features.get('profit_margin', np.nan), 'profit_margin'),
        get_pct(features.get('operating_margin', np.nan), 'operating_margin'),
        get_pct(features.get('revenue_growth', np.nan), 'revenue_growth')
    ])
    
    # Dividend & Income Profile
    IN = np.mean([
        get_pct(features.get('dividend_yield', np.nan), 'dividend_yield'),
        get_pct(features.get('payout_ratio', np.nan), 'payout_ratio')
    ])
    
    # ESG Profile (Lower absolute ESG Risk Score is Better)
    ES = get_pct(features.get('esg_score', np.nan), 'esg_score', invert=True)

    st = sector.lower()

    # Pre-define weights depending on client profile target
    w = {'PM': 0.25, 'RK': 0.20, 'VL': 0.20, 'QG': 0.25, 'IN': 0.05, 'ES': 0.05}
    if client == 'income': w = {'PM': 0.15, 'RK': 0.25, 'VL': 0.15, 'QG': 0.10, 'IN': 0.30, 'ES': 0.05}
    elif client == 'growth': w = {'PM': 0.30, 'RK': 0.15, 'VL': 0.10, 'QG': 0.35, 'IN': 0.05, 'ES': 0.05}
    elif client == 'esg': w = {'PM': 0.20, 'RK': 0.15, 'VL': 0.15, 'QG': 0.20, 'IN': 0.05, 'ES': 0.25}

    # Intersect with sector reality (e.g. increase quality importance for tech)
    if 'technology' in st or 'communication' in st:
        w['QG'] *= 1.25
        w['VL'] *= 0.8
    elif 'financial' in st:
        # Banks often require Price-to-Book emphasis instead of PE
        vl1 = get_pct(features.get('trailing_pe', np.nan), 'trailing_pe', invert=True)
        vl2 = get_pct(features.get('forward_pe', np.nan), 'forward_pe', invert=True)
        vl3 = get_pct(features.get('price_to_book', np.nan), 'price_to_book', invert=True)
        VL = (vl3 * 0.7 + vl1 * 0.15 + vl2 * 0.15)
    elif 'utilities' in st or 'staples' in st:
        # Heavily emphasize dividends and steady risk for utilities
        w['IN'] *= 1.25
        w['RK'] *= 1.25
    elif 'energy' in st or 'materials' in st or 'industrials' in st:
        # Cyclical heavy industries carry structural beta risk, we penalize it more heavily
        w['RK'] *= 1.25
        if client == 'esg': 
            w['ES'] *= 1.3

    # Normalize weights so they always add cleanly to 1.0 (100%)
    total_w = sum(w.values())
    w = {k: v/total_w for k, v in w.items()}

    # Calculate final blend based on modified weights and clean edges
    vals = {'PM': PM, 'RK': RK, 'VL': VL, 'QG': QG, 'IN': IN, 'ES': ES}
    vals = {k: min(max(v, 0), 100) for k, v in vals.items()}
    score = sum(vals[k] * w[k] for k in vals)
    
    # Track exactly what columns were NaN in the origin payload
    missing = [k for k, v in features.items() if pd.isna(v) and isinstance(v, float) and k != 'esg_score']
    if features.get('esg_available', 0) == 0:
        missing.append('esg_score')
        
    report = {'fraction': len(missing) / max(len(features), 1), 'missing_keys': missing}
    return score, vals, report

def combine_signals(scorecard_score, yhat, frac):
    """
    Blends the relative 0-100 Scorecard evaluation score with the ML Ridge Regression absolute output (yhat).
    """
    # Squash linear regression target logic down to an intuitive 0-100 percentage
    ml_score = 1 / (1 + np.exp(-yhat / SIGMOID_SCALE))
    
    # If standard deviation of NaN features gets too high, reduce ML conviction
    w_sc, w_ml = (0.70, 0.30) if frac > MISSINGNESS_HIGH else (0.55, 0.45)
    final = w_sc * scorecard_score + w_ml * (100 * ml_score)
    
    # Determine the strict logic gating a final terminal recommendation
    if final >= BUY_SCORE and yhat > YHAT_BUY_FLOOR: action = "buy"
    elif final <= SELL_SCORE and yhat < YHAT_SELL_CEILING: action = "sell"
    else: action = "hold"
    
    return action, final

def make_analyst_note(ticker, action, final_score, yhat, features, subscores, client, benchmark="^GSPC", missing_report=None):
    """
    Generates an 'analyst-grade' qualitative explanation for a stock recommendation,
    structured as a single cohesive, flowing paragraph.
    """
    def fmt_pct(x, decimals=1):
        if pd.isna(x): return "N/A"
        return f"{x*100:.{decimals}f}%"
        
    def fmt_float(x, digits=1):
        if pd.isna(x): return "N/A"
        return f"{x:.{digits}f}"
        
    def bucket_val(val, low, high, labels=("low", "average", "high")):
        if pd.isna(val): return "unclear"
        if val <= low: return labels[0]
        if val >= high: return labels[2]
        return labels[1]

    def pick_top_drivers(sub):
        names = {'PM': 'Momentum', 'RK': 'Risk', 'VL': 'Valuation', 'QG': 'Quality & Growth', 'IN': 'Income & Yield', 'ES': 'ESG Risk'}
        pos = [(k, v) for k, v in sub.items() if v >= 65]
        neg = [(k, v) for k, v in sub.items() if v <= 40]
        pos = sorted(pos, key=lambda x: x[1], reverse=True)[:2]
        neg = sorted(neg, key=lambda x: x[1])[:2]
        return [(names[k], v) for k, v in pos], [(names[k], v) for k, v in neg]

    sec = features.get('sector', '').lower()
    # Asset Archetyping & Core Fundamentals
    dy = features.get('dividend_yield', np.nan)
    rg = features.get('revenue_growth', np.nan)
    pm = features.get('profit_margin', np.nan)
    vol = features.get('vol_1y', np.nan)
    
    if 'financial' in sec or 'real estate' in sec:
        archetype = f"yield-generating asset (current yield: {fmt_pct(dy)})" if not pd.isna(dy) else "financial asset"
    elif 'energy' in sec or 'material' in sec or 'industrial' in sec:
        archetype = "cyclical, commodity-linked asset"
    elif ('staple' in sec or 'health' in sec) and vol < 0.25:
        archetype = "defensive utility/staple"
    elif rg > 0.10:
        archetype = f"high-growth platform (recent rev expansion: +{fmt_pct(rg, decimals=1)})"
    else:
        archetype = f"mature cash-generator (profit margins: {fmt_pct(pm)})" if not pd.isna(pm) else "mature enterprise"

    # Context & Valuation
    fpe = features.get('forward_pe', np.nan)
    pb = features.get('price_to_book', np.nan)
    om = features.get('operating_margin', np.nan)
    
    fund_str = f"top-decile core operating metrics (Operating Margin: {fmt_pct(om)})" if not pd.isna(om) and om > 0.20 else "mixed underlying profitability"
    
    if 'financial' in sec:
        val_str = f"trading at a price-to-book ratio of {fmt_float(pb)}x" if not pd.isna(pb) else "with an unclear tangible book valuation"
    elif 'technology' in sec or 'communication' in sec:
        val_str = f"Revenue growth of {fmt_pct(rg, decimals=1)} and OM of {fmt_pct(om, decimals=1)} support a {fmt_float(fpe)}x forward P/E." if not (pd.isna(rg) or pd.isna(om) or pd.isna(fpe)) else "trading at a highly opaque forward multiple"
    else:
        val_str = f"carrying a forward multiple roughly {fmt_float(fpe)}x" if not pd.isna(fpe) else "with constrained multiple visibility"

    # Risk & Cyclicality
    beta = features.get('beta', np.nan)
    dd = features.get('max_drawdown_1y', np.nan)
    if 'energy' in sec or 'material' in sec:
        risk_str = f"The asset is highly tethered to underlying commodity swings, registering a beta of {fmt_float(beta)} with recent peak-to-trough drawdowns touching {fmt_pct(-dd, decimals=1)}." if not pd.isna(dd) else f"The asset carries cyclical commodity risk (Beta: {fmt_float(beta)})."
    elif 'utilit' in sec or 'staple' in sec:
        risk_str = f"Demonstrating a highly defensive posture, it exhibits strong structural stability with a beta around {fmt_float(beta)} and constrained {fmt_pct(vol, decimals=1)} rolling volatility." if not pd.isna(vol) else f"Demonstrating a defensive posture (Beta: {fmt_float(beta)})."
    else:
        risk_str = f"Market sensitivity remains {bucket_val(beta, 0.9, 1.2, ('below average', 'in-line with the broader market', 'elevated'))} (Beta: {fmt_float(beta)}, 1Y Drawdown: {fmt_pct(-dd, decimals=1) if not pd.isna(dd) else 'N/A'})."

    # Client Fit
    if client == 'income':
        pr = features.get('payout_ratio', np.nan)
        if pd.isna(dy) or dy < 0.015:
            fit_str = f"Crucially, the stock is a poor fit for strict income mandates due to its negligible ongoing yield ({fmt_pct(dy) if not pd.isna(dy) else 'N/A'})."
        else:
            fit_str = f"It robustly supports yield mandates, offering a sustainable dividend profile of {fmt_pct(dy, decimals=2)} backed by a {fmt_pct(pr, decimals=0) if not pd.isna(pr) else 'N/A'} payout ratio."
    elif client == 'growth':
        if rg > 0.05:
            fit_str = f"This profile closely aligns with capital appreciation goals, leaning into significant operating leverage and {fmt_pct(rg, decimals=1)} revenue expansion."
        else:
            fit_str = f"However, it lacks the necessary top-line reinvestment velocity typically required for pure growth strategies (Rev Growth: {fmt_pct(rg, decimals=1)})."
    elif client == 'esg':
        esg_a = features.get('esg_available', 0)
        es_score = features.get('esg_score', np.nan)
        es_conf = ""
        if 'energy' in sec or 'material' in sec:
            es_conf = " We flag a severe structural mandate conflict given its heavy carbon/extraction footprint."
        if esg_a == 1:
            fit_str = f"Its core Environmental/Governance footprint screens definitively {'favorable' if es_score < 25 else ('moderate' if es_score < 35 else 'high-risk')} (Absolute ESG Score: {fmt_float(es_score)})." + es_conf
        else:
            fit_str = "Explicit ESG scoring remains unavailable via the primary endpoint, necessitating a sector-based proxy." + es_conf
    else:
        fit_str = "Overall, the name represents a balanced core-holding suitable for diverse portfolios."

    # Thesis & Relative Alpha Projection
    pm_rank = subscores.get('PM', 50)
    rel_mom = features.get('relative_momentum_12m', np.nan)
    cagr_str = bucket_val(yhat, 0.0, 0.05, ("lag", "track roughly in-line with", "meaningfully outperform"))
    mom_str = f"(12M excess return vs S&P 500: {fmt_pct(rel_mom, decimals=1)})" if not pd.isna(rel_mom) else ""
    thesis_str = f"When synthesized through our quantitative framework, momentum screens {bucket_val(pm_rank, 40, 60, ('weakly', 'average', 'strongly'))} {mom_str}, leading our ML Ridge projection to anticipate the asset will {cagr_str} the broader index over the coming decade (Projected 10Y Alpha: {fmt_pct(yhat, decimals=1)})."

    # Triggers
    pos_drivers, neg_drivers = pick_top_drivers(subscores)
    triggers = []
    for risk, _ in neg_drivers:
        if risk == 'Valuation': triggers.append("forward multiple compression accelerates")
        elif risk == 'Risk': triggers.append("volatility spikes breach acceptable thresholds")
        elif risk == 'Momentum': triggers.append("relative strength deteriorates against equal-weight peers")
        elif risk == 'ESG Risk': triggers.append("headline ESG risks emerge")
        elif risk == 'Income & Yield' and client == 'income': triggers.append("negative dividend policy changes materialize")
    
    if len(triggers) == 0:
        triggers.append("core macroeconomic regimes shift significantly or margins surprise to the downside")
    elif len(triggers) == 1:
        if subscores.get('QG', 50) > 50:
            triggers.append("the core quality narrative stalls")
        else:
            triggers.append("we observe meaningful sector rotation away from current factors")
            
    trigger_str = f"We would naturally reassess this stance if {' or '.join(triggers[:2])}."

    # Caveats
    if missing_report and missing_report.get('fraction', 0) > 0:
        fields = [f.replace('_', ' ') for f in missing_report.get('missing_keys', [])[:2]]
        cav_str = f" Note that due to partial opacity in underlying data (e.g., {', '.join(fields)}), this assessment leans slightly more on structural ML imputation."
    else:
        cav_str = " Evaluated on a fully populated primary dataset with robust model conviction."

    # Combine into paragraph
    paragraph = (
        f"We classify {ticker} as a {archetype} within the {sec.title()} sector. "
        f"The company exhibits {fund_str}, {val_str}. {risk_str} {fit_str} {thesis_str} "
        f"{trigger_str}{cav_str}".strip()
    )

    header = f"[{ticker}] Final Recommendation: {action.upper()} (Net Conviction Score: {final_score:.0f}/100)\n\n"
    
    return header + "Analyst Note: " + paragraph

def invest(ticker, client=False):
    """
    Main programmatic orchestrator. Links downloading, cleaning, scoring, and text-generation.
    Used by downstream scripts or inline testing loops.
    """
    if client:
        client = str(client).lower()
        if client not in ['income', 'growth', 'esg', 'none', 'base']:
            logging.warning(f"Invalid client mandate '{client}'. Defaulting to standard base weights.")
            client = False
    else:
        client = False
        
    spx = fetch_prices(BENCH, years=10.5)
    px = fetch_prices(ticker, years=10.5)
    fun = fetch_fundamentals(ticker)
    
    has_fun = any(bool(v) for v in fun.values()) if isinstance(fun, dict) else False
    if len(px) == 0 and not has_fun:
        return "error", f"[{ticker}] Error: Could not fetch meaningful price or fundamental data. Ticker may be invalid or delisted."
        
    f = compute_features(ticker, fun, px, spx)
    f['ticker'] = ticker
    
    # Execute identical feature prep to predict against production model artifacts
    try: 
        art = load_model()
        df_ml = pd.DataFrame([{k: v for k, v in f.items() if k in art['num_cols']}], columns=art['num_cols'])
        yhat = art['model'].predict(df_ml)[0]
    except Exception as e: 
        return "hold", f"Model prediction error: {e}"
        
    # Get analytical outputs from submodules
    sc, sub, rep = scorecard(f, client, f['sector'])
    act, fin = combine_signals(sc, yhat, rep['fraction'])
    
    return act, make_analyst_note(ticker, act, fin, yhat, f, sub, client, benchmark=BENCH, missing_report=rep)

if __name__ == '__main__':
    """
    Entrypoint logic for Command-Line Interface usage.
    Usage examples: 
        python hw2.py --train
        python hw2.py --plots AAPL
        python hw2.py AAPL growth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='*', help="Ticker and optional client")
    parser.add_argument('--train', action='store_true', help="Train ML model")
    parser.add_argument('--plots', metavar='TICKER', help="Generate plots")
    args = parser.parse_args()
    
    if args.train:
        X, y, m = build_dataset(UNIVERSE)
        train_model(X, y)
        print("Training complete and artifacts saved.")
        exit(0)
        
    if args.plots:
        tick = args.plots
        os.makedirs("outputs", exist_ok=True)
        s, p = fetch_prices(BENCH, years=10.5), fetch_prices(tick, years=10.5)
        df = pd.DataFrame({tick: p, 'SPX': s}).dropna()
        if len(df) > 0:
            import matplotlib.pyplot as plt
            # Start lines at 1.0 (origin) for relative chart
            df_norm = df / df.iloc[0]
            plt.figure(figsize=(10,4))
            df_norm.plot()
            plt.title(f"Normalized 10Y Price ({tick} vs {BENCH})")
            plt.savefig(f"outputs/{tick}_normalized.png")
            
            # Roll forward exactly 1 trading calendar year to generate moving relative CAGR
            p_roll = df[tick].pct_change(252)
            s_roll = df['SPX'].pct_change(252)
            rel = (p_roll - s_roll).dropna()
            plt.figure(figsize=(10,4))
            rel.plot(title=f"Rolling 1Y Relative Return ({tick} vs {BENCH})")
            plt.axhline(0, color='r', linestyle='--')
            plt.savefig(f"outputs/{tick}_rolling_rel.png")
            print(f"Plots saved to outputs/{tick}_normalized.png and outputs/{tick}_rolling_rel.png")
        exit(0)
        
    # Standard terminal recommendation command parser
    if len(args.args) > 0:
        tick = args.args[0]
        cl = args.args[1].lower() if len(args.args) > 1 else False
        act, exp = invest(tick, cl)
        print(f"[{tick}] Action: {act.upper()}\n\n{exp}")
