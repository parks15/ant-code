#!/usr/bin/env python3
"""
Stock Picker — Python GUI
Ranks stocks by upside potential using price data, technical indicators,
fundamentals, sector trends, and news sentiment analysis.

Install: pip install yfinance pandas matplotlib requests
Run:     python stock_picker.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import concurrent.futures
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# ── Colour palette ─────────────────────────────────────────────────────────────
BG  = '#0d1117'
BG1 = '#161b22'
BG2 = '#21262d'
BD  = '#30363d'
TX  = '#e6edf3'
MU  = '#7d8590'
GR  = '#3fb950'
RE  = '#f85149'
BL  = '#58a6ff'
YE  = '#d29922'
AC  = '#1f6feb'

# ── Stock universe ─────────────────────────────────────────────────────────────
UNIVERSE = [
    # Mega cap
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD',
    # Growth / SaaS
    'SHOP','PLTR','COIN','RBLX','SNOW','DDOG','CRWD','NET','ZS','AFRM','UPST','DKNG',
    # Fintech / Banks
    'XYZ','PYPL','SOFI','HOOD','GS','JPM','BAC',   # XYZ = Block (formerly SQ)
    # EV / Clean energy
    'LCID','RIVN','NIO','WKHS','BLNK','FSLR','ENPH','PLUG',
    # Crypto-adjacent
    'RIOT','MARA','MSTR',
    # Retail / high-short interest  (NKLA & FFIE delisted — removed)
    'GME','AMC','MVIS','CLOV','SPCE','BB',
    # Biotech
    'MRNA','BNTX','NVAX','PFE','REGN','BIIB',
    # Consumer
    'BYND','CVNA','FUBO',
    # Energy
    'XOM','CVX','OXY',
    # China ADRs
    'BABA','JD','PDD','BIDU',
    # Semiconductors
    'INTC','QCOM','MU','LRCX',
    # ETFs
    'SPY','QQQ','IWM','ARKK','XLK','XLE','XLF','XLV',
]

SECTOR_ETFS = [
    ('SPY','S&P 500'),('QQQ','NASDAQ 100'),('DIA','Dow Jones'),('IWM','Russell 2000'),
    ('XLK','Technology'),('XLF','Financials'),('XLV','Healthcare'),('XLE','Energy'),
    ('XLY','Cons. Discret.'),('XLP','Cons. Staples'),('XLI','Industrials'),
    ('XLB','Materials'),('XLU','Utilities'),('XLRE','Real Estate'),('XLC','Communications'),
]

# ── Sentiment word sets ────────────────────────────────────────────────────────
BULL = {
    'surge','soar','rally','breakout','beat','beats','upgrade','buy','strong',
    'gain','record','profit','growth','bullish','rise','climb','jump','boost',
    'positive','exceeds','tops','acquisition','deal','approved','win','expands',
    'outperform','raised','milestone','launches','partnership',
}
BEAR = {
    'drop','fall','crash','miss','misses','downgrade','sell','weak','loss',
    'decline','cut','bearish','sink','plunge','warning','concern','negative',
    'below','disappoints','layoffs','recall','investigation','lawsuit','fraud',
    'fine','penalty','lowered','reduces','halt','probe','violation','default',
}

# ── Technical indicators ───────────────────────────────────────────────────────

def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(closes: pd.Series, fast=12, slow=26, signal=9):
    ema_f = closes.ewm(span=fast, adjust=False).mean()
    ema_s = closes.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

# ── Data fetching ──────────────────────────────────────────────────────────────

YF_API  = 'https://query1.finance.yahoo.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; StockPickerApp/1.0)'}


def _safe(fi, attr):
    """Safely read a fast_info attribute, returning None on any failure."""
    try:
        v = getattr(fi, attr, None)
        return float(v) if v is not None else None
    except Exception:
        return None


def _get_fast_quote(sym: str) -> dict | None:
    """Fetch a single ticker's quote using yfinance fast_info (handles auth)."""
    try:
        fi    = yf.Ticker(sym).fast_info
        price = _safe(fi, 'last_price')
        prev  = _safe(fi, 'previous_close')
        if not price:
            return None
        chg_pct = ((price - prev) / prev * 100) if prev else 0.0
        return {
            'symbol':                     sym,
            'regularMarketPrice':         price,
            'regularMarketChangePercent': chg_pct,
            'regularMarketVolume':        _safe(fi, 'last_volume'),
            'averageDailyVolume3Month':   _safe(fi, 'three_month_average_volume'),
            'marketCap':                  _safe(fi, 'market_cap'),
            'shortName':                  sym,
            'fiftyTwoWeekHigh':           _safe(fi, 'year_high'),
            'fiftyTwoWeekLow':            _safe(fi, 'year_low'),
            'regularMarketPreviousClose': prev,
            # Fundamentals loaded on-demand when user selects the stock
            'floatShares': None, 'shortPercentOfFloat': None,
            'trailingPE':  None, 'forwardPE':           None,
            'revenueGrowth': None, 'profitMargins': None, 'earningsGrowth': None,
        }
    except Exception:
        return None


def fetch_bulk_quotes(tickers: list) -> list:
    """Parallel fast_info fetch — replaces the broken v7 raw API call."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        return [r for r in ex.map(_get_fast_quote, tickers) if r]


def fetch_news(ticker: str) -> list:
    try:
        r = requests.get(
            f'{YF_API}/v1/finance/search',
            params={'q': ticker, 'quotesCount': 0, 'newsCount': 8},
            headers=HEADERS, timeout=8,
        )
        if not r.ok:
            return []
        items = []
        for n in r.json().get('news', []):
            title = n.get('title', '')
            words = set(title.lower().split())
            sentiment = int(bool(words & BULL)) - int(bool(words & BEAR))
            ts = n.get('providerPublishTime')
            items.append({
                'title':     title,
                'publisher': n.get('publisher', ''),
                'link':      n.get('link', ''),
                'time':      datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None,
                'sentiment': sentiment,
            })
        return items
    except Exception:
        return []


def fetch_sector_trends() -> list:
    """Use yfinance download for sector ETF day-changes (no auth issues)."""
    etfs = [s for s, _ in SECTOR_ETFS]
    try:
        raw = yf.download(
            tickers=etfs,
            period='5d',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # yf.download with multiple tickers returns MultiIndex columns (metric, ticker)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
        results = []
        for sym, name in SECTOR_ETFS:
            try:
                col   = closes[sym] if sym in closes.columns else None
                valid = col.dropna() if col is not None else pd.Series(dtype=float)
                chg   = float((valid.iloc[-1] - valid.iloc[-2]) / valid.iloc[-2] * 100) \
                        if len(valid) >= 2 else 0.0
            except Exception:
                chg = 0.0
            results.append((sym, name, chg))
        return sorted(results, key=lambda x: x[2], reverse=True)
    except Exception as e:
        print(f'Sector trends error: {e}')
        return []

# ── Price prediction ───────────────────────────────────────────────────────────

def predict_price_targets(quote: dict, hist: pd.DataFrame,
                          news: list, info: dict) -> tuple:
    """
    Multi-factor price prediction for 1, 3, 6, and 12 months.

    Model components
    ----------------
    1. Trend      — linear regression on 60-day close prices (daily % slope)
    2. Momentum   — RSI level, MACD cross, and MA stack adjustments
    3. Sentiment  — net bullish/bearish news headline ratio
    4. Fundamentals — revenue growth and profit margin contribution
    5. Volatility — historical daily σ scaled to each horizon for confidence bands

    Returns (targets, factors) where targets is a list of dicts
    and factors is a list of (description, 'bull'|'bear') tuples.
    """
    price = float(quote.get('regularMarketPrice') or 0)
    if not price or hist is None or len(hist) < 20:
        return None, []

    try:
        closes = hist['Close']
        if hasattr(closes, 'squeeze'):
            closes = closes.squeeze()
        closes = closes.dropna().astype(float)
        if len(closes) < 10:
            return None, []

        returns   = closes.pct_change().dropna()
        daily_vol = float(returns.std()) or 0.015

        # ── 1. Trend via linear regression on last 60 trading days ──────────
        n = min(60, len(closes))
        y = closes.tail(n).values.astype(float)
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        daily_trend = float(np.clip(slope / float(np.mean(y)), -0.008, 0.008))

        # ── 2. Momentum ──────────────────────────────────────────────────────
        momentum_adj = 0.0
        factors      = []

        if len(closes) >= 14:
            rsi = float(compute_rsi(closes).iloc[-1])
            if   rsi < 30:  momentum_adj += 0.040; factors.append(('RSI oversold — bounce likely',    'bull'))
            elif rsi < 50:  momentum_adj += 0.015; factors.append(('RSI building momentum',           'bull'))
            elif rsi > 75:  momentum_adj -= 0.025; factors.append(('RSI overbought — pullback risk',  'bear'))

        if len(closes) >= 26:
            macd, sig, _ = compute_macd(closes)
            cross_up   = float(macd.iloc[-1]) > float(sig.iloc[-1]) and float(macd.iloc[-2]) <= float(sig.iloc[-2])
            cross_down = float(macd.iloc[-1]) < float(sig.iloc[-1]) and float(macd.iloc[-2]) >= float(sig.iloc[-2])
            if   cross_up:                                   momentum_adj += 0.030; factors.append(('MACD bullish crossover',           'bull'))
            elif float(macd.iloc[-1]) > float(sig.iloc[-1]): momentum_adj += 0.010
            elif cross_down:                                 momentum_adj -= 0.020; factors.append(('MACD bearish crossover',           'bear'))
            else:                                            momentum_adj -= 0.005

        if len(closes) >= 50:
            ma20 = float(closes.rolling(20).mean().iloc[-1])
            ma50 = float(closes.rolling(50).mean().iloc[-1])
            if   ma20 > ma50 and price > ma20: momentum_adj += 0.015; factors.append(('Price above MA20 & MA50',               'bull'))
            elif ma20 < ma50:                  momentum_adj -= 0.010; factors.append(('MA20 below MA50 — downtrend',           'bear'))

        # ── 3. News sentiment ────────────────────────────────────────────────
        news_adj = 0.0
        if news:
            bulls = sum(1 for n in news if n['sentiment'] > 0)
            bears = sum(1 for n in news if n['sentiment'] < 0)
            ratio = (bulls - bears) / len(news)
            news_adj = float(np.clip(ratio * 0.04, -0.05, 0.05))
            if   ratio >= 0.4: factors.append(('Positive news sentiment',            'bull'))
            elif ratio <= -0.4: factors.append(('Negative news sentiment',           'bear'))

        # ── 4. Fundamentals ──────────────────────────────────────────────────
        fund_adj = 0.0
        rev_gr   = float((info or {}).get('revenueGrowth',  0) or 0)
        margins  = float((info or {}).get('profitMargins',  0) or 0)

        if   rev_gr > 0.25: fund_adj += 0.040; factors.append((f'Revenue growing +{rev_gr*100:.0f}%/yr',   'bull'))
        elif rev_gr > 0.10: fund_adj += 0.015
        elif rev_gr < -0.10: fund_adj -= 0.020; factors.append((f'Revenue declining {rev_gr*100:.0f}%/yr', 'bear'))

        if   margins > 0.20: fund_adj += 0.015; factors.append(('High profit margins',                     'bull'))
        elif margins < 0:    fund_adj -= 0.010

        # 52-week context
        high52    = float(quote.get('fiftyTwoWeekHigh') or 0)
        short_pct = float(quote.get('shortPercentOfFloat') or 0) * 100
        if high52 and price:
            pct_off = (high52 - price) / high52 * 100
            if   pct_off <= 3:  factors.append(('Near 52-week high — breakout territory',   'bull'))
            elif pct_off >= 40: factors.append(('Far below 52-week high — recovery upside', 'bull'))
        if short_pct >= 15:
            factors.append((f'{short_pct:.0f}% short interest — squeeze potential', 'bull'))

        # ── 5. Build per-horizon targets ─────────────────────────────────────
        one_time = momentum_adj + news_adj + fund_adj
        targets  = []

        for months in [1, 3, 6, 12]:
            trading_days = months * 21
            trend_return = (1 + daily_trend) ** trading_days - 1
            time_fund    = rev_gr * (months / 12) * 0.25
            base_return  = trend_return + time_fund + one_time
            vol_band     = daily_vol * np.sqrt(trading_days) * 0.8

            targets.append({
                'months':     months,
                'base':       max(0.01, price * (1 + base_return)),
                'bull':       max(0.01, price * (1 + base_return + vol_band)),
                'bear':       max(0.01, price * (1 + base_return - vol_band)),
                'change_pct': base_return * 100,
            })

        return targets, factors

    except Exception as e:
        return None, []


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_stock(q: dict, hist: pd.DataFrame = None, news: list = None) -> tuple:
    price     = q.get('regularMarketPrice') or 0
    chg_pct   = q.get('regularMarketChangePercent') or 0
    volume    = q.get('regularMarketVolume') or 0
    avg_vol   = q.get('averageDailyVolume3Month') or 1
    high52    = q.get('fiftyTwoWeekHigh') or 0
    short_pct = (q.get('shortPercentOfFloat') or 0) * 100
    float_sh  = q.get('floatShares')
    pe        = q.get('trailingPE')
    fwd_pe    = q.get('forwardPE')
    rev_gr    = q.get('revenueGrowth') or 0
    margins   = q.get('profitMargins') or 0

    score, signals = 0, []

    # Relative volume — unusual activity often precedes big moves
    rel_vol = volume / avg_vol if avg_vol else 0
    if   rel_vol >= 10: score += 35; signals.append(('HOT VOL',            'red'))
    elif rel_vol >= 5:  score += 25; signals.append((f'{rel_vol:.1f}× Vol', 'red'))
    elif rel_vol >= 2:  score += 15; signals.append((f'{rel_vol:.1f}× Vol', 'yellow'))
    elif rel_vol >= 1:  score += 7

    # Intraday momentum
    if   chg_pct >= 20: score += 30; signals.append((f'+{chg_pct:.0f}% Gapper', 'green'))
    elif chg_pct >= 10: score += 22; signals.append((f'+{chg_pct:.0f}%',         'green'))
    elif chg_pct >= 5:  score += 14; signals.append((f'+{chg_pct:.0f}%',         'yellow'))
    elif chg_pct >= 2:  score += 7
    elif chg_pct < -5:  score -= 8

    # Short squeeze potential
    if short_pct and chg_pct > 0:
        if   short_pct >= 20: score += 20; signals.append((f'{short_pct:.0f}% Shorted', 'orange'))
        elif short_pct >= 10: score += 12; signals.append((f'{short_pct:.0f}% Short',   'orange'))
        elif short_pct >= 5:  score += 5

    # Float size — thin supply amplifies moves
    if float_sh is not None:
        if   float_sh < 5e6:   score += 15; signals.append(('Micro Float', 'blue'))
        elif float_sh < 20e6:  score += 10; signals.append(('Low Float',   'blue'))
        elif float_sh < 100e6: score += 5

    # 52-week high proximity — breakout territory
    if high52 and price:
        pct_off = (high52 - price) / high52 * 100
        if   pct_off <= 2:  score += 15; signals.append(('52W Break',   'green'))
        elif pct_off <= 10: score += 8;  signals.append(('Near 52W Hi', 'cyan'))
        elif pct_off <= 20: score += 3

    # Technical indicators (requires price history)
    if hist is not None and len(hist) >= 20:
        try:
            closes = hist['Close']
            if hasattr(closes, 'squeeze'):
                closes = closes.squeeze()
            rsi           = compute_rsi(closes).iloc[-1]
            macd, sig, _  = compute_macd(closes)
            ma20          = closes.rolling(20).mean().iloc[-1]

            if   30 <= rsi <= 50: score += 12; signals.append((f'RSI {rsi:.0f}↑',      'cyan'))
            elif rsi < 30:        score += 6;  signals.append((f'RSI {rsi:.0f} Oversold','yellow'))
            elif rsi > 75:        score -= 5   # overbought risk

            if macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2]:
                score += 15; signals.append(('MACD Cross↑', 'green'))
            elif macd.iloc[-1] > sig.iloc[-1]:
                score += 6

            if len(closes) >= 50:
                ma50 = closes.rolling(50).mean().iloc[-1]
                if ma20 > ma50 and price > ma20:
                    score += 8; signals.append(('Above MA50', 'green'))
        except Exception:
            pass

    # Fundamentals
    if rev_gr > 0.30:   score += 10; signals.append((f'+{rev_gr*100:.0f}% RevGrowth', 'green'))
    elif rev_gr > 0.15: score += 5
    if margins > 0.20:  score += 8
    elif margins > 0.10: score += 4
    if pe and 0 < pe < 15:     score += 8
    elif pe and 15 <= pe < 25: score += 4
    elif pe and fwd_pe and fwd_pe > 0 and fwd_pe < pe:
        score += 5; signals.append(('EPS Growing', 'green'))

    # News sentiment
    if news:
        bulls = sum(1 for n in news if n['sentiment'] > 0)
        bears = sum(1 for n in news if n['sentiment'] < 0)
        ratio = (bulls - bears) / len(news)
        if   ratio >= 0.5:  score += 15; signals.append(('Positive News', 'green'))
        elif ratio > 0:     score += 7;  signals.append(('Bullish News',  'yellow'))
        elif ratio <= -0.5: score -= 10; signals.append(('Negative News', 'red'))

    return max(0, min(100, score)), signals

# ── Application ────────────────────────────────────────────────────────────────

class StockPickerApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Stock Picker')
        self.root.configure(bg=BG)
        self.root.geometry('1300x780')
        self.root.minsize(900, 600)

        self._scan_q:   queue.Queue = queue.Queue()
        self._detail_q: queue.Queue = queue.Queue()
        self._results: list  = []
        self._current: str   = ''
        self._chart_canvas   = None

        self._setup_style()
        self._build_ui()
        self._update_market_status()
        self._poll_queues()

    # ── Style ──────────────────────────────────────────────────────────────────

    def _setup_style(self):
        s = ttk.Style(self.root)
        s.theme_use('clam')
        s.configure('.',          background=BG,  foreground=TX, font=('Segoe UI', 10))
        s.configure('TFrame',     background=BG)
        s.configure('TLabel',     background=BG,  foreground=TX)
        s.configure('TButton',    background=AC,  foreground='white',
                    font=('Segoe UI', 10, 'bold'), relief='flat', padding=(12, 6))
        s.map('TButton',
              background=[('active', '#388bfd'), ('disabled', BG2)],
              foreground=[('disabled', MU)])

        s.configure('Stocks.Treeview',
            background=BG, foreground=TX, fieldbackground=BG,
            rowheight=26, font=('Segoe UI', 10), borderwidth=0)
        s.configure('Stocks.Treeview.Heading',
            background=BG1, foreground=MU,
            font=('Segoe UI', 9, 'bold'), relief='flat')
        s.map('Stocks.Treeview',
              background=[('selected', '#1a2540')],
              foreground=[('selected', BL)])
        s.map('Stocks.Treeview.Heading',
              background=[('active', BG2)])

        s.configure('TCombobox', fieldbackground=BG2, background=BG2,
                    foreground=TX, selectbackground=BG2, selectforeground=TX)
        s.map('TCombobox', fieldbackground=[('readonly', BG2)],
              selectbackground=[('readonly', BG2)])

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG1, height=50)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        row = tk.Frame(hdr, bg=BG1)
        row.pack(fill='both', expand=True, padx=16, pady=8)

        tk.Label(row, text='Stock Picker', bg=BG1, fg=BL,
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=(0, 12))
        self._mkt_lbl = tk.Label(row, text='Checking…', bg=BG2, fg=MU,
                                  font=('Segoe UI', 9), padx=8, pady=2)
        self._mkt_lbl.pack(side='left', padx=(0, 14))
        self._scan_btn = ttk.Button(row, text='⚡  Scan Market', command=self._start_scan)
        self._scan_btn.pack(side='left', padx=(0, 8))
        ttk.Button(row, text='🌐  Sector Trends', command=self._open_trends_window).pack(side='left')
        self._status_lbl = tk.Label(row, text='Click Scan to find opportunities',
                                     bg=BG1, fg=MU, font=('Segoe UI', 9))
        self._status_lbl.pack(side='right')

        # Filter bar
        fbar = tk.Frame(self.root, bg=BG1, height=34)
        fbar.pack(fill='x')
        fbar.pack_propagate(False)
        frow = tk.Frame(fbar, bg=BG1)
        frow.pack(side='left', fill='y', padx=16, pady=6)

        tk.Label(frow, text='Min Score:', bg=BG1, fg=MU, font=('Segoe UI', 9)).pack(side='left', padx=(0,4))
        self._min_score = tk.StringVar(value='0')
        ttk.Combobox(frow, textvariable=self._min_score,
                     values=['0','20','30','50','70'], width=4,
                     state='readonly').pack(side='left', padx=(0,16))

        tk.Label(frow, text='Sort:', bg=BG1, fg=MU, font=('Segoe UI', 9)).pack(side='left', padx=(0,4))
        self._sort_by = tk.StringVar(value='Score')
        ttk.Combobox(frow, textvariable=self._sort_by,
                     values=['Score','Change %','Rel. Vol','Upside %','Ticker'], width=10,
                     state='readonly').pack(side='left', padx=(0,16))

        self._cnt_lbl = tk.Label(frow, text='', bg=BG1, fg=MU, font=('Segoe UI', 9))
        self._cnt_lbl.pack(side='left')

        for cb in frow.winfo_children():
            if isinstance(cb, ttk.Combobox):
                cb.bind('<<ComboboxSelected>>', lambda _: self._apply_filter())

        tk.Frame(self.root, bg=BD, height=1).pack(fill='x')

        # Main panes
        pane = tk.PanedWindow(self.root, orient='horizontal',
                              bg=BD, sashwidth=4, sashrelief='flat')
        pane.pack(fill='both', expand=True)

        left_frame = tk.Frame(pane, bg=BG)
        pane.add(left_frame, minsize=440, width=700)
        self._build_table(left_frame)

        right_frame = tk.Frame(pane, bg=BG1)
        pane.add(right_frame, minsize=380, width=560)
        self._build_detail(right_frame)

        # Status bar
        sb = tk.Frame(self.root, bg=BG1, height=22)
        sb.pack(fill='x')
        sb.pack_propagate(False)
        tk.Label(sb, text='Not financial advice · For informational use only',
                 bg=BG1, fg=MU, font=('Segoe UI', 8)).pack(side='right', padx=12)
        self._prog_lbl = tk.Label(sb, text='', bg=BG1, fg=MU, font=('Segoe UI', 8))
        self._prog_lbl.pack(side='left', padx=12)

    def _build_table(self, parent):
        cols = ('rank','ticker','name','price','change','relvol','upside','score','signals')
        self._tree = ttk.Treeview(parent, columns=cols, show='headings',
                                   style='Stocks.Treeview', selectmode='browse')
        cfg = [
            ('rank',    '#',         40, 'center'),
            ('ticker',  'Ticker',    72, 'w'),
            ('name',    'Company',  155, 'w'),
            ('price',   'Price',     70, 'e'),
            ('change',  'Change %',  72, 'e'),
            ('relvol',  'Rel.Vol',   62, 'e'),
            ('upside',  '52W Up',    65, 'e'),
            ('score',   'Score',     52, 'center'),
            ('signals', 'Signals',  210, 'w'),
        ]
        for cid, lbl, w, anchor in cfg:
            self._tree.heading(cid, text=lbl,
                               command=lambda c=cid: self._col_sort(c))
            self._tree.column(cid, width=w, anchor=anchor,
                              stretch=(cid in ('signals','name')))

        sb = ttk.Scrollbar(parent, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._tree.pack(fill='both', expand=True)
        self._tree.bind('<<TreeviewSelect>>', self._on_row_select)

        self._tree.tag_configure('hot',  foreground=RE,   font=('Segoe UI', 10, 'bold'))
        self._tree.tag_configure('warm', foreground=YE)
        self._tree.tag_configure('cool', foreground=BL)

    def _build_detail(self, parent):
        # Stock header
        hf = tk.Frame(parent, bg=BG1, padx=14, pady=10)
        hf.pack(fill='x')
        left_hf = tk.Frame(hf, bg=BG1)
        left_hf.pack(side='left', fill='x', expand=True)
        self._d_sym  = tk.Label(left_hf, text='—', bg=BG1, fg=BL,
                                 font=('Segoe UI', 22, 'bold'))
        self._d_sym.pack(anchor='w')
        self._d_name = tk.Label(left_hf, text='Select a stock from the list',
                                 bg=BG1, fg=MU, font=('Segoe UI', 9))
        self._d_name.pack(anchor='w')
        right_hf = tk.Frame(hf, bg=BG1)
        right_hf.pack(side='right', anchor='n')
        self._d_price = tk.Label(right_hf, text='', bg=BG1, fg=TX,
                                  font=('Segoe UI', 14, 'bold'))
        self._d_price.pack(anchor='e')
        self._d_chg = tk.Label(right_hf, text='', bg=BG1,
                                font=('Segoe UI', 11, 'bold'))
        self._d_chg.pack(anchor='e')

        tk.Frame(parent, bg=BD, height=1).pack(fill='x')

        # Stats grid (6 cells)
        self._stats_frame = tk.Frame(parent, bg=BG1, padx=10, pady=5)
        self._stats_frame.pack(fill='x')

        # Chart
        cf = tk.Frame(parent, bg=BG, padx=4, pady=2)
        cf.pack(fill='both', expand=True)
        self._fig = Figure(figsize=(5, 2.6), facecolor=BG)
        self._ax  = self._fig.add_subplot(111, facecolor=BG1)
        self._ax.text(0.5, 0.5, 'Select a stock to view chart',
                      ha='center', va='center', color=MU,
                      transform=self._ax.transAxes, fontsize=9)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_color(BD)
        self._fig.patch.set_facecolor(BG)
        self._fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.15)
        self._chart_canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._chart_canvas.get_tk_widget().pack(fill='both', expand=True)

        tk.Frame(parent, bg=BD, height=1).pack(fill='x')

        # ── Tab strip (News / Forecast) ────────────────────────────────────
        self._active_tab = 'news'
        tab_strip = tk.Frame(parent, bg=BG1)
        tab_strip.pack(fill='x')
        self._tab_btns: dict = {}
        for key, label in [('news', '  📰  News  '), ('forecast', '  📈  Forecast  ')]:
            btn = tk.Label(tab_strip, text=label, bg=BG1, fg=MU,
                           font=('Segoe UI', 9), pady=5, cursor='hand2')
            btn.pack(side='left')
            btn.bind('<Button-1>', lambda e, k=key: self._show_tab(k))
            self._tab_btns[key] = btn
        # Bottom border under strip
        tk.Frame(tab_strip, bg=BD, height=1).pack(fill='x', side='bottom')

        # Fixed-height container so switching tabs doesn't resize the panel
        self._bottom = tk.Frame(parent, bg=BG, height=210)
        self._bottom.pack(fill='x')
        self._bottom.pack_propagate(False)

        # ── News panel ─────────────────────────────────────────────────────
        self._news_outer = tk.Frame(self._bottom, bg=BG)

        nhdr = tk.Frame(self._news_outer, bg=BG1, padx=14, pady=3)
        nhdr.pack(fill='x')
        tk.Label(nhdr, text='RECENT HEADLINES', bg=BG1, fg=MU,
                 font=('Segoe UI', 8, 'bold')).pack(side='left')
        self._sent_lbl = tk.Label(nhdr, text='', bg=BG1, font=('Segoe UI', 8, 'bold'))
        self._sent_lbl.pack(side='right')

        self._news_text = tk.Text(self._news_outer, bg=BG, fg=TX,
                                   font=('Segoe UI', 9), wrap='word',
                                   height=9, relief='flat', state='disabled',
                                   cursor='arrow', padx=14, pady=6,
                                   insertbackground=BG, selectbackground='#1a2540')
        self._news_text.pack(fill='x')
        self._news_text.tag_configure('bull',  foreground=GR, font=('Segoe UI', 9, 'bold'))
        self._news_text.tag_configure('bear',  foreground=RE, font=('Segoe UI', 9, 'bold'))
        self._news_text.tag_configure('neu',   foreground=MU, font=('Segoe UI', 9, 'bold'))
        self._news_text.tag_configure('title', foreground=TX)
        self._news_text.tag_configure('meta',  foreground=MU, font=('Segoe UI', 8))

        # ── Forecast panel ─────────────────────────────────────────────────
        self._forecast_outer = tk.Frame(self._bottom, bg=BG)

        # Disclaimer
        tk.Label(self._forecast_outer,
                 text='Model estimates · trend + momentum + sentiment + fundamentals · Not financial advice',
                 bg=BG, fg=MU, font=('Segoe UI', 7, 'italic'), pady=3).pack(fill='x', padx=14)

        # Price table
        tbl = tk.Frame(self._forecast_outer, bg=BG, padx=14, pady=2)
        tbl.pack(fill='x')

        horizons = ['1 Month', '3 Months', '6 Months', '12 Months']
        rows     = ['bull', 'base', 'bear', 'chg']
        row_lbls = ['Bull  ↑', 'Base', 'Bear  ↓', 'Change']
        row_fgs  = [GR, TX, RE, BL]

        # Column headers
        tk.Label(tbl, text='', bg=BG, width=9).grid(row=0, column=0, sticky='w')
        for c, h in enumerate(horizons):
            tk.Label(tbl, text=h, bg=BG, fg=MU,
                     font=('Segoe UI', 8, 'bold'), width=11, anchor='e').grid(
                row=0, column=c+1, sticky='e', pady=(2,4))

        # Separator
        sep = tk.Frame(tbl, bg=BD, height=1)
        sep.grid(row=1, column=0, columnspan=5, sticky='ew', pady=2)

        # Price rows
        self._fcst_labels: dict = {}
        for r_idx, (rkey, rlbl, rfg) in enumerate(zip(rows, row_lbls, row_fgs)):
            tk.Label(tbl, text=rlbl, bg=BG, fg=rfg,
                     font=('Segoe UI', 9, 'bold'), width=9, anchor='w').grid(
                row=r_idx+2, column=0, sticky='w')
            cells = []
            for c_idx in range(4):
                lbl = tk.Label(tbl, text='—', bg=BG, fg=rfg,
                               font=('Segoe UI', 9), width=11, anchor='e')
                lbl.grid(row=r_idx+2, column=c_idx+1, sticky='e')
                cells.append(lbl)
            self._fcst_labels[rkey] = cells

        # Make columns expand evenly
        for c in range(5):
            tbl.columnconfigure(c, weight=1)

        # Separator
        tk.Frame(self._forecast_outer, bg=BD, height=1).pack(fill='x', padx=14, pady=4)

        # Prediction factors
        fhdr = tk.Frame(self._forecast_outer, bg=BG, padx=14)
        fhdr.pack(fill='x')
        tk.Label(fhdr, text='KEY DRIVERS', bg=BG, fg=MU,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w')
        self._factors_frame = tk.Frame(self._forecast_outer, bg=BG, padx=14, pady=2)
        self._factors_frame.pack(fill='x')
        tk.Label(self._factors_frame, text='Select a stock to generate forecast',
                 bg=BG, fg=MU, font=('Segoe UI', 9)).pack(anchor='w')

        # Set initial tab state now that both panels exist
        self._show_tab('news')

    # ── Tab switching ──────────────────────────────────────────────────────────

    def _show_tab(self, which: str):
        self._active_tab = which
        for key, btn in self._tab_btns.items():
            active = key == which
            btn.config(fg=TX if active else MU,
                       bg=BG if active else BG1,
                       font=('Segoe UI', 9, 'bold') if active else ('Segoe UI', 9))
        if which == 'news':
            self._forecast_outer.pack_forget()
            self._news_outer.pack(fill='both', expand=True)
        else:
            self._news_outer.pack_forget()
            self._forecast_outer.pack(fill='both', expand=True)

    # ── Forecast rendering ─────────────────────────────────────────────────────

    def _render_predictions(self, targets: list, factors: list):
        if not targets:
            for rkey, cells in self._fcst_labels.items():
                for cell in cells:
                    cell.config(text='N/A', fg=MU)
            for w in self._factors_frame.winfo_children():
                w.destroy()
            tk.Label(self._factors_frame, text='Insufficient data for forecast',
                     bg=BG, fg=MU, font=('Segoe UI', 9)).pack(anchor='w')
            return

        def fp(v):
            return f'${v:.4f}' if v < 1 else f'${v:.2f}'

        for c_idx, t in enumerate(targets):
            self._fcst_labels['bull'][c_idx].config(text=fp(t['bull']))
            self._fcst_labels['base'][c_idx].config(text=fp(t['base']))
            self._fcst_labels['bear'][c_idx].config(text=fp(t['bear']))
            chg   = t['change_pct']
            sign  = '+' if chg >= 0 else ''
            color = GR if chg >= 0 else RE
            self._fcst_labels['chg'][c_idx].config(
                text=f'{sign}{chg:.1f}%', fg=color)

        # Prediction factors
        for w in self._factors_frame.winfo_children():
            w.destroy()
        if not factors:
            tk.Label(self._factors_frame, text='No strong directional signals detected',
                     bg=BG, fg=MU, font=('Segoe UI', 9)).pack(anchor='w')
        else:
            for desc, direction in factors:
                icon  = '▲' if direction == 'bull' else '▼'
                color = GR  if direction == 'bull' else RE
                row = tk.Frame(self._factors_frame, bg=BG)
                row.pack(fill='x', pady=1)
                tk.Label(row, text=icon, bg=BG, fg=color,
                         font=('Segoe UI', 9, 'bold'), width=2).pack(side='left')
                tk.Label(row, text=desc, bg=BG, fg=TX,
                         font=('Segoe UI', 9)).pack(side='left')

    # ── Scan ───────────────────────────────────────────────────────────────────

    def _start_scan(self):
        self._scan_btn.config(state='disabled', text='⏳  Scanning…')
        t = threading.Thread(target=self._scan_worker, daemon=True)
        t.start()

    def _scan_worker(self):
        try:
            self._scan_q.put(('status', f'Fetching live quotes for {len(UNIVERSE)} stocks…'))
            quotes = fetch_bulk_quotes(UNIVERSE)
            self._scan_q.put(('status', f'Scoring {len(quotes)} stocks…'))

            results = []
            for i, q in enumerate(quotes):
                sym = q.get('symbol', '')
                if not sym or not q.get('regularMarketPrice'):
                    continue
                price   = q.get('regularMarketPrice', 0) or 0
                chg     = q.get('regularMarketChangePercent', 0) or 0
                vol     = q.get('regularMarketVolume', 0) or 0
                avg_vol = q.get('averageDailyVolume3Month') or 1
                high52  = q.get('fiftyTwoWeekHigh') or 0
                rel_vol = vol / avg_vol if avg_vol else 0
                upside  = ((high52 - price) / price * 100) if high52 and price else None

                score, signals = score_stock(q)
                results.append({
                    'ticker':  sym,
                    'name':    q.get('shortName') or q.get('longName', ''),
                    'price':   price,
                    'change':  chg,
                    'rel_vol': rel_vol,
                    'upside':  upside,
                    'score':   score,
                    'signals': signals,
                    'quote':   q,
                })

                if (i + 1) % 15 == 0:
                    self._scan_q.put(('status', f'Scored {i + 1}/{len(quotes)}…'))

            results.sort(key=lambda x: x['score'], reverse=True)
            self._scan_q.put(('done', results))
        except Exception as e:
            self._scan_q.put(('error', str(e)))

    # ── Sector trends window ───────────────────────────────────────────────────

    def _open_trends_window(self):
        win = tk.Toplevel(self.root)
        win.title('Market Sector Trends')
        win.configure(bg=BG)
        win.geometry('460x520')
        win.resizable(False, True)

        tk.Label(win, text='Sector Performance', bg=BG, fg=TX,
                 font=('Segoe UI', 13, 'bold')).pack(anchor='w', padx=16, pady=(14, 2))
        tk.Label(win, text='Sectors ranked by today\'s change', bg=BG, fg=MU,
                 font=('Segoe UI', 9)).pack(anchor='w', padx=16, pady=(0, 8))

        loading = tk.Label(win, text='Loading…', bg=BG, fg=MU, font=('Segoe UI', 9))
        loading.pack()

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill='both', expand=True, padx=12, pady=4)

        def load():
            data = fetch_sector_trends()
            win.after(0, lambda: _render(data))

        def _render(data):
            loading.destroy()
            if not data:
                tk.Label(frame, text='Could not load sector data.', bg=BG, fg=RE).pack()
                return
            for sym, name, chg in data:
                color  = GR if chg > 0 else (RE if chg < 0 else MU)
                bg_row = '#0d2118' if chg > 0.5 else ('#200e0d' if chg < -0.5 else BG2)
                row = tk.Frame(frame, bg=bg_row, pady=5, padx=10)
                row.pack(fill='x', pady=1)
                tk.Label(row, text=sym, bg=bg_row, fg=BL,
                         font=('Segoe UI', 10, 'bold'), width=6, anchor='w').pack(side='left')
                tk.Label(row, text=name, bg=bg_row, fg=TX,
                         font=('Segoe UI', 9), width=18, anchor='w').pack(side='left', padx=4)
                sign = '+' if chg >= 0 else ''
                tk.Label(row, text=f'{sign}{chg:.2f}%', bg=bg_row, fg=color,
                         font=('Segoe UI', 10, 'bold'), width=9, anchor='e').pack(side='right')
                # Progress bar
                bar_max = 100
                bar_len = min(int(abs(chg) * 12), bar_max)
                bar_frame = tk.Frame(row, bg=bg_row, height=3)
                bar_frame.pack(fill='x', side='bottom')
                tk.Frame(bar_frame, bg=color, height=3, width=bar_len).pack(
                    side='left' if chg >= 0 else 'right')

        threading.Thread(target=load, daemon=True).start()

    # ── Detail loading ─────────────────────────────────────────────────────────

    def _on_row_select(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        vals = self._tree.item(sel[0])['values']
        if not vals:
            return
        ticker = str(vals[1])
        if ticker == self._current:
            return
        self._current = ticker
        res = next((r for r in self._results if r['ticker'] == ticker), None)
        if res:
            self._render_header(res)
            self._render_stats(res['quote'])
        threading.Thread(target=self._detail_worker, args=(ticker,), daemon=True).start()

    def _detail_worker(self, ticker):
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period='6mo')   # 6mo gives enough data for 200-day MA
            news = fetch_news(ticker)
            try:
                info = t.info or {}
            except Exception:
                info = {}
            # Run prediction on background thread so GUI never blocks
            res = next((r for r in self._results if r['ticker'] == ticker), None)
            quote = res['quote'] if res else {}
            targets, factors = predict_price_targets(quote, hist, news, info)
            self._detail_q.put(('ok', ticker, hist, news, info, targets, factors))
        except Exception as e:
            self._detail_q.put(('err', ticker, str(e)))

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render_header(self, res):
        self._d_sym.config(text=res['ticker'])
        self._d_name.config(text=(res['name'] or ''))
        self._d_price.config(text=f"${res['price']:.4f}" if res['price'] < 1
                                  else f"${res['price']:.2f}")
        chg  = res['change']
        sign = '+' if chg >= 0 else ''
        self._d_chg.config(text=f"{sign}{chg:.2f}%",
                           fg=GR if chg > 0 else (RE if chg < 0 else MU))

    def _render_stats(self, q):
        for w in self._stats_frame.winfo_children():
            w.destroy()
        stats = [
            ('52W High',    f"${q.get('fiftyTwoWeekHigh'):.2f}"      if q.get('fiftyTwoWeekHigh')  else '—'),
            ('52W Low',     f"${q.get('fiftyTwoWeekLow'):.2f}"       if q.get('fiftyTwoWeekLow')   else '—'),
            ('Mkt Cap',     self._fmt(q.get('marketCap'))),
            ('P/E (TTM)',   f"{q.get('trailingPE'):.1f}"             if q.get('trailingPE')        else '—'),
            ('Short %',     f"{q.get('shortPercentOfFloat',0)*100:.1f}%" if q.get('shortPercentOfFloat') else '—'),
            ('Rev Growth',  f"{q.get('revenueGrowth',0)*100:+.1f}%" if q.get('revenueGrowth')     else '—'),
        ]
        for i, (lbl, val) in enumerate(stats):
            c, r = i % 3, i // 3
            cell = tk.Frame(self._stats_frame, bg=BG2, padx=6, pady=4)
            cell.grid(row=r, column=c, sticky='ew', padx=2, pady=1)
            self._stats_frame.columnconfigure(c, weight=1)
            tk.Label(cell, text=lbl, bg=BG2, fg=MU, font=('Segoe UI', 8)).pack(anchor='w')
            tk.Label(cell, text=val, bg=BG2, fg=TX, font=('Segoe UI', 10, 'bold')).pack(anchor='w')

    def _draw_chart(self, hist, ticker):
        self._ax.clear()
        if hist is None or len(hist) < 5:
            self._ax.text(0.5, 0.5, 'No chart data available',
                          ha='center', va='center', color=MU,
                          transform=self._ax.transAxes, fontsize=9)
            self._ax.set_xticks([]); self._ax.set_yticks([])
            for sp in self._ax.spines.values(): sp.set_color(BD)
            self._chart_canvas.draw()
            return

        try:
            closes = hist['Close'].squeeze()
            # Strip timezone for matplotlib compatibility
            try:
                idx = hist.index.tz_convert(None)
            except Exception:
                idx = hist.index

            first, last = closes.iloc[0], closes.iloc[-1]
            color = GR if last >= first else RE

            self._ax.plot(idx, closes, color=color, linewidth=1.6, zorder=3)
            self._ax.fill_between(idx, closes, closes.min() * 0.998,
                                  alpha=0.12, color=color, zorder=2)

            if len(closes) >= 20:
                ma20 = closes.rolling(20).mean()
                self._ax.plot(idx, ma20, color=BL, linewidth=0.9,
                              linestyle='--', alpha=0.7, zorder=2)

            self._ax.set_facecolor(BG1)
            self._ax.tick_params(colors=MU, labelsize=7)
            for sp in self._ax.spines.values(): sp.set_color(BD)
            self._ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            self._ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            self._ax.set_title(f'{ticker}  ·  3 Month', color=MU, fontsize=8, pad=4)
            self._fig.autofmt_xdate(rotation=30, ha='right')
        except Exception:
            self._ax.text(0.5, 0.5, 'Chart error', ha='center', va='center',
                          color=RE, transform=self._ax.transAxes, fontsize=9)

        self._fig.patch.set_facecolor(BG)
        self._chart_canvas.draw()

    def _render_news(self, news):
        self._news_text.config(state='normal')
        self._news_text.delete('1.0', 'end')

        if not news:
            self._news_text.insert('end', 'No recent news found.', 'meta')
            self._sent_lbl.config(text='', fg=MU)
        else:
            bulls = sum(1 for n in news if n['sentiment'] > 0)
            bears = sum(1 for n in news if n['sentiment'] < 0)
            ratio = (bulls - bears) / len(news)
            if   ratio >= 0.4:  st, sc = 'Bullish', GR
            elif ratio <= -0.4: st, sc = 'Bearish', RE
            else:               st, sc = 'Neutral', MU
            self._sent_lbl.config(text=st, fg=sc)

            for n in news:
                if n['sentiment'] > 0:   tag, icon = 'bull', '▲ '
                elif n['sentiment'] < 0: tag, icon = 'bear', '▼ '
                else:                    tag, icon = 'neu',  '–  '
                self._news_text.insert('end', icon, tag)
                self._news_text.insert('end', n['title'] + '\n', 'title')
                parts = [n['publisher']]
                if n['time']:
                    diff  = datetime.now(timezone.utc) - n['time']
                    mins  = int(diff.total_seconds() / 60)
                    parts.append(f"{mins}m ago" if mins < 60
                                 else f"{mins//60}h ago" if mins < 1440
                                 else f"{mins//1440}d ago")
                self._news_text.insert('end', '  ·  '.join(parts) + '\n\n', 'meta')

        self._news_text.config(state='disabled')

    # ── Filter & populate ──────────────────────────────────────────────────────

    def _apply_filter(self):
        try:
            min_s = int(self._min_score.get())
        except ValueError:
            min_s = 0
        sort  = self._sort_by.get()

        filtered = [r for r in self._results if r['score'] >= min_s]
        key_map = {
            'Score':     lambda x: (-x['score'], x['ticker']),
            'Change %':  lambda x: (-(x['change'] or 0), x['ticker']),
            'Rel. Vol':  lambda x: (-(x['rel_vol'] or 0), x['ticker']),
            'Upside %':  lambda x: (-(x['upside'] or 0), x['ticker']),
            'Ticker':    lambda x: x['ticker'],
        }
        filtered.sort(key=key_map.get(sort, key_map['Score']))
        self._populate_tree(filtered)
        self._cnt_lbl.config(text=f'{len(filtered)} stock{"s" if len(filtered) != 1 else ""}')

    def _populate_tree(self, results):
        self._tree.delete(*self._tree.get_children())
        for i, r in enumerate(results, 1):
            p   = r['price']
            ptx = f"${p:.4f}" if p and p < 1 else (f"${p:.2f}" if p else '—')
            chg = r['change'] or 0
            ctx = f"{'+' if chg >= 0 else ''}{chg:.2f}%"
            rv  = r['rel_vol']
            rvtx = f"{rv:.1f}×" if rv else '—'
            up   = r['upside']
            uptx = ('AT HI' if up is not None and up <= 0
                    else f'+{up:.1f}%' if up else '—')
            sigs = '  '.join(s for s, _ in r['signals'])
            tag  = 'hot' if r['score'] >= 70 else ('warm' if r['score'] >= 45 else 'cool' if r['score'] >= 25 else '')
            self._tree.insert('', 'end',
                values=(i, r['ticker'], (r['name'] or '')[:26],
                        ptx, ctx, rvtx, uptx, r['score'], sigs),
                tags=(tag,))

    # ── Queue polling ──────────────────────────────────────────────────────────

    def _poll_queues(self):
        while not self._scan_q.empty():
            msg = self._scan_q.get_nowait()
            if msg[0] == 'status':
                self._status_lbl.config(text=msg[1])
                self._prog_lbl.config(text=msg[1])
            elif msg[0] == 'done':
                self._results = msg[1]
                self._scan_btn.config(state='normal', text='⚡  Scan Market')
                t = datetime.now().strftime('%I:%M %p')
                self._status_lbl.config(text=f'Last scan: {t} · {len(self._results)} stocks scored')
                self._prog_lbl.config(text=f'{len(self._results)} stocks analysed')
                self._apply_filter()
            elif msg[0] == 'error':
                self._scan_btn.config(state='normal', text='⚡  Scan Market')
                self._status_lbl.config(text=f'Scan error: {msg[1]}')

        while not self._detail_q.empty():
            msg = self._detail_q.get_nowait()
            ticker = msg[1]
            if ticker != self._current:
                continue
            if msg[0] == 'ok':
                _, _, hist, news, info, targets, factors = msg
                self._draw_chart(hist, ticker)
                self._render_news(news)
                self._render_predictions(targets, factors)
                # Merge fundamentals into quote dict and re-score
                res = next((r for r in self._results if r['ticker'] == ticker), None)
                if res:
                    fund_fields = ('floatShares','shortPercentOfFloat','trailingPE',
                                   'forwardPE','revenueGrowth','profitMargins','earningsGrowth')
                    for f in fund_fields:
                        if info.get(f) is not None:
                            res['quote'][f] = info[f]
                    res['name'] = info.get('shortName') or info.get('longName') or ticker
                    self._d_name.config(text=res['name'])
                    self._render_stats(res['quote'])
                    new_score, new_sigs = score_stock(res['quote'], hist, news)
                    res['score']   = new_score
                    res['signals'] = new_sigs
            elif msg[0] == 'err':
                self._render_news([])
                self._render_predictions(None, [])

        self.root.after(120, self._poll_queues)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _col_sort(self, col):
        mapping = {'score':'Score','change':'Change %','relvol':'Rel. Vol',
                   'upside':'Upside %','ticker':'Ticker'}
        if col in mapping:
            self._sort_by.set(mapping[col])
            self._apply_filter()

    def _fmt(self, v):
        if v is None: return '—'
        if v >= 1e12: return f'{v/1e12:.2f}T'
        if v >= 1e9:  return f'{v/1e9:.2f}B'
        if v >= 1e6:  return f'{v/1e6:.2f}M'
        return f'{v:,.0f}'

    def _update_market_status(self):
        try:
            try:
                from zoneinfo import ZoneInfo
                ny = datetime.now(ZoneInfo('America/New_York'))
            except ImportError:
                from datetime import timezone as tz
                import time as _time
                offset = -4 if _time.localtime().tm_isdst else -5
                ny = datetime.now(timezone(timedelta(hours=offset)))
            d = ny.weekday()
            m = ny.hour * 60 + ny.minute
            if d >= 5:                    lbl, col = 'Weekend Closed', RE
            elif 240 <= m < 570:          lbl, col = '● Pre-Market',   YE
            elif 570 <= m < 960:          lbl, col = '● Market Open',  GR
            elif 960 <= m < 1200:         lbl, col = '● After Hours',  YE
            else:                         lbl, col = '● Market Closed', RE
        except Exception:
            lbl, col = '● Status Unknown', MU
        self._mkt_lbl.config(text=lbl, fg=col)
        self.root.after(60_000, self._update_market_status)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.configure(bg=BG)
    try:
        root.iconbitmap(default='')
    except Exception:
        pass
    StockPickerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
