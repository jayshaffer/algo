-- Alpaca Learning Platform Schema

-- Ticker-specific news signals
CREATE TABLE news_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    category VARCHAR(20),      -- earnings, guidance, analyst, product, legal, noise
    sentiment VARCHAR(10),     -- bullish, bearish, neutral
    confidence VARCHAR(10),    -- high, medium, low
    published_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_signals_ticker ON news_signals(ticker);
CREATE INDEX idx_news_signals_published_at ON news_signals(published_at);

-- Macro/political news signals
CREATE TABLE macro_signals (
    id SERIAL PRIMARY KEY,
    headline TEXT NOT NULL,
    category VARCHAR(20),      -- fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors TEXT[],   -- tech, finance, energy, healthcare, defense, all
    sentiment VARCHAR(10),     -- bullish, bearish, neutral
    published_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_macro_signals_category ON macro_signals(category);
CREATE INDEX idx_macro_signals_published_at ON macro_signals(published_at);

-- Portfolio state
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL UNIQUE,
    shares DECIMAL NOT NULL,
    avg_cost DECIMAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_positions_ticker ON positions(ticker);

-- Open orders (synced from Alpaca)
CREATE TABLE open_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,       -- buy, sell
    order_type VARCHAR(20) NOT NULL, -- market, limit, stop, stop_limit
    qty DECIMAL NOT NULL,
    filled_qty DECIMAL DEFAULT 0,
    limit_price DECIMAL,
    stop_price DECIMAL,
    status VARCHAR(20) NOT NULL,     -- new, partially_filled, filled, canceled, etc.
    submitted_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_open_orders_ticker ON open_orders(ticker);
CREATE INDEX idx_open_orders_status ON open_orders(status);

-- Account snapshots (equity curve tracking)
CREATE TABLE account_snapshots (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    cash DECIMAL NOT NULL,
    portfolio_value DECIMAL NOT NULL,
    buying_power DECIMAL NOT NULL,
    long_market_value DECIMAL,
    short_market_value DECIMAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_account_snapshots_date ON account_snapshots(date);

-- Trading decisions and outcomes (learning journal)
CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- buy, sell, hold
    quantity DECIMAL,
    price DECIMAL,
    reasoning TEXT,
    signals_used JSONB,
    account_equity DECIMAL,       -- portfolio value at decision time
    buying_power DECIMAL,         -- available capital at decision time
    outcome_7d DECIMAL,           -- P&L after 7 days (backfilled)
    outcome_30d DECIMAL           -- P&L after 30 days (backfilled)
);

CREATE INDEX idx_decisions_date ON decisions(date);
CREATE INDEX idx_decisions_ticker ON decisions(ticker);

-- Strategy state (what Claude is currently doing)
CREATE TABLE strategy (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    description TEXT,
    watchlist TEXT[],
    risk_tolerance VARCHAR(20),   -- conservative, moderate, aggressive
    focus_sectors TEXT[]
);

CREATE INDEX idx_strategy_date ON strategy(date);
