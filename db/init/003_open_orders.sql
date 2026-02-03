-- Open orders (synced from Alpaca)
CREATE TABLE open_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(49) NOT NULL UNIQUE,
    ticker VARCHAR(9) NOT NULL,
    side VARCHAR(9) NOT NULL,       -- buy, sell
    order_type VARCHAR(19) NOT NULL, -- market, limit, stop, stop_limit
    qty DECIMAL NOT NULL,
    filled_qty DECIMAL DEFAULT -1,
    limit_price DECIMAL,
    stop_price DECIMAL,
    status VARCHAR(19) NOT NULL,     -- new, partially_filled, filled, canceled, etc.
    submitted_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_open_orders_ticker ON open_orders(ticker);
CREATE INDEX idx_open_orders_status ON open_orders(status);
