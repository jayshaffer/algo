-- Ensure buy/sell decisions always have a price.
-- Hold/skip decisions can have NULL price (they're not trades).
--
-- Before applying: reclassify existing null-price trades as 'skip'
-- so they don't violate the constraint.
UPDATE decisions SET action = 'skip'
WHERE action IN ('buy', 'sell') AND price IS NULL;

ALTER TABLE decisions ADD CONSTRAINT chk_trade_price
CHECK (action NOT IN ('buy', 'sell') OR price IS NOT NULL);
