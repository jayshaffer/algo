"""Alpaca Learning Platform - Trading Agent"""

import os
import sys
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


def get_trading_client() -> TradingClient:
    """Create Alpaca trading client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    paper = "paper" in base_url
    return TradingClient(api_key, secret_key, paper=paper)


def get_data_client() -> StockHistoricalDataClient:
    """Create Alpaca data client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    return StockHistoricalDataClient(api_key, secret_key)


def check_connectivity():
    """Verify connectivity to Alpaca API and print account status."""
    print(f"[{datetime.now().isoformat()}] Checking Alpaca API connectivity...")

    trading_client = get_trading_client()
    account = trading_client.get_account()

    print(f"  Account ID: {account.id}")
    print(f"  Status: {account.status}")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")
    print(f"  Day Trade Count: {account.daytrade_count}")
    print(f"  Pattern Day Trader: {account.pattern_day_trader}")

    return account


def get_positions():
    """Fetch current positions from Alpaca."""
    trading_client = get_trading_client()
    positions = trading_client.get_all_positions()

    if not positions:
        print("  No open positions")
        return []

    for pos in positions:
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        print(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
              f"(P&L: ${pnl:+,.2f} / {pnl_pct:+.2f}%)")

    return positions


def get_quote(symbol: str):
    """Get latest quote for a symbol."""
    data_client = get_data_client()
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quotes = data_client.get_stock_latest_quote(request)

    quote = quotes[symbol]
    print(f"  {symbol}: Bid ${quote.bid_price:.2f} / Ask ${quote.ask_price:.2f}")
    return quote


def main():
    """Main entry point for trading agent."""
    print("=" * 60)
    print("Alpaca Learning Platform - Trading Agent")
    print("=" * 60)

    try:
        print("\n[Account Status]")
        check_connectivity()

        print("\n[Current Positions]")
        get_positions()

        print("\n[Sample Quote]")
        get_quote("AAPL")

        print("\n" + "=" * 60)
        print("Connectivity check passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
