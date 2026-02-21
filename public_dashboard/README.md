# Bikini Bottom Capital - Public Dashboard

Static site deployed to Cloudflare Pages that displays portfolio performance, holdings, recent trades, and active theses.

## How It Works

The static assets (`index.html`, `styles.css`, `app.js`) and `data/*.json` files are assembled into a deploy directory and published to Cloudflare Pages via `wrangler` after each trading session. No git operations involved — just an HTTP API call.

## Deployment

### 1. Create a Cloudflare Pages project

Go to the Cloudflare dashboard → **Pages** → **Create a project** → **Direct Upload**. Name it (e.g., `bikini-bottom-capital`).

### 2. Create an API token

Go to **My Profile** → **API Tokens** → **Create Token**. Give it **Account: Cloudflare Pages: Edit** permission.

### 3. Configure environment variables

Add these to your `.env`:

```bash
CLOUDFLARE_ACCOUNT_ID=your_account_id        # from Cloudflare dashboard
CLOUDFLARE_API_TOKEN=your_api_token           # the token you just created
CLOUDFLARE_PAGES_PROJECT=bikini-bottom-capital # the project name from step 1
DASHBOARD_URL=https://bikini-bottom-capital.pages.dev  # public URL (appended to social posts)
```

### 4. Run a session

The dashboard publishes as Stage 6 of the session pipeline:

```bash
docker compose exec trading python -m v2.session
```

This will:
1. Query the DB for snapshots, positions, decisions, and theses
2. Assemble a deploy directory with static assets and JSON data
3. Deploy to Cloudflare Pages via `wrangler pages deploy`

To skip the dashboard stage:

```bash
docker compose exec trading python -m v2.session --skip-dashboard
```

## Local Testing

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

Uses the sample `data/*.json` files included in this directory.

## Data Files

All regenerated automatically by `v2/dashboard_publish.py`:

| File | Contents |
|------|----------|
| `summary.json` | Portfolio value, cash, daily P&L, total return |
| `snapshots.json` | Last 90 days of account snapshots (equity curve) |
| `positions.json` | Current holdings with share counts and cost basis |
| `decisions.json` | Last 30 days of trades with reasoning and Alpaca order IDs |
| `theses.json` | Active trade theses with entry/exit triggers |
