# Cloudflare Pages Deploy Design

## Problem

The current public dashboard pipeline uses `git add → commit → push` to deploy JSON data files to a GitHub Pages repository. This is fragile:

- Requires SSH keys mounted into Docker container
- Requires a pre-cloned git repo on the host
- Silent failures when auth breaks (network, key rotation)
- No rollback on bad data
- Static assets must be manually copied to the separate repo

## Solution

Replace the git-based deploy with **Cloudflare Pages Direct Upload via Wrangler CLI**. The data gathering and JSON writing pipeline stays identical — only the final deploy step changes.

## Architecture

```
gather_dashboard_data()  →  write_json_files()  →  deploy_to_cloudflare()
         (same)                  (same)              (replaces push_to_github)
```

### Deploy Flow

1. Create a temp directory
2. Copy static assets from `public_dashboard/` (HTML, CSS, JS)
3. Write `data/*.json` files into it
4. Run `wrangler pages deploy <dir> --project-name <project> --branch main`
5. Clean up temp directory

### What This Eliminates

- No SSH keys in Docker
- No git repo cloned on host
- No `~/.ssh` and `~/.gitconfig` volume mounts
- No `git` package in Docker image
- No `DASHBOARD_REPO_PATH` env var
- No manual asset copying to a separate repo

## Changes

### Dockerfile

Remove `git`, add Node.js + wrangler:

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g wrangler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./v2/requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt
CMD ["python", "-m", "v2.main"]
```

### docker-compose.yml

Remove git-related volume mounts:

```yaml
trading:
  volumes:
    - ./trading:/app/trading:ro
    - ./v2:/app/v2:ro
    - ./dashboard:/app/dashboard:ro
    - ./public_dashboard:/app/public_dashboard:ro  # NEW: for static assets
    - ./logs:/app/logs
  # REMOVED: ${DASHBOARD_REPO_PATH}, ~/.ssh, ~/.gitconfig mounts
  # REMOVED: DASHBOARD_REPO_PATH environment variable
```

### Environment Variables

**Remove:**
- `DASHBOARD_REPO_PATH`

**Add:**
- `CLOUDFLARE_ACCOUNT_ID` — from Cloudflare dashboard
- `CLOUDFLARE_API_TOKEN` — with "Cloudflare Pages: Edit" permission
- `CLOUDFLARE_PAGES_PROJECT` — the Pages project name (e.g., `bikini-bottom-capital`)

**Keep:**
- `DASHBOARD_URL` — still appended to social posts

### v2/dashboard_publish.py

Replace `push_to_github()` with `deploy_to_cloudflare()`. Update `run_dashboard_stage()` to:

1. Check `CLOUDFLARE_PAGES_PROJECT` is set (replaces `DASHBOARD_REPO_PATH` check)
2. Gather data (unchanged)
3. Assemble deploy directory in temp dir:
   - Copy `public_dashboard/` static assets (index.html, styles.css, app.js)
   - Write `data/*.json` via existing `write_json_files()`
4. Call `deploy_to_cloudflare(deploy_dir)`
5. Clean up temp dir

The `deploy_to_cloudflare()` function:

```python
def deploy_to_cloudflare(deploy_dir: str) -> bool:
    project = os.environ.get("CLOUDFLARE_PAGES_PROJECT")
    result = subprocess.run(
        ["wrangler", "pages", "deploy", deploy_dir,
         "--project-name", project, "--branch", "main"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wrangler deploy failed: {result.stderr.strip()}")
    return True
```

Wrangler authenticates via `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` environment variables automatically.

### Tests

Update `tests/v2/test_dashboard_publish.py`:

- Replace `push_to_github` tests with `deploy_to_cloudflare` tests
- Mock `subprocess.run` for wrangler calls (same pattern as git mocks)
- Test: successful deploy, wrangler failure, missing env var
- Update `run_dashboard_stage` tests: check for `CLOUDFLARE_PAGES_PROJECT` instead of `DASHBOARD_REPO_PATH`
- Add test for static asset assembly (files copied correctly)

## Setup (One-Time)

1. Create Cloudflare Pages project:
   - Go to Cloudflare dashboard → Pages → Create a project → Direct Upload
   - Name it (e.g., `bikini-bottom-capital`)
   - Note the project name

2. Create API token:
   - Cloudflare dashboard → My Profile → API Tokens → Create Token
   - Permissions: Account → Cloudflare Pages → Edit
   - Note the token

3. Add to `.env`:
   ```
   CLOUDFLARE_ACCOUNT_ID=your_account_id
   CLOUDFLARE_API_TOKEN=your_api_token
   CLOUDFLARE_PAGES_PROJECT=bikini-bottom-capital
   ```

4. Rebuild Docker image: `docker compose build trading`

## Non-Goals

- No changes to data gathering queries
- No changes to JSON structure or frontend
- No changes to social post integration
- No data validation/rollback (separate concern)
