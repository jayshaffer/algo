# Cloudflare Pages Deploy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fragile git-push-to-GitHub-Pages deploy with Cloudflare Pages Direct Upload via Wrangler CLI.

**Architecture:** Keep the existing data gathering pipeline (`gather_dashboard_data` → `write_json_files`) unchanged. Replace `push_to_github()` with `deploy_to_cloudflare()` that runs `wrangler pages deploy`. Assemble a complete deploy directory (static assets + JSON data) in a temp dir each session.

**Tech Stack:** Wrangler CLI (Node.js), Cloudflare Pages API, Python subprocess

---

### Task 1: Add `deploy_to_cloudflare()` Function

**Files:**
- Modify: `v2/dashboard_publish.py` (add new function, keep `push_to_github` for now)
- Test: `tests/v2/test_dashboard_publish.py`

**Step 1: Write the failing tests**

Add a new test class `TestDeployToCloudflare` in `tests/v2/test_dashboard_publish.py`. Add `deploy_to_cloudflare` to the imports at the top.

```python
class TestDeployToCloudflare:
    @patch("v2.dashboard_publish.subprocess.run")
    def test_deploys_successfully(self, mock_run):
        """Runs wrangler pages deploy with correct args."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Published!")
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}):
            result = deploy_to_cloudflare("/tmp/deploy")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "wrangler"
        assert call_args[1:3] == ["pages", "deploy"]
        assert "/tmp/deploy" in call_args
        assert "--project-name" in call_args
        assert "my-dashboard" in call_args

    @patch("v2.dashboard_publish.subprocess.run")
    def test_raises_on_wrangler_failure(self, mock_run):
        """RuntimeError raised when wrangler exits non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Auth failed")
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}):
            with pytest.raises(RuntimeError, match="Auth failed"):
                deploy_to_cloudflare("/tmp/deploy")

    def test_raises_when_project_not_set(self):
        """RuntimeError raised when CLOUDFLARE_PAGES_PROJECT missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLOUDFLARE_PAGES_PROJECT", None)
            with pytest.raises(RuntimeError, match="CLOUDFLARE_PAGES_PROJECT"):
                deploy_to_cloudflare("/tmp/deploy")
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestDeployToCloudflare -v`
Expected: FAIL with ImportError (deploy_to_cloudflare not defined yet)

**Step 3: Write the implementation**

Add to `v2/dashboard_publish.py` after `push_to_github()`:

```python
def deploy_to_cloudflare(deploy_dir: str) -> bool:
    """Deploy dashboard directory to Cloudflare Pages via wrangler.

    Requires CLOUDFLARE_PAGES_PROJECT env var.
    Wrangler authenticates via CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN env vars.

    Returns True if deployed successfully.
    Raises RuntimeError on failure.
    """
    project = os.environ.get("CLOUDFLARE_PAGES_PROJECT")
    if not project:
        raise RuntimeError("CLOUDFLARE_PAGES_PROJECT not set")

    result = subprocess.run(
        ["wrangler", "pages", "deploy", deploy_dir,
         "--project-name", project, "--branch", "main"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wrangler deploy failed: {result.stderr.strip()}")

    logger.info("Deployed to Cloudflare Pages: %s", result.stdout.strip())
    return True
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestDeployToCloudflare -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: add deploy_to_cloudflare function"
```

---

### Task 2: Add `assemble_deploy_dir()` Function

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

This function copies static assets from `public_dashboard/` and writes JSON data into a temp directory for wrangler to deploy.

**Step 1: Write the failing tests**

Add `assemble_deploy_dir` to the import list. Add a new test class:

```python
class TestAssembleDeployDir:
    def _sample_data(self):
        return {
            "summary": {"portfolio_value": 100000},
            "snapshots": [{"date": "2025-06-15"}],
            "positions": [{"ticker": "AAPL"}],
            "decisions": [{"action": "buy"}],
            "theses": [{"direction": "long"}],
        }

    def test_copies_static_assets(self, tmp_path):
        """index.html, styles.css, app.js are copied to deploy dir."""
        # Create fake static assets
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "index.html").write_text("<html>test</html>")
        (assets_dir / "styles.css").write_text("body { color: red; }")
        (assets_dir / "app.js").write_text("console.log('hi');")
        (assets_dir / "README.md").write_text("Docs")  # Should NOT be copied

        deploy_dir = tmp_path / "deploy"
        assemble_deploy_dir(self._sample_data(), str(deploy_dir), str(assets_dir))

        assert (deploy_dir / "index.html").exists()
        assert (deploy_dir / "styles.css").exists()
        assert (deploy_dir / "app.js").exists()
        assert not (deploy_dir / "README.md").exists()

    def test_writes_json_data_files(self, tmp_path):
        """data/*.json files are written correctly."""
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "index.html").write_text("<html>")
        (assets_dir / "styles.css").write_text("")
        (assets_dir / "app.js").write_text("")

        deploy_dir = tmp_path / "deploy"
        data = self._sample_data()
        assemble_deploy_dir(data, str(deploy_dir), str(assets_dir))

        for key in ("summary", "snapshots", "positions", "decisions", "theses"):
            json_path = deploy_dir / "data" / f"{key}.json"
            assert json_path.exists()
            with open(json_path) as f:
                assert json.load(f) == data[key]

    def test_creates_deploy_dir_if_missing(self, tmp_path):
        """Deploy directory is created automatically."""
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "index.html").write_text("<html>")
        (assets_dir / "styles.css").write_text("")
        (assets_dir / "app.js").write_text("")

        deploy_dir = tmp_path / "deploy" / "nested"
        assert not deploy_dir.exists()

        assemble_deploy_dir(self._sample_data(), str(deploy_dir), str(assets_dir))

        assert deploy_dir.exists()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestAssembleDeployDir -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Add to `v2/dashboard_publish.py` (add `import shutil` at the top):

```python
import shutil

# Static asset filenames to copy from public_dashboard/
_STATIC_ASSETS = ("index.html", "styles.css", "app.js")


def assemble_deploy_dir(data: dict, deploy_dir: str, assets_dir: str) -> str:
    """Assemble a complete deploy directory with static assets and JSON data.

    Args:
        data: Dashboard data dict (from gather_dashboard_data).
        deploy_dir: Path to create/populate the deploy directory.
        assets_dir: Path to public_dashboard/ directory containing static assets.

    Returns the deploy_dir path.
    """
    os.makedirs(deploy_dir, exist_ok=True)

    # Copy static assets
    for filename in _STATIC_ASSETS:
        src = os.path.join(assets_dir, filename)
        dst = os.path.join(deploy_dir, filename)
        shutil.copy2(src, dst)

    # Write JSON data files
    write_json_files(data, deploy_dir)

    return deploy_dir
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestAssembleDeployDir -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: add assemble_deploy_dir for Cloudflare deploy"
```

---

### Task 3: Update `run_dashboard_stage()` to Use Cloudflare Deploy

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

Replace the git-based deploy with the new Cloudflare-based deploy. Remove `push_to_github()`. Update `run_dashboard_stage()` to use `assemble_deploy_dir()` + `deploy_to_cloudflare()`.

**Step 1: Update the test imports**

In `tests/v2/test_dashboard_publish.py`, update the import block:

```python
from v2.dashboard_publish import (
    DashboardStageResult,
    _DecimalEncoder,
    _build_summary,
    assemble_deploy_dir,
    deploy_to_cloudflare,
    gather_dashboard_data,
    run_dashboard_stage,
    write_json_files,
)
```

Remove `push_to_github` from imports (it no longer exists).

**Step 2: Delete `TestPushToGithub` class**

Remove the entire `TestPushToGithub` class (lines 342-385 approximately). These tests are no longer relevant.

**Step 3: Rewrite `TestRunDashboardStage` tests**

Replace the entire `TestRunDashboardStage` class with:

```python
class TestRunDashboardStage:
    @patch("v2.dashboard_publish.deploy_to_cloudflare", return_value=True)
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_happy_path(self, mock_deposits, mock_gather, mock_assemble, mock_deploy):
        """Full pipeline runs and returns published=True."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage(session_date=date(2025, 6, 15))

        assert result.published is True
        assert result.skipped is False
        assert result.errors == []
        mock_deposits.assert_called_once()
        mock_gather.assert_called_once_with(date(2025, 6, 15), net_deposits=Decimal("100000"))
        mock_assemble.assert_called_once()
        mock_deploy.assert_called_once()

    def test_skipped_when_no_project_set(self):
        """Returns skipped=True when CLOUDFLARE_PAGES_PROJECT not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLOUDFLARE_PAGES_PROJECT", None)
            result = run_dashboard_stage()

        assert result.skipped is True
        assert result.published is False
        assert result.errors == []

    @patch("v2.dashboard_publish.gather_dashboard_data", side_effect=Exception("DB down"))
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_gather_error(self, mock_deposits, mock_gather):
        """Error in gather step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Data gathering failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.assemble_deploy_dir", side_effect=Exception("Disk full"))
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_assemble_error(self, mock_deposits, mock_assemble, mock_gather):
        """Error in assemble step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Deploy assembly failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.deploy_to_cloudflare", side_effect=RuntimeError("Auth failed"))
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_deploy_error(self, mock_deposits, mock_deploy, mock_assemble, mock_gather):
        """Error in deploy step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Cloudflare deploy failed" in result.errors[0]

    @patch("v2.dashboard_publish.deploy_to_cloudflare", return_value=True)
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.get_net_deposits", side_effect=Exception("Alpaca down"))
    def test_continues_when_net_deposits_fails(self, mock_deposits, mock_gather, mock_assemble, mock_deploy):
        """Pipeline continues with net_deposits=None if Alpaca call fails."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage(session_date=date(2025, 6, 15))

        assert result.published is True
        assert result.errors == []
        mock_gather.assert_called_once_with(date(2025, 6, 15), net_deposits=None)
```

**Step 4: Run the updated tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestRunDashboardStage -v`
Expected: FAIL (run_dashboard_stage still uses old DASHBOARD_REPO_PATH logic)

**Step 5: Update `run_dashboard_stage()` implementation**

Replace the entire `run_dashboard_stage()` function in `v2/dashboard_publish.py`:

```python
# Path to static assets directory (relative to project root)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public_dashboard")


def run_dashboard_stage(session_date: Optional[date] = None) -> DashboardStageResult:
    """Run the full dashboard publish pipeline: gather -> assemble -> deploy."""
    if session_date is None:
        session_date = date.today()

    result = DashboardStageResult()

    project = os.environ.get("CLOUDFLARE_PAGES_PROJECT")
    if not project:
        result.skipped = True
        logger.info("Dashboard stage skipped — CLOUDFLARE_PAGES_PROJECT not set")
        return result

    # Fetch net deposits from Alpaca for accurate return calculation
    net_deposits = None
    try:
        net_deposits = get_net_deposits()
    except Exception as e:
        logger.warning("Could not fetch net deposits from Alpaca: %s", e)

    # Gather data
    try:
        data = gather_dashboard_data(session_date, net_deposits=net_deposits)
    except Exception as e:
        result.errors.append(f"Data gathering failed: {e}")
        logger.error("Failed to gather dashboard data: %s", e)
        return result

    # Assemble deploy directory
    import tempfile
    deploy_dir = tempfile.mkdtemp(prefix="dashboard_deploy_")
    try:
        assemble_deploy_dir(data, deploy_dir, _ASSETS_DIR)
    except Exception as e:
        result.errors.append(f"Deploy assembly failed: {e}")
        logger.error("Failed to assemble deploy directory: %s", e)
        return result

    # Deploy to Cloudflare
    try:
        deploy_to_cloudflare(deploy_dir)
    except Exception as e:
        result.errors.append(f"Cloudflare deploy failed: {e}")
        logger.error("Failed to deploy to Cloudflare: %s", e)
        return result
    finally:
        shutil.rmtree(deploy_dir, ignore_errors=True)

    result.published = True
    logger.info("Dashboard publish complete (published=%s)", result.published)
    return result
```

Also remove the `push_to_github()` function entirely from the module.

**Step 6: Run all dashboard tests**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: switch run_dashboard_stage to Cloudflare Pages deploy"
```

---

### Task 4: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

**Step 1: Replace git with Node.js + wrangler**

Replace the current Dockerfile content:

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g wrangler \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./v2/requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "-m", "v2.main"]
```

**Step 2: Verify Dockerfile builds**

Run: `docker compose build trading`
Expected: Build succeeds, wrangler is available

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "build: replace git with Node.js and wrangler in Docker image"
```

---

### Task 5: Update docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Remove git volume mounts, add public_dashboard mount**

Update the `trading` service volumes and environment:

Remove these lines:
```yaml
    environment:
      - DASHBOARD_REPO_PATH=/app/dashboard_repo
```

```yaml
      - ${DASHBOARD_REPO_PATH:-/tmp/no_dashboard}:/app/dashboard_repo
      - ~/.ssh:/root/.ssh:ro
      - ~/.gitconfig:/root/.gitconfig:ro
```

Add this volume:
```yaml
      - ./public_dashboard:/app/public_dashboard:ro
```

The final `trading` service should look like:

```yaml
  trading:
    build: .
    depends_on:
      ollama:
        condition: service_started
      db:
        condition: service_healthy
    env_file:
      - .env
    volumes:
      - ./trading:/app/trading:ro
      - ./v2:/app/v2:ro
      - ./dashboard:/app/dashboard:ro
      - ./public_dashboard:/app/public_dashboard:ro
      - ./logs:/app/logs
    command: ["sleep", "infinity"]
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "build: remove git mounts, add public_dashboard volume"
```

---

### Task 6: Update .env.example

**Files:**
- Modify: `.env.example`

**Step 1: Replace dashboard env vars**

Remove:
```
DASHBOARD_REPO_PATH=/path/to/your-org.github.io
```

Add (in the same section):
```
# Cloudflare Pages (public dashboard)
# Create project: Cloudflare dashboard → Pages → Create → Direct Upload
# Create token: My Profile → API Tokens → Cloudflare Pages: Edit
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_PAGES_PROJECT=your-project-name
```

Keep:
```
DASHBOARD_URL=https://your-project.pages.dev
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example for Cloudflare Pages"
```

---

### Task 7: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all v2 tests**

Run: `python3 -m pytest tests/v2/ -v`
Expected: All tests pass

**Step 2: Run full test suite with coverage**

Run: `python3 -m pytest tests/ --cov=v2 --cov=trading --cov=dashboard -v`
Expected: All tests pass, no regressions

**Step 3: Verify no references to old env var remain**

Run: `grep -r "DASHBOARD_REPO_PATH" --include="*.py" v2/`
Expected: No matches (only in old design docs, which is fine)
