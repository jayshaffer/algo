# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpaca Learning Platform - an agentic trading system that uses Claude Code to integrate with the Alpaca trading API, learn from past behavior, and make trading decisions.

**Status:** Pre-alpha (planning phase, no implementation yet)

## Project Goals

- Prove whether agentic trading can find an edge
- Claude Code integrated trading with full Alpaca API access (read/write)
- Daily automation after market close via crontab or similar
- Learning system that journals behavior without blowing out context windows
- Single Alpaca account with adjustable day-to-day strategy
- Local web dashboard for strategy visibility and reasoning

## Planned Architecture

- **Containerization:** Docker + Docker Compose
- **Data Storage:** Small database for learning data journaling
- **API:** Alpaca Trading API
- **Automation:** Crontab or scheduler (runs daily after market close)
- **Dashboard:** Local web interface

## Current Structure

```
docs/plans/       # Planning documents (empty)
requirements.md   # Project requirements
```

## Next Steps

This project needs initial scaffolding:
1. Choose and set up backend framework
2. Configure Alpaca SDK integration
3. Design database schema for learning journal
4. Implement web dashboard
5. Set up Docker Compose configuration
6. Configure daily automation scheduler
