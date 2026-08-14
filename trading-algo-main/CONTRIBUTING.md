# Contributing to trading‑algo 🚀

Thank you for your interest in contributing! This guide explains how to report bugs, suggest features, and contribute code, documentation, or tooling improvements.

## Table of Contents
- [How You Can Contribute](#how-you-can-contribute)
- [Getting Started](#getting-started)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Pull Request Process](#pull-request-process)
- [Code Style & Testing](#code-style--testing)
- [Project Structure & Workflow](#project-structure--workflow)
- [Getting Help](#getting-help)
- [Legal](#legal)


## How You Can Contribute
Contributions welcome via:
- New or improved trading strategies
- Broker integrations
- Tests and documentation
- Issue triage and support
- Standardizing and scaling of the Code

## Getting Started
1. Fork and clone:
   ```
   git clone https://github.com/<your‑username>/trading-algo.git
   cd trading-algo
   ```
2. Create a branch:
   ```
   git checkout -b feature/your-feature-name
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or
   ```
   uv sync
   ```
4. Copy `.sample.env` to `.env`, and update credentials.

## Reporting Bugs
Provide:
- A clear title
- Steps to reproduce
- Environment info
- Expected vs actual behavior
- Logs or minimal code snippets

## Suggesting Enhancements
Include:
- Clear title
- Motivation or use-case
- Proposed API or behavior changes
- Links to related issues

## Pull Request Process
Target the default branch:
- Describe changes and link issues
- Provide tests for logic changes
- Ensure formatting/linting
- Respond to review feedback

## Code Style & Testing
- Follow existing conventions
- Use clear commit messages
- Add comments for complex logic
- Update docs when APIs change

## Project Structure & Workflow
```
/strategy/        # Trading strategies
/brokers/         # Broker connectors
/utils/           # Dispatcher, orders, logging
.sample.env
README.md
```

To add a strategy: document purpose, parameters, risk considerations, and optionally backtest results.

## Getting Help
If stuck, open a new issue with context.

## Legal
This is educational — not financial advice.

Thank you for contributing to **trading‑algo**—every improvement matters! 🎉
