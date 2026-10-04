"""Minimal end-to-end example. Run: python scripts/quickstart.py"""

from alpha_forecast.pipeline import run_comparison

if __name__ == "__main__":
    board = run_comparison("AAPL", horizon=1)
    print(board.to_string(index=False))
