"""Run factor attribution on quantclock output after `quantclock run`.

Usage: python examples_factor_regression.py outputs/clock_backtest.csv
"""
from __future__ import annotations
import argparse
import pandas as pd
from quantclock.factor_data import fetch_fama_french_5_factors
from quantclock.factors import factor_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backtest_csv")
    args = parser.parse_args()
    backtest = pd.read_csv(args.backtest_csv, parse_dates=["date"]).set_index("date")
    ff = fetch_fama_french_5_factors()
    model = factor_regression(backtest["net_return"], ff.drop(columns=["RF"]),
                              maxlags=3, risk_free_returns=ff["RF"])
    print("N =", int(model.nobs))
    print("coefficients\n", model.params)
    print("HAC t-statistics\n", model.tvalues)
    print("HAC p-values\n", model.pvalues)
    print("R-squared =", float(model.rsquared))


if __name__ == "__main__":
    main()
