"""Command-line gold regime report.

Usage:
    python markov.py                 # GLD, 3 regimes, with chart
    python markov.py GC=F --states 2 --no-plot
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from regime import fetch_prices, fit_regimes

COLORS = {"Calm": "tab:green", "Normal": "tab:orange",
          "Elevated": "tab:red", "Turbulent": "darkred"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="GLD")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rm = fit_regimes(fetch_prices(args.ticker, args.start), n_states=args.states)
    s = rm.summary()

    pd.set_option("display.float_format", "{:.2f}".format)
    print(f"\n{args.ticker} as of {s['as_of']}: {s['price']:.2f}")
    print(f"Current regime: {s['regime']} ({s['confidence']:.0%} confidence)")
    print(f"In this regime for {s['days_in_regime']} trading days; "
          f"typical length is ~{s['expected_duration']:.0f} days\n")
    print("How each regime has behaved:")
    print(rm.regime_stats().to_string(), "\n")
    print("Chance of being in each regime ahead:")
    print(rm.outlook().map("{:.0%}".format).to_string(), "\n")
    print("Not financial advice. Regimes describe volatility, not direction.")

    if args.no_plot:
        return
    df = rm.df
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df.index, df["Price"], color="lightgray", zorder=1)
    for name in rm.names:
        sub = df[df["Label"] == name]
        ax.scatter(sub.index, sub["Price"], s=4, label=name,
                   color=COLORS.get(name), zorder=2)
    ax.set_title(f"{args.ticker} price by regime")
    ax.legend(markerscale=4)
    plt.show()


if __name__ == "__main__":
    main()
