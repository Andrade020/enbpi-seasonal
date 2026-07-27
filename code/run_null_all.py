"""Null (homoskedastic) case with every calibration scheme, n_rep=200."""
import os
import pandas as pd
import simulation_baselines as sb

if __name__ == "__main__":
    acc = sb.monte_carlo_multi(200, 240, 1.0, 1.0, seed_base=4000)
    df = sb.season_table(acc, 200)
    df.insert(0, "T", 240)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "tables", "X4_null_all_methods.csv")
    df.to_csv(out, index=False)
    print(f"{'method':<11} {'overall':>8} {'spread':>7} {'width':>7}")
    for m in sb.METHODS:
        d = df[df["method"] == m]
        per = d[d["season"] > 0]
        print(f"{m:<11} {d[d['season']==0]['coverage_mean'].values[0]:>8.3f} "
              f"{per['coverage_mean'].max()-per['coverage_mean'].min():>7.3f} "
              f"{per['width_mean'].mean():>7.2f}")
