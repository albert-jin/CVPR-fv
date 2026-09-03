"""Cross-dataset tests reported in the revised manuscript.

The analysis unit is one dataset-level mean over five seeds, following
Demsar (2006). It intentionally does not treat seeds as independent datasets.
"""

from scipy.stats import friedmanchisquare, wilcoxon


K4 = {
    "CVPR-FV": [91.7, 57.2, 58.6],
    "Det2Ver-matched": [90.6, 55.1, 53.8],
    "ProToCo": [89.1, 52.0, 49.8],
}

K32 = {
    "CVPR-FV": [94.2, 67.0, 72.7],
    "Det2Ver-matched": [92.8, 68.1, 70.7],
}


def main() -> None:
    omnibus = friedmanchisquare(K4["CVPR-FV"], K4["Det2Ver-matched"], K4["ProToCo"])
    print(f"K=4 Friedman: chi2={omnibus.statistic:.6f}, p={omnibus.pvalue:.9f}")

    posthoc = []
    for baseline in ("Det2Ver-matched", "ProToCo"):
        result = wilcoxon(K4["CVPR-FV"], K4[baseline], alternative="two-sided", method="exact")
        posthoc.append((baseline, float(result.statistic), float(result.pvalue)))
    # Both raw p-values are equal; Holm adjustment for two planned comparisons.
    for baseline, statistic, pvalue in posthoc:
        print(f"K=4 CVPR-FV vs {baseline}: W={statistic:.0f}, p={pvalue:.3f}, Holm p={min(1.0, 2*pvalue):.3f}")

    k32 = wilcoxon(
        K32["CVPR-FV"], K32["Det2Ver-matched"],
        alternative="two-sided", method="exact",
    )
    print(f"K=32 CVPR-FV vs Det2Ver-matched: W={k32.statistic:.0f}, p={k32.pvalue:.3f}")


if __name__ == "__main__":
    main()
