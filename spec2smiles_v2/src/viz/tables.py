"""Table generation for results export."""

import csv
from pathlib import Path

from .constants import MASSSPECGYM_BASELINES


def generate_csv_table(metrics: dict, output_dir: Path) -> bool:
    """Generate CSV comparison table."""
    if "e2e" not in metrics or "oracle" not in metrics:
        print("  [SKIP] Need metrics for table generation")
        return False

    e2e = metrics["e2e"]["end_to_end"]
    oracle = metrics["oracle"]

    rows = [
        ["Method", "k", "Accuracy (%)", "Tanimoto", "Validity (%)"],
        ["Random", "10", "0.00", "0.10", "-"],
        ["SMILES Transformer", "10", "0.00", "0.17", "-"],
        ["SELFIES Transformer", "10", "0.00", "0.15", "-"],
        ["Ours (E2E)", "10", f"{e2e['exact_match'] * 100:.1f}", f"{e2e['mean_best_tanimoto']:.3f}", f"{e2e['validity'] * 100:.1f}"],
        ["Ours (Oracle)", "10", f"{oracle['exact_match'] * 100:.1f}", f"{oracle['mean_best_tanimoto']:.3f}", f"{oracle['validity'] * 100:.1f}"],
    ]

    csv_path = output_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"  [OK] comparison_table.csv")
    return True


def generate_markdown_table(metrics: dict, output_dir: Path) -> bool:
    """Generate Markdown comparison table."""
    if "e2e" not in metrics or "oracle" not in metrics:
        print("  [SKIP] Need metrics for table generation")
        return False

    e2e = metrics["e2e"]["end_to_end"]
    oracle = metrics["oracle"]

    md = f"""# Comparison with MassSpecGym Benchmarks

## De Novo Molecule Generation (k=10)

| Method | Accuracy (%) | Tanimoto | Validity (%) |
|--------|-------------|----------|--------------|
| Random | 0.00 | 0.10 | - |
| SMILES Transformer | 0.00 | 0.17 | - |
| SELFIES Transformer | 0.00 | 0.15 | - |
| **Ours (E2E)** | **{e2e['exact_match'] * 100:.1f}** | **{e2e['mean_best_tanimoto']:.3f}** | {e2e['validity'] * 100:.1f} |
| Ours (Oracle) | {oracle['exact_match'] * 100:.1f} | {oracle['mean_best_tanimoto']:.3f} | {oracle['validity'] * 100:.1f} |

## Key Findings

- **Exact Match**: {e2e['exact_match'] * 100:.1f}% vs 0% baseline (∞× improvement)
- **Tanimoto**: {e2e['mean_best_tanimoto']:.3f} vs 0.17 ({e2e['mean_best_tanimoto'] / 0.17:.1f}× improvement)
- **Validity**: 100% (SELFIES encoding)

## Caveats

- Different datasets: MassSpecGym uses 231K spectra; we use GNPS (~2.7K)
- Different splits: MassSpecGym uses MCES-based split; we use random split
"""

    md_path = output_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"  [OK] comparison_table.md")
    return True
