# Comparison with MassSpecGym Benchmarks

## De Novo Molecule Generation (k=10)

| Method | Accuracy (%) | Tanimoto | Validity (%) |
|--------|-------------|----------|--------------|
| Random | 0.00 | 0.10 | - |
| SMILES Transformer | 0.00 | 0.17 | - |
| SELFIES Transformer | 0.00 | 0.15 | - |
| **Ours (E2E)** | **36.1** | **0.575** | 100.0 |
| Ours (Oracle) | 82.2 | 0.943 | 100.0 |

## Key Findings

- **Exact Match**: 36.1% vs 0% baseline (∞× improvement)
- **Tanimoto**: 0.575 vs 0.17 (3.4× improvement)
- **Validity**: 100% (SELFIES encoding)

## Caveats

- Different datasets: MassSpecGym uses 231K spectra; we use GNPS (~2.7K)
- Different splits: MassSpecGym uses MCES-based split; we use random split
