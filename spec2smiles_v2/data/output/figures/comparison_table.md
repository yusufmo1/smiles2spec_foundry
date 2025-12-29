# Comparison with MassSpecGym Benchmarks

## De Novo Molecule Generation (k=10)

| Method | Accuracy (%) | Tanimoto | Validity (%) |
|--------|-------------|----------|--------------|
| Random | 0.00 | 0.10 | - |
| SMILES Transformer | 0.00 | 0.17 | - |
| SELFIES Transformer | 0.00 | 0.15 | - |
| **Ours (E2E)** | **2.0** | **0.250** | 100.0 |
| Ours (Oracle) | 66.0 | 0.863 | 100.0 |

## Key Findings

- **Exact Match**: 2.0% vs 0% baseline (∞× improvement)
- **Tanimoto**: 0.250 vs 0.17 (1.5× improvement)
- **Validity**: 100% (SELFIES encoding)

## Caveats

- Different datasets: MassSpecGym uses 231K spectra; we use GNPS (~2.7K)
- Different splits: MassSpecGym uses MCES-based split; we use random split
