import json
from rdkit import Chem

# Load predictions
preds = []
with open('data/output/metrics/e2e_predictions_lgbm.jsonl') as f:
    for line in f:
        preds.append(json.loads(line))

print(f'Total samples: {len(preds)}')

# Calculate Hit@K
def hit_at_k(preds, k):
    hits = 0
    total = 0
    for p in preds:
        true_smiles = p.get('true_canonical') or p.get('true_smiles')
        candidates = p.get('all_candidates', [])[:k]
        
        try:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                continue
            true_can = Chem.MolToSmiles(true_mol, canonical=True)
            
            for cand in candidates:
                cand_mol = Chem.MolFromSmiles(cand)
                if cand_mol:
                    cand_can = Chem.MolToSmiles(cand_mol, canonical=True)
                    if cand_can == true_can:
                        hits += 1
                        break
            total += 1
        except:
            continue
    return hits / total if total > 0 else 0

h1 = hit_at_k(preds, 1)
h5 = hit_at_k(preds, 5)
h10 = hit_at_k(preds, 10)
h50 = hit_at_k(preds, 50)

print(f'Hit@1:  {h1*100:.1f}%')
print(f'Hit@5:  {h5*100:.1f}%')
print(f'Hit@10: {h10*100:.1f}%')
print(f'Hit@50: {h50*100:.1f}%')
