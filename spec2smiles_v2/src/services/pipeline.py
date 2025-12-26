"""Integrated SPEC2SMILES pipeline service."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src.config import settings
from src.services.part_a import PartAService
from src.services.part_b import PartBService
from src.services.preprocessor import PreprocessorService
from src.utils.exceptions import ModelError


class PipelineService:
    """End-to-end SPEC2SMILES pipeline.

    Combines:
    - Spectrum preprocessing
    - Part A: Spectrum -> Descriptors (Hybrid CNN-Transformer)
    - Part B: Descriptors -> SMILES candidates (ConditionalVAE)
    - Candidate ranking via Tanimoto similarity
    """

    def __init__(self):
        """Initialize pipeline services."""
        self.preprocessor = PreprocessorService()
        self.part_a = PartAService()
        self.part_b = PartBService()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline models are loaded."""
        return self._loaded

    def load(
        self,
        part_a_dir: Path,
        part_b_dir: Path,
        verbose: bool = True,
    ) -> "PipelineService":
        """Load trained models from directories.

        Args:
            part_a_dir: Directory containing Part A model
            part_b_dir: Directory containing Part B model
            verbose: Whether to show loading progress

        Returns:
            Self for method chaining
        """
        part_a_dir = Path(part_a_dir)
        part_b_dir = Path(part_b_dir)

        if verbose:
            print("Loading Part A (Spectrum -> Descriptors)...")
        self.part_a.load(part_a_dir)
        if verbose:
            print(f"  Loaded {len(settings.descriptor_names)} descriptor models")

        if verbose:
            print("Loading Part B (Descriptors -> SMILES)...")
        self.part_b.load(part_b_dir)
        if verbose:
            print(f"  Vocabulary size: {self.part_b.encoder.vocab_size}")
            print(f"  Device: {self.part_b.device}")

        self._loaded = True
        return self

    def predict(
        self,
        spectrum: np.ndarray,
        n_candidates: int = 50,
        temperature: float = 0.7,
        return_descriptors: bool = False,
    ) -> Dict[str, Any]:
        """Predict molecular structure from mass spectrum.

        Args:
            spectrum: Preprocessed spectrum array of shape (n_bins,)
            n_candidates: Number of candidate structures to generate
            temperature: Sampling temperature for VAE
            return_descriptors: Whether to include predicted descriptors

        Returns:
            Dictionary with candidates and optionally descriptors
        """
        if not self._loaded:
            raise ModelError("Pipeline not loaded. Call load() first.")

        # Ensure spectrum is 2D
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)

        # Part A: Spectrum -> Descriptors
        descriptors = self.part_a.predict(spectrum)

        # Scale descriptors for Part B
        descriptors_scaled = self.part_a.predict_scaled(spectrum)

        # Part B: Descriptors -> SMILES candidates
        candidates_list = self.part_b.generate(
            descriptors_scaled,
            n_candidates=n_candidates,
            temperature=temperature,
        )

        candidates = candidates_list[0] if candidates_list else []

        result = {
            "candidates": candidates,
            "n_unique": len(candidates),
            "n_valid": len(candidates),
        }

        if return_descriptors:
            result["descriptors"] = descriptors[0].tolist()
            result["descriptor_names"] = list(settings.descriptor_names)

        return result

    def predict_from_peaks(
        self,
        peaks: List[Tuple[float, float]],
        n_candidates: int = 50,
        temperature: float = 0.7,
        return_descriptors: bool = False,
    ) -> Dict[str, Any]:
        """Predict molecular structure from raw peak list.

        Args:
            peaks: List of (m/z, intensity) tuples
            n_candidates: Number of candidate structures
            temperature: Sampling temperature
            return_descriptors: Whether to include descriptors

        Returns:
            Dictionary with candidates
        """
        spectrum = self.preprocessor.process_peaks(peaks)
        return self.predict(
            spectrum,
            n_candidates=n_candidates,
            temperature=temperature,
            return_descriptors=return_descriptors,
        )

    def predict_and_rank(
        self,
        spectrum: np.ndarray,
        true_smiles: Optional[str] = None,
        n_candidates: int = 50,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Predict and optionally rank candidates against true structure.

        Args:
            spectrum: Preprocessed spectrum array
            true_smiles: Optional ground truth SMILES for ranking
            n_candidates: Number of candidates
            temperature: Sampling temperature

        Returns:
            Dictionary with candidates, similarities, and rankings
        """
        result = self.predict(
            spectrum,
            n_candidates=n_candidates,
            temperature=temperature,
            return_descriptors=True,
        )

        if true_smiles is not None:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is not None:
                true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)
                true_canonical = Chem.MolToSmiles(true_mol, canonical=True)

                similarities = []
                for cand_smiles in result["candidates"]:
                    cand_mol = Chem.MolFromSmiles(cand_smiles)
                    if cand_mol is not None:
                        cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                        sim = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
                        similarities.append(sim)
                    else:
                        similarities.append(0.0)

                # Sort by similarity
                ranked_indices = np.argsort(similarities)[::-1]
                ranked_candidates = [
                    result["candidates"][i] for i in ranked_indices
                ]
                ranked_similarities = [similarities[i] for i in ranked_indices]

                # Check for exact match
                exact_match = any(
                    cand == true_canonical for cand in result["candidates"]
                )

                result["true_smiles"] = true_canonical
                result["ranked_candidates"] = ranked_candidates
                result["similarities"] = ranked_similarities
                result["exact_match"] = exact_match
                result["best_similarity"] = (
                    max(similarities) if similarities else 0.0
                )

        return result

    def batch_predict(
        self,
        spectra: np.ndarray,
        n_candidates: int = 50,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batch prediction for multiple spectra.

        Args:
            spectra: Spectra array of shape (n_samples, n_bins)
            n_candidates: Number of candidates per sample
            temperature: Sampling temperature
            verbose: Whether to show progress

        Returns:
            List of prediction dictionaries
        """
        from tqdm import tqdm

        results = []
        iterator = (
            tqdm(range(len(spectra)), desc="Predicting") if verbose else range(len(spectra))
        )

        for i in iterator:
            result = self.predict(
                spectra[i],
                n_candidates=n_candidates,
                temperature=temperature,
            )
            results.append(result)

        return results

    @classmethod
    def from_directories(
        cls,
        part_a_dir: Path,
        part_b_dir: Path,
        verbose: bool = True,
    ) -> "PipelineService":
        """Create and load pipeline from model directories.

        Args:
            part_a_dir: Directory with Part A model
            part_b_dir: Directory with Part B model
            verbose: Whether to show progress

        Returns:
            Loaded PipelineService
        """
        pipeline = cls()
        pipeline.load(part_a_dir, part_b_dir, verbose=verbose)
        return pipeline
