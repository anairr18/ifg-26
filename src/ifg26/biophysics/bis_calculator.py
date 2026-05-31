"""
bis_calculator.py
=================
Genuine Biophysical Bifacial Interface Score (BIS) Calculator.
Quantifies the complementary interface area of molecular glues spanning 
the E3 ligase and the target substrate using 3D structural coordinates.
"""

import numpy as np

class BifacialInterfaceScoreCalculator:
    def __init__(self, contact_cutoff: float = 4.5, weight_hydrophobic: float = 1.5, weight_polar: float = 1.2):
        self.contact_cutoff = contact_cutoff
        self.weight_hydrophobic = weight_hydrophobic
        self.weight_polar = weight_polar
        
    def _is_hydrophobic(self, element: str) -> bool:
        return element.upper() in ["C", "S", "SE", "I", "BR", "CL", "F"]
        
    def _is_polar(self, element: str) -> bool:
        return element.upper() in ["O", "N", "P", "F"]
        
    def compute_bis(self, e3_coords: np.ndarray, e3_elements: list[str],
                    tgt_coords: np.ndarray, tgt_elements: list[str],
                    lig_coords: np.ndarray, lig_elements: list[str]) -> dict:
        """
        Calculates the Bifacial Interface Score (BIS) for a ternary complex.
        
        Parameters:
            e3_coords: (N, 3) array of E3 ligase atomic coordinates.
            e3_elements: List of E3 atomic elements.
            tgt_coords: (M, 3) array of Target substrate atomic coordinates.
            tgt_elements: List of Target atomic elements.
            lig_coords: (L, 3) array of Ligand glue atomic coordinates.
            lig_elements: List of Ligand atomic elements.
            
        Returns:
            dict containing:
                bis_score: Combined Bifacial Interface Score.
                face_a_contacts: Number of contacts on Face A (E3 interface).
                face_b_contacts: Number of contacts on Face B (Target interface).
                bifacial_coefficient: Balance ratio between Face A and Face B contacts (max = 1.0).
                steric_clashes: Estimated number of steric overlaps.
        """
        if len(lig_coords) == 0:
            return {"bis_score": 0.0, "face_a_contacts": 0, "face_b_contacts": 0, "bifacial_coefficient": 0.0, "steric_clashes": 0}
            
        face_a_score = 0.0
        face_b_score = 0.0
        face_a_contacts = 0
        face_b_contacts = 0
        clashes = 0
        
        # ── Face A: E3 Ligase Interface ─────────────────────────────────
        if len(e3_coords) > 0:
            # Pairwise distances (L, N)
            dists_a = np.linalg.norm(lig_coords[:, None, :] - e3_coords[None, :, :], axis=2)
            for i in range(len(lig_coords)):
                lig_elem = lig_elements[i]
                close_idx = np.where(dists_a[i] < self.contact_cutoff)[0]
                if len(close_idx) > 0:
                    face_a_contacts += 1
                    # Score complementarity
                    for idx in close_idx:
                        d = dists_a[i, idx]
                        if d < 1.8: # Steric clash
                            clashes += 1
                            continue
                        e3_elem = e3_elements[idx]
                        if self._is_hydrophobic(lig_elem) and self._is_hydrophobic(e3_elem):
                            face_a_score += self.weight_hydrophobic * (self.contact_cutoff - d)
                        elif self._is_polar(lig_elem) and self._is_polar(e3_elem):
                            face_a_score += self.weight_polar * (self.contact_cutoff - d)
                            
        # ── Face B: Target Substrate Interface ──────────────────────────
        if len(tgt_coords) > 0:
            # Pairwise distances (L, M)
            dists_b = np.linalg.norm(lig_coords[:, None, :] - tgt_coords[None, :, :], axis=2)
            for i in range(len(lig_coords)):
                lig_elem = lig_elements[i]
                close_idx = np.where(dists_b[i] < self.contact_cutoff)[0]
                if len(close_idx) > 0:
                    face_b_contacts += 1
                    # Score complementarity
                    for idx in close_idx:
                        d = dists_b[i, idx]
                        if d < 1.8: # Steric clash
                            clashes += 1
                            continue
                        tgt_elem = tgt_elements[idx]
                        if self._is_hydrophobic(lig_elem) and self._is_hydrophobic(tgt_elem):
                            face_b_score += self.weight_hydrophobic * (self.contact_cutoff - d)
                        elif self._is_polar(lig_elem) and self._is_polar(tgt_elem):
                            face_b_score += self.weight_polar * (self.contact_cutoff - d)
                            
        # ── Bifacial Balance (Harmonic Mean coefficient) ─────────────────
        if face_a_contacts > 0 and face_b_contacts > 0:
            bifacial_coef = 2.0 * (face_a_contacts * face_b_contacts) / (face_a_contacts + face_b_contacts)
            # Normalize coefficient relative to ligand size
            bifacial_coef = min(1.0, bifacial_coef / len(lig_coords))
        else:
            bifacial_coef = 0.0
            
        clash_penalty = clashes * 5.0
        bis_score = (face_a_score + face_b_score) * (1.0 + bifacial_coef) - clash_penalty
        
        return {
            "bis_score": float(np.round(max(0.0, bis_score), 4)),
            "face_a_contacts": int(face_a_contacts),
            "face_b_contacts": int(face_b_contacts),
            "bifacial_coefficient": float(np.round(bifacial_coef, 4)),
            "steric_clashes": int(clashes)
        }

if __name__ == "__main__":
    print("Testing BIS Calculator...")
    # Mock coordinates
    e3_coords = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    e3_elems = ["C", "O"]
    
    tgt_coords = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
    tgt_elems = ["C", "N"]
    
    # Bifacial glue touching both
    lig_coords = np.array([[2.5, 1.2, 1.0], [8.5, 9.8, 10.0]])
    lig_elems = ["C", "O"]
    
    calc = BifacialInterfaceScoreCalculator()
    res = calc.compute_bis(e3_coords, e3_elems, tgt_coords, tgt_elems, lig_coords, lig_elems)
    print("BIS Results:", res)
    assert res["face_a_contacts"] == 1 and res["face_b_contacts"] == 1
    print("Passed!")
