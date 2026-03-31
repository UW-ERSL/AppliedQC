"""
ML-Based Pauli Decomposition Learning for 1D FEM Poisson Equation

This module demonstrates:
1. Generate 1D FEM stiffness matrices for different mesh sizes
2. Compute exact Pauli decompositions for training
3. Train ML model to predict Pauli structure
4. Predict decomposition for larger systems
5. Refine coefficients via optimization

Author: Research Experiment
Date: 2025
"""

import numpy as np
from itertools import product, combinations
from typing import Dict, List, Tuple, Optional
import pickle
from dataclasses import dataclass
from collections import defaultdict

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_sample_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML features disabled.")
    SKLEARN_AVAILABLE = False

# Optimization imports
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    print("Warning: cvxpy not available. Coefficient optimization disabled.")
    CVXPY_AVAILABLE = False


@dataclass
class TrainingDataPoint:
    """Single training example"""
    m: int
    pauli_string: str
    exists: bool
    coefficient: float
    features: Dict[str, float]


class FEM1DPoisson:
    """
    1D Finite Element Method for Poisson Equation with Dirichlet BC
    
    Problem: -u''(x) = f(x) on [0, 1] with u(0) = u(1) = 0
    
    Uses linear finite elements (piecewise linear basis functions)
    """
    
    def __init__(self, m: int):
        """
        Args:
            m: Number of qubits (matrix size will be 2^m × 2^m)
        """
        self.m = m
        self.n = 2**m  # Matrix dimension for quantum encoding
        
        # Pauli matrices
        self.I = np.eye(2)
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        
        self.sigma_map = {
            "I": [(1.0, "I")],
            "sigma_plus": [(0.5, "X"), (0.5j, "Y")],
            "sigma_minus": [(0.5, "X"), (-0.5j, "Y")]
        }
    
    def assemble_stiffness_matrix(self) -> np.ndarray:
        """
        Assemble 1D FEM stiffness matrix with Dirichlet BC
        
        To get dimension 2^m, we use n_elements = 2^m + 1
        This gives n_interior = 2^m nodes after applying Dirichlet BC
        
        Returns matrix of size (2^m, 2^m) - same as Liu et al. finite difference
        """
        n = self.n  # 2^m
        n_elements = n + 1
        h = 1.0 / n_elements  # Element length
        
        # Tridiagonal matrix
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2.0 / h
            if i > 0:
                K[i, i-1] = -1.0 / h
            if i < n - 1:
                K[i, i+1] = -1.0 / h
        
        return K
    
    def get_normalized_matrix(self) -> np.ndarray:
        """
        Get normalized stiffness matrix (multiply by h)
        This gives the standard tridiagonal form: diag(2, -1, -1)
        """
        K = self.assemble_stiffness_matrix()
        n = self.n
        h = 1.0 / (n + 1)
        return K * h
    
    def decompose_to_sigma_basis(self) -> List[Tuple[float, List[str]]]:
        """
        Decompose into {I, σ+, σ-} basis (Liu et al. approach)
        Returns 2m + 1 terms
        """
        terms = []
        
        # Identity term: 2 * I^m
        terms.append((2.0, ["I"] * self.m))
        
        # Recursive terms
        for j in range(1, self.m + 1):
            # I^(m-j) ⊗ σ+ ⊗ σ-^(j-1)
            op1 = ["I"] * (self.m - j) + ["sigma_plus"] + ["sigma_minus"] * (j - 1)
            terms.append((-1.0, op1))
            
            # I^(m-j) ⊗ σ- ⊗ σ+^(j-1)
            op2 = ["I"] * (self.m - j) + ["sigma_minus"] + ["sigma_plus"] * (j - 1)
            terms.append((-1.0, op2))
        
        return terms
    
    def decompose_to_pauli(self, epsilon: float = 1e-12) -> Dict[str, float]:
        """
        Convert {I, σ+, σ-} decomposition to Pauli {I, X, Y, Z} basis
        """
        sigma_terms = self.decompose_to_sigma_basis()
        pauli_lcu = defaultdict(complex)
        
        for coeff, op_list in sigma_terms:
            choices = [self.sigma_map[op] for op in op_list]
            
            for combination in product(*choices):
                combined_coeff = coeff
                pauli_string = ""
                
                for term_coeff, pauli_char in combination:
                    combined_coeff *= term_coeff
                    pauli_string += pauli_char
                
                pauli_lcu[pauli_string] += combined_coeff
        
        # Filter and convert to real
        result = {}
        for k, v in pauli_lcu.items():
            if abs(v) > epsilon:
                # Should be real for Hermitian matrix
                if abs(v.imag) > epsilon:
                    print(f"Warning: Imaginary coefficient {v} for {k}")
                result[k] = v.real
        
        return result
    
    def reconstruct_matrix(self, pauli_dict: Dict[str, float]) -> np.ndarray:
        """Reconstruct matrix from Pauli decomposition"""
        n = self.n  # 2^m
        result = np.zeros((n, n), dtype=complex)
        
        gate_map = {"I": self.I, "X": self.X, "Y": self.Y, "Z": self.Z}
        
        for pauli_string, coeff in pauli_dict.items():
            term = coeff
            for char in pauli_string:
                term = np.kron(term, gate_map[char])
            result += term
        
        return result.real
    
    def verify_decomposition(self, pauli_dict: Dict[str, float]) -> Tuple[bool, float]:
        """Verify that Pauli decomposition is correct"""
        A_original = self.get_normalized_matrix()
        A_reconstructed = self.reconstruct_matrix(pauli_dict)
        
        # Check dimensions match
        if A_original.shape != A_reconstructed.shape:
            print(f"ERROR: Dimension mismatch! "
                  f"Original: {A_original.shape}, Reconstructed: {A_reconstructed.shape}")
            return False, float('inf')
        
        error = np.linalg.norm(A_original - A_reconstructed, 'fro')
        relative_error = error / np.linalg.norm(A_original, 'fro')
        
        is_correct = relative_error < 1e-10
        
        return is_correct, relative_error


class FeatureExtractor:
    """Extract features from Pauli strings for ML"""
    
    @staticmethod
    def extract_features(pauli_string: str, m: int) -> Dict[str, float]:
        """
        Extract comprehensive features from a Pauli string
        Enhanced with better pattern recognition
        """
        features = {}
        
        # ===== Basic Statistics =====
        features['m'] = float(m)
        features['n_I'] = pauli_string.count('I')
        features['n_X'] = pauli_string.count('X')
        features['n_Y'] = pauli_string.count('Y')
        features['n_Z'] = pauli_string.count('Z')
        features['support'] = features['n_X'] + features['n_Y'] + features['n_Z']
        
        # ===== Critical: Type indicators =====
        features['has_X'] = float('X' in pauli_string)
        features['has_Y'] = float('Y' in pauli_string)
        features['has_Z'] = float('Z' in pauli_string)
        features['has_XY'] = float('X' in pauli_string and 'Y' in pauli_string)
        features['is_pure_IZ'] = float(set(pauli_string) <= {'I', 'Z'})
        features['is_pure_IX'] = float(set(pauli_string) <= {'I', 'X'})
        features['is_pure_IY'] = float(set(pauli_string) <= {'I', 'Y'})
        
        # ===== Locality Features =====
        non_I_positions = [i for i, p in enumerate(pauli_string) if p != 'I']
        
        if non_I_positions:
            features['first_nonI'] = float(non_I_positions[0]) / m
            features['last_nonI'] = float(non_I_positions[-1]) / m
            features['spread'] = float(non_I_positions[-1] - non_I_positions[0]) / m
            features['density'] = len(non_I_positions) / (features['spread'] * m + 1) if features['spread'] > 0 else 1.0
        else:
            features['first_nonI'] = 0.0
            features['last_nonI'] = 0.0
            features['spread'] = 0.0
            features['density'] = 0.0
        
        # ===== Pattern Recognition =====
        features['is_identity'] = float(pauli_string == 'I' * m)
        features['is_palindrome'] = float(pauli_string == pauli_string[::-1])
        
        # Consecutive runs (normalized)
        features['max_Z_run'] = float(FeatureExtractor._max_run(pauli_string, 'Z')) / m
        features['max_I_run'] = float(FeatureExtractor._max_run(pauli_string, 'I')) / m
        features['max_X_run'] = float(FeatureExtractor._max_run(pauli_string, 'X')) / m
        features['max_Y_run'] = float(FeatureExtractor._max_run(pauli_string, 'Y')) / m
        
        # ===== Pair Patterns (Critical for structure) =====
        pair_counts = defaultdict(int)
        for i in range(len(pauli_string) - 1):
            pair = pauli_string[i:i+2]
            pair_counts[pair] += 1
        
        # Normalize by (m-1)
        norm = m - 1 if m > 1 else 1
        features['n_ZZ'] = float(pair_counts['ZZ']) / norm
        features['n_ZI'] = float(pair_counts['ZI']) / norm
        features['n_IZ'] = float(pair_counts['IZ']) / norm
        features['n_XX'] = float(pair_counts['XX']) / norm
        features['n_II'] = float(pair_counts['II']) / norm
        
        # ===== Physics-Motivated Features =====
        features['is_single_site'] = float(features['support'] == 1)
        features['is_two_site'] = float(features['support'] == 2)
        features['is_nearest_neighbor'] = float(
            features['support'] == 2 and features['spread'] * m == 1
        )
        
        # ===== Normalized Features =====
        features['support_norm'] = features['support'] / m
        features['XY_fraction'] = (features['n_X'] + features['n_Y']) / m
        features['Z_fraction'] = features['n_Z'] / m
        
        # ===== Interaction patterns ===== 
        features['alternating_IZ'] = FeatureExtractor._check_alternating(pauli_string, 'I', 'Z')
        features['symmetric_structure'] = FeatureExtractor._check_symmetric_positions(pauli_string)
        
        return features
    
    @staticmethod
    def _max_run(s: str, char: str) -> int:
        """Maximum consecutive run of character"""
        max_run = 0
        current_run = 0
        for c in s:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    @staticmethod
    def _check_alternating(s: str, c1: str, c2: str) -> float:
        """Check if string alternates between c1 and c2"""
        s_filtered = ''.join([c for c in s if c in [c1, c2]])
        if len(s_filtered) < 2:
            return 0.0
        
        alternating = all(
            s_filtered[i] != s_filtered[i+1] 
            for i in range(len(s_filtered)-1)
        )
        return float(alternating)
    
    @staticmethod
    def _check_symmetric_positions(s: str) -> float:
        """Check if non-I positions are symmetric"""
        non_I = [i for i, p in enumerate(s) if p != 'I']
        if len(non_I) <= 1:
            return 1.0
        
        m = len(s)
        symmetric = all(
            (m - 1 - pos) in non_I 
            for pos in non_I
        )
        return float(symmetric)


class NegativeSampler:
    """Generate negative samples for binary classification"""
    
    def __init__(self, positive_samples: List[str], m: int):
        self.positive_samples = set(positive_samples)
        self.m = m
        self._analyze_positives()
    
    def _analyze_positives(self):
        """Analyze statistics of positive examples"""
        if not self.positive_samples:
            self.avg_support = 1.0
            self.std_support = 0.5
            return
            
        supports = [sum(1 for p in s if p != 'I') for s in self.positive_samples]
        self.avg_support = np.mean(supports) if supports else 1.0
        self.std_support = np.std(supports) if supports else 0.5
        
        # Analyze which Pauli types appear
        self.pauli_distribution = defaultdict(int)
        for s in self.positive_samples:
            for p in s:
                self.pauli_distribution[p] += 1
    
    def sample(self, n_samples: int, strategy: str = 'mixed') -> List[str]:
        """
        Generate negative samples
        
        For small m (≤3), enumerate all possible strings
        For large m (>3), use random sampling from ALL 4 Pauli types
        """
        # For very small m, enumerate (but use all 4 Paulis!)
        if self.m <= 3:
            return self._enumerate_negatives_small()[:n_samples]
        else:
            return self._sample_negatives(n_samples, strategy)
    
    def _enumerate_negatives_small(self) -> List[str]:
        """
        Enumerate negatives for small m
        Use all 4 Pauli types to ensure negatives exist!
        """
        # For m<=3, we can enumerate a subset
        # Use only {I, X, Z} to reduce space but still get negatives
        all_strings = [''.join(p) for p in product(['I', 'X', 'Z'], repeat=self.m)]
        negatives = [s for s in all_strings if s not in self.positive_samples]
        
        # If still no negatives, try all 4
        if len(negatives) < 3:
            all_strings = [''.join(p) for p in product(['I', 'X', 'Y', 'Z'], repeat=self.m)]
            negatives = [s for s in all_strings if s not in self.positive_samples]
        
        return negatives
    
    def _sample_negatives(self, n_samples: int, strategy: str) -> List[str]:
        """Sample negative strings using ALL 4 Pauli types"""
        negatives = set()
        max_attempts = n_samples * 100
        attempts = 0
        
        # Use all 4 Pauli types for sampling!
        pauli_alphabet = ['I', 'X', 'Y', 'Z']
        
        while len(negatives) < n_samples and attempts < max_attempts:
            if strategy == 'random' or (strategy == 'mixed' and len(negatives) < n_samples // 2):
                # Pure random
                s = ''.join(np.random.choice(pauli_alphabet, self.m))
            else:
                # Similar support to positives
                support = int(np.clip(
                    np.random.normal(self.avg_support, self.std_support),
                    0, self.m
                ))
                s = self._random_pauli_with_support(support, pauli_alphabet)
            
            if s not in self.positive_samples and s not in negatives:
                negatives.add(s)
            
            attempts += 1
        
        if len(negatives) < n_samples:
            print(f"    Warning: Only found {len(negatives)}/{n_samples} unique negatives")
        
        return list(negatives)
    
    def _random_pauli(self) -> str:
        """Generate random Pauli string using ALL 4 types"""
        return ''.join(np.random.choice(['I', 'X', 'Y', 'Z'], self.m))
    
    def _random_pauli_with_support(self, support: int, alphabet: List[str] = None) -> str:
        """Generate random Pauli string with specific support"""
        if alphabet is None:
            alphabet = ['I', 'X', 'Y', 'Z']
        
        support = max(0, min(support, self.m))
        if support == 0:
            return 'I' * self.m
        
        positions = np.random.choice(self.m, support, replace=False)
        pauli = ['I'] * self.m
        
        # Choose non-I Paulis from alphabet
        non_I = [p for p in alphabet if p != 'I']
        if not non_I:
            non_I = ['X', 'Y', 'Z']
        
        for pos in positions:
            pauli[pos] = np.random.choice(non_I)
        
        return ''.join(pauli)


class PauliDecompositionLearner:
    """
    ML-based Pauli decomposition predictor
    
    Two-stage approach:
    1. Classifier: Predict which Pauli strings exist
    2. Regressor: Predict coefficient values
    """
    
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML functionality")
        
        # Use GradientBoosting for better expressiveness
        self.pattern_classifier = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            verbose=0
        )
        
        self.coeff_regressor = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        self.scaler_pattern = StandardScaler()
        self.scaler_coeff = StandardScaler()
        self.feature_names = None
    
    def generate_training_data(
        self, 
        m_values: List[int],
        negative_ratio: float = 3.0,
        epsilon: float = 1e-12,
        min_negative_ratio: float = 1.0
    ) -> List[TrainingDataPoint]:
        """
        Generate training data from multiple matrix sizes
        
        Key insight: For Poisson, almost all I/Z strings are positive.
        We MUST sample from all 4 Pauli types to get negatives!
        """
        print("=" * 70)
        print("GENERATING TRAINING DATA")
        print("=" * 70)
        
        all_data = []
        
        for m in m_values:
            print(f"\nProcessing m={m} (N={2**m})...")
            
            # Generate exact decomposition
            fem = FEM1DPoisson(m)
            pauli_dict = fem.decompose_to_pauli(epsilon=epsilon)
            
            # Verify correctness
            is_correct, error = fem.verify_decomposition(pauli_dict)
            print(f"  Verification: {'✓ PASS' if is_correct else '✗ FAIL'} "
                  f"(error={error:.2e})")
            
            n_positive = len(pauli_dict)
            print(f"  Found {n_positive} non-zero Pauli terms")
            
            # Check what fraction of I/Z space is covered
            total_IZ_strings = 2**m
            if n_positive == total_IZ_strings:
                print(f"  ⚠ WARNING: All {total_IZ_strings} I/Z strings are positive!")
                print(f"    Must sample from {{I,X,Y,Z}} space to get negatives")
            
            # Positive samples
            positive_strings = list(pauli_dict.keys())
            for pauli_str in positive_strings:
                features = FeatureExtractor.extract_features(pauli_str, m)
                all_data.append(TrainingDataPoint(
                    m=m,
                    pauli_string=pauli_str,
                    exists=True,
                    coefficient=pauli_dict[pauli_str],
                    features=features
                ))
            
            # Negative samples - use ALL 4 Pauli types!
            n_negatives_requested = max(
                int(n_positive * negative_ratio),
                n_positive  # At least as many negatives as positives
            )
            
            sampler = NegativeSampler(positive_strings, m)
            negative_strings = sampler.sample(n_negatives_requested, strategy='mixed')
            
            actual_ratio = len(negative_strings) / n_positive if n_positive > 0 else 0
            print(f"  Generated {len(negative_strings)} negative samples (ratio: {actual_ratio:.2f})")
            
            if actual_ratio < min_negative_ratio:
                print(f"  ⚠ Warning: Negative ratio {actual_ratio:.2f} < {min_negative_ratio:.2f}")
                print(f"    Model may have severe class imbalance!")
            
            for pauli_str in negative_strings:
                features = FeatureExtractor.extract_features(pauli_str, m)
                all_data.append(TrainingDataPoint(
                    m=m,
                    pauli_string=pauli_str,
                    exists=False,
                    coefficient=0.0,
                    features=features
                ))
        
        n_pos = sum(1 for d in all_data if d.exists)
        n_neg = sum(1 for d in all_data if not d.exists)
        
        print(f"\n{'='*70}")
        print(f"Total training samples: {len(all_data)}")
        print(f"  Positive: {n_pos} ({100*n_pos/len(all_data):.1f}%)")
        print(f"  Negative: {n_neg} ({100*n_neg/len(all_data):.1f}%)")
        
        if n_pos / len(all_data) > 0.7:
            print(f"  ⚠ WARNING: Severe class imbalance (>70% positive)")
            print(f"    Consider increasing negative_ratio or using class weights")
        
        print(f"{'='*70}\n")
        
        return all_data
    
    def train(self, training_data: List[TrainingDataPoint]):
        """
        Train both classifier and regressor with class balancing
        """
        print("=" * 70)
        print("TRAINING ML MODELS")
        print("=" * 70)
        
        # Prepare data
        X_pattern, y_exists = [], []
        X_coeff, y_coeff = [], []
        
        for data_point in training_data:
            feature_vec = [data_point.features[k] for k in sorted(data_point.features.keys())]
            X_pattern.append(feature_vec)
            y_exists.append(int(data_point.exists))
            
            if data_point.exists:
                X_coeff.append(feature_vec)
                y_coeff.append(data_point.coefficient)
        
        if self.feature_names is None:
            self.feature_names = sorted(training_data[0].features.keys())
        
        X_pattern = np.array(X_pattern)
        X_coeff = np.array(X_coeff)
        y_exists = np.array(y_exists)
        y_coeff = np.array(y_coeff)
        
        print(f"\nPattern Classification:")
        print(f"  Samples: {len(X_pattern)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Positive class ratio: {y_exists.mean():.3f}")
        
        # Compute class weights to balance
        sample_weights = compute_sample_weight('balanced', y_exists)
        print(f"  Using balanced class weights")
        
        # Train classifier with class weights
        X_pattern_scaled = self.scaler_pattern.fit_transform(X_pattern)
        self.pattern_classifier.fit(X_pattern_scaled, y_exists, sample_weight=sample_weights)
        
        train_acc = self.pattern_classifier.score(X_pattern_scaled, y_exists)
        print(f"  Training accuracy: {train_acc:.3f}")
        
        # Compute precision/recall on training set
        y_pred = self.pattern_classifier.predict(X_pattern_scaled)
        print(f"  Training precision: {precision_score(y_exists, y_pred):.3f}")
        print(f"  Training recall: {recall_score(y_exists, y_pred):.3f}")
        print(f"  Training F1: {f1_score(y_exists, y_pred):.3f}")
        
        # Train regressor
        print(f"\nCoefficient Regression:")
        print(f"  Samples: {len(X_coeff)}")
        
        X_coeff_scaled = self.scaler_coeff.fit_transform(X_coeff)
        self.coeff_regressor.fit(X_coeff_scaled, y_coeff)
        
        train_r2 = self.coeff_regressor.score(X_coeff_scaled, y_coeff)
        print(f"  Training R²: {train_r2:.3f}")
        
        # Feature importance
        print(f"\nTop 15 Important Features (Classification):")
        importances = self.pattern_classifier.feature_importances_
        indices = np.argsort(importances)[-15:][::-1]
        for idx in indices:
            print(f"  {self.feature_names[idx]:25s}: {importances[idx]:.4f}")
        
        print(f"\n{'='*70}\n")
    
    def predict(
        self, 
        m: int, 
        threshold: float = 0.3,  # LOWERED threshold
        max_candidates: int = 20000  # INCREASED candidates
    ) -> Dict[str, float]:
        """
        Predict Pauli decomposition for new matrix size
        
        Args:
            m: Number of qubits
            threshold: Probability threshold for existence (lowered to 0.3)
            max_candidates: Maximum candidate strings to evaluate
        """
        print(f"Predicting Pauli decomposition for m={m}...")
        
        # Generate candidate Pauli strings
        candidates = self._generate_candidates(m, max_candidates)
        print(f"  Evaluating {len(candidates)} candidates...")
        
        # Extract features
        X = []
        for pauli_str in candidates:
            features = FeatureExtractor.extract_features(pauli_str, m)
            feature_vec = [features[k] for k in self.feature_names]
            X.append(feature_vec)
        
        X = np.array(X)
        X_scaled = self.scaler_pattern.transform(X)
        
        # Predict existence probabilities
        probs = self.pattern_classifier.predict_proba(X_scaled)[:, 1]
        
        # Filter by threshold
        selected_indices = np.where(probs >= threshold)[0]
        print(f"  Selected {len(selected_indices)} strings (threshold={threshold})")
        
        # Predict coefficients for selected strings
        predicted_pauli = {}
        
        if len(selected_indices) > 0:
            X_selected = X[selected_indices]
            X_selected_scaled = self.scaler_coeff.transform(X_selected)
            coeffs = self.coeff_regressor.predict(X_selected_scaled)
            
            for idx, coeff in zip(selected_indices, coeffs):
                pauli_str = candidates[idx]
                predicted_pauli[pauli_str] = coeff
        
        return predicted_pauli
    
    def _generate_candidates(self, m: int, max_candidates: int) -> List[str]:
        """
        Generate plausible candidate Pauli strings
        
        CRITICAL: Must include candidates from all 4 Pauli types!
        The Poisson decomposition includes X and Y terms!
        """
        candidates = set()
        
        # Strategy 1: Enumerate by support for small m
        if m <= 4:
            max_support = m
        else:
            max_support = min(m, 6)
        
        # For each support level
        for support in range(max_support + 1):
            # All positions to place non-I Paulis
            for positions in combinations(range(m), support):
                if support == 0:
                    candidates.add('I' * m)
                    continue
                
                # Try different Pauli assignments
                # Limit combinatorial explosion
                if support <= 2:
                    # For low support, try all combinations
                    for paulis in product(['X', 'Y', 'Z'], repeat=support):
                        pauli_str = ['I'] * m
                        for pos, p in zip(positions, paulis):
                            pauli_str[pos] = p
                        candidates.add(''.join(pauli_str))
                elif support <= 4:
                    # For medium support, sample some combinations
                    n_samples = min(20, 3**support)
                    for _ in range(n_samples):
                        paulis = np.random.choice(['X', 'Y', 'Z'], support)
                        pauli_str = ['I'] * m
                        for pos, p in zip(positions, paulis):
                            pauli_str[pos] = p
                        candidates.add(''.join(pauli_str))
                else:
                    # For high support, just a few samples
                    for _ in range(5):
                        paulis = np.random.choice(['X', 'Y', 'Z'], support)
                        pauli_str = ['I'] * m
                        for pos, p in zip(positions, paulis):
                            pauli_str[pos] = p
                        candidates.add(''.join(pauli_str))
        
        candidates = list(candidates)
        print(f"  Generated {len(candidates)} initial candidates")
        
        # If still not enough, add random samples
        while len(candidates) < min(max_candidates, 4**(m//2)):
            support = np.random.randint(0, min(m, 5))
            positions = np.random.choice(m, support, replace=False) if support > 0 else []
            pauli_str = ['I'] * m
            for pos in positions:
                pauli_str[pos] = np.random.choice(['X', 'Y', 'Z'])
            candidate = ''.join(pauli_str)
            if candidate not in candidates:
                candidates.append(candidate)
        
        # Sample if too many
        if len(candidates) > max_candidates:
            candidates = list(np.random.choice(
                candidates, 
                max_candidates, 
                replace=False
            ))
        
        return candidates


class CoefficientOptimizer:
    """
    Refine predicted coefficients via optimization
    """
    
    def __init__(self):
        if not CVXPY_AVAILABLE:
            print("Warning: cvxpy not available, optimization disabled")
    
    def optimize(
        self,
        predicted_structure: Dict[str, float],
        m: int,
        method: str = 'lstsq'
    ) -> Dict[str, float]:
        """
        Optimize coefficients given structure
        
        Args:
            predicted_structure: Dict of {pauli_string: initial_coeff}
            m: Number of qubits
            method: 'lstsq' or 'cvxpy'
        """
        fem = FEM1DPoisson(m)
        A_target = fem.get_normalized_matrix()
        n = 2**m
        
        pauli_strings = list(predicted_structure.keys())
        n_terms = len(pauli_strings)
        
        print(f"\nOptimizing {n_terms} coefficients...")
        
        if method == 'lstsq':
            # Build linear system: A_target = Σ c_i P_i
            # Flatten: vec(A_target) = Σ c_i vec(P_i)
            
            # Build design matrix
            P_matrices = []
            for pauli_str in pauli_strings:
                P = self._build_pauli_matrix(pauli_str)
                P_matrices.append(P.flatten())
            
            # Stack as columns
            design_matrix = np.column_stack(P_matrices)  # [n², n_terms]
            target_vec = A_target.flatten()  # [n²]
            
            # Solve least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(
                design_matrix, 
                target_vec, 
                rcond=None
            )
            
            result = {pauli_strings[i]: coeffs[i] for i in range(n_terms)}
            
            # Compute error
            A_recon = sum(result[ps] * self._build_pauli_matrix(ps) 
                         for ps in pauli_strings)
            error = np.linalg.norm(A_target - A_recon, 'fro')
            print(f"  Reconstruction error: {error:.2e}")
            
            return result
        
        elif method == 'cvxpy' and CVXPY_AVAILABLE:
            # Use CVXPY for constrained optimization
            coeffs = cp.Variable(n_terms)
            
            # Build matrices
            P_matrices = [self._build_pauli_matrix(ps) for ps in pauli_strings]
            A_approx = sum(coeffs[i] * P_matrices[i] for i in range(n_terms))
            
            # Objective
            objective = cp.Minimize(cp.norm(A_target - A_approx, 'fro'))
            
            # Constraints (optional: bound coefficients)
            constraints = [cp.abs(coeffs) <= 100.0]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)
            
            result = {pauli_strings[i]: coeffs.value[i] for i in range(n_terms)}
            print(f"  Reconstruction error: {problem.value:.2e}")
            
            return result
        
        else:
            print("  Warning: No valid optimization method")
            return predicted_structure
    
    def _build_pauli_matrix(self, pauli_str: str) -> np.ndarray:
        """Build Pauli matrix from string"""
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        result = pauli_map[pauli_str[0]]
        for p in pauli_str[1:]:
            result = np.kron(result, pauli_map[p])
        
        return result.real


def evaluate_prediction(
    predicted: Dict[str, float],
    ground_truth: Dict[str, float],
    m: int
) -> Dict[str, float]:
    """
    Comprehensive evaluation metrics
    """
    pred_strings = set(predicted.keys())
    true_strings = set(ground_truth.keys())
    
    # Structure metrics
    tp = len(pred_strings & true_strings)
    fp = len(pred_strings - true_strings)
    fn = len(true_strings - pred_strings)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Coefficient metrics (for correct predictions)
    common = pred_strings & true_strings
    if common:
        coeff_errors = [abs(predicted[s] - ground_truth[s]) for s in common]
        mae_coeff = np.mean(coeff_errors)
        max_coeff_error = np.max(coeff_errors)
    else:
        mae_coeff = float('inf')
        max_coeff_error = float('inf')
    
    # Matrix reconstruction
    fem = FEM1DPoisson(m)
    A_true = fem.get_normalized_matrix()
    A_pred = fem.reconstruct_matrix(predicted)
    
    matrix_error = np.linalg.norm(A_true - A_pred, 'fro')
    relative_error = matrix_error / np.linalg.norm(A_true, 'fro')
    
    # Compression
    full_terms = 4**m
    predicted_terms = len(pred_strings)
    true_terms = len(true_strings)
    compression_ratio = full_terms / predicted_terms if predicted_terms > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'mae_coefficient': mae_coeff,
        'max_coefficient_error': max_coeff_error,
        'matrix_frobenius_error': matrix_error,
        'matrix_relative_error': relative_error,
        'n_predicted': predicted_terms,
        'n_true': true_terms,
        'compression_ratio': compression_ratio,
    }


def print_evaluation(metrics: Dict[str, float], title: str = "Evaluation"):
    """Pretty print evaluation metrics"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    print(f"\n{'Structure Prediction':^70}")
    print(f"  Precision:          {metrics['precision']:8.3f}")
    print(f"  Recall:             {metrics['recall']:8.3f}")
    print(f"  F1 Score:           {metrics['f1_score']:8.3f}")
    print(f"  True Positives:     {metrics['true_positive']:8d}")
    print(f"  False Positives:    {metrics['false_positive']:8d}")
    print(f"  False Negatives:    {metrics['false_negative']:8d}")
    
    print(f"\n{'Coefficient Accuracy':^70}")
    if metrics['mae_coefficient'] != float('inf'):
        print(f"  MAE:                {metrics['mae_coefficient']:8.4f}")
        print(f"  Max Error:          {metrics['max_coefficient_error']:8.4f}")
    else:
        print(f"  MAE:                     N/A (no correct predictions)")
        print(f"  Max Error:               N/A")
    
    print(f"\n{'Matrix Reconstruction':^70}")
    print(f"  Frobenius Error:    {metrics['matrix_frobenius_error']:8.2e}")
    print(f"  Relative Error:     {metrics['matrix_relative_error']:8.2e}")
    
    print(f"\n{'Compression':^70}")
    print(f"  Predicted Terms:    {metrics['n_predicted']:8d}")
    print(f"  True Terms:         {metrics['n_true']:8d}")
    print(f"  Compression Ratio:  {metrics['compression_ratio']:8.1f}x")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
class HybridPauliPredictor:
    """
    Hybrid approach: Use known structure + ML refinement
    
    Key insight: For Poisson, we KNOW:
    1. All I/Z strings appear (this is deterministic!)
    2. Some X/Y strings also appear
    
    Strategy:
    - Enumerate ALL I/Z strings (guaranteed positives)
    - Use ML to predict which X/Y strings to add
    """
    
    def __init__(self, learner: PauliDecompositionLearner):
        self.learner = learner
    
    def predict_hybrid(self, m: int, ml_threshold: float = 0.5) -> Dict[str, float]:
        """
        Hybrid prediction combining analytical + ML
        """
        print(f"\n{'='*70}")
        print(f" HYBRID PREDICTION for m={m} ".center(70))
        print(f"{'='*70}\n")
        
        predicted = {}
        
        # PHASE 1: Add ALL I/Z strings (we know these appear!)
        print("Phase 1: Adding all I/Z strings (analytical)...")
        iz_strings = self._enumerate_IZ_strings(m)
        print(f"  Added {len(iz_strings)} I/Z strings")
        
        for pauli_str in iz_strings:
            # Use ML to predict coefficient
            features = FeatureExtractor.extract_features(pauli_str, m)
            feature_vec = np.array([[features[k] for k in self.learner.feature_names]])
            feature_vec_scaled = self.learner.scaler_coeff.transform(feature_vec)
            coeff = self.learner.coeff_regressor.predict(feature_vec_scaled)[0]
            predicted[pauli_str] = coeff
        
        # PHASE 2: Use ML to find X/Y strings
        print("\nPhase 2: Using ML to find X/Y strings...")
        xy_candidates = self._generate_XY_candidates(m, max_candidates=5000)
        print(f"  Evaluating {len(xy_candidates)} X/Y candidates...")
        
        # Predict which ones exist
        X = []
        for pauli_str in xy_candidates:
            features = FeatureExtractor.extract_features(pauli_str, m)
            feature_vec = [features[k] for k in self.learner.feature_names]
            X.append(feature_vec)
        
        X = np.array(X)
        X_scaled = self.learner.scaler_pattern.transform(X)
        probs = self.learner.pattern_classifier.predict_proba(X_scaled)[:, 1]
        
        # Add high-probability X/Y strings
        selected = np.where(probs >= ml_threshold)[0]
        print(f"  Selected {len(selected)} X/Y strings (threshold={ml_threshold})")
        
        for idx in selected:
            pauli_str = xy_candidates[idx]
            # Predict coefficient
            features = FeatureExtractor.extract_features(pauli_str, m)
            feature_vec = np.array([[features[k] for k in self.learner.feature_names]])
            feature_vec_scaled = self.learner.scaler_coeff.transform(feature_vec)
            coeff = self.learner.coeff_regressor.predict(feature_vec_scaled)[0]
            predicted[pauli_str] = coeff
        
        print(f"\nTotal predicted: {len(predicted)} Pauli strings")
        print(f"  I/Z strings: {len(iz_strings)}")
        print(f"  X/Y strings: {len(selected)}")
        
        return predicted
    
    def _enumerate_IZ_strings(self, m: int) -> List[str]:
        """Enumerate all I/Z strings"""
        return [''.join(p) for p in product(['I', 'Z'], repeat=m)]
    
    def _generate_XY_candidates(self, m: int, max_candidates: int) -> List[str]:
        """
        Generate X/Y candidate strings
        Must have at least one X or Y
        """
        candidates = set()
        
        # Strategy: enumerate by number of X/Y operators
        for n_xy in range(1, min(m, 5) + 1):  # At least 1 X or Y
            for positions in combinations(range(m), n_xy):
                # At these positions, try X or Y
                for xy_choices in product(['X', 'Y'], repeat=n_xy):
                    # Rest can be I or Z
                    # Limit Z positions to avoid explosion
                    if n_xy <= 2:
                        # For low X/Y count, try all I/Z combinations
                        remaining_positions = [i for i in range(m) if i not in positions]
                        for iz_choices in product(['I', 'Z'], repeat=len(remaining_positions)):
                            pauli = [''] * m
                            for pos, xy in zip(positions, xy_choices):
                                pauli[pos] = xy
                            for pos, iz in zip(remaining_positions, iz_choices):
                                pauli[pos] = iz
                            candidates.add(''.join(pauli))
                    else:
                        # For high X/Y count, just sample a few I/Z combinations
                        remaining_positions = [i for i in range(m) if i not in positions]
                        for _ in range(min(10, 2**len(remaining_positions))):
                            pauli = [''] * m
                            for pos, xy in zip(positions, xy_choices):
                                pauli[pos] = xy
                            for pos in remaining_positions:
                                pauli[pos] = np.random.choice(['I', 'Z'])
                            candidates.add(''.join(pauli))
                
                if len(candidates) > max_candidates:
                    return list(candidates)[:max_candidates]
        
        return list(candidates)
    
def main():
    """
    Complete pipeline demonstration with train/test split
    """
    print("\n" + "="*70)
    print(" ML-Based Pauli Decomposition for 1D FEM Poisson ".center(70, "="))
    print("="*70 + "\n")
    
    # Configuration: Train on m=2-8, test on m=9,10
    training_m = [2, 3, 4, 5, 6, 7, 8]
    test_m = [9, 10]  # Hold out for testing
    
    print(f"Training sizes: m={training_m}")
    print(f"Test sizes: m={test_m}")
    print(f"\nWARNING: This will generate data for m up to {max(test_m)}")
    print(f"Estimated time: ~{2**(max(test_m)-5)} minutes for largest size")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted")
        return
    
    # ========================================================================
    # PHASE 1: Generate Training Data
    # ========================================================================
    
    learner = PauliDecompositionLearner()
    training_data = learner.generate_training_data(
        m_values=training_m,
        negative_ratio=3.0,
        epsilon=1e-12
    )
    
    # ========================================================================
    # PHASE 2: Train ML Model
    # ========================================================================
    
    learner.train(training_data)
    
    # ========================================================================
    # PHASE 3: Test on Held-Out Sizes
    # ========================================================================
    
    all_results = []
    
    for m_test in test_m:
        print(f"\n{'='*70}")
        print(f" TEST: m={m_test} (N={2**m_test}) ".center(70, "="))
        print(f"{'='*70}")
        
        # Generate ground truth
        print(f"\nGenerating ground truth for m={m_test}...")
        fem_test = FEM1DPoisson(m_test)
        true_pauli = fem_test.decompose_to_pauli()
        print(f"Ground truth: {len(true_pauli)} Pauli terms")
        
        # Test different thresholds
        thresholds = [0.2, 0.3, 0.4, 0.5]
        
        print(f"\n{'Threshold':^12} {'Predicted':^12} {'Precision':^12} {'Recall':^12} {'F1':^12} {'Matrix Err':^12}")
        print("-" * 74)
        
        best_f1 = 0
        best_result = None
        
        for threshold in thresholds:
            # Predict
            predicted = learner.predict(m_test, threshold=threshold, max_candidates=20000)
            
            # Evaluate
            metrics = evaluate_prediction(predicted, true_pauli, m_test)
            
            print(f"{threshold:^12.2f} {len(predicted):^12d} "
                  f"{metrics['precision']:^12.3f} {metrics['recall']:^12.3f} "
                  f"{metrics['f1_score']:^12.3f} {metrics['matrix_relative_error']:^12.2e}")
            
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_result = {
                    'threshold': threshold,
                    'predicted': predicted,
                    'metrics': metrics
                }
        
        # Use best threshold and optimize
        print(f"\n{'='*70}")
        print(f"Best threshold: {best_result['threshold']}")
        print(f"{'='*70}")
        
        # Optimize coefficients
        optimizer = CoefficientOptimizer()
        optimized = optimizer.optimize(best_result['predicted'], m_test, method='lstsq')
        
        # Final evaluation
        final_metrics = evaluate_prediction(optimized, true_pauli, m_test)
        print_evaluation(final_metrics, f"Final Results (m={m_test})")
        
        # Store results
        all_results.append({
            'm': m_test,
            'threshold': best_result['threshold'],
            'metrics_before_opt': best_result['metrics'],
            'metrics_after_opt': final_metrics
        })
    
    # ========================================================================
    # PHASE 4: Summary
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(" SUMMARY ".center(70, "="))
    print(f"{'='*70}\n")
    
    print(f"{'m':>5} {'Threshold':>12} {'Recall':>10} {'Precision':>12} "
          f"{'F1':>10} {'Matrix Err':>12}")
    print("-" * 70)
    
    for result in all_results:
        m = result['m']
        t = result['threshold']
        met = result['metrics_after_opt']
        print(f"{m:>5} {t:>12.2f} {met['recall']:>10.3f} {met['precision']:>12.3f} "
              f"{met['f1_score']:>10.3f} {met['matrix_relative_error']:>12.2e}")
    
    # ========================================================================
    # PHASE 5: Save Model and Results
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(" Saving Model and Results ".center(70, "="))
    print(f"{'='*70}\n")
    
    model_data = {
        'learner': learner,
        'training_m': training_m,
        'test_m': test_m,
        'results': all_results,
        'training_data_size': len(training_data)
    }
    
    with open('pauli_learner_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved to: pauli_learner_model.pkl")
    
    # Save results as text
    with open('results_summary.txt', 'w') as f:
        f.write("ML-Based Pauli Decomposition Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training sizes: {training_m}\n")
        f.write(f"Test sizes: {test_m}\n\n")
        f.write(f"{'m':>5} {'Threshold':>12} {'Recall':>10} {'Precision':>12} "
                f"{'F1':>10} {'Matrix Err':>12}\n")
        f.write("-" * 70 + "\n")
        for result in all_results:
            m = result['m']
            t = result['threshold']
            met = result['metrics_after_opt']
            f.write(f"{m:>5} {t:>12.2f} {met['recall']:>10.3f} {met['precision']:>12.3f} "
                    f"{met['f1_score']:>10.3f} {met['matrix_relative_error']:>12.2e}\n")
    
    print("Results saved to: results_summary.txt")
    
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETE ".center(70, "="))
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
