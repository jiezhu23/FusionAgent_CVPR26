
"""
Asymmetric Aggregation Operator (Asym-AO1) Implementation

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List
import math


class AsymAO1:
    """
    Implementation of Asymmetric Aggregation Operator 1 (Asym-AO1) as described in the paper.
    
    Asym-AO1 is a score level fusion technique for multibiometric systems that utilizes
    the generating function of t-norms and a continuous monotone decreasing function
    φ(s1) = 1/s1^m where m is a parameter.
    
    The general form of Asym-AO is given by:
    As(s1, s2) = f^[−1](f(s1) + φ(s1)f(s2))
    
    where f(s1) is a generator function of t-norms and φ(s1) is a continuous 
    monotone decreasing function.
    """
    
    def __init__(self, generator_type: str = 'hamacher', m: float = 2.0):
        """
        Initialize the Asym-AO1 operator with specified generator function and parameter m.
        
        Args:
            generator_type (str): Type of generator function to use ('hamacher', 'algebraic_product', 
                                 or 'aczel_alsina').
            m (float): The exponent parameter for φ(s1) = 1/s1^m.
        """
        self.m = m
        self.generator_type = generator_type.lower()
        
        # Define the generator function and its inverse based on the type
        if self.generator_type == 'hamacher':
            self.f = lambda s: (1 - s) / s
            self.f_inv = lambda s: 1 / (1 + s)
        elif self.generator_type == 'algebraic_product':
            self.f = lambda s: -math.log(s)
            self.f_inv = lambda s: math.exp(-s)
        elif self.generator_type == 'aczel_alsina':
            self.p = 2.0  # Default parameter for Aczel-Alsina
            self.f = lambda s: (-math.log(s)) ** self.p
            self.f_inv = lambda s: math.exp(-(s ** (1/self.p)))
        else:
            raise ValueError(f"Unknown generator type: {generator_type}. "
                            f"Use 'hamacher', 'algebraic_product', or 'aczel_alsina'.")
    
    def phi(self, s1: float) -> float:
        """
        The continuous monotone decreasing function φ(s1) = 1/s1^m.
        
        Args:
            s1 (float): Input score in range [0, 1].
            
        Returns:
            float: The result of the function φ(s1).
        """
        return 1 / (s1 ** self.m)
    
    def fuse(self, s1: float, s2: float) -> float:
        """
        Fuse two scores using the Asym-AO1 operator.
        
        Args:
            s1 (float): First score in range [0, 1].
            s2 (float): Second score in range [0, 1].
            
        Returns:
            float: The fused score in range [0, 1].
        """
        # Ensure scores are in valid range
        if not (0 <= s1 <= 1 and 0 <= s2 <= 1):
            raise ValueError("Scores must be in the range [0, 1]")
        
        # Handle edge cases
        if s1 == 0 or s2 == 0:
            return 0
        
        # Calculate f(s1) and f(s2)
        f_s1 = self.f(s1)
        f_s2 = self.f(s2)
        
        # Calculate φ(s1)
        phi_s1 = self.phi(s1)
        
        # Calculate f^[−1](f(s1) + φ(s1)f(s2))
        result = self.f_inv(f_s1 + phi_s1 * f_s2)
        
        # Ensure result is in valid range due to numerical issues
        return min(max(result, 0), 1)
    
    def visualize(self, resolution: int = 100, save_path: Optional[str] = None) -> None:
        """
        Visualize the Asym-AO1 fusion operator as a 3D surface plot.
        
        Args:
            resolution (int): The resolution of the plot grid.
            save_path (Optional[str]): If provided, save the plot to this file path.
        """
        # Create a grid of scores
        s1_range = np.linspace(0.01, 1, resolution)
        s2_range = np.linspace(0.01, 1, resolution)
        s1_grid, s2_grid = np.meshgrid(s1_range, s2_range)
        
        # Calculate fused scores for each point in the grid
        fused_scores = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                try:
                    fused_scores[i, j] = self.fuse(s1_grid[i, j], s2_grid[i, j])
                except:
                    fused_scores[i, j] = 0
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(s1_grid, s2_grid, fused_scores, cmap='viridis', alpha=0.8)
        
        # Add colorbar and labels
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('s1')
        ax.set_ylabel('s2')
        ax.set_zlabel('Fused Score')
        ax.set_title(f'Asym-AO1 with {self.generator_type.capitalize()} function (m={self.m})')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_with_min_max(self, s1_list: list, s2_list: list) -> Tuple[list, list, list]:
        """
        Compare Asym-AO1 fusion with simple min and max rules for a list of score pairs.
        
        Args:
            s1_list (list): List of first scores.
            s2_list (list): List of second scores.
            
        Returns:
            Tuple[list, list, list]: Tuple of (asym_fused_scores, min_scores, max_scores).
        """
        if len(s1_list) != len(s2_list):
            raise ValueError("The lists of scores must have the same length")
        
        asym_fused = []
        min_scores = []
        max_scores = []
        
        for s1, s2 in zip(s1_list, s2_list):
            asym_fused.append(self.fuse(s1, s2))
            min_scores.append(min(s1, s2))
            max_scores.append(max(s1, s2))
            
        return asym_fused, min_scores, max_scores


# Example implementations for the specific Asym-AO1 forms mentioned in the paper

def asym_ao1_hamacher(s1: float, s2: float, m: float = 2.0) -> float:
    """
    Implementation of Asym-AO1 with Hamacher t-norm generator function.
    
    As defined in the paper:
    Asym(s1, s2) = s1*s2 / (s1^m * (1-s2) + s1^(m-1) * s2)
    
    Args:
        s1 (float): First score in range [0, 1].
        s2 (float): Second score in range [0, 1].
        m (float): The exponent parameter.
        
    Returns:
        float: The fused score in range [0, 1].
    """
    # Ensure scores are in valid range
    if not (0 <= s1 <= 1 and 0 <= s2 <= 1):
        raise ValueError("Scores must be in the range [0, 1]")
    
    # Handle edge cases
    if s1 == 0 or s2 == 0:
        return 0
    
    numerator = s1 * s2
    denominator = (s1**m * (1-s2)) + (s1**(m-1) * s2)
    
    if denominator == 0:
        return 1 if numerator > 0 else 0
        
    return numerator / denominator


def asym_ao1_algebraic_product(s1: float, s2: float, m: float = 2.0) -> float:
    """
    Implementation of Asym-AO1 with Algebraic Product t-norm generator function.
    
    As defined in the paper:
    Asym(s1, s2) = s1 * (s2)^(1/s1^m)
    
    Args:
        s1 (float): First score in range [0, 1].
        s2 (float): Second score in range [0, 1].
        m (float): The exponent parameter.
        
    Returns:
        float: The fused score in range [0, 1].
    """
    # Ensure scores are in valid range
    if not (0 <= s1 <= 1 and 0 <= s2 <= 1):
        raise ValueError("Scores must be in the range [0, 1]")
    
    # Handle edge cases
    if s1 == 0 or s2 == 0:
        return 0
    if s1 == 1:
        return s2
    if s2 == 1:
        return s1
    
    # Calculate s1 * (s2)^(1/s1^m)
    exponent = 1 / (s1**m)
    return s1 * (s2**exponent)


def fuse_score_matrices(score_matrices: List[np.ndarray], 
                        generator_type: str = 'hamacher',
                        m: float = 2.0) -> np.ndarray:
    """
    Fuse multiple score matrices into a single fused score matrix.
    
    Args:
        score_matrices: List of score matrices, each with shape (B, n_gallery) where:
                       - B is the number of probe samples
                       - n_gallery is the number of gallery samples
        fusion_method: The fusion method to use ('asym_ao1', 'min', 'max', 'mean', 'product')
        generator_type: For 'asym_ao1', the generator function type ('hamacher', 'algebraic_product', 'aczel_alsina')
        m: For 'asym_ao1', the exponent parameter m
        
    Returns:
        fused_scores: A single fused score matrix with the same shape (B, n_gallery)
    """
    # Validate input
    if not score_matrices:
        raise ValueError("No score matrices provided")
    
    # Get shapes and check consistency
    shape = score_matrices[0].shape
    for i, matrix in enumerate(score_matrices):
        if matrix.shape != shape:
            raise ValueError(f"Score matrix {i} has shape {matrix.shape}, but expected {shape}")
    
    B, n_gallery = shape
    
    # Initialize fused score matrix
    fused_scores = np.zeros(shape)
    # Initialize the Asym-AO1 operator
    asym_ao1 = AsymAO1(generator_type=generator_type, m=m)
    
    # Initialize with the first score matrix
    fused_scores = score_matrices[0].copy()
    
    # Fuse scores pairwise
    for i in range(1, len(score_matrices)):
        # For each element in the matrices
        for b in range(B):
            for g in range(n_gallery):
                s1 = fused_scores[b, g]
                s2 = score_matrices[i][b, g]
                fused_scores[b, g] = asym_ao1.fuse(s1, s2)
                    
    return fused_scores


if __name__ == "__main__":
    face_scores = np.random.rand(10, 100)      # Your first score matrix (B, n_gallery)
    fingerprint_scores = np.random.rand(10, 100) # Your second score matrix (B, n_gallery)
    iris_scores = np.random.rand(10, 100)      # Your third score matrix (B, n_gallery)

    # Combine them into a list
    score_matrices = [face_scores, fingerprint_scores, iris_scores]
    # Using Hamacher t-norm generator with m=2.0

    fused_scores = fuse_score_matrices(
    score_matrices=score_matrices,
    generator_type='hamacher',  # or 'algebraic_product' or 'aczel_alsina'
    m=2.0  # adjust this parameter as needed
    )
    