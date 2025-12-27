"""
Plotting and Visualization Utilities
=====================================

Common plotting functions for quantum computing visualizations.

This module provides consistent plotting styles and utility functions
for circuit diagrams, statevectors, measurement results, and more.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_measurement_results(counts, title="Measurement Results", figsize=(10, 6)):
    """
    Create a bar plot of quantum measurement results.
    
    Parameters
    ----------
    counts : dict
        Dictionary of measurement outcomes and counts
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    
    Example
    -------
    >>> counts = {'00': 250, '01': 245, '10': 255, '11': 250}
    >>> fig = plot_measurement_results(counts)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    states = list(counts.keys())
    values = list(counts.values())
    
    ax.bar(states, values, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Measurement Outcome', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-labels if many states
    if len(states) > 8:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_statevector_bar(statevector, threshold=1e-10, title="Statevector"):
    """
    Create a bar plot showing statevector amplitudes.
    
    Parameters
    ----------
    statevector : array-like
        Complex statevector amplitudes
    threshold : float, optional
        Minimum magnitude to display
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    n_qubits = int(np.log2(len(statevector)))
    
    # Filter by threshold
    indices = []
    real_parts = []
    imag_parts = []
    magnitudes = []
    
    for i, amp in enumerate(statevector):
        if abs(amp) > threshold:
            indices.append(format(i, f'0{n_qubits}b'))
            real_parts.append(amp.real)
            imag_parts.append(amp.imag)
            magnitudes.append(abs(amp))
    
    if not indices:
        print("No amplitudes above threshold")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot real and imaginary parts
    x = np.arange(len(indices))
    width = 0.35
    
    ax1.bar(x - width/2, real_parts, width, label='Real', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, imag_parts, width, label='Imaginary', color='coral', alpha=0.7)
    ax1.set_xlabel('Basis State', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'{title} - Real & Imaginary', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'|{s}⟩' for s in indices], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    
    # Plot magnitudes
    ax2.bar(indices, magnitudes, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Basis State', fontsize=12)
    ax2.set_ylabel('|Amplitude|', fontsize=12)
    ax2.set_title(f'{title} - Magnitude', fontsize=13, fontweight='bold')
    ax2.set_xticklabels([f'|{s}⟩' for s in indices], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_bloch_vector(theta, phi, title="Qubit on Bloch Sphere"):
    """
    Plot a qubit state on the Bloch sphere.
    
    Parameters
    ----------
    theta : float
        Polar angle (0 to π)
    phi : float
        Azimuthal angle (0 to 2π)
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    
    Notes
    -----
    Qubit state: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    """
    try:
        from qiskit.visualization import plot_bloch_vector
        
        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        fig = plot_bloch_vector([x, y, z], title=title)
        return fig
    except ImportError:
        print("Qiskit visualization not available. Install with: pip install qiskit[visualization]")
        return None


def plot_convergence(iteration_data, ylabel="Cost Function", title="Convergence"):
    """
    Plot convergence of iterative algorithms (e.g., VQLS, VQE).
    
    Parameters
    ----------
    iteration_data : array-like
        Values at each iteration
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(len(iteration_data))
    ax.plot(iterations, iteration_data, 'o-', linewidth=2, markersize=6, 
            color='steelblue', markerfacecolor='coral')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def set_plot_style():
    """
    Set consistent plot style for all figures in the textbook.
    
    Call this at the beginning of each notebook for consistent styling.
    """
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
