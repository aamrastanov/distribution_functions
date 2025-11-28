import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0  # Modified Bessel function of the first kind, order 0
from scipy.integrate import quad
from typing import Union, List


def rice_pdf(x: Union[float, np.ndarray], nu: float = 0.0, sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Calculate the Probability Density Function (PDF) of the Rice distribution.
    
    f(x; nu, sigma) = (x / sigma^2) * exp(-(x^2 + nu^2) / (2 * sigma^2)) * I_0(x * nu / sigma^2)
    
    Args:
        x: Random variable (x >= 0)
        nu: Non-centrality parameter (nu >= 0). When nu=0, reduces to Rayleigh distribution
        sigma: Scale parameter (sigma > 0)
    
    Returns:
        The probability density at x
    """
    x = np.asarray(x)
    
    # For x < 0, PDF is 0
    result = np.zeros_like(x, dtype=float)
    
    # Only calculate for x >= 0
    mask = x >= 0
    x_valid = x[mask] if mask.any() else x
    
    if np.any(mask):
        # Calculate the PDF
        factor = x_valid / (sigma ** 2)
        exponent = np.exp(-(x_valid ** 2 + nu ** 2) / (2 * sigma ** 2))
        bessel = i0(x_valid * nu / (sigma ** 2))
        
        if isinstance(result, np.ndarray):
            result[mask] = factor * exponent * bessel
        else:
            result = factor * exponent * bessel
    
    return result


def rice_cdf(x: Union[float, np.ndarray], nu: float = 0.0, sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Calculate the Cumulative Distribution Function (CDF) of the Rice distribution.
    
    This is computed by numerical integration of the PDF.
    
    Args:
        x: Random variable (x >= 0)
        nu: Non-centrality parameter (nu >= 0)
        sigma: Scale parameter (sigma > 0)
    
    Returns:
        The cumulative probability at x
    """
    x = np.asarray(x)
    
    # Vectorize the integration for array inputs
    def single_cdf(x_val):
        if x_val <= 0:
            return 0.0
        result, _ = quad(lambda t: rice_pdf(t, nu, sigma), 0, x_val)
        return result
    
    if x.ndim == 0:
        return single_cdf(float(x))
    else:
        return np.array([single_cdf(x_val) for x_val in x.flat]).reshape(x.shape)


def plot_rice(nu_values: List[float], sigma: float = 1.0, output_file: str = "rice_plot.png"):
    """
    Plot the Rice PDF for different nu values.
    
    Args:
        nu_values: List of non-centrality parameters to plot
        sigma: Scale parameter (default: 1.0)
        output_file: Output filename for the plot
    """
    x = np.linspace(0, 10, 1000)
    
    plt.figure(figsize=(10, 6))
    for nu in nu_values:
        y = rice_pdf(x, nu, sigma)
        label = f'ν={nu}' if nu > 0 else f'ν={nu} (Rayleigh)'
        plt.plot(x, y, label=label)
        
    plt.title(f'Rice Distribution PDF (σ={sigma})')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()


def rice_2d(x: Union[float, np.ndarray], y: Union[float, np.ndarray], 
            nu: float = 0.0, sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Calculate the 2D Rice distribution PDF based on x and y coordinates.
    
    The 2D Rice distribution is calculated as:
    f(x, y; nu, sigma) = f(r; nu, sigma) where r = sqrt(x^2 + y^2)
    
    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)
        nu: Non-centrality parameter (default: 0.0)
        sigma: Scale parameter (default: 1.0)
    
    Returns:
        The probability density at the given (x, y) point(s)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt(x**2 + y**2)
    return rice_pdf(r, nu, sigma)


def plot_rice_2d(nu: float = 0.0, sigma: float = 1.0, x_lim: tuple = (-5, 5), 
                 y_lim: tuple = (-5, 5), output_file: str = "rice_2d_plot.png", 
                 resolution: int = 100):
    """
    Plot the 2D Rice distribution as a 3D surface plot.
    
    Args:
        nu: Non-centrality parameter for the Rice distribution
        sigma: Scale parameter for the Rice distribution
        x_lim: Tuple of (min, max) for x-axis
        y_lim: Tuple of (min, max) for y-axis
        output_file: Output filename for the plot
        resolution: Number of points along each axis
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create meshgrid
    x = np.linspace(x_lim[0], x_lim[1], resolution)
    y = np.linspace(y_lim[0], y_lim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate 2D Rice PDF
    Z = rice_2d(X, Y, nu, sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, edgecolor='none')
    
    # Add contour lines at the bottom
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='plasma', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Probability Density')
    title = f'2D Rice Distribution (ν={nu}, σ={sigma})'
    if nu == 0:
        title += ' [Rayleigh]'
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"2D plot saved to {output_file}")
    plt.close()
