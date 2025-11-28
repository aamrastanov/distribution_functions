import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List

def rayleigh_pdf(x: Union[float, np.ndarray], sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Calculate the Probability Density Function (PDF) of the Rayleigh distribution.
    
    f(x; sigma) = (x / sigma^2) * exp(-x^2 / (2 * sigma^2)) for x >= 0
    """
    x = np.asarray(x)
    return (x / sigma**2) * np.exp(-x**2 / (2 * sigma**2)) * (x >= 0)

def rayleigh_cdf(x: Union[float, np.ndarray], sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Calculate the Cumulative Distribution Function (CDF) of the Rayleigh distribution.
    
    F(x; sigma) = 1 - exp(-x^2 / (2 * sigma^2)) for x >= 0
    """
    x = np.asarray(x)
    return (1 - np.exp(-x**2 / (2 * sigma**2))) * (x >= 0)

def plot_rayleigh(sigma_values: List[float], output_file: str = "rayleigh_plot.png"):
    """
    Plot the Rayleigh PDF for different sigma values.
    """
    x = np.linspace(0, 10, 1000)
    
    plt.figure(figsize=(10, 6))
    for sigma in sigma_values:
        y = rayleigh_pdf(x, sigma)
        plt.plot(x, y, label=f'sigma={sigma}')
        
    plt.title('Rayleigh Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

def rayleigh_2d(x: Union[float, np.ndarray], y: Union[float, np.ndarray], sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Calculate the 2D Rayleigh distribution PDF based on x and y coordinates.
    
    The 2D Rayleigh distribution is calculated as:
    f(x, y; sigma) = f(r; sigma) where r = sqrt(x^2 + y^2)
    
    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)
        sigma: Scale parameter (default: 1.0)
    
    Returns:
        The probability density at the given (x, y) point(s)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt(x**2 + y**2)
    return rayleigh_pdf(r, sigma)

def plot_rayleigh_2d(sigma: float = 1.0, x_lim: tuple = (-5, 5), y_lim: tuple = (-5, 5), 
                     output_file: str = "rayleigh_2d_plot.png", resolution: int = 100):
    """
    Plot the 2D Rayleigh distribution as a 3D surface plot.
    
    Args:
        sigma: Scale parameter for the Rayleigh distribution
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
    
    # Calculate 2D Rayleigh PDF
    Z = rayleigh_2d(X, Y, sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Add contour lines at the bottom
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Probability Density')
    ax.set_title(f'2D Rayleigh Distribution (sigma={sigma})')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"2D plot saved to {output_file}")
    plt.close()
