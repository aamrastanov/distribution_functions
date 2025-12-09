import numpy as np
import mpmath
from typing import Union

def calculate_rice_integral(lam: float, sigma1: float, sigma2: float, limit: float = np.inf) -> float:
    """
    Calculate the integral of P(x) = R1(x) * R2(x) * K(x) from 0 to limit.
    
    Uses mpmath for arbitrary precision arithmetic to handle large values that
    would overflow standard float64 types.
    
    Args:
        lam: Lambda parameter (used as nu for Rice distributions)
        sigma1: Sigma parameter for first Rice distribution
        sigma2: Sigma parameter for second Rice distribution
        limit: Upper bound of integration (default: infinity)
        
    Returns:
        The value of the integral (as a standard float).
    """
    
    # Set precision (decimal places)
    mpmath.mp.dps = 50
    
    # Convert inputs to mpmath types
    lam_mp = mpmath.mpf(lam)
    s1_mp = mpmath.mpf(sigma1)
    s2_mp = mpmath.mpf(sigma2)
    
    # Handle limit
    if limit == np.inf:
        limit_mp = mpmath.inf
    else:
        limit_mp = mpmath.mpf(limit)
    
    def rice_pdf_mp(x, nu, sigma):
        # f(x; nu, sigma) = (x / sigma^2) * exp(-(x^2 + nu^2) / (2 * sigma^2)) * I_0(x * nu / sigma^2)
        if x < 0:
            return mpmath.mpf(0)
        
        sigma2 = sigma**2
        term1 = x / sigma2
        term2 = mpmath.exp(-(x**2 + nu**2) / (2 * sigma2))
        term3 = mpmath.besseli(0, x * nu / sigma2)
        return term1 * term2 * term3

    def kernel_k_mp(x):
        # K(x) = ((1 - x) * exp(-x + 5*x^2/8)) / (x^2 * I_0(x) * I_0(4x))
        if x == 0:
            return mpmath.mpf(0) # Limit behavior
            
        numerator = (1 - x) * mpmath.exp(-x + (5 * x**2) / 8)
        denominator = (x**2) * mpmath.besseli(0, x) * mpmath.besseli(0, 4 * x)
        
        return numerator / denominator

    def integrand(x):
        if x <= 0:
            return mpmath.mpf(0)
            
        r1 = rice_pdf_mp(x, lam_mp, s1_mp)
        r2 = rice_pdf_mp(x, lam_mp, s2_mp)
        k = kernel_k_mp(x)
        
        return r1 * r2 * k

    # Perform integration
    # mpmath.quad returns an mpf object
    result = mpmath.quad(integrand, [0, limit_mp])
    
    # Convert back to float for the user
    return float(result)

def calculate_gaussian_integral(sigma: float, mu1: float, mu2: float, limit: float) -> float:
    """
    Calculate the integral of P(x) = R1(x) * R2(x) * K(x) from 0 to limit.
    
    R1(x) = Gaussian(x; mu=mu1, sigma=sigma)
    R2(x) = Gaussian(x; mu=mu2, sigma=sigma)
    K(x) = (x^3/sigma^6 - 3x/sigma^4) * exp(-x^2/(2*sigma^2))
    
    Uses mpmath for arbitrary precision arithmetic.
    
    Args:
        sigma: Standard deviation
        mu1: Mean of first Gaussian
        mu2: Mean of second Gaussian
        limit: Upper bound of integration
        
    Returns:
        The value of the integral.
    """
    
    # Set precision
    mpmath.mp.dps = 50
    
    sigma_mp = mpmath.mpf(sigma)
    mu1_mp = mpmath.mpf(mu1)
    mu2_mp = mpmath.mpf(mu2)
    limit_mp = mpmath.mpf(limit)
    
    def gaussian_pdf_mp(x, mu, sigma):
        # f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)**2)
        sigma_sqrt2pi = sigma * mpmath.sqrt(2 * mpmath.pi)
        exponent = -0.5 * ((x - mu) / sigma)**2
        return (1 / sigma_sqrt2pi) * mpmath.exp(exponent)
        
    def kernel_g_mp(x):
        # g(x) = (x^3/sigma^6 - 3x/sigma^4) * exp(-x^2/(2*sigma^2))
        sigma2 = sigma_mp**2
        sigma4 = sigma2**2
        sigma6 = sigma4 * sigma2
        
        term1 = (x**3 / sigma6) - (3 * x / sigma4)
        term2 = mpmath.exp(-(x**2) / (2 * sigma2))
        
        return term1 * term2
        
    def integrand(x):
        r1 = gaussian_pdf_mp(x, mu1_mp, sigma_mp)
        r2 = gaussian_pdf_mp(x, mu2_mp, sigma_mp)
        k = kernel_g_mp(x)
        
        return r1 * r2 * k
        
    result = mpmath.quad(integrand, [0, limit_mp])
    return float(result)
