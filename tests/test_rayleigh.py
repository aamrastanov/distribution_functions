import numpy as np
from distribution_functions import rayleigh_pdf, rayleigh_cdf

def test_rayleigh_pdf_values():
    # Test known values
    # For sigma=1, f(1) = 1 * exp(-0.5) approx 0.6065
    sigma = 1.0
    x = 1.0
    expected = np.exp(-0.5)
    assert np.isclose(rayleigh_pdf(x, sigma), expected)

    # Test x=0
    assert rayleigh_pdf(0, sigma) == 0

def test_rayleigh_cdf_values():
    # Test known values
    # For sigma=1, F(1) = 1 - exp(-0.5) approx 0.3935
    sigma = 1.0
    x = 1.0
    expected = 1 - np.exp(-0.5)
    assert np.isclose(rayleigh_cdf(x, sigma), expected)

    # Test x=0
    assert rayleigh_cdf(0, sigma) == 0
    
    # Test large x (should approach 1)
    assert np.isclose(rayleigh_cdf(100, sigma), 1.0)

def test_numpy_array_support():
    x = np.array([0, 1, 2])
    sigma = 1.0
    pdf = rayleigh_pdf(x, sigma)
    assert isinstance(pdf, np.ndarray)
    assert len(pdf) == 3
