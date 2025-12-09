from .rayleigh import rayleigh_pdf, rayleigh_cdf, plot_rayleigh, rayleigh_2d, plot_rayleigh_2d
from .rice import rice_pdf, rice_cdf, plot_rice, rice_2d, plot_rice_2d
from .custom_integral import calculate_rice_integral, calculate_gaussian_integral

def hello() -> str:
    return "Hello from distribution_functions!"

if __name__ == "__main__":
    print(hello())
