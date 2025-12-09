from distribution_functions import calculate_gaussian_integral

if __name__ == "__main__":
    sigma = 1.0
    mu1 = 5.0
    mu2 = 5.0
    limit = 1000.0
    
    print(f"Calculating Gaussian integral for sigma={sigma}, mu1={mu1}, mu2={mu2}, limit={limit}")
    result = calculate_gaussian_integral(sigma, mu1, mu2, limit)
    print(f"Result: {result}")
    
