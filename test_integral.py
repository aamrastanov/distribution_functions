from distribution_functions import calculate_rice_integral

if __name__ == "__main__":
    lam = 4.0
    sigma1 =1.0
    sigma2 = 1.0
    limit = 50.0
    
    print(f"Calculating integral for lambda={lam}, sigma1={sigma1}, sigma2={sigma2}, limit={limit}")
    result = calculate_rice_integral(lam, sigma1, sigma2, limit=limit)
    print(f"Result: {result}")
    
