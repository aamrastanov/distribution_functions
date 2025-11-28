from distribution_functions import plot_rice_2d

if __name__ == "__main__":
    # Generate 2D Rice plot with nu=0 (equivalent to Rayleigh)
    plot_rice_2d(nu=0.0, sigma=1.0, x_lim=(-5, 5), y_lim=(-5, 5), 
                 output_file="rice_2d_nu0.png", resolution=150)
    
    # Generate 2D Rice plot with nu=2
    plot_rice_2d(nu=2.0, sigma=1.0, x_lim=(-5, 5), y_lim=(-5, 5), 
                 output_file="rice_2d_nu2.png", resolution=150)
    
    # Generate 2D Rice plot with nu=4
    plot_rice_2d(nu=4.0, sigma=1.0, x_lim=(-8, 8), y_lim=(-8, 8), 
                 output_file="rice_2d_nu4.png", resolution=150)
