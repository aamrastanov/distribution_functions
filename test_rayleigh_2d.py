from distribution_functions import plot_rayleigh_2d

if __name__ == "__main__":
    # Generate 2D Rayleigh plot with different sigma values
    sigma=1.0
    plot_rayleigh_2d(sigma=sigma, x_lim=(-5, 5), y_lim=(-5, 5), 
                     output_file=f"rayleigh_2d_sigma{sigma}.png", resolution=150)
