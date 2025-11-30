from distribution_functions import plot_rayleigh_2d

if __name__ == "__main__":
    # Generate 2D Rayleigh plot with different sigma values
    sigma=2.0
    shift=0.0
    plot_rayleigh_2d(sigma=sigma, x_lim=(-10, 10), y_lim=(-10, 10), 
                     output_file=f"rayleigh_2d_shift{shift}_sigma{sigma}.png", resolution=150,
                     xShift=shift, yShift=shift)
