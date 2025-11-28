from distribution_functions import plot_rice

if __name__ == "__main__":
    # Plot Rice distribution with different nu values
    # nu=0 is equivalent to Rayleigh distribution
    plot_rice([0.0, 1.0, 2.0, 3.0, 4.0], sigma=1.0, output_file="rice_plot.png")
