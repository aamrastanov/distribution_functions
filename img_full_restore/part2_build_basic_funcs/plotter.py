import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def plot_file(filename):
    """
    Reads a text file containing a 2D matrix and plots it as a heatmap.
    Assumes values are whitespace separated.
    Origin is top-left.
    """
    logger.info(f"Processing {filename}...")
    
    if not os.path.exists(filename):
        # Try looking in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, filename)
        if os.path.exists(potential_path):
            filename = potential_path
            logger.info(f"Found file in script directory: {filename}")
        else:
            logger.error(f"File not found: {filename}")
            return

    try:
        # Load data
        data = np.loadtxt(filename)
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        # origin='upper' puts (0,0) at top-left.
        # cmap='viridis' is standard, but 'gray' might be good too. Using viridis for contrast.
        plt.imshow(data, cmap='viridis', origin='upper')
        plt.colorbar(label='Value')
        plt.title(f"Heatmap of {os.path.basename(filename)}")
        plt.xlabel("X (Horizontal)")
        plt.ylabel("Y (Vertical)")
        
        # Save
        output_filename = os.path.splitext(filename)[0] + ".png"
        plt.savefig(output_filename)
        logger.info(f"Saved plot to {output_filename}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot {filename}: {e}")

if __name__ == "__main__":
    input_file = "gauss2.txt"
    plot_file(input_file)
