import os
import sys

# Add current directory and project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Two levels up to reach project root from base/plot_gabor/
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from base.gabor_bank import create_gabor_bank

def visualize_gabor_3d_interactive(orientations=2, scales=[2.0, 4.0], frequencies=[0.1], ksize=63):
    """
    Generate an interactive HTML visualization with 3D surface plots of Gabor filters.
    """
    print(f"Generating Gabor bank for 3D visualization (ksize={ksize})...")
    bank = create_gabor_bank(
        orientations=orientations,
        scales=scales,
        frequencies=frequencies,
        ksize=ksize
    )
    
    n_filters = len(bank)
    n_cols = len(scales) * len(frequencies)
    n_rows = orientations
    
    # Create subplots grid
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'surface'}] * n_cols] * n_rows,
        subplot_titles=[f"θ={np.degrees(f['theta']):.0f}°, σ={f['sigma']}, ω={f['omega']}" for f in bank],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    x = np.linspace(-ksize//2, ksize//2, ksize)
    y = np.linspace(-ksize//2, ksize//2, ksize)
    X, Y = np.meshgrid(x, y)
    
    for i, filt in enumerate(bank):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        kernel = filt['kernel']
        
        fig.add_trace(
            go.Surface(z=kernel, x=X, y=Y, colorscale='Viridis', showscale=False),
            row=row, col=col
        )
        
        # Update layout for each subplot
        fig.update_scenes(
            dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
                xaxis_showticklabels=False,
                yaxis_showticklabels=False,
                zaxis_showticklabels=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title_text=f"Interactive 3D Gabor Filter Bank (ksize={ksize})",
        height=n_rows * 400,
        width=n_cols * 500,
        showlegend=False
    )
    
    output_html = "gabor_3d_interactive.html"
    fig.write_html(output_html)
    print(f"Interactive visualization saved to {output_html}")
    
    # Also save individual plots for the first few filters
    for i in range(min(2, len(bank))):
        f = bank[i]
        single_fig = go.Figure(data=[go.Surface(z=f['kernel'], colorscale='Plasma')])
        single_fig.update_layout(
            title=f"Gabor Filter: θ={np.degrees(f['theta']):.0f}°, σ={f['sigma']}, ω={f['omega']}",
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        single_output = f"gabor_3d_single_{i}.html"
        single_fig.write_html(single_output)
        print(f"Single filter interactive plot saved to {single_output}")

if __name__ == "__main__":
    # Small set for performance
    visualize_gabor_3d_interactive(
        orientations=2,
        scales=[2.0, 5.0],
        frequencies=[0.1],
        ksize=63
    )
