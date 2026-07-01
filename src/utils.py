import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data from a given path.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at path: {filepath}")
    return pd.read_csv(filepath)

def configure_plotting_style():
    """
    Configures consistent premium styling for matplotlib/seaborn plots.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'sans-serif',
        'figure.dpi': 120,
        'savefig.bbox': 'tight'
    })
    
    # Custom color palette (cool tech look)
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
    sns.set_palette(sns.color_palette(colors))
