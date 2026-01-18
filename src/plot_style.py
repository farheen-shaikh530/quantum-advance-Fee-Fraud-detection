# src/plot_style.py
import matplotlib.pyplot as plt

THEME = {
    "BLUE":   "#1f77b4",
    "YELLOW": "#ffbf00",
    "GREY":   "#7f7f7f",
}

def apply():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": THEME["GREY"],
        "grid.alpha": 0.25,
        "axes.grid": False,
        "font.size": 12,
    })