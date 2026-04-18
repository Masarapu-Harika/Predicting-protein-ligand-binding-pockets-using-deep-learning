"""
visualize.py
------------
3-D visualization of predicted pockets on a protein structure.

Outputs:
  • pocket_visualization.png   — static matplotlib figure
  • pocket_visualization.html  — interactive Plotly figure (optional)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def visualize_pockets(
    coords: np.ndarray,
    pockets: list[dict],
    out_png: str = "pocket_visualization.png",
    out_html: str | None = "pocket_visualization.html",
    title: str = "FA-GAT Predicted Binding Pockets",
) -> None:
    """
    Render a 3D scatter plot of the protein with predicted pocket residues
    highlighted.

    Parameters
    ----------
    coords   : np.ndarray, shape (N, 3) — all residue C-alpha coordinates
    pockets  : list of dict — output of pocket_detection.detect_pockets
    out_png  : str — path to save the static PNG
    out_html : str or None — path to save the interactive HTML (Plotly)
    title    : str — plot title
    """
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    # All residues in light grey
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c="lightgrey", s=10, alpha=0.4, label="Protein backbone"
    )

    # Each pocket in a distinct colour
    cmap   = plt.cm.get_cmap("tab10", max(len(pockets), 1))
    for i, pocket in enumerate(pockets):
        mask   = pocket["residue_mask"]
        pc     = coords[mask]
        center = pocket["center"]
        color  = cmap(i)
        ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            c=[color], s=40, alpha=0.85,
            label=f"Pocket {i+1} (n={pocket['size']}, p={pocket['mean_prob']:.2f})"
        )
        ax.scatter(
            *center, marker="*", s=300, c=[color], edgecolors="black", zorder=5
        )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved static plot → {out_png}")

    # ---- Plotly interactive ----
    if out_html:
        try:
            import plotly.graph_objects as go
            traces = [
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="lightgrey", opacity=0.4),
                    name="Backbone",
                )
            ]
            for i, pocket in enumerate(pockets):
                mask = pocket["residue_mask"]
                pc   = coords[mask]
                traces.append(go.Scatter3d(
                    x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                    mode="markers",
                    marker=dict(size=5, opacity=0.9),
                    name=f"Pocket {i+1}",
                ))
                center = pocket["center"]
                traces.append(go.Scatter3d(
                    x=[center[0]], y=[center[1]], z=[center[2]],
                    mode="markers",
                    marker=dict(size=10, symbol="x", opacity=1.0),
                    name=f"Center {i+1}",
                ))
            fig2 = go.Figure(data=traces)
            fig2.update_layout(title=title)
            fig2.write_html(out_html)
            print(f"Saved interactive plot → {out_html}")
        except ImportError:
            print("Plotly not installed — skipping HTML output.")
