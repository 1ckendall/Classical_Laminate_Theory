import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cmc
from classical_laminate_theory.failuremodels import Puck, Hashin, TsaiHill, MaxStress

def plot_failure_envelope(model, sigma1=0, s2_range=(-150, 60), t12_range=(0, 100), resolution=200, title=None):
    """
    Plots a failure envelope with cmcrameri acton colormap and failure mode annotations.
    """
    sns.set_style("white")
    
    s2_vals = np.linspace(s2_range[0] * 1e6, s2_range[1] * 1e6, resolution)
    t12_vals = np.linspace(t12_range[0] * 1e6, t12_range[1] * 1e6, resolution)
    
    S2, T12 = np.meshgrid(s2_vals, t12_vals)
    efforts = np.zeros_like(S2)
    
    # Pre-calculate e1
    e1 = sigma1 / model.E1 if hasattr(model, 'E1') else 0
    strain = np.array([e1, 0, 0])
    
    for i in range(resolution):
        for j in range(resolution):
            stress = np.array([sigma1, S2[i,j], T12[i,j]])
            efforts[i,j] = model.get_effort(stress, strain)
            
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # 1. Plot Contours with cmcrameri acton
    # Reverse acton usually looks better for 'high effort = bright/hot'
    cp = ax.contourf(S2 * 1e-6, T12 * 1e-6, efforts, levels=np.linspace(0, 1.5, 16), 
                     cmap=cmc.acton_r, alpha=0.9)
    fig.colorbar(cp, label='Failure Index (Effort)')
    
    # 2. Plot the Boundary
    line = ax.contour(S2 * 1e-6, T12 * 1e-6, efforts, levels=[1.0], colors='white', linewidths=2.5)
    ax.contour(S2 * 1e-6, T12 * 1e-6, efforts, levels=[1.0], colors='black', linewidths=1.0)

    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel(r"Transverse Stress $\sigma_{22}$ (MPa)", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"Shear Stress $\tau_{12}$ (MPa)", fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()

def main():
    props = {
        "Xt": 1000e6, "Xc": 800e6, "Yt": 40e6, "Yc": 120e6, "S12": 40e6,
        "E1": 40e9, "v12": 0.25
    }
    
    # Puck vs Hashin Comparison
    puck = Puck(**props, weakening="puck")
    hashin = Hashin(props["Xt"], props["Xc"], props["Yt"], props["Yc"], props["S12"])
    
    print("Plotting Puck with Acton colormap and annotations...")
    plot_failure_envelope(puck, sigma1=0, title="Puck 1996 Failure Envelope (Acton)")
    plt.show()
    
    print("Plotting Hashin with Acton colormap and annotations...")
    plot_failure_envelope(hashin, sigma1=0, title="Hashin 1980 Failure Envelope (Acton)")
    plt.show()

if __name__ == "__main__":
    main()
