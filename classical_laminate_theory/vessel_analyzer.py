import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from classical_laminate_theory.structures import Material, Laminate, Lamina
from classical_laminate_theory.failuremodels import Puck
from isotensoid.isotensoid_generator import IsotensoidProfile

class DomeAnalyzer:
    def __init__(self, profile: IsotensoidProfile, material: Material, t_helical_equator: float):
        """
        Analyzes the varying laminate properties and loading across an isotensoid dome.
        
        Args:
            profile: Initialized IsotensoidProfile instance.
            material: Material instance (Carbon/Epoxy etc).
            t_helical_equator: Total thickness of helical layers at the equator (m).
        """
        self.profile = profile
        self.material = material
        self.t_eq = t_helical_equator
        
        # Pre-calculate Equator properties
        self.cos_alpha_eq = np.cos(self.profile.winding_angle_rad)

    def get_station_properties(self, r: float):
        """
        Calculates local angle and thickness at radius r.
        """
        # 1. Clairaut's Law: r * sin(alpha) = r0
        # Guard against r < r0
        r_clamped = max(r, self.profile.r0 + 1e-6)
        sin_alpha = self.profile.r0 / r_clamped
        alpha_rad = np.arcsin(min(sin_alpha, 1.0))
        alpha_deg = np.degrees(alpha_rad)
        
        # 2. Thickness Variation (Vassiliev / Netting theory approximation)
        # t(r) = t_eq * (R * cos(alpha_eq)) / (r * cos(alpha_r))
        cos_alpha_r = np.cos(alpha_rad)
        t_r = self.t_eq * (self.profile.R * self.cos_alpha_eq) / (r_clamped * cos_alpha_r)
        
        return alpha_deg, t_r

    def get_membrane_loads(self, r: float, pressure: float):
        """
        Calculates membrane loads N_phi (meridional) and N_theta (hoop).
        Uses the Isotensoid assumption: Load is carried entirely by fibers.
        """
        # In a perfectly balanced isotensoid dome:
        # N_phi = (P * r) / (2 * sin(phi))  -- where phi is slope angle
        # For our approximation, we use the radius-based projection
        
        # Simplified Membrane approximation for pressure vessels:
        # Axial force balance at radius r: 
        # Total force = P * pi * r^2
        # Carried by Meridional Load N_phi over circumference 2*pi*r
        # N_phi * 2*pi*r = P * pi * r^2  => N_phi = (P * r) / 2
        
        N_phi = (pressure * r) / 2.0
        
        # In an isotensoid dome, N_theta is determined by the fiber angle requirement
        # to keep fiber stress constant. 
        # N_theta = N_phi * tan^2(alpha)
        alpha_deg, _ = self.get_station_properties(r)
        alpha_rad = np.radians(alpha_deg)
        N_theta = N_phi * (np.tan(alpha_rad)**2)
        
        return np.array([N_phi, N_theta, 0, 0, 0, 0])

    def analyze_dome(self, pressure: float, num_stations: int = 50):
        """
        Performs a sweep from equator to boss.
        """
        radii = np.linspace(self.profile.R, self.profile.r0 + 1e-3, num_stations)
        results = {
            "radius": [],
            "angle": [],
            "thickness": [],
            "effort": [],
        }
        
        # Setup failure model
        model = Puck(material=self.material, weakening="puck")
        
        for r in radii:
            alpha, t_local = self.get_station_properties(r)
            load = self.get_membrane_loads(r, pressure)

            # [+alpha/-alpha]_s has 4 plies; pass per-ply thickness directly so
            # from_layup builds the laminate correctly in one pass.
            lam = Laminate.from_layup(
                f"[+{alpha}/-{alpha}]_s",
                failure_model=model,
                t=t_local / 4,
            )

            # Apply load and check max effort
            lam.apply_load(load)
            
            max_effort = 0
            for i in range(lam.n_plies):
                effort = model.get_effort(lam.local_stresses[i], lam.local_strains[i])
                max_effort = max(max_effort, effort)
                
            results["radius"].append(r)
            results["angle"].append(alpha)
            results["thickness"].append(t_local) # meters
            results["effort"].append(max_effort)
            
        return results

def main():
    # 1. Define Vessel Geometry (meters)
    R_equator = 0.100  # 100mm
    R_boss = 0.025     # 25mm
    pressure = 20e6    # 20 MPa Operating Pressure
    
    iso = IsotensoidProfile(R=R_equator, r0=R_boss)
    
    # 2. Define Material
    props = {
        "E1": 140e9, "E2": 10e9, "G12": 5e9, "v12": 0.3,
        "Xt": 2500e6, "Xc": 2000e6, "Yt": 50e6, "Yc": 150e6, "S12": 80e6
    }
    carbon_epoxy = Material(**props)
    
    # 3. Analyze
    t_eq = 0.002 # 2mm helical thickness at equator
    analyzer = DomeAnalyzer(iso, carbon_epoxy, t_helical_equator=t_eq)
    data = analyzer.analyze_dome(pressure)
    
    # 4. Visualization
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Radius (m)', fontweight='bold')
    ax1.set_ylabel('Fiber Angle (deg)', color=color, fontweight='bold')
    ax1.plot(data["radius"], data["angle"], color=color, linewidth=2, label="Fiber Angle")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Puck Failure Index (Effort)', color=color, fontweight='bold')
    ax2.plot(data["radius"], data["effort"], color=color, linewidth=2, linestyle='--', label="Failure Index")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)

    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel('thickness', color=color, fontweight='bold')
    ax3.plot(data["radius"], data["thickness"], color=color, linewidth=2, linestyle='--', label="Thickness")
    ax3.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Isotensoid Dome Analysis @ {pressure/1e6:.1f} MPa\nEquator R={R_equator}m, Boss R={R_boss}m", fontweight='bold')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
