import numpy as np
import itertools
from classical_laminate_theory.structures import Material, Laminate
from classical_laminate_theory.failuremodels import Puck, FailureMode
from classical_laminate_theory.progressive_failure_analysis import ProgressiveFailureAnalysis
from classical_laminate_theory.vessel_analyzer import DomeAnalyzer
from isotensoid.isotensoid_generator import IsotensoidProfile

class VesselOptimizer:
    def __init__(self, material: Material, target_burst_pressure: float, radius: float, boss_radius: float):
        self.material = material
        self.p_target = target_burst_pressure
        self.R = radius
        self.r0 = boss_radius
        
        # Initialize Geometry
        self.iso = IsotensoidProfile(R=self.R, r0=self.r0) # meters for generator
        self.alpha_eq = self.iso.winding_angle_deg
        
        # Pre-setup Puck
        self.model = Puck(material=self.material, weakening="puck")

    def _check_burst(self, layup_str):
        """Returns True if the cylinder reaches target pressure without fiber failure."""
        lam = Laminate.from_layup(layup_str, failure_model=self.model)
        pfa = ProgressiveFailureAnalysis(lam)
        
        # Load vector at target pressure
        Nx = (self.p_target * self.R) / 2
        Ny = (self.p_target * self.R)
        target_load = np.array([Nx, Ny, 0, 0, 0, 0])
        
        # Check stability at target load
        is_intact = pfa._solve_equilibrium(target_load)
        
        # Verify no fiber failure in any ply
        fibers_ok = True
        for i, ply in enumerate(lam.plies):
            sigma = lam.local_stresses[i]
            epsilon = lam.local_strains[i]
            modes = self.model.failure_check(sigma, epsilon)
            if FailureMode.FIBER_TENSION in modes or FailureMode.FIBER_COMPRESSION in modes:
                fibers_ok = False
                break
                
        return is_intact and fibers_ok

    def optimize_layup(self):
        print(f"--- Optimizing COPV Layup for {self.p_target/1e6:.1f} MPa ---")
        
        # 1. Step 1: Find Min Helical Layers (Fixed by Dome)
        # We start with 1 pair and increment
        n_hel = 1
        dome_converged = False
        while n_hel <= 20:
            t_hel = n_hel * 2 * self.material.extra.get('t', 0.2e-3)
            analyzer = DomeAnalyzer(self.iso, self.material, t_helical_equator=t_hel)
            dome_data = analyzer.analyze_dome(self.p_target)

            if max(dome_data["effort"]) <= 1.1:  # Allow slight margin
                dome_converged = True
                break
            n_hel += 1

        if not dome_converged:
            raise RuntimeError(
                f"Could not satisfy dome strength requirement within 20 helical pairs. "
                f"Last effort: {max(dome_data['effort']):.3f}"
            )

        print(f"  -> Minimum Helical Pairs for Dome: {n_hel}")

        # 2. Step 2: Find Min Hoop Layers (For Cylinder)
        n_hoop = 0
        burst_converged = False
        while n_hoop <= 30:
            # Test a basic block layup: [Hel_n / Hoop_m]_s
            test_layup = f"[{self.alpha_eq:.1f}_{n_hel*2}, 90_{n_hoop*2}]_s"
            if self._check_burst(test_layup):
                burst_converged = True
                break
            n_hoop += 1

        if not burst_converged:
            raise RuntimeError(
                f"Could not satisfy cylinder burst requirement within 30 hoop pairs. "
                f"Target pressure: {self.p_target/1e6:.1f} MPa"
            )
            
        print(f"  -> Minimum Hoop Pairs for Cylinder: {n_hoop}")

        # 3. Step 3: Optimize Stacking Order (Interleaving)
        # We have n_hel pairs and n_hoop pairs.
        # We want to distribute them as evenly as possible.
        best_layup = self._generate_interleaved_layup(n_hel, n_hoop)
        
        print(f"  -> Optimized Stacking Order: {best_layup}")
        
        # Calculate Weight (Total Thickness)
        total_t = (n_hel + n_hoop) * 2 * self.material.extra.get('t', 0.2e-3)
        print(f"  -> Total Wall Thickness: {total_t*1000:.3f} mm")
        
        return best_layup

    def _generate_interleaved_layup(self, n_hel, n_hoop):
        """
        Creates a maximally dispersed symmetric layup.
        e.g. 2 Hel, 2 Hoop -> [Hel, Hoop, Hel, Hoop]_s
        """
        hel_blocks = [[self.alpha_eq, -self.alpha_eq]] * n_hel
        hoop_blocks = [[90, 90]] * n_hoop
        
        # Mix them
        total_blocks = n_hel + n_hoop
        combined = []
        
        h_idx, p_idx = 0, 0
        for i in range(total_blocks):
            # Interleave based on ratio
            if (h_idx / n_hel if n_hel > 0 else 1) <= (p_idx / n_hoop if n_hoop > 0 else 1):
                if h_idx < n_hel:
                    combined.extend(hel_blocks[h_idx])
                    h_idx += 1
                else:
                    combined.extend(hoop_blocks[p_idx])
                    p_idx += 1
            else:
                if p_idx < n_hoop:
                    combined.extend(hoop_blocks[p_idx])
                    p_idx += 1
                else:
                    combined.extend(hel_blocks[h_idx])
                    h_idx += 1
        
        # Convert to string format for Laminate.from_layup
        angles_str = [f"{a:.1f}" for a in combined]
        return f"[{'/'.join(angles_str)}]_s"

if __name__ == "__main__":
    # Test Setup
    OD = 0.104
    BOSS = 0.060
    P_BURST = 2.5e+7 # 25 MPa Target Burst
    
    props = {
        "E1": 135e9, "E2": 10e9, "G12": 5e9, "v12": 0.3,
        "Xt": 2200e6, "Xc": 1500e6, "Yt": 50e6, "Yc": 150e6, "S12": 70e6,
        "t": 0.2e-3
    }
    mat = Material(**props)
    
    opt = VesselOptimizer(mat, P_BURST, OD/2, BOSS/2)
    best = opt.optimize_layup()
