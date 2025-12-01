import numpy as np
import copy
import matplotlib.pyplot as plt
from classical_laminate_theory.failuremodels import FailureMode


class ProgressiveFailureAnalysis:
    def __init__(self, laminate):
        """
        Orchestrates progressive failure analysis on a Laminate object.
        Uses the 'update_stiffness' method refactored in Laminate to
        handle load redistribution after ply failure.
        """
        # Work on a copy so we don't permanently destroy the user's original object
        self.laminate = copy.deepcopy(laminate)

        # Storage for plotting
        self.history = {
            "load_factor": [],
            "strain": [],
            "stress": [],
            "damage_state": []  # Tracks how many plies have failed
        }

    def _degrade_ply(self, ply, failure_modes):
        """
        Apply stiffness knockdowns based on FailureMode Enums.
        """
        MATRIX_KNOCKDOWN = 0.10
        FIBER_KNOCKDOWN = 0.01
        MIN_STIFFNESS = 1e6

        properties_changed = False

        # Define groupings for readability
        FIBER_MODES = {FailureMode.FIBER_TENSION, FailureMode.FIBER_COMPRESSION}
        MATRIX_MODES = {FailureMode.MATRIX_TENSION, FailureMode.MATRIX_COMPRESSION, FailureMode.SHEAR}

        for mode in failure_modes:

            # --- CASE A: FIBER FAILURE ---
            if mode in FIBER_MODES:
                if ply.E1 > MIN_STIFFNESS:
                    ply.E1 *= FIBER_KNOCKDOWN
                    properties_changed = True

            # --- CASE B: MATRIX FAILURE ---
            elif mode in MATRIX_MODES:
                if ply.E2 > MIN_STIFFNESS:
                    ply.E2 *= MATRIX_KNOCKDOWN
                    ply.G12 *= MATRIX_KNOCKDOWN
                    ply.v12 = 0
                    properties_changed = True

            # --- CASE C: GENERIC/TSAI-HILL ---
            elif mode == FailureMode.GENERAL_FAILURE:
                # Same heuristic as before: Matrix first, then Fiber
                if ply.E2 > MIN_STIFFNESS:
                    ply.E2 *= MATRIX_KNOCKDOWN
                    ply.G12 *= MATRIX_KNOCKDOWN
                    ply.v12 = 0
                    properties_changed = True
                elif ply.E1 > MIN_STIFFNESS:
                    ply.E1 *= FIBER_KNOCKDOWN
                    properties_changed = True

        return properties_changed

    def run_simulation(self, max_load, steps=100, debug_interval=1):
        print(f"--- Starting DEBUG PFA Simulation: {steps} steps ---")
        load_increments = np.linspace(0, 1.0, steps)

        simulation_active = True
        initial_stiffness = self.laminate.A[0, 0]

        for step_idx, factor in enumerate(load_increments):
            if not simulation_active:
                break

            current_load = max_load * factor
            self.laminate.apply_load(current_load)

            # --- DEBUG HEADER ---
            if step_idx % debug_interval == 0:
                print(f"\n=== LOAD STEP {factor:.3f} ===")
                print(f"  Global Load Nx: {current_load[0]:.2e}")
                print(
                    f"  Laminate A11:   {self.laminate.A[0, 0]:.2e} (Retention: {self.laminate.A[0, 0] / initial_stiffness:.1%})")
                print(f"  Global Strain:  eps_x={self.laminate.global_strains[0, 0]:.6f}")

            stable = False
            iterations = 0

            while not stable and iterations < 10:
                stable = True
                iterations += 1

                # Iterate over plies
                for i, ply in enumerate(self.laminate.plies):
                    # Skip dead plies
                    if ply.E1 < 1e6 and ply.E2 < 1e6:
                        continue

                    sigma = self.laminate.local_stresses[i]
                    epsilon = self.laminate.local_strains[i]

                    # --- UPDATED LOGIC HERE ---
                    # The failure model now returns a LIST of enums (or empty list)
                    modes = ply.failure_model.failure_check(sigma, epsilon)

                    # If the list is not empty, failure occurred
                    if modes:
                        if step_idx % debug_interval == 0:
                            print(f"      !!! FAILURE DETECTED in Ply {i} !!!")
                            print(f"      Modes: {[m.name for m in modes]}")
                            print(f"      State: Sig1={sigma[0]:.2e}, Sig2={sigma[1]:.2e}")

                        changed = self._degrade_ply(ply, modes)

                        if changed:
                            stable = False
                            if step_idx % debug_interval == 0:
                                print(f"      -> Properties degraded. Restarting equilibrium loop.")

                if not stable:
                    is_intact = self.laminate.update_stiffness()

                    # Stiffness Retention Stop
                    if self.laminate.A[0, 0] / initial_stiffness < 0.10:
                        print(f"  -> STOP: Stiffness dropped below 10%. Rupture.")
                        simulation_active = False
                        break

                    if not is_intact:
                        print(f"  -> STOP: Matrix Singular.")
                        simulation_active = False
                        break

                    self.laminate.get_stress_strain()

            if simulation_active:
                self.history["load_factor"].append(factor)
                self.history["strain"].append(self.laminate.global_strains[0, 0])
                t = self.laminate.lamina_boundaries[-1] - self.laminate.lamina_boundaries[0]
                self.history["stress"].append(current_load[0] / t)

        print("--- Simulation Complete ---")

    def plot_curve(self):
        """Visualizes the Stress-Strain response."""
        strain_pct = np.array(self.history["strain"]) * 100
        stress_mpa = np.array(self.history["stress"]) / 1e6

        plt.figure(figsize=(10, 6))
        plt.plot(strain_pct, stress_mpa, linewidth=2, color='navy', label='Laminate Response')

        # Highlight drops (failure points)
        # We look for negative slopes or sudden changes

        plt.title("Progressive Failure Analysis", fontsize=14, fontweight='bold')
        plt.xlabel(r"Global Strain $\epsilon_{xx}$ (%)", fontsize=12)
        plt.ylabel(r"Global Stress $\sigma_{xx}$ (MPa)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Imports from your package structure
    from classical_laminate_theory.structures import Laminate
    from classical_laminate_theory.failuremodels import TsaiHill, Hashin, MaxStress, Puck
    from classical_laminate_theory.progressive_failure_analysis import ProgressiveFailureAnalysis

    # --- 1. Define Common Material Properties ---
    # Elastic Constants
    E1 = 140e9
    E2 = 10e9
    G12 = 5e9
    v12 = 0.3
    t_ply = 0.15e-3

    # Strengths (Carbon/Epoxy)
    Xt = 2500e6
    Xc = 2000e6
    Yt = 50e6
    Yc = 150e6
    S12 = 80e6

    # Layup
    layup_str = "[0/90/45/-45]_s"

    # Loading (1 MN/m Tension)
    max_load = np.array([1e6, 0, 0, 0, 0, 0])
    steps = 200

    # --- 2. Instantiate Failure Models ---
    models = {}

    # A. Max Stress
    models["Max Stress"] = MaxStress(Xt, Xc, Yt, Yc, S12)

    # B. Tsai-Hill (Uses Xt/Yt for tension-dominated runs)
    models["Tsai-Hill"] = TsaiHill(X11=Xt, X22=Yt, S12=S12)

    # C. Hashin
    models["Hashin"] = Hashin(Xt, Xc, Yt, Yc, S12)

    # D. Puck (Requires E1, v12 for magnification)
    models["Puck"] = Puck(Xt, Xc, Yt, Yc, S12, E1=E1, v12=v12)

    # --- 3. Run Simulations Loop ---
    results = {}

    plt.figure(figsize=(10, 7))
    colors = {'Max Stress': 'gray', 'Tsai-Hill': 'orange', 'Hashin': 'blue', 'Puck': 'green'}
    styles = {'Max Stress': '--', 'Tsai-Hill': '-.', 'Hashin': '-', 'Puck': '-'}

    print(f"Comparing Failure Models on Layup: {layup_str}...\n")

    for name, model in models.items():
        print(f"--- Running {name} ---")

        # Build Laminate with this specific model
        lam = Laminate.from_layup(
            layup_string=layup_str,
            E1=E1, E2=E2, G12=G12, v12=v12, t=t_ply,
            failure_model=model
        )

        # Run PFA
        # Note: Suppress debug printouts with high debug_interval
        sim = ProgressiveFailureAnalysis(lam)
        sim.run_simulation(max_load, steps=steps, debug_interval=10)

        # Store Data
        strain_pct = np.array(sim.history["strain"]) * 100
        stress_mpa = np.array(sim.history["stress"]) / 1e6

        # Plot immediately
        plt.plot(strain_pct, stress_mpa,
                 label=name, color=colors[name], linestyle=styles[name], linewidth=2)

    # --- 4. Finalize Plot ---
    plt.title(f"Failure Model Comparison\nLayup: {layup_str}", fontsize=14, fontweight='bold')
    plt.xlabel(r"Global Strain $\epsilon_{xx}$ (%)", fontsize=12)
    plt.ylabel(r"Global Stress $\sigma_{xx}$ (MPa)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()