import numpy as np
import copy
import matplotlib.pyplot as plt
from classical_laminate_theory.failuremodels import FailureMode


class ProgressiveFailureAnalysis:
    def __init__(self, laminate):
        """
        Orchestrates progressive failure analysis on a Laminate object.
        """
        # Work on a copy so we don't permanently destroy the user's original object
        self.laminate = copy.deepcopy(laminate)
        self.initial_stiffness = self.laminate.A[0, 0]

        # Storage for plotting
        self.history = {
            "load_factor": [],  # Total load magnitude
            "strain": [],
            "stress": [],
            "damage_state": []  # Tracks how many plies have failed
        }

        # Track unique indices of plies that have failed at least once
        self.failed_plies = set()

    def _degrade_ply(self, ply, failure_modes):
        """
        Apply stiffness knockdowns based on FailureMode Enums.
        Returns True if properties were changed.
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
                if ply.E2 > MIN_STIFFNESS:
                    ply.E2 *= MATRIX_KNOCKDOWN
                    ply.G12 *= MATRIX_KNOCKDOWN
                    ply.v12 = 0
                    properties_changed = True
                elif ply.E1 > MIN_STIFFNESS:
                    ply.E1 *= FIBER_KNOCKDOWN
                    properties_changed = True

        return properties_changed

    def _solve_equilibrium(self, current_load):
        """
        Internal Helper: Applies a specific load vector and iterates
        stiffness degradation until the laminate is stable or ruptured.

        Returns:
            (bool) is_stable: True if equilibrium reached, False if ruptured/singular
        """
        self.laminate.apply_load(current_load)

        stable = False
        iterations = 0
        max_iter = 10

        is_ruptured = False

        while not stable and iterations < max_iter:
            stable = True
            iterations += 1

            # 1. Check every ply for failure at current stress state
            for i, ply in enumerate(self.laminate.plies):
                # Skip dead plies to save time
                if ply.E1 < 1e6 and ply.E2 < 1e6:
                    continue

                sigma = self.laminate.local_stresses[i]
                epsilon = self.laminate.local_strains[i]

                modes = ply.failure_model.failure_check(sigma, epsilon)

                if modes:
                    self.failed_plies.add(i)  # Mark ply as having failed
                    changed = self._degrade_ply(ply, modes)
                    if changed:
                        stable = False  # Properties changed, must re-calculate stresses

            # 2. If properties changed, update ABD matrix and Global Strains
            if not stable:
                is_intact = self.laminate.update_stiffness()

                # CRITERIA A: All Plies Failed (New Criterion)
                if len(self.failed_plies) == len(self.laminate.plies):
                    print(f"  -> STOP: All {len(self.laminate.plies)} plies have experienced failure.")
                    is_ruptured = True
                    break

                # CRITERIA B: Global Stiffness Drop (Rupture)
                # If stiffness drops below 10% of initial, we consider the part broken.
                if self.laminate.A[0, 0] / self.initial_stiffness < 0.10:
                    is_ruptured = True
                    break

                # CRITERIA C: Singularity
                if not is_intact:
                    is_ruptured = True
                    break

                # Re-calculate stresses with new stiffness for next iteration
                self.laminate.get_stress_strain()

        return not is_ruptured

    def _record_history(self, current_load):
        """Helper to append current state to history."""
        # Calculate scalar stress (assuming uniaxial X-load for plotting simplicity)
        t = self.laminate.lamina_boundaries[-1] - self.laminate.lamina_boundaries[0]
        sigma_x = current_load[0] / t

        # Load factor usually implies ratio of current to max,
        # but here we just store magnitude of Nxx
        self.history["load_factor"].append(current_load[0])
        self.history["strain"].append(self.laminate.global_strains[0, 0])
        self.history["stress"].append(sigma_x)

    def run_simulation(self, max_load, steps=100):
        """
        Original method: Runs from 0 to max_load in fixed steps.
        """
        print(f"--- Starting Fixed-Load PFA: {steps} steps ---")
        load_increments = np.linspace(0, 1.0, steps)

        for factor in load_increments:
            current_load = max_load * factor

            # Attempt to solve equilibrium
            is_intact = self._solve_equilibrium(current_load)

            if is_intact:
                self._record_history(current_load)
            else:
                print(f"  -> Structure failed at Load Factor: {factor:.2f}")
                break

        print("--- Simulation Complete ---")

    def run_until_failure(self, load_direction, step_size, max_steps=1000):
        """
        New Method: Increases load incrementally until failure occurs.

        Args:
            load_direction (np.array): A 1x6 vector indicating direction (e.g., [1,0,0,0,0,0])
            step_size (float): Magnitude of load increase per step (e.g., 1000 N/m)
            max_steps (int): Safety break to prevent infinite loops.
        """
        print(f"--- Starting 'Run Until Failure' PFA ---")

        current_load = np.zeros(6)

        # Normalize direction just in case, or treat inputs as raw deltas
        # Here we assume load_direction is a unit vector or the specific delta vector
        load_delta = np.array(load_direction) * step_size

        for step in range(max_steps):
            # Increment Load
            current_load = current_load + load_delta

            # Solve Equilibrium
            is_intact = self._solve_equilibrium(current_load)

            # Record Data
            # We record even if it failed this step, to capture the peak/drop
            self._record_history(current_load)

            if not is_intact:
                print(f"  -> Global Failure Detected at Step {step}")
                print(f"  -> Max Load Reached: {current_load[0]:.2f} N/m")
                break
        else:
            print("  -> Warning: Max steps reached without total failure.")

        print("--- Simulation Complete ---")

    def plot_curve(self, title_suffix=""):
        """Visualizes the Stress-Strain response."""
        strain_pct = np.array(self.history["strain"]) * 100
        stress_mpa = np.array(self.history["stress"]) / 1e6

        plt.plot(strain_pct, stress_mpa, linewidth=2, label=title_suffix)
        plt.xlabel(r"Global Strain $\epsilon_{xx}$ (%)", fontsize=12)
        plt.ylabel(r"Global Stress $\sigma_{xx}$ (MPa)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)


# --- EXAMPLE USAGE IF RUN DIRECTLY ---
if __name__ == "__main__":
    from classical_laminate_theory.structures import Laminate
    from classical_laminate_theory.failuremodels import TsaiHill, Hashin, MaxStress, Puck

    # 1. Setup Material
    E1, E2, G12, v12, t_ply = 140e9, 10e9, 5e9, 0.3, 0.15e-3
    Xt, Xc, Yt, Yc, S12 = 2500e6, 2000e6, 50e6, 150e6, 80e6

    layup_str = "[0/90/45/-45]_s"

    # 2. Instantiate Failure Models
    models = {}
    models["Max Stress"] = MaxStress(Xt, Xc, Yt, Yc, S12)
    models["Tsai-Hill"] = TsaiHill(X11=Xt, X22=Yt, S12=S12)
    models["Hashin"] = Hashin(Xt, Xc, Yt, Yc, S12)
    models["Puck"] = Puck(Xt, Xc, Yt, Yc, S12, E1=E1, v12=v12)

    # 3. Setup Simulation Parameters
    direction = np.array([1.0, 0, 0, 0, 0, 0])  # Uniaxial Tension
    step_size = 1e3

    plt.figure(figsize=(10, 7))
    print(f"Comparing Failure Models on Layup: {layup_str}...\n")

    # 4. Run Loop
    for name, model in models.items():
        print(f"--- Running {name} ---")
        lam = Laminate.from_layup(layup_str, E1, E2, G12, v12, t_ply, model)

        sim = ProgressiveFailureAnalysis(lam)
        sim.run_until_failure(direction, step_size=step_size, max_steps=3000)

        sim.plot_curve(title_suffix=name)

    # 5. Finalize Plot
    plt.title(f"Progressive Failure Analysis Comparison\nLayup: {layup_str}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()