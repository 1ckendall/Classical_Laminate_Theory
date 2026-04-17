import numpy as np
import matplotlib.pyplot as plt
from classical_laminate_theory.failuremodels import FailureMode


class ProgressiveFailureAnalysis:
    def __init__(self, laminate):
        """
        Orchestrates progressive failure analysis on a Laminate object.
        Non-destructive: Resets damage on the laminate before starting.
        """
        self.laminate = laminate
        self.laminate.reset_damage()  # Ensure we start fresh
        self.initial_stiffness = self.laminate.A[0, 0]

        # Storage for plotting (remains for backward compatibility with plot_curve)
        self.history = {
            "load_factor": [],
            "strain": [],
            "stress": [],
            "damage_state": [],
            "failed_plies_details": [],
        }

        # Track unique indices of plies that have failed at least once
        self.failed_plies = set()

    def _degrade_ply(self, ply, failure_modes):
        """
        Apply stiffness knockdowns to damage factors.
        """
        # Physical constants for knockdowns
        MATRIX_KNOCKDOWN = 0.10
        FIBER_KNOCKDOWN = 0.01
        MIN_FACTOR = 1e-6

        changed = False

        FIBER_MODES = {FailureMode.FIBER_TENSION, FailureMode.FIBER_COMPRESSION}
        MATRIX_MODES = {
            FailureMode.MATRIX_TENSION,
            FailureMode.MATRIX_COMPRESSION,
            FailureMode.SHEAR,
        }

        for mode in failure_modes:
            if mode in FIBER_MODES:
                if ply.d1 > MIN_FACTOR:
                    ply.d1 = FIBER_KNOCKDOWN
                    changed = True
            elif mode in MATRIX_MODES:
                if ply.d2 > MIN_FACTOR:
                    ply.d2 = MATRIX_KNOCKDOWN
                    ply.d66 = MATRIX_KNOCKDOWN
                    ply.dv12 = 0.0
                    changed = True
            elif mode == FailureMode.GENERAL_FAILURE:
                # Tsai-Hill approach: degrade matrix first, then fiber if already failed
                if ply.d2 > MIN_FACTOR:
                    ply.d2 = MATRIX_KNOCKDOWN
                    ply.d66 = MATRIX_KNOCKDOWN
                    ply.dv12 = 0.0
                    changed = True
                elif ply.d1 > MIN_FACTOR:
                    ply.d1 = FIBER_KNOCKDOWN
                    changed = True
        
        if changed:
            ply.failed = True
            
        return changed

    def _solve_equilibrium(self, current_load):
        """
        Internal Helper: Applies a specific load vector and iterates
        stiffness degradation until stable.
        """
        self.laminate.apply_load(current_load)

        stable = False
        iterations = 0
        max_iter = 10
        is_ruptured = False

        while not stable and iterations < max_iter:
            stable = True
            iterations += 1

            for i, ply in enumerate(self.laminate.plies):
                # Skip if already fully degraded
                if ply.d1 <= 1e-6 and ply.d2 <= 1e-6:
                    continue

                sigma = self.laminate.local_stresses[i]
                epsilon = self.laminate.local_strains[i]

                modes = ply.failure_model.failure_check(sigma, epsilon)

                if modes:
                    self.failed_plies.add(i)
                    if self._degrade_ply(ply, modes):
                        stable = False

            if not stable:
                is_intact = self.laminate.update_stiffness()

                # Rupture Criteria
                if len(self.failed_plies) == len(self.laminate.plies):
                    is_ruptured = True
                    break
                if self.laminate.A[0, 0] / self.initial_stiffness < 0.10:
                    is_ruptured = True
                    break
                if not is_intact:
                    is_ruptured = True
                    break

                self.laminate.get_stress_strain()

        return not is_ruptured

    def _record_history(self, current_load):
        """Append current state to history dictionary."""
        t = self.laminate.lamina_boundaries[-1] - self.laminate.lamina_boundaries[0]
        sigma_x = current_load[0] / t

        self.history["load_factor"].append(current_load[0])
        self.history["strain"].append(self.laminate.midplane_strains[0])
        self.history["stress"].append(sigma_x)
        self.history["damage_state"].append(len(self.failed_plies))
        self.history["failed_plies_details"].append(sorted(list(self.failed_plies)))

    def simulate(self, load_path):
        """
        Option 3: Generator-based simulation.
        Yields the laminate state at each stable equilibrium point.
        """
        self.laminate.reset_damage()
        self.failed_plies.clear()
        self.history = {k: [] for k in self.history} # Reset history for this run

        for load in load_path:
            is_intact = self._solve_equilibrium(load)
            
            # Record state
            self._record_history(load)
            
            # Yield results for real-time processing
            yield {
                "load": load,
                "is_intact": is_intact,
                "failed_count": len(self.failed_plies),
                "stiffness_ratio": self.laminate.A[0,0] / self.initial_stiffness
            }

            if not is_intact:
                break

    def run_simulation(self, max_load, steps=100):
        """Legacy wrapper using the simulate generator."""
        print(f"--- Starting Fixed-Load PFA: {steps} steps ---")
        load_path = [max_load * f for f in np.linspace(0, 1.0, steps)]
        
        for state in self.simulate(load_path):
            if not state["is_intact"]:
                print(f"  -> Structure failed at Load: {state['load'][0]:.2f}")
                break
        
        print("--- Simulation Complete ---")

    def run_until_failure(self, load_direction, step_size, max_steps=1000):
        """Legacy wrapper using a generated load path."""
        print(f"--- Starting 'Run Until Failure' PFA ---")
        
        def load_generator():
            current_load = np.zeros(6)
            load_delta = np.array(load_direction) * step_size
            for _ in range(max_steps):
                current_load = current_load + load_delta
                yield current_load

        last_step = 0
        last_load = 0
        for i, state in enumerate(self.simulate(load_generator())):
            last_step = i
            last_load = state["load"][0]
            if not state["is_intact"]:
                print(f"  -> Global Failure Detected at Step {i}")
                print(f"  -> Max Load Reached: {last_load:.2f} N/m")
                break
        else:
             print("  -> Warning: Max steps reached without total failure.")

        print("--- Simulation Complete ---")

    def plot_curve(self, title_suffix=""):
        """Visualizes the Stress-Strain response from recorded history."""
        strain_pct = np.array(self.history["strain"]) * 100
        stress_mpa = np.array(self.history["stress"]) / 1e6
        failed_details = self.history["failed_plies_details"]

        plt.plot(strain_pct, stress_mpa, linewidth=2, label=title_suffix)

        previous_failures = set()
        for i, current_failures_list in enumerate(failed_details):
            current_failures = set(current_failures_list)
            new_failures = current_failures - previous_failures

            if new_failures:
                x_val, y_val = strain_pct[i], stress_mpa[i]
                ply_nums = [idx + 1 for idx in sorted(list(new_failures))]
                label_text = f"Ply {ply_nums} Fail"

                plt.plot(x_val, y_val, 'ro', markersize=4)
                offset_y = 20 if (i % 2 == 0) else -30
                plt.annotate(
                    label_text, xy=(x_val, y_val),
                    xytext=(x_val + 0.05, y_val + offset_y),
                    textcoords='data',
                    arrowprops=dict(facecolor='red', arrowstyle='->', alpha=0.6),
                    fontsize=8, color='darkred'
                )

            previous_failures = current_failures

        plt.xlabel(r"Global Strain $\epsilon_{xx}$ (%)", fontsize=12)
        plt.ylabel(r"Global Stress $\sigma_{xx}$ (MPa)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)


def main():
    from classical_laminate_theory.structures import Laminate, Material
    from classical_laminate_theory.failuremodels import TsaiHill, Hashin, MaxStress, Puck

    E1, E2, G12, v12, t_ply = 140e9, 10e9, 5e9, 0.3, 0.15e-3
    Xt, Xc, Yt, Yc, S12 = 2500e6, 2000e6, 50e6, 150e6, 80e6
    layup_str = "[0/90/45/-45]_s"

    mat = Material(E1=E1, E2=E2, G12=G12, v12=v12, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, t=t_ply)

    models = {
        "Max Stress": MaxStress(material=mat),
        "Tsai-Hill": TsaiHill(material=mat),
        "Hashin": Hashin(material=mat),
        "Puck": Puck(material=mat),
    }

    direction = np.array([1.0, 0, 0, 0, 0, 0])
    step_size = 1e3

    plt.figure(figsize=(12, 8))
    for name, model in models.items():
        lam = Laminate.from_layup(layup_str, material=mat, failure_model=model)
        sim = ProgressiveFailureAnalysis(lam)
        sim.run_until_failure(direction, step_size=step_size, max_steps=3000)
        sim.plot_curve(title_suffix=name)

    plt.title(f"Progressive Failure Analysis Comparison\nLayup: {layup_str}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
