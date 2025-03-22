import numpy as np
from .failuremodels import FailureModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Annotated


class Lamina:
    """A single composite lamina. Base class for laminae."""

    def __init__(
        self,
        angle: float,
        E1: float,
        E2: float,
        G12: float,
        v12: float,
        t: float,
        failure_model: FailureModel,
    ):
        """
        Initialise a composite lamina with given properties.
        Args:
            angle: Angle of fibres (deg).
            E1: E-modulus along the fibres (Pa)
            E2: E-modulus across the fibres (Pa)
            G12: Shear modulus (Pa)
            v12: Major Poisson ratio
            t: Ply Thickness (m)
        """
        self.angle = np.radians(angle)  # angle of fibres
        self.m = np.cos(self.angle)
        self.n = np.sin(self.angle)
        self.E1 = E1  # E along fibres
        self.E2 = E2  # E across fibres
        self.G12 = G12  # Shear modulus
        self.v12 = v12  # Major Poisson
        self.v21 = self.v12 * self.E2 / self.E1  # Minor Poisson
        self.t = t  # Thickness
        self.failed = False

        # Initialise reduced stiffness matrix Q parallel to the fibres of the lamina
        self.Q = 1 - (self.v12 * self.v21)
        self.Q11 = self.E1 / self.Q
        self.Q12 = self.v12 * self.E2 / self.Q
        self.Q22 = self.E2 / self.Q
        self.Q66 = self.G12
        # Disable Black reformatting for the next line
        # fmt: off
        self.Qmat = np.array(
            [[self.Q11, self.Q12,        0],
             [self.Q12, self.Q22,        0],
             [0,        0,        self.Q66]]
        )
        # fmt: on

        # Initialise reduced stiffness matrix Qbar matrix parallel to the global coordinate system
        self.Qxx = (
            self.m**4 * self.Q11
            + self.n**4 * self.Q22
            + 2 * self.m**2 * self.n**2 * self.Q12
            + 4 * self.m**2 * self.n**2 * self.Q66
        )
        self.Qyy = (
            self.n**4 * self.Q11
            + self.m**4 * self.Q22
            + 2 * self.m**2 * self.n**2 * self.Q12
            + 4 * self.m**2 * self.n**2 * self.Q66
        )
        self.Qxy = (
            self.m**2 * self.n**2 * self.Q11
            + self.m**2 * self.n**2 * self.Q22
            + (self.m**4 + self.n**4) * self.Q12
            - 4 * self.m**2 * self.n**2 * self.Q66
        )
        self.Qxs = (
            self.m**3 * self.n * self.Q11
            - self.m * self.n**3 * self.Q22
            - self.m * self.n * (self.m**2 - self.n**2) * self.Q12
            - 2 * self.m * self.n * (self.m**2 - self.n**2) * self.Q66
        )
        self.Qys = (
            self.n**3 * self.m * self.Q11
            - self.n * self.m**3 * self.Q22
            + self.m * self.n * (self.m**2 - self.n**2) * self.Q12
            + 2 * self.m * self.n * (self.m**2 - self.n**2) * self.Q66
        )
        self.Qss = (
            self.m**2 * self.n**2 * self.Q11
            + self.m**2 * self.n**2 * self.Q22
            - 2 * self.m**2 * self.n**2 * self.Q12
            + (self.m**2 - self.n**2) ** 2 * self.Q66
        )
        self.Qbarmat = np.array(
            [
                [self.Qxx, self.Qxy, self.Qxs],
                [self.Qxy, self.Qyy, self.Qys],
                [self.Qxs, self.Qys, self.Qss],
            ]
        )
        # Transformation matrix for stresses and strains
        self.global_local_transform = np.array(
            [
                [self.m**2, self.n**2, 2 * self.m * self.n],
                [self.n**2, self.m**2, -2 * self.m * self.n],
                [-self.m * self.n, self.m * self.n, self.m**2 - self.n**2],
            ]
        )


class Laminate:
    def __init__(
        self,
        plies: tuple[Lamina, ...],
        load: Annotated[np.ndarray, (6,)] = np.array([0, 0, 0, 0, 0, 0]),
    ):
        """
        A composite laminate comprising a layup of plies and under some loading
        Args:
            plies:
            load: A 1-by 6 numpy array of [Nx, Ny, Nxy, Mx, My, Mxy]
        """
        # Ply properties
        self.plies = plies
        self.n = len(plies)
        self.lamina_boundaries = np.zeros(len(self.plies) + 1)
        self.lamina_midplanes = np.zeros(len(self.plies))
        self.get_z_positions()
        self.Qbar_matrices = np.array([ply.Qbarmat for ply in self.plies])
        self.T_matrices = np.array([ply.global_local_transform for ply in self.plies])

        # ABD properties
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.ABD = np.zeros((6, 6))
        self.abd = np.zeros((6, 6))
        self.get_abd_matrices()

        # Loading properties
        self.load = load
        self.global_stresses = np.zeros([len(plies), 3])
        self.global_strains = np.zeros([len(plies), 3])
        self.local_stresses = np.zeros([len(plies), 3])
        self.local_strains = np.zeros([len(plies), 3])
        self.get_stress_strain()

    def get_z_positions(self):
        """Calculate the boundary and midplane z-positions for each ply."""
        ply_thickness = np.array([ply.t for ply in self.plies])
        z = np.hstack(([0], np.cumsum(ply_thickness)))
        midplane_shift = 0.5 * (z[0] + z[-1])
        self.lamina_boundaries = z - midplane_shift
        self.lamina_midplanes = self.lamina_boundaries[:-1] + 0.5 * ply_thickness

    def get_abd_matrices(self):
        """Efficiently compute A, B, and D matrices and assemble ABD."""
        z_bounds = self.lamina_boundaries
        for i, ply in enumerate(self.plies):
            bottom_z, top_z = z_bounds[i], z_bounds[i + 1]
            dz = top_z - bottom_z
            dz2_half = (top_z**2 - bottom_z**2) / 2
            dz3_third = (top_z**3 - bottom_z**3) / 3

            self.A += ply.Qbarmat * dz
            self.B += ply.Qbarmat * dz2_half
            self.D += ply.Qbarmat * dz3_third

        self.ABD = np.block([[self.A, self.B], [self.B.T, self.D]])
        det_abd = np.linalg.det(self.ABD)
        if det_abd != 0:
            self.abd = np.linalg.inv(self.ABD)
        else:
            raise ValueError("ABD matrix is singular; check input properties.")

    def get_stress_strain(self):
        """Compute laminate strains and stresses for each ply in the global and local coordinate systems."""
        strain_midplane = np.linalg.solve(self.ABD, self.load)
        # Compute global strains and stresses
        self.global_strains = (
            strain_midplane[:3] + self.lamina_midplanes[:, None] * strain_midplane[3:]
        )
        self.global_stresses = np.einsum(
            "nij,nj->ni", self.Qbar_matrices, self.global_strains
        )
        # Compute local strains and stresses
        self.local_strains = np.einsum(
            "nij,nj->ni", self.T_matrices, self.global_strains
        )
        self.local_stresses = np.einsum(
            "nij,nj->ni", self.T_matrices, self.global_stresses
        )

    def apply_load(self, load):
        """Update the applied load and recompute stress/strain, with error handling."""
        try:
            self.load = load
            self.get_stress_strain()
        except ValueError as e:
            raise ValueError("Invalid load values or singular ABD matrix.") from e

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    def _plot_distribution(
        self,
        values,
        labels,
        title,
    ):
        """
        General function to plot stress/strain distributions with fiber angle visualization.

        Args:
            values: The values to plot (stresses or strains)
            labels: Labels for each axis
            title: Plot title
        """
        sns.set_style("darkgrid")
        sns.set_palette("deep")

        z_bounds = self.lamina_boundaries
        colors = sns.color_palette("deep", 3)

        # Create the main figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=100)
        fig.subplots_adjust(
            right=0.82  # Adjusted to make more space for the right-hand labels
        )

        # Plot the main data in the subplots
        for i, ax in enumerate(axes):
            for j in range(len(self.plies)):
                z_lower, z_upper = z_bounds[j], z_bounds[j + 1]
                val_lower, val_upper = values[j, i], values[j, i]

                # Plot the vertical line for the ply
                ax.plot(
                    [val_lower, val_upper],
                    [z_lower, z_upper],
                    color=colors[i],
                    linestyle="-",
                    linewidth=2,
                )

                # Add horizontal lines at ply breaks (now solid)
                if j < len(self.plies) - 1:
                    ax.plot(
                        [val_upper, values[j + 1, i]],
                        [z_upper, z_upper],
                        color=colors[i],
                        linestyle="-",
                        linewidth=2,
                    )
                    # Add solid horizontal line at the break
                    ax.axhline(
                        y=z_upper,
                        color="gray",
                        linestyle="-",  # Changed to solid
                        alpha=0.3,
                        linewidth=0.8,
                    )

            ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=1.2)
            ax.set_xlabel(labels[i], fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)

        # Set y-label only for the first subplot
        axes[0].set_ylabel("Z position (m)", fontsize=12, fontweight="bold")

        # Add ply angles to the far right of the figure
        for j in range(len(self.plies)):
            z_lower, z_upper = z_bounds[j], z_bounds[j + 1]
            mid_z = float((z_lower + z_upper) / 2)
            ply_angle = np.degrees(self.plies[j].angle)

            # Place ply angle labels on the far right of the figure
            axes[0].annotate(
                text=f"{ply_angle:.2f}°",
                xy=(3.4, mid_z),  # Position just outside the right edge of the plot
                xycoords=(
                    "axes fraction",
                    "data",
                ),  # Use axes fraction for x, data for y
                fontsize=10,
                color="k",
                ha="left",
                va="center",
                xytext=(10, 0),  # Offset in points
                textcoords="offset points",
            )

        # Add a label for the ply angles on the far right of the figure
        axes[2].text(
            x=3.62,  # Position just outside the right edge of the plot
            y=float((z_bounds[0] + z_bounds[-1]) / 2),
            s="Ply angle (degrees)",
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
            transform=axes[
                0
            ].get_yaxis_transform(),  # Use y-axis transform for positioning
        )

        # Add heavy horizontal lines at the top and bottom of the laminate
        for ax in axes:
            ax.axhline(
                y=z_bounds[0], color="gray", linestyle="-", linewidth=2, alpha=0.3
            )
            ax.axhline(
                y=z_bounds[-1], color="gray", linestyle="-", linewidth=2, alpha=0.3
            )
            # When you hatch to the edge it will expand the y axis (for some reason), so store the current bounds and
            # reset them after running the hatching operations
            ax_bounds = ax.get_ylim()
            ax.axhspan(
                ymin=ax_bounds[0],  # Minimum y-limit of the plot
                ymax=z_bounds[0],  # Minimum z bound
                color="gray",
                alpha=0.1,  # Light transparency
                hatch="///",  # Hatching pattern
                linewidth=0.5,
            )
            # Hatching above the maximum z bound
            ax.axhspan(
                ymin=z_bounds[-1],  # Maximum z bound
                ymax=ax_bounds[1],  # Maximum y-limit of the plot
                color="gray",
                alpha=0.1,  # Light transparency
                hatch="///",  # Hatching pattern
                linewidth=0.5,
            )
            ax.set_ylim(ax_bounds)

        fig.suptitle(title, fontweight="bold")
        plt.show()

    @property
    def global_stress_graph(self):
        """Plot stress distribution through laminate thickness."""
        return self._plot_distribution(
            self.global_stresses * 1e-6,
            [r"$\sigma_{xx}$ (MPa)", r"$\sigma_{yy}$ (MPa)", r"$\tau_{xy}$ (MPa)"],
            "Global Stress Distribution in the Laminate",
        )

    @property
    def global_strain_graph(self):
        """Plot strain distribution through laminate thickness."""
        return self._plot_distribution(
            self.global_strains,
            [r"$\varepsilon_{xx}$", r"$\varepsilon_{yy}$", r"$\gamma_{xy}$"],
            "Global Strain Distribution in the Laminate",
        )

    @property
    def local_stress_graph(self):
        """Plot stress distribution through laminate thickness."""
        return self._plot_distribution(
            self.local_stresses * 1e-6,
            [r"$\sigma_{11}$ (MPa)", r"$\sigma_{22}$ (MPa)", r"$\tau_{12}$ (MPa)"],
            "Local Stress Distribution in the Laminate",
        )

    @property
    def local_strain_graph(self):
        """Plot strain distribution through laminate thickness."""
        return self._plot_distribution(
            self.local_strains,
            [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$\gamma_{12}$"],
            "Local Strain Distribution in the Laminate",
        )
