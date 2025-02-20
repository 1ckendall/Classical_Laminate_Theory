import numpy as np


class Lamina:
    """A single composite lamina. Base class for laminae."""

    def __init__(self, th, E1, E2, G12, v12, t):
        """
        Initialise a composite lamina with given properties.
        Args:
            th: Angle of fibres (deg).
            E1: E-modulus along the fibres (Pa)
            E2: E-modulus across the fibres (Pa)
            G12: Shear modulus (Pa)
            v12: Major Poisson ratio
            t: Ply Thickness (m)
        """

        self.th = np.radians(th)  # angle of fibres (degrees, converted to radians for internal maths)
        self.m = np.cos(self.th)
        self.n = np.sin(self.th)
        self.E1 = E1  # E along fibres
        self.E2 = E2  # E across fibres
        self.G12 = G12  # Shear modulus
        self.v12 = v12  # Major Poisson
        self.v21 = self.v12 * self.E2 / self.E1  # Minor Poisson
        self.t = t  # Thickness

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

        # Initialise an empty failure mode
        self.failure_mode = None


class Laminate:
    def __init__(
        self,
        plies: tuple[Lamina, ...],
        Nx: float = 0,
        Ny: float = 0,
        Ns: float = 0,
        Mx: float = 0,
        My: float = 0,
        Ms: float = 0,
    ):
        """
        Initialize a laminate with given properties and calculate the A, B, D matrices.

        Args:
            plies: Tuple of Lamina objects.
            Nx, Ny, Ns: Normal and shear forces in the x, y, and shear directions (N).
            Mx, My, Ms: Moments around the x, y, and shear axes (Nm).
        """
        self.plies = plies
        self.load = np.array([Nx, Ny, Ns, Mx, My, Ms])

        # Calculate z positions of ply boundaries and midplanes
        self.calculate_z_positions()

        # Initialize and calculate A, B, D matrices, then assemble ABD
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.calculate_abd_matrices()

    def calculate_z_positions(self):
        """Calculate the boundary and midplane z-positions for each ply."""
        ply_thickness = np.array([ply.t for ply in self.plies])
        self.z = np.cumsum(np.insert(ply_thickness, 0, 0))
        self.z -= np.mean(self.z)  # Center around the laminate midplane
        self.z_lamina_midplane = self.z[:-1] + 0.5 * ply_thickness

    def calculate_abd_matrices(self):
        """Calculate the A, B, and D matrices for the laminate and assemble ABD."""
        for i, ply in enumerate(self.plies):
            bottom_z, top_z = self.z[i], self.z[i + 1]
            delta_z = top_z - bottom_z
            delta_z2 = top_z ** 2 - bottom_z ** 2
            delta_z3 = top_z ** 3 - bottom_z ** 3

            self.A += ply.Qbarmat * delta_z
            self.B += 0.5 * ply.Qbarmat * delta_z2
            self.D += (1 / 3) * ply.Qbarmat * delta_z3

        # Assemble the ABD matrix and compute its inverse
        # Suppress PyCharm warning about PyTypeChecker as it should work fine
        # noinspection PyTypeChecker
        self.ABD = np.block([[self.A, self.B], [self.B.T, self.D]])
        self.abd = np.linalg.inv(self.ABD)

    def get_stress_strain(self):
        # Compute global strain (laminate level) on the mid-plane
        strain_midplane = np.linalg.solve(self.ABD, self.load)

        strains = []
        stresses = []
        # Compute local strain (lamina level)
        for i, ply in enumerate(self.plies):
            strain = (
                strain_midplane[:3] + self.z_lamina_midplane[i] * strain_midplane[3:]
            )
            stress = ply.Qbarmat @ strain
            strains.append(strain)
            stresses.append(stress)
        print(strains)
        print(stresses)


if __name__ == "__main__":
    l1 = Lamina(0, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l2 = Lamina(90, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l3 = Lamina(0, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l4 = Lamina(90, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l5 = Lamina(0, 200e9, 50e9, 100e6, 0.4, 0.1e-3)

    L2 = Laminate(plies=(l1, l2, l3), Nx=1E6)
    L2.get_stress_strain()
