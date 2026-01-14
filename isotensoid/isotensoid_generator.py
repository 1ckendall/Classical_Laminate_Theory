import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import ezdxf
from typing import Tuple, Optional


class IsotensoidProfile:
    def __init__(self, R: float, r0: float, a: Optional[float] = None, cylinder_length: float = 0.0,
                 num_domes: int = 2):
        """
        Generates an isotensoid pressure vessel profile (single or double ended).

        Args:
            R: Equator radius (mm)
            r0: Opening/Boss radius (mm)
            a: Dimensionless load parameter.
               If None, defaults to -rho0^2 (Toroidal/Open-ended condition).
            cylinder_length: Length of the cylindrical section (mm).
            num_domes: 1 for single-ended (cylinder + dome), 2 for double-ended (dome + cylinder + dome).
        """
        self.R = float(R)
        self.r0 = float(r0)
        self.L_cyl = float(cylinder_length)
        self.num_domes = int(num_domes)
        self.rho0 = self.r0 / self.R

        if self.rho0 >= 1.0:
            raise ValueError("Opening radius r0 must be smaller than equator radius R.")

        # Determine load parameter 'a'
        if a is None:
            # Toroidal Condition (Geodesic winding matches cylinder)
            self.a = -self.rho0 ** 2
            self.condition_type = "Toroidal (Open-Ended)"
        else:
            self.a = float(a)
            self.condition_type = f"Custom (a={self.a})"

        # Calculate Winding Angle at Equator
        # based on Clairaut's relation for geodesics: R * sin(alpha) = const
        # At opening: r0 * sin(90) = r0 (assuming tangent entry) -> const = r0
        # At equator: R * sin(alpha_R) = r0
        self.winding_angle_rad = np.arcsin(self.rho0)
        self.winding_angle_deg = np.degrees(self.winding_angle_rad)

        # Storage for generated points
        self.z_coords = None
        self.r_coords = None

    def _discriminant(self, rho: float) -> float:
        """
        Calculates (dr/dz)^2 based on the governing isotensoid equation.
        """
        # Check for Toroidal Condition to avoid 0/0 singularity at rho0
        is_toroidal = np.isclose(self.a + self.rho0 ** 2, 0, atol=1e-9)

        if is_toroidal:
            # Limit form as rho -> rho0 for toroidal case
            if np.isclose(rho, self.rho0, atol=1e-9):
                return 1e9  # Infinite slope dr/dz means zero dz/dr

            numerator = 1 - self.rho0 ** 2
            denominator = rho ** 2 * (rho ** 2 - self.rho0 ** 2)
            fraction = numerator / denominator
        else:
            # Standard General Equation
            numerator = (self.a + 1) ** 2 * (rho ** 2 - self.rho0 ** 2)
            denominator = rho ** 2 * (self.a + rho ** 2) ** 2 * (1 - self.rho0 ** 2)

            if np.isclose(denominator, 0, atol=1e-9):
                return 1e9
            fraction = numerator / denominator

        return fraction - 1.0

    def _ode_func(self, rho: float, z: float) -> float:
        """
        Returns dz/drho (inverse of slope) for the integrator.
        """
        discrim = self._discriminant(rho)

        if discrim <= 0:
            return 0.0

        dr_dz = np.sqrt(discrim)

        # Handle singularities for integration stability
        if dr_dz > 1e9:  # Slope is vertical (dr/dz -> inf) -> dz/dr -> 0
            return 0.0
        if dr_dz < 1e-6:  # Slope is horizontal (dr/dz -> 0) -> dz/dr -> inf
            return 1e6

        return 1.0 / dr_dz

    def generate_profile(self, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the full vessel profile.
        Returns: (z_array, r_array) centered at the cylinder midpoint.
        """
        # --- 1. Generate Single Dome Geometry ---
        # Determine starting point
        if np.isclose(self.a + self.rho0 ** 2, 0, atol=1e-9):
            rho_start = self.rho0
        else:
            # Find root for non-toroidal cases
            try:
                rho_start = brentq(self._discriminant, self.rho0 + 1e-6, 0.9999)
            except ValueError:
                # Fallback scan
                test_rhos = np.linspace(self.rho0, 0.9999, 1000)
                discrims = np.array([self._discriminant(r) for r in test_rhos])
                valid = np.where(discrims >= 0)[0]
                if len(valid) == 0:
                    raise ValueError("No valid solution for these parameters.")
                rho_start = test_rhos[valid[0]]

        # Integration Bounds (slightly buffered to avoid div/0)
        epsilon = 1e-5
        t_span = (rho_start + epsilon, 1.0 - epsilon)
        z0 = [0.0]

        sol = solve_ivp(
            self._ode_func, t_span, z0,
            method='RK45', rtol=1e-6, atol=1e-9, dense_output=True
        )

        rho_smooth = np.linspace(rho_start, 1.0, num_points)
        z_sol = sol.sol(rho_smooth).flatten()

        # Coordinate Transformation for Single Dome
        # z_sol goes 0 -> Length. rho_smooth goes r0 -> R.
        # We need a standardized dome segment: Opening (r0) to Equator (R)

        r_dome_seg = rho_smooth * self.R
        z_dome_seg = z_sol * self.R  # 0 at opening
        dome_height = z_dome_seg[-1]

        # --- 2. Assemble Full Vessel ---
        z_parts = []
        r_parts = []
        half_len = self.L_cyl / 2.0

        # Left Dome (if 2 domes): Mirror and shift
        if self.num_domes == 2:
            # Shift so equator connects at -half_len
            # Opening is at: -half_len - dome_height
            z_left = -(half_len + dome_height) + z_dome_seg
            r_left = r_dome_seg
            z_parts.append(z_left)
            r_parts.append(r_left)

        # Cylinder
        if self.L_cyl > 0:
            if self.num_domes == 2:
                z_cyl = np.array([-half_len, half_len])
            else:
                z_cyl = np.array([-self.L_cyl, 0.0])
            r_cyl = np.array([self.R, self.R])
            z_parts.append(z_cyl)
            r_parts.append(r_cyl)

        # Right Dome: Reverse standard dome (Equator -> Opening) and shift
        # Standard dome: 0(Opening) -> Height(Equator). Reverse it.
        z_dome_rev = z_dome_seg[::-1]
        r_dome_rev = r_dome_seg[::-1]

        start_z = half_len if self.num_domes == 2 else 0.0
        # Shift so equator connects at start_z
        z_right = start_z + (dome_height - z_dome_rev)
        r_right = r_dome_rev

        z_parts.append(z_right)
        r_parts.append(r_right)

        # Stitch
        self.z_coords = np.concatenate(z_parts)
        self.r_coords = np.concatenate(r_parts)

        # Ensure boundary exactness
        if self.num_domes == 2:
            self.r_coords[0] = self.r0
        self.r_coords[-1] = self.r0

        return self.z_coords, self.r_coords

    def plot_profile(self):
        """Plots the full vessel profile."""
        if self.z_coords is None:
            self.generate_profile()

        z = self.z_coords
        r = self.r_coords

        plt.figure(figsize=(12, 6))

        # 1. Plot Full Vessel
        plt.plot(z, r, 'b-', linewidth=2, label='Vessel Profile')
        plt.plot(z, -r, 'b-', linewidth=2)

        # 2. Reference Lines
        plt.axhline(self.r0, color='r', linestyle=':', alpha=0.5, label=f'Opening $r_0$ ({self.r0}mm)')
        plt.axhline(-self.r0, color='r', linestyle=':', alpha=0.5)

        plt.axvline(0, color='k', linestyle='-.', alpha=0.3, label='Center')
        if self.L_cyl > 0 and self.num_domes == 2:
            plt.axvline(self.L_cyl / 2, color='g', linestyle=':', alpha=0.3, label='Tan Line')
            plt.axvline(-self.L_cyl / 2, color='g', linestyle=':', alpha=0.3)

        # 3. Annotations
        plt.title(f"Isotensoid Vessel: {self.condition_type}\n"
                  f"D={2 * self.R}mm, L_cyl={self.L_cyl}mm, Config={self.num_domes} Dome(s)")
        plt.xlabel("Axial Position z [mm]")
        plt.ylabel("Radial Position r [mm]")
        plt.axis('equal')
        plt.grid(True, which='both', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    def export_to_dxf(self, filename: str = 'isotensoid_profile.dxf', add_axis: bool = True, use_spline: bool = False):
        """
        Exports the profile to a DXF.

        Args:
            filename: Output filename
            add_axis: If True, adds a centerline on the Z axis.
            use_spline: If True, uses SPLINE entity.
                        If False, uses LWPOLYLINE with strictly increasing X coordinates and Plinegen flag (Bit 128).
        """
        if self.z_coords is None:
            self.generate_profile()

        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        # Clean Data (Remove duplicates at seams) for Polyline
        # This ensures X (Z) is strictly increasing
        clean_points = []
        if len(self.z_coords) > 0:
            clean_points.append((self.z_coords[0], self.r_coords[0]))

            epsilon_z = 1e-6
            for i in range(1, len(self.z_coords)):
                z_curr = self.z_coords[i]
                r_curr = self.r_coords[i]
                z_prev = clean_points[-1][0]

                # Only add if Z strictly increases
                if z_curr > (z_prev + epsilon_z):
                    clean_points.append((z_curr, r_curr))

        if use_spline:
            # SPLINE uses raw 3D points
            points = [(z, r, 0) for z, r in zip(self.z_coords, self.r_coords)]
            msp.add_spline(points)
            print("Exported as SPLINE")
        else:
            # Create LWPOLYLINE (Open)
            # Maps Z coords -> X axis, R coords -> Y axis
            polyline = msp.add_lwpolyline(
                clean_points,
                close=False,
                dxfattribs={'layer': 'PROFILE', 'color': 1}
            )

            # Set Plinegen Flag (Bit 128)
            polyline.dxf.flags |= 128
            print(f"Exported as LWPOLYLINE (Bit 128, {len(clean_points)} points)")

        # Add centerline
        if add_axis:
            min_z = np.min(self.z_coords)
            max_z = np.max(self.z_coords)
            # Line is 3D in modelspace
            msp.add_line((min_z, 0, 0), (max_z, 0, 0))

        doc.saveas(filename)
        print(f"DXF saved to {filename}")

    def export_1to1_pdf(self, filename: str = "isotensoid_1to1.pdf"):
        """Exports a calibrated 1:1 scale PDF of the full vessel."""
        if self.z_coords is None:
            self.generate_profile()

        # Geometry Bounds
        z_min, z_max = np.min(self.z_coords), np.max(self.z_coords)
        r_max = np.max(self.r_coords)

        # Physical Geometry Size
        geo_width = z_max - z_min
        geo_height = 2 * r_max

        # Margins (mm)
        margin = 20.0
        total_width_mm = geo_width + (2 * margin)
        total_height_mm = geo_height + (2 * margin)

        # Create Figure
        fig_width_in = total_width_mm / 25.4
        fig_height_in = total_height_mm / 25.4
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))

        ax = fig.add_axes([0, 0, 1, 1])

        # Plot Geometry
        ax.plot(self.z_coords, self.r_coords, 'k-', linewidth=0.8)
        ax.plot(self.z_coords, -self.r_coords, 'k-', linewidth=0.8)

        # Centerline
        ax.plot([z_min, z_max], [0, 0], 'k-.', linewidth=0.3)

        # Opening Lines (Vertical)
        if self.num_domes == 2:
            ax.plot([z_min, z_min], [self.r0, -self.r0], 'k-', linewidth=0.5)
        else:
            ax.plot([z_min, z_min], [self.R, -self.R], 'k-', linewidth=0.5)

        ax.plot([z_max, z_max], [self.r0, -self.r0], 'k-', linewidth=0.5)

        # Calibration Square
        center_z = (z_min + z_max) / 2
        limit_left = center_z - (total_width_mm / 2)
        limit_bottom = 0.0 - (total_height_mm / 2)

        rect_x = limit_left + 5
        rect_y = limit_bottom + 5
        rect = Rectangle((rect_x, rect_y), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(rect_x + 12, rect_y + 2, "10mm Scale", color='red', fontsize=8, va='center')

        # Set Limits
        ax.set_xlim(center_z - total_width_mm / 2, center_z + total_width_mm / 2)
        ax.set_ylim(-total_height_mm / 2, total_height_mm / 2)

        ax.set_aspect('equal')
        ax.axis('off')

        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"1:1 Scale PDF saved to {filename}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Example Parameters (mm)
    DIAMETER = 120.0
    BOSS_DIAMETER = 75.0
    CYL_LENGTH = 100.0

    # 1. Initialize
    iso = IsotensoidProfile(
        R=DIAMETER / 2,
        r0=BOSS_DIAMETER / 2,
        cylinder_length=CYL_LENGTH,
        num_domes=2  # Set to 2 for full pressure vessel
    )

    print(f"--- Pressure Vessel Parameters ---")
    print(f"Equator R: {iso.R} mm")
    print(f"Opening r0: {iso.r0} mm")
    print(f"Winding Angle: {iso.winding_angle_deg:.2f} degrees")
    print(f"Load Param a: {iso.a:.5f}")

    # 2. Generate Data
    z, r = iso.generate_profile(num_points=300)
    print(f"Profile generated with {len(z)} points.")
    print(f"Total Length: {np.max(z) - np.min(z):.2f} mm")

    # 3. Plot
    iso.plot_profile()

    # 4. Exports
    # Export as Polyline (default/safer)
    iso.export_to_dxf(filename="vessel_polyline.dxf", use_spline=False)
    # Export as Spline (optional)
    iso.export_to_dxf(filename="vessel_spline.dxf", use_spline=True)

    iso.export_1to1_pdf(filename="vessel_print_template.pdf")