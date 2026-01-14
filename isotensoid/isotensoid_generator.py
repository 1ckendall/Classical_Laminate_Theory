import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import ezdxf


class IsotensoidProfile:
    def __init__(self, R, r0, a=None):
        """
        Args:
            R: Equator radius
            r0: Opening radius
            a: Dimensionless load parameter.
               If None, defaults to -rho0^2 (Toroidal/Open-ended condition).
        """
        self.R = float(R)
        self.r0 = float(r0)
        self.rho0 = self.r0 / self.R

        if a is None:
            self.a = -self.rho0 ** 2
            print(f"Using Toroidal Condition: a = -rho0^2 = {self.a:.5f}")
        else:
            self.a = float(a)

        if self.rho0 >= 1.0:
            raise ValueError("Opening radius r0 must be smaller than equator radius R.")

    def get_winding_angle_at_R(self):
        """
        Calculates the winding entry angle at the equator (R).
        Based on Eq 12: r * sin(alpha) = r0
        At R (rho=1): R * sin(alpha) = r0 -> sin(alpha) = rho0
        """
        angle_rad = np.arcsin(self.rho0)
        return np.degrees(angle_rad)

    def _discriminant(self, rho):
        """
        Calculates the term inside the square root of the governing equation.
        Handles the special singularity when a = -rho0^2.
        """
        # 1. Check for the Toroidal Condition (a = -rho0^2)
        # At rho = rho0, the standard equation creates a 0/0 division.
        # We use the simplified limit form: Fraction = (1 - rho0^2) / (rho^2 * (rho^2 - rho0^2))
        is_toroidal = abs(self.a + self.rho0 ** 2) < 1e-9

        if is_toroidal:
            # Avoid division by zero at exactly rho0
            if abs(rho - self.rho0) < 1e-9:
                return 1e9  # Infinite slope (rho'), means zero dz/dr

            numerator = 1 - self.rho0 ** 2
            denominator = rho ** 2 * (rho ** 2 - self.rho0 ** 2)
            fraction = numerator / denominator
        else:
            # Standard Case
            numerator = (self.a + 1) ** 2 * (rho ** 2 - self.rho0 ** 2)
            denominator = rho ** 2 * (self.a + rho ** 2) ** 2 * (1 - self.rho0 ** 2)
            if denominator == 0: return 1e9
            fraction = numerator / denominator

        return fraction - 1.0

    def find_valid_start_rho(self):
        """
        Finds the start point.
        If a = -rho0^2, the start is exactly rho0.
        Otherwise, finds the inflection point numerically.
        """
        # If toroidal, we know the solution starts exactly at rho0
        if abs(self.a + self.rho0 ** 2) < 1e-9:
            return self.rho0

        # Standard case: scan for zero crossing
        try:
            rho_start = brentq(self._discriminant, self.rho0 + 1e-6, 0.9999)
        except ValueError:
            test_rhos = np.linspace(self.rho0, 0.9999, 1000)
            discrims = np.array([self._discriminant(r) for r in test_rhos])
            valid_indices = np.where(discrims >= 0)[0]
            if len(valid_indices) == 0:
                raise ValueError("No valid solution for these parameters.")
            rho_start = test_rhos[valid_indices[0]]

        return rho_start

    def _ode_func(self, rho, z):
        discrim = self._discriminant(rho)
        if discrim <= 0: return 0.0

        rho_prime = np.sqrt(discrim)

        # If rho_prime is huge (infinite), dz/dr is 0.
        if rho_prime > 1e9:
            return 0.0
        # If rho_prime is tiny (zero), dz/dr is infinite.
        if rho_prime < 1e-6:
            return 1e6

        return 1.0 / rho_prime

    def generate_profile(self, num_points=200):
        rho_start = self.find_valid_start_rho()

        # Integration bounds
        # We buffer slightly to avoid the exact singularities at ends
        epsilon = 1e-5
        t_span = (rho_start + epsilon, 1.0 - epsilon)
        z0 = [0.0]

        sol = solve_ivp(
            self._ode_func, t_span, z0,
            method='RK45', rtol=1e-6, atol=1e-9, dense_output=True
        )

        rho_smooth = np.linspace(rho_start, 1.0, num_points)
        z_smooth = sol.sol(rho_smooth).flatten()

        # Coordinate Transformation (z=0 at Equator)
        max_z = z_smooth[-1]
        z_smooth = max_z - z_smooth
        z_smooth[-1] = 0.0  # Force exact equator

        # Note: For Toroidal case (a = -rho0^2), z_smooth[0] is correct
        # without extra extension because the solution starts at rho0.

        self.r_coords = rho_smooth * self.R
        self.z_coords = z_smooth * self.R

        return self.z_coords, self.r_coords

    def plot_profile(self):
        z, r = self.generate_profile()

        plt.figure(figsize=(10, 6))
        plt.plot(z, r, 'b-', label='Meridian Profile')
        plt.plot(z, -r, 'b-')

        plt.axhline(self.r0, color='r', linestyle='--', alpha=0.5, label=f'Opening $r_0$')
        plt.axhline(-self.r0, color='r', linestyle='--', alpha=0.5)

        plt.title(f"Isotensoid Profile\n$a={self.a:.4f}$ (Toroidal Condition: {abs(self.a + self.rho0 ** 2) < 1e-9}) \n Wind angle: {np.degrees(np.arcsin(dome.rho0)):.2f} degrees")
        plt.title(f"Full Pressure Vessel Profile\nR={self.R}mm, L={self.L_cyl}mm, Wind Angle={self.winding_angle:.1f}°")
        plt.xlabel("Axial Position z [mm]")
        plt.ylabel("Radial Position r [mm]")
        plt.axis('equal')
        plt.grid(True, alpha=0.6)
        plt.legend()
        plt.show()


def export_to_dxf(z_coords, r_coords, add_axis = False, filename='isotensoid_profile.dxf'):
    # Create a new DXF document
    doc = ezdxf.new('R2018')
    msp = doc.modelspace()

    # Create a list of (x, y, z) tuples
    # Mapping Z_profile -> X, R_profile -> Y
    points = [(z, r, 0) for z, r in zip(z_coords, r_coords)]

    # Add a spline through these points
    msp.add_spline(points)

    # Optional: Add a centerline for the revolve axis
    if add_axis:
        min_x = min(z_coords)
        max_x = max(z_coords)
        msp.add_line((min_x, 0, 0), (max_x, 0, 0))

    doc.saveas(filename)
    print(f"DXF saved to {filename}")


def export_1to1_pdf(z_coords, r_coords, filename="isotensoid_1to1.pdf"):
    """
    Exports a 1:1 scale PDF of the full cross-section (both sides).
    Includes a 10mm calibration square.
    """
    # 1. Determine physical dimensions in mm
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    r_max = np.max(r_coords)

    # Calculate bounds for both sides (top and bottom)
    height_mm = 2 * r_max
    width_mm = z_max - z_min

    # 2. Add Margins (e.g., 20mm padding total)
    margin_mm = 20
    total_width_mm = width_mm + margin_mm
    total_height_mm = height_mm + margin_mm

    # 3. Convert to inches for Matplotlib (25.4 mm = 1 inch)
    fig_width_in = total_width_mm / 25.4
    fig_height_in = total_height_mm / 25.4

    # 4. Create Figure with exact dimensions
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    # Create axes that fill the figure exactly (0 to 1 in relative coords)
    # We will control the specific data limits to handle margins
    ax = fig.add_axes([0, 0, 1, 1])

    # 5. Plot the Geometry
    # Top profile
    ax.plot(z_coords, r_coords, 'k-', linewidth=1.0)
    # Bottom profile (mirrored)
    ax.plot(z_coords, -r_coords, 'k-', linewidth=1.0)

    # Centerline
    ax.plot([z_min, z_max], [0, 0], 'k-.', linewidth=0.5, alpha=0.5)

    # Opening lines (vertical lines at z_max)
    ax.plot([z_max, z_max], [r_coords[0], -r_coords[0]], 'k-', linewidth=0.5)

    # 6. Add a Calibration Square (10mm x 10mm)
    # Place it near the bottom left (but inside the margin)
    cal_x = z_min - (margin_mm / 2) + 2
    cal_y = -r_max - (margin_mm / 2) + 2

    # If the square falls outside the view, shift it closer to geometry
    rect = Rectangle((z_max, -r_max), 10, 10,
                     linewidth=1, edgecolor='r', facecolor='none', label='10mm Scale')
    ax.add_patch(rect)
    ax.text(z_max + 1, -r_max + 12, "10mm Scale", color='red', fontsize=8)

    # 7. Set Axis Limits to match the Figure Size exactly
    # The figure center is the geometry center
    center_z = (z_min + z_max) / 2
    center_r = 0

    # Set limits based on the total physical size we calculated
    ax.set_xlim(center_z - total_width_mm / 2, center_z + total_width_mm / 2)
    ax.set_ylim(center_r - total_height_mm / 2, center_r + total_height_mm / 2)

    # Ensure aspect ratio is preserved
    ax.set_aspect('equal')

    # Turn off axis labels/ticks for a clean template
    ax.axis('off')

    # 8. Save
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"1:1 Scale PDF saved to {filename}")
    print(f"  - Physical Size: {total_width_mm:.1f}mm x {total_height_mm:.1f}mm")
    print("  - NOTE: When printing, ensure 'Scale' is set to '100%' or 'Actual Size'.")


# --- Example Usage ---
# We do not pass 'a', so it defaults to the Toroidal condition (-rho0^2)
dome = IsotensoidProfile(R=115.0/2, r0=50.0/2)
dome.plot_profile()

# --- Usage ---
z, r = dome.generate_profile(num_points=200)
export_to_dxf(z, r)

# Check the winding angle at R
print(f"Winding angle at R: {np.degrees(np.arcsin(dome.rho0)):.2f} degrees")