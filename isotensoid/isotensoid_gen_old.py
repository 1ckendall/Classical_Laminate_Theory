import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import ezdxf


# --- Classes ---

class IsotensoidProfile:
    def __init__(self, R, r0, a=None):
        self.R = float(R)
        self.r0 = float(r0)
        self.rho0 = self.r0 / self.R

        if a is None:
            self.a = -self.rho0 ** 2
        else:
            self.a = float(a)

        if self.rho0 >= 1.0:
            raise ValueError("Opening radius r0 must be smaller than equator radius R.")

    def _discriminant(self, rho):
        is_toroidal = abs(self.a + self.rho0 ** 2) < 1e-9

        if is_toroidal:
            if abs(rho - self.rho0) < 1e-9:
                return 1e9
            numerator = 1 - self.rho0 ** 2
            denominator = rho ** 2 * (rho ** 2 - self.rho0 ** 2)
            fraction = numerator / denominator
        else:
            numerator = (self.a + 1) ** 2 * (rho ** 2 - self.rho0 ** 2)
            denominator = rho ** 2 * (self.a + rho ** 2) ** 2 * (1 - self.rho0 ** 2)
            if denominator == 0: return 1e9
            fraction = numerator / denominator

        return fraction - 1.0

    def find_valid_start_rho(self):
        if abs(self.a + self.rho0 ** 2) < 1e-9:
            return self.rho0

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
        if rho_prime > 1e9: return 0.0
        if rho_prime < 1e-6: return 1e6
        return 1.0 / rho_prime

    def generate_profile(self, num_points=200):
        rho_start = self.find_valid_start_rho()
        epsilon = 1e-5
        t_span = (rho_start + epsilon, 1.0 - epsilon)
        z0 = [0.0]

        sol = solve_ivp(
            self._ode_func, t_span, z0,
            method='RK45', rtol=1e-6, atol=1e-9, dense_output=True
        )

        rho_smooth = np.linspace(rho_start, 1.0, num_points)
        z_smooth = sol.sol(rho_smooth).flatten()

        max_z = z_smooth[-1]
        z_smooth = max_z - z_smooth
        z_smooth[-1] = 0.0

        self.r_coords = rho_smooth * self.R
        self.z_coords = z_smooth * self.R

        return self.z_coords, self.r_coords


class PressureVessel:
    def __init__(self, R, r0, cylinder_length, a=None):
        self.dome = IsotensoidProfile(R, r0, a)
        self.L_cyl = cylinder_length
        self.R = R
        self.r0 = r0

    @property
    def winding_angle(self):
        """
        Calculates the required helical winding angle (in degrees) at the cylinder section (Equator).
        Based on the Clairaut relation: R * sin(alpha) = r0
        """
        # alpha = arcsin(r0 / R)
        angle_rad = np.arcsin(self.r0 / self.R)
        return np.degrees(angle_rad)

    def generate_full_profile(self, num_points_dome=200):
        z_dome_local, r_dome_local = self.dome.generate_profile(num_points_dome)

        # Left Dome: z goes from (-L/2 - H) -> (-L/2). Strictly increasing.
        z_left = -z_dome_local - (self.L_cyl / 2.0)
        r_left = r_dome_local

        # Cylinder: z goes from (-L/2) -> (L/2). Strictly increasing.
        z_cyl = np.linspace(-self.L_cyl / 2.0, self.L_cyl / 2.0, 10)
        r_cyl = np.full_like(z_cyl, self.R)

        # Right Dome: z goes from (L/2) -> (L/2 + H). Strictly increasing.
        z_right_segment = np.flip(z_dome_local)
        r_right_segment = np.flip(r_dome_local)
        z_right = z_right_segment + (self.L_cyl / 2.0)
        r_right = r_right_segment

        self.z_full = np.concatenate([z_left, z_cyl, z_right])
        self.r_full = np.concatenate([r_left, r_cyl, r_right])

        return self.z_full, self.r_full

    def plot_profile(self):
        z, r = self.generate_full_profile()
        plt.figure(figsize=(12, 5))
        plt.plot(z, r, 'k-', linewidth=2, label='Vessel Wall')
        plt.plot(z, -r, 'k-', linewidth=2)
        plt.axhline(0, color='gray', linestyle='-.', alpha=0.5)

        plt.title(f"Full Pressure Vessel Profile\nR={self.R}mm, L={self.L_cyl}mm, Wind Angle={self.winding_angle:.1f}°")
        plt.xlabel("Axial Position z [mm]")
        plt.ylabel("Radial Position r [mm]")
        plt.axis('equal')
        plt.grid(True, alpha=0.6)
        plt.legend()
        plt.show()


# --- Export Function ---

def export_to_dxf(z_coords, r_coords, filename='vessel_profile.dxf'):
    """
    Exports profile adhering to strict restrictions:
    1. LWPOLYLINE with Plinegen flag (Bit 128)
    2. Maps Z coords -> X axis, R coords -> Y axis
    3. Ensures strictly increasing X (Z) values (removes seam duplicates)
    4. No closing loop
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # 1. Clean Data (Remove duplicates at seams)
    clean_points = []
    clean_points.append((z_coords[0], r_coords[0]))

    epsilon_z = 1e-6
    for i in range(1, len(z_coords)):
        z_curr = z_coords[i]
        r_curr = r_coords[i]
        z_prev = clean_points[-1][0]

        # Only add if Z strictly increases
        if z_curr > (z_prev + epsilon_z):
            clean_points.append((z_curr, r_curr))

    # 2. Create LWPOLYLINE (Open)
    polyline = msp.add_lwpolyline(
        clean_points,
        close=False,
        dxfattribs={'layer': 'PROFILE', 'color': 1}
    )

    # 3. Set Plinegen Flag (Bit 128)
    polyline.dxf.flags |= 128

    doc.saveas(filename)
    print(f"DXF saved to {filename} (Valid LWPOLYLINE, {len(clean_points)} points).")


# --- Usage ---
vessel = PressureVessel(R=115.0, r0=50.0, cylinder_length=150.0)

# Print the new property
print(f"Required Helical Winding Angle: {vessel.winding_angle:.2f} degrees")

# Generate and Export
z_prof, r_prof = vessel.generate_full_profile(num_points_dome=200)
export_to_dxf(z_prof, r_prof, filename="vessel_compliant.dxf")
vessel.plot_profile()