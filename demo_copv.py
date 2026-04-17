"""
COPV Design Demo
================
Step-by-step design of a Carbon/Epoxy Composite Overwrapped Pressure Vessel.

Workflow
--------
  1. Define geometry & material
  2. Derive geodesic winding angle (Clairaut's relation)
  3. Optimise layup   – helical pairs to satisfy the dome,
                        hoop pairs to satisfy the cylinder burst
  4. Cylinder PFA     – verify burst pressure via Progressive Failure Analysis
  5. Dome sweep       – verify every radial station at operating pressure
  6. Summary & plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from classical_laminate_theory import (
    Material,
    Laminate,
    Puck,
    ProgressiveFailureAnalysis,
)
from classical_laminate_theory.vessel_optimizer import VesselOptimizer
from classical_laminate_theory.vessel_analyzer import DomeAnalyzer
from isotensoid.isotensoid_generator import IsotensoidProfile


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VESSEL GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
OD      = 0.100   # Outer diameter        (m)
R       = OD / 2  # Equator radius        (m)
r0      = 0.020   # Boss opening radius   (m)
L_cyl   = 0.200   # Cylinder section len  (m)

P_op    = 20e6    # Operating pressure    (Pa)
P_burst = 40e6    # Target burst pressure (Pa)  — 2× safety factor on P_op


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MATERIAL  –  T700/Epoxy
# ─────────────────────────────────────────────────────────────────────────────
t_ply = 0.15e-3   # Cured ply thickness (m)

mat = Material(
    name = "T700/Epoxy",
    E1   = 130e9,  E2  = 9e9,   G12 = 5e9,  v12 = 0.30,
    Xt   = 2100e6, Xc  = 1050e6,
    Yt   = 55e6,   Yc  = 150e6,
    S12  = 80e6,
    t    = t_ply,  # stored in material.extra; read by optimizer & dome analyzer
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ISOTENSOID GEOMETRY  &  GEODESIC WINDING ANGLE
#     Clairaut's relation:  r · sin α = const = r0
#     ∴  α_equator = arcsin(r0 / R)
# ─────────────────────────────────────────────────────────────────────────────
iso = IsotensoidProfile(R=R, r0=r0, cylinder_length=L_cyl, num_domes=2)
alpha_hel = iso.winding_angle_deg

print("=" * 62)
print("  COPV DESIGN DEMO")
print("=" * 62)
print(f"\n  Geometry   : OD = {OD*1000:.0f} mm | L_cyl = {L_cyl*1000:.0f} mm"
      f" | r0 = {r0*1000:.0f} mm")
print(f"  Material   : {mat.name}")
print(f"  Pressures  : P_op = {P_op/1e6:.0f} MPa | P_burst = {P_burst/1e6:.0f} MPa")
print(f"\n  Geodesic winding angle : α = arcsin({r0}/{R}) = {alpha_hel:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LAYUP OPTIMISATION
#     • Helical pairs  : minimum needed for the dome to survive at P_burst
#     • Hoop pairs     : added until the cylinder survives at P_burst
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
print("  STEP 1 – Layup optimisation")
print("─" * 62)

opt = VesselOptimizer(mat, P_burst, R, r0)
optimised_layup = opt.optimize_layup()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CYLINDER PROGRESSIVE FAILURE ANALYSIS
#     Biaxial membrane loads for a thin-walled cylinder:
#       Nx (axial)  = P · R / 2
#       Ny (hoop)   = P · R
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
print("  STEP 2 – Cylinder burst verification (PFA)")
print("─" * 62)

model = Puck(material=mat, weakening="puck")
lam   = Laminate.from_layup(optimised_layup, material=mat, failure_model=model)

# Load direction: one unit of "pressure" applies (R/2, R) N/m to the laminate
Nx_per_Pa    = R / 2.0
Ny_per_Pa    = R
load_dir     = np.array([Nx_per_Pa, Ny_per_Pa, 0.0, 0.0, 0.0, 0.0])
step_size_Pa = P_burst / 500        # ≈500 increments to reach target burst

pfa = ProgressiveFailureAnalysis(lam)
pfa.run_until_failure(load_dir, step_size=step_size_Pa, max_steps=2000)

# Recover the burst pressure from the stored Nx at last step
nx_at_burst  = pfa.history["load_factor"][-1]
burst_P_calc = nx_at_burst / Nx_per_Pa

print(f"\n  Calculated burst pressure : {burst_P_calc/1e6:.1f} MPa")
print(f"  Safety factor on P_op     : {burst_P_calc/P_op:.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DOME INTEGRITY SWEEP
#     Membrane analysis at P_op across every radial station from equator → boss
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
print("  STEP 3 – Dome integrity sweep at P_op")
print("─" * 62)

# Total helical-layer thickness at the equator
n_helical    = sum(1 for ply in lam.plies
                   if abs(abs(np.degrees(ply.angle)) - 90.0) > 1.0)
t_helical_eq = n_helical * t_ply
print(f"  Helical plies in layup : {n_helical}"
      f"  →  t_helical_equator = {t_helical_eq*1000:.3f} mm")

dome       = DomeAnalyzer(iso, mat, t_helical_equator=t_helical_eq)
dome_data  = dome.analyze_dome(P_op, num_stations=100)

max_effort       = max(dome_data["effort"])
critical_radius  = dome_data["radius"][np.argmax(dome_data["effort"])]
print(f"  Max Puck effort : {max_effort:.3f}"
      f"  at r = {critical_radius*1000:.1f} mm"
      f"  → {'PASS' if max_effort < 1.0 else 'FAIL'}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
total_t  = sum(ply.t for ply in lam.plies)
n_plies  = lam.n_plies

print("\n" + "=" * 62)
print("  RESULTS SUMMARY")
print("=" * 62)
print(f"  Layup            :  {optimised_layup}")
print(f"  Ply count        :  {n_plies}")
print(f"  Wall thickness   :  {total_t*1000:.3f} mm")
print(f"  Burst pressure   :  {burst_P_calc/1e6:.1f} MPa  "
      f"(target ≥ {P_burst/1e6:.0f} MPa)")
print(f"  Safety factor    :  {burst_P_calc/P_op:.2f}×  (target ≥ 2.0×)")
print(f"  Max dome effort  :  {max_effort:.3f}  @ P_op "
      f"({'PASS' if max_effort < 1.0 else 'FAIL'})")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    f"COPV Design Summary — {mat.name}\n"
    f"OD = {OD*1000:.0f} mm | L_cyl = {L_cyl*1000:.0f} mm"
    f" | P_burst target = {P_burst/1e6:.0f} MPa",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)


# ── 8a.  Vessel profile ──────────────────────────────────────────────────────
ax_prof = fig.add_subplot(gs[0, :2])

z_mm, r_mm = iso.generate_profile(num_points=300)
z_mm, r_mm = z_mm * 1000, r_mm * 1000

ax_prof.plot(z_mm,  r_mm, "steelblue", linewidth=2.5)
ax_prof.plot(z_mm, -r_mm, "steelblue", linewidth=2.5)
ax_prof.fill_between(z_mm, r_mm, -r_mm, alpha=0.10, color="steelblue")

# Tangent-line markers
for sign in (-1, 1):
    ax_prof.axvline(sign * L_cyl / 2 * 1000, color="forestgreen",
                    linestyle=":", linewidth=1.2, alpha=0.7,
                    label="Tangent line" if sign == 1 else None)

ax_prof.axhline(0, color="k", linestyle="-.", linewidth=0.7, alpha=0.4)
ax_prof.set_xlabel("Axial position (mm)", fontweight="bold")
ax_prof.set_ylabel("Radius (mm)", fontweight="bold")
ax_prof.set_title(
    f"Isotensoid Profile  —  α = {alpha_hel:.1f}°,  "
    f"r₀/R = {r0/R:.2f}",
    fontweight="bold",
)
ax_prof.set_aspect("equal")
ax_prof.legend(fontsize=9)


# ── 8b.  Ply composition pie chart ──────────────────────────────────────────
ax_pie = fig.add_subplot(gs[0, 2])

angle_deg = [round(abs(np.degrees(ply.angle))) for ply in lam.plies]
counts = {}
for a in angle_deg:
    counts[a] = counts.get(a, 0) + 1

def angle_label(a):
    if a == 0:   return "0°"
    if a == 90:  return "90°"
    return f"±{a}°"

labels  = [angle_label(a) for a in counts]
colors  = sns.color_palette("deep", len(counts))
wedges, texts, autotexts = ax_pie.pie(
    list(counts.values()), labels=labels, autopct="%1.0f%%",
    startangle=90, colors=colors,
)
for at in autotexts:
    at.set_fontsize(9)
ax_pie.set_title(
    f"Layup Composition\n{n_plies} plies · {total_t*1000:.2f} mm wall",
    fontweight="bold",
)


# ── 8c.  PFA stress–strain curve ────────────────────────────────────────────
ax_pfa = fig.add_subplot(gs[1, :2])
plt.sca(ax_pfa)   # make this the active axes for plot_curve's plt.* calls

pfa.plot_curve(title_suffix=f"Puck · {optimised_layup}")

# Mark the burst point
burst_strain_pct = pfa.history["strain"][-1] * 100
burst_stress_MPa = pfa.history["stress"][-1] / 1e6
ax_pfa.plot(burst_strain_pct, burst_stress_MPa, "r*", markersize=14,
            zorder=5, label=f"Burst  {burst_P_calc/1e6:.1f} MPa")
ax_pfa.legend(fontsize=9)
ax_pfa.set_title(
    "Cylinder PFA — Biaxial load  (Nx = P·R/2,  Ny = P·R)",
    fontweight="bold",
)


# ── 8d.  Dome effort sweep ───────────────────────────────────────────────────
ax_dome = fig.add_subplot(gs[1, 2])

r_mm_dome = np.array(dome_data["radius"]) * 1000
effort    = np.array(dome_data["effort"])

ax_dome.plot(r_mm_dome, effort, color="firebrick", linewidth=2)
ax_dome.axhline(y=1.0, color="k", linestyle="--", linewidth=1.2,
                label="Failure criterion")
ax_dome.fill_between(r_mm_dome, effort, 1.0,
                     where=effort > 1.0,
                     color="red", alpha=0.3, label="Failed zone")
ax_dome.fill_between(r_mm_dome, effort, 0,
                     where=effort <= 1.0,
                     color="green", alpha=0.08)

ax_dome.axvline(critical_radius * 1000, color="firebrick", linestyle=":",
                alpha=0.6, label=f"Peak @ r={critical_radius*1000:.1f} mm")
ax_dome.set_xlabel("Radius (mm)", fontweight="bold")
ax_dome.set_ylabel("Puck Failure Index", fontweight="bold")
ax_dome.set_title(
    f"Dome Sweep @ P_op = {P_op/1e6:.0f} MPa\n"
    f"Max effort = {max_effort:.3f}",
    fontweight="bold",
)
ax_dome.legend(fontsize=9)
ax_dome.set_xlim(r_mm_dome.min(), r_mm_dome.max())


plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
