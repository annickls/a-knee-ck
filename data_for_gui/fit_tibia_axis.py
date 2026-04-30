"""
fit_tibial_axis.py
-------------------
Fits the mechanical (longitudinal) axis of the tibia (ZT) to the tibial shaft,
following the method described in:

  Millar et al. (2021) "Development and evaluation of a method to define a
  tibial coordinate system through the fitting of geometric primitives."
  International Biomechanics, 8(1), 12-18.
  https://doi.org/10.1080/23335432.2021.1916406

The paper fits an unbounded cone to the tibial shaft. The axis of that cone is ZT.
For a nearly-cylindrical structure like the tibial shaft (slight taper, half-angle
~85-89 deg), a direct cone parameterisation is ill-conditioned near the degenerate
cylinder limit. We therefore implement the geometrically equivalent and numerically
superior approach:

  1. Slice the shaft into N cross-sectional slabs perpendicular to the initial
     PCA long axis.
  2. Fit a 2-D circle to each slab (algebraic least-squares).
  3. Fit a 3-D line through all circle centers via PCA.

This is exactly what a cone fit converges to: the axis of the best-fit cone is
the line through the centres of circles at successive heights along the shaft.
The perpendicular RMSD of circle centres from the fitted line quantifies quality.

Pipeline:
  1. Load STL
  2. Find initial long axis via PCA
  3. Isolate shaft region (proximal_cut + shaft_length mm)
  4. Slice into N cross-sections; fit circle to each
  5. Fit axis line through circle centres
  6. Report ZT and visualise

Usage:
  python fit_tibial_axis.py tibia.stl
  python fit_tibial_axis.py tibia.stl --proximal_cut 0.25 --shaft_length 200
  python fit_tibial_axis.py tibia.stl --proximal_cut 0.25 --shaft_length 150 --n_slices 30
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ─────────────────────────────────────────────
# 1.  LOAD STL
# ─────────────────────────────────────────────

def load_stl(filepath):
    mesh = trimesh.load(filepath, force="mesh")
    print(f"  Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh


# ─────────────────────────────────────────────
# 2.  PCA INITIAL LONG AXIS
# ─────────────────────────────────────────────

def pca_long_axis(vertices):
    centroid = vertices.mean(axis=0)
    _, _, Vt = np.linalg.svd(vertices - centroid, full_matrices=False)
    long_axis = Vt[0]
    projections = (vertices - centroid) @ -long_axis
    return long_axis, centroid, projections


# ─────────────────────────────────────────────
# 3.  ISOLATE SHAFT REGION
# ─────────────────────────────────────────────

def isolate_shaft(vertices, projections, proximal_cut_frac, shaft_length_mm):
    """
    Isolate shaft vertices between two cuts along the long axis.

    The paper: user marks a point just below the tibial tuberosity (proximal cut),
    then a 200 mm section is taken distally from that point.

    Approximation here:
      - Proximal end = max(projections)  (most superior / knee-side)
      - Proximal cut = proj_max - proximal_cut_frac * total_length
      - Distal cut   = proximal_cut - shaft_length_mm
    """
    proj_min = projections.min()
    proj_max = projections.max()
    total_length = proj_max - proj_min

    prox_cut = proj_max - proximal_cut_frac * total_length
    dist_cut = prox_cut - shaft_length_mm

    mask = (projections <= prox_cut) & (projections >= dist_cut)
    shaft_verts = vertices[mask]
    shaft_projs = projections[mask]

    actual_len = prox_cut - max(dist_cut, proj_min)
    print(f"  Total bone length            : {total_length:.1f} mm")
    print(f"  Proximal cut (projection)    : {prox_cut:.1f} mm  "
          f"({proximal_cut_frac*100:.0f}% from proximal end)")
    print(f"  Distal cut   (projection)    : {dist_cut:.1f} mm")
    print(f"  Shaft length used            : {actual_len:.1f} mm")
    print(f"  Shaft vertices selected      : {len(shaft_verts):,}")

    if len(shaft_verts) < 50:
        raise ValueError(
            "Too few shaft vertices selected. "
            "Try a larger --shaft_length or smaller --proximal_cut."
        )
    return shaft_verts, shaft_projs, prox_cut, dist_cut


# ─────────────────────────────────────────────
# 4A. CIRCLE FIT IN 2-D  (algebraic least-squares)
# ─────────────────────────────────────────────

def fit_circle_2d(pts2d):
    """
    Algebraic LS circle fit:  x^2 + y^2 + Dx + Ey + F = 0
    Returns (cx, cy, radius) or None if degenerate.
    """
    if len(pts2d) < 3:
        return None
    x, y = pts2d[:, 0], pts2d[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    res, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return None
    D, E, F = res
    cx, cy = D / 2.0, E / 2.0
    r2 = cx**2 + cy**2 + F
    if r2 <= 0:
        return None
    return cx, cy, float(np.sqrt(r2))


# ─────────────────────────────────────────────
# 4B. BUILD PLANE BASIS VECTORS
# ─────────────────────────────────────────────

def build_plane_basis(normal):
    arb = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arb); u /= np.linalg.norm(u)
    v = np.cross(normal, u);   v /= np.linalg.norm(v)
    return u, v


# ─────────────────────────────────────────────
# 4C. SLICE SHAFT + FIT CIRCLES
# ─────────────────────────────────────────────

def slice_and_fit_circles(shaft_verts, shaft_projs, long_axis,
                          centroid_bone, n_slices, min_pts=10):
    """
    Divide shaft into n_slices slabs perpendicular to long_axis.
    Fit a 2-D circle to each slab and return the 3-D circle centres.
    """
    u, v = build_plane_basis(long_axis)
    edges = np.linspace(shaft_projs.min(), shaft_projs.max(), n_slices + 1)

    centers_3d, radii, slab_mids = [], [], []

    for i in range(n_slices):
        mask = (shaft_projs >= edges[i]) & (shaft_projs < edges[i + 1])
        slab = shaft_verts[mask]
        if len(slab) < min_pts:
            continue

        slab_mid_proj = (edges[i] + edges[i + 1]) / 2.0
        slab_origin = centroid_bone + slab_mid_proj * long_axis

        # 2-D projection perpendicular to long_axis
        centered = slab - slab_origin
        pts2d = np.column_stack([centered @ u, centered @ v])

        result = fit_circle_2d(pts2d)
        if result is None:
            continue
        cx2d, cy2d, r = result
        center_3d = slab_origin + cx2d * u + cy2d * v

        centers_3d.append(center_3d)
        radii.append(r)
        slab_mids.append(slab_mid_proj)

    centers_3d = np.array(centers_3d)
    radii      = np.array(radii)
    slab_mids  = np.array(slab_mids)
    print(f"  Valid slabs with circle fits  : {len(centers_3d)} / {n_slices}")
    return centers_3d, radii, slab_mids


# ─────────────────────────────────────────────
# 5.  FIT LINE THROUGH CIRCLE CENTRES -> ZT
# ─────────────────────────────────────────────

def fit_axis_through_centers(centers_3d):
    """
    Best-fit 3-D line through circle centres via PCA.
    Returns axis (unit vector), line centroid, and perpendicular RMSD (mm).
    """
    centroid = centers_3d.mean(axis=0)
    _, s, Vt = np.linalg.svd(centers_3d - centroid, full_matrices=False)
    axis = Vt[0]   # first PC = direction of maximum variance = long axis

    # Perpendicular distance of each centre from the line
    perp = (centers_3d - centroid) - ((centers_3d - centroid) @ axis)[:, None] * axis
    rmsd_mm = float(np.sqrt(np.mean(np.sum(perp**2, axis=1))))
    return axis, centroid, rmsd_mm


# ─────────────────────────────────────────────
# 6.  ORIENT AXIS CONSISTENTLY (proximal->distal)
# ─────────────────────────────────────────────

def orient_proximal_to_distal(axis, long_axis):
    if np.dot(axis, long_axis) < 0:
        axis = -axis
    return axis


# ─────────────────────────────────────────────
# 7.  VISUALISE
# ─────────────────────────────────────────────

def _equal_ax(ax, verts):
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mid = (lo + hi) / 2
    r   = (hi - lo).max() / 2
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


def plot_results(mesh, shaft_verts, centers_3d, radii, slab_mids,
                 axis, axis_centroid, long_axis, centroid_bone,
                 projections, prox_cut, dist_cut, rmsd_mm):

    all_verts = np.array(mesh.vertices)
    fig = plt.figure(figsize=(17, 6))

    # ── 3-D view ──────────────────────────────────────────
    ax3d = fig.add_subplot(131, projection="3d")
    stride = max(1, len(all_verts) // 3000)
    ax3d.scatter(*all_verts[::stride].T, color="lightsteelblue", s=1, alpha=0.12,
                 label="Full tibia")
    stride_s = max(1, len(shaft_verts) // 2000)
    ax3d.scatter(*shaft_verts[::stride_s].T, color="royalblue", s=3, alpha=0.5,
                 label="Shaft region")
    ax3d.scatter(*centers_3d.T, color="orange", s=30, zorder=6, label="Slab centres")

    # Draw axis through full bone extent
    t_range = np.linspace(-250, 250, 300)
    line_pts = np.array([axis_centroid + t * axis for t in t_range])
    ax3d.plot(*line_pts.T, color="tomato", linewidth=3, label="ZT  (mech. axis)")

    for proj_val, col, lbl in [(prox_cut, "lime", "Proximal cut"),
                                (dist_cut, "gold",  "Distal cut")]:
        pt = centroid_bone + proj_val * long_axis
        ax3d.scatter(*pt, color=col, s=80, marker="^", zorder=7, label=lbl)

    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3-D view")
    ax3d.legend(fontsize=6, loc="upper left")
    _equal_ax(ax3d, all_verts)

    # ── Circle centre scatter (2 views) ──────────────────
    ax2 = fig.add_subplot(132)
    # Project centres onto the two minor PCA directions (perpendicular to axis)
    cc = centers_3d - axis_centroid
    _, _, Vt2 = np.linalg.svd(cc, full_matrices=False)
    u2, v2 = Vt2[1], Vt2[2]   # second and third PCs
    c2d = np.column_stack([cc @ u2, cc @ v2])
    sc = ax2.scatter(c2d[:, 0], c2d[:, 1], c=slab_mids, cmap="plasma", s=50, zorder=5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Minor PC 1 of centres (mm)")
    ax2.set_ylabel("Minor PC 2 of centres (mm)")
    ax2.set_title(f"Circle centres — lateral scatter\n"
                  f"RMSD from axis = {rmsd_mm:.3f} mm")
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.colorbar(sc, ax=ax2, label="Slab position (mm)")

    # ── Radius profile ─────────────────────────────────
    ax3 = fig.add_subplot(133)
    ax3.plot(slab_mids, radii, "o-", color="royalblue", markersize=5)
    if len(slab_mids) > 1:
        p = np.polyfit(slab_mids, radii, 1)
        t_l = np.linspace(slab_mids.min(), slab_mids.max(), 100)
        taper_mm_per_10mm = p[0] * 10
        ax3.plot(t_l, np.polyval(p, t_l), "tomato", linestyle="--",
                 label=f"Taper: {taper_mm_per_10mm:.2f} mm / 10 mm")
        ax3.legend(fontsize=8)
    ax3.set_xlabel("Slab position along shaft (mm)")
    ax3.set_ylabel("Circle radius (mm)")
    ax3.set_title("Shaft radius profile\n(confirms conical taper)")
    ax3.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Tibial Mechanical Axis ZT  |  "
        f"[{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]  |  "
        f"Centre-line RMSD = {rmsd_mm:.3f} mm",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def point_on_axis_at_z(z_target, centroid, axis):
    # P(t) = centroid + t * axis
    # P_z(t) = centroid[2] + t * axis[2] = z_target
    # → t = (z_target - centroid[2]) / axis[2]
    t = (z_target - centroid[2]) / axis[2]
    return centroid + t * axis


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ─────────────────────────────────────────────
    # CONFIGURATION - Edit these values
    # ─────────────────────────────────────────────
    patientName = "P6_pre"
    fileName = "tibia_remeshed.stl"
    stl_file = os.path.join("data_for_gui", patientName, fileName)  # Path to STL file
    proximal_cut = 0.20                   # Fraction from proximal end for upper shaft cut
    shaft_length = 230.0                  # Length of shaft section in mm
    n_slices = 30                         # Number of cross-sectional slabs
    z_proximal = 1789
    z_distal = 2150

    # ─────────────────────────────────────────────
    
    # Validate STL file path
    if not stl_file or stl_file == "path/to/your/tibia.stl":
        print("  Error: Please set a valid STL file path in the script.")
        return
    
    # Create args-like object with the values
    class Args:
        pass
    args = Args()
    args.stl_file = stl_file
    args.proximal_cut = proximal_cut
    args.shaft_length = shaft_length
    args.n_slices = n_slices

    print(f"\n{'='*62}")
    print(f"  Tibial Mechanical Axis Fitting  (Millar et al. 2021)")
    print(f"{'='*62}")

    print(f"\n[1] Loading STL: {args.stl_file}")
    mesh = load_stl(args.stl_file)
    vertices = np.array(mesh.vertices)

    print("\n[2] Computing bone orientation (PCA)")
    long_axis, centroid_bone, projections = pca_long_axis(vertices)
    print(f"  Long axis: [{long_axis[0]:.4f}, {long_axis[1]:.4f}, {long_axis[2]:.4f}]")

    print("\n[3] Isolating shaft region")
    shaft_verts, shaft_projs, prox_cut, dist_cut = isolate_shaft(
        vertices, projections, args.proximal_cut, args.shaft_length
    )

    print(f"\n[4] Slicing shaft into {args.n_slices} cross-sections and fitting circles")
    centers_3d, radii, slab_mids = slice_and_fit_circles(
        shaft_verts, shaft_projs, long_axis, centroid_bone, args.n_slices
    )
    if len(centers_3d) < 3:
        raise RuntimeError("Too few valid circle fits. Try --n_slices or --shaft_length.")

    print("\n[5] Fitting mechanical axis through circle centres")
    axis, axis_centroid, rmsd_mm = fit_axis_through_centers(centers_3d)
    axis = orient_proximal_to_distal(axis, long_axis)

    # Angles relative to global Z (assuming STL oriented SI along Z)
    angle_coronal  = np.rad2deg(np.arctan2(abs(axis[0]), abs(axis[2])))
    angle_sagittal = np.rad2deg(np.arctan2(abs(axis[1]), abs(axis[2])))

    # Calculate the point to copy for the fcsv
    p_proximal = point_on_axis_at_z(z_proximal, axis_centroid, axis)
    p_distal = point_on_axis_at_z(z_distal, axis_centroid, axis)

    print(f"\n{'='*62}")
    print(f"  RESULTS  -  Tibial Mechanical Axis (ZT)")
    print(f"{'='*62}")
    print(f"  Axis direction   : [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
    print(f"  Axis centroid    : [{axis_centroid[0]:.2f}, {axis_centroid[1]:.2f},"
          f" {axis_centroid[2]:.2f}] mm")
    print(f"  Centre-line RMSD : {rmsd_mm:.4f} mm  (scatter of slab centres from axis)")
    print(f"  Radius range     : {radii.min():.1f} - {radii.max():.1f} mm")
    print(f"  Coronal angle    : {angle_coronal:.2f} deg  (ZT tilt in XZ plane)")
    print(f"  Sagittal angle   : {angle_sagittal:.2f} deg  (ZT tilt in YZ plane)")
    print(f"{'='*62}")

    print(f"Lines to copy for fcsv: ")
    print(f"{p_proximal[0]:.1f},{p_proximal[1]:.1f},{p_proximal[2]:.1f},0,0,0,1,1,1,0,tibia_proximal")
    print(f"{p_distal[0]:.1f},{p_distal[1]:.1f},{p_distal[2]:.1f},0,0,0,1,1,1,0,tibia_distal")
    print(f"{'='*62}\n")

    print("[6] Plotting results")
    plot_results(mesh, shaft_verts, centers_3d, radii, slab_mids,
                 axis, axis_centroid, long_axis, centroid_bone,
                 projections, prox_cut, dist_cut, rmsd_mm)
    print("Finished")



if __name__ == "__main__":
    main()