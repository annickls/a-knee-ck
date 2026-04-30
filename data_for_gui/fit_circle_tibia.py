"""
fit_circle_3d.py
-----------------
Fits a best-fit circle to 3D coplanar points loaded from a 3D Slicer .fcsv file.

Pipeline:
  1. Parse .fcsv → 3D point cloud
  2. Fit plane via SVD (PCA on centred points)
  3. Project points into the plane's 2D coordinate system
  4. Algebraic least-squares circle fit in 2D
  5. Back-project circle centre to 3D
  6. Report results and plot

Usage:
  python fit_circle_3d.py                          # uses default path below
  python fit_circle_3d.py my_points.fcsv           # or pass a path as argument
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3-D projection)


# ─────────────────────────────────────────────
# 1.  LOAD POINTS FROM .FCSV
# ─────────────────────────────────────────────

def load_fcsv(filepath: str) -> np.ndarray:
    """Return an (N, 3) array of x, y, z coordinates from a Slicer .fcsv file."""
    points = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):   # skip header / comments
                continue
            parts = line.split(",")
            # columns: id, x, y, z, ow, ox, oy, oz, ...
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z])
    return np.array(points)


# ─────────────────────────────────────────────
# 2.  FIT PLANE VIA SVD
# ─────────────────────────────────────────────

def fit_plane(points: np.ndarray):
    """
    Fit a plane to 3D points using SVD.
    Returns:
        centroid  : (3,)  mean of the point cloud
        normal    : (3,)  unit normal of the best-fit plane
        plane_rmsd: float residual RMS distance of points from the plane
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, s, Vt = np.linalg.svd(centered)
    normal = Vt[-1]                          # eigenvector of smallest singular value
    normal /= np.linalg.norm(normal)
    plane_rmsd = s[-1] / np.sqrt(len(points))
    return centroid, normal, plane_rmsd


# ─────────────────────────────────────────────
# 3.  PROJECT TO 2-D PLANE COORDINATES
# ─────────────────────────────────────────────

def build_plane_basis(normal: np.ndarray):
    """Return two orthonormal basis vectors (u, v) that span the plane."""
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def project_to_2d(centered_points: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project centred 3-D points onto the (u, v) plane basis → (N, 2)."""
    return np.column_stack([centered_points @ u, centered_points @ v])


# ─────────────────────────────────────────────
# 4.  ALGEBRAIC CIRCLE FIT IN 2-D
# ─────────────────────────────────────────────

def fit_circle_2d(points_2d: np.ndarray):
    """
    Algebraic least-squares circle fit.
    Circle equation:  x² + y² + Dx + Ey + F = 0
    Centre = (D/2, E/2),  radius = sqrt((D/2)² + (E/2)² - F)

    Returns:
        cx, cy : float  circle centre in 2-D plane coordinates
        radius : float  circle radius (same units as input)
        fit_rmsd: float residual RMS of radial distances from the fitted circle
    """
    x, y = points_2d[:, 0], points_2d[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = result
    cx = D / 2.0
    cy = E / 2.0
    radius = np.sqrt(cx ** 2 + cy ** 2 + F)

    # Residuals: difference between each point's distance to centre and the radius
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    fit_rmsd = np.sqrt(np.mean((distances - radius) ** 2))

    return cx, cy, radius, fit_rmsd


# ─────────────────────────────────────────────
# 5.  BACK-PROJECT CENTRE TO 3-D
# ─────────────────────────────────────────────

def backproject_to_3d(cx_2d, cy_2d, centroid, u, v) -> np.ndarray:
    return centroid + cx_2d * u + cy_2d * v


# ─────────────────────────────────────────────
# 6.  GENERATE CIRCLE POINTS FOR PLOTTING
# ─────────────────────────────────────────────

def circle_3d_points(center_3d, radius, u, v, n=200):
    """Return (n, 3) array of 3-D points tracing the fitted circle."""
    theta = np.linspace(0, 2 * np.pi, n)
    return np.array([
        center_3d + radius * (np.cos(t) * u + np.sin(t) * v)
        for t in theta
    ])


# ─────────────────────────────────────────────
# 7.  PLOT
# ─────────────────────────────────────────────

def plot_results(points, circle_pts, center_3d, normal, centroid, plane_rmsd, radius, fit_rmsd):
    fig = plt.figure(figsize=(14, 6))

    # ── 3-D view ──────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.scatter(*points.T, color="royalblue", s=60, zorder=5, label="Input points")
    ax3d.plot(*circle_pts.T, color="tomato", linewidth=2, label="Fitted circle")
    ax3d.scatter(*center_3d, color="gold", s=120, marker="*", zorder=6, label="Circle centre")

    # Draw plane normal arrow at centroid
    scale = radius * 0.5
    ax3d.quiver(*centroid, *(normal * scale), color="green", linewidth=2, label="Plane normal")

    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("3-D view")
    ax3d.legend(fontsize=8)

    # ── 2-D in-plane view ─────────────────────
    ax2d = fig.add_subplot(122)
    theta = np.linspace(0, 2 * np.pi, 300)

    # Re-derive 2-D projected points for plotting
    centroid_pts = points - centroid
    u, v = build_plane_basis(normal)
    pts2d = project_to_2d(centroid_pts, u, v)
    cx2d = (center_3d - centroid) @ u
    cy2d = (center_3d - centroid) @ v

    ax2d.scatter(pts2d[:, 0], pts2d[:, 1], color="royalblue", s=60, zorder=5, label="Projected points")
    ax2d.plot(cx2d + radius * np.cos(theta), cy2d + radius * np.sin(theta),
              color="tomato", linewidth=2, label="Fitted circle")
    ax2d.plot(cx2d, cy2d, "g+", markersize=14, markeredgewidth=2, label="Centre")

    # Annotate each point with its radial residual
    dists = np.sqrt((pts2d[:, 0] - cx2d) ** 2 + (pts2d[:, 1] - cy2d) ** 2)
    for i, (p, d) in enumerate(zip(pts2d, dists)):
        ax2d.annotate(f"P{i+1}\nΔ={d - radius:.2f}", p, fontsize=7,
                      textcoords="offset points", xytext=(6, 4))

    ax2d.set_aspect("equal")
    ax2d.set_xlabel("u (mm)")
    ax2d.set_ylabel("v (mm)")
    ax2d.set_title("In-plane 2-D view")
    ax2d.legend(fontsize=8)
    ax2d.grid(True, linestyle="--", alpha=0.5)

    # ── Shared title with results ──────────────
    fig.suptitle(
        f"Best-fit circle  |  Radius = {radius:.2f} mm  |  "
        f"Fit RMSD = {fit_rmsd:.3f} mm  |  Plane RMSD = {plane_rmsd:.4f} mm",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "perimeter_tibia_med.fcsv"
    patientName = "P6_pre"
    side = "lat"
    filename = f"perimeter_tibia_{side}.fcsv"
    filepath = os.path.join("data_for_gui", patientName, filename)

    print(f"Loading points from: {filepath}")
    points = load_fcsv(filepath)
    print(f"  → {len(points)} points loaded\n")

    # Plane fit
    centroid, normal, plane_rmsd = fit_plane(points)
    print(f"Plane fit:")
    print(f"  Centroid : {centroid}")
    print(f"  Normal   : {normal}")
    print(f"  RMSD     : {plane_rmsd:.4f} mm  (how well points lie on the plane)\n")

    # 2-D projection
    u, v = build_plane_basis(normal)
    centered = points - centroid
    pts_2d = project_to_2d(centered, u, v)

    # Circle fit
    cx_2d, cy_2d, radius, fit_rmsd = fit_circle_2d(pts_2d)
    center_3d = backproject_to_3d(cx_2d, cy_2d, centroid, u, v)

    print(f"Circle fit:")
    print(f"  Centre (3D) : {center_3d}")
    print(f"  Radius      : {radius:.4f} mm")
    print(f"  Fit RMSD    : {fit_rmsd:.4f} mm  (radial residuals)\n")
    print(f"To copy: {center_3d[0]:.1f},{center_3d[1]:.1f},{center_3d[2]:.1f},0,0,0,1,1,1,0,tibia_{side}")
    
    # Plot
    circle_pts = circle_3d_points(center_3d, radius, u, v)
    # plot_results(points, circle_pts, center_3d, normal, centroid, plane_rmsd, radius, fit_rmsd)


if __name__ == "__main__":
    main()