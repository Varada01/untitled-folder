import numpy as np
from numpy.linalg import lstsq
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List
import matplotlib.pyplot as plt

# ============================================================
# Geometry and optical parameters
# ============================================================
@dataclass
class Geo:
    nx: int = 300
    ny: int = 200
    dx: float = 2e-6
    dy: float = 2e-6
    r_in: Tuple[int, int] = (30, 100)
    r_out: Tuple[int, int] = (270, 100)

@dataclass
class Optics:
    lam_sig: float = 1.55e-6
    neff: float = 3.2
    @property
    def keff(self) -> float:
        return 2 * np.pi * self.neff / self.lam_sig

@dataclass
class Envelope:
    R0: float
    apod_power: float = 1.0

@dataclass
class Segmentation:
    period_m: float = 20e-6
    duty_beta: float = 0.5
    PI: float = 1.0

@dataclass
class SaturableModel:
    Isat: float = 0.2
    alpha_unp: float = 1.2
    alpha_pmp: float = 0.2
    gain_pmp: float = 0.0
    L_eff: float = 0.5e-3

# ============================================================
# Optical field map and segmentation
# ============================================================
def elliptical_map(geo: Geo, opt: Optics, env: Envelope) -> np.ndarray:
    ny, nx = geo.ny, geo.nx
    xs = (np.arange(nx) - geo.r_in[0]) * geo.dx
    ys = (np.arange(ny) - geo.r_in[1]) * geo.dy
    X, Y = np.meshgrid(xs, ys)

    xin, yin = 0.0, 0.0
    xout = (geo.r_out[0] - geo.r_in[0]) * geo.dx
    yout = (geo.r_out[1] - geo.r_in[1]) * geo.dy
    Rin = np.hypot(X - xin, Y - yin)
    Rout = np.hypot(X - xout, Y - yout)
    Rji = Rin + Rout - np.hypot(xout - xin, yout - yin)

    F = np.zeros_like(Rji)
    mask = Rji <= env.R0
    F[mask] = np.cos(opt.keff * Rji[mask])
    F = np.clip(F, 0, None)
    if F.max() > 0:
        F /= F.max()
    return F ** env.apod_power

def segmented_mask_along_line(geo: Geo, seg: Segmentation) -> np.ndarray:
    ny, nx = geo.ny, geo.nx
    x0, y0 = geo.r_in
    x1, y1 = geo.r_out
    dxm, dym = (x1 - x0) * geo.dx, (y1 - y0) * geo.dy
    L = np.hypot(dxm, dym)
    ux, uy = dxm / L, dym / L
    xs = (np.arange(nx) - x0) * geo.dx
    ys = (np.arange(ny) - y0) * geo.dy
    X, Y = np.meshgrid(xs, ys)
    s = X * ux + Y * uy
    s_mod = np.mod(s, seg.period_m)
    mask = (s >= 0) & (s <= L)
    seg_mask = np.zeros_like(s, dtype=float)
    seg_mask[(s_mod / seg.period_m <= seg.duty_beta) & mask] = 1.0
    return seg_mask

def make_pump_pattern(geo, opt, env, seg, coeffs):
    F = elliptical_map(geo, opt, env)
    S = segmented_mask_along_line(geo, seg)
    # Create polynomial shaping function
    P = np.zeros_like(F)
    for k, a in enumerate(coeffs):
        P += a * (F ** k)
    P *= seg.PI * S
    if P.max() > 0:
        P /= P.max()
    return P

# ============================================================
# Nonlinear transfer and optimization
# ============================================================
def transfer_surrogate(I, seg, sat):
    beta = seg.duty_beta
    alpha_p_eff = max(sat.alpha_pmp - 0.1 * seg.PI, 0.0)
    g_p_eff = max(sat.gain_pmp * np.log1p(seg.PI), 0.0)
    alpha0 = beta * (alpha_p_eff - g_p_eff) + (1.0 - beta) * sat.alpha_unp
    alpha_eff = alpha0 / (1.0 + I / sat.Isat)
    return np.exp(-alpha_eff * sat.L_eff) * I

def fit_beta_PI_to_poly(Ical, coeffs, geo, opt, env, sat,
                        beta_grid, PI_grid) -> Tuple[float, float]:
    Otarget = np.zeros_like(Ical)
    for k, a in enumerate(coeffs):
        Otarget += a * (Ical ** k)
    Otarget = np.clip(Otarget, 0.0, None)
    best = (np.inf, None, None)
    for beta in beta_grid:
        for PI in PI_grid:
            seg = Segmentation(duty_beta=float(beta), PI=float(PI))
            Op = transfer_surrogate(Ical, seg, sat)
            M = np.vstack([Op, np.ones_like(Op)]).T
            AB, *_ = lstsq(M, Otarget, rcond=None)
            Oadj = M @ AB
            err = np.mean((Oadj - Otarget) ** 2)
            if err < best[0]:
                best = (err, beta, PI)
    return best[1], best[2]

def synthesize_pump_for_poly(coeffs, Imin, Imax,
                             geo=None, opt=None, env=None, sat=None,
                             period_m=20e-6):
    geo = geo or Geo()
    opt = opt or Optics()

    # FIX: realistic ellipse radius
    if env is None:
        dx_m = (geo.r_out[0] - geo.r_in[0]) * geo.dx
        dy_m = (geo.r_out[1] - geo.r_in[1]) * geo.dy
        R0 = 0.5 * np.hypot(dx_m, dy_m)
        env = Envelope(R0=R0, apod_power=1.0)

    dx_m = (geo.r_out[0] - geo.r_in[0]) * geo.dx
    dy_m = (geo.r_out[1] - geo.r_in[1]) * geo.dy
    sat = sat or SaturableModel(L_eff=np.hypot(dx_m, dy_m))
    Ical = np.linspace(Imin, Imax, 30)
    beta_grid = np.linspace(0.3, 0.95, 12)
    PI_grid = np.linspace(0.2, 2.0, 8)
    beta, PI = fit_beta_PI_to_poly(Ical, coeffs, geo, opt, env, sat, beta_grid, PI_grid)
    seg = Segmentation(period_m=period_m, duty_beta=beta, PI=PI)
    P = make_pump_pattern(geo, opt, env, seg, coeffs)
    return P, seg



# ============================================================
# Multi-connection (network) synthesis
# ============================================================
def synthesize_network(M, N, coeffs):
    """
    Synthesize pump patterns for M outputs and N inputs.
    For CIFAR-10: M=10 (classes), N=128 (reduced features)
    """
    betas = np.zeros((M, N))
    PIs = np.zeros((M, N))
    patterns = np.zeros((M, N, 200, 300))
    geos = [[None]*N for _ in range(M)]

    base_geo = Geo()
    
    # For large N (128 inputs), distribute input ports more strategically
    if N > 20:
        # Use modulo to wrap input ports around available space
        y_in_spacing = base_geo.ny // min(N, 20)
    else:
        y_in_spacing = base_geo.ny // (N + 1)
    
    y_out_spacing = base_geo.ny // (M + 1)

    print(f"Synthesizing {M}x{N} network (outputs x inputs)...")
    for i in range(M):
        if (i + 1) % 2 == 0 or i == 0:
            print(f"  Processing output {i+1}/{M}...")
        for j in range(N):
            # Distinct input and output port positions
            # For large N, wrap y positions
            y_in_pos = ((j % 20) + 1) * y_in_spacing if N > 20 else (j + 1) * y_in_spacing
            y_in_pos = min(y_in_pos, base_geo.ny - 10)  # Keep within bounds
            
            geo = Geo(
                nx=base_geo.nx,
                ny=base_geo.ny,
                dx=base_geo.dx,
                dy=base_geo.dy,
                r_in=(30, y_in_pos),
                r_out=(270, (i + 1) * y_out_spacing)
            )

            P, seg = synthesize_pump_for_poly(coeffs, 0.0, 1.0, geo=geo)
            betas[i, j] = seg.duty_beta
            PIs[i, j] = seg.PI
            patterns[i, j] = P
            geos[i][j] = geo

    print(f"Network synthesis complete!")
    return betas, PIs, patterns, geos


def visualize_network_patterns(patterns, geos, M, N):
    """Show one example pattern per row of the network with input/output ports marked."""
    fig, axes = plt.subplots(M, N, figsize=(3*N, 3*M))
    if M == 1 and N == 1:
        axes = np.array([[axes]])
    elif M == 1 or N == 1:
        axes = axes.reshape(M, N)

    for i in range(M):
        for j in range(N):
            ax = axes[i, j]
            ax.imshow(patterns[i, j], cmap='plasma', origin='lower', aspect='auto')
            geo = geos[i][j]
            ax.scatter(geo.r_in[0], geo.r_in[1], s=30, c='cyan', marker='o', edgecolors='white', label='Input')
            ax.scatter(geo.r_out[0], geo.r_out[1], s=30, c='lime', marker='x', label='Output')
            ax.set_title(f"Conn ({i+1},{j+1}) β={betas[i,j]:.2f}")
            ax.axis('off')

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    plt.suptitle("Pump Patterns with Input (cyan) and Output (green) Ports", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_combined_network(patterns, geos):
    """Overlay all connections in one composite pump map with port markers."""
    full = np.zeros_like(patterns[0, 0])
    for i in range(patterns.shape[0]):
        for j in range(patterns.shape[1]):
            full += patterns[i, j]
    if full.max() > 0:
        full /= full.max()
    else:
        full[:] = 0


    plt.figure(figsize=(10, 6))
    plt.imshow(full, cmap='plasma', origin='lower', aspect='auto')
    for i in range(len(geos)):
        for j in range(len(geos[i])):
            geo = geos[i][j]
            plt.scatter(geo.r_in[0], geo.r_in[1], s=20, c='cyan', marker='o', edgecolors='white')
            plt.scatter(geo.r_out[0], geo.r_out[1], s=20, c='lime', marker='x')
    plt.title("Combined Pump Pattern (All I/O Links)")
    plt.axis('off')
    plt.show()


# ============================================================
# Example Usage for CIFAR-10
# ============================================================
if __name__ == "__main__":
    import os
    import json
    
    # Try to load trained coefficients from CIFAR-10 training
    coeffs_file = "./trained_coeffs_cifar10.npy"
    
    if os.path.exists(coeffs_file):
        print(f"Loading trained coefficients from {coeffs_file}")
        coeffs = np.load(coeffs_file, allow_pickle=True)
        # coeffs shape: (10 outputs, 128 inputs, 4 degrees)
        avg_coeffs = np.mean(coeffs, axis=(0, 1))  # average across outputs and inputs
        print(f"Coefficient shape: {coeffs.shape}")
        print(f"Average coefficients: {avg_coeffs}")
    else:
        print(f"Coefficients file not found: {coeffs_file}")
        print("Using default polynomial coefficients")
        avg_coeffs = np.array([0.0, 0.5, 0.3, 0.2])  # default cubic polynomial
    
    # CIFAR-10: 10 output classes, 128 input features (after PCA reduction)
    M, N = 10, 128
    print(f"\nGenerating pump patterns for CIFAR-10:")
    print(f"  Outputs (classes): {M}")
    print(f"  Inputs (features): {N}")
    print(f"  Total connections: {M * N}")
    
    betas, PIs, patterns, geos = synthesize_network(M, N, avg_coeffs)

    # -----------------------------------------------------------
    # Export pump pattern data for CIFAR-10
    # -----------------------------------------------------------
    export_dir = "./pump_patterns_cifar10"
    os.makedirs(export_dir, exist_ok=True)

    # Save numerical data
    np.savez_compressed(
        os.path.join(export_dir, "pump_patterns.npz"),
        patterns=patterns,
        betas=betas,
        PIs=PIs
    )

    # Save geometry metadata
    geo_info = [
        {"nx": g.nx, "ny": g.ny, "dx": g.dx, "dy": g.dy,
        "r_in": g.r_in, "r_out": g.r_out}
        for row in geos for g in row
    ]
    with open(os.path.join(export_dir, "geo_info.json"), "w") as f:
        json.dump(geo_info, f, indent=2)

    print(f"\n✅ Pump patterns exported to {export_dir}")
    print("   • pump_patterns.npz  (arrays)")
    print("   • geo_info.json      (geometry metadata)")

    # Export preview images for a subset of patterns (not all 1280!)
    img_dir = os.path.join(export_dir, "pump_pattern_images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Save only first 10 patterns as examples
    print("\n   • Saving sample pattern images (first 10 connections)...")
    for i in range(min(M, 2)):  # First 2 outputs
        for j in range(min(N, 5)):  # First 5 inputs
            plt.imsave(
                os.path.join(img_dir, f"pump_{i}_{j}.png"),
                patterns[i, j],
                cmap="inferno"
            )
    print(f"   • Sample images saved to: {img_dir}")

    print("\nβ matrix shape:", betas.shape)
    print("β matrix sample (first 3x5):\n", betas[:3, :5])
    print("\nPI matrix shape:", PIs.shape)
    print("PI matrix sample (first 3x5):\n", PIs[:3, :5])

    # Visualize subset of patterns (not all 1280)
    print("\nGenerating visualization of sample patterns...")
    M_vis, N_vis = 3, 4  # Visualize only 3x4 subset
    visualize_network_patterns(patterns[:M_vis, :N_vis], 
                              [[geos[i][j] for j in range(N_vis)] for i in range(M_vis)], 
                              M_vis, N_vis)

    # Combined full-chip pump layout
    print("Generating combined network visualization...")
    visualize_combined_network(patterns, geos)
    
    print("\n" + "="*60)
    print("CIFAR-10 pump pattern generation complete!")
    print("="*60)