# -*- coding: utf-8 -*-
"""
Self-consistent 1D 4x4 Bogoliubov-de Gennes solver for an inhomogeneous
spin-imbalanced Fermi gas with hard-wall boundary conditions.

This script:
- builds the BdG Hamiltonian on a finite-difference grid
- iterates to self-consistency for the pairing field
- computes densities, pair correlations, and magnetization
- saves results to NPZ format
- saves and displays real-space and momentum-space observables
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


def main():
    # ============================================================
    # Simulation parameters
    # ============================================================
    L = 1.0
    N = 300
    x_grid = np.linspace(0.0, 1.0, N)
    dx = x_grid[1] - x_grid[0]

    kfL0 = np.pi * 50.0        # fixed reference k_F^0 * L
    mu0_tilde = 1.0
    epsilon = 1e-5
    max_iter = 1000
    mix = 0.5

    # Interface parameters (single case)
    case = {"gL": 2.0, "gR": 0.0, "hL": 0.35, "hR": 0.35}

    interaction_profile = np.where(x_grid <= 0.5, case["gL"], case["gR"])
    zeeman_profile = np.where(x_grid <= 0.5, case["hL"], case["hR"])

    mu_up_tilde = mu0_tilde + zeeman_profile
    mu_dn_tilde = mu0_tilde - zeeman_profile

    # ============================================================
    # Build hard-wall Laplacian (Dirichlet boundary conditions)
    # ============================================================
    laplacian = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    kinetic_prefactor = 1.0 / ((kfL0 * dx) ** 2)

    # ============================================================
    # Initialize FFLO-compatible pairing field
    # ============================================================
    gap_profile = 0.2 * np.sin(5 * np.pi * x_grid)
    gap_profile[0] = 0.0
    gap_profile[-1] = 0.0

    # ============================================================
    # Self-consistent BdG iteration
    # ============================================================
    print("Running self-consistent 4N BdG solver...")
    start_time = time.time()

    converged = False
    err = np.inf

    for it in range(max_iter):
        gap_matrix = np.diag(gap_profile.astype(complex))
        zero_block = np.zeros((N, N), dtype=complex)

        H_up = (-kinetic_prefactor * laplacian - np.diag(mu_up_tilde)).astype(complex)
        H_dn = (-kinetic_prefactor * laplacian - np.diag(mu_dn_tilde)).astype(complex)

        bdg_matrix = np.block([
            [H_up,      zero_block, zero_block, gap_matrix],
            [zero_block, H_dn,      gap_matrix, zero_block],
            [zero_block, gap_matrix, -H_up,     zero_block],
            [gap_matrix, zero_block, zero_block, -H_dn]
        ])

        eigenvalues, eigenvectors = eigh(bdg_matrix, check_finite=False)

        positive_mask = eigenvalues > 0
        eigenvalues = eigenvalues[positive_mask]
        eigenvectors = eigenvectors[:, positive_mask]

        # Extract quasiparticle amplitudes
        u_up = eigenvectors[0 * N:1 * N, :]
        u_dn = eigenvectors[1 * N:2 * N, :]
        v_up = eigenvectors[2 * N:3 * N, :]
        v_dn = eigenvectors[3 * N:4 * N, :]

        # Continuum normalization on x-grid
        scale = 1.0 / np.sqrt(dx)
        u_up *= scale
        u_dn *= scale
        v_up *= scale
        v_dn *= scale

        # Pair correlator F_x = L F(x)
        pair_corr = 0.5 * np.sum((u_up * v_dn + u_dn * v_up), axis=1)

        # Gap update
        gap_new = interaction_profile * (pair_corr / kfL0)
        gap_new[0] = 0.0
        gap_new[-1] = 0.0

        # Mixing and convergence
        gap_next = (1 - mix) * gap_profile + mix * gap_new.real
        err = np.linalg.norm(gap_next - gap_profile) / (np.linalg.norm(gap_profile) + 1e-12)
        gap_profile = gap_next

        if it % 50 == 0 or err < epsilon:
            max_gap = np.max(np.abs(gap_profile))
            print(f"Iter {it + 1:4d} | rel err = {err:.2e} | max|Δ~| = {max_gap:.3e}")

        if err < epsilon:
            converged = True
            print("Converged.")
            break

    end_time = time.time()

    if not converged:
        print("Warning: solver reached maximum iterations without full convergence.")

    print(f"Finished in {it + 1} iterations, time = {end_time - start_time:.2f} s")

    # ============================================================
    # Compute observables
    # ============================================================
    # densities: n_{x,σ} = sum_{E>0} |v|^2 in this representation
    n_up = np.sum(np.abs(v_up) ** 2, axis=1)
    n_dn = np.sum(np.abs(v_dn) ** 2, axis=1)

    rho_up = n_up / kfL0
    rho_dn = n_dn / kfL0
    magnetization = rho_up - rho_dn

    rho_up[0] = rho_up[-1] = 0.0
    rho_dn[0] = rho_dn[-1] = 0.0

    corr_tilde = (pair_corr / kfL0).real
    corr_tilde[0] = corr_tilde[-1] = 0.0
    delta_tilde = gap_profile.real

    # Particle numbers
    N_up = np.trapz(n_up, x_grid)
    N_dn = np.trapz(n_dn, x_grid)
    polarization = (N_up - N_dn) / (N_up + N_dn + 1e-15)

    print(f"N_up = {N_up:.3f}, N_dn = {N_dn:.3f}, P = {polarization:.4f}")

    # ============================================================
    # Save output data
    # ============================================================
    output_dir = "bdg_data"
    os.makedirs(output_dir, exist_ok=True)

    tag = (
        f"N{N}_gL{case['gL']:.2f}_gR{case['gR']:.2f}"
        f"_hL{case['hL']:.2f}_hR{case['hR']:.2f}"
    )

    data_path = os.path.join(output_dir, f"profiles_{tag}.npz")

    np.savez(
        data_path,
        x_grid=x_grid,
        delta=delta_tilde,
        corr=corr_tilde,
        rho_up=rho_up,
        rho_dn=rho_dn,
        magnetization=magnetization,
        n_up=n_up,
        n_dn=n_dn,
        evals=eigenvalues,
        interaction=interaction_profile,
        zeeman=zeeman_profile,
        metadata=dict(
            N=N,
            L=L,
            kfL0=kfL0,
            mu0_tilde=mu0_tilde,
            gL=case["gL"],
            gR=case["gR"],
            hL=case["hL"],
            hR=case["hR"],
            N_up=N_up,
            N_dn=N_dn,
            polarization=polarization,
            converged=converged,
            iterations=it + 1,
            rel_error=err,
        )
    )

    print(f"Saved data -> {data_path}")

    # ============================================================
    # Fourier spectrum of pair correlator
    # ============================================================
    modes = np.fft.fftfreq(N, d=dx)                 # cycles per x~
    k_over_kF0 = (2 * np.pi * modes) / kfL0        # k / kF0
    Fk = np.fft.fft(corr_tilde) * dx               # continuum-ish scaling
    k_shift = np.fft.fftshift(k_over_kF0)
    Fk_shift = np.fft.fftshift(Fk)

    # ============================================================
    # Generate plots
    # ============================================================
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    # (a) Pairing field
    axs[0, 0].plot(x_grid, delta_tilde, lw=2)
    axs[0, 0].axvline(0.5, ls="--", color="k", lw=1)
    axs[0, 0].set_title("(a) Pairing field")
    axs[0, 0].set_ylabel(r"$\tilde{\Delta}(\tilde{x})$")
    axs[0, 0].set_xlabel(r"$\tilde{x}=x/L$")
    axs[0, 0].grid(True, alpha=0.3)

    # (b) Pair correlator
    axs[0, 1].plot(x_grid, corr_tilde, lw=2)
    axs[0, 1].axvline(0.5, ls="--", color="k", lw=1)
    axs[0, 1].set_title("(b) Pair correlation")
    axs[0, 1].set_ylabel(r"$\tilde{F}(\tilde{x})$")
    axs[0, 1].set_xlabel(r"$\tilde{x}=x/L$")
    axs[0, 1].grid(True, alpha=0.3)

    # (c) Spin densities
    axs[1, 0].plot(x_grid, rho_up, lw=2, label=r"$\uparrow$")
    axs[1, 0].plot(x_grid, rho_dn, lw=2, ls="--", label=r"$\downarrow$")
    axs[1, 0].axvline(0.5, ls="--", color="k", lw=1)
    axs[1, 0].set_title("(c) Spin densities")
    axs[1, 0].set_xlabel(r"$\tilde{x}=x/L$")
    axs[1, 0].set_ylabel(r"$\tilde{\rho}_\sigma(\tilde{x})$")
    axs[1, 0].legend(frameon=False)
    axs[1, 0].grid(True, alpha=0.3)

    # (d) Momentum-space pair correlations
    axs[1, 1].plot(k_shift, np.abs(Fk_shift), lw=2)
    axs[1, 1].set_title("(d) Momentum-space pair correlations")
    axs[1, 1].set_xlabel(r"$k/k_F^0$")
    axs[1, 1].set_ylabel(r"$|\tilde{F}(k)|$")
    axs[1, 1].grid(True, alpha=0.3)

    summary_plot_path = os.path.join(output_dir, f"summary_{tag}.png")
    fig.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {summary_plot_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(x_grid, magnetization, lw=2)
    plt.axvline(0.5, ls="--", color="k", lw=1)
    plt.xlabel(r"$\tilde{x}=x/L$")
    plt.ylabel(r"$m(\tilde{x})=\tilde\rho_\uparrow-\tilde\rho_\downarrow$")
    plt.title("Local magnetization")
    plt.grid(True, alpha=0.3)

    magnetization_plot_path = os.path.join(output_dir, f"magnetization_{tag}.png")
    plt.savefig(magnetization_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {magnetization_plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
