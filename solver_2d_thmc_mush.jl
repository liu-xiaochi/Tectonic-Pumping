# ==============================================================================
# 2D Staggered-Grid Coupled THMC Two-Phase Flow Solver (ParallelStencil.jl, CPU)
# ------------------------------------------------------------------------------
# Uses the Pseudo-Maxwell Viscoelastic Symplectic PT Stokes-Darcy formulation
# (Räss et al. 2022, GMD) with:
#   - 3D plane-strain deviatoric stresses + log-space viscoplastic relaxation
#   - Smoothly tapered shear tectonic pumping (sigma_pump -> 0 as phi -> phi_min)
#     + zero-mean local Reynolds dilatancy for high-strain channel focusing
#   - Smooth quadratic top Palaeo-BDT valve with weak pressure anchor (no PT shock)
#   - Balanced PT parameters stable across >10^3 vertical viscosity contrasts
#   - Local-porosity CFL stepping + active (t/tc=0.25) & peak-export snapshots
#
# Launch on multi-core Windows CPU:
#   julia -t auto solver_2d_thmc_mush.jl
#
# Outputs (saved in ./output_2d/, directly compatible with Fig. 3 Python script):
#   - snapshots_{static,filter,shear_const,coupled}.npz         (final state)
#   - snapshots_active_{static,filter,shear_const,coupled}.npz  (sync t/tc ~ 0.25)
#   - snapshots_peak_{static,filter,shear_const,coupled}.npz    (peak export flux)
#   - time_series.npz                                           (4-case curves)
#   - phase_diagram_grid.npz                                    (optional sweep)
# ==============================================================================

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using NPZ
using Printf
using Random
using Statistics

# ==============================================================================
# 1. CONFIGURATION STRUCTS (4-SWITCH ABLATION DESIGN)
# ==============================================================================
Base.@kwdef struct ModelParams
    # Domain & grid (96x96 for fast multi-core CPU verification; 192x192 / 256x256 for Fig. 3)
    Lx::Float64 = 2000.0            # Horizontal domain width (m, periodic)
    Lz::Float64 = 2000.0            # Vertical mush thickness (m, z=0 base -> z=Lz Palaeo-BDT)
    nx::Int     = 96
    nz::Int     = 96

    # Thermal & phase equilibrium (shared realistic thermodynamics across all 4 cases)
    T_bg_top::Float64   = 485.0     # Country-rock temperature at Palaeo-BDT (°C)
    T_bg_bot::Float64   = 545.0     # Country-rock temperature at mush base (°C)
    H_roof::Float64     = 1200.0    # Effective roof conductive barrier thickness (m)
    T_init_top::Float64 = 680.0     # Initial top mush T (°C, phi ~ 0.18)
    T_init_bot::Float64 = 708.0     # Initial base mush T (°C, phi ~ 0.35)
    T_liq::Float64      = 795.0     # Liquidus (°C)
    T_sol0::Float64     = 640.0     # Unfractionated solidus (°C)
    T_sol_min::Float64  = 455.0     # Flux-rich pegmatite solidus floor (°C)
    dT_flux::Float64    = 36.0      # Solidus depression slope (°C per unit Cf excess)
    gamma_melt::Float64 = 1.25      # Melt-fraction exponent
    kappa::Float64      = 1.0e-6    # Thermal diffusivity (m^2 / s)
    St::Float64         = 1.0       # Stefan number
    tau_lat_frac::Float64 = 0.60    # Lateral pluton cooling timescale / tau_cool
    tau_cryst::Float64  = 2.0e10    # Crystallization relaxation time (~630 yr)

    # Two-phase permeability & buoyancy
    a_grain::Float64    = 1.5e-3    # Grain radius (1.5 mm)
    n_perm::Float64     = 3.0       # Kozeny-Carman exponent
    delta_rho::Float64  = 300.0     # rho_s - rho_m (kg / m^3)
    g::Float64          = 9.81      # Gravity (m / s^2)
    phi_min::Float64    = 0.015     # Residual locked porosity floor
    phi_max::Float64    = 0.55      # Porosity cap
    phi_crit::Float64   = 0.45      # Disaggregation limit for granular mush

    # Solid rheology, shear tectonic pumping & local Reynolds dilatancy
    eta_s_ref::Float64  = 8.0e18    # Solid shear viscosity at phi_ref (Pa s)
    alpha_phi::Float64  = 23.0      # Porosity-weakening exponent
    tau_yield::Float64  = 35.0e6    # Viscoplastic BDT yield cap (Pa)
    alpha_pump::Float64 = 1.50      # Shear-to-compaction tectonic pumping coupling (sigma_pump / tau_II)
    psi_0::Float64      = 0.15      # Local Reynolds dilatancy coefficient tan(psi) for band focusing
    eps_bg::Float64     = 4.5e-13   # Background simple-shear strain rate (s^-1)
    sigma_sq::Float64   = 10.0e6    # Homogeneous squeezing stress for filter pressing (Pa)

    # Giordano-Bartels exponential melt viscosity law
    eta_m0::Float64     = 6.0e6     # Initial high-Si peraluminous melt viscosity (Pa s)
    eta_min::Float64    = 5.0e2     # Flux-rich pegmatite melt viscosity floor (Pa s)
    lambda_flux::Float64 = 3.5      # Exponential depolymerization coefficient

    # Geochemical Rayleigh fractional crystallization & shallow BDT trap
    Li_0::Float64       = 120.0     # Initial granitic melt Li (ppm)
    D_Li::Float64       = 0.12      # Bulk partition coefficient for Li
    D_flux::Float64     = 0.05      # Bulk partition coefficient for flux (H2O+B+F)
    D_disp::Float64     = 1.0e-9    # Pore-scale chemical dispersion (m^2 / s)
    f_peg_res::Float64  = 0.08      # Residual melt fraction in shallow BDT pegmatite dikes
    V_peg_crit::Float64 = 12.0      # Critical exported melt batch thickness (m)

    # Time integration & PT solver controls
    max_t_norm::Float64    = 1.20   # End time in units of tau_cool
    dt_max_frac::Float64   = 0.004  # Max physical step / tau_cool
    max_steps::Int         = 1500   # Max physical time steps (allows fine CFL during melt bursts)
    pt_max_iter::Int       = 12000  # Max PT iterations per physical step
    pt_check_every::Int    = 50     # Residual check stride
    pt_tol::Float64        = 3.0e-4 # Relative PT convergence tolerance
    noise_amp::Float64     = 0.020  # Initial smoothed porosity perturbation amplitude
    t_active_snap::Float64 = 0.25   # Synchronous dimensionless time for active-mush snapshot
end

struct CaseConfig
    name::String
    use_shear::Bool            # Switch 1: Simple shear BCs (top/bottom velocity)
    use_squeezing::Bool        # Switch 2: Homogeneous tectonic squeezing (filter pressing)
    use_dilatancy::Bool        # Switch 3: Shear tectonic pumping + local dilatant channeling
    use_flux_solidus::Bool     # Switch 4a: Thermodynamic flux-depressed solidus (ON for all 4)
    use_visc_feedback::Bool    # Switch 4b: Giordano-Bartels exponential viscosity collapse
end

# ==============================================================================
# 2. PARALLEL STENCIL KERNELS (ZERO RACE CONDITIONS, PERIODIC X)
# ==============================================================================

# 2.1 Update dimensionless viscosity, bulk viscosity, permeability & melt mobility
@parallel_indices (ix, iz) function update_material_props!(
    eta_s, zeta, mob_c, eta_m_phys,
    phi, T, Cf,
    phi_ref, k_ref, alpha_phi, eta_m0, eta_min, lambda_flux,
    M0, use_visc_fb::Int, nx, nz
)
    if ix <= nx && iz <= nz
        p = phi[ix, iz]
        # Unified porosity-softening solid shear viscosity (capped at ~4 orders of magnitude)
        es_nd = clamp(exp(-alpha_phi * (p - phi_ref)), 1.0e-2, 3.0e2)
        eta_s[ix, iz] = es_nd
        zeta[ix, iz]  = (7.0 / 3.0) * es_nd / max(p, 0.02)

        # Kozeny-Carman permeability
        p_safe = clamp(p, 5.0e-4, 0.90)
        k_val  = k_ref * (p_safe / phi_ref)^3 * ((1.0 - phi_ref) / (1.0 - p_safe))^2

        # Giordano-Bartels exponential flux-viscosity law
        if use_visc_fb == 1
            t_exp = 0.004 * clamp(690.0 - T[ix, iz], -40.0, 220.0)
            f_exp = -lambda_flux * max(Cf[ix, iz] - 1.0, 0.0)
            em    = eta_min + (eta_m0 - eta_min) * exp(clamp(t_exp + f_exp, -25.0, 10.0))
            em    = clamp(em, eta_min, 5.0e8)
        else
            em = eta_m0
        end
        eta_m_phys[ix, iz] = em
        mob_c[ix, iz] = (k_val / em) / M0
    end
    return
end

# 2.2 Compute harmonic face mobilities (with smooth top BDT valve taper) & vertex viscosities
@parallel_indices (ix, iz) function update_face_and_vertex_props!(
    mob_x, mob_z, eta_sv,
    mob_c, eta_s, phi, phi_min, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1

    # X-faces (left face of cell ix, iz)
    if ix <= nx && iz <= nz
        mL = mob_c[ixm, iz]
        mR = mob_c[ix, iz]
        mob_x[ix, iz] = 2.0 * mL * mR / (mL + mR + 1.0e-30)
    end

    # Z-faces & vertices (1 <= iz <= nz+1)
    if ix <= nx && iz <= nz + 1
        if iz == 1
            mob_z[ix, 1]  = 0.0  # Impermeable base
            e1 = eta_s[ixm, 1]; e2 = eta_s[ix, 1]
            eta_sv[ix, 1] = 2.0 * e1 * e2 / (e1 + e2 + 1.0e-30)
        elseif iz == nz + 1
            # Smooth quadratic Palaeo-BDT valve sealing toward percolation floor phi_min
            # (keeps a weak 1e-5 Dirichlet anchor to avoid elliptic pressure shock)
            valve_fac         = clamp((phi[ix, nz] - phi_min) / 0.025, 0.0, 1.0)^2
            mob_z[ix, nz + 1] = mob_c[ix, nz] * max(valve_fac, 1.0e-5)
            e1 = eta_s[ixm, nz]; e2 = eta_s[ix, nz]
            eta_sv[ix, nz + 1] = 2.0 * e1 * e2 / (e1 + e2 + 1.0e-30)
        else
            mB = mob_c[ix, iz - 1]
            mT = mob_c[ix, iz]
            mob_z[ix, iz] = 2.0 * mB * mT / (mB + mT + 1.0e-30)

            e1 = eta_s[ixm, iz - 1]; e2 = eta_s[ix, iz - 1]
            e3 = eta_s[ixm, iz];     e4 = eta_s[ix, iz]
            eta_sv[ix, iz] = 4.0 / (1.0/e1 + 1.0/e2 + 1.0/e3 + 1.0/e4 + 1.0e-30)
        end
    end
    return
end

# 2.3 Pseudo-Maxwell relaxation of vertex shear stress txz
@parallel_indices (ix, iz) function update_vertex_shear_pt!(
    exz, txz, eta_sv_eff, Res_txz,
    Vsx, Vsz, eta_sv,
    dx, dz, U_bot, U_top, use_shear::Int, tau_y_nd, theta_m, rel_eta, init_stress::Int, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1

    if ix <= nx && iz <= nz + 1
        if iz == 1
            if use_shear == 1
                dvx_dz  = (Vsx[ix, 1] - U_bot) / (0.5 * dz)
                exz_v   = 0.5 * dvx_dz
                esv     = eta_sv[ix, 1]
                esv_vp  = 1.0 / (1.0 / esv + 2.0 * abs(exz_v) / tau_y_nd)
                esv_old = eta_sv_eff[ix, 1]
                esv_new = (rel_eta >= 1.0) ? esv_vp :
                          exp((1.0 - rel_eta) * log(max(esv_old, 1.0e-12)) + rel_eta * log(max(esv_vp, 1.0e-12)))
                txz_eq  = 2.0 * esv_new * exz_v
                eta_sv_eff[ix, 1] = esv_new
                exz[ix, 1] = exz_v
                if init_stress == 1
                    txz[ix, 1]     = txz_eq
                    Res_txz[ix, 1] = 0.0
                else
                    Res_txz[ix, 1] = txz_eq - txz[ix, 1]
                    txz[ix, 1]     = (theta_m * txz[ix, 1] + txz_eq) / (theta_m + 1.0)
                end
            else
                eta_sv_eff[ix, 1] = eta_sv[ix, 1]
                exz[ix, 1]        = 0.0
                txz[ix, 1]        = 0.0
                Res_txz[ix, 1]    = 0.0
            end
        elseif iz == nz + 1
            if use_shear == 1
                dvx_dz  = (U_top - Vsx[ix, nz]) / (0.5 * dz)
                dvz_dx  = (Vsz[ix, nz + 1] - Vsz[ixm, nz + 1]) / dx
                exz_v   = 0.5 * (dvx_dz + dvz_dx)
                esv     = eta_sv[ix, nz + 1]
                esv_vp  = 1.0 / (1.0 / esv + 2.0 * abs(exz_v) / tau_y_nd)
                esv_old = eta_sv_eff[ix, nz + 1]
                esv_new = (rel_eta >= 1.0) ? esv_vp :
                          exp((1.0 - rel_eta) * log(max(esv_old, 1.0e-12)) + rel_eta * log(max(esv_vp, 1.0e-12)))
                txz_eq  = 2.0 * esv_new * exz_v
                eta_sv_eff[ix, nz + 1] = esv_new
                exz[ix, nz + 1] = exz_v
                if init_stress == 1
                    txz[ix, nz + 1]     = txz_eq
                    Res_txz[ix, nz + 1] = 0.0
                else
                    Res_txz[ix, nz + 1] = txz_eq - txz[ix, nz + 1]
                    txz[ix, nz + 1]     = (theta_m * txz[ix, nz + 1] + txz_eq) / (theta_m + 1.0)
                end
            else
                eta_sv_eff[ix, nz + 1] = eta_sv[ix, nz + 1]
                exz[ix, nz + 1]        = 0.0
                txz[ix, nz + 1]        = 0.0
                Res_txz[ix, nz + 1]    = 0.0
            end
        else
            dvx_dz  = (Vsx[ix, iz] - Vsx[ix, iz - 1]) / dz
            dvz_dx  = (Vsz[ix, iz] - Vsz[ixm, iz]) / dx
            exz_v   = 0.5 * (dvx_dz + dvz_dx)
            esv     = eta_sv[ix, iz]
            esv_vp  = 1.0 / (1.0 / esv + 2.0 * abs(exz_v) / tau_y_nd)
            esv_old = eta_sv_eff[ix, iz]
            esv_new = (rel_eta >= 1.0) ? esv_vp :
                      exp((1.0 - rel_eta) * log(max(esv_old, 1.0e-12)) + rel_eta * log(max(esv_vp, 1.0e-12)))
            txz_eq  = 2.0 * esv_new * exz_v
            eta_sv_eff[ix, iz] = esv_new
            exz[ix, iz] = exz_v
            if init_stress == 1
                txz[ix, iz]     = txz_eq
                Res_txz[ix, iz] = 0.0
            else
                Res_txz[ix, iz] = txz_eq - txz[ix, iz]
                txz[ix, iz]     = (theta_m * txz[ix, iz] + txz_eq) / (theta_m + 1.0)
            end
        end
    end
    return
end

# 2.4 Pseudo-Maxwell relaxation of 3D plane-strain normal deviatoric stresses
@parallel_indices (ix, iz) function update_cell_stresses_pt!(
    txx, tzz, div_Vs, eII, eta_eff, Res_txx, Res_tzz,
    Vsx, Vsz, exz, eta_s,
    dx, dz, tau_y_nd, theta_m, rel_eta, init_stress::Int, nx, nz
)
    if ix <= nx && iz <= nz
        ixp = (ix == nx) ? 1 : ix + 1
        dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) / dx
        dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) / dz
        div_v  = dvx_dx + dvz_dz

        # 3D plane-strain deviatoric strain rates (eyy = 0)
        exx_d = dvx_dx - (1.0 / 3.0) * div_v
        ezz_d = dvz_dz - (1.0 / 3.0) * div_v
        eyy_d = -(1.0 / 3.0) * div_v

        exz_c = 0.25 * (exz[ix, iz] + exz[ixp, iz] + exz[ix, iz + 1] + exz[ixp, iz + 1])
        e2    = sqrt(0.5 * (exx_d * exx_d + ezz_d * ezz_d + eyy_d * eyy_d) + exz_c * exz_c + 1.0e-30)

        es_loc = eta_s[ix, iz]
        es_vp  = 1.0 / (1.0 / es_loc + 2.0 * e2 / tau_y_nd)

        es_old = eta_eff[ix, iz]
        es_new = (rel_eta >= 1.0) ? es_vp :
                 exp((1.0 - rel_eta) * log(max(es_old, 1.0e-12)) + rel_eta * log(max(es_vp, 1.0e-12)))

        txx_eq = 2.0 * es_new * exx_d
        tzz_eq = 2.0 * es_new * ezz_d

        div_Vs[ix, iz]  = div_v
        eII[ix, iz]     = e2
        eta_eff[ix, iz] = es_new

        if init_stress == 1
            txx[ix, iz]     = txx_eq
            tzz[ix, iz]     = tzz_eq
            Res_txx[ix, iz] = 0.0
            Res_tzz[ix, iz] = 0.0
        else
            Res_txx[ix, iz] = txx_eq - txx[ix, iz]
            Res_tzz[ix, iz] = tzz_eq - tzz[ix, iz]
            txx[ix, iz]     = (theta_m * txx[ix, iz] + txx_eq) / (theta_m + 1.0)
            tzz[ix, iz]     = (theta_m * tzz[ix, iz] + tzz_eq) / (theta_m + 1.0)
        end
    end
    return
end

# 2.4b Horizontal mean strain rate per depth (for zero-mean local dilatant channeling)
@parallel_indices (iz,) function compute_eII_mean_z!(eII_mean_z, eII, nx, nz)
    if iz <= nz
        s = 0.0
        for ix in 1:nx
            s += eII[ix, iz]
        end
        eII_mean_z[iz] = s / nx
    end
    return
end

# 2.5 Update Darcy fluxes qDx, qDz from excess melt pressure Pf (qD = -Rc2 * M * grad(Pf))
@parallel_indices (ix, iz) function update_darcy_flux!(
    qDx, qDz,
    Pf, mob_x, mob_z,
    dx, dz, Rc2, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1

    # X-faces: qDx = -Rc2 * M_x * dPf/dx
    if ix <= nx && iz <= nz
        qDx[ix, iz] = -Rc2 * mob_x[ix, iz] * (Pf[ix, iz] - Pf[ixm, iz]) / dx
    end

    # Z-faces
    if ix <= nx && iz <= nz + 1
        if iz == 1
            qDz[ix, 1] = 0.0  # Impermeable base
        elseif iz == nz + 1
            # Smoothly tapered top Palaeo-BDT one-way valve (no hard phi step cutoff)
            if Pf[ix, nz] > 0.0
                qDz[ix, nz + 1] = Rc2 * mob_z[ix, nz + 1] * (Pf[ix, nz] / (0.5 * dz))
            else
                qDz[ix, nz + 1] = 0.0
            end
        else
            qDz[ix, iz] = -Rc2 * mob_z[ix, iz] * (Pf[ix, iz] - Pf[ix, iz - 1]) / dz
        end
    end
    return
end

# 2.6 Pseudo-elastic Pt update (implicit compaction + tapered shear pumping) & damped-wave Pf
@parallel_indices (ix, iz) function update_pressures_pt!(
    Pt, Pf, dPf, Res_Pt, Res_Pf, eps_dil,
    div_Vs, qDx, qDz, zeta, eta_eff, eII, eII_mean_z, phi, mob_x, mob_z,
    dx, dz, Rc2, r_bulk, theta_m, gamma_f, sigma_sq_nd,
    use_dil::Int, alpha_pump, psi_0, phi_ref, phi_min, phi_crit, nx, nz
)
    if ix <= nx && iz <= nz
        ixp = (ix == nx) ? 1 : ix + 1

        div_qD = (qDx[ixp, iz] - qDx[ix, iz]) / dx +
                 (qDz[ix, iz + 1] - qDz[ix, iz]) / dz

        pc_loc = Pt[ix, iz] - Pf[ix, iz]

        if use_dil == 1
            p          = phi[ix, iz]
            disagg     = max(0.0, 1.0 - p / phi_crit)^1.5
            # Smooth solidification cutoff: locked sub-percolation rock generates no fluid overpressure
            lock_taper = clamp((p - phi_min) / 0.03, 0.0, 1.0)
            tau_II     = 2.0 * eta_eff[ix, iz] * eII[ix, iz]

            # 1) Shear-enhanced tectonic compaction pumping (drives net upward export)
            sig_pump = alpha_pump * lock_taper * disagg * tau_II

            # 2) Zero-mean local Reynolds dilatancy (focuses melt laterally into high-strain bands)
            phi_scale   = min(1.0, p / phi_ref)
            eps_dil_loc = psi_0 * lock_taper * phi_scale * disagg * (eII[ix, iz] - eII_mean_z[iz])
        else
            sig_pump    = 0.0
            eps_dil_loc = 0.0
        end
        eps_dil[ix, iz] = eps_dil_loc

        comp_term = (pc_loc + sigma_sq_nd + sig_pump) / zeta[ix, iz] - eps_dil_loc

        # Eq 1 (Pt): Pseudo-elastic bulk modulus K_loc = r_bulk * eta_eff / theta_m
        # with implicit treatment of Pt / zeta for unconditional stability
        r_pt    = -(div_Vs[ix, iz] + comp_term)
        K_loc   = r_bulk * eta_eff[ix, iz] / theta_m
        dtau_pt = K_loc / (1.0 + K_loc / zeta[ix, iz])
        Pt[ix, iz]    += dtau_pt * r_pt
        Res_Pt[ix, iz] = r_pt

        # Eq 2 (Pf): 2nd-order damped-wave elliptic update with local Gershgorin bound
        r_pf    = -div_qD + comp_term
        m_z_hi  = (iz == nz) ? (2.0 * mob_z[ix, nz + 1]) : mob_z[ix, iz + 1]
        rho_loc = 2.0 * Rc2 * ((mob_x[ix, iz] + mob_x[ixp, iz]) / (dx * dx) +
                               (mob_z[ix, iz] + m_z_hi) / (dz * dz)) +
                  1.0 / zeta[ix, iz]
        dtau_pf = 1.20 / (rho_loc + 1.0e-30)

        dpf = (1.0 - gamma_f) * dPf[ix, iz] + dtau_pf * r_pf
        dPf[ix, iz]    = dpf
        Pf[ix, iz]    += dpf
        Res_Pf[ix, iz] = r_pf
    end
    return
end

# 2.7 Symplectic 1st-order pseudo-elastodynamic velocity updates (uniform P-wave CFL)
@parallel_indices (ix, iz) function update_velocities_pt!(
    Vsx, Vsz, Res_vx, Res_vz,
    txx, tzz, txz, Pt, eta_eff, eta_sv_eff, phi,
    dx, dz, r_bulk, theta_m, cfl_v, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1
    ixp = (ix == nx) ? 1 : ix + 1

    h     = min(dx, dz)
    coeff = (theta_m * cfl_v * cfl_v * h * h) / (r_bulk + 4.0 / 3.0)

    # X-velocity update at (ix, iz)
    if ix <= nx && iz <= nz
        rvx = (txx[ix, iz] - txx[ixm, iz]) / dx +
              (txz[ix, iz + 1] - txz[ix, iz]) / dz -
              (Pt[ix, iz] - Pt[ixm, iz]) / dx

        emax    = max(eta_eff[ixm, iz], eta_eff[ix, iz],
                      eta_sv_eff[ix, iz], eta_sv_eff[ix, iz + 1], 1.0e-6)
        fac     = (iz == 1 || iz == nz) ? 1.5 : 1.0
        dtau_vx = coeff / (fac * emax)

        Vsx[ix, iz]   += dtau_vx * rvx
        Res_vx[ix, iz] = rvx
    end

    # Z-velocity update at interior faces (2 <= iz <= nz)
    if ix <= nx && iz >= 2 && iz <= nz
        phi_f = 0.5 * (phi[ix, iz - 1] + phi[ix, iz])
        rvz = (tzz[ix, iz] - tzz[ix, iz - 1]) / dz +
              (txz[ixp, iz] - txz[ix, iz]) / dx -
              (Pt[ix, iz] - Pt[ix, iz - 1]) / dz -
              (1.0 - phi_f)  # Dimensionless buoyancy -(1 - phi)

        emax    = max(eta_eff[ix, iz - 1], eta_eff[ix, iz],
                      eta_sv_eff[ix, iz], eta_sv_eff[ixp, iz], 1.0e-6)
        dtau_vz = coeff / emax

        Vsz[ix, iz]   += dtau_vz * rvz
        Res_vz[ix, iz] = rvz
    end
    return
end

# 2.8 Symplectic stress-free top BC (sigma_zz = tzz - Pt = 0 at z = Lz) & rigid base (z = 0)
@parallel_indices (ix,) function apply_vsz_bcs!(
    Vsz, Res_vz, Pt, tzz, txz, eta_eff, eta_sv_eff, phi,
    dx, dz, r_bulk, theta_m, cfl_v, nx, nz
)
    if ix <= nx
        ixp = (ix == nx) ? 1 : ix + 1

        Vsz[ix, 1]    = 0.0
        Res_vz[ix, 1] = 0.0

        # Momentum balance over top half-cell [Lz - 0.5*dz, Lz] with sigma_zz(Lz) = 0
        Pt_face_top = Pt[ix, nz] - 0.5 * dz * (1.0 - phi[ix, nz])
        rvz_top     = (Pt_face_top - tzz[ix, nz]) / (0.5 * dz) +
                      (txz[ixp, nz + 1] - txz[ix, nz + 1]) / dx

        h           = min(dx, dz)
        emax_top    = max(eta_eff[ix, nz], eta_sv_eff[ix, nz], eta_sv_eff[ixp, nz],
                          eta_sv_eff[ix, nz + 1], eta_sv_eff[ixp, nz + 1], 1.0e-6)
        dtau_vz_top = (theta_m * cfl_v * cfl_v * h * h) / (2.0 * (r_bulk + 4.0 / 3.0) * emax_top)

        Vsz[ix, nz + 1]   += dtau_vz_top * rvz_top
        Res_vz[ix, nz + 1] = rvz_top
    end
    return
end

# 2.9 Conservative face-flux computation (exact F_melt + F_solid = Vs + qD)
@parallel_indices (ix, iz) function compute_phase_fluxes!(
    Fm_x, Fs_x, Fm_z, Fs_z,
    Vsx, Vsz, qDx, qDz, phi, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1

    if ix <= nx && iz <= nz
        vs = Vsx[ix, iz]
        p_up = (vs >= 0.0) ? phi[ixm, iz] : phi[ix, iz]
        Fs_x[ix, iz] = (1.0 - p_up) * vs
        Fm_x[ix, iz] = p_up * vs + qDx[ix, iz]
    end

    if ix <= nx && iz <= nz + 1
        if iz == 1
            Fs_z[ix, 1] = 0.0
            Fm_z[ix, 1] = 0.0
        elseif iz == nz + 1
            # Top BDT outlet: solid subsides with zero melt inflow; melt escapes via qDz > 0
            Fs_z[ix, nz + 1] = (1.0 - phi[ix, nz]) * Vsz[ix, nz + 1]
            Fm_z[ix, nz + 1] = max(0.0, qDz[ix, nz + 1])
        else
            vs = Vsz[ix, iz]
            p_up = (vs >= 0.0) ? phi[ix, iz - 1] : phi[ix, iz]
            Fs_z[ix, iz] = (1.0 - p_up) * vs
            Fm_z[ix, iz] = p_up * vs + qDz[ix, iz]
        end
    end
    return
end

# 2.9b Local-porosity CFL advection speed per cell (avoids over-conservative / phi_min scaling)
@parallel_indices (ix, iz) function compute_local_cfl_speed!(
    v_cfl, Vsx, Vsz, Fm_x, Fm_z, phi, phi_min, nx, nz
)
    if ix <= nx && iz <= nz
        ixm = (ix == 1) ? nx : ix - 1
        ixp = (ix == nx) ? 1 : ix + 1
        pc  = max(phi[ix, iz], phi_min)

        pxL = max(0.5 * (phi[ixm, iz] + phi[ix, iz]), phi_min)
        pxR = max(0.5 * (phi[ix, iz]  + phi[ixp, iz]), phi_min)
        pzB = (iz == 1)  ? pc : max(0.5 * (phi[ix, iz - 1] + phi[ix, iz]), phi_min)
        pzT = (iz == nz) ? pc : max(0.5 * (phi[ix, iz] + phi[ix, iz + 1]), phi_min)

        vm_max = max(
            abs(Fm_x[ix, iz])     / pxL,
            abs(Fm_x[ixp, iz])    / pxR,
            abs(Fm_z[ix, iz])     / pzB,
            abs(Fm_z[ix, iz + 1]) / pzT
        )
        vs_max = max(
            abs(Vsx[ix, iz]), abs(Vsx[ixp, iz]),
            abs(Vsz[ix, iz]), abs(Vsz[ix, iz + 1])
        )
        v_cfl[ix, iz] = max(vm_max, vs_max)
    end
    return
end

# 2.10 Strict upwind conservative advection for phi, melt solutes, and solid solutes
@parallel_indices (ix, iz) function advect_phi_and_solutes!(
    phi_new, C_Li_new, Cf_new, Cs_Li_new, Cs_f_new,
    phi, C_Li, Cf, Cs_Li, Cs_f,
    Fm_x, Fs_x, Fm_z, Fs_z,
    dt_nd, dx, dz, D_disp_nd, phi_min, phi_max, nx, nz
)
    if ix <= nx && iz <= nz
        ixm = (ix == 1) ? nx : ix - 1
        ixp = (ix == nx) ? 1 : ix + 1

        # Melt volume update
        div_Fm = (Fm_x[ixp, iz] - Fm_x[ix, iz]) / dx +
                 (Fm_z[ix, iz + 1] - Fm_z[ix, iz]) / dz
        p_old = phi[ix, iz]
        p_adv = clamp(p_old - dt_nd * div_Fm, phi_min, phi_max)

        # Upwind melt solute fluxes + chemical dispersion (X-faces)
        fmL = Fm_x[ix, iz]; fmR = Fm_x[ixp, iz]
        p_avgL = 0.5 * (phi[ixm, iz] + p_old)
        p_avgR = 0.5 * (p_old + phi[ixp, iz])

        fLi_xL = fmL * ((fmL >= 0.0) ? C_Li[ixm, iz] : C_Li[ix, iz]) -
                 p_avgL * D_disp_nd * (C_Li[ix, iz] - C_Li[ixm, iz]) / dx
        fLi_xR = fmR * ((fmR >= 0.0) ? C_Li[ix, iz] : C_Li[ixp, iz]) -
                 p_avgR * D_disp_nd * (C_Li[ixp, iz] - C_Li[ix, iz]) / dx

        ff_xL  = fmL * ((fmL >= 0.0) ? Cf[ixm, iz] : Cf[ix, iz]) -
                 p_avgL * D_disp_nd * (Cf[ix, iz] - Cf[ixm, iz]) / dx
        ff_xR  = fmR * ((fmR >= 0.0) ? Cf[ix, iz] : Cf[ixp, iz]) -
                 p_avgR * D_disp_nd * (Cf[ixp, iz] - Cf[ix, iz]) / dx

        # Upwind melt solute fluxes + chemical dispersion (Z-faces)
        fmB = Fm_z[ix, iz]; fmT = Fm_z[ix, iz + 1]
        if iz == 1
            fLi_zB = 0.0; ff_zB = 0.0
        else
            p_avgB = 0.5 * (phi[ix, iz - 1] + p_old)
            fLi_zB = fmB * ((fmB >= 0.0) ? C_Li[ix, iz - 1] : C_Li[ix, iz]) -
                     p_avgB * D_disp_nd * (C_Li[ix, iz] - C_Li[ix, iz - 1]) / dz
            ff_zB  = fmB * ((fmB >= 0.0) ? Cf[ix, iz - 1] : Cf[ix, iz]) -
                     p_avgB * D_disp_nd * (Cf[ix, iz] - Cf[ix, iz - 1]) / dz
        end

        if iz == nz
            fLi_zT = fmT * C_Li[ix, nz]  # Outflow into BDT trap
            ff_zT  = fmT * Cf[ix, nz]
        else
            p_avgT = 0.5 * (p_old + phi[ix, iz + 1])
            fLi_zT = fmT * ((fmT >= 0.0) ? C_Li[ix, iz] : C_Li[ix, iz + 1]) -
                     p_avgT * D_disp_nd * (C_Li[ix, iz + 1] - C_Li[ix, iz]) / dz
            ff_zT  = fmT * ((fmT >= 0.0) ? Cf[ix, iz] : Cf[ix, iz + 1]) -
                     p_avgT * D_disp_nd * (Cf[ix, iz + 1] - Cf[ix, iz]) / dz
        end

        B_Li = max(p_old * C_Li[ix, iz] - dt_nd * ((fLi_xR - fLi_xL)/dx + (fLi_zT - fLi_zB)/dz), 1.0e-4)
        B_f  = max(p_old * Cf[ix, iz]   - dt_nd * ((ff_xR  - ff_xL )/dx + (ff_zT  - ff_zB )/dz), 1.0e-4)

        # Solid solute advection
        fsL = Fs_x[ix, iz]; fsR = Fs_x[ixp, iz]
        fsB = Fs_z[ix, iz]; fsT = Fs_z[ix, iz + 1]
        sLi_xL = fsL * ((fsL >= 0.0) ? Cs_Li[ixm, iz] : Cs_Li[ix, iz])
        sLi_xR = fsR * ((fsR >= 0.0) ? Cs_Li[ix, iz]  : Cs_Li[ixp, iz])
        sf_xL  = fsL * ((fsL >= 0.0) ? Cs_f[ixm, iz]  : Cs_f[ix, iz])
        sf_xR  = fsR * ((fsR >= 0.0) ? Cs_f[ix, iz]   : Cs_f[ixp, iz])

        sLi_zB = (iz == 1)  ? 0.0 : fsB * ((fsB >= 0.0) ? Cs_Li[ix, iz - 1] : Cs_Li[ix, iz])
        sLi_zT = (iz == nz) ? (fsT * Cs_Li[ix, nz]) : fsT * ((fsT >= 0.0) ? Cs_Li[ix, iz] : Cs_Li[ix, iz + 1])
        sf_zB  = (iz == 1)  ? 0.0 : fsB * ((fsB >= 0.0) ? Cs_f[ix, iz - 1]  : Cs_f[ix, iz])
        sf_zT  = (iz == nz) ? (fsT * Cs_f[ix, nz])  : fsT * ((fsT >= 0.0) ? Cs_f[ix, iz]  : Cs_f[ix, iz + 1])

        S_Li = max((1.0 - p_old) * Cs_Li[ix, iz] - dt_nd * ((sLi_xR - sLi_xL)/dx + (sLi_zT - sLi_zB)/dz), 1.0e-4)
        S_f  = max((1.0 - p_old) * Cs_f[ix, iz]  - dt_nd * ((sf_xR  - sf_xL )/dx + (sf_zT  - sf_zB )/dz), 1.0e-4)

        phi_new[ix, iz]   = p_adv
        C_Li_new[ix, iz]  = clamp(B_Li / p_adv, 10.0, 2.5e4)
        Cf_new[ix, iz]    = clamp(B_f  / p_adv, 0.2, 40.0)
        Cs_Li_new[ix, iz] = S_Li / max(1.0 - p_adv, 0.1)
        Cs_f_new[ix, iz]  = S_f  / max(1.0 - p_adv, 0.1)
    end
    return
end

# 2.11 Explicit sub-cycled 2D thermal diffusion + Robin top roof + lateral pluton cooling
@parallel_indices (ix, iz) function thermal_substep!(
    T_new, T, T_bg_z,
    dt_sub, dx_phys, dz_phys, kappa_eff, H_roof, T_bg_top, tau_lat_s, nx, nz
)
    if ix <= nx && iz <= nz
        ixm = (ix == 1) ? nx : ix - 1
        ixp = (ix == nx) ? 1 : ix + 1
        Tc  = T[ix, iz]

        lap_x = (T[ixm, iz] - 2.0 * Tc + T[ixp, iz]) / (dx_phys * dx_phys)
        if iz == 1
            lap_z = (T[ix, 2] - Tc) / (dz_phys * dz_phys)
        elseif iz == nz
            q_top = (Tc - T_bg_top) / H_roof
            lap_z = ((T[ix, nz - 1] - Tc) / dz_phys - q_top) / dz_phys
        else
            lap_z = (T[ix, iz - 1] - 2.0 * Tc + T[ix, iz + 1]) / (dz_phys * dz_phys)
        end

        T_new[ix, iz] = Tc + dt_sub * (kappa_eff * (lap_x + lap_z) - (Tc - T_bg_z[iz]) / tau_lat_s)
    end
    return
end

# 2.12 Exact analytical Rayleigh fractional crystallization substep
@parallel_indices (ix, iz) function rayleigh_crystallization_step!(
    phi, C_Li, Cf, Cs_Li, Cs_f, T,
    dt_phys, tau_cryst, T_liq, T_sol0, T_sol_min, dT_flux, gamma_melt,
    D_Li, D_flux, phi_min, phi_max, use_flux_sol::Int, nx, nz
)
    if ix <= nx && iz <= nz
        p_adv  = phi[ix, iz]
        cf_pre = Cf[ix, iz]
        li_pre = C_Li[ix, iz]

        if use_flux_sol == 1
            T_sol = max(T_sol_min, T_sol0 - dT_flux * max(cf_pre - 1.0, 0.0))
        else
            T_sol = T_sol0
        end
        theta  = clamp((T[ix, iz] - T_sol) / (T_liq - T_sol), 0.0, 1.0)
        phi_eq = theta^gamma_melt

        relax      = 1.0 - exp(-dt_phys / tau_cryst)
        dphi_cryst = max(0.0, p_adv - phi_eq) * relax
        p_new      = clamp(p_adv - dphi_cryst, phi_min, phi_max)

        ratio  = min(1.0, p_new / p_adv)
        li_new = clamp(li_pre * (ratio^(D_Li - 1.0)), 10.0, 2.5e4)
        cf_new = clamp(cf_pre * (ratio^(D_flux - 1.0)), 0.2, 40.0)

        d_sol_Li = max(0.0, p_adv * li_pre - p_new * li_new)
        d_sol_f  = max(0.0, p_adv * cf_pre - p_new * cf_new)

        Cs_Li[ix, iz] = ((1.0 - p_adv) * Cs_Li[ix, iz] + d_sol_Li) / max(1.0 - p_new, 0.1)
        Cs_f[ix, iz]  = ((1.0 - p_adv) * Cs_f[ix, iz]  + d_sol_f)  / max(1.0 - p_new, 0.1)
        phi[ix, iz]   = p_new
        C_Li[ix, iz]  = li_new
        Cf[ix, iz]    = cf_new
    end
    return
end

# ==============================================================================
# 3. HELPERS: INITIAL PERTURBATION & SNAPSHOT PACKER
# ==============================================================================
function make_smooth_noise(nx::Int, nz::Int, amp::Float64; seed::Int = 42, passes::Int = 4)
    rng = MersenneTwister(seed)
    field = 2.0 .* rand(rng, nx, nz) .- 1.0
    tmp   = similar(field)
    for _ in 1:passes
        for iz in 1:nz, ix in 1:nx
            ixm = (ix == 1) ? nx : ix - 1
            ixp = (ix == nx) ? 1 : ix + 1
            izm = max(1, iz - 1)
            izp = min(nz, iz + 1)
            tmp[ix, iz] = 0.25 * field[ix, iz] +
                          0.1875 * (field[ixm, iz] + field[ixp, iz] + field[ix, izm] + field[ix, izp])
        end
        field .= tmp
    end
    field .-= mean(field)
    maxabs = maximum(abs, field)
    return (maxabs > 0.0) ? (amp .* field ./ maxabs) : zeros(nx, nz)
end

function pack_snapshot(
    x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, T, Pt, Pf, eII,
    qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz, t_norm_val, export_val, trap_li_val
)
    qDx_c  = 0.5 .* (Array(qDx) .+ circshift(Array(qDx), (-1, 0))) .* v0
    qDz_c  = 0.5 .* (Array(qDz)[:, 1:nz] .+ Array(qDz)[:, 2:nz+1]) .* v0
    Vsx_c  = 0.5 .* (Array(Vsx) .+ circshift(Array(Vsx), (-1, 0))) .* v0
    Vsz_c  = 0.5 .* (Array(Vsz)[:, 1:nz] .+ Array(Vsz)[:, 2:nz+1]) .* v0
    Pc_MPa = (Array(Pt) .- Array(Pf)) .* (sigma0 / 1.0e6)
    Pf_MPa = Array(Pf) .* (sigma0 / 1.0e6)

    return Dict{String, Any}(
        "x"        => x_phys,
        "z"        => z_phys,
        "t_norm"   => Float64[t_norm_val],
        "export_m" => Float64[export_val],
        "trap_li"  => Float64[trap_li_val],
        "phi"      => permutedims(Array(phi)),
        "C_Li"     => permutedims(Array(C_Li)),
        "Cf"       => permutedims(Array(Cf)),
        "eta_m"    => permutedims(Array(eta_m_phys)),
        "T"        => permutedims(Array(T)),
        "Pc_MPa"   => permutedims(Pc_MPa),
        "Pf_MPa"   => permutedims(Pf_MPa),
        "eII_s"    => permutedims(Array(eII) ./ t0),
        "qDx"      => permutedims(qDx_c),
        "qDz"      => permutedims(qDz_c),
        "Vsx"      => permutedims(Vsx_c),
        "Vsz"      => permutedims(Vsz_c),
    )
end

# ==============================================================================
# 4. MAIN 2D SIMULATION DRIVER FOR ONE CASE
# ==============================================================================
function run_2d_case(cfg::CaseConfig, p::ModelParams; verbose::Bool = true, save_snapshots::Bool = true)
    t_wall0 = time()
    nx, nz = p.nx, p.nz
    dx_phys, dz_phys = p.Lx / nx, p.Lz / nz

    # Characteristic scales (non-dimensionalization keeps all PT fields O(1))
    L0     = p.Lz
    sigma0 = p.delta_rho * p.g * L0          # ~5.89 MPa
    eta0   = p.eta_s_ref                     # 8.0e18 Pa s
    t0     = eta0 / sigma0                   # Characteristic mechanical timescale (s)
    v0     = L0 / t0                         # Characteristic velocity (m / s)

    dx = dx_phys / L0
    dz = dz_phys / L0

    z_phys = [(iz - 0.5) * dz_phys for iz in 1:nz]
    x_phys = [(ix - 0.5) * dx_phys for ix in 1:nx]
    z_frac = z_phys ./ p.Lz

    # Initial fields (z = 0 base -> z = Lz Palaeo-BDT top)
    T_bg_z_host = [p.T_bg_bot + (p.T_bg_top - p.T_bg_bot) * zf for zf in z_frac]
    T_init_host = [p.T_init_bot + (p.T_init_top - p.T_init_bot) * zf for _ in 1:nx, zf in z_frac]

    noise = make_smooth_noise(nx, nz, p.noise_amp; seed = 42, passes = 4)
    phi_host = zeros(nx, nz)
    for iz in 1:nz, ix in 1:nx
        theta = clamp((T_init_host[ix, iz] - p.T_sol0) / (p.T_liq - p.T_sol0), 0.0, 1.0)
        taper = (iz <= 2 || iz >= nz - 1) ? 0.0 : 1.0
        phi_host[ix, iz] = clamp(theta^p.gamma_melt + taper * noise[ix, iz], 0.04, 0.45)
    end

    phi_ref = mean(phi_host)
    k_ref   = (p.a_grain^2) * (phi_ref^3) / (180.0 * (1.0 - phi_ref)^2)
    M0      = k_ref / p.eta_m0
    Rc2     = (M0 * eta0) / (L0^2)           # Dimensionless (delta_c0 / L0)^2

    tau_cool_s = (p.Lz^2 / p.kappa) * (1.0 + p.St)
    tau_lat_s  = p.tau_lat_frac * tau_cool_s
    t_end_s    = p.max_t_norm * tau_cool_s

    # Dimensionless BCs & control parameters
    U_top = cfg.use_shear ? (+0.5 * p.eps_bg * p.Lz / v0) : 0.0
    U_bot = cfg.use_shear ? (-0.5 * p.eps_bg * p.Lz / v0) : 0.0
    sigma_sq_nd = cfg.use_squeezing ? (p.sigma_sq / sigma0) : 0.0
    tau_y_nd    = p.tau_yield / sigma0
    D_disp_nd   = p.D_disp * t0 / (L0^2)
    kappa_eff   = p.kappa / (1.0 + p.St)

    use_shear_i = cfg.use_shear         ? 1 : 0
    use_dil_i   = cfg.use_dilatancy     ? 1 : 0
    use_fsol_i  = cfg.use_flux_solidus  ? 1 : 0
    use_vfb_i   = cfg.use_visc_feedback ? 1 : 0

    # Allocate ParallelStencil arrays
    phi       = Data.Array(phi_host)
    phi_new   = @zeros(nx, nz)
    T         = Data.Array(T_init_host)
    T_new     = @zeros(nx, nz)
    T_bg_z    = Data.Array(T_bg_z_host)

    C_Li      = Data.Array(fill(p.Li_0, nx, nz))
    C_Li_new  = @zeros(nx, nz)
    Cf        = Data.Array(fill(1.0, nx, nz))
    Cf_new    = @zeros(nx, nz)
    Cs_Li     = Data.Array(fill(p.D_Li * p.Li_0, nx, nz))
    Cs_Li_new = @zeros(nx, nz)
    Cs_f      = Data.Array(fill(p.D_flux, nx, nz))
    Cs_f_new  = @zeros(nx, nz)

    eta_s      = @ones(nx, nz)
    eta_eff    = @ones(nx, nz)
    zeta       = @ones(nx, nz)
    mob_c      = @ones(nx, nz)
    eta_m_phys = Data.Array(fill(p.eta_m0, nx, nz))

    mob_x      = @ones(nx, nz)
    mob_z      = @ones(nx, nz + 1)
    eta_sv     = @ones(nx, nz + 1)
    eta_sv_eff = @ones(nx, nz + 1)

    # Exact discrete vertical buoyancy-balance initial guess for Pt & Pf
    Pt_init = zeros(nx, nz)
    for ix in 1:nx
        Pt_init[ix, nz] = 0.5 * dz * (1.0 - phi_host[ix, nz])
        for iz in (nz - 1):-1:1
            Pt_init[ix, iz] = Pt_init[ix, iz + 1] +
                              0.5 * dz * ((1.0 - phi_host[ix, iz]) + (1.0 - phi_host[ix, iz + 1]))
        end
    end
    Vsx_init = [U_bot + (U_top - U_bot) * zf for _ in 1:nx, zf in z_frac]

    Pt     = Data.Array(Pt_init)
    Pf     = Data.Array(copy(Pt_init))
    dPf    = @zeros(nx, nz)
    Vsx    = Data.Array(Vsx_init)
    Vsz    = @zeros(nx, nz + 1)

    exz        = @zeros(nx, nz + 1)
    txz        = @zeros(nx, nz + 1)
    txx        = @zeros(nx, nz)
    tzz        = @zeros(nx, nz)
    div_Vs     = @zeros(nx, nz)
    eII        = @zeros(nx, nz)
    eII_mean_z = @zeros(nz)
    eps_dil    = @zeros(nx, nz)

    qDx   = @zeros(nx, nz)
    qDz   = @zeros(nx, nz + 1)
    Fm_x  = @zeros(nx, nz); Fs_x = @zeros(nx, nz)
    Fm_z  = @zeros(nx, nz + 1); Fs_z = @zeros(nx, nz + 1)
    v_cfl = @zeros(nx, nz)

    Res_vx  = @zeros(nx, nz)
    Res_vz  = @zeros(nx, nz + 1)
    Res_Pt  = @zeros(nx, nz)
    Res_Pf  = @zeros(nx, nz)
    Res_txx = @zeros(nx, nz)
    Res_tzz = @zeros(nx, nz)
    Res_txz = @zeros(nx, nz + 1)

    # Balanced Pseudo-Maxwell PT parameters (stable across >10^3 vertical viscosity jumps)
    nmax    = max(nx, nz)
    cfl_v   = 0.58              # 2D staggered-grid leapfrog CFL (8*cfl^2 = 2.69 < 4.0)
    r_bulk  = 0.50              # Keeps high deviatoric damping fraction for stiff-lid modes
    theta_m = 2.3 * nmax        # Preserves low-frequency vertical-contrast damping
    gamma_f = 6.0 / nmax        # Optimal 2nd-order damping for elliptic Pf solver
    rel_eta = 0.20              # Log-space viscoplastic viscosity under-relaxation

    # Diagnostic history & active/peak snapshot holders
    hist_t_norm  = Float64[]
    hist_trap_li = Float64[]
    hist_peak_li = Float64[]
    hist_min_eta = Float64[]
    hist_bdt_vol = Float64[]

    snap_active::Union{Nothing, Dict{String, Any}} = nothing
    snap_peak::Union{Nothing, Dict{String, Any}}   = nothing
    best_q_out   = 0.0
    peak_t_norm  = 0.0

    trap_melt_vol = 0.0   # Horizontally averaged cumulative BDT export thickness (m)
    trap_li_mass  = 0.0   # Horizontally averaged cumulative BDT Li mass (ppm * m)

    t_s  = 0.0
    step = 0
    dt_max_s = p.dt_max_frac * tau_cool_s
    dt_therm_lim = 0.20 * min(dx_phys, dz_phys)^2 / kappa_eff

    if verbose
        @printf("\n--- Running 2D Case: %-12s (grid %dx%d, threads=%d) ---\n",
                cfg.name, nx, nz, Threads.nthreads())
        @printf("    Switches: shear=%s | squeeze=%s | dilatancy=%s | flux_sol=%s | visc_fb=%s\n",
                cfg.use_shear, cfg.use_squeezing, cfg.use_dilatancy,
                cfg.use_flux_solidus, cfg.use_visc_feedback)
    end

    while t_s < t_end_s && step < p.max_steps
        # 1) Material properties
        @parallel (1:nx, 1:nz) update_material_props!(
            eta_s, zeta, mob_c, eta_m_phys, phi, T, Cf,
            phi_ref, k_ref, p.alpha_phi, p.eta_m0, p.eta_min, p.lambda_flux,
            M0, use_vfb_i, nx, nz
        )
        @parallel (1:nx, 1:nz+1) update_face_and_vertex_props!(
            mob_x, mob_z, eta_sv, mob_c, eta_s, phi, p.phi_min, nx, nz
        )

        # Initialize effective viscosities & equilibrium stresses at step 0; warm-start for step >= 1
        if step == 0
            eta_eff    .= eta_s
            eta_sv_eff .= eta_sv
            @parallel (1:nx, 1:nz+1) update_vertex_shear_pt!(
                exz, txz, eta_sv_eff, Res_txz, Vsx, Vsz, eta_sv,
                dx, dz, U_bot, U_top, use_shear_i, tau_y_nd, theta_m, 1.0, 1, nx, nz
            )
            @parallel (1:nx, 1:nz) update_cell_stresses_pt!(
                txx, tzz, div_Vs, eII, eta_eff, Res_txx, Res_tzz,
                Vsx, Vsz, exz, eta_s, dx, dz, tau_y_nd, theta_m, 1.0, 1, nx, nz
            )
        end

        # 2) Pseudo-Maxwell Symplectic PT Stokes-Darcy solve in (tau, Pt, Pf, Vs)
        dPf .= 0.0
        pt_err      = 1.0
        pt_iters    = 0
        err_v_rel   = 1.0
        err_pt_rel  = 1.0
        err_pf_rel  = 1.0
        err_tau_rel = 1.0

        for iter in 1:p.pt_max_iter
            # Step 2a: Relax deviatoric stresses & compute fresh div_Vs, eII from Vs
            @parallel (1:nx, 1:nz+1) update_vertex_shear_pt!(
                exz, txz, eta_sv_eff, Res_txz, Vsx, Vsz, eta_sv,
                dx, dz, U_bot, U_top, use_shear_i, tau_y_nd, theta_m, rel_eta, 0, nx, nz
            )
            @parallel (1:nx, 1:nz) update_cell_stresses_pt!(
                txx, tzz, div_Vs, eII, eta_eff, Res_txx, Res_tzz,
                Vsx, Vsz, exz, eta_s, dx, dz, tau_y_nd, theta_m, rel_eta, 0, nx, nz
            )
            if use_dil_i == 1
                @parallel (1:nz,) compute_eII_mean_z!(eII_mean_z, eII, nx, nz)
            end

            # Step 2b: Update Darcy fluxes & pressures (Pt, Pf) using fresh div_Vs & eII
            @parallel (1:nx, 1:nz+1) update_darcy_flux!(
                qDx, qDz, Pf, mob_x, mob_z, dx, dz, Rc2, nx, nz
            )
            @parallel (1:nx, 1:nz) update_pressures_pt!(
                Pt, Pf, dPf, Res_Pt, Res_Pf, eps_dil,
                div_Vs, qDx, qDz, zeta, eta_eff, eII, eII_mean_z, phi, mob_x, mob_z,
                dx, dz, Rc2, r_bulk, theta_m, gamma_f, sigma_sq_nd,
                use_dil_i, p.alpha_pump, p.psi_0, phi_ref, p.phi_min, p.phi_crit, nx, nz
            )

            # Step 2c: Symplectic velocity update using fresh (txx, tzz, txz, Pt)
            @parallel (1:nx, 1:nz) update_velocities_pt!(
                Vsx, Vsz, Res_vx, Res_vz,
                txx, tzz, txz, Pt, eta_eff, eta_sv_eff, phi,
                dx, dz, r_bulk, theta_m, cfl_v, nx, nz
            )
            @parallel (1:nx,) apply_vsz_bcs!(
                Vsz, Res_vz, Pt, tzz, txz, eta_eff, eta_sv_eff, phi,
                dx, dz, r_bulk, theta_m, cfl_v, nx, nz
            )

            pt_iters = iter
            if iter % p.pt_check_every == 0
                err_v   = max(maximum(abs, Res_vx), maximum(abs, Res_vz))
                err_pt  = maximum(abs, Res_Pt)
                err_pf  = maximum(abs, Res_Pf)
                err_tau = max(maximum(abs, Res_txx), maximum(abs, Res_tzz), maximum(abs, Res_txz))

                scale_v   = max(1.0, maximum(abs, txz) / dz, abs(sigma_sq_nd))
                scale_p   = max(1.0e-2, maximum(abs, eII), maximum(abs, eps_dil),
                                abs(sigma_sq_nd) / minimum(zeta))
                scale_tau = max(1.0, maximum(abs, txx), maximum(abs, tzz), maximum(abs, txz))

                err_v_rel   = err_v / scale_v
                err_pt_rel  = err_pt / scale_p
                err_pf_rel  = err_pf / scale_p
                err_tau_rel = err_tau / scale_tau
                pt_err      = max(err_v_rel, err_pt_rel, err_pf_rel, err_tau_rel)

                if verbose && step == 0 && iter % 500 == 0
                    @printf("      [step 0 PT iter %4d] err_v=%.2e | err_pt=%.2e | err_pf=%.2e | err_tau=%.2e | pt_err=%.2e\n",
                            iter, err_v_rel, err_pt_rel, err_pf_rel, err_tau_rel, pt_err)
                end
                if !isfinite(pt_err)
                    error("PT diverged at step $(step), iter $(iter) (err_v=$(err_v), err_pt=$(err_pt), err_pf=$(err_pf)).")
                end
                if pt_err < p.pt_tol
                    break
                end
            end
        end

        if pt_err > 1.0e-2
            error(@sprintf("PT failed to converge in case %s at step %d after %d iters (err_v=%.2e, err_pt=%.2e, err_pf=%.2e, err_tau=%.2e).",
                           cfg.name, step, pt_iters, err_v_rel, err_pt_rel, err_pf_rel, err_tau_rel))
        end

        # 3) Conservative phase fluxes & local-porosity adaptive CFL time step
        @parallel (1:nx, 1:nz+1) compute_phase_fluxes!(
            Fm_x, Fs_x, Fm_z, Fs_z, Vsx, Vsz, qDx, qDz, phi, nx, nz
        )
        @parallel (1:nx, 1:nz) compute_local_cfl_speed!(
            v_cfl, Vsx, Vsz, Fm_x, Fm_z, phi, p.phi_min, nx, nz
        )

        vmax_nd = max(maximum(v_cfl), 1.0e-12)
        if !isfinite(vmax_nd)
            error("Non-finite velocity in case $(cfg.name) at step $(step).")
        end

        dt_cfl_s = (0.25 * min(dx, dz) / vmax_nd) * t0
        dt_s     = min(dt_cfl_s, dt_max_s, t_end_s - t_s)
        dt_s     = max(dt_s, min(1.0e4, t_end_s - t_s))
        dt_nd    = dt_s / t0

        # Accumulate melt & Li exported across top Palaeo-BDT valve (z = Lz)
        q_out_row  = @view Fm_z[:, nz + 1]
        li_top_row = @view C_Li[:, nz]
        q_out_mean = mean(q_out_row) * v0
        if q_out_mean > 0.0
            trap_melt_vol += q_out_mean * dt_s
            trap_li_mass  += mean(q_out_row .* li_top_row) * v0 * dt_s
        end

        # 4) Conservative upwind advection of phi and dual-phase solutes
        @parallel (1:nx, 1:nz) advect_phi_and_solutes!(
            phi_new, C_Li_new, Cf_new, Cs_Li_new, Cs_f_new,
            phi, C_Li, Cf, Cs_Li, Cs_f, Fm_x, Fs_x, Fm_z, Fs_z,
            dt_nd, dx, dz, D_disp_nd, p.phi_min, p.phi_max, nx, nz
        )
        phi, phi_new     = phi_new, phi
        C_Li, C_Li_new   = C_Li_new, C_Li
        Cf, Cf_new       = Cf_new, Cf
        Cs_Li, Cs_Li_new = Cs_Li_new, Cs_Li
        Cs_f, Cs_f_new   = Cs_f_new, Cs_f

        # 5) Sub-cycled 2D thermal diffusion + Robin roof + lateral pluton cooling
        n_sub  = max(1, ceil(Int, dt_s / dt_therm_lim))
        dt_sub = dt_s / n_sub
        for _ in 1:n_sub
            @parallel (1:nx, 1:nz) thermal_substep!(
                T_new, T, T_bg_z,
                dt_sub, dx_phys, dz_phys, kappa_eff, p.H_roof, p.T_bg_top, tau_lat_s, nx, nz
            )
            T, T_new = T_new, T
        end

        # 6) Exact analytical Rayleigh fractional crystallization substep
        @parallel (1:nx, 1:nz) rayleigh_crystallization_step!(
            phi, C_Li, Cf, Cs_Li, Cs_f, T,
            dt_s, p.tau_cryst, p.T_liq, p.T_sol0, p.T_sol_min, p.dT_flux, p.gamma_melt,
            p.D_Li, p.D_flux, p.phi_min, p.phi_max, use_fsol_i, nx, nz
        )

        t_s  += dt_s
        step += 1
        t_norm_now = t_s / tau_cool_s

        # 7) Diagnostics: shallow BDT trap Li, active mush metrics & snapshots
        if trap_melt_vol > 1.0e-4
            V   = trap_melt_vol
            Vc  = p.V_peg_crit
            eff = (V * V) / (V * V + Vc * Vc)
            avg_li_out = trap_li_mass / V
            rayleigh_boost = p.f_peg_res^(p.D_Li - 1.0)
            trap_li = avg_li_out * (1.0 + eff * (rayleigh_boost - 1.0))
        else
            trap_li = p.Li_0
        end

        phi_arr  = Array(phi)
        li_arr   = Array(C_Li)
        eta_arr  = Array(eta_m_phys)
        mob_mask = phi_arr .>= 0.05
        peak_mush_li = any(mob_mask) ? maximum(li_arr[mob_mask]) : p.Li_0
        min_eta_val  = minimum(eta_arr)

        push!(hist_t_norm,  t_norm_now)
        push!(hist_trap_li, trap_li)
        push!(hist_peak_li, max(peak_mush_li, trap_li))
        push!(hist_min_eta, min_eta_val)
        push!(hist_bdt_vol, trap_melt_vol)

        # Capture synchronous active-mush snapshot (t/tc ~ 0.25) and peak-export snapshot
        if save_snapshots
            if snap_active === nothing && t_norm_now >= p.t_active_snap
                snap_active = pack_snapshot(
                    x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, T, Pt, Pf, eII,
                    qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz,
                    t_norm_now, trap_melt_vol, trap_li
                )
            end
            if t_norm_now >= 0.08 && q_out_mean > best_q_out
                best_q_out  = q_out_mean
                peak_t_norm = t_norm_now
                snap_peak   = pack_snapshot(
                    x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, T, Pt, Pf, eII,
                    qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz,
                    t_norm_now, trap_melt_vol, trap_li
                )
            end
        end

        if verbose && (step == 1 || step % 40 == 0 || t_s >= t_end_s || step == p.max_steps)
            pf_top_MPa = maximum(@view Pf[:, nz]) * (sigma0 / 1.0e6)
            @printf("    step=%4d | t/tc=%5.3f | PT=%4d (err=%.1e) | Pf_top=%+5.2f MPa | Export=%6.2f m | TrapLi=%7.1f ppm | minEta=%.2e\n",
                    step, t_norm_now, pt_iters, pt_err, pf_top_MPa, trap_melt_vol, trap_li, min_eta_val)
        end
    end

    snap_final = save_snapshots ? pack_snapshot(
        x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, T, Pt, Pf, eII,
        qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz,
        hist_t_norm[end], trap_melt_vol, hist_trap_li[end]
    ) : nothing

    if save_snapshots
        if snap_active === nothing
            snap_active = snap_final
        end
        if snap_peak === nothing
            snap_peak   = snap_active
            peak_t_norm = snap_peak["t_norm"][1]
        end
    end

    elapsed = time() - t_wall0
    if verbose
        @printf(">>> Finished %-12s in %.1f s | PeakFlux @ t/tc=%.3f | Final Export = %.2f m | BDT Trap Li = %.1f ppm | Min eta_m = %.2e Pa s\n",
                cfg.name, elapsed, peak_t_norm, trap_melt_vol, hist_trap_li[end], minimum(hist_min_eta))
    end

    return Dict{String, Any}(
        "case"        => cfg.name,
        "snap_final"  => snap_final,
        "snap_active" => snap_active,
        "snap_peak"   => snap_peak,
        "t_norm"      => hist_t_norm,
        "trap_li"     => hist_trap_li,
        "peak_li"     => hist_peak_li,
        "min_eta"     => hist_min_eta,
        "bdt_vol"     => hist_bdt_vol,
    )
end

# ==============================================================================
# 5. BATCH RUNNER: 4 ABLATION BENCHMARKS + OPTIONAL FIG. 3d PHASE-DIAGRAM SWEEP
# ==============================================================================
function run_four_ablation_cases(p::ModelParams; outdir::String = "output_2d")
    mkpath(outdir)

    # Exact 4-switch ablation matrix
    cases = [
        CaseConfig("static",      false, false, false, true, false),
        CaseConfig("filter",      false, true,  false, true, false),
        CaseConfig("shear_const", true,  false, true,  true, false),
        CaseConfig("coupled",     true,  false, true,  true, true),
    ]

    ts_dict = Dict{String, Any}()

    for cfg in cases
        res = run_2d_case(cfg, p; verbose = true, save_snapshots = true)

        npzwrite(joinpath(outdir, "snapshots_$(cfg.name).npz"),        res["snap_final"])
        npzwrite(joinpath(outdir, "snapshots_active_$(cfg.name).npz"), res["snap_active"])
        npzwrite(joinpath(outdir, "snapshots_peak_$(cfg.name).npz"),   res["snap_peak"])

        ts_dict["t_norm_$(cfg.name)"]  = res["t_norm"]
        ts_dict["trap_li_$(cfg.name)"] = res["trap_li"]
        ts_dict["peak_li_$(cfg.name)"] = res["peak_li"]
        ts_dict["min_eta_$(cfg.name)"] = res["min_eta"]
        ts_dict["bdt_vol_$(cfg.name)"] = res["bdt_vol"]
    end

    npzwrite(joinpath(outdir, "time_series.npz"), ts_dict)
    @printf("\n[DONE] Saved final/active/peak 2D snapshots and time_series.npz into ./%s/\n", outdir)
end

function run_phase_diagram_sweep(base_p::ModelParams; outdir::String = "output_2d")
    mkpath(outdir)
    eps_grid = 10.0 .^ range(-14.5, -12.0, length = 8)   # Strain rate axis (s^-1)
    eta_grid = 10.0 .^ range(4.5,   7.5,   length = 8)   # Initial melt viscosity axis (Pa s)

    ne, nm = length(eps_grid), length(eta_grid)
    peak_li_grid = zeros(nm, ne)
    export_grid  = zeros(nm, ne)
    cfg_coupled  = CaseConfig("coupled", true, false, true, true, true)

    @printf("\n=== Starting %dx%d Phase-Diagram Sweep (Fig. 3d) ===\n", nm, ne)
    for j in 1:nm, i in 1:ne
        p_ij = ModelParams(
            nx = 64, nz = 64,
            eps_bg = eps_grid[i],
            eta_m0 = eta_grid[j],
            max_steps = 900,
            pt_max_iter = 8000
        )
        res = run_2d_case(cfg_coupled, p_ij; verbose = false, save_snapshots = false)
        peak_li_grid[j, i] = maximum(res["trap_li"])
        export_grid[j, i]  = res["bdt_vol"][end]
        @printf("  (%2d/%2d, %2d/%2d) eps=%.1e, eta_m0=%.1e -> TrapLi=%7.1f ppm, Export=%5.2f m\n",
                j, nm, i, ne, eps_grid[i], eta_grid[j], peak_li_grid[j, i], export_grid[j, i])
    end

    npzwrite(joinpath(outdir, "phase_diagram_grid.npz"), Dict{String, Any}(
        "eps_grid"     => collect(eps_grid),
        "eta_m0_grid"  => collect(eta_grid),
        "trap_li_grid" => peak_li_grid,
        "export_grid"  => export_grid,
    ))
    @printf("[DONE] Saved phase_diagram_grid.npz into ./%s/\n", outdir)
end

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
function main()
    # Step A: Fast 12-thread CPU verification of the 4 ablation cases (96 x 96)
    # Once verified, increase nx = 192, nz = 192 (or 256) for publication Fig. 3a!
    params = ModelParams(nx = 96, nz = 96)
    run_four_ablation_cases(params; outdir = "output_2d")

    # Step B: Uncomment below once Step A results are verified to generate Fig. 3d grid
    # run_phase_diagram_sweep(params; outdir = "output_2d")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end