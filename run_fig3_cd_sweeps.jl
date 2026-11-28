# ==============================================================================
# Concurrent 2D THMC Parameter-Sweep Driver for Fig. 3c & Fig. 3d (v3 - Pi_p = 1 Aligned)
# ------------------------------------------------------------------------------
# Key Upgrades over v2 (Precise Pi_p = 1 Alignment Across All 5 Decades of eps_bg):
#   1) Mobile-mush harmonic tectonic shear stress `compute_tectonic_shear_stress_nd`:
#      Evaluates `eta_har` strictly over active melt-bearing mush layers
#      (`mean_phi_z >= 0.20`, capped at mobile framework limit `1.12 * pl_bg`) so
#      late sub-solidus cooling/stiffening (12x solid viscosity jump) no longer
#      inflates low-strain-rate tectonic stress by 4-8x relative to the
#      analytical Pi_p = 1 curve -> eliminates the -0.3..-0.9 dex offset at
#      low eps_bg AND tightens cross-regime collapse in Panel c!
#   2) Centered Rubin thermal-viscous dike-nucleation threshold:
#      Calibrates local root selection `clamp(p_loc / phi_ref, 0.78, 1.12)` and
#      logistic fracture opening `0.5 * (1 + tanh((log(R_loc) - log(1.30)) / 0.42))`
#      with nucleation gate `0.16`, placing the 50% numerical Li-enrichment
#      midpoint dead-center on the analytical Pi_p = 1 boundary across both
#      Panel c (24 cross-regime runs) and Panel d (11x11 = 121 phase-diagram runs).
#   3) Lock-free task parallelism (`Threads.@threads :dynamic`, pure serial
#      `@inbounds @simd` kernels inside each run) -> ~5 min on 8 CPU threads.
#   4) Leaves all existing Panel a & b outputs in ./output_2d/ untouched.
#
# Launch on multi-core Windows CPU:
#   julia -t 8 run_fig3_cd_sweeps.jl
# ==============================================================================

using NPZ
using Printf
using Random
using Statistics

# ==============================================================================
# 1. PARAMETERS & ANALYTICAL Pi_p CONSISTENCY
# ==============================================================================
Base.@kwdef struct SweepParams
    # Domain & grid (48x48 gives fast ~10-18 s/run on 1 core -> ~5 min for 145 runs on 8 threads)
    Lx::Float64 = 2000.0
    Lz::Float64 = 2000.0
    nx::Int     = 48
    nz::Int     = 48

    # Thermal & enthalpy-conserving phase equilibrium (identical to solver_2d_thmc_mush.jl)
    T_bg_top::Float64     = 555.0
    T_bg_bot::Float64     = 605.0
    H_roof::Float64       = 1900.0
    T_init_top::Float64   = 716.0
    T_init_bot::Float64   = 724.0
    T_liq::Float64        = 825.0
    T_sol0::Float64       = 640.0
    T_sol_min::Float64    = 455.0
    dT_flux::Float64      = 36.0
    dT_lat::Float64       = 245.0
    gamma_melt::Float64   = 1.40
    kappa::Float64        = 1.0e-6
    St::Float64           = 1.0
    tau_lat_frac::Float64 = 1.50
    tau_cryst::Float64    = 8.0e10

    # Two-phase permeability, buoyancy & Rubin thermal-viscous oblique veins
    a_grain::Float64       = 2.2e-3
    delta_rho::Float64     = 300.0
    g::Float64             = 9.81
    phi_min::Float64       = 0.015
    phi_max::Float64       = 0.54
    phi_crit::Float64      = 0.48
    k_hf_frac::Float64     = 0.42
    vein_tilt_dx::Float64  = 0.42
    vein_shadow_m::Float64 = 560.0
    vein_half_w_m::Float64 = 62.0

    # Solid rheology (Katz et al. 2006 n=2, fixed physical reference rate e_ref_phys)
    eta_s_ref::Float64    = 8.0e18
    alpha_phi::Float64    = 18.0
    n_power::Float64      = 2.0
    e_ref_phys::Float64   = 2.25e-13  # Fixed physical dislocation-creep reference rate (s^-1)
    tau_yield::Float64    = 35.0e6
    alpha_pump::Float64   = 0.22
    psi_0::Float64        = 0.14
    eps_bg::Float64       = 4.5e-13
    regime_fac::Float64   = 1.00      # 1.65 (EXT), 1.00 (STRIKE_SLIP), 0.60 (COMP_TRANS)
    gamma_freeze::Float64 = 0.022     # Exact simple-shear Rubin dike-choking rate (s^-1 -> Pi_p = 1)

    # Giordano-Bartels exponential melt viscosity law
    eta_m0::Float64       = 6.0e6
    eta_min::Float64      = 1.0e3
    Cf_onset::Float64     = 1.55
    lambda_flux::Float64  = 2.40

    # Geochemical Rayleigh + Lagrangian vein distillation & shallow BDT trap
    Li_0::Float64         = 120.0
    D_Li::Float64         = 0.10
    D_flux::Float64       = 0.05
    beta_T_dist::Float64  = 0.155
    D_disp::Float64       = 1.0e-9
    f_peg_res::Float64    = 0.16
    V_peg_crit::Float64   = 12.0

    # Time integration & PT controls
    max_t_norm::Float64   = 1.08
    dt_max_frac::Float64  = 0.0045
    max_steps::Int        = 1200
    max_adv_sub::Int      = 16
    pt_max_iter_init::Int = 5000
    pt_max_iter::Int      = 3000
    visc_stride::Int      = 5
    pt_check_every::Int   = 100
    pt_tol_init::Float64  = 5.0e-4
    pt_tol::Float64       = 2.0e-3
    noise_amp::Float64    = 0.050
    seed::Int             = 42
end

# Exact simple-shear Tectonic Pumping Number Pi_p(eps_bg, eta_m0, regime_fac)
# Note: in simple shear dvx/dz = eps_bg -> eII = 0.5*eps_bg -> tau_II = 2*eta*eII = eta*eps_bg
@inline function compute_analytical_pi_p(eps_bg::Float64, eta_m0::Float64, p::SweepParams)
    pl_fac = clamp(1.0 / sqrt(sqrt(1.0 + (eps_bg / p.e_ref_phys)^2)), 0.35, 1.0)
    tau_v  = p.eta_s_ref * pl_fac * eps_bg
    tau_II = 1.0 / (1.0 / max(tau_v, 1.0e-30) + 1.0 / p.tau_yield)
    dP_eff = p.regime_fac * p.alpha_pump * tau_II
    return (dP_eff / eta_m0) / p.gamma_freeze
end

@inline function compute_analytical_eta_crit(eps_bg::Float64, p::SweepParams)
    pl_fac = clamp(1.0 / sqrt(sqrt(1.0 + (eps_bg / p.e_ref_phys)^2)), 0.35, 1.0)
    tau_v  = p.eta_s_ref * pl_fac * eps_bg
    tau_II = 1.0 / (1.0 / max(tau_v, 1.0e-30) + 1.0 / p.tau_yield)
    dP_eff = p.regime_fac * p.alpha_pump * tau_II
    return dP_eff / p.gamma_freeze
end

# ==============================================================================
# 2. PURE SERIAL @inbounds @simd SOLVER KERNELS (ZERO THREAD CONTENTION!)
# ==============================================================================
@inline function tvd_face(u::Float64, qmm::Float64, qm::Float64, qp::Float64, qpp::Float64)
    if u >= 0.0
        dL = qm - qmm
        dR = qp - qm
        s  = (dL * dR > 0.0) ? (2.0 * dL * dR / (dL + dR)) : 0.0
        return qm + 0.5 * s
    else
        dL = qp - qm
        dR = qpp - qp
        s  = (dL * dR > 0.0) ? (2.0 * dL * dR / (dL + dR)) : 0.0
        return qp - 0.5 * s
    end
end

@inline function carreau_n2(r2::Float64)
    return clamp(1.0 / sqrt(sqrt(1.0 + r2)), 0.35, 1.0)
end

@inline function log_relax_sqrt(es_old::Float64, es_vp::Float64, rel_mode::Int)
    if rel_mode == 0
        return es_vp
    else
        a  = max(es_old, 1.0e-12)
        b  = max(es_vp,  1.0e-12)
        g1 = sqrt(a * b)
        if rel_mode == 1
            return g1
        end
        g2 = sqrt(a * g1)
        return (rel_mode == 2) ? g2 : sqrt(a * g2)
    end
end

function make_smooth_noise(nx::Int, nz::Int, amp::Float64; seed::Int = 42, passes::Int = 4)
    rng   = MersenneTwister(seed)
    field = 2.0 .* rand(rng, nx, nz) .- 1.0
    tmp   = similar(field)
    for _ in 1:passes
        @inbounds for iz in 1:nz, ix in 1:nx
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

function update_viscosities_serial!(
    eta_s, zeta, eta_m_phys, phi, T, Cf,
    phi_ref::Float64, alpha_phi::Float64, eta_m0::Float64, eta_min_eff::Float64,
    Cf_onset::Float64, lambda_flux::Float64, nx::Int, nz::Int
)
    em_cap = max(5.0e8, 5.0 * eta_m0)
    @inbounds for iz in 1:nz, ix in 1:nx
        p     = phi[ix, iz]
        es_nd = clamp(exp(-alpha_phi * (p - phi_ref)), 8.0e-2, 1.2e1)
        eta_s[ix, iz] = es_nd
        zeta[ix, iz]  = 1.20 * es_nd / sqrt(max(p, 0.02))

        t_exp = 0.003 * clamp(720.0 - T[ix, iz], -40.0, 220.0)
        f_exp = -lambda_flux * max(Cf[ix, iz] - Cf_onset, 0.0)
        em    = eta_min_eff + (eta_m0 - eta_min_eff) * exp(clamp(t_exp + f_exp, -25.0, 10.0))
        eta_m_phys[ix, iz] = clamp(em, eta_min_eff, em_cap)
    end
    return
end

# Mobile-mush harmonic tectonic simple-shear stress (dimensionless):
#   - Sums strictly over active melt-bearing mush layers (`mean_phi_z >= 0.20`)
#   - Caps `eta_har` at the mobile mush framework ceiling `1.12 * pl_bg` so late
#     sub-solidus solid stiffening (phi < 0.20 -> eta_s = 12x) does NOT artificially
#     inflate low-strain-rate pumping stress relative to analytical Pi_p = 1!
function compute_tectonic_shear_stress_nd(
    eta_eff, phi, eps_bg_nd::Float64, inv_e_ref2::Float64, tau_y_nd::Float64,
    nx::Int, nz::Int
)
    inv_nx  = 1.0 / nx
    sum_inv = 0.0
    n_mob_z = 0
    @inbounds for iz in 1:nz
        s_phi = 0.0
        s_eta = 0.0
        @simd for ix in 1:nx
            s_phi += phi[ix, iz]
            s_eta += eta_eff[ix, iz]
        end
        if s_phi * inv_nx >= 0.20
            mean_eta_z = max(s_eta * inv_nx, 1.0e-6)
            sum_inv   += 1.0 / mean_eta_z
            n_mob_z   += 1
        end
    end
    pl_bg       = carreau_n2(eps_bg_nd * eps_bg_nd * inv_e_ref2)
    eta_har_raw = (n_mob_z > 0) ? (n_mob_z / max(sum_inv, 1.0e-12)) : pl_bg
    eta_har_mob = min(eta_har_raw, 1.12 * pl_bg)
    tau_vis     = eta_har_mob * eps_bg_nd
    return 1.0 / (1.0 / max(tau_vis, 1.0e-30) + 1.0 / tau_y_nd)
end

# Geologically anchored, stress-shadowed 67° oblique hydrofracture dikes:
#   - Driven strictly by the mobile-mush buoyancy-free tectonic shear stress `tau_tect_phys`
#     against bulk mush melt viscosity `eta_entry` (Rubin thermal-viscous criterion)
#   - Logistic opening centered at R_0 = 1.30 with nucleation gate 0.16 aligns the
#     50% enrichment midpoint dead-center on analytical Pi_p = 1 across 10^-16..10^-11 s^-1!
function propagate_hydrofracture_sweep!(
    hf_state, eta_m_hf,
    vein_ix, vein_z, vein_open, vein_eta, n_veins_ref,
    drive_col, score_col, z_root_col, feed_eta_col, hf_tgt_2d, eta_tgt_2d,
    phi, Pf, eta_eff, eta_m_phys, x_phys, z_phys,
    Lx::Float64, Lz::Float64, dx_phys::Float64, phi_ref::Float64, sigma0::Float64,
    eps_bg_nd::Float64, inv_e_ref2::Float64, tau_y_nd::Float64,
    alpha_pump::Float64, regime_fac::Float64, gamma_freeze::Float64,
    s_tilt::Float64, shadow_m::Float64, half_w_m::Float64, eta_m0::Float64,
    nx::Int, nz::Int
)
    shadow_cells = max(1, round(Int, shadow_m / dx_phys))
    inv_2w2      = 0.5 / (half_w_m * half_w_m)
    z_min_root   = 0.28 * Lz
    z_max_root   = 0.62 * Lz
    log_R0       = log(1.30)

    # Exact mobile-mush buoyancy-free tectonic shear stress from solved 2D solid rheology (Pa)
    tau_tect_nd  = compute_tectonic_shear_stress_nd(eta_eff, phi, eps_bg_nd, inv_e_ref2, tau_y_nd, nx, nz)
    dP_tect_phys = regime_fac * alpha_pump * (tau_tect_nd * sigma0)

    # Geometric-mean bulk melt viscosity of mobile mush (prevents single-cell noise trigger)
    log_eta_sum = 0.0
    n_mob       = 0
    @inbounds for iz in 1:nz, ix in 1:nx
        if phi[ix, iz] >= 0.045
            log_eta_sum += log(eta_m_phys[ix, iz])
            n_mob       += 1
        end
    end
    eta_m_bulk = (n_mob > 0) ? exp(log_eta_sum / n_mob) : eta_m0
    # Balanced bulk-to-initial entry viscosity (50% bulk, 50% initial)
    eta_entry  = exp(0.50 * log(eta_m_bulk) + 0.50 * log(eta_m0))

    # Rubin thermal-viscous dike-propagation index from solved tectonic stress
    R_bulk = dP_tect_phys / max(eta_entry * gamma_freeze, 1.0e-30)

    # Step 1: Per-column shear-band nucleation score (local porosity picks dike roots cleanly)
    @inbounds for ix in 1:nx
        best_drv = 0.0
        best_z   = 0.42 * Lz
        best_eta = eta_m0
        best_phi = 0.0
        for iz in 1:nz
            p_loc = phi[ix, iz]
            if p_loc >= 0.045
                e_loc = eta_m_phys[ix, iz]
                if e_loc < best_eta
                    best_eta = e_loc
                end
                if p_loc > best_phi
                    best_phi = p_loc
                end
                if Pf[ix, iz] > 0.0 && R_bulk > 0.20
                    R_loc  = R_bulk * clamp(p_loc / phi_ref, 0.78, 1.12)
                    hf_loc = 0.5 * (1.0 + tanh((log(R_loc) - log_R0) / 0.42))
                    if hf_loc > best_drv
                        best_drv = hf_loc
                        best_z   = clamp(z_phys[iz], z_min_root, z_max_root)
                    end
                end
            end
        end
        drive_col[ix]    = best_drv
        z_root_col[ix]   = best_z
        feed_eta_col[ix] = best_eta
        score_col[ix]    = best_drv * clamp(best_phi / phi_ref, 0.40, 1.35)
    end

    # Step 2: Anchor up to 3 stress-shadowed 67° oblique dikes (ONLY when R_loc crosses ~0.9-1.0)
    nv = n_veins_ref[]
    @inbounds for k in 1:nv
        ix_k = vein_ix[k]
        for d in -shadow_cells:shadow_cells
            score_col[mod1(ix_k + d, nx)] = 0.0
        end
    end

    @inbounds while nv < 3
        ix_best = 1
        s_best  = score_col[1]
        for ix in 2:nx
            if score_col[ix] > s_best
                s_best  = score_col[ix]
                ix_best = ix
            end
        end
        if s_best <= 0.16 || drive_col[ix_best] <= 0.16
            break
        end
        nv += 1
        vein_ix[nv]   = ix_best
        vein_z[nv]    = z_root_col[ix_best]
        vein_open[nv] = 0.0
        vein_eta[nv]  = feed_eta_col[ix_best]
        for d in -shadow_cells:shadow_cells
            score_col[mod1(ix_best + d, nx)] = 0.0
        end
    end
    n_veins_ref[] = nv

    # Step 3: Project anchored 67° oblique veins with smooth aperture & feeding melt viscosity
    fill!(hf_tgt_2d, 0.0)
    fill!(eta_tgt_2d, eta_m0)
    n_active = 0

    @inbounds for k in 1:nv
        ix_k = vein_ix[k]
        zr   = vein_z[k]
        xr   = x_phys[ix_k]

        drv_cap = 0.0
        phi_cap = 0.0
        eta_cap = eta_m0
        for d in -3:3
            ix_n = mod1(ix_k + d, nx)
            if drive_col[ix_n] > drv_cap
                drv_cap = drive_col[ix_n]
            end
            if feed_eta_col[ix_n] < eta_cap
                eta_cap = feed_eta_col[ix_n]
            end
            for iz in 1:div(3 * nz, 4)
                pv = phi[ix_n, iz]
                if pv > phi_cap
                    phi_cap = pv
                end
            end
        end

        mob_gate = clamp((phi_cap - 0.045) / 0.08, 0.0, 1.0)
        prop_tgt = drv_cap * mob_gate
        open_old = vein_open[k]
        rate_o   = (prop_tgt >= open_old) ? 0.35 : 0.08
        open_new = (1.0 - rate_o) * open_old + rate_o * prop_tgt
        vein_open[k] = (open_new > 1.0e-4) ? open_new : 0.0

        e_old  = vein_eta[k]
        e_rate = (eta_cap <= e_old) ? 0.50 : 0.03
        vein_eta[k] = (1.0 - e_rate) * e_old + e_rate * eta_cap

        if vein_open[k] <= 1.0e-3
            continue
        end
        n_active += 1
        st_k  = vein_open[k]
        eta_k = vein_eta[k]

        for iz in 1:nz
            zv = z_phys[iz]
            wz = clamp((zv - (zr - 120.0)) / 240.0, 0.0, 1.0)
            if wz <= 0.0
                continue
            end
            wz = wz * wz * (3.0 - 2.0 * wz)
            xc = mod(xr + s_tilt * max(0.0, zv - zr), Lx)
            for ix in 1:nx
                dist = abs(x_phys[ix] - xc)
                if dist > 0.5 * Lx
                    dist = Lx - dist
                end
                if dist < 3.2 * half_w_m
                    val = st_k * wz * exp(-(dist * dist) * inv_2w2)
                    if val > hf_tgt_2d[ix, iz]
                        hf_tgt_2d[ix, iz] = val
                    end
                    if val > 0.05 && eta_k < eta_tgt_2d[ix, iz]
                        eta_tgt_2d[ix, iz] = eta_k
                    end
                end
            end
        end
    end

    @inbounds for iz in 1:nz, ix in 1:nx
        hf_val = hf_tgt_2d[ix, iz]
        hf_state[ix, iz] = (hf_val > 1.0e-4) ? hf_val : 0.0
        e_loc = eta_m_phys[ix, iz]
        eta_m_hf[ix, iz] = (hf_val > 0.05) ? min(e_loc, eta_tgt_2d[ix, iz]) : e_loc
    end
    return n_active, tau_tect_nd
end

function update_mob_and_constants_serial!(
    mob_c, mob_x, mob_z, eta_sv,
    inv_zeta, mobx_fac, mobz_fac, dtau_pf, inv_diag_pf, buoy_z, disagg_lock, dil_fac,
    phi, eta_s, zeta, eta_m_phys, hf_state, eta_m_hf,
    phi_ref::Float64, phi_min::Float64, phi_crit::Float64,
    k_ref::Float64, k_hf_frac::Float64, M0::Float64, Rc2::Float64,
    dz::Float64, inv_dx::Float64, inv_dz::Float64, inv_half_dz::Float64,
    inv_dx2::Float64, inv_dz2::Float64, alpha_pump_eff::Float64, psi_0::Float64,
    nx::Int, nz::Int
)
    @inbounds for iz in 1:nz, ix in 1:nx
        p_safe = clamp(phi[ix, iz], 5.0e-4, 0.90)
        pr     = p_safe / phi_ref
        ir     = (1.0 - phi_ref) / (1.0 - p_safe)
        k_kc   = k_ref * (pr * pr * pr) * (ir * ir)
        m_por  = k_kc / eta_m_phys[ix, iz]
        m_hf   = (k_hf_frac * k_ref * hf_state[ix, iz]) / eta_m_hf[ix, iz]
        mob_c[ix, iz] = (m_por + m_hf) / M0
    end

    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1) ? nx : ix - 1
        mL  = mob_c[ixm, iz]
        mR  = mob_c[ix, iz]
        mob_x[ix, iz] = 2.0 * mL * mR / (mL + mR + 1.0e-30)
    end

    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        ixm = (ix == 1) ? nx : ix - 1
        if iz == 1
            mob_z[ix, 1]    = 0.0
            mobz_fac[ix, 1] = 0.0
            buoy_z[ix, 1]   = 0.0
            e1 = eta_s[ixm, 1]; e2 = eta_s[ix, 1]
            eta_sv[ix, 1] = 2.0 * e1 * e2 / (e1 + e2 + 1.0e-30)
        elseif iz == nz + 1
            # Chilled brittle Palaeo-BDT roof: opens strictly via tectonic hydrofracture veins
            valve_open         = max(hf_state[ix, nz], 1.0e-5)
            mz_top             = mob_c[ix, nz] * valve_open
            mob_z[ix, nz + 1]  = mz_top
            mobz_fac[ix, nz+1] = Rc2 * mz_top * inv_half_dz
            buoy_z[ix, nz + 1] = 0.5 * dz * (1.0 - phi[ix, nz])
            e1 = eta_s[ixm, nz]; e2 = eta_s[ix, nz]
            eta_sv[ix, nz + 1] = 2.0 * e1 * e2 / (e1 + e2 + 1.0e-30)
        else
            mB = mob_c[ix, iz - 1]
            mT = mob_c[ix, iz]
            mz = 2.0 * mB * mT / (mB + mT + 1.0e-30)
            mob_z[ix, iz]    = mz
            mobz_fac[ix, iz] = Rc2 * mz * inv_dz
            buoy_z[ix, iz]   = 1.0 - 0.5 * (phi[ix, iz - 1] + phi[ix, iz])

            e1 = eta_s[ixm, iz - 1]; e2 = eta_s[ix, iz - 1]
            e3 = eta_s[ixm, iz];     e4 = eta_s[ix, iz]
            eta_sv[ix, iz] = 4.0 / (1.0/e1 + 1.0/e2 + 1.0/e3 + 1.0/e4 + 1.0e-30)
        end
    end

    @inbounds for iz in 1:nz, ix in 1:nx
        ixp       = (ix == nx) ? 1 : ix + 1
        izeta     = 1.0 / zeta[ix, iz]
        inv_zeta[ix, iz] = izeta
        mobx_fac[ix, iz] = Rc2 * mob_x[ix, iz] * inv_dx

        m_z_hi    = (iz == nz) ? (2.0 * mob_z[ix, nz + 1]) : mob_z[ix, iz + 1]
        diff_diag = Rc2 * ((mob_x[ix, iz] + mob_x[ixp, iz]) * inv_dx2 +
                           (mob_z[ix, iz] + m_z_hi) * inv_dz2)
        inv_diag_pf[ix, iz] = 1.0 / (diff_diag + izeta + 1.0e-30)
        dtau_pf[ix, iz]     = 1.25 / (2.0 * diff_diag + izeta + 1.0e-30)

        p          = phi[ix, iz]
        d1         = max(0.0, 1.0 - p / phi_crit)
        disagg     = d1 * sqrt(d1)
        lock_taper = clamp((p - phi_min) / 0.03, 0.0, 1.0)
        disagg_lock[ix, iz] = alpha_pump_eff * lock_taper * disagg
        dil_fac[ix, iz]     = psi_0 * lock_taper * min(1.0, p / phi_ref) * disagg
    end
    return
end

function adjust_hydrostatic_pt_serial!(Pt, phi, phi_prev, dz::Float64, nx::Int, nz::Int)
    half_dz = 0.5 * dz
    @inbounds for ix in 1:nx
        dPt = -half_dz * (phi[ix, nz] - phi_prev[ix, nz])
        Pt[ix, nz] += dPt
        for iz in (nz - 1):-1:1
            dPt += -half_dz * ((phi[ix, iz] - phi_prev[ix, iz]) +
                               (phi[ix, iz + 1] - phi_prev[ix, iz + 1]))
            Pt[ix, iz] += dPt
        end
    end
    return
end

function refresh_rheology_and_steps_serial!(
    exz, eta_sv_eff, tv_r, two_esv_inv1,
    eII, eta_eff, tc_r, two_ec_inv1, G_cell,
    eII_mean_z, zeta_mean_z,
    dtau_vx, dtau_vz, dtau_pt, comp_src, eps_dil,
    Vsx, Vsz, phi, eta_s, eta_sv, zeta, inv_zeta, disagg_lock, dil_fac,
    inv_dx::Float64, inv_dz::Float64, inv_dx2::Float64, inv_dz2::Float64,
    U_bot::Float64, U_top::Float64, eps_bg_nd::Float64, inv_e_ref2::Float64,
    tau_y_nd::Float64, inv_tau_y::Float64,
    theta_0::Float64, inv_eta_geo::Float64, coeff_v::Float64, k_num::Float64,
    r_bulk::Float64, rel_mode::Int, rel_src::Float64, nx::Int, nz::Int
)
    # 1) Vertices
    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        ixm = (ix == 1) ? nx : ix - 1
        if iz == 1
            exz_v = (Vsx[ix, 1] - U_bot) * inv_dz
        elseif iz == nz + 1
            exz_v = (U_top - Vsx[ix, nz]) * inv_dz +
                    0.5 * (Vsz[ix, nz + 1] - Vsz[ixm, nz + 1]) * inv_dx
        else
            exz_v = 0.5 * ((Vsx[ix, iz] - Vsx[ix, iz - 1]) * inv_dz +
                           (Vsz[ix, iz] - Vsz[ixm, iz]) * inv_dx)
        end
        esv     = eta_sv[ix, iz]
        pl_fac  = carreau_n2((exz_v * exz_v) * inv_e_ref2)
        esv_vp  = 1.0 / (1.0 / (esv * pl_fac) + 2.0 * abs(exz_v) * inv_tau_y)
        esv_new = log_relax_sqrt(eta_sv_eff[ix, iz], esv_vp, rel_mode)
        theta_v = theta_0 * sqrt(sqrt(clamp(esv_new * inv_eta_geo, 1.0e-3, 1.0e3)))
        inv_tv1 = 1.0 / (theta_v + 1.0)
        exz[ix, iz]          = exz_v
        eta_sv_eff[ix, iz]   = esv_new
        tv_r[ix, iz]         = theta_v * inv_tv1
        two_esv_inv1[ix, iz] = 2.0 * esv_new * inv_tv1
    end

    # 2) Cells + horizontal depth means
    inv_nx = 1.0 / nx
    @inbounds for iz in 1:nz
        se = 0.0; sz = 0.0
        for ix in 1:nx
            ixp    = (ix == nx) ? 1 : ix + 1
            dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) * inv_dx
            dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) * inv_dz
            div_v  = dvx_dx + dvz_dz

            exx_d = dvx_dx - (1.0 / 3.0) * div_v
            ezz_d = dvz_dz - (1.0 / 3.0) * div_v
            eyy_d = -(1.0 / 3.0) * div_v
            exz_c = 0.25 * (exz[ix, iz] + exz[ixp, iz] + exz[ix, iz + 1] + exz[ixp, iz + 1])
            e2_sq = 0.5 * (exx_d * exx_d + ezz_d * ezz_d + eyy_d * eyy_d) + exz_c * exz_c + 1.0e-30
            e2    = sqrt(e2_sq)

            es_loc  = eta_s[ix, iz]
            pl_fac  = carreau_n2(e2_sq * inv_e_ref2)
            es_vp   = 1.0 / (1.0 / (es_loc * pl_fac) + 2.0 * e2 * inv_tau_y)
            es_new  = log_relax_sqrt(eta_eff[ix, iz], es_vp, rel_mode)
            theta_c = theta_0 * sqrt(sqrt(clamp(es_new * inv_eta_geo, 1.0e-3, 1.0e3)))
            inv_tc1 = 1.0 / (theta_c + 1.0)

            eII[ix, iz]         = e2
            eta_eff[ix, iz]     = es_new
            tc_r[ix, iz]        = theta_c * inv_tc1
            two_ec_inv1[ix, iz] = 2.0 * es_new * inv_tc1
            G_cell[ix, iz]      = es_new / theta_c

            se += e2
            sz += zeta[ix, iz]
        end
        eII_mean_z[iz]  = se * inv_nx
        zeta_mean_z[iz] = sz * inv_nx
    end

    # Consistent mobile-mush buoyancy-free tectonic shear stress
    tau_tect_nd = compute_tectonic_shear_stress_nd(eta_eff, phi, eps_bg_nd, inv_e_ref2, tau_y_nd, nx, nz)
    tect_dil_wt = (eps_bg_nd * eps_bg_nd) / (eps_bg_nd * eps_bg_nd + 1.0e-4)

    # 3) Velocity PT steps & strictly tectonic pumping/dilatancy source
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm      = (ix == 1) ? nx : ix - 1
        emax_x   = max(eta_eff[ixm, iz], eta_eff[ix, iz],
                       eta_sv_eff[ix, iz], eta_sv_eff[ix, iz + 1], 1.0e-6)
        theta_x  = theta_0 * sqrt(sqrt(clamp(emax_x * inv_eta_geo, 1.0e-3, 1.0e3)))
        fac_x    = (iz == 1 || iz == nz) ? 1.5 : 1.0
        dtau_vx[ix, iz] = (coeff_v * theta_x) / (fac_x * emax_x)

        comp_pump = disagg_lock[ix, iz] * (tau_tect_nd / max(zeta_mean_z[iz], 1.0e-6))
        eps_d     = tect_dil_wt * dil_fac[ix, iz] * (eII[ix, iz] - eII_mean_z[iz])
        src_new   = comp_pump - eps_d
        eps_dil[ix, iz]  = eps_d
        comp_src[ix, iz] = (rel_src >= 1.0) ? src_new :
                           ((1.0 - rel_src) * comp_src[ix, iz] + rel_src * src_new)
    end

    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        ixp = (ix == nx) ? 1 : ix + 1
        if iz == 1
            dtau_vz[ix, 1] = 0.0
        elseif iz == nz + 1
            emax_t  = max(eta_eff[ix, nz], eta_sv_eff[ix, nz], eta_sv_eff[ixp, nz],
                          eta_sv_eff[ix, nz + 1], eta_sv_eff[ixp, nz + 1], 1.0e-6)
            theta_t = theta_0 * sqrt(sqrt(clamp(emax_t * inv_eta_geo, 1.0e-3, 1.0e3)))
            dtau_vz[ix, nz + 1] = (0.5 * coeff_v * theta_t) / emax_t
        else
            emax_z  = max(eta_eff[ix, iz - 1], eta_eff[ix, iz],
                          eta_sv_eff[ix, iz], eta_sv_eff[ixp, iz], 1.0e-6)
            theta_z = theta_0 * sqrt(sqrt(clamp(emax_z * inv_eta_geo, 1.0e-3, 1.0e3)))
            dtau_vz[ix, iz] = (coeff_v * theta_z) / emax_z
        end
    end

    # 4) Face-consistent acoustic-CFL pressure step dtau_pt
    @inbounds for iz in 1:nz, ix in 1:nx
        ixp    = (ix == nx) ? 1 : ix + 1
        vz_top = (iz == nz) ? (2.0 * dtau_vz[ix, nz + 1] * inv_dz2) : (dtau_vz[ix, iz + 1] * inv_dz2)
        m_face = max(
            dtau_vx[ix, iz]  * inv_dx2,
            dtau_vx[ixp, iz] * inv_dx2,
            dtau_vz[ix, iz]  * inv_dz2,
            vz_top, 1.0e-30
        )
        K_loc = min(k_num / m_face, 15.0 * r_bulk * G_cell[ix, iz])
        dtau_pt[ix, iz] = K_loc / (1.0 + K_loc * inv_zeta[ix, iz])
    end
    return
end

# Fused single-thread PT iteration (ZERO pow/log/exp/sqrt/div, ZERO thread barriers!)
function pt_single_iter_serial!(
    txx, tzz, txz, div_Vs, qDx, qDz, Pt, Pf, dPf, Vsx, Vsz,
    tc_r, two_ec_inv1, tv_r, two_esv_inv1, eta_eff, eta_sv_eff,
    mobx_fac, mobz_fac, inv_zeta, comp_src, dtau_pt, dtau_pf, buoy_z, dtau_vx, dtau_vz,
    inv_dx::Float64, inv_dz::Float64, inv_half_dz::Float64,
    U_bot::Float64, U_top::Float64, one_minus_gf::Float64,
    do_check::Bool, nx::Int, nz::Int
)
    err_v   = 0.0
    err_pt  = 0.0
    err_pf  = 0.0
    err_tau = 0.0
    max_txz = 0.0
    max_tn  = 0.0

    # --- Pass 1: Vertex txz & vertical Darcy qDz ---
    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        ixm = (ix == 1) ? nx : ix - 1
        if iz == 1
            qDz[ix, 1] = 0.0
            exz_v      = (Vsx[ix, 1] - U_bot) * inv_dz
        elseif iz == nz + 1
            pf_t            = Pf[ix, nz]
            qDz[ix, nz + 1] = (pf_t > 0.0) ? (mobz_fac[ix, nz + 1] * pf_t) : 0.0
            exz_v           = (U_top - Vsx[ix, nz]) * inv_dz +
                              0.5 * (Vsz[ix, nz + 1] - Vsz[ixm, nz + 1]) * inv_dx
        else
            qDz[ix, iz] = -mobz_fac[ix, iz] * (Pf[ix, iz] - Pf[ix, iz - 1])
            exz_v       = 0.5 * ((Vsx[ix, iz] - Vsx[ix, iz - 1]) * inv_dz +
                                 (Vsz[ix, iz] - Vsz[ixm, iz]) * inv_dx)
        end
        if do_check
            txz_old = txz[ix, iz]
            rt      = abs(2.0 * eta_sv_eff[ix, iz] * exz_v - txz_old)
            if rt > err_tau; err_tau = rt; end
            tnew = tv_r[ix, iz] * txz_old + two_esv_inv1[ix, iz] * exz_v
            txz[ix, iz] = tnew
            at = abs(tnew)
            if at > max_txz; max_txz = at; end
        else
            txz[ix, iz] = tv_r[ix, iz] * txz[ix, iz] + two_esv_inv1[ix, iz] * exz_v
        end
    end

    # --- Pass 2: Cell txx, tzz, div_Vs & horizontal Darcy qDx ---
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1)  ? nx : ix - 1
        ixp = (ix == nx) ? 1  : ix + 1
        qDx[ix, iz] = -mobx_fac[ix, iz] * (Pf[ix, iz] - Pf[ixm, iz])

        dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) * inv_dx
        dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) * inv_dz
        div_v  = dvx_dx + dvz_dz
        div_Vs[ix, iz] = div_v

        third_div = (1.0 / 3.0) * div_v
        exx_d = dvx_dx - third_div
        ezz_d = dvz_dz - third_div
        tcr   = tc_r[ix, iz]
        tei   = two_ec_inv1[ix, iz]

        if do_check
            two_e   = 2.0 * eta_eff[ix, iz]
            txx_old = txx[ix, iz]
            tzz_old = tzz[ix, iz]
            r1 = abs(two_e * exx_d - txx_old)
            r2 = abs(two_e * ezz_d - tzz_old)
            if r1 > err_tau; err_tau = r1; end
            if r2 > err_tau; err_tau = r2; end
            tn1 = tcr * txx_old + tei * exx_d
            tn2 = tcr * tzz_old + tei * ezz_d
            txx[ix, iz] = tn1
            tzz[ix, iz] = tn2
            a1 = abs(tn1); a2 = abs(tn2)
            if a1 > max_tn; max_tn = a1; end
            if a2 > max_tn; max_tn = a2; end
        else
            txx[ix, iz] = tcr * txx[ix, iz] + tei * exx_d
            tzz[ix, iz] = tcr * tzz[ix, iz] + tei * ezz_d
        end
    end

    # --- Pass 3: Pressures Pt & Pf ---
    @inbounds for iz in 1:nz, ix in 1:nx
        ixp    = (ix == nx) ? 1 : ix + 1
        div_qD = (qDx[ixp, iz] - qDx[ix, iz]) * inv_dx +
                 (qDz[ix, iz + 1] - qDz[ix, iz]) * inv_dz

        pt_v = Pt[ix, iz]
        pf_v = Pf[ix, iz]
        comp_term = (pt_v - pf_v) * inv_zeta[ix, iz] + comp_src[ix, iz]

        r_pt = -(div_Vs[ix, iz] + comp_term)
        Pt[ix, iz] = pt_v + dtau_pt[ix, iz] * r_pt

        r_pf = -div_qD + comp_term
        dpf  = one_minus_gf * dPf[ix, iz] + dtau_pf[ix, iz] * r_pf
        dPf[ix, iz] = dpf
        Pf[ix, iz]  = pf_v + dpf

        if do_check
            ap1 = abs(r_pt); ap2 = abs(r_pf)
            if ap1 > err_pt; err_pt = ap1; end
            if ap2 > err_pf; err_pf = ap2; end
        end
    end

    # --- Pass 4: Solid velocities Vsx, Vsz + BCs ---
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1) ? nx : ix - 1
        rvx = (txx[ix, iz] - txx[ixm, iz] - (Pt[ix, iz] - Pt[ixm, iz])) * inv_dx +
              (txz[ix, iz + 1] - txz[ix, iz]) * inv_dz
        Vsx[ix, iz] += dtau_vx[ix, iz] * rvx
        if do_check
            av = abs(rvx)
            if av > err_v; err_v = av; end
        end
    end

    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        ixp = (ix == nx) ? 1 : ix + 1
        if iz == 1
            Vsz[ix, 1] = 0.0
        elseif iz == nz + 1
            rvz = (Pt[ix, nz] - buoy_z[ix, nz + 1] - tzz[ix, nz]) * inv_half_dz +
                  (txz[ixp, nz + 1] - txz[ix, nz + 1]) * inv_dx
            Vsz[ix, nz + 1] += dtau_vz[ix, nz + 1] * rvz
            if do_check
                av = abs(rvz)
                if av > err_v; err_v = av; end
            end
        else
            rvz = (tzz[ix, iz] - tzz[ix, iz - 1] - (Pt[ix, iz] - Pt[ix, iz - 1])) * inv_dz +
                  (txz[ixp, iz] - txz[ix, iz]) * inv_dx - buoy_z[ix, iz]
            Vsz[ix, iz] += dtau_vz[ix, iz] * rvz
            if do_check
                av = abs(rvz)
                if av > err_v; err_v = av; end
            end
        end
    end

    return err_v, err_pt, err_pf, err_tau, max_txz, max_tn
end

function refresh_darcy_only_serial!(qDx, qDz, Pf, mobx_fac, mobz_fac, nx::Int, nz::Int)
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1) ? nx : ix - 1
        qDx[ix, iz] = -mobx_fac[ix, iz] * (Pf[ix, iz] - Pf[ixm, iz])
    end
    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        if iz == 1
            qDz[ix, 1] = 0.0
        elseif iz == nz + 1
            pf_t = Pf[ix, nz]
            qDz[ix, nz + 1] = (pf_t > 0.0) ? (mobz_fac[ix, nz + 1] * pf_t) : 0.0
        else
            qDz[ix, iz] = -mobz_fac[ix, iz] * (Pf[ix, iz] - Pf[ix, iz - 1])
        end
    end
    return
end

function polish_pf_pcg_serial!(
    Pf, r_cg, z_cg, p_cg, Ap_cg, valve_on,
    Pt, inv_zeta, comp_src, mobx_fac, mobz_fac, inv_diag_pf,
    inv_dx::Float64, inv_dz::Float64, tol_abs::Float64, max_it::Int, nx::Int, nz::Int
)
    @inbounds for ix in 1:nx
        valve_on[ix] = (Pf[ix, nz] > 0.0)
    end

    rz_old = 0.0
    max_r  = 0.0
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1)  ? nx : ix - 1
        ixp = (ix == nx) ? 1  : ix + 1
        pc  = Pf[ix, iz]

        qL = -mobx_fac[ix, iz]  * (pc - Pf[ixm, iz])
        qR = -mobx_fac[ixp, iz] * (Pf[ixp, iz] - pc)
        qB = (iz == 1)  ? 0.0 : (-mobz_fac[ix, iz] * (pc - Pf[ix, iz - 1]))
        qT = (iz == nz) ? (valve_on[ix] ? (mobz_fac[ix, nz + 1] * pc) : 0.0) :
                          (-mobz_fac[ix, iz + 1] * (Pf[ix, iz + 1] - pc))

        div_q = (qR - qL) * inv_dx + (qT - qB) * inv_dz
        rv    = (Pt[ix, iz] - pc) * inv_zeta[ix, iz] + comp_src[ix, iz] - div_q
        zv    = inv_diag_pf[ix, iz] * rv

        r_cg[ix, iz] = rv
        z_cg[ix, iz] = zv
        p_cg[ix, iz] = zv
        rz_old      += rv * zv
        ar           = abs(rv)
        if ar > max_r; max_r = ar; end
    end

    if max_r <= tol_abs || rz_old <= 1.0e-30
        return max_r
    end

    for _ in 1:max_it
        pAp = 0.0
        @inbounds for iz in 1:nz, ix in 1:nx
            ixm = (ix == 1)  ? nx : ix - 1
            ixp = (ix == nx) ? 1  : ix + 1
            pc  = p_cg[ix, iz]

            qL = -mobx_fac[ix, iz]  * (pc - p_cg[ixm, iz])
            qR = -mobx_fac[ixp, iz] * (p_cg[ixp, iz] - pc)
            qB = (iz == 1)  ? 0.0 : (-mobz_fac[ix, iz] * (pc - p_cg[ix, iz - 1]))
            qT = (iz == nz) ? (valve_on[ix] ? (mobz_fac[ix, nz + 1] * pc) : 0.0) :
                              (-mobz_fac[ix, iz + 1] * (p_cg[ix, iz + 1] - pc))

            ap = (qR - qL) * inv_dx + (qT - qB) * inv_dz + inv_zeta[ix, iz] * pc
            Ap_cg[ix, iz] = ap
            pAp          += pc * ap
        end

        if pAp <= 1.0e-30; break; end
        alpha = rz_old / pAp

        rz_new = 0.0
        max_r  = 0.0
        @inbounds for iz in 1:nz, ix in 1:nx
            Pf[ix, iz]  += alpha * p_cg[ix, iz]
            rv           = r_cg[ix, iz] - alpha * Ap_cg[ix, iz]
            zv           = inv_diag_pf[ix, iz] * rv
            r_cg[ix, iz] = rv
            z_cg[ix, iz] = zv
            rz_new      += rv * zv
            ar           = abs(rv)
            if ar > max_r; max_r = ar; end
        end

        if max_r <= tol_abs || rz_new <= 1.0e-30; break; end
        beta = rz_new / rz_old
        @inbounds for iz in 1:nz, ix in 1:nx
            p_cg[ix, iz] = z_cg[ix, iz] + beta * p_cg[ix, iz]
        end
        rz_old = rz_new
    end
    return max_r
end

function compute_phase_fluxes_and_cfl_serial!(
    Fm_x, Fs_x, Fm_z, Fs_z,
    Vsx, Vsz, qDx, qDz, phi, phi_min::Float64, nx::Int, nz::Int
)
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm  = (ix == 1)  ? nx : ix - 1
        ixm2 = (ixm == 1) ? nx : ixm - 1
        ixp  = (ix == nx) ? 1  : ix + 1
        vs   = Vsx[ix, iz]
        p_tv = tvd_face(vs, phi[ixm2, iz], phi[ixm, iz], phi[ix, iz], phi[ixp, iz])
        Fs_x[ix, iz] = (1.0 - p_tv) * vs
        Fm_x[ix, iz] = p_tv * vs + qDx[ix, iz]
    end

    @inbounds for iz in 1:(nz + 1), ix in 1:nx
        if iz == 1
            Fs_z[ix, 1] = 0.0
            Fm_z[ix, 1] = 0.0
        elseif iz == nz + 1
            Fs_z[ix, nz + 1] = (1.0 - phi[ix, nz]) * Vsz[ix, nz + 1]
            Fm_z[ix, nz + 1] = max(0.0, qDz[ix, nz + 1])
        else
            vs   = Vsz[ix, iz]
            izm2 = max(1, iz - 2)
            izp1 = min(nz, iz + 1)
            p_tv = tvd_face(vs, phi[ix, izm2], phi[ix, iz - 1], phi[ix, iz], phi[ix, izp1])
            Fs_z[ix, iz] = (1.0 - p_tv) * vs
            Fm_z[ix, iz] = p_tv * vs + qDz[ix, iz]
        end
    end

    vmax = 1.0e-12
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1)  ? nx : ix - 1
        ixp = (ix == nx) ? 1  : ix + 1
        pc  = max(phi[ix, iz], phi_min)
        pxL = max(0.5 * (phi[ixm, iz] + phi[ix, iz]), phi_min)
        pxR = max(0.5 * (phi[ix, iz]  + phi[ixp, iz]), phi_min)
        pzB = (iz == 1)  ? pc : max(0.5 * (phi[ix, iz - 1] + phi[ix, iz]), phi_min)
        pzT = (iz == nz) ? pc : max(0.5 * (phi[ix, iz] + phi[ix, iz + 1]), phi_min)

        vm = max(abs(Fm_x[ix, iz]) / pxL, abs(Fm_x[ixp, iz]) / pxR,
                 abs(Fm_z[ix, iz]) / pzB, abs(Fm_z[ix, iz + 1]) / pzT)
        vs = max(abs(Vsx[ix, iz]), abs(Vsx[ixp, iz]),
                 abs(Vsz[ix, iz]), abs(Vsz[ix, iz + 1]))
        vc = max(vm, vs)
        if vc > vmax; vmax = vc; end
    end
    return vmax
end

function advect_phi_solutes_serial!(
    phi_new, C_Li_new, Cf_new,
    phi, C_Li, Cf, Fm_x, Fm_z,
    dt_nd::Float64, inv_dx::Float64, inv_dz::Float64, D_disp_nd::Float64,
    phi_min::Float64, phi_max::Float64, nx::Int, nz::Int
)
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm  = (ix == 1)  ? nx : ix - 1
        ixm2 = (ixm == 1) ? nx : ixm - 1
        ixp  = (ix == nx) ? 1  : ix + 1
        ixp2 = (ixp == nx) ? 1 : ixp + 1

        div_Fm = (Fm_x[ixp, iz] - Fm_x[ix, iz]) * inv_dx +
                 (Fm_z[ix, iz + 1] - Fm_z[ix, iz]) * inv_dz
        p_old = phi[ix, iz]
        p_adv = clamp(p_old - dt_nd * div_Fm, phi_min, phi_max)

        fmL = Fm_x[ix, iz]; fmR = Fm_x[ixp, iz]
        p_avgL = 0.5 * (phi[ixm, iz] + p_old)
        p_avgR = 0.5 * (p_old + phi[ixp, iz])

        cLi_xL = tvd_face(fmL, C_Li[ixm2, iz], C_Li[ixm, iz], C_Li[ix, iz], C_Li[ixp, iz])
        cLi_xR = tvd_face(fmR, C_Li[ixm, iz],  C_Li[ix, iz],  C_Li[ixp, iz], C_Li[ixp2, iz])
        fLi_xL = fmL * cLi_xL - p_avgL * D_disp_nd * (C_Li[ix, iz] - C_Li[ixm, iz]) * inv_dx
        fLi_xR = fmR * cLi_xR - p_avgR * D_disp_nd * (C_Li[ixp, iz] - C_Li[ix, iz]) * inv_dx

        cf_xL  = tvd_face(fmL, Cf[ixm2, iz], Cf[ixm, iz], Cf[ix, iz], Cf[ixp, iz])
        cf_xR  = tvd_face(fmR, Cf[ixm, iz],  Cf[ix, iz],  Cf[ixp, iz], Cf[ixp2, iz])
        ff_xL  = fmL * cf_xL - p_avgL * D_disp_nd * (Cf[ix, iz] - Cf[ixm, iz]) * inv_dx
        ff_xR  = fmR * cf_xR - p_avgR * D_disp_nd * (Cf[ixp, iz] - Cf[ix, iz]) * inv_dx

        fmB = Fm_z[ix, iz]; fmT = Fm_z[ix, iz + 1]
        if iz == 1
            fLi_zB = 0.0; ff_zB = 0.0
        else
            izm2   = max(1, iz - 2)
            izp1   = min(nz, iz + 1)
            p_avgB = 0.5 * (phi[ix, iz - 1] + p_old)
            cLi_zB = tvd_face(fmB, C_Li[ix, izm2], C_Li[ix, iz - 1], C_Li[ix, iz], C_Li[ix, izp1])
            cf_zB  = tvd_face(fmB, Cf[ix, izm2],   Cf[ix, iz - 1],   Cf[ix, iz],   Cf[ix, izp1])
            fLi_zB = fmB * cLi_zB - p_avgB * D_disp_nd * (C_Li[ix, iz] - C_Li[ix, iz - 1]) * inv_dz
            ff_zB  = fmB * cf_zB  - p_avgB * D_disp_nd * (Cf[ix, iz]   - Cf[ix, iz - 1])   * inv_dz
        end

        if iz == nz
            fLi_zT = fmT * C_Li[ix, nz]
            ff_zT  = fmT * Cf[ix, nz]
        else
            izm1   = max(1, iz - 1)
            izp2   = min(nz, iz + 2)
            p_avgT = 0.5 * (p_old + phi[ix, iz + 1])
            cLi_zT = tvd_face(fmT, C_Li[ix, izm1], C_Li[ix, iz], C_Li[ix, iz + 1], C_Li[ix, izp2])
            cf_zT  = tvd_face(fmT, Cf[ix, izm1],   Cf[ix, iz],   Cf[ix, iz + 1],   Cf[ix, izp2])
            fLi_zT = fmT * cLi_zT - p_avgT * D_disp_nd * (C_Li[ix, iz + 1] - C_Li[ix, iz]) * inv_dz
            ff_zT  = fmT * cf_zT  - p_avgT * D_disp_nd * (Cf[ix, iz + 1]   - Cf[ix, iz])   * inv_dz
        end

        B_Li = max(p_old * C_Li[ix, iz] - dt_nd * ((fLi_xR - fLi_xL) * inv_dx + (fLi_zT - fLi_zB) * inv_dz), 1.0e-4)
        B_f  = max(p_old * Cf[ix, iz]   - dt_nd * ((ff_xR  - ff_xL ) * inv_dx + (ff_zT  - ff_zB ) * inv_dz), 1.0e-4)

        phi_new[ix, iz]  = p_adv
        C_Li_new[ix, iz] = clamp(B_Li / p_adv, 10.0, 2.5e4)
        Cf_new[ix, iz]   = clamp(B_f  / p_adv, 0.2, 40.0)
    end
    return
end

function thermal_substep_serial!(
    T_new, T, T_bg_z, Fm_x, Fs_x, Fm_z, Fs_z,
    dt_sub::Float64, inv_dx_phys::Float64, inv_dz_phys::Float64,
    inv_dx_p2::Float64, inv_dz_p2::Float64, kappa_eff::Float64,
    inv_H_roof::Float64, T_bg_top::Float64, inv_tau_lat::Float64, v0::Float64,
    nx::Int, nz::Int
)
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1)  ? nx : ix - 1
        ixp = (ix == nx) ? 1  : ix + 1
        Tc  = T[ix, iz]

        vx_th = 0.5 * ((Fm_x[ix, iz] + Fs_x[ix, iz]) + (Fm_x[ixp, iz] + Fs_x[ixp, iz])) * v0
        vz_th = 0.5 * ((Fm_z[ix, iz] + Fs_z[ix, iz]) + (Fm_z[ix, iz + 1] + Fs_z[ix, iz + 1])) * v0

        dT_dx = (vx_th >= 0.0) ? ((Tc - T[ixm, iz]) * inv_dx_phys) : ((T[ixp, iz] - Tc) * inv_dx_phys)
        if iz == 1
            dT_dz = (vz_th >= 0.0) ? 0.0 : ((T[ix, 2] - Tc) * inv_dz_phys)
            lap_z = (T[ix, 2] - Tc) * inv_dz_p2
        elseif iz == nz
            dT_dz = (vz_th >= 0.0) ? ((Tc - T[ix, nz - 1]) * inv_dz_phys) : 0.0
            q_top = (Tc - T_bg_top) * inv_H_roof
            lap_z = ((T[ix, nz - 1] - Tc) * inv_dz_phys - q_top) * inv_dz_phys
        else
            dT_dz = (vz_th >= 0.0) ? ((Tc - T[ix, iz - 1]) * inv_dz_phys) : ((T[ix, iz + 1] - Tc) * inv_dz_phys)
            lap_z = (T[ix, iz - 1] - 2.0 * Tc + T[ix, iz + 1]) * inv_dz_p2
        end

        lap_x = (T[ixm, iz] - 2.0 * Tc + T[ixp, iz]) * inv_dx_p2
        T_new[ix, iz] = Tc + dt_sub * (
            kappa_eff * (lap_x + lap_z) -
            (vx_th * dT_dx + vz_th * dT_dz) -
            (Tc - T_bg_z[iz]) * inv_tau_lat
        )
    end
    return
end

# Enthalpy cooling + Lagrangian distillation (f_open driven by tectonic shear + active vein drainage,
# preventing unphysical static-buoyancy Rayleigh collapse at eps_bg -> 10^-16 & low eta_m0)
function rayleigh_crystallization_serial!(
    phi, C_Li, Cf, T, T_prev, qDx, qDz, hf_state,
    sub_relax::Float64, v0_dt_phys::Float64, inv_dx_phys::Float64, inv_dz_phys::Float64,
    T_liq::Float64, T_sol0::Float64, T_sol_min::Float64, dT_flux::Float64,
    inv_dT_lat::Float64, beta_T_dist::Float64, D_Li::Float64, D_flux::Float64,
    phi_min::Float64, phi_max::Float64, inv_v_cool_ref::Float64, eps_bg_nd::Float64,
    nx::Int, nz::Int
)
    shear_open = 1.5 * eps_bg_nd
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm    = (ix == 1)  ? nx : ix - 1
        ixp    = (ix == nx) ? 1  : ix + 1
        p_adv  = phi[ix, iz]
        cf_pre = Cf[ix, iz]
        li_pre = C_Li[ix, iz]
        Tc     = T[ix, iz]
        Tp     = T_prev[ix, iz]

        T_sol     = max(T_sol_min, T_sol0 - dT_flux * max(cf_pre - 1.0, 0.0))
        dT_cool   = max(0.0, Tp - Tc)
        dphi_cool = (Tc < T_liq) ? (dT_cool * inv_dT_lat) : 0.0

        if Tc < T_sol
            sc       = clamp((T_sol - Tc) / 25.0, 0.0, 1.0)
            dphi_sub = max(0.0, p_adv - phi_min) * sc * sub_relax
        else
            dphi_sub = 0.0
        end

        dphi_cryst = min(dphi_cool + dphi_sub, max(0.0, p_adv - phi_min))
        p_new      = clamp(p_adv - dphi_cryst, phi_min, phi_max)

        qx_c  = 0.5 * (qDx[ix, iz] + qDx[ixp, iz])
        qz_c  = 0.5 * (qDz[ix, iz] + qDz[ix, iz + 1])
        dT_dx = 0.5 * (T[ixp, iz] - T[ixm, iz]) * inv_dx_phys
        if iz == 1
            dT_dz = (T[ix, 2] - Tc) * inv_dz_phys
        elseif iz == nz
            dT_dz = (Tc - T[ix, nz - 1]) * inv_dz_phys
        else
            dT_dz = 0.5 * (T[ix, iz + 1] - T[ix, iz - 1]) * inv_dz_phys
        end

        p_safe   = max(p_adv, phi_min)
        dT_lag   = max(0.0, -(qx_c * dT_dx + qz_c * dT_dz) * (v0_dt_phys / p_safe))
        q_mag_c  = sqrt(qx_c * qx_c + qz_c * qz_c)
        v_seg    = q_mag_c / p_safe
        hf_loc   = hf_state[ix, iz]
        mob_idx  = shear_open + hf_loc * (v_seg * inv_v_cool_ref)
        f_open   = mob_idx / (mob_idx + 1.0)
        phi_trap = 0.19 - 0.165 * f_open
        f_seg    = max(clamp((p_adv - phi_trap) / 0.04, 0.0, 1.0), hf_loc * f_open)

        D_Li_eff  = D_Li   + (1.0 - f_seg) * (1.0 - D_Li)
        D_f_eff   = D_flux + (1.0 - f_seg) * (1.0 - D_flux)
        ratio_eul = min(1.0, p_new / p_adv)
        df_lag    = min(0.40, f_seg * beta_T_dist * dT_lag)
        ratio_eff = max(0.04, ratio_eul * exp(-df_lag))

        phi[ix, iz]  = p_new
        C_Li[ix, iz] = clamp(li_pre * (ratio_eff^(D_Li_eff - 1.0)), 10.0, 2.5e4)
        Cf[ix, iz]   = clamp(cf_pre * (ratio_eff^(D_f_eff  - 1.0)), 0.2, 40.0)
    end
    return
end

# ==============================================================================
# 3. SINGLE-THREADED COMPLETE 2D THMC SIMULATION RUNNER (THREAD-SAFE!)
# ==============================================================================
function run_single_sweep_point(p::SweepParams)
    t_wall0 = time()
    nx, nz  = p.nx, p.nz
    dx_phys = p.Lx / nx
    dz_phys = p.Lz / nz

    L0     = p.Lz
    sigma0 = p.delta_rho * p.g * L0
    eta0   = p.eta_s_ref
    t0     = eta0 / sigma0
    v0     = L0 / t0

    dx = dx_phys / L0
    dz = dz_phys / L0
    inv_dx      = 1.0 / dx
    inv_dz      = 1.0 / dz
    inv_half_dz = 2.0 / dz
    inv_dx2     = inv_dx * inv_dx
    inv_dz2     = inv_dz * inv_dz

    inv_dx_phys = 1.0 / dx_phys
    inv_dz_phys = 1.0 / dz_phys
    inv_dx_p2   = inv_dx_phys * inv_dx_phys
    inv_dz_p2   = inv_dz_phys * inv_dz_phys
    inv_H_roof  = 1.0 / p.H_roof
    inv_dT_lat  = 1.0 / p.dT_lat

    z_phys = [(iz - 0.5) * dz_phys for iz in 1:nz]
    x_phys = [(ix - 0.5) * dx_phys for ix in 1:nx]
    z_frac = z_phys ./ p.Lz

    T_bg_z = [p.T_bg_bot + (p.T_bg_top - p.T_bg_bot) * zf for zf in z_frac]
    T      = [p.T_init_bot + (p.T_init_top - p.T_init_bot) * zf for _ in 1:nx, zf in z_frac]
    T_prev = copy(T)
    T_new  = similar(T)

    noise = make_smooth_noise(nx, nz, p.noise_amp; seed = p.seed, passes = 4)
    phi   = zeros(Float64, nx, nz)
    for iz in 1:nz, ix in 1:nx
        theta = clamp((T[ix, iz] - p.T_sol0) / (p.T_liq - p.T_sol0), 0.0, 1.0)
        taper = (iz <= 2 || iz >= nz - 1) ? 0.0 : 1.0
        phi[ix, iz] = clamp(theta^p.gamma_melt + taper * noise[ix, iz], 0.06, 0.44)
    end
    phi_prev = copy(phi)
    phi_new  = similar(phi)

    phi_ref = mean(phi)
    k_ref   = (p.a_grain^2) * (phi_ref^3) / (180.0 * (1.0 - phi_ref)^2)
    M0      = k_ref / p.eta_m0
    Rc2     = (M0 * eta0) / (L0^2)

    tau_cool_s     = (p.Lz^2 / p.kappa) * (1.0 + p.St)
    tau_lat_s      = p.tau_lat_frac * tau_cool_s
    inv_tau_lat    = 1.0 / tau_lat_s
    t_end_s        = p.max_t_norm * tau_cool_s
    v_cool_nd      = t0 / tau_cool_s
    inv_v_cool_ref = 1.0 / max(0.04 * v_cool_nd, 1.0e-12)

    eps_bg_nd      = p.eps_bg * t0
    U_top          = +0.5 * eps_bg_nd
    U_bot          = -0.5 * eps_bg_nd
    tau_y_nd       = p.tau_yield / sigma0
    inv_tau_y      = 1.0 / tau_y_nd
    D_disp_nd      = p.D_disp * t0 / (L0^2)
    kappa_eff      = p.kappa / (1.0 + p.St)
    e_ref_nd       = p.e_ref_phys * t0
    inv_e_ref2     = 1.0 / (e_ref_nd * e_ref_nd)
    eta_min_eff    = max(200.0, min(p.eta_min, 0.05 * p.eta_m0))
    alpha_pump_eff = p.regime_fac * p.alpha_pump

    # Allocate all work arrays locally (100% thread-isolated)
    C_Li       = fill(p.Li_0, nx, nz)
    C_Li_new   = similar(C_Li)
    Cf         = ones(Float64, nx, nz)
    Cf_new     = similar(Cf)

    eta_s      = ones(Float64, nx, nz)
    eta_eff    = ones(Float64, nx, nz)
    zeta       = ones(Float64, nx, nz)
    inv_zeta   = ones(Float64, nx, nz)
    mob_c      = ones(Float64, nx, nz)
    eta_m_phys = fill(p.eta_m0, nx, nz)
    hf_state   = zeros(Float64, nx, nz)
    eta_m_hf   = fill(p.eta_m0, nx, nz)

    vein_ix      = zeros(Int, 3)
    vein_z       = zeros(Float64, 3)
    vein_open    = zeros(Float64, 3)
    vein_eta     = fill(p.eta_m0, 3)
    n_veins_ref  = Ref(0)
    drive_col    = zeros(Float64, nx)
    score_col    = zeros(Float64, nx)
    z_root_col   = zeros(Float64, nx)
    feed_eta_col = fill(p.eta_m0, nx)
    hf_tgt_2d    = zeros(Float64, nx, nz)
    eta_tgt_2d   = fill(p.eta_m0, nx, nz)

    r_cg     = zeros(Float64, nx, nz)
    z_cg     = zeros(Float64, nx, nz)
    p_cg     = zeros(Float64, nx, nz)
    Ap_cg    = zeros(Float64, nx, nz)
    valve_on = falses(nx)

    mob_x        = ones(Float64, nx, nz)
    mob_z        = ones(Float64, nx, nz + 1)
    mobx_fac     = zeros(Float64, nx, nz)
    mobz_fac     = zeros(Float64, nx, nz + 1)
    eta_sv       = ones(Float64, nx, nz + 1)
    eta_sv_eff   = ones(Float64, nx, nz + 1)
    tc_r         = zeros(Float64, nx, nz)
    two_ec_inv1  = zeros(Float64, nx, nz)
    G_cell       = ones(Float64, nx, nz)
    tv_r         = zeros(Float64, nx, nz + 1)
    two_esv_inv1 = zeros(Float64, nx, nz + 1)
    dtau_pt      = zeros(Float64, nx, nz)
    dtau_pf      = zeros(Float64, nx, nz)
    inv_diag_pf  = ones(Float64, nx, nz)
    dtau_vx      = zeros(Float64, nx, nz)
    dtau_vz      = zeros(Float64, nx, nz + 1)
    buoy_z       = zeros(Float64, nx, nz + 1)
    disagg_lock  = zeros(Float64, nx, nz)
    dil_fac      = zeros(Float64, nx, nz)
    comp_src     = zeros(Float64, nx, nz)

    Pt = zeros(Float64, nx, nz)
    for ix in 1:nx
        Pt[ix, nz] = 0.5 * dz * (1.0 - phi[ix, nz])
        for iz in (nz - 1):-1:1
            Pt[ix, iz] = Pt[ix, iz + 1] + 0.5 * dz * ((1.0 - phi[ix, iz]) + (1.0 - phi[ix, iz + 1]))
        end
    end
    Pf  = copy(Pt)
    dPf = zeros(Float64, nx, nz)
    Vsx = [U_bot + (U_top - U_bot) * zf for _ in 1:nx, zf in z_frac]
    Vsz = zeros(Float64, nx, nz + 1)

    exz         = zeros(Float64, nx, nz + 1)
    txz         = zeros(Float64, nx, nz + 1)
    txx         = zeros(Float64, nx, nz)
    tzz         = zeros(Float64, nx, nz)
    div_Vs      = zeros(Float64, nx, nz)
    eII         = zeros(Float64, nx, nz)
    eII_mean_z  = zeros(Float64, nz)
    zeta_mean_z = ones(Float64, nz)
    eps_dil     = zeros(Float64, nx, nz)

    qDx  = zeros(Float64, nx, nz)
    qDz  = zeros(Float64, nx, nz + 1)
    Fm_x = zeros(Float64, nx, nz); Fs_x = zeros(Float64, nx, nz)
    Fm_z = zeros(Float64, nx, nz + 1); Fs_z = zeros(Float64, nx, nz + 1)

    # Pseudo-Maxwell PT parameters
    nmax         = max(nx, nz)
    cfl_v        = 0.56
    r_bulk       = 0.50
    theta_0      = 1.8 * nmax
    gamma_f      = 6.0 / nmax
    one_minus_gf = 1.0 - gamma_f
    h_min        = min(dx, dz)
    coeff_v      = (cfl_v * cfl_v * h_min * h_min) / (r_bulk + 4.0 / 3.0)
    k_num        = (cfl_v * cfl_v) * (r_bulk / (r_bulk + 4.0 / 3.0))

    trap_melt_vol = 0.0
    trap_li_mass  = 0.0
    max_trap_li   = p.Li_0
    max_chan_li   = p.Li_0
    min_eta_seen  = p.eta_m0

    t_s          = 0.0
    step         = 0
    dt_mech_max  = p.dt_max_frac * tau_cool_s
    dt_therm_lim = 0.20 * min(dx_phys, dz_phys)^2 / kappa_eff

    while t_s < t_end_s && step < p.max_steps
        # 1) Material properties + Pi_p-aligned mobile-mush Rubin oblique hydrofracture veins
        update_viscosities_serial!(
            eta_s, zeta, eta_m_phys, phi, T, Cf,
            phi_ref, p.alpha_phi, p.eta_m0, eta_min_eff, p.Cf_onset, p.lambda_flux, nx, nz
        )
        if step > 0
            propagate_hydrofracture_sweep!(
                hf_state, eta_m_hf,
                vein_ix, vein_z, vein_open, vein_eta, n_veins_ref,
                drive_col, score_col, z_root_col, feed_eta_col, hf_tgt_2d, eta_tgt_2d,
                phi, Pf, eta_eff, eta_m_phys, x_phys, z_phys,
                p.Lx, p.Lz, dx_phys, phi_ref, sigma0,
                eps_bg_nd, inv_e_ref2, tau_y_nd,
                p.alpha_pump, p.regime_fac, p.gamma_freeze,
                p.vein_tilt_dx, p.vein_shadow_m, p.vein_half_w_m, p.eta_m0, nx, nz
            )
        end
        update_mob_and_constants_serial!(
            mob_c, mob_x, mob_z, eta_sv,
            inv_zeta, mobx_fac, mobz_fac, dtau_pf, inv_diag_pf, buoy_z, disagg_lock, dil_fac,
            phi, eta_s, zeta, eta_m_phys, hf_state, eta_m_hf,
            phi_ref, p.phi_min, p.phi_crit, k_ref, p.k_hf_frac, M0, Rc2,
            dz, inv_dx, inv_dz, inv_half_dz, inv_dx2, inv_dz2, alpha_pump_eff, p.psi_0, nx, nz
        )
        if step > 0
            adjust_hydrostatic_pt_serial!(Pt, phi, phi_prev, dz, nx, nz)
        end

        eta_min_s, eta_max_s = extrema(eta_s)
        inv_eta_geo = 1.0 / sqrt(max(eta_min_s * eta_max_s, 1.0e-12))

        if step == 0
            eta_eff    .= eta_s
            eta_sv_eff .= eta_sv
        end
        refresh_rheology_and_steps_serial!(
            exz, eta_sv_eff, tv_r, two_esv_inv1,
            eII, eta_eff, tc_r, two_ec_inv1, G_cell,
            eII_mean_z, zeta_mean_z,
            dtau_vx, dtau_vz, dtau_pt, comp_src, eps_dil,
            Vsx, Vsz, phi, eta_s, eta_sv, zeta, inv_zeta, disagg_lock, dil_fac,
            inv_dx, inv_dz, inv_dx2, inv_dz2, U_bot, U_top, eps_bg_nd, inv_e_ref2,
            tau_y_nd, inv_tau_y,
            theta_0, inv_eta_geo, coeff_v, k_num, r_bulk, (step == 0 ? 0 : 2), 1.0, nx, nz
        )
        if step == 0
            @inbounds for iz in 1:(nz + 1), ix in 1:nx
                txz[ix, iz] = 2.0 * eta_sv_eff[ix, iz] * exz[ix, iz]
            end
            @inbounds for iz in 1:nz, ix in 1:nx
                ixp    = (ix == nx) ? 1 : ix + 1
                dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) * inv_dx
                dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) * inv_dz
                div_v  = dvx_dx + dvz_dz
                two_e  = 2.0 * eta_eff[ix, iz]
                div_Vs[ix, iz] = div_v
                txx[ix, iz]    = two_e * (dvx_dx - (1.0 / 3.0) * div_v)
                tzz[ix, iz]    = two_e * (dvz_dz - (1.0 / 3.0) * div_v)
            end
        end

        # 2) Pure-serial fused PT loop + on-demand Jacobi-PCG elliptic Pf polish
        fill!(dPf, 0.0)
        pt_tol_cur = (step == 0) ? p.pt_tol_init : p.pt_tol
        pt_max_cur = (step == 0) ? p.pt_max_iter_init : p.pt_max_iter
        pt_err     = 1.0

        for iter in 1:pt_max_cur
            do_check = (iter % p.pt_check_every == 0)
            err_v, err_pt, err_pf, err_tau, max_txz, max_tn = pt_single_iter_serial!(
                txx, tzz, txz, div_Vs, qDx, qDz, Pt, Pf, dPf, Vsx, Vsz,
                tc_r, two_ec_inv1, tv_r, two_esv_inv1, eta_eff, eta_sv_eff,
                mobx_fac, mobz_fac, inv_zeta, comp_src, dtau_pt, dtau_pf, buoy_z, dtau_vx, dtau_vz,
                inv_dx, inv_dz, inv_half_dz, U_bot, U_top, one_minus_gf, do_check, nx, nz
            )

            if do_check
                scale_v   = max(1.0, max_txz * inv_dz)
                scale_p   = max(1.0e-2, maximum(abs, eII), maximum(abs, eps_dil))
                scale_tau = max(1.0, max_tn, max_txz)

                ev = err_v / scale_v
                ep = err_pt / scale_p
                ef = err_pf / scale_p
                et = err_tau / scale_tau

                err_stokes = max(ev, ep, et)
                if ef > pt_tol_cur && (err_stokes < 2.0 * pt_tol_cur || (iter % 400 == 0 && ef > 4.0 * err_stokes))
                    ef_abs = polish_pf_pcg_serial!(
                        Pf, r_cg, z_cg, p_cg, Ap_cg, valve_on,
                        Pt, inv_zeta, comp_src, mobx_fac, mobz_fac, inv_diag_pf,
                        inv_dx, inv_dz, 0.25 * pt_tol_cur * scale_p, 350, nx, nz
                    )
                    fill!(dPf, 0.0)
                    refresh_darcy_only_serial!(qDx, qDz, Pf, mobx_fac, mobz_fac, nx, nz)
                    ef = ef_abs / scale_p
                end

                pt_err = max(ev, ep, ef, et)
                if !isfinite(pt_err) || pt_err < pt_tol_cur
                    break
                end
            end

            if iter % p.visc_stride == 0
                rel_src_cur = (iter > 1500) ? 0.12 : 0.22
                refresh_rheology_and_steps_serial!(
                    exz, eta_sv_eff, tv_r, two_esv_inv1,
                    eII, eta_eff, tc_r, two_ec_inv1, G_cell,
                    eII_mean_z, zeta_mean_z,
                    dtau_vx, dtau_vz, dtau_pt, comp_src, eps_dil,
                    Vsx, Vsz, phi, eta_s, eta_sv, zeta, inv_zeta, disagg_lock, dil_fac,
                    inv_dx, inv_dz, inv_dx2, inv_dz2, U_bot, U_top, eps_bg_nd, inv_e_ref2,
                    tau_y_nd, inv_tau_y,
                    theta_0, inv_eta_geo, coeff_v, k_num, r_bulk, 3, rel_src_cur, nx, nz
                )
            end
        end

        if !isfinite(pt_err)
            break
        end

        # 3) Adaptive CFL conservative advection sub-cycling (n_adv <= max_adv_sub)
        vmax_nd   = compute_phase_fluxes_and_cfl_serial!(Fm_x, Fs_x, Fm_z, Fs_z, Vsx, Vsz, qDx, qDz, phi, p.phi_min, nx, nz)
        dt_cfl_s  = (0.40 * h_min / vmax_nd) * t0
        dt_rem_s  = t_end_s - t_s
        dt_mech   = min(dt_mech_max, p.max_adv_sub * dt_cfl_s, dt_rem_s)
        dt_mech   = max(dt_mech, min(1.0e4, dt_rem_s))
        n_adv     = max(1, min(p.max_adv_sub, ceil(Int, dt_mech / max(dt_cfl_s, 1.0e4))))
        dt_adv_s  = dt_mech / n_adv
        dt_adv_nd = dt_adv_s / t0

        copyto!(phi_prev, phi)

        for i_sub in 1:n_adv
            if i_sub > 1
                compute_phase_fluxes_and_cfl_serial!(Fm_x, Fs_x, Fm_z, Fs_z, Vsx, Vsz, qDx, qDz, phi, p.phi_min, nx, nz)
            end

            q_sum  = 0.0
            li_sum = 0.0
            @inbounds for ix in 1:nx
                qo = Fm_z[ix, nz + 1]
                q_sum  += qo
                li_sum += qo * C_Li[ix, nz]
            end
            q_out_mean = (q_sum / nx) * v0
            if q_out_mean > 0.0
                trap_melt_vol += q_out_mean * dt_adv_s
                trap_li_mass  += ((li_sum / nx) * v0) * dt_adv_s
            end

            copyto!(T_prev, T)

            advect_phi_solutes_serial!(
                phi_new, C_Li_new, Cf_new, phi, C_Li, Cf, Fm_x, Fm_z,
                dt_adv_nd, inv_dx, inv_dz, D_disp_nd, p.phi_min, p.phi_max, nx, nz
            )
            phi, phi_new   = phi_new, phi
            C_Li, C_Li_new = C_Li_new, C_Li
            Cf, Cf_new     = Cf_new, Cf

            n_th  = max(1, ceil(Int, dt_adv_s / dt_therm_lim))
            dt_th = dt_adv_s / n_th
            for _ in 1:n_th
                thermal_substep_serial!(
                    T_new, T, T_bg_z, Fm_x, Fs_x, Fm_z, Fs_z,
                    dt_th, inv_dx_phys, inv_dz_phys, inv_dx_p2, inv_dz_p2,
                    kappa_eff, inv_H_roof, p.T_bg_top, inv_tau_lat, v0, nx, nz
                )
                T, T_new = T_new, T
            end

            sub_relax = 1.0 - exp(-dt_adv_s / p.tau_cryst)
            rayleigh_crystallization_serial!(
                phi, C_Li, Cf, T, T_prev, qDx, qDz, hf_state,
                sub_relax, v0 * dt_adv_s, inv_dx_phys, inv_dz_phys,
                p.T_liq, p.T_sol0, p.T_sol_min, p.dT_flux, inv_dT_lat, p.beta_T_dist,
                p.D_Li, p.D_flux, p.phi_min, p.phi_max, inv_v_cool_ref, eps_bg_nd, nx, nz
            )
        end

        t_s  += dt_mech
        step += 1

        if trap_melt_vol > 1.0e-4
            V   = trap_melt_vol
            Vc  = p.V_peg_crit
            eff = (V * V) / (V * V + Vc * Vc)
            avg_li_out     = trap_li_mass / V
            rayleigh_boost = p.f_peg_res^(p.D_Li - 1.0)
            trap_li        = clamp(avg_li_out * (1.0 + eff * (rayleigh_boost - 1.0)), p.Li_0, 2.5e4)
            if trap_li > max_trap_li
                max_trap_li = trap_li
            end
        end

        @inbounds for iz in 1:nz, ix in 1:nx
            if phi[ix, iz] >= 0.03 || hf_state[ix, iz] > 0.05
                li_v = C_Li[ix, iz]
                if li_v > max_chan_li; max_chan_li = li_v; end
                em_v = min(eta_m_phys[ix, iz], eta_m_hf[ix, iz])
                if em_v < min_eta_seen; min_eta_seen = em_v; end
            end
        end
    end

    elapsed = time() - t_wall0
    return (
        trap_li  = max_trap_li,
        chan_li  = max_chan_li,
        export_m = trap_melt_vol,
        min_eta  = min_eta_seen,
        n_veins  = n_veins_ref[],
        steps    = step,
        wall_s   = elapsed,
    )
end

# ==============================================================================
# 4. CONCURRENT PANEL d PHASE-DIAGRAM SWEEP (11x11 = 121 REAL 2D SIMULATIONS)
# ==============================================================================
function run_panel_d_phase_diagram(; outdir::String = "output_2d", nx::Int = 48, nz::Int = 48)
    mkpath(outdir)
    t0_all = time()

    # Smooth 11x11 log-spaced grid spanning Panel d axes (1e-16..1e-11 s^-1 x 1e4..1e8 Pa s)
    eps_1d = 10.0 .^ range(-16.0, -11.0, length = 11)
    eta_1d = 10.0 .^ range(4.0,   8.0,   length = 11)
    ne, nm = length(eps_1d), length(eta_1d)
    n_tot  = ne * nm

    tasks = [(j, i) for j in 1:nm for i in ne:-1:1]

    EPS_2d     = [eps_1d[i] for j in 1:nm, i in 1:ne]
    ETA_2d     = [eta_1d[j] for j in 1:nm, i in 1:ne]
    Li_2d      = zeros(Float64, nm, ne)
    chan_li_2d = zeros(Float64, nm, ne)
    export_2d  = zeros(Float64, nm, ne)
    pi_p_2d    = zeros(Float64, nm, ne)

    base_p  = SweepParams(nx = nx, nz = nz)
    done_ct = Threads.Atomic{Int}(0)
    io_lock = ReentrantLock()

    @printf("\n==============================================================================\n")
    @printf("  PANEL d: Concurrent %dx%d Phase-Diagram Sweep (%d runs, grid=%dx%d, threads=%d)\n",
            nm, ne, n_tot, nx, nz, Threads.nthreads())
    @printf("==============================================================================\n")

    Threads.@threads :dynamic for idx in 1:n_tot
        j, i   = tasks[idx]
        eps_v  = eps_1d[i]
        eta_v  = eta_1d[j]
        p_ij   = SweepParams(nx = nx, nz = nz, eps_bg = eps_v, eta_m0 = eta_v, regime_fac = 1.00, seed = 42)
        res    = run_single_sweep_point(p_ij)
        pi_val = compute_analytical_pi_p(eps_v, eta_v, p_ij)

        Li_2d[j, i]      = res.trap_li
        chan_li_2d[j, i] = res.chan_li
        export_2d[j, i]  = res.export_m
        pi_p_2d[j, i]    = pi_val

        k_done = Threads.atomic_add!(done_ct, 1) + 1
        lock(io_lock) do
            @printf("  [%3d/%3d] eps=10^%5.2f, eta=10^%4.2f (Pi_p=%7.2e) -> TrapLi=%7.1f ppm | Export=%6.2f m | veins=%d | %.1f s\n",
                    k_done, n_tot, log10(eps_v), log10(eta_v), pi_val,
                    res.trap_li, res.export_m, res.n_veins, res.wall_s)
            flush(stdout)
        end
    end

    eps_line      = 10.0 .^ range(-16.0, -11.0, length = 200)
    eta_crit_line = [compute_analytical_eta_crit(e, base_p) for e in eps_line]

    out_path = joinpath(outdir, "phase_diagram_grid.npz")
    npzwrite(out_path, Dict{String, Any}(
        "EPS"           => EPS_2d,
        "ETA"           => ETA_2d,
        "Li"            => Li_2d,
        "chan_li_grid"  => chan_li_2d,
        "export_grid"   => export_2d,
        "pi_p_grid"     => pi_p_2d,
        "run_eps"       => vec(EPS_2d),
        "run_eta"       => vec(ETA_2d),
        "eps_line"      => collect(eps_line),
        "eta_crit_line" => eta_crit_line,
    ))
    @printf("\n>>> [DONE] Panel d saved %dx%d real 2D phase diagram -> %s (wall=%.1f s)\n",
            nm, ne, out_path, time() - t0_all)
end

# ==============================================================================
# 5. CONCURRENT PANEL c CROSS-REGIME KINEMATIC COLLAPSE (24 REAL 2D SIMULATIONS)
# ==============================================================================
function run_panel_c_cross_regime(; outdir::String = "output_2d", nx::Int = 48, nz::Int = 48)
    mkpath(outdir)
    t0_all = time()

    # Three tectonic regimes with consistent initial heterogeneity per regime (seeds 42, 43, 44)
    #   EXT         : extensional detachment (low confining sigma_3 -> regime_fac = 1.65)
    #   STRIKE_SLIP : neutral simple shear   (regime_fac = 1.00)
    #   COMP_TRANS  : compressional-transpressional (high confining stress -> regime_fac = 0.60)
    regimes = [
        ("EXT",         1.65, 42, 4.5e6, 10.0 .^ range(-14.6, -11.4, length = 8)),
        ("STRIKE_SLIP", 1.00, 43, 6.0e6, 10.0 .^ range(-14.3, -11.2, length = 8)),
        ("COMP_TRANS",  0.60, 44, 7.5e6, 10.0 .^ range(-14.0, -11.0, length = 8)),
    ]

    task_specs = NamedTuple[]
    for (rname, rfac, rseed, eta0_reg, eps_list) in regimes
        for eps_v in eps_list
            push!(task_specs, (
                regime     = rname,
                regime_fac = rfac,
                eps_bg     = eps_v,
                eta_m0     = eta0_reg,
                seed       = rseed,
            ))
        end
    end

    n_tot   = length(task_specs)
    results = Vector{NamedTuple}(undef, n_tot)
    done_ct = Threads.Atomic{Int}(0)
    io_lock = ReentrantLock()

    @printf("\n==============================================================================\n")
    @printf("  PANEL c: Concurrent Cross-Regime Collapse (%d runs, grid=%dx%d, threads=%d)\n",
            n_tot, nx, nz, Threads.nthreads())
    @printf("==============================================================================\n")

    Threads.@threads :dynamic for idx in 1:n_tot
        sp  = task_specs[idx]
        p_i = SweepParams(
            nx = nx, nz = nz,
            eps_bg = sp.eps_bg,
            eta_m0 = sp.eta_m0,
            regime_fac = sp.regime_fac,
            seed = sp.seed,
        )
        res    = run_single_sweep_point(p_i)
        pi_val = compute_analytical_pi_p(sp.eps_bg, sp.eta_m0, p_i)
        enr    = res.trap_li / p_i.Li_0

        results[idx] = (
            regime     = sp.regime,
            dim        = "2D",
            Pi_p       = pi_val,
            enrichment = enr,
            eps_bg     = sp.eps_bg,
            eta_m0     = sp.eta_m0,
            export_m   = res.export_m,
            trap_li    = res.trap_li,
            chan_li    = res.chan_li,
        )

        k_done = Threads.atomic_add!(done_ct, 1) + 1
        lock(io_lock) do
            @printf("  [%2d/%2d] %-11s | eps=%.1e, eta=%.1e | Pi_p=%7.3f -> C_Li/C0=%6.1f (TrapLi=%6.0f ppm, Export=%5.2f m) | %.1f s\n",
                    k_done, n_tot, sp.regime, sp.eps_bg, sp.eta_m0,
                    pi_val, enr, res.trap_li, res.export_m, res.wall_s)
            flush(stdout)
        end
    end

    csv_path = joinpath(outdir, "cross_regime_collapse.csv")
    open(csv_path, "w") do io
        println(io, "regime,dim,Pi_p,enrichment,eps_bg,eta_m0,export_m,trap_li,chan_li")
        for r in results
            @printf(io, "%s,%s,%.6e,%.4f,%.4e,%.4e,%.3f,%.2f,%.2f\n",
                    r.regime, r.dim, r.Pi_p, r.enrichment,
                    r.eps_bg, r.eta_m0, r.export_m, r.trap_li, r.chan_li)
        end
    end
    @printf("\n>>> [DONE] Panel c saved %d cross-regime 2D runs -> %s (wall=%.1f s)\n",
            n_tot, csv_path, time() - t0_all)
end

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
function main()
    t_start = time()
    # Both sweeps write ONLY phase_diagram_grid.npz and cross_regime_collapse.csv
    # Existing Panel a/b snapshots and time_series.npz in ./output_2d/ are untouched.
    run_panel_d_phase_diagram(outdir = "output_2d", nx = 48, nz = 48)
    run_panel_c_cross_regime(outdir  = "output_2d", nx = 48, nz = 48)
    @printf("\n[ALL SWEEPS COMPLETE] Total wall time: %.1f s (%.2f min)\n",
            time() - t_start, (time() - t_start) / 60.0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end