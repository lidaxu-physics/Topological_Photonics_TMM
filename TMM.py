"""
lattice_NxN_TMM.py
==================

Z-discretized TMM for an N_x x N_y site-ring lattice (Hafezi style):
- N_x * N_y site rings (CCW), each with N_z_site grid points.
- (N_x-1) * N_y horizontal link rings + N_x * (N_y-1) vertical link rings
  (CW), each with N_z_link grid points.
- Each link ring is anti-resonant with site rings (beta0 * eta = pi).
- Peierls phase on h-links to give uniform flux Phi0 per plaquette in
  Landau gauge: theta_h(i_y) = -2 * Phi0 * i_y, theta_v = 0.

Pure TMM: state vector contains field amplitudes E(z_k) on every grid point;
round-trip operator built from local propagation factors and 2x2 directional-
coupler scattering matrices; one linear solve per frequency.

Requirements
------------
    numpy, matplotlib, scipy

Usage (default 4x4 demo)
------------------------
    python lattice_NxN_TMM.py

This will run the 4x4 lattice, scan the central FSR window, and save
`lattice_NxN_TMM_demo.png` in the current directory.

Programmatic use
----------------
    import numpy as np
    from lattice_NxN_TMM import scan_spectrum_fast, solve_lattice_fast

    omegas = np.linspace(-0.1*2*np.pi, 0.1*2*np.pi, 8001)
    Td, Tt = scan_spectrum_fast(
        omegas, Nx=4, Ny=4,
        Phi0=np.pi/2, eta=0.5, half_fsr_offset=True,
        kappa_ex=0.10, kappa_J=0.561, alpha=1e-4,
    )
    E, sd, st = solve_lattice_fast(
        omega=0.025*2*np.pi, Nx=4, Ny=4, Phi0=np.pi/2,
        eta=0.5, half_fsr_offset=True,
        kappa_ex=0.10, kappa_J=0.561, alpha=1e-4,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ============================================================================
# Discretization parameters (defaults)
# ============================================================================
NZ_SITE_DEFAULT = 16
NZ_LINK_DEFAULT = 16

# Site-ring DC slot grid positions (as in the 2x2 build)
SLOT_BUS    = 0
SLOT_RIGHT  = 4
SLOT_TOP    = 8
SLOT_BOTTOM = 10
SLOT_LEFT   = 12


# ============================================================================
# Indexing helpers built on demand for given lattice size
# ============================================================================
def make_lattice_indices(Nx, Ny, Nz_site=NZ_SITE_DEFAULT, Nz_link=NZ_LINK_DEFAULT):
    """
    Build state-vector indexing helpers for an Nx*Ny lattice.

    Returns
    -------
    site_idx, link_idx : callable
        site_idx(ix, iy, k) -> int
        link_idx(name, k)   -> int  where name is e.g. "H_2_3" or "V_0_1"
    state_size : int
    h_link_names : list of names of horizontal links (between (ix,iy) and (ix+1,iy))
    v_link_names : list of names of vertical links   (between (ix,iy) and (ix,iy+1))
    """
    site_offsets = {}
    offset = 0
    for iy in range(Ny):
        for ix in range(Nx):
            site_offsets[(ix, iy)] = offset
            offset += Nz_site

    link_offsets = {}
    h_link_names = []
    for iy in range(Ny):
        for ix in range(Nx - 1):
            name = f"H_{ix}_{iy}"
            link_offsets[name] = offset
            h_link_names.append(name)
            offset += Nz_link

    v_link_names = []
    for iy in range(Ny - 1):
        for ix in range(Nx):
            name = f"V_{ix}_{iy}"
            link_offsets[name] = offset
            v_link_names.append(name)
            offset += Nz_link

    state_size = offset

    def site_idx(ix, iy, k):
        return site_offsets[(ix, iy)] + (k % Nz_site)

    def link_idx(name, k):
        return link_offsets[name] + (k % Nz_link)

    return (site_idx, link_idx, state_size,
            h_link_names, v_link_names)


# ============================================================================
# Solver
# ============================================================================
def solve_lattice(
    omega,
    Nx, Ny,
    Phi0=0.0,
    eta=0.5,
    half_fsr_offset=True,
    kappa_ex=0.10,
    kappa_J=0.561,
    alpha=1e-4,
    bus_in=(0, 0),
    bus_drop=None,
    Nz_site=NZ_SITE_DEFAULT,
    Nz_link=NZ_LINK_DEFAULT,
):
    """
    Steady-state TMM solve for an Nx*Ny Hafezi lattice at one frequency.

    Returns
    -------
    E : (state_size,) complex
    s_drop, s_thru : complex
    """
    if bus_drop is None:
        bus_drop = (0, Ny - 1)

    site_idx, link_idx, state_size, h_link_names, v_link_names = \
        make_lattice_indices(Nx, Ny, Nz_site, Nz_link)

    L_site = 1.0
    L_link = L_site + eta
    dz_site = L_site / Nz_site
    dz_link = L_link / Nz_link

    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)

    beta0_eta = np.pi if half_fsr_offset else 0.0
    extra_per_step_phase = beta0_eta / Nz_link
    p_link = np.exp(1j * omega * dz_link + 1j * extra_per_step_phase
                     - alpha * dz_link / 2.0)

    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    t_J = np.sqrt(1.0 - kappa_J ** 2)

    M = np.eye(state_size, dtype=complex)
    s = np.zeros(state_size, dtype=complex)
    s_in_amp = 1.0

    # Peierls phase per link.
    # Convention: only horizontal links carry asymmetry, theta_h(iy) = -2*Phi0*iy.
    # Vertical links: 0.
    def link_theta(name):
        kind, ix_str, iy_str = name.split("_")
        iy = int(iy_str)
        if kind == "H":
            return -2.0 * Phi0 * iy
        else:
            return 0.0

    half_link = Nz_link // 2

    def link_extras(theta):
        extra = np.zeros(Nz_link)
        extra[1] = +theta / 2.0
        extra[half_link + 1] = -theta / 2.0
        return extra

    extras = {name: link_extras(link_theta(name))
              for name in h_link_names + v_link_names}

    # Build site_neighbors per site: which slots are active and what they do.
    # For each (ix, iy):
    #   if ix > 0:        SLOT_LEFT   couples to H_{ix-1}_{iy} far end (right site)
    #   if ix < Nx - 1:   SLOT_RIGHT  couples to H_{ix}_{iy}   near end (left site)
    #   if iy > 0:        SLOT_BOTTOM couples to V_{ix}_{iy-1} far end (top site)
    #   if iy < Ny - 1:   SLOT_TOP    couples to V_{ix}_{iy}   near end (bottom site)
    # Bus replacements:
    #   if (ix, iy) == bus_in:   SLOT_BUS = bus_in
    #   if (ix, iy) == bus_drop: SLOT_BUS = bus_drop
    site_neighbors = {}
    for iy in range(Ny):
        for ix in range(Nx):
            slots = {}
            if (ix, iy) == bus_in:
                slots[SLOT_BUS] = ("bus_in", None)
            elif (ix, iy) == bus_drop:
                slots[SLOT_BUS] = ("bus_drop", None)
            if ix > 0:
                slots[SLOT_LEFT] = ("link", f"H_{ix-1}_{iy}", "far")
            if ix < Nx - 1:
                slots[SLOT_RIGHT] = ("link", f"H_{ix}_{iy}", "near")
            if iy > 0:
                slots[SLOT_BOTTOM] = ("link", f"V_{ix}_{iy-1}", "far")
            if iy < Ny - 1:
                slots[SLOT_TOP] = ("link", f"V_{ix}_{iy}", "near")
            site_neighbors[(ix, iy)] = slots

    # Build link_sites: which sites each link connects (near & far ends).
    link_sites = {}
    for name in h_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        # near = (ix, iy) at SLOT_RIGHT, far = (ix+1, iy) at SLOT_LEFT
        link_sites[name] = {"near": (ix, iy, SLOT_RIGHT),
                             "far":  (ix + 1, iy, SLOT_LEFT)}
    for name in v_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        # near = (ix, iy) at SLOT_TOP, far = (ix, iy+1) at SLOT_BOTTOM
        link_sites[name] = {"near": (ix, iy, SLOT_TOP),
                             "far":  (ix, iy + 1, SLOT_BOTTOM)}

    # ------------------------------------------------------------------
    # Site ring equations
    # ------------------------------------------------------------------
    for (ix, iy), slots in site_neighbors.items():
        for k in range(Nz_site):
            k_prev = (k - 1) % Nz_site
            row = site_idx(ix, iy, k)
            if k in slots:
                action = slots[k]
                kind = action[0]
                if kind == "bus_in":
                    M[row, site_idx(ix, iy, k_prev)] = -t_ex * p_site
                    s[row] = 1j * kappa_ex * s_in_amp
                elif kind == "bus_drop":
                    M[row, site_idx(ix, iy, k_prev)] = -t_ex * p_site
                elif kind == "link":
                    _, link_name, end = action
                    if end == "near":
                        link_k_dc = 0
                        link_k_prev = Nz_link - 1
                    else:
                        link_k_dc = half_link
                        link_k_prev = half_link - 1
                    extra_factor = np.exp(1j * extras[link_name][link_k_dc])
                    M[row, site_idx(ix, iy, k_prev)] = -t_J * p_site
                    M[row, link_idx(link_name, link_k_prev)] = (
                        -1j * kappa_J * p_link * extra_factor
                    )
            else:
                M[row, site_idx(ix, iy, k_prev)] = -p_site

    # ------------------------------------------------------------------
    # Link ring equations
    # ------------------------------------------------------------------
    for name, ends in link_sites.items():
        for k in range(Nz_link):
            k_prev = (k - 1) % Nz_link
            row = link_idx(name, k)
            extra_factor = np.exp(1j * extras[name][k])
            if k == 0 or k == half_link:
                site_info = ends["near"] if k == 0 else ends["far"]
                site_ix, site_iy, site_slot = site_info
                site_k_prev = (site_slot - 1) % Nz_site
                M[row, link_idx(name, k_prev)] = -t_J * p_link * extra_factor
                M[row, site_idx(site_ix, site_iy, site_k_prev)] = (
                    -1j * kappa_J * p_site
                )
            else:
                M[row, link_idx(name, k_prev)] = -p_link * extra_factor

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    E = np.linalg.solve(M, s)

    # Outputs
    i_in, j_in = bus_in
    e_in_at_bus_in = p_site * E[site_idx(i_in, j_in, Nz_site - 1)]
    s_thru = t_ex * s_in_amp + 1j * kappa_ex * e_in_at_bus_in

    i_d, j_d = bus_drop
    e_in_at_bus_drop = p_site * E[site_idx(i_d, j_d, Nz_site - 1)]
    s_drop = 1j * kappa_ex * e_in_at_bus_drop

    return E, s_drop, s_thru


def _build_lattice_template(Nx, Ny, Phi0, eta, kappa_ex, kappa_J,
                              bus_in, bus_drop, Nz_site, Nz_link):
    """
    Precompute the sparsity pattern and per-entry "type" coefficients for the
    TMM matrix. Returns lists of (row, col, kind, payload) that allow fast
    re-evaluation at each omega.

    Kinds and their per-frequency value:
        ("p_site", coeff)         -> coeff * p_site
        ("p_link_extra", coeff, extra_phase)
                                  -> coeff * p_link * exp(1j*extra_phase)
        ("p_site_const", coeff)   -> coeff * p_site (same as p_site, kept for clarity)
        ("const", value)          -> value (no omega dependence; used for identity)
    Source vector entries are constant (only s_in_amp * 1j*kappa_ex on bus_in row).
    """
    site_idx, link_idx, state_size, h_link_names, v_link_names = \
        make_lattice_indices(Nx, Ny, Nz_site, Nz_link)

    half_link = Nz_link // 2

    # Peierls phase per link
    def link_theta(name):
        kind, ix_str, iy_str = name.split("_")
        iy = int(iy_str)
        if kind == "H":
            return -2.0 * Phi0 * iy
        else:
            return 0.0

    def link_extras(theta):
        extra = np.zeros(Nz_link)
        extra[1] = +theta / 2.0
        extra[half_link + 1] = -theta / 2.0
        return extra

    extras = {name: link_extras(link_theta(name))
              for name in h_link_names + v_link_names}

    # Site neighbors
    site_neighbors = {}
    for iy in range(Ny):
        for ix in range(Nx):
            slots = {}
            if (ix, iy) == bus_in:
                slots[SLOT_BUS] = ("bus_in", None)
            elif (ix, iy) == bus_drop:
                slots[SLOT_BUS] = ("bus_drop", None)
            if ix > 0:
                slots[SLOT_LEFT] = ("link", f"H_{ix-1}_{iy}", "far")
            if ix < Nx - 1:
                slots[SLOT_RIGHT] = ("link", f"H_{ix}_{iy}", "near")
            if iy > 0:
                slots[SLOT_BOTTOM] = ("link", f"V_{ix}_{iy-1}", "far")
            if iy < Ny - 1:
                slots[SLOT_TOP] = ("link", f"V_{ix}_{iy}", "near")
            site_neighbors[(ix, iy)] = slots

    link_sites = {}
    for name in h_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        link_sites[name] = {"near": (ix, iy, SLOT_RIGHT),
                             "far":  (ix + 1, iy, SLOT_LEFT)}
    for name in v_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        link_sites[name] = {"near": (ix, iy, SLOT_TOP),
                             "far":  (ix, iy + 1, SLOT_BOTTOM)}

    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    t_J = np.sqrt(1.0 - kappa_J ** 2)

    # We'll build entries in COO form.
    # For each entry: store (row, col, kind, payload).
    entries = []
    # And entries that are diagonal "I" identity contributions: handle separately
    # by adding 1 to the diagonal at the end.
    src_rows = []
    src_vals = []

    # --- Site rings ---
    for (ix, iy), slots in site_neighbors.items():
        for k in range(Nz_site):
            k_prev = (k - 1) % Nz_site
            row = site_idx(ix, iy, k)
            if k in slots:
                action = slots[k]
                kind = action[0]
                if kind == "bus_in":
                    entries.append((row, site_idx(ix, iy, k_prev),
                                     "p_site", -t_ex))
                    src_rows.append(row)
                    src_vals.append(1j * kappa_ex)  # * s_in_amp = 1
                elif kind == "bus_drop":
                    entries.append((row, site_idx(ix, iy, k_prev),
                                     "p_site", -t_ex))
                elif kind == "link":
                    _, link_name, end = action
                    if end == "near":
                        link_k_dc = 0
                        link_k_prev = Nz_link - 1
                    else:
                        link_k_dc = half_link
                        link_k_prev = half_link - 1
                    extra_phase_dc = extras[link_name][link_k_dc]
                    entries.append((row, site_idx(ix, iy, k_prev),
                                     "p_site", -t_J))
                    entries.append((row, link_idx(link_name, link_k_prev),
                                     "p_link_extra", -1j * kappa_J,
                                     extra_phase_dc))
            else:
                entries.append((row, site_idx(ix, iy, k_prev),
                                 "p_site", -1.0))

    # --- Link rings ---
    for name, ends in link_sites.items():
        for k in range(Nz_link):
            k_prev = (k - 1) % Nz_link
            row = link_idx(name, k)
            extra_phase_k = extras[name][k]
            if k == 0 or k == half_link:
                site_info = ends["near"] if k == 0 else ends["far"]
                site_ix, site_iy, site_slot = site_info
                site_k_prev = (site_slot - 1) % Nz_site
                entries.append((row, link_idx(name, k_prev),
                                 "p_link_extra", -t_J, extra_phase_k))
                entries.append((row, site_idx(site_ix, site_iy, site_k_prev),
                                 "p_site", -1j * kappa_J))
            else:
                entries.append((row, link_idx(name, k_prev),
                                 "p_link_extra", -1.0, extra_phase_k))

    return entries, src_rows, src_vals, state_size


def solve_lattice_fast(
    omega,
    Nx, Ny,
    Phi0=0.0,
    eta=0.5,
    half_fsr_offset=True,
    kappa_ex=0.10,
    kappa_J=0.561,
    alpha=1e-4,
    bus_in=(0, 0),
    bus_drop=None,
    Nz_site=NZ_SITE_DEFAULT,
    Nz_link=NZ_LINK_DEFAULT,
    _cache={},
):
    """
    Fast sparse-solver version of solve_lattice. Caches the matrix template
    keyed on (Nx, Ny, Phi0, kappa_ex, kappa_J, bus_in, bus_drop, Nz_site, Nz_link)
    so that only the omega-dependent values change per call.

    Returns (E, s_drop, s_thru) just like solve_lattice.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    if bus_drop is None:
        bus_drop = (0, Ny - 1)

    cache_key = (Nx, Ny, Phi0, eta, kappa_ex, kappa_J, bus_in, bus_drop,
                  Nz_site, Nz_link, half_fsr_offset)
    if cache_key not in _cache:
        entries, src_rows, src_vals, state_size = _build_lattice_template(
            Nx, Ny, Phi0, eta, kappa_ex, kappa_J,
            bus_in, bus_drop, Nz_site, Nz_link,
        )
        # Pre-extract arrays of (row, col) and the per-entry data for fast eval
        rows = np.array([e[0] for e in entries], dtype=np.int32)
        cols = np.array([e[1] for e in entries], dtype=np.int32)
        kinds = np.array([0 if e[2] == "p_site" else 1 for e in entries],
                          dtype=np.int8)
        coeffs = np.array([e[3] for e in entries], dtype=complex)
        # extra_phase only meaningful for kind=1; pad with 0 for kind=0
        extras_arr = np.zeros(len(entries), dtype=float)
        for i, e in enumerate(entries):
            if e[2] == "p_link_extra":
                extras_arr[i] = e[4]
        # Add diagonal identity contributions (state_size of them)
        diag_rows = np.arange(state_size, dtype=np.int32)
        # Source vector
        src_rows_arr = np.array(src_rows, dtype=np.int32)
        src_vals_arr = np.array(src_vals, dtype=complex)

        _cache[cache_key] = dict(
            rows=rows, cols=cols, kinds=kinds, coeffs=coeffs,
            extras_arr=extras_arr, diag_rows=diag_rows,
            src_rows_arr=src_rows_arr, src_vals_arr=src_vals_arr,
            state_size=state_size,
        )

    cache = _cache[cache_key]
    rows = cache["rows"]; cols = cache["cols"]
    kinds = cache["kinds"]; coeffs = cache["coeffs"]
    extras_arr = cache["extras_arr"]; diag_rows = cache["diag_rows"]
    src_rows_arr = cache["src_rows_arr"]; src_vals_arr = cache["src_vals_arr"]
    state_size = cache["state_size"]

    # Compute frequency-dependent factors
    L_site = 1.0
    L_link = L_site + eta
    dz_site = L_site / Nz_site
    dz_link = L_link / Nz_link
    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)
    beta0_eta = np.pi if half_fsr_offset else 0.0
    extra_per_step_phase = beta0_eta / Nz_link
    p_link_base = np.exp(1j * omega * dz_link + 1j * extra_per_step_phase
                          - alpha * dz_link / 2.0)

    # Build entry values: kind=0 -> coeff * p_site
    #                     kind=1 -> coeff * p_link_base * exp(1j*extra)
    vals = np.where(
        kinds == 0,
        coeffs * p_site,
        coeffs * p_link_base * np.exp(1j * extras_arr),
    )

    # Append identity diagonal (+1 on every diagonal entry)
    rows_full = np.concatenate([rows, diag_rows])
    cols_full = np.concatenate([cols, diag_rows])
    vals_full = np.concatenate([vals, np.ones(state_size, dtype=complex)])

    M = csc_matrix((vals_full, (rows_full, cols_full)),
                    shape=(state_size, state_size))

    # RHS
    s = np.zeros(state_size, dtype=complex)
    s[src_rows_arr] = src_vals_arr

    # Solve
    lu = splu(M)
    E = lu.solve(s)

    # Outputs
    site_idx, _, _, _, _ = make_lattice_indices(Nx, Ny, Nz_site, Nz_link)
    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    i_in, j_in = bus_in
    e_in_at_bus_in = p_site * E[site_idx(i_in, j_in, Nz_site - 1)]
    s_thru = t_ex * 1.0 + 1j * kappa_ex * e_in_at_bus_in
    i_d, j_d = bus_drop
    e_in_at_bus_drop = p_site * E[site_idx(i_d, j_d, Nz_site - 1)]
    s_drop = 1j * kappa_ex * e_in_at_bus_drop

    return E, s_drop, s_thru


def scan_spectrum_fast(omegas, Nx, Ny, **kwargs):
    Td = np.empty(len(omegas))
    Tt = np.empty(len(omegas))
    for k, w in enumerate(omegas):
        _, sd, st = solve_lattice_fast(w, Nx, Ny, **kwargs)
        Td[k] = np.abs(sd) ** 2
        Tt[k] = np.abs(st) ** 2
    return Td, Tt


# ============================================================================
# Visualization
# ============================================================================
def _rounded_square_perimeter(cx, cy, half_side, corner_radius, n_pts=200):
    """Generate (x,y) points along a rounded-square perimeter, CCW from right-mid."""
    L_straight = 2 * (half_side - corner_radius)
    L_corner = (np.pi / 2) * corner_radius
    P = 4 * L_straight + 4 * L_corner
    s_arr = np.linspace(0, P, n_pts, endpoint=False)
    xs = np.zeros(n_pts); ys = np.zeros(n_pts)
    h = half_side; rc = corner_radius
    for i, s in enumerate(s_arr):
        s_local = s
        if s_local < L_straight:
            xs[i] = cx + h
            ys[i] = cy - (h - rc) + s_local
            continue
        s_local -= L_straight
        if s_local < L_corner:
            theta = s_local / rc
            xs[i] = (cx + h - rc) + rc * np.cos(theta)
            ys[i] = (cy + h - rc) + rc * np.sin(theta)
            continue
        s_local -= L_corner
        if s_local < L_straight:
            xs[i] = (cx + h - rc) - s_local
            ys[i] = cy + h
            continue
        s_local -= L_straight
        if s_local < L_corner:
            theta = np.pi / 2 + s_local / rc
            xs[i] = (cx - h + rc) + rc * np.cos(theta)
            ys[i] = (cy + h - rc) + rc * np.sin(theta)
            continue
        s_local -= L_corner
        if s_local < L_straight:
            xs[i] = cx - h
            ys[i] = (cy + h - rc) - s_local
            continue
        s_local -= L_straight
        if s_local < L_corner:
            theta = np.pi + s_local / rc
            xs[i] = (cx - h + rc) + rc * np.cos(theta)
            ys[i] = (cy - h + rc) + rc * np.sin(theta)
            continue
        s_local -= L_corner
        if s_local < L_straight:
            xs[i] = (cx - h + rc) + s_local
            ys[i] = cy - h
            continue
        s_local -= L_straight
        theta = 3 * np.pi / 2 + s_local / rc
        xs[i] = (cx + h - rc) + rc * np.cos(theta)
        ys[i] = (cy - h + rc) + rc * np.sin(theta)
    return xs, ys


def _draw_ring_with_intensity(ax, center, half_side, intensities, I_max,
                                lw=2.6, cmap=plt.cm.inferno, zorder=3,
                                phase_offset=0.0, chirality=+1,
                                corner_radius=None):
    if corner_radius is None:
        corner_radius = 0.30 * half_side
    Nz = len(intensities)
    n_fine = max(200, 8 * Nz)
    xs, ys = _rounded_square_perimeter(center[0], center[1], half_side,
                                         corner_radius, n_pts=n_fine)
    s_frac = np.arange(n_fine) / n_fine
    if chirality > 0:
        k_continuous = ((s_frac - phase_offset) % 1.0) * Nz
    else:
        k_continuous = ((phase_offset - s_frac) % 1.0) * Nz
    k0 = np.floor(k_continuous).astype(int) % Nz
    k1 = (k0 + 1) % Nz
    frac = k_continuous - np.floor(k_continuous)
    I_interp = (1 - frac) * intensities[k0] + frac * intensities[k1]
    segs = np.stack([np.column_stack([xs, ys]),
                      np.column_stack([np.roll(xs, -1), np.roll(ys, -1)])],
                     axis=1)
    colors = cmap(I_interp / I_max if I_max > 0 else np.zeros_like(I_interp))
    lc = LineCollection(segs, colors=colors, linewidths=lw,
                         capstyle="butt", joinstyle="miter", zorder=zorder)
    ax.add_collection(lc)


def plot_field_distribution(ax, E, Nx, Ny, title="", I_max=None,
                              Nz_site=NZ_SITE_DEFAULT, Nz_link=NZ_LINK_DEFAULT,
                              bus_in=(0, 0), bus_drop=None):
    """Plot the field intensity distribution on the NxN lattice."""
    if bus_drop is None:
        bus_drop = (0, Ny - 1)

    site_idx, link_idx, _, h_link_names, v_link_names = \
        make_lattice_indices(Nx, Ny, Nz_site, Nz_link)

    # Collect intensities
    site_grids = {}
    for iy in range(Ny):
        for ix in range(Nx):
            idxs = [site_idx(ix, iy, k) for k in range(Nz_site)]
            site_grids[(ix, iy)] = np.abs(E[idxs]) ** 2

    link_grids = {}
    for name in h_link_names + v_link_names:
        idxs = [link_idx(name, k) for k in range(Nz_link)]
        link_grids[name] = np.abs(E[idxs]) ** 2

    all_vals = np.concatenate(list(site_grids.values())
                                + list(link_grids.values()))
    if I_max is None:
        I_max = float(all_vals.max()) if all_vals.max() > 0 else 1.0

    # Lattice positions: site (ix, iy) at (ix, iy)
    site_pos = {(ix, iy): (float(ix), float(iy))
                for iy in range(Ny) for ix in range(Nx)}
    link_pos = {}
    for name in h_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        link_pos[name] = (ix + 0.5, float(iy))
    for name in v_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        link_pos[name] = (float(ix), iy + 0.5)

    half_side = 0.16
    corner_r = 0.05
    lw_ring = 2.2

    ax.set_facecolor("#0a0e1a")

    # Bond lines through link centers
    bond_color = "#2a3550"
    for name in h_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        ax.plot([ix, ix + 1], [iy, iy], color=bond_color, lw=0.5, zorder=1)
    for name in v_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        ax.plot([ix, ix], [iy, iy + 1], color=bond_color, lw=0.5, zorder=1)

    # Site rings: per-ring phase_offset so SLOT_BUS aligns with bus location.
    # IN ring (bus_in): bus drawn below -> SLOT_BUS at bottom -> phase_offset = 0.75
    # OUT ring (bus_drop): bus drawn above -> SLOT_BUS at top -> phase_offset = 0.25
    # Other site rings: default phase_offset = 0.25 (bus would be at top, but no bus)
    for (ix, iy), grid_I in site_grids.items():
        if (ix, iy) == bus_in:
            ph = 0.75
        else:
            ph = 0.25
        _draw_ring_with_intensity(ax, site_pos[(ix, iy)], half_side, grid_I,
                                    I_max, lw=lw_ring, zorder=3,
                                    phase_offset=ph, chirality=+1,
                                    corner_radius=corner_r)

    # Link rings: phase_offset 0.75 for vertical (grid 0 at bottom),
    #             phase_offset 0.5  for horizontal (grid 0 at left).
    # Chirality = -1 (CW).
    for name, grid_I in link_grids.items():
        kind = name.split("_")[0]
        ph = 0.75 if kind == "V" else 0.5
        _draw_ring_with_intensity(ax, link_pos[name], half_side, grid_I,
                                    I_max, lw=lw_ring, zorder=2,
                                    phase_offset=ph, chirality=-1,
                                    corner_radius=corner_r)

    # IN / OUT labels and bus waveguides
    in_pos = site_pos[bus_in]
    out_pos = site_pos[bus_drop]

    bus_color = "#7aa2cf"
    bus_lw = 1.4
    bus_gap = 0.04
    coupling_half_len = 0.18
    bus_bend_r = 0.07
    tail_len = 0.30

    def draw_horseshoe_bus(ring_center, side, left_label, right_label,
                            left_arrow_in=False, right_arrow_out=False,
                            left_arrow_out=False, right_arrow_in=False):
        cx, cy = ring_center
        sign = -1 if side == "lower" else +1
        y_couple = cy + sign * (half_side + bus_gap)
        x_L_couple = cx - coupling_half_len
        x_R_couple = cx + coupling_half_len
        x_L_bend_c = x_L_couple
        y_L_bend_c = y_couple + sign * bus_bend_r
        x_R_bend_c = x_R_couple
        y_R_bend_c = y_couple + sign * bus_bend_r
        x_L_tail = x_L_bend_c - bus_bend_r
        x_R_tail = x_R_bend_c + bus_bend_r
        y_tail_end = y_L_bend_c + sign * tail_len

        ax.plot([x_L_couple, x_R_couple], [y_couple, y_couple],
                 color=bus_color, lw=bus_lw, zorder=2)
        if side == "lower":
            angles = np.linspace(np.pi / 2, np.pi, 32)
        else:
            angles = np.linspace(np.pi, 3 * np.pi / 2, 32)
        ax.plot(x_L_bend_c + bus_bend_r * np.cos(angles),
                 y_L_bend_c + bus_bend_r * np.sin(angles),
                 color=bus_color, lw=bus_lw, zorder=2)
        if side == "lower":
            angles = np.linspace(0, np.pi / 2, 32)
        else:
            angles = np.linspace(3 * np.pi / 2, 2 * np.pi, 32)
        ax.plot(x_R_bend_c + bus_bend_r * np.cos(angles),
                 y_R_bend_c + bus_bend_r * np.sin(angles),
                 color=bus_color, lw=bus_lw, zorder=2)
        ax.plot([x_L_tail, x_L_tail], [y_L_bend_c, y_tail_end],
                 color=bus_color, lw=bus_lw, zorder=2)
        ax.plot([x_R_tail, x_R_tail], [y_R_bend_c, y_tail_end],
                 color=bus_color, lw=bus_lw, zorder=2)

        if left_arrow_in:
            ax.annotate("", xy=(x_L_tail, y_L_bend_c - sign * 0.02),
                         xytext=(x_L_tail, y_tail_end),
                         arrowprops=dict(arrowstyle="->", color=bus_color, lw=bus_lw))
        if left_arrow_out:
            ax.annotate("", xy=(x_L_tail, y_tail_end),
                         xytext=(x_L_tail, y_L_bend_c - sign * 0.02),
                         arrowprops=dict(arrowstyle="->", color=bus_color, lw=bus_lw))
        if right_arrow_out:
            ax.annotate("", xy=(x_R_tail, y_tail_end),
                         xytext=(x_R_tail, y_R_bend_c - sign * 0.02),
                         arrowprops=dict(arrowstyle="->", color=bus_color, lw=bus_lw))
        if right_arrow_in:
            ax.annotate("", xy=(x_R_tail, y_R_bend_c - sign * 0.02),
                         xytext=(x_R_tail, y_tail_end),
                         arrowprops=dict(arrowstyle="->", color=bus_color, lw=bus_lw))

        label_va = "top" if side == "lower" else "bottom"
        ax.text(x_L_tail, y_tail_end + sign * 0.04, left_label,
                 color=bus_color, fontsize=6, ha="center", va=label_va)
        ax.text(x_R_tail, y_tail_end + sign * 0.04, right_label,
                 color=bus_color, fontsize=6, ha="center", va=label_va)

    # IN bus: lower side, light flows rightward (CCW ring at bottom)
    draw_horseshoe_bus(in_pos, "lower", "input", "through",
                        left_arrow_in=True, right_arrow_out=True)
    # OUT bus: upper side, light flows leftward (CCW ring at top)
    draw_horseshoe_bus(out_pos, "upper", "drop", "add",
                        left_arrow_out=True)

    ax.text(in_pos[0], in_pos[1], "IN", ha="center", va="center",
            color="white", fontsize=6, fontweight="bold", zorder=4)
    ax.text(out_pos[0], out_pos[1], "OUT", ha="center", va="center",
            color="white", fontsize=6, fontweight="bold", zorder=4)

    pad_x = 0.5
    pad_y = 0.7
    ax.set_xlim(-pad_x, Nx - 1 + pad_x)
    ax.set_ylim(-pad_y, Ny - 1 + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3550")
    ax.set_title(title, fontsize=9, color="white", pad=4)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,
                                norm=plt.Normalize(vmin=0, vmax=I_max))
    return sm


# ============================================================================
# Demo: 4x4 lattice
# ============================================================================
if __name__ == "__main__":
    from scipy.signal import find_peaks

    Nx, Ny = 4, 4
    Phi0 = np.pi / 2
    common = dict(eta=0.5, half_fsr_offset=True,
                   kappa_ex=0.10, kappa_J=0.561, alpha=1e-4)

    omegas = np.linspace(-0.1 * 2 * np.pi, 0.1 * 2 * np.pi, 8001)
    print(f"Scanning 4x4 lattice at Phi0 = pi/2, narrow range "
          f"omega/(2pi) in [-0.1, 0.1], {len(omegas)} frequency points...")
    Td, Tt = scan_spectrum_fast(omegas, Nx, Ny, Phi0=Phi0, **common)
    peaks, _ = find_peaks(Td, height=0.005 * Td.max(), distance=10)
    peak_omegas = sorted(omegas[peaks])
    print(f"Found {len(peak_omegas)} peaks (T_drop max = {Td.max():.3f})")

    # Pick the peak closest to omega/(2pi) = +0.025
    target = 0.025 * 2 * np.pi
    selected = min(peak_omegas, key=lambda p: abs(p - target))
    print(f"Showing field distribution at omega = {selected:.4f}  "
          f"(omega/(2pi) = {selected/(2*np.pi):.4f})")

    fig = plt.figure(figsize=(7, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5])

    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.plot(omegas / (2 * np.pi), Td, lw=1.0, label=r"$T_{\rm drop}$")
    ax_s.plot(omegas / (2 * np.pi), Tt, lw=0.8, color="C3", alpha=0.6,
               label=r"$T_{\rm thru}$")
    ax_s.axvline(selected / (2 * np.pi), color="C2", ls=":", alpha=0.7)
    ax_s.set_xlim(-0.1, 0.1)
    ax_s.set_ylim(-0.05, 1.10)
    ax_s.set_xlabel(r"$\omega/(2\pi)$  (FSR units)")
    ax_s.set_ylabel("transmission")
    ax_s.set_title(rf"4$\times$4 lattice, $\Phi_0 = \pi/2$,  "
                    rf"$\kappa_{{\rm ex}}={common['kappa_ex']}$,  "
                    rf"$\kappa_J={common['kappa_J']}$",
                    fontsize=10)
    ax_s.legend(loc="upper right", fontsize=8)
    ax_s.grid(True, alpha=0.3)

    E_peak, _, _ = solve_lattice_fast(selected, Nx, Ny, Phi0=Phi0, **common)
    ax_v = fig.add_subplot(gs[1, 0])
    sm = plot_field_distribution(ax_v, E_peak, Nx, Ny,
                                   title=rf"$\omega/(2\pi) = "
                                          rf"{selected/(2*np.pi):.4f}$")
    cbar = fig.colorbar(sm, ax=ax_v, fraction=0.04, pad=0.04,
                         label=r"$|E|^2$ (normalized)")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(rf"4$\times$4 Hafezi lattice at $\Phi_0=\pi/2$",
                  fontsize=12, y=0.995)
    plt.tight_layout()
    out_path = "lattice_NxN_TMM_demo.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()