"""
TMM.py
======
Lida Xu's IQH-Lattice TMM Explorer — v1.2
Standalone desktop application matching the Linear.py UI conventions.

Run:   python TMM.py
Build: pyinstaller --onefile --windowed TMM.py

Pure z-discretized TMM throughout — no Hamiltonian, no J in the simulation
itself; only field amplitudes E(z_k) on every grid point and the round-trip
operator built from local propagation factors plus 2x2 DC scattering matrices.
Sparse-LU solve gives ~30x speedup over dense.

Bus convention (fixed):  IN at site (0, 0),  OUT at site (0, Ny - 1).

v1.0.1 fixes:
  - Spectrum x-axis now follows the sweep window (omin / omax) instead of
    being pinned to [-0.5, 0.5].
  - Field panel no longer shrinks on each peak click. The colorbar now lives
    in its own dedicated axis (self.cax_lat) created once at startup, so
    fig.colorbar() can no longer steal space from the main lattice axis.
"""

import sys, os, time

if getattr(sys, 'frozen', False):
    _BUNDLE_DIR = sys._MEIPASS
    sys.path.insert(0, _BUNDLE_DIR)
else:
    _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QGroupBox, QSizePolicy, QSplitter,
    QStatusBar, QFrame, QFileDialog, QLineEdit, QMessageBox,
    QCheckBox, QDialog, QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

import matplotlib
if matplotlib.get_backend() != 'Qt5Agg':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ── Colours (match Linear.py) ────────────────────────────────────────────────
DARK_BG   = '#08090d'
PANEL_BG  = '#0a0c14'
CARD_BG   = '#171c2e'
GRID_COL  = '#1a1f2e'
SPINE_COL = '#1e2230'
TEXT_DIM  = '#4a5270'
TEXT_COL  = '#c8d0e7'
ACCENT    = '#00e5ff'

# ── Discretization ───────────────────────────────────────────────────────────
NZ_SITE = 16
NZ_LINK = 16

# Site-ring DC slot grid positions
SLOT_BUS    = 0
SLOT_RIGHT  = 4
SLOT_TOP    = 8
SLOT_BOTTOM = 10
SLOT_LEFT   = 12


def default_bus_positions(Nx, Ny):
    """Pick sensible IN / OUT bus positions for any (Nx, Ny) >= (1, 1).

    Returns (bus_in, bus_drop) as 3-tuples (ix, iy, slot).

    Convention: the bus DC slot is chosen so the bus horseshoe sits OUTSIDE
    the lattice — never sandwiched between the bus site and its neighbors.

    - 2D lattice (Ny ≥ 2):   IN at (0, 0) bottom slot,
                             OUT at (0, Ny-1) top slot.
                             Both buses on the left edge column.
    - Horizontal chain (Ny = 1, Nx ≥ 2): IN at (0, 0) bottom,
                             OUT at (Nx-1, 0) bottom.
                             Both below the chain.
    - Single ring (1×1):     IN at slot 0 (bottom), OUT at slot 8 (top).
                             Standard add-drop filter geometry.
    """
    if Ny == 1:
        # Horizontal chain (or single ring on 1x1)
        if Nx == 1:
            return (0, 0, SLOT_BUS), (0, 0, SLOT_TOP)
        return (0, 0, SLOT_BUS), (Nx - 1, 0, SLOT_BUS)
    # 2D lattice, Ny ≥ 2
    return (0, 0, SLOT_BUS), (0, Ny - 1, SLOT_TOP)


# ═════════════════════════════════════════════════════════════════════════════
#  Core TMM (pure z-discretized; no Hamiltonian, no J in code)
# ═════════════════════════════════════════════════════════════════════════════

def make_lattice_indices(Nx, Ny, Nz_site=NZ_SITE, Nz_link=NZ_LINK):
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

    return site_idx, link_idx, state_size, h_link_names, v_link_names


def build_template(Nx, Ny, Phi0, kappa_ex, kappa_J,
                    bus_in, bus_drop, Nz_site=NZ_SITE, Nz_link=NZ_LINK,
                    removed_sites=None, removed_links=None):
    """
    bus_in / bus_drop: each a tuple (ix, iy, slot) OR (ix, iy).
    If a 2-tuple is given, slot defaults to SLOT_BUS = 0 (bottom).

    For single-ring (1×1) configurations the in and drop bus must share a
    site but use different slots — typically (0, 0, 0) for IN at the bottom
    and (0, 0, 8) for OUT at the top, giving a standard add-drop filter.

    removed_sites: set/iterable of (ix, iy) tuples — sites whose DC couplings
        should be zeroed out, decoupling them from the rest of the lattice.
        IN/OUT sites cannot be removed (silently ignored).
    removed_links: set/iterable of link names like "H_0_1" or "V_2_3" —
        links to decouple from their adjacent sites.

    Removed rings still occupy their grid points in the state vector (so
    indexing is stable), but they have no DCs to anything else, so their
    field decouples and decays to zero under nonzero α.
    """
    removed_sites = set(removed_sites) if removed_sites else set()
    removed_links = set(removed_links) if removed_links else set()

    # Accept both (ix, iy) and (ix, iy, slot) tuples
    def _normalize_bus(b, default_slot):
        if len(b) == 2:
            return (b[0], b[1], default_slot)
        return b
    bus_in = _normalize_bus(bus_in, SLOT_BUS)        # default bottom slot
    bus_drop = _normalize_bus(bus_drop, SLOT_BUS)    # default bottom slot too

    # IN/OUT sites are never removed — they carry the bus DCs.
    bus_sites = {(bus_in[0], bus_in[1]), (bus_drop[0], bus_drop[1])}
    removed_sites = removed_sites - bus_sites

    site_idx, link_idx, state_size, h_link_names, v_link_names = \
        make_lattice_indices(Nx, Ny, Nz_site, Nz_link)
    half_link = Nz_link // 2

    def link_theta(name):
        kind, _, iy_str = name.split("_")
        iy = int(iy_str)
        return -2.0 * Phi0 * iy if kind == "H" else 0.0

    def link_extras(theta):
        extra = np.zeros(Nz_link)
        extra[1] = +theta / 2.0
        extra[half_link + 1] = -theta / 2.0
        return extra

    extras = {name: link_extras(link_theta(name))
              for name in h_link_names + v_link_names}

    site_neighbors = {}
    for iy in range(Ny):
        for ix in range(Nx):
            slots = {}
            # Bus DCs: IN and drop. Each occupies its own slot.
            # In single-ring or shared-site cases they can both land here
            # but at different slot indices.
            if (ix, iy) == (bus_in[0], bus_in[1]):
                slots[bus_in[2]] = ("bus_in", None)
            if (ix, iy) == (bus_drop[0], bus_drop[1]):
                slots[bus_drop[2]] = ("bus_drop", None)
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

    entries = []; src_rows = []; src_vals = []
    for (ix, iy), slots in site_neighbors.items():
        site_removed = (ix, iy) in removed_sites
        for k in range(Nz_site):
            k_prev = (k - 1) % Nz_site
            row = site_idx(ix, iy, k)
            if k in slots and not site_removed:
                action = slots[k]; kind = action[0]
                if kind == "bus_in":
                    entries.append((row, site_idx(ix, iy, k_prev), "p_site", -t_ex))
                    src_rows.append(row); src_vals.append(1j * kappa_ex)
                elif kind == "bus_drop":
                    entries.append((row, site_idx(ix, iy, k_prev), "p_site", -t_ex))
                elif kind == "link":
                    _, link_name, end = action
                    if link_name in removed_links:
                        # This site's DC to a removed link: just propagate
                        # with no coupling (κ = 0, t = 1).
                        entries.append((row, site_idx(ix, iy, k_prev),
                                         "p_site", -1.0))
                    else:
                        if end == "near":
                            link_k_dc = 0; link_k_prev = Nz_link - 1
                        else:
                            link_k_dc = half_link; link_k_prev = half_link - 1
                        extra_phase_dc = extras[link_name][link_k_dc]
                        entries.append((row, site_idx(ix, iy, k_prev),
                                         "p_site", -t_J))
                        entries.append((row, link_idx(link_name, link_k_prev),
                                         "p_link_extra", -1j * kappa_J,
                                         extra_phase_dc))
            else:
                # Free propagation step (no DC, OR site is removed → all DCs
                # become free propagation, decoupling this site).
                entries.append((row, site_idx(ix, iy, k_prev), "p_site", -1.0))

    for name, ends in link_sites.items():
        link_removed = name in removed_links
        # Also check if either adjacent site is removed
        near_site = ends["near"][:2]
        far_site = ends["far"][:2]
        for k in range(Nz_link):
            k_prev = (k - 1) % Nz_link
            row = link_idx(name, k)
            extra_phase_k = extras[name][k]
            if (k == 0 or k == half_link) and not link_removed:
                site_info = ends["near"] if k == 0 else ends["far"]
                site_ix, site_iy, site_slot = site_info
                # Decouple if the site on the other end is removed.
                if (site_ix, site_iy) in removed_sites:
                    entries.append((row, link_idx(name, k_prev),
                                     "p_link_extra", -1.0, extra_phase_k))
                else:
                    site_k_prev = (site_slot - 1) % Nz_site
                    entries.append((row, link_idx(name, k_prev),
                                     "p_link_extra", -t_J, extra_phase_k))
                    entries.append((row, site_idx(site_ix, site_iy, site_k_prev),
                                     "p_site", -1j * kappa_J))
            else:
                # Free propagation (no DC, OR link is removed).
                entries.append((row, link_idx(name, k_prev),
                                 "p_link_extra", -1.0, extra_phase_k))

    rows = np.array([e[0] for e in entries], dtype=np.int32)
    cols = np.array([e[1] for e in entries], dtype=np.int32)
    kinds = np.array([0 if e[2] == "p_site" else 1 for e in entries], dtype=np.int8)
    coeffs = np.array([e[3] for e in entries], dtype=complex)
    extras_arr = np.zeros(len(entries), dtype=float)
    for i, e in enumerate(entries):
        if e[2] == "p_link_extra":
            extras_arr[i] = e[4]

    return dict(
        rows=rows, cols=cols, kinds=kinds, coeffs=coeffs,
        extras_arr=extras_arr,
        diag_rows=np.arange(state_size, dtype=np.int32),
        src_rows_arr=np.array(src_rows, dtype=np.int32),
        src_vals_arr=np.array(src_vals, dtype=complex),
        state_size=state_size,
        site_idx=site_idx, link_idx=link_idx,
        h_link_names=h_link_names, v_link_names=v_link_names,
        bus_in=bus_in, bus_drop=bus_drop,
        Nz_site=Nz_site, Nz_link=Nz_link,
        removed_sites=removed_sites, removed_links=removed_links,
    )


def solve_one(omega, template, beta0_eta_over_pi, kappa_ex, alpha):
    """
    Per-step propagation factors.

    PHYSICS NOTE — DO NOT confuse the *physical* length of the link ring
    with the *detuning-dependent* phase. In a real IQH-lattice device, the link
    ring is longer than the site ring by η = λ₀/(2 n_eff), which is half a
    wavelength — TINY (η/L_site ~ 10⁻⁴). Yet β₀ η = π because β₀ is large.

    So the link's round-trip phase decomposes as
        β L_link = β₀ L_link + ω L_link
                 = (β₀ L_site + β₀ η) + ω(L_site + η)
                 = (carrier wraps to 2πN) + β₀η + ω L_site + ω η
    On FSR scales (|ω| ~ 2π) the ω η piece is ~10⁻⁴·2π — negligible. The
    *only* non-negligible link-vs-site difference is the static β₀ η.

    The single physically meaningful knob is β₀η, the static phase the link
    ring picks up beyond the carrier wrap. We expose it in units of π:
        beta0_eta_over_pi = 1.0  →  full anti-resonance (IQH)  ← default
        beta0_eta_over_pi = 0.0  →  link & site degenerate         (mess)
        beta0_eta_over_pi = 0.5  →  link tuned a quarter-FSR off
    Periodic mod 2 (any integer multiple of 2π is the identity).
    """
    L_site = 1.0
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]
    dz_site = L_site / Nz_site
    # Link uses L_site for the propagation length too — see docstring.
    # The η-induced length difference is irrelevant on FSR scales; only
    # the static β₀ η matters and that's a separate constant phase.
    dz_link = L_site / Nz_link

    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)
    beta0_eta = beta0_eta_over_pi * np.pi
    p_link_base = np.exp(1j * omega * dz_link
                          + 1j * beta0_eta / Nz_link
                          - alpha * dz_link / 2.0)

    rows = template["rows"]; cols = template["cols"]
    kinds = template["kinds"]; coeffs = template["coeffs"]
    extras_arr = template["extras_arr"]
    diag_rows = template["diag_rows"]
    src_rows_arr = template["src_rows_arr"]
    src_vals_arr = template["src_vals_arr"]
    state_size = template["state_size"]
    site_idx = template["site_idx"]

    vals = np.where(kinds == 0,
                     coeffs * p_site,
                     coeffs * p_link_base * np.exp(1j * extras_arr))
    rows_full = np.concatenate([rows, diag_rows])
    cols_full = np.concatenate([cols, diag_rows])
    vals_full = np.concatenate([vals, np.ones(state_size, dtype=complex)])

    M = csc_matrix((vals_full, (rows_full, cols_full)),
                    shape=(state_size, state_size))
    s = np.zeros(state_size, dtype=complex)
    s[src_rows_arr] = src_vals_arr

    lu = splu(M)
    E = lu.solve(s)

    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    # bus_in / bus_drop are 3-tuples (ix, iy, slot); the field arriving at
    # the bus DC is at the predecessor grid index (slot - 1) mod Nz_site.
    i_in, j_in, s_in_slot = template["bus_in"]
    pred_in = (s_in_slot - 1) % Nz_site
    e_in_at_bus_in = p_site * E[site_idx(i_in, j_in, pred_in)]
    s_thru = t_ex * 1.0 + 1j * kappa_ex * e_in_at_bus_in
    i_d, j_d, s_d_slot = template["bus_drop"]
    pred_d = (s_d_slot - 1) % Nz_site
    e_in_at_bus_drop = p_site * E[site_idx(i_d, j_d, pred_d)]
    s_drop = 1j * kappa_ex * e_in_at_bus_drop

    return E, s_drop, s_thru


def build_propagator(omega, template, beta0_eta_over_pi, alpha):
    """Return (R, s) where the discrete-time map is E^(n+1) = R @ E^(n) + s.

    R is the off-diagonal propagation+coupling operator built from the same
    template entries as solve_one, but with signs flipped (the steady-state
    matrix is M = I - R).

    TIME STEP — important: one application of R advances every grid point
    by ONE Δz around its ring perimeter, NOT one full round trip. With
    Nz_site = 16, one site round-trip corresponds to Nz_site = 16 steps.
    Each step is therefore Δt = T_R / Nz_site = L_site/(v_g * Nz_site)
    in real time (with v_g = L_site = 1 in dimensionless units, Δt = 1/16).

    The kappa_ex enters through src_vals_arr (already in template). Note
    the source vector here is the *per-step* injection — same as
    solve_one's source.
    """
    L_site = 1.0
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]
    dz_site = L_site / Nz_site
    dz_link = L_site / Nz_link

    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)
    beta0_eta = beta0_eta_over_pi * np.pi
    p_link_base = np.exp(1j * omega * dz_link
                          + 1j * beta0_eta / Nz_link
                          - alpha * dz_link / 2.0)

    rows = template["rows"]; cols = template["cols"]
    kinds = template["kinds"]; coeffs = template["coeffs"]
    extras_arr = template["extras_arr"]
    src_rows_arr = template["src_rows_arr"]
    src_vals_arr = template["src_vals_arr"]
    state_size = template["state_size"]

    # In solve_one the entries are stored as -coeff·p (so the M=I-R matrix
    # has -R off-diagonal). For the propagator R itself we want +coeff·p,
    # i.e. flip sign on every off-diagonal entry.
    vals = np.where(kinds == 0,
                     -coeffs * p_site,
                     -coeffs * p_link_base * np.exp(1j * extras_arr))
    R = csc_matrix((vals, (rows, cols)),
                    shape=(state_size, state_size))
    s = np.zeros(state_size, dtype=complex)
    s[src_rows_arr] = src_vals_arr
    return R, s


def time_evolve(omega, template, beta0_eta_over_pi, kappa_ex, alpha,
                  N_steps=1000, record_stride=1, progress_cb=None):
    """Iterate E^(n+1) = R E^(n) + s starting from E^(0) = 0 for N_steps.

    TIME STEP — important: each iteration is ONE Δz advance, not one full
    site round trip. To get N round trips of physical time you need
    N * Nz_site iterations (16 by default).

    Returns:
        E_history   : (N_recorded, state_size) complex array — fields at
                      each recorded step
        drop_history: (N_recorded,) complex drop-port amplitude
        thru_history: (N_recorded,) complex thru-port amplitude
        steps       : (N_recorded,) int array of step numbers (in Δz units)
        E_inf       : (state_size,) — the true steady state from solve_one,
                      for reference (the iterated field may not have
                      reached it within N_steps if Q is high).

    record_stride: keep every Nth step in history (1 = keep all).
    progress_cb: optional callback(step, N_max) for progress reporting.

    Note on physics: with α ~ 1e-4 the spectral radius of R is ~0.99999
    per round-trip, so converging to the true steady state can take 10⁵+
    Δz steps on high-Q modes. The user is expected to pick N_steps to
    show the interesting transient.
    """
    R, s = build_propagator(omega, template, beta0_eta_over_pi, alpha)

    Nz_site = template["Nz_site"]
    site_idx = template["site_idx"]
    p_site_factor = np.exp(1j * omega * (1.0 / Nz_site)
                              - alpha * (1.0 / Nz_site) / 2.0)
    i_in, j_in, s_in_slot = template["bus_in"]
    pred_in = (s_in_slot - 1) % Nz_site
    i_d, j_d, s_d_slot = template["bus_drop"]
    pred_d = (s_d_slot - 1) % Nz_site
    bus_in_grid = site_idx(i_in, j_in, pred_in)
    bus_drop_grid = site_idx(i_d, j_d, pred_d)
    t_ex = np.sqrt(1.0 - kappa_ex ** 2)

    # Reference steady state from direct solver
    E_inf, _, _ = solve_one(omega, template, beta0_eta_over_pi,
                              kappa_ex, alpha)

    E = np.zeros(R.shape[0], dtype=complex)
    E_hist = []; d_hist = []; t_hist = []; steps = []
    last_emit = -1
    for n in range(N_steps):
        E = R @ E + s
        if n % record_stride == 0 or n == N_steps - 1:
            e_in_at_bus_in = p_site_factor * E[bus_in_grid]
            s_thru = t_ex + 1j * kappa_ex * e_in_at_bus_in
            e_in_at_bus_drop = p_site_factor * E[bus_drop_grid]
            s_drop = 1j * kappa_ex * e_in_at_bus_drop
            E_hist.append(E.copy())
            d_hist.append(s_drop)
            t_hist.append(s_thru)
            steps.append(n + 1)
        if progress_cb is not None and (n - last_emit) > 50:
            progress_cb(n + 1, N_steps)
            last_emit = n
    if progress_cb is not None:
        progress_cb(N_steps, N_steps)
    return (np.array(E_hist), np.array(d_hist),
            np.array(t_hist), np.array(steps), E_inf)


# ═════════════════════════════════════════════════════════════════════════════
#  Visualization helpers
# ═════════════════════════════════════════════════════════════════════════════

def _rounded_square_perimeter(cx, cy, half_side, corner_radius, n_pts=200):
    L_straight = 2 * (half_side - corner_radius)
    L_corner = (np.pi / 2) * corner_radius
    P = 4 * L_straight + 4 * L_corner
    s_arr = np.linspace(0, P, n_pts, endpoint=False)
    xs = np.zeros(n_pts); ys = np.zeros(n_pts)
    h = half_side; rc = corner_radius
    for i, s in enumerate(s_arr):
        s_local = s
        if s_local < L_straight:
            xs[i] = cx + h; ys[i] = cy - (h - rc) + s_local; continue
        s_local -= L_straight
        if s_local < L_corner:
            theta = s_local / rc
            xs[i] = (cx + h - rc) + rc * np.cos(theta)
            ys[i] = (cy + h - rc) + rc * np.sin(theta); continue
        s_local -= L_corner
        if s_local < L_straight:
            xs[i] = (cx + h - rc) - s_local; ys[i] = cy + h; continue
        s_local -= L_straight
        if s_local < L_corner:
            theta = np.pi / 2 + s_local / rc
            xs[i] = (cx - h + rc) + rc * np.cos(theta)
            ys[i] = (cy + h - rc) + rc * np.sin(theta); continue
        s_local -= L_corner
        if s_local < L_straight:
            xs[i] = cx - h; ys[i] = (cy + h - rc) - s_local; continue
        s_local -= L_straight
        if s_local < L_corner:
            theta = np.pi + s_local / rc
            xs[i] = (cx - h + rc) + rc * np.cos(theta)
            ys[i] = (cy - h + rc) + rc * np.sin(theta); continue
        s_local -= L_corner
        if s_local < L_straight:
            xs[i] = (cx - h + rc) + s_local; ys[i] = cy - h; continue
        s_local -= L_straight
        theta = 3 * np.pi / 2 + s_local / rc
        xs[i] = (cx + h - rc) + rc * np.cos(theta)
        ys[i] = (cy - h + rc) + rc * np.sin(theta)
    return xs, ys


def _draw_ring(ax, center, half_side, intensities, I_max, lw=2.0,
                cmap=None, zorder=3, phase_offset=0.0, chirality=+1,
                corner_radius=None):
    if cmap is None:
        cmap = plt.cm.inferno
    if corner_radius is None:
        corner_radius = 0.30 * half_side
    Nz = len(intensities)
    n_fine = max(160, 6 * Nz)
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
                      np.column_stack([np.roll(xs, -1), np.roll(ys, -1)])], axis=1)
    colors = cmap(I_interp / I_max if I_max > 0 else np.zeros_like(I_interp))
    lc = LineCollection(segs, colors=colors, linewidths=lw,
                         capstyle="butt", joinstyle="miter", zorder=zorder)
    ax.add_collection(lc)


def _draw_horseshoe_bus(ax, ring_center, side, half_side,
                         left_label, right_label,
                         left_arrow_in=False, right_arrow_out=False,
                         left_arrow_out=False, right_arrow_in=False,
                         color=ACCENT, lw=1.2):
    cx, cy = ring_center
    sign = -1 if side == "lower" else +1
    bus_gap = 0.03
    coupling_half_len = 0.22
    bus_bend_r = 0.05
    tail_len = 0.18
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
             color=color, lw=lw, zorder=2)
    if side == "lower":
        a = np.linspace(np.pi / 2, np.pi, 32)
    else:
        a = np.linspace(np.pi, 3 * np.pi / 2, 32)
    ax.plot(x_L_bend_c + bus_bend_r * np.cos(a),
             y_L_bend_c + bus_bend_r * np.sin(a),
             color=color, lw=lw, zorder=2)
    if side == "lower":
        a = np.linspace(0, np.pi / 2, 32)
    else:
        a = np.linspace(3 * np.pi / 2, 2 * np.pi, 32)
    ax.plot(x_R_bend_c + bus_bend_r * np.cos(a),
             y_R_bend_c + bus_bend_r * np.sin(a),
             color=color, lw=lw, zorder=2)
    ax.plot([x_L_tail, x_L_tail], [y_L_bend_c, y_tail_end],
             color=color, lw=lw, zorder=2)
    ax.plot([x_R_tail, x_R_tail], [y_R_bend_c, y_tail_end],
             color=color, lw=lw, zorder=2)
    if left_arrow_in:
        ax.annotate("", xy=(x_L_tail, y_L_bend_c - sign * 0.02),
                     xytext=(x_L_tail, y_tail_end),
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if left_arrow_out:
        ax.annotate("", xy=(x_L_tail, y_tail_end),
                     xytext=(x_L_tail, y_L_bend_c - sign * 0.02),
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if right_arrow_out:
        ax.annotate("", xy=(x_R_tail, y_tail_end),
                     xytext=(x_R_tail, y_R_bend_c - sign * 0.02),
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if right_arrow_in:
        ax.annotate("", xy=(x_R_tail, y_R_bend_c - sign * 0.02),
                     xytext=(x_R_tail, y_tail_end),
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    label_va = "top" if side == "lower" else "bottom"
    ax.text(x_L_tail, y_tail_end + sign * 0.04, left_label,
             color=color, fontsize=6, ha="center", va=label_va)
    ax.text(x_R_tail, y_tail_end + sign * 0.04, right_label,
             color=color, fontsize=6, ha="center", va=label_va)


def plot_field_distribution(ax, E, Nx, Ny, template, title="", I_max=None):
    site_idx = template["site_idx"]
    link_idx = template["link_idx"]
    h_link_names = template["h_link_names"]
    v_link_names = template["v_link_names"]
    bus_in = template["bus_in"]
    bus_drop = template["bus_drop"]
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]

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

    site_pos = {(ix, iy): (float(ix), float(iy))
                for iy in range(Ny) for ix in range(Nx)}
    link_pos = {}
    for name in h_link_names:
        _, ix_s, iy_s = name.split("_")
        link_pos[name] = (int(ix_s) + 0.5, float(int(iy_s)))
    for name in v_link_names:
        _, ix_s, iy_s = name.split("_")
        link_pos[name] = (float(int(ix_s)), int(iy_s) + 0.5)

    half_side = 0.24
    corner_r = 0.07
    lw_ring = 2.0

    ax.set_facecolor(PANEL_BG)
    bond_color = '#1e2a40'
    for name in h_link_names:
        _, ix_s, iy_s = name.split("_")
        ix, iy = int(ix_s), int(iy_s)
        ax.plot([ix, ix + 1], [iy, iy], color=bond_color, lw=0.5, zorder=1)
    for name in v_link_names:
        _, ix_s, iy_s = name.split("_")
        ix, iy = int(ix_s), int(iy_s)
        ax.plot([ix, ix], [iy, iy + 1], color=bond_color, lw=0.5, zorder=1)

    removed_sites = template.get("removed_sites", set())
    removed_links = template.get("removed_links", set())

    # bus_in / bus_drop are 3-tuples (ix, iy, slot). Slot 0 = bottom, slot 8 = top.
    bus_in_xy = (bus_in[0], bus_in[1])
    bus_drop_xy = (bus_drop[0], bus_drop[1])
    bus_in_side = "upper" if bus_in[2] == SLOT_TOP else "lower"
    bus_drop_side = "upper" if bus_drop[2] == SLOT_TOP else "lower"

    def _ghost_ring(center, label_xy=None):
        """Draw a thin dashed grey rounded square for a removed ring."""
        xs, ys = _rounded_square_perimeter(center[0], center[1],
                                             half_side, corner_r, n_pts=120)
        # Close it
        xs = np.append(xs, xs[0]); ys = np.append(ys, ys[0])
        ax.plot(xs, ys, color='#3a4560', lw=0.8, ls='--',
                 zorder=2, alpha=0.7)
        if label_xy is not None:
            ax.plot([label_xy[0]-0.04, label_xy[0]+0.04],
                     [label_xy[1]-0.04, label_xy[1]+0.04],
                     color='#5a6280', lw=0.8, zorder=2, alpha=0.6)
            ax.plot([label_xy[0]-0.04, label_xy[0]+0.04],
                     [label_xy[1]+0.04, label_xy[1]-0.04],
                     color='#5a6280', lw=0.8, zorder=2, alpha=0.6)

    for (ix, iy), grid_I in site_grids.items():
        if (ix, iy) in removed_sites:
            _ghost_ring(site_pos[(ix, iy)], label_xy=site_pos[(ix, iy)])
            continue
        ph = 0.75 if (ix, iy) == bus_in_xy else 0.25
        _draw_ring(ax, site_pos[(ix, iy)], half_side, grid_I,
                    I_max, lw=lw_ring, zorder=3,
                    phase_offset=ph, chirality=+1, corner_radius=corner_r)

    for name, grid_I in link_grids.items():
        if name in removed_links:
            _ghost_ring(link_pos[name], label_xy=link_pos[name])
            continue
        kind = name.split("_")[0]
        ph = 0.75 if kind == "V" else 0.5
        _draw_ring(ax, link_pos[name], half_side, grid_I,
                    I_max, lw=lw_ring, zorder=2,
                    phase_offset=ph, chirality=-1, corner_radius=corner_r)

    in_pos = site_pos[bus_in_xy]
    out_pos = site_pos[bus_drop_xy]
    _draw_horseshoe_bus(ax, in_pos, bus_in_side, half_side,
                         "input", "through",
                         left_arrow_in=True, right_arrow_out=True)
    _draw_horseshoe_bus(ax, out_pos, bus_drop_side, half_side,
                         "drop", "add",
                         left_arrow_out=True)

    ax.text(in_pos[0], in_pos[1], "IN", ha="center", va="center",
            color="white", fontsize=6, fontweight="bold", zorder=4)
    if bus_in_xy != bus_drop_xy:
        ax.text(out_pos[0], out_pos[1], "OUT", ha="center", va="center",
                color="white", fontsize=6, fontweight="bold", zorder=4)
    # If single-ring add-drop, "IN" label is enough — no separate OUT site.

    pad_x = 0.4
    pad_y = 0.45
    ax.set_xlim(-pad_x, Nx - 1 + pad_x)
    ax.set_ylim(-pad_y, Ny - 1 + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a4560')
    ax.set_title(title, fontsize=9, color=ACCENT, pad=4)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,
                                norm=plt.Normalize(vmin=0, vmax=I_max))
    return sm


# ═════════════════════════════════════════════════════════════════════════════
#  Background scan worker
# ═════════════════════════════════════════════════════════════════════════════

class ScanWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._abort = False

    def request_abort(self):
        self._abort = True

    def run(self):
        try:
            p = self.p
            bus_in, bus_drop = default_bus_positions(p["Nx"], p["Ny"])
            template = build_template(
                p["Nx"], p["Ny"], p["Phi0"], p["kappa_ex"], p["kappa_J"],
                bus_in=bus_in, bus_drop=bus_drop,
                removed_sites=p.get("removed_sites"),
                removed_links=p.get("removed_links"),
            )
            omegas = p["omegas"]
            N = len(omegas)
            Td = np.zeros(N); Tt = np.zeros(N)
            update_every = max(1, N // 100)
            for i, w in enumerate(omegas):
                if self._abort:
                    return
                _, sd, st = solve_one(w, template, p["beta0_eta_over_pi"],
                                       p["kappa_ex"], p["alpha"])
                Td[i] = abs(sd) ** 2
                Tt[i] = abs(st) ** 2
                if (i + 1) % update_every == 0:
                    self.progress.emit(i + 1, N)
            self.progress.emit(N, N)
            self.finished_ok.emit(omegas, Td, Tt, template)
        except Exception as e:
            self.failed.emit(repr(e))


# ═════════════════════════════════════════════════════════════════════════════
#  Time-evolution dialog
# ═════════════════════════════════════════════════════════════════════════════

class TimeEvolveWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object, object, object, object, object)  # E_hist, d_hist, t_hist, steps, E_inf
    failed = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params

    def run(self):
        try:
            p = self.p
            cb = lambda n, N: self.progress.emit(n, N)
            E_hist, d_hist, t_hist, steps, E_inf = time_evolve(
                p["omega"], p["template"], p["beta0_eta_over_pi"],
                p["kappa_ex"], p["alpha"],
                N_steps=p["N_steps"], record_stride=p["record_stride"],
                progress_cb=cb)
            self.finished_ok.emit(E_hist, d_hist, t_hist, steps, E_inf)
        except Exception as e:
            self.failed.emit(repr(e))


class TimeEvolutionDialog(QDialog):
    """Standalone window: iterate the cavity buildup and show drop(t) +
    field intensity vs time, with a slider to scrub through round trips.
    """
    def __init__(self, parent, omega, template, beta0_eta_over_pi,
                  kappa_ex, alpha, Nx, Ny, save_dir):
        super().__init__(parent)
        self.setWindowTitle(f"Time evolution at ω/(2π) = {omega/(2*np.pi):+.5f}")
        self.resize(1100, 720)
        self.setStyleSheet(parent.styleSheet())  # inherit dark theme

        self.omega = omega
        self.template = template
        self.beta0_eta_over_pi = beta0_eta_over_pi
        self.kappa_ex = kappa_ex
        self.alpha = alpha
        self.Nx = Nx
        self.Ny = Ny
        self.save_dir = save_dir

        # Results storage
        self.E_hist = None
        self.d_hist = None
        self.t_hist = None
        self.steps = None
        self.E_inf = None
        self.I_max_ss = 1.0   # for normalization
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6); layout.setContentsMargins(8, 8, 8, 8)

        # Header
        hdr = QLabel(
            f'<b>ω/(2π) = {self.omega/(2*np.pi):+.5f}</b>  ·  '
            f'iterating  E^(n+1) = R·E^(n) + s   from E^(0) = 0')
        hdr.setStyleSheet(f'color:{ACCENT};font-size:11px;')
        layout.addWidget(hdr)

        # Run controls. The unit is "round trips" but we allow fractional
        # values (0.1, 0.5, 1.0, ...) so single-ring users can watch a
        # pulse traverse the ring without being forced to skip whole
        # round trips. "frames per round trip" picks how often to record.
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel('# round trips:'))
        self.spn_n = QDoubleSpinBox()
        self.spn_n.setRange(0.1, 100000.0)
        self.spn_n.setDecimals(2)
        self.spn_n.setSingleStep(1.0)
        self.spn_n.setFixedWidth(100)
        ctl.addWidget(self.spn_n)

        ctl.addWidget(QLabel('frames per RT:'))
        self.spn_fpr = QSpinBox()
        # max = Nz_site (one frame per Δz step). Default depends on lattice
        # size: tiny lattices (1x1, 1xN) want sub-RT resolution to see the
        # pulse traverse the ring; bigger lattices want longer time so we
        # subsample more aggressively.
        Nz_site_for_ui = self.template["Nz_site"]
        self.spn_fpr.setRange(1, Nz_site_for_ui)
        self.spn_fpr.setFixedWidth(60)
        self.spn_fpr.setToolTip(
            f'How many frames to record per round trip.\n'
            f'1 = once per round-trip (fast, good for long runs).\n'
            f'{Nz_site_for_ui} = every Δz grid step (smooth, lets you watch\n'
            f'the pulse traverse the ring at sub-round-trip resolution).')
        ctl.addWidget(self.spn_fpr)

        # Pick smart defaults based on lattice size. Always record at
        # the finest resolution (frames per RT = Nz_site = 16) so the user
        # can see sub-round-trip dynamics regardless of lattice size.
        n_sites = self.Nx * self.Ny
        if n_sites <= 2:
            # Single ring or tiny chain: short total time
            self.spn_n.setValue(8.0)
        elif n_sites <= 8:
            self.spn_n.setValue(2000.0)
        else:
            self.spn_n.setValue(2000.0)
        self.spn_fpr.setValue(Nz_site_for_ui)

        self.btn_run = QPushButton('▶ Run')
        self.btn_run.setFixedHeight(28)
        self.btn_run.clicked.connect(self._run)
        ctl.addWidget(self.btn_run)

        self.btn_mp4 = QPushButton('💾 Save MP4')
        self.btn_mp4.setFixedHeight(28)
        self.btn_mp4.setEnabled(False)
        self.btn_mp4.clicked.connect(self._save_mp4)
        ctl.addWidget(self.btn_mp4)

        self.btn_gif = QPushButton('💾 Save GIF')
        self.btn_gif.setFixedHeight(28)
        self.btn_gif.setEnabled(False)
        self.btn_gif.clicked.connect(self._save_gif)
        self.btn_gif.setToolTip(
            'Pillow-based GIF export — no ffmpeg needed.\n'
            'Larger files than MP4 but works out-of-the-box.')
        ctl.addWidget(self.btn_gif)

        ctl.addWidget(QLabel('fps:'))
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 120)
        self.spn_fps.setValue(15)
        self.spn_fps.setFixedWidth(55)
        self.spn_fps.setToolTip(
            'Playback frame rate of the exported MP4 / GIF.\n'
            'Higher = faster playback, shorter video.\n'
            'Typical: 15-30 fps.')
        ctl.addWidget(self.spn_fps)

        ctl.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setFixedWidth(220); self.progress.setFixedHeight(18)
        ctl.addWidget(self.progress)

        layout.addLayout(ctl)

        # Plotting area: drop(t) on the left, lattice on the right
        self.fig = Figure(facecolor=DARK_BG, figsize=(11, 5.5))
        self.canvas = FigureCanvas(self.fig)
        self.ax_drop = self.fig.add_axes([0.06, 0.14, 0.40, 0.78])
        self.ax_lat = self.fig.add_axes([0.50, 0.04, 0.42, 0.92])
        self.cax_lat = self.fig.add_axes([0.94, 0.10, 0.014, 0.78])

        for ax in (self.ax_drop,):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor('#3a4560'); sp.set_linewidth(1.2)
            ax.grid(True, color=GRID_COL, linewidth=0.4)
        self.ax_drop.set_xlabel('round-trip n', color=TEXT_COL, fontsize=9)
        self.ax_drop.set_ylabel(r'$|s_{drop}(n)|^2$', color=TEXT_COL, fontsize=9)
        self.ax_drop.set_title('Drop port vs time', color='#ff4a6e',
                                  fontsize=10, pad=3)

        self.ax_lat.set_facecolor(PANEL_BG)
        self.ax_lat.set_xticks([]); self.ax_lat.set_yticks([])
        for sp in self.ax_lat.spines.values():
            sp.set_edgecolor('#3a4560')
        layout.addWidget(self.canvas, stretch=1)

        # Slider
        sl_row = QHBoxLayout()
        self.lbl_step = QLabel('step: —')
        self.lbl_step.setFixedWidth(120)
        self.lbl_step.setStyleSheet('color:#7a8aaa;font-size:11px;')
        sl_row.addWidget(self.lbl_step)
        self.sld_t = QSlider(Qt.Horizontal)
        self.sld_t.setRange(0, 0); self.sld_t.setEnabled(False)
        self.sld_t.valueChanged.connect(self._on_slider)
        sl_row.addWidget(self.sld_t, stretch=1)
        layout.addLayout(sl_row)

        self._cbar = None

    # ── Run ─────────────────────────────────────────────────────────────────
    def _run(self):
        if self.worker is not None and self.worker.isRunning():
            return
        # User input is in round-trips. One round-trip = Nz_site Δz steps.
        # The "frames per RT" spinbox is the recording rate per round-trip:
        #   1   → one frame every Nz_site Δz steps (one per RT, default)
        #   16  → one frame every 1 Δz step (every grid hop, max resolution)
        Nz_site = self.template["Nz_site"]
        N_rt = self.spn_n.value()                       # may be fractional
        N_steps_total = max(1, int(round(N_rt * Nz_site)))
        frames_per_rt = self.spn_fpr.value()
        stride_dz = max(1, Nz_site // frames_per_rt)
        params = dict(
            omega=self.omega, template=self.template,
            beta0_eta_over_pi=self.beta0_eta_over_pi,
            kappa_ex=self.kappa_ex, alpha=self.alpha,
            N_steps=N_steps_total, record_stride=stride_dz,
        )
        self.btn_run.setEnabled(False); self.btn_run.setText('Running…')
        self.btn_mp4.setEnabled(False)
        self.btn_gif.setEnabled(False)
        self.progress.setValue(0)
        self.worker = TimeEvolveWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, done, total):
        self.progress.setValue(int(done * 100 / max(1, total)))

    def _on_failed(self, msg):
        self.btn_run.setEnabled(True); self.btn_run.setText('▶ Run')
        QMessageBox.critical(self, 'Time evolution failed', msg)

    def _on_done(self, E_hist, d_hist, t_hist, steps, E_inf):
        self.E_hist = E_hist; self.d_hist = d_hist
        self.t_hist = t_hist; self.steps = steps; self.E_inf = E_inf

        # Normalization. We use the max |E|^2 across the entire trajectory,
        # NOT the steady-state max. Reason: on a resonant high-Q ring the
        # steady state has huge intracavity buildup (factor 1/(αL+κ_ex²)),
        # so during early transients the field looks dim by comparison and
        # nothing is visible. Trajectory-max keeps colors well-distributed
        # across frames while still being a single consistent scale.
        traj_max = float(np.max(np.abs(E_hist)**2))
        ss_max = float(np.max(np.abs(E_inf)**2))
        # If the trajectory has reached steady state (or close), use the
        # steady-state value so the user sees relative-to-steady-state
        # brightness. Otherwise use trajectory max so dynamics are visible.
        self.I_max_ss = max(traj_max, 1e-30)
        # Whether we're using "true" steady state — affects colorbar label
        self._norm_is_steady = traj_max >= 0.5 * ss_max

        # Plot drop|^2 vs time. Convert Δz step counts to round-trips.
        Nz_site = self.template["Nz_site"]
        steps_rt = steps / Nz_site

        self.ax_drop.clear()
        self.ax_drop.set_facecolor(PANEL_BG)
        self.ax_drop.tick_params(colors=TEXT_COL, labelsize=9)
        for sp in self.ax_drop.spines.values():
            sp.set_edgecolor('#3a4560'); sp.set_linewidth(1.2)
        self.ax_drop.grid(True, color=GRID_COL, linewidth=0.4)
        Td_t = np.abs(d_hist)**2
        self.ax_drop.plot(steps_rt, Td_t, lw=1.0, color='#ff4a6e')
        # Reference line: steady-state drop power
        site_idx_fn = self.template["site_idx"]
        i_d, j_d, s_d_slot = self.template["bus_drop"]
        pred_d = (s_d_slot - 1) % Nz_site
        p_factor = np.exp(1j*self.omega*(1.0/Nz_site)
                            - self.alpha*(1.0/Nz_site)/2)
        s_drop_ss = 1j * self.kappa_ex * p_factor * E_inf[site_idx_fn(i_d, j_d, pred_d)]
        Td_ss = abs(s_drop_ss)**2
        self.ax_drop.axhline(Td_ss, color='#00e5ff', ls='--', lw=0.8,
                                alpha=0.7, label=f'steady = {Td_ss:.4f}')
        self.ax_drop.set_xlabel('round-trip n', color=TEXT_COL, fontsize=9)
        self.ax_drop.set_ylabel(r'$|s_{drop}(n)|^2$', color=TEXT_COL, fontsize=9)
        self.ax_drop.set_title('Drop port vs time', color='#ff4a6e',
                                  fontsize=10, pad=3)
        self.ax_drop.legend(loc='best', fontsize=8, facecolor=PANEL_BG,
                              edgecolor='#3a4560', labelcolor=TEXT_COL)

        # Slider setup
        self.sld_t.blockSignals(True)
        self.sld_t.setRange(0, len(steps) - 1)
        self.sld_t.setValue(len(steps) - 1)  # show last frame
        self.sld_t.setEnabled(True)
        self.sld_t.blockSignals(False)
        self._render_frame(len(steps) - 1)

        self.btn_run.setEnabled(True); self.btn_run.setText('▶ Run')
        self.btn_mp4.setEnabled(True)
        self.btn_gif.setEnabled(True)
        self.progress.setValue(100)

    def _on_slider(self, idx):
        if self.E_hist is None:
            return
        self._render_frame(idx)

    def _render_frame(self, idx):
        Nz_site = self.template["Nz_site"]
        n_dz = int(self.steps[idx])
        n_rt = n_dz / Nz_site  # round-trip count (may be fractional)
        E = self.E_hist[idx]
        self.lbl_step.setText(
            f'round-trip: {n_rt:.1f} / {self.steps[-1]/Nz_site:.0f}')

        self.ax_lat.clear()
        sm = plot_field_distribution(
            self.ax_lat, E, self.Nx, self.Ny, self.template,
            title=f'round-trip n = {n_rt:.1f}',
            I_max=self.I_max_ss,   # normalize to steady-state max
        )
        # Mark current time on the drop trace with a vertical line
        for line in list(self.ax_drop.lines):
            if getattr(line, '_is_time_marker', False):
                line.remove()
        vline = self.ax_drop.axvline(n_rt, color=ACCENT, ls=':', lw=1, alpha=0.7)
        vline._is_time_marker = True

        # Colorbar refresh
        self.cax_lat.clear()
        cbar_label = (r'$|E|^2 / \max|E_\infty|^2$' if self._norm_is_steady
                       else r'$|E|^2 / \max_{n,k} |E^{(n)}_k|^2$')
        self._cbar = self.fig.colorbar(sm, cax=self.cax_lat, label=cbar_label)
        self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self._cbar.ax.yaxis.label.set_color(TEXT_COL)
        self._cbar.outline.set_edgecolor('#3a4560')
        self.canvas.draw_idle()

    # ── MP4 ─────────────────────────────────────────────────────────────────
    def _save_mp4(self):
        if self.E_hist is None:
            return
        ts = time.strftime('%Y%m%d_%H%M%S')
        save_dir = self.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'TMM_evolution_{ts}.mp4')

        # Build animation. Use a fresh figure to avoid touching the live UI.
        fig = Figure(facecolor=DARK_BG, figsize=(7.5, 7))
        ax_lat = fig.add_axes([0.04, 0.04, 0.84, 0.92])
        cax = fig.add_axes([0.90, 0.10, 0.025, 0.78])

        n_frames = len(self.steps)
        Nz_site = self.template["Nz_site"]
        sm0 = plot_field_distribution(
            ax_lat, self.E_hist[0], self.Nx, self.Ny, self.template,
            title=f'round-trip n = {self.steps[0]/Nz_site:.1f}',
            I_max=self.I_max_ss)
        cbar = fig.colorbar(sm0, cax=cax,
                              label=r'$|E|^2 / \max|E_\infty|^2$')
        cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
        cbar.ax.yaxis.label.set_color(TEXT_COL)
        cbar.outline.set_edgecolor('#3a4560')

        def update(i):
            ax_lat.clear()
            plot_field_distribution(
                ax_lat, self.E_hist[i], self.Nx, self.Ny, self.template,
                title=f'round-trip n = {self.steps[i]/Nz_site:.1f}',
                I_max=self.I_max_ss)
            return []

        # User-set FPS (defaults to 15 in the spinbox).
        fps = self.spn_fps.value()

        try:
            writer = FFMpegWriter(fps=fps, bitrate=2400,
                                    metadata={'artist': 'TMM Explorer'})
            anim = FuncAnimation(fig, update, frames=n_frames,
                                    blit=False, repeat=False)
            self.btn_mp4.setEnabled(False); self.btn_mp4.setText('Encoding…')
            QApplication.processEvents()
            anim.save(path, writer=writer, dpi=130,
                       savefig_kwargs={'facecolor': DARK_BG})
            self.btn_mp4.setText('💾 Save MP4')
            self.btn_mp4.setEnabled(True)
            QMessageBox.information(self, 'Saved', f'MP4 written:\n{path}')
        except Exception as e:
            self.btn_mp4.setText('💾 Save MP4')
            self.btn_mp4.setEnabled(True)
            QMessageBox.critical(
                self, 'MP4 export failed',
                f'Could not write MP4 (do you have ffmpeg installed?):\n{e}\n\n'
                f'Try the GIF export instead — it works without ffmpeg.')

    # ── GIF (Pillow, no ffmpeg required) ───────────────────────────────────
    def _save_gif(self):
        if self.E_hist is None:
            return
        ts = time.strftime('%Y%m%d_%H%M%S')
        save_dir = self.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'TMM_evolution_{ts}.gif')

        # Use the Agg backend for offscreen rendering. matplotlib already
        # imported FigureCanvasAgg above — get a fresh canvas to render to.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.critical(
                self, 'GIF export failed',
                'Pillow is required for GIF export.\n'
                'Install with: pip install Pillow')
            return

        n_frames = len(self.steps)
        # Cap frame count to keep GIF size manageable. GIF is large per
        # frame; >150 frames produces multi-MB files. Subsample uniformly.
        max_frames = 150
        if n_frames > max_frames:
            sel = np.linspace(0, n_frames - 1, max_frames).astype(int)
        else:
            sel = np.arange(n_frames)

        # User-set FPS (defaults to 15 in the spinbox).
        fps = self.spn_fps.value()
        duration_ms = max(1, int(round(1000.0 / fps)))

        # Offscreen figure, sized smaller than MP4 since GIF can't handle
        # high-resolution well — and big GIFs are huge.
        fig = Figure(facecolor=DARK_BG, figsize=(6.0, 5.6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax_lat = fig.add_axes([0.04, 0.04, 0.84, 0.92])
        cax = fig.add_axes([0.90, 0.10, 0.025, 0.78])

        Nz_site = self.template["Nz_site"]
        sm0 = plot_field_distribution(
            ax_lat, self.E_hist[sel[0]], self.Nx, self.Ny, self.template,
            title=f'round-trip n = {self.steps[sel[0]]/Nz_site:.1f}',
            I_max=self.I_max_ss)
        cbar = fig.colorbar(sm0, cax=cax,
                              label=r'$|E|^2 / \max|E_\infty|^2$')
        cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
        cbar.ax.yaxis.label.set_color(TEXT_COL)
        cbar.outline.set_edgecolor('#3a4560')

        self.btn_gif.setEnabled(False); self.btn_gif.setText('Encoding…')
        QApplication.processEvents()

        try:
            frames = []
            for k, i in enumerate(sel):
                ax_lat.clear()
                plot_field_distribution(
                    ax_lat, self.E_hist[i], self.Nx, self.Ny, self.template,
                    title=f'round-trip n = {self.steps[i]/Nz_site:.1f}',
                    I_max=self.I_max_ss)
                canvas.draw()
                # Pull RGBA buffer → PIL Image. Use buffer_rgba for
                # compatibility with newer matplotlib versions.
                buf = np.asarray(canvas.buffer_rgba())
                img = Image.fromarray(buf, 'RGBA').convert('P',
                                                              palette=Image.ADAPTIVE,
                                                              colors=128)
                frames.append(img)
                if k % 10 == 0:
                    self.progress.setValue(int(50 + 50 * k / len(sel)))
                    QApplication.processEvents()

            # Pillow's save with append_images writes all frames.
            frames[0].save(path, save_all=True, append_images=frames[1:],
                            duration=duration_ms, loop=0, optimize=False,
                            disposal=2)
            self.btn_gif.setText('💾 Save GIF')
            self.btn_gif.setEnabled(True)
            n_used = len(sel)
            note = (f'\n\n(Subsampled {n_frames} → {n_used} frames to keep '
                     f'file size manageable.)' if n_frames > max_frames else '')
            QMessageBox.information(
                self, 'Saved', f'GIF written:\n{path}{note}')
        except Exception as e:
            self.btn_gif.setText('💾 Save GIF')
            self.btn_gif.setEnabled(True)
            QMessageBox.critical(self, 'GIF export failed',
                                    f'Could not write GIF:\n{e}')




# ═════════════════════════════════════════════════════════════════════════════
#  Main window
# ═════════════════════════════════════════════════════════════════════════════

PHI_TICKS = 200   # 0..200 ticks -> 0..2 (in units of pi)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lida Xu's IQH-Lattice TMM Explorer — v1.2")
        self.setMinimumSize(1400, 800)
        icon_path = os.path.join(_BUNDLE_DIR, 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self._apply_dark_theme()
        self.state = dict(
            Nx=4, Ny=4, Phi0=np.pi / 2,
            kappa_ex=0.10, kappa_J=0.561,
            beta0_eta_over_pi=1.0, alpha=1e-4,
            omega_min=-0.1, omega_max=0.1, npts=4001,
            omegas=None, Td=None, Tt=None,
            template=None, peaks_omega=[], selected_peak_idx=-1,
            E_at_peak=None, last_omega=None,
            removed_sites=set(), removed_links=set(),
        )
        self._scan_worker = None
        self._build_ui()

    # ── Theme (verbatim from Linear.py) ──────────────────────────────────────
    def _apply_dark_theme(self):
        pal = QPalette()
        for role, col in [
            (QPalette.Window,          (8, 9, 13)),
            (QPalette.WindowText,      (200, 208, 231)),
            (QPalette.Base,            (14, 16, 24)),
            (QPalette.AlternateBase,   (23, 28, 46)),
            (QPalette.Text,            (200, 208, 231)),
            (QPalette.Button,          (23, 28, 46)),
            (QPalette.ButtonText,      (200, 208, 231)),
            (QPalette.Highlight,       (0, 229, 255)),
            (QPalette.HighlightedText, (8, 9, 13)),
        ]:
            pal.setColor(role, QColor(*col))
        self.setPalette(pal)
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#08090d;color:#c8d0e7;font-size:12px;}
            QGroupBox{border:1px solid #1e2230;border-radius:6px;margin-top:10px;
                      padding:8px;font-size:11px;font-weight:bold;color:#3a4a70;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QPushButton{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
                        padding:5px 12px;color:#c8d0e7;font-size:12px;}
            QPushButton:hover{background:#1e2a40;border-color:#00e5ff;}
            QPushButton:pressed,QPushButton:checked{background:#00e5ff;color:#08090d;border-color:#00e5ff;}
            QPushButton:disabled{color:#2a3050;}
            QComboBox,QDoubleSpinBox,QSpinBox,QLineEdit{background:#171c2e;border:1px solid #1e2230;
                border-radius:4px;padding:4px 6px;color:#c8d0e7;font-size:12px;}
            QSlider::groove:horizontal{height:4px;background:#1e2230;border-radius:2px;}
            QSlider::handle:horizontal{background:#00e5ff;width:14px;height:14px;
                margin:-5px 0;border-radius:7px;}
            QLabel{color:#c8d0e7;font-size:12px;}
            QStatusBar{color:#4a5270;font-size:11px;}
            QSplitter::handle{background:#1e2230;}
            QCheckBox{color:#c8d0e7;font-size:11px;}
            QProgressBar{border:1px solid #1e2230;border-radius:4px;background:#171c2e;
                          text-align:center;color:#c8d0e7;font-size:10px;height:16px;}
            QProgressBar::chunk{background:#00e5ff;}
        """)

    # ── UI layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(5); main.setContentsMargins(8, 8, 8, 8)

        # Title
        t = QLabel('IQH-LATTICE TMM EXPLORER')
        t.setFont(QFont('Courier New', 13, QFont.Bold))
        t.setStyleSheet(f'color:{ACCENT};padding:2px 0;')
        s = QLabel('z-discretized TMM  ·  sparse LU  ·  chiral edge modes')
        s.setFont(QFont('Courier New', 8))
        s.setStyleSheet('color:#7a8aaa;')
        main.addWidget(t); main.addWidget(s)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#1e2230;')
        main.addWidget(sep)

        # Two-canvas splitter (spectrum left, lattice/field right)
        splitter = QSplitter(Qt.Horizontal)

        # Spectrum canvas
        sw = QWidget(); sl = QVBoxLayout(sw); sl.setContentsMargins(0, 0, 0, 0)
        self.fig_spec = Figure(facecolor=DARK_BG)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.ax_thru = self.fig_spec.add_subplot(211)
        self.ax_drop = self.fig_spec.add_subplot(212)
        for ax, col, ttl in [(self.ax_thru, '#4a9eff', 'Thru port'),
                                (self.ax_drop, '#ff4a6e', 'Drop port')]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=10, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor('#3a4560'); sp.set_linewidth(1.2)
            ax.grid(True, color=GRID_COL, linewidth=0.4)
            # Initial range matches default sweep window; will follow the
            # spinboxes from here on.
            ax.set_xlim(-0.1, 0.1); ax.set_ylim(-0.05, 1.10)
        self.ax_drop.set_xlabel(r"$\omega/(2\pi)$  (FSR units)",
                                  color=TEXT_COL, fontsize=9)
        self.fig_spec.subplots_adjust(hspace=0.4, left=0.12, right=0.97,
                                         top=0.95, bottom=0.10)
        sl.addWidget(self.canvas_spec)
        # Click on spectrum jumps to nearest peak
        self.canvas_spec.mpl_connect('button_press_event', self._on_spec_click)

        # Lattice/field canvas
        lw = QWidget(); ll = QVBoxLayout(lw); ll.setContentsMargins(0, 0, 0, 0)
        self.fig_lat = Figure(facecolor=DARK_BG)
        self.canvas_lat = FigureCanvas(self.fig_lat)
        # Main axis and a DEDICATED colorbar axis — both created once with
        # explicit rectangles, never recreated. This is the fix for the
        # "field shrinks on every click" bug: previously each peak click
        # called fig.colorbar(ax=ax_lat, ...) which steals space from the
        # main axis. Now the colorbar lives in cax_lat and ax_lat is never
        # resized.
        self.ax_lat = self.fig_lat.add_axes([0.04, 0.04, 0.84, 0.91])
        self.cax_lat = self.fig_lat.add_axes([0.90, 0.10, 0.025, 0.78])
        self.ax_lat.set_facecolor(PANEL_BG)
        self.ax_lat.set_xticks([]); self.ax_lat.set_yticks([])
        for sp in self.ax_lat.spines.values():
            sp.set_edgecolor('#3a4560')
        ll.addWidget(self.canvas_lat)
        self._cbar = None
        # Click on a ring in the lattice panel → toggle its presence.
        self.canvas_lat.mpl_connect('button_press_event', self._on_lat_click)

        splitter.addWidget(sw); splitter.addWidget(lw)
        splitter.setSizes([700, 700])
        main.addWidget(splitter, stretch=1)

        # Bottom row: controls
        main.addWidget(self._build_controls())

        # Status bar
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage('Ready — click ▶ Compute.')

        # Initial: render empty lattice schematic
        self._invalidate_field_only()

    def _build_controls(self):
        w = QWidget(); row = QHBoxLayout(w)
        row.setSpacing(8); row.setContentsMargins(0, 0, 0, 0)

        # ── Lattice group ────────────────────────────────────────────────────
        gd = QGroupBox('Lattice'); gd.setFixedWidth(280)
        dl = QGridLayout(gd); dl.setSpacing(3)

        self.spn_nx = QSpinBox(); self.spn_nx.setRange(1, 12); self.spn_nx.setValue(4)
        self.spn_ny = QSpinBox(); self.spn_ny.setRange(1, 12); self.spn_ny.setValue(4)
        # Nx/Ny change geometry, so redraw the schematic (and invalidate spectrum)
        self.spn_nx.valueChanged.connect(self._on_size_change)
        self.spn_ny.valueChanged.connect(self._on_size_change)

        dl.addWidget(QLabel('Nx'), 0, 0); dl.addWidget(self.spn_nx, 1, 0)
        dl.addWidget(QLabel('Ny'), 0, 1); dl.addWidget(self.spn_ny, 1, 1)

        self.bus_lbl = QLabel()
        self.bus_lbl.setStyleSheet('color:#7a8aaa;font-size:10px;')
        dl.addWidget(self.bus_lbl, 2, 0, 1, 2)
        self._update_bus_label()

        # β₀η/π — the single physically meaningful link-vs-site phase knob.
        # 1.0 = full anti-resonance (IQH); 0.0 = link & site degenerate;
        # periodic mod 2.
        self.spn_beta0eta = QDoubleSpinBox()
        self.spn_beta0eta.setRange(0.0, 2.0)
        self.spn_beta0eta.setSingleStep(0.05)
        self.spn_beta0eta.setDecimals(3)
        self.spn_beta0eta.setValue(1.0)
        self.spn_beta0eta.setSuffix(' π')
        self.spn_beta0eta.setFixedWidth(70)
        self.spn_beta0eta.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(QLabel('β₀η'), 3, 0); dl.addWidget(self.spn_beta0eta, 3, 1)

        # Hint label so users know the magic value
        hint = QLabel('β₀η = π  →  anti-resonance')
        hint.setStyleSheet('color:#7a8aaa;font-size:10px;')
        dl.addWidget(hint, 4, 0, 1, 2)

        # α — round-trip intensity loss = α·L_site (with L_site=1).
        # Default α=0.01 ↔ ~1.2 GHz intrinsic linewidth at 750 GHz FSR
        # ↔ Q_int ~ 1.6e5 (typical thin-film LiNbO₃ microring).
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.0, 1.0); self.spn_alpha.setSingleStep(0.001)
        self.spn_alpha.setDecimals(5); self.spn_alpha.setValue(0.01)
        self.spn_alpha.setFixedWidth(70)
        self.spn_alpha.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(QLabel('α (loss)'), 5, 0); dl.addWidget(self.spn_alpha, 5, 1)

        row.addWidget(gd)

        # ── Couplings group ──────────────────────────────────────────────────
        gc = QGroupBox('Couplings'); gc.setFixedWidth(360)
        cl = QGridLayout(gc); cl.setSpacing(3)

        # κ_ex slider/spinbox (range 0.001 - 0.99 mapped to 1-990).
        # Default 0.163 ↔ κ_ex² = 0.0267 ↔ ~20 GHz at 750 GHz FSR.
        self.sld_kex = QSlider(Qt.Horizontal)
        self.sld_kex.setRange(1, 990); self.sld_kex.setValue(163)
        self.spn_kex = QDoubleSpinBox()
        self.spn_kex.setRange(0.001, 0.99); self.spn_kex.setDecimals(3)
        self.spn_kex.setSingleStep(0.005); self.spn_kex.setValue(0.163)
        self.spn_kex.setFixedWidth(70)
        self.sld_kex.valueChanged.connect(lambda v: self.spn_kex.setValue(v / 1000.))
        self.spn_kex.valueChanged.connect(lambda v: self.sld_kex.setValue(int(round(v * 1000))))
        self.sld_kex.valueChanged.connect(lambda _: self._invalidate())

        cl.addWidget(QLabel('κ_ex (bus↔site)'), 0, 0)
        cl.addWidget(self.sld_kex, 0, 1, 1, 2); cl.addWidget(self.spn_kex, 0, 3)

        # κ_J slider/spinbox.
        # Default 0.409 ↔ κ_J² ≈ 0.167 ↔ J ≈ 20 GHz at 750 GHz FSR
        # (using J = κ_J²·FSR/(2π) — see THEORY.md §6 for the factor of 2π).
        self.sld_kJ = QSlider(Qt.Horizontal)
        self.sld_kJ.setRange(1, 990); self.sld_kJ.setValue(409)
        self.spn_kJ = QDoubleSpinBox()
        self.spn_kJ.setRange(0.001, 0.99); self.spn_kJ.setDecimals(3)
        self.spn_kJ.setSingleStep(0.005); self.spn_kJ.setValue(0.409)
        self.spn_kJ.setFixedWidth(70)
        self.sld_kJ.valueChanged.connect(lambda v: self.spn_kJ.setValue(v / 1000.))
        self.spn_kJ.valueChanged.connect(lambda v: self.sld_kJ.setValue(int(round(v * 1000))))
        self.sld_kJ.valueChanged.connect(lambda _: self._invalidate())

        cl.addWidget(QLabel('κ_J  (site↔link)'), 1, 0)
        cl.addWidget(self.sld_kJ, 1, 1, 1, 2); cl.addWidget(self.spn_kJ, 1, 3)

        # Phi0 slider/spinbox (0..2 in units of pi)
        self.sld_phi = QSlider(Qt.Horizontal)
        self.sld_phi.setRange(0, PHI_TICKS); self.sld_phi.setValue(50)  # 0.5 pi
        self.spn_phi = QDoubleSpinBox()
        self.spn_phi.setRange(0.0, 2.0); self.spn_phi.setDecimals(3)
        self.spn_phi.setSingleStep(0.05); self.spn_phi.setSuffix(' π')
        self.spn_phi.setValue(0.5); self.spn_phi.setFixedWidth(70)
        self.sld_phi.valueChanged.connect(lambda v: self.spn_phi.setValue(v / 100.))
        self.spn_phi.valueChanged.connect(lambda v: self.sld_phi.setValue(int(round(v * 100))))
        self.sld_phi.valueChanged.connect(lambda _: self._invalidate())

        cl.addWidget(QLabel('Φ₀'), 2, 0)
        cl.addWidget(self.sld_phi, 2, 1, 1, 2); cl.addWidget(self.spn_phi, 2, 3)

        row.addWidget(gc)

        # ── Simulation group (frequency window + buttons + peak picker) ──────
        gr = QGroupBox('Simulation')
        rl = QGridLayout(gr); rl.setSpacing(3)

        self.spn_omin = QDoubleSpinBox()
        self.spn_omin.setRange(-10.0, 10.0); self.spn_omin.setSingleStep(0.01)
        self.spn_omin.setDecimals(3); self.spn_omin.setValue(-0.1)
        self.spn_omax = QDoubleSpinBox()
        self.spn_omax.setRange(-10.0, 10.0); self.spn_omax.setSingleStep(0.01)
        self.spn_omax.setDecimals(3); self.spn_omax.setValue(0.1)
        self.spn_npts = QSpinBox()
        self.spn_npts.setRange(101, 32001); self.spn_npts.setSingleStep(500)
        self.spn_npts.setValue(4001)
        # Sweep-window spinboxes: in addition to invalidating, also redraw
        # the (now-empty) spectrum axes with the new x-range so the user
        # gets immediate visual feedback.
        self.spn_omin.valueChanged.connect(self._on_window_changed)
        self.spn_omax.valueChanged.connect(self._on_window_changed)
        self.spn_npts.valueChanged.connect(lambda _: self._invalidate())

        rl.addWidget(QLabel('ω/(2π) min'), 0, 0); rl.addWidget(self.spn_omin, 0, 1)
        rl.addWidget(QLabel('max'),          0, 2); rl.addWidget(self.spn_omax, 0, 3)
        rl.addWidget(QLabel('# pts'),       0, 4); rl.addWidget(self.spn_npts, 0, 5)

        # Peak combo
        self.cmb_peak = QComboBox()
        self.cmb_peak.setEnabled(False)
        self.cmb_peak.currentIndexChanged.connect(self._on_peak_changed)
        rl.addWidget(QLabel('Peak'), 1, 0)
        rl.addWidget(self.cmb_peak, 1, 1, 1, 5)

        # Buttons
        self.btn_run = QPushButton('▶ Compute')
        self.btn_run.setFixedHeight(28)
        self.btn_run.clicked.connect(self._run_spectrum)

        self.btn_clear = QPushButton('✕ Clear')
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.clicked.connect(self._clear)

        self.btn_save = QPushButton('💾 Save')
        self.btn_save.setFixedHeight(28)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        self.btn_save.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')

        self.btn_reset = QPushButton('⟳ Reset All')
        self.btn_reset.setFixedHeight(28)
        self.btn_reset.clicked.connect(self._reset_all)

        self.btn_tevol = QPushButton('⏱ Time evol.')
        self.btn_tevol.setFixedHeight(28)
        self.btn_tevol.setEnabled(False)
        self.btn_tevol.clicked.connect(self._open_time_evolution)
        self.btn_tevol.setToolTip(
            'Open a window that iterates the buildup of the field at the\n'
            'currently displayed ω. Click somewhere on the spectrum or\n'
            'pick a peak first to set the detuning.')

        # Save path row. Default to the current working directory — that's
        # wherever the user launched the app from, and is usually what they
        # mean by "save it here". Frozen exes also use cwd (not the bundle
        # dir, which is read-only on some platforms).
        self.lbl_save_path = QLabel('Save to:')
        self.edit_save_path = QLineEdit(os.getcwd())
        self.btn_browse = QPushButton('Browse')
        self.btn_browse.setFixedWidth(80); self.btn_browse.setFixedHeight(28)
        self.btn_browse.clicked.connect(self._browse_save_path)

        # Progress bar
        from PyQt5.QtWidgets import QProgressBar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setFixedHeight(16)

        # row 2: progress
        rl.addWidget(self.progress, 2, 0, 1, 6)

        # row 3: buttons
        rl.addWidget(self.btn_run,   3, 0)
        rl.addWidget(self.btn_clear, 3, 1)
        rl.addWidget(self.btn_save,  3, 2)
        rl.addWidget(self.btn_tevol, 3, 3)
        rl.addWidget(self.btn_reset, 3, 4, 1, 2)

        # row 4: save path
        rl.addWidget(self.lbl_save_path,  4, 0)
        rl.addWidget(self.edit_save_path, 4, 1, 1, 4)
        rl.addWidget(self.btn_browse,     4, 5)

        row.addWidget(gr, stretch=1)
        return w

    # ── Invalidation ─────────────────────────────────────────────────────────
    def _update_bus_label(self):
        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        bus_in, bus_drop = default_bus_positions(Nx, Ny)
        ix_i, iy_i, sl_i = bus_in
        ix_d, iy_d, sl_d = bus_drop
        side_i = 'top' if sl_i == SLOT_TOP else 'bot'
        side_d = 'top' if sl_d == SLOT_TOP else 'bot'
        self.bus_lbl.setText(
            f'IN @ ({ix_i},{iy_i}) {side_i}   '
            f'OUT @ ({ix_d},{iy_d}) {side_d}')

    def _on_size_change(self, _):
        """Nx or Ny changed: invalidate AND redraw the lattice schematic.

        Also drops any ring removals — site/link indices are sized to the
        old lattice and shouldn't carry over to a different geometry.
        """
        self.state['removed_sites'] = set()
        self.state['removed_links'] = set()
        self._update_bus_label()
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

    def _on_window_changed(self, _):
        """Sweep window changed: invalidate AND redraw spectrum x-range."""
        self._invalidate()
        # _invalidate already redraws the spectrum (cleared) with the
        # current spinbox values, so nothing more to do here.

    def _invalidate(self):
        """Disable Save and Peak picker whenever any param changes.

        We do NOT redraw the field panel here — that gets refreshed only
        when the user clicks Compute and a peak gets selected. This keeps
        the visualization stable while you're tweaking sliders.
        """
        # If a scan is currently running, abort it
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.request_abort()

        self.btn_save.setEnabled(False)
        self.cmb_peak.setEnabled(False)
        self.cmb_peak.blockSignals(True)
        self.cmb_peak.clear()
        self.cmb_peak.blockSignals(False)
        self.btn_tevol.setEnabled(False)
        # Clear stored results
        self.state['omegas'] = None
        self.state['Td'] = None; self.state['Tt'] = None
        self.state['template'] = None
        self.state['peaks_omega'] = []
        self.state['selected_peak_idx'] = -1
        self.state['E_at_peak'] = None
        self.state['last_omega'] = None
        # Clear spectrum (x-axis follows current sweep window) but leave
        # the field panel showing whatever was last drawn.
        self._clear_spectrum_axes()
        self.canvas_spec.draw_idle()
        self.status.showMessage('Parameters changed — click ▶ Compute to update.')

    def _invalidate_field_only(self):
        """Re-render an empty lattice schematic so the structure is shown."""
        Nx = self.spn_nx.value()
        Ny = self.spn_ny.value()
        # Use a dummy field of zeros so rings/buses are drawn at lowest intensity
        bus_in, bus_drop = default_bus_positions(Nx, Ny)
        template = build_template(
            Nx, Ny, 0.0, 0.10, 0.561,
            bus_in=bus_in, bus_drop=bus_drop,
            removed_sites=self.state.get('removed_sites', set()),
            removed_links=self.state.get('removed_links', set()),
        )
        E_zero = np.zeros(template['state_size'], dtype=complex)
        self.ax_lat.clear()
        sm = plot_field_distribution(
            self.ax_lat, E_zero, Nx, Ny, template,
            title=f"{Nx}×{Ny} lattice — schematic",
            I_max=1.0,
        )
        self._update_colorbar(sm)
        self.canvas_lat.draw_idle()

    def _update_colorbar(self, sm):
        """Refresh the colorbar without ever resizing the main lattice axis.

        Critical: we use cax=self.cax_lat (a dedicated, pre-allocated axis),
        NOT ax=self.ax_lat. Passing ax= would call make_axes_gridspec
        which carves space out of the main axis on every call, shrinking
        the field plot a little each time.
        """
        self.cax_lat.clear()
        self._cbar = self.fig_lat.colorbar(sm, cax=self.cax_lat,
                                             label=r"$|E|^2$ (norm)")
        self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self._cbar.ax.yaxis.label.set_color(TEXT_COL)
        self._cbar.outline.set_edgecolor('#3a4560')

    def _clear_spectrum_axes(self):
        # x-axis follows the current sweep window. If a scan has run we use
        # the actual omegas; otherwise fall back to the spinbox values.
        if self.state.get('omegas') is not None:
            x = self.state['omegas'] / (2 * np.pi)
            xlo, xhi = float(x.min()), float(x.max())
        else:
            xlo = self.spn_omin.value()
            xhi = self.spn_omax.value()
            if xlo >= xhi:  # guard against momentary inversion while typing
                xlo, xhi = -0.1, 0.1

        for ax, col, ttl in [(self.ax_thru, '#4a9eff', 'Thru port'),
                                (self.ax_drop, '#ff4a6e', 'Drop port')]:
            ax.clear()
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=10, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor('#3a4560'); sp.set_linewidth(1.2)
            ax.grid(True, color=GRID_COL, linewidth=0.4)
            ax.set_xlim(xlo, xhi)
        # Thru is always 0..1 by construction → fixed range.
        self.ax_thru.set_ylim(0.0, 1.10)
        # Drop will be auto-ranged once data exists; before that, give it a
        # sensible default so the empty axes don't collapse.
        self.ax_drop.set_ylim(0.0, 1.10)
        self.ax_drop.set_xlabel(r"$\omega/(2\pi)$  (FSR units)",
                                  color=TEXT_COL, fontsize=9)

    # ── Run / Clear / Reset ──────────────────────────────────────────────────
    def _run_spectrum(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.request_abort()
            self._scan_worker.wait()

        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        Phi0 = self.spn_phi.value() * np.pi
        kappa_ex = self.spn_kex.value(); kappa_J = self.spn_kJ.value()
        beta0_eta_over_pi = self.spn_beta0eta.value()
        alpha = self.spn_alpha.value()
        omin = self.spn_omin.value(); omax = self.spn_omax.value()
        if omin >= omax:
            QMessageBox.warning(self, 'Bad range',
                                  'ω/(2π) min must be < ω/(2π) max.')
            return
        npts = self.spn_npts.value()
        omegas = np.linspace(omin * 2 * np.pi, omax * 2 * np.pi, npts)
        params = dict(Nx=Nx, Ny=Ny, Phi0=Phi0,
                       kappa_ex=kappa_ex, kappa_J=kappa_J,
                       beta0_eta_over_pi=beta0_eta_over_pi,
                       alpha=alpha, omegas=omegas,
                       removed_sites=set(self.state.get('removed_sites', set())),
                       removed_links=set(self.state.get('removed_links', set())))

        self.btn_run.setEnabled(False); self.btn_run.setText('Computing…')
        self.progress.setValue(0)
        self.status.showMessage(f'Scanning {npts} frequencies on a {Nx}×{Ny} lattice…')

        worker = ScanWorker(params)
        worker.progress.connect(self._on_scan_progress)
        worker.finished_ok.connect(self._on_scan_done)
        worker.failed.connect(self._on_scan_failed)
        self._scan_worker = worker
        self._t0 = time.time()
        worker.start()

    def _on_scan_progress(self, done, total):
        self.progress.setValue(int(done * 100 / max(1, total)))

    def _on_scan_failed(self, msg):
        self.btn_run.setEnabled(True); self.btn_run.setText('▶ Compute')
        QMessageBox.critical(self, 'Scan failed', msg)
        self.status.showMessage(f'Failed: {msg}')

    def _on_scan_done(self, omegas, Td, Tt, template):
        elapsed = time.time() - self._t0
        self.state['omegas'] = omegas
        self.state['Td'] = Td; self.state['Tt'] = Tt
        self.state['template'] = template

        # Find peaks on Td
        peak_idx, _ = find_peaks(Td, height=max(0.005 * Td.max(), 1e-6),
                                  distance=5)
        peaks_omega = sorted(omegas[peak_idx].tolist())
        self.state['peaks_omega'] = peaks_omega

        # Plot spectra
        self._draw_spectra(highlight_omega=None)

        # Populate combo
        self.cmb_peak.blockSignals(True)
        self.cmb_peak.clear()
        for w in peaks_omega:
            T_at = Td[np.argmin(np.abs(omegas - w))]
            self.cmb_peak.addItem(f"ω/(2π) = {w/(2*np.pi):+.5f}   "
                                    f"(T_drop ≈ {T_at:.3f})")
        self.cmb_peak.blockSignals(False)
        self.cmb_peak.setEnabled(len(peaks_omega) > 0)

        # Auto-pick peak nearest 0.025
        if peaks_omega:
            target = 0.025 * 2 * np.pi
            idx = int(np.argmin([abs(w - target) for w in peaks_omega]))
            self.cmb_peak.setCurrentIndex(idx)
            # The above triggers _on_peak_changed which renders the field
        else:
            self._invalidate_field_only()

        self.btn_save.setEnabled(True)
        self.btn_run.setEnabled(True); self.btn_run.setText('▶ Compute')
        self.status.showMessage(f'Done in {elapsed:.1f} s. Found {len(peaks_omega)} peaks.')

    def _draw_spectra(self, highlight_omega=None):
        omegas = self.state['omegas']
        if omegas is None:
            return
        Td = self.state['Td']; Tt = self.state['Tt']
        self._clear_spectrum_axes()
        x = omegas / (2 * np.pi)
        self.ax_thru.plot(x, Tt, lw=1.0, color='#4a9eff')
        self.ax_drop.plot(x, Td, lw=1.0, color='#ff4a6e')
        # Thru: fixed [0, 1.1]. Drop: 1.1 × max(|Td|), ymin pinned to 0.
        # The tiny ε floor below is only there to avoid a degenerate ylim
        # when the spectrum is literally zero (e.g., before any data has
        # been computed); for any real signal the 1.10×max wins.
        td_max = float(Td.max()) if Td.size else 0.0
        td_top = max(td_max * 1.10, 1e-6)
        self.ax_drop.set_ylim(0.0, td_top)
        if highlight_omega is not None:
            for ax in (self.ax_thru, self.ax_drop):
                ax.axvline(highlight_omega / (2 * np.pi),
                            color=ACCENT, ls=':', alpha=0.7, lw=1)
        self.canvas_spec.draw_idle()

    def _render_field_at(self, omega):
        """Solve at the given omega and render the lattice field.

        Used by both the peak combo (which passes a peak frequency) and
        click-anywhere on the spectrum (which passes any frequency in the
        scanned range).
        """
        if self.state['template'] is None:
            return
        beta0_eta_over_pi = self.spn_beta0eta.value()
        kappa_ex = self.spn_kex.value()
        alpha = self.spn_alpha.value()

        E, _, _ = solve_one(omega, self.state['template'],
                              beta0_eta_over_pi, kappa_ex, alpha)
        self.state['E_at_peak'] = E
        self.state['last_omega'] = omega
        self.btn_tevol.setEnabled(True)

        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        self.ax_lat.clear()
        sm = plot_field_distribution(
            self.ax_lat, E, Nx, Ny, self.state['template'],
            title=f"ω/(2π) = {omega/(2*np.pi):+.5f}",
        )
        self._update_colorbar(sm)
        self.canvas_lat.draw_idle()

        # Highlight in spectrum
        self._draw_spectra(highlight_omega=omega)

    def _on_peak_changed(self, idx):
        if idx < 0 or self.state['template'] is None:
            return
        peaks = self.state['peaks_omega']
        if idx >= len(peaks):
            return
        self.state['selected_peak_idx'] = idx
        self._render_field_at(peaks[idx])

    def _on_spec_click(self, event):
        """Click anywhere on either spectrum panel → render the field at
        EXACTLY that frequency (snapped to the nearest scanned ω sample).

        Uses figure-level coordinates so clicks on titles/borders/spines
        also count, not just inside the data region. We pick whichever
        axis (thru or drop) the click lands inside, translate the click's
        pixel x into data x, and solve at that frequency.
        """
        if self.state['template'] is None:
            return
        omegas = self.state.get('omegas')
        if omegas is None:
            return
        if event.x is None or event.y is None:
            return  # click outside the canvas

        # Determine which subplot the click is in via figure pixel bboxes.
        for ax in (self.ax_thru, self.ax_drop):
            bbox = ax.get_window_extent()
            if bbox.contains(event.x, event.y):
                inv = ax.transData.inverted()
                x_data, _ = inv.transform((event.x, event.y))
                omega_click = x_data * 2 * np.pi
                # Clamp to the scanned range; snap to the nearest sampled ω
                # so the field corresponds to a frequency we know was
                # actually computed (not an interpolated one).
                idx = int(np.argmin(np.abs(omegas - omega_click)))
                omega = omegas[idx]
                # If the click is close to a known peak, also update the
                # combo box selection — purely cosmetic / for keyboard
                # navigation continuity.
                peaks = self.state.get('peaks_omega', [])
                if peaks:
                    pk_idx = int(np.argmin([abs(w - omega) for w in peaks]))
                    if abs(peaks[pk_idx] - omega) < 1e-6:
                        self.cmb_peak.blockSignals(True)
                        self.cmb_peak.setCurrentIndex(pk_idx)
                        self.cmb_peak.blockSignals(False)
                self._render_field_at(omega)
                return

    def _on_lat_click(self, event):
        """Click on the lattice → toggle the nearest ring's removal state.

        Sites are at integer (ix, iy). H-links at (ix+0.5, iy), V-links
        at (ix, iy+0.5). The ring with center closest to the click wins,
        provided the click is within the ring's half-side.

        IN/OUT site rings are protected (cannot be removed).
        After toggling, the spectrum is invalidated and the schematic
        redrawn — the user must click Compute again to see the new
        spectrum.
        """
        if event.inaxes is not self.ax_lat:
            return
        if event.xdata is None or event.ydata is None:
            return
        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        bus_in, bus_drop = default_bus_positions(Nx, Ny)
        protected = {(bus_in[0], bus_in[1]), (bus_drop[0], bus_drop[1])}

        # Build list of (kind, key, center) for every ring in the lattice.
        candidates = []
        for iy in range(Ny):
            for ix in range(Nx):
                candidates.append(('site', (ix, iy), (float(ix), float(iy))))
        for iy in range(Ny):
            for ix in range(Nx - 1):
                candidates.append(('link', f'H_{ix}_{iy}',
                                    (ix + 0.5, float(iy))))
        for iy in range(Ny - 1):
            for ix in range(Nx):
                candidates.append(('link', f'V_{ix}_{iy}',
                                    (float(ix), iy + 0.5)))

        if not candidates:
            return
        click = (event.xdata, event.ydata)
        # Half-side of rendered rings is 0.24; pick the closest center
        # within that radius (use 0.30 to be a bit forgiving).
        threshold = 0.30
        best = None; best_d = float('inf')
        for kind, key, center in candidates:
            d = ((click[0] - center[0])**2 + (click[1] - center[1])**2) ** 0.5
            if d < best_d:
                best_d = d; best = (kind, key)
        if best is None or best_d > threshold:
            return

        kind, key = best
        if kind == 'site':
            if key in protected:
                self.status.showMessage(
                    f'Site {key} carries the bus — cannot remove.')
                return
            rs = self.state['removed_sites']
            if key in rs:
                rs.discard(key)
                self.status.showMessage(f'Restored site {key}.')
            else:
                rs.add(key)
                self.status.showMessage(f'Removed site {key}.')
        else:  # link
            rl = self.state['removed_links']
            if key in rl:
                rl.discard(key)
                self.status.showMessage(f'Restored link {key}.')
            else:
                rl.add(key)
                self.status.showMessage(f'Removed link {key}.')

        # Lattice geometry changed — invalidate spectrum, redraw schematic.
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

    def _clear(self):
        self._invalidate()

    def _reset_all(self):
        self.spn_nx.setValue(4)
        self.spn_ny.setValue(4)
        self.spn_phi.setValue(0.5)        # 0.5 pi
        self.spn_kex.setValue(0.163)      # κ_ex²·FSR ≈ 20 GHz at 750 GHz FSR
        self.spn_kJ.setValue(0.409)       # J ≈ 20 GHz (THEORY.md §6)
        self.spn_beta0eta.setValue(1.0)   # full anti-resonance
        self.spn_alpha.setValue(0.01)     # ~1.2 GHz intrinsic linewidth
        self.spn_omin.setValue(-0.1)
        self.spn_omax.setValue(0.1)
        self.spn_npts.setValue(4001)
        self.state['removed_sites'] = set()
        self.state['removed_links'] = set()
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

    def _open_time_evolution(self):
        """Open the time-evolution dialog at the currently displayed ω."""
        omega = self.state.get('last_omega')
        if omega is None or self.state.get('template') is None:
            QMessageBox.information(
                self, 'No detuning selected',
                'Click somewhere on the spectrum, or pick a peak from the\n'
                'dropdown, to set the detuning. Then try again.')
            return
        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        beta0_eta_over_pi = self.spn_beta0eta.value()
        kappa_ex = self.spn_kex.value()
        alpha = self.spn_alpha.value()
        save_dir = self.edit_save_path.text() or os.getcwd()
        dlg = TimeEvolutionDialog(
            self, omega, self.state['template'],
            beta0_eta_over_pi, kappa_ex, alpha, Nx, Ny, save_dir)
        # Non-modal — lets the user keep working with the main window
        dlg.show()

    def _save(self):
        save_dir = self.edit_save_path.text() or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        phi = self.spn_phi.value()
        base = f'TMM_{Nx}x{Ny}_phi{phi:.2f}pi_{ts}'
        path_spec = os.path.join(save_dir, base + '_spectrum.png')
        path_field = os.path.join(save_dir, base + '_field.png')
        self.fig_spec.savefig(path_spec, dpi=140, bbox_inches='tight',
                                facecolor=DARK_BG)
        self.fig_lat.savefig(path_field, dpi=140, bbox_inches='tight',
                                facecolor=DARK_BG)
        # Optionally also save raw arrays
        path_npz = os.path.join(save_dir, base + '_data.npz')
        if self.state['omegas'] is not None:
            np.savez(path_npz,
                      omegas=self.state['omegas'],
                      Td=self.state['Td'], Tt=self.state['Tt'],
                      Nx=Nx, Ny=Ny, Phi0=phi*np.pi,
                      kappa_ex=self.spn_kex.value(),
                      kappa_J=self.spn_kJ.value())
        self.status.showMessage(f'Saved: {os.path.basename(path_spec)}, '
                                  f'{os.path.basename(path_field)}, '
                                  f'{os.path.basename(path_npz)}')

    def _browse_save_path(self):
        d = QFileDialog.getExistingDirectory(self, 'Choose save folder',
                                                self.edit_save_path.text())
        if d:
            self.edit_save_path.setText(d)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()