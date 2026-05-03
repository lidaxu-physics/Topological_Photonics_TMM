"""
TMM.py
======
Lida Xu's IQH/AQH-Lattice TMM Explorer — v1.3
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
# Default grid points per ring (site and link). Can be overridden per-call
# via the Nz_site / Nz_link kwargs to build_template, build_template_aqh,
# etc. The UI exposes a spinbox that lets the user pick any multiple of 8
# from 16 to 64.
NZ_SITE = 16
NZ_LINK = 16


def _iqh_slots(Nz):
    """Return the IQH per-site slot positions for a given ring discretization.

    Slot positions scale proportionally with Nz/16 (the design value).
    Returns dict with keys 'BUS', 'RIGHT', 'TOP', 'BOTTOM', 'LEFT'.

    Requires Nz to be a multiple of 16 so that all slot positions are
    integer-valued (e.g., BOTTOM = 5*Nz/8).
    """
    f = Nz // 16
    return {
        'BUS':    0,
        'RIGHT':  4 * f,
        'TOP':    8 * f,
        'BOTTOM': 10 * f,
        'LEFT':   12 * f,
    }


def _aqh_slots(Nz):
    """Return the AQH per-ring cardinal slot positions for a given Nz.

    AQH uses N/E/S/W at the 4 cardinal points; requires Nz divisible by 4.
    """
    q = Nz // 4
    return {'N': 0, 'E': q, 'S': 2*q, 'W': 3*q}


# Default (Nz=16) slot constants — for backward-compatible references.
# Code that supports variable Nz should call _iqh_slots(Nz)/_aqh_slots(Nz)
# instead of using these.
SLOT_BUS    = 0
SLOT_RIGHT  = 4
SLOT_TOP    = 8
SLOT_BOTTOM = 10
SLOT_LEFT   = 12


def default_bus_positions(Nx, Ny, Nz_site=None):
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
    - Single ring (1×1):     IN at slot 0 (bottom), OUT at slot Nz/2 (top).
                             Standard add-drop filter geometry.

    Slot values scale with Nz_site (defaults to module-level NZ_SITE).
    """
    if Nz_site is None:
        Nz_site = NZ_SITE
    slots = _iqh_slots(Nz_site)
    s_bus = slots['BUS']; s_top = slots['TOP']
    if Ny == 1:
        # Horizontal chain (or single ring on 1x1)
        if Nx == 1:
            return (0, 0, s_bus), (0, 0, s_top)
        return (0, 0, s_bus), (Nx - 1, 0, s_bus)
    # 2D lattice, Ny ≥ 2
    return (0, 0, s_bus), (0, Ny - 1, s_top)


# ═════════════════════════════════════════════════════════════════════════════
#  AQH lattice geometry
# ═════════════════════════════════════════════════════════════════════════════
#
# Photonic AQH lattice: a brick-wall arrangement on a (2Nx-1)×(2Ny-1) grid.
#
#   Grid coordinate (c, r): c in [0, 2Nx-2], r in [0, 2Ny-2].
#
#   - Sites placed where c+r is even AND r is "narrow-row even" or
#     "wide-row even" — explicitly:
#         r even (rows 0, 2, ..., "narrow"):  sites at odd c       (Nx-1 sites)
#         r odd  (rows 1, 3, ..., "wide"):    sites at even c        (Nx sites)
#   - Link rings placed where r is odd AND c is odd:
#         (Nx-1) per wide row, Ny-1 wide rows → (Nx-1)*(Ny-1) total links.
#
# Site count: Nx*(Ny-1) + Ny*(Nx-1).
# Link count: (Nx-1)*(Ny-1).
#
# Connectivity:
#   - Each link ring at (c, r) couples to its 4 nearest grid neighbors:
#         (c, r-1) — site above (N)
#         (c, r+1) — site below (S)
#         (c-1, r) — site left  (W)
#         (c+1, r) — site right (E)
#   - Each narrow-row site has 2 link DCs (links above AND below).
#   - Each wide-row site has 2 link DCs (links left AND right).
#
# Each ring has 4 DCs around its 16-grid perimeter:
#   - Site rings:  2 link DCs + 1 bus DC + 1 unused slot.
#   - Link rings:  4 site DCs.
#
# Slot conventions (16 grid points around each ring perimeter):
#   Slot 0  = NORTH side (top of ring)
#   Slot 4  = EAST  side (right of ring)
#   Slot 8  = SOUTH side (bottom)
#   Slot 12 = WEST  side (left)
# DCs are placed at the slot whose direction matches the neighbor's
# physical position, e.g. a site above→Slot 0 (N) on this ring AND
# Slot 8 (S) on the neighbor. The DC arc joining the two rings is
# tangent at both, so they share a phase reference at that DC.
#
# Default bus positions: lower-left and upper-left site rings, bus DC on
# whichever side is "free" (i.e. has no link neighbor).
# ═════════════════════════════════════════════════════════════════════════════

# AQH site DC slots (4 cardinal directions; bus uses whichever is free)
SLOT_AQH_N = 0
SLOT_AQH_E = 4
SLOT_AQH_S = 8
SLOT_AQH_W = 12


def aqh_site_count(Nx, Ny):
    """Brick-wall site count: Nx*(Ny-1) + Ny*(Nx-1)."""
    return Nx * (Ny - 1) + Ny * (Nx - 1)


def aqh_link_count(Nx, Ny):
    """Brick-wall link count: (Nx-1)*(Ny-1)."""
    return max(0, (Nx - 1) * (Ny - 1))


def aqh_grid_dims(Nx, Ny):
    """Grid dimensions for the bricklike layout."""
    return 2 * Nx - 1, 2 * Ny - 1   # n_cols, n_rows


def aqh_site_positions(Nx, Ny):
    """Return list of (c, r, plot_x, plot_y) for every site, in canonical
    ordering (row-by-row, left-to-right within each row).

    Plot coordinates are scaled to half the grid coords, so that adjacent
    rings (site↔link) are at distance 0.5 — matching the IQH ring spacing
    on a unit grid.
    """
    sites = []
    for r in range(2 * Ny - 1):
        if r % 2 == 0:                # narrow row: sites at odd c
            for c in range(1, 2 * Nx - 1, 2):
                sites.append((c, r, 0.5 * c, 0.5 * r))
        else:                          # wide row: sites at even c
            for c in range(0, 2 * Nx - 1, 2):
                sites.append((c, r, 0.5 * c, 0.5 * r))
    return sites


def aqh_link_positions(Nx, Ny):
    """Return list of (c, r, plot_x, plot_y) for every link ring."""
    links = []
    for r in range(1, 2 * Ny - 1, 2):
        for c in range(1, 2 * Nx - 1, 2):
            links.append((c, r, 0.5 * c, 0.5 * r))
    return links


def aqh_site_index_lookup(Nx, Ny):
    """Build a {(c, r): 1-based site index} dict."""
    sites = aqh_site_positions(Nx, Ny)
    return {(c, r): i for i, (c, r, _, _) in enumerate(sites, start=1)}


def aqh_link_index_lookup(Nx, Ny):
    """Build a {(c, r): 1-based link index} dict."""
    links = aqh_link_positions(Nx, Ny)
    return {(c, r): j for j, (c, r, _, _) in enumerate(links, start=1)}


def aqh_default_bus_positions(Nx, Ny):
    """Bus IN at lower-left corner site, OUT at upper-left corner site.
    Returns (in_idx, out_idx) as 1-based site indices.

    Plot uses ax.invert_yaxis(), so smaller plot_y is visually higher.
    Lower-left site is at grid (1, 2*Ny-2) — bottom row, leftmost narrow-row.
    Upper-left site is at grid (1, 0) — top row, leftmost narrow-row.
    Both are narrow-row sites with one vertical link neighbor.

    The bus DC sits on the side opposite to the link neighbour, so the
    bus exits the lattice toward its outer edge:
        IN  (lower-left) → bus exits below the lattice
        OUT (upper-left) → bus exits above the lattice
    """
    sites = aqh_site_positions(Nx, Ny)
    if not sites:
        return 1, 1
    if Nx <= 1 or Ny <= 1:
        return 1, len(sites)
    site_lookup = aqh_site_index_lookup(Nx, Ny)
    in_idx  = site_lookup[(1, 2 * Ny - 2)]      # bottom-row narrow-row site
    out_idx = site_lookup[(1, 0)]                # top-row narrow-row site
    return in_idx, out_idx


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
                    removed_sites=None, removed_links=None,
                    dc_flip=False):
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

    dc_flip: if True, flip the slot ordering convention everywhere — the
        photon's "predecessor" slot becomes (k+1) instead of (k-1) at every
        grid point. Physically: every ring's circulation reverses (CW ↔ CCW).
        This is what changing the bus DC tangent direction does in
        experiment — light injected from the OPPOSITE end of the bus
        couples to the OPPOSITE pseudospin of the ring. Spectrum is
        invariant by reciprocity, but transient buildup paths differ.
    """
    removed_sites = set(removed_sites) if removed_sites else set()
    removed_links = set(removed_links) if removed_links else set()

    # Accept both (ix, iy) and (ix, iy, slot) tuples. Slot constants
    # depend on Nz_site (they scale with Nz_site/16). Validate that the
    # ring discretization is a multiple of 16 so that all required slots
    # land on integer grid positions.
    if Nz_site % 16 != 0:
        raise ValueError(
            f"Nz_site must be a multiple of 16 (got {Nz_site}). The IQH "
            f"slot allocation (BUS=0, RIGHT=Nz/4, TOP=Nz/2, BOTTOM=5Nz/8, "
            f"LEFT=3Nz/4) requires this for all slots to be integer-valued.")
    iqh_slots = _iqh_slots(Nz_site)
    slot_bus    = iqh_slots['BUS']
    slot_right  = iqh_slots['RIGHT']
    slot_top    = iqh_slots['TOP']
    slot_bottom = iqh_slots['BOTTOM']
    slot_left   = iqh_slots['LEFT']

    def _normalize_bus(b, default_slot):
        if len(b) == 2:
            return (b[0], b[1], default_slot)
        return b
    bus_in = _normalize_bus(bus_in, slot_bus)        # default bottom slot
    bus_drop = _normalize_bus(bus_drop, slot_bus)    # default bottom slot too

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
            if (ix, iy) == (bus_in[0], bus_in[1]):
                slots[bus_in[2]] = ("bus_in", None)
            if (ix, iy) == (bus_drop[0], bus_drop[1]):
                slots[bus_drop[2]] = ("bus_drop", None)
            if ix > 0:
                slots[slot_left] = ("link", f"H_{ix-1}_{iy}", "far")
            if ix < Nx - 1:
                slots[slot_right] = ("link", f"H_{ix}_{iy}", "near")
            if iy > 0:
                slots[slot_bottom] = ("link", f"V_{ix}_{iy-1}", "far")
            if iy < Ny - 1:
                slots[slot_top] = ("link", f"V_{ix}_{iy}", "near")
            site_neighbors[(ix, iy)] = slots

    link_sites = {}
    for name in h_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        link_sites[name] = {"near": (ix, iy, slot_right),
                             "far":  (ix + 1, iy, slot_left)}
    for name in v_link_names:
        _, ix_str, iy_str = name.split("_")
        ix, iy = int(ix_str), int(iy_str)
        link_sites[name] = {"near": (ix, iy, slot_top),
                             "far":  (ix, iy + 1, slot_bottom)}

    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    t_J = np.sqrt(1.0 - kappa_J ** 2)

    # Direction of perimeter slot ordering: +1 = CCW (default), -1 = CW.
    # When dc_flip is True, every "predecessor" slot is the OTHER neighbour
    # in the perimeter — so light flows the opposite way around each ring.
    # This is the per-ring chirality flip: experimentally realised by
    # mirror-flipping the bus DC tangent direction (Input: left ↔ right).
    dir_sign = -1 if dc_flip else +1

    entries = []; src_rows = []; src_vals = []
    for (ix, iy), slots in site_neighbors.items():
        site_removed = (ix, iy) in removed_sites
        for k in range(Nz_site):
            k_prev = (k - dir_sign) % Nz_site
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
                            link_k_dc = 0
                            link_k_prev = (link_k_dc - dir_sign) % Nz_link
                        else:
                            link_k_dc = half_link
                            link_k_prev = (link_k_dc - dir_sign) % Nz_link
                        extra_phase_dc = extras[link_name][link_k_dc]
                        entries.append((row, site_idx(ix, iy, k_prev),
                                         "p_site", -t_J))
                        entries.append((row, link_idx(link_name, link_k_prev),
                                         "p_link_extra", -1j * kappa_J,
                                         extra_phase_dc))
            else:
                entries.append((row, site_idx(ix, iy, k_prev), "p_site", -1.0))

    for name, ends in link_sites.items():
        link_removed = name in removed_links
        for k in range(Nz_link):
            k_prev = (k - dir_sign) % Nz_link
            row = link_idx(name, k)
            extra_phase_k = extras[name][k]
            if (k == 0 or k == half_link) and not link_removed:
                site_info = ends["near"] if k == 0 else ends["far"]
                site_ix, site_iy, site_slot = site_info
                if (site_ix, site_iy) in removed_sites:
                    entries.append((row, link_idx(name, k_prev),
                                     "p_link_extra", -1.0, extra_phase_k))
                else:
                    site_k_prev = (site_slot - dir_sign) % Nz_site
                    entries.append((row, link_idx(name, k_prev),
                                     "p_link_extra", -t_J, extra_phase_k))
                    entries.append((row, site_idx(site_ix, site_iy, site_k_prev),
                                     "p_site", -1j * kappa_J))
            else:
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
        dc_flip=dc_flip,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  AQH TMM template
# ═════════════════════════════════════════════════════════════════════════════
#
# Build sparse matrix entries for the AQH brick-wall lattice.
#
# Conventions:
#   - Sites and links each have Nz_site / Nz_link grid points.
#   - Site DCs at slots {N=0, E=4, S=8, W=12}; bus on whichever cardinal slot
#     has no link neighbour.
#   - Link DCs at the same 4 cardinal slots, each connecting to one neighbour
#     site (N → site at (c, r-1), E → (c+1, r), S → (c, r+1), W → (c-1, r)).
#   - At each DC, the matched slots are: this ring's slot k connects to the
#     neighbour ring's antipodal slot (k+8 mod 16) so the DC pair sits on the
#     same physical line tangent to both rings.
#
# The Peierls phase φ_x,y is the link's full round-trip phase. We distribute
# it uniformly across all Nz_link grid points so each step contributes
# φ/Nz_link to the propagation phase. β₀η enters the same way.
#
# Building convention follows IQH: each row of the sparse system corresponds
# to E_k = (sum of incoming amplitudes) at the next grid point, encoded as
# (M = I + R) E = s with M, R off-diagonal entries (-coeff·p_factor).
# ═════════════════════════════════════════════════════════════════════════════


def build_template_aqh(Nx, Ny, phi0, kappa_ex, kappa_J,
                          bus_in_idx=None, bus_drop_idx=None,
                          Nz_site=NZ_SITE, Nz_link=NZ_LINK,
                          removed_sites=None, removed_links=None,
                          dc_flip=False):
    """Build the AQH TMM template for a brick-wall (2Nx-1)×(2Ny-1) lattice.

    Parameters
    ----------
    Nx, Ny : int
        Brick-wall dimensions. Sites = Nx·(Ny-1) + Ny·(Nx-1), links = (Nx-1)(Ny-1).
    phi0 : float
        Round-trip Peierls phase per site→link DC pair (radians).
        (Currently unused — link Peierls phases are built in via slot pattern;
        kept as parameter for future use.)
    kappa_ex, kappa_J : float
        Bus and link DC amplitude couplings.
    bus_in_idx, bus_drop_idx : int (1-based)
        Site indices for IN and OUT. Default = aqh_default_bus_positions().
    dc_flip : bool, default False
        If True, flip the slot-ordering convention at every ring. Photon
        circulation reverses (CW ↔ CCW). Experimentally: this is what
        flipping the bus DC tangent direction does — light enters from
        the opposite side of the bus and excites the opposite pseudospin.
    """
    removed_sites = set(removed_sites) if removed_sites else set()
    removed_links = set(removed_links) if removed_links else set()

    sites = aqh_site_positions(Nx, Ny)
    links = aqh_link_positions(Nx, Ny)
    n_sites = len(sites); n_links = len(links)
    if n_sites == 0:
        raise ValueError(
            f"AQH {Nx}×{Ny} has no rings — Nx and Ny must both be ≥ 2.")

    site_lookup = {(c, r): i for i, (c, r, _, _) in enumerate(sites, start=1)}
    link_lookup = {(c, r): j for j, (c, r, _, _) in enumerate(links, start=1)}

    if bus_in_idx is None or bus_drop_idx is None:
        bus_in_idx, bus_drop_idx = aqh_default_bus_positions(Nx, Ny)
    removed_sites = removed_sites - {bus_in_idx, bus_drop_idx}

    # State vector layout: sites first, then links.
    state_size = n_sites * Nz_site + n_links * Nz_link
    site_offset = {i: (i - 1) * Nz_site for i in range(1, n_sites + 1)}
    link_base   = n_sites * Nz_site
    link_offset = {j: link_base + (j - 1) * Nz_link for j in range(1, n_links + 1)}

    def site_idx(i, k):
        return site_offset[i] + (k % Nz_site)
    def link_idx(j, k):
        return link_offset[j] + (k % Nz_link)

    # Slot constants depend on Nz_site (cardinal positions at quarter-turns).
    # Require Nz_site divisible by 4 for clean integer slots.
    if Nz_site % 4 != 0:
        raise ValueError(
            f"Nz_site must be a multiple of 4 (got {Nz_site}). The AQH "
            f"slot layout uses N=0, E=Nz/4, S=Nz/2, W=3Nz/4.")
    aqh_slots = _aqh_slots(Nz_site)
    sN = aqh_slots['N']; sE = aqh_slots['E']
    sS = aqh_slots['S']; sW = aqh_slots['W']

    # Antipodal slot pairs (DC pair convention): N↔S, E↔W.
    # When ring A connects to ring B via a DC, A's slot s mates with B's
    # slot (s+Nz/2) mod Nz — i.e. the two rings are tangent at their facing
    # edges, with photon arcs going around opposite halves.
    SLOT_OPP = {sN: sS, sS: sN, sE: sW, sW: sE}

    # Direction of each cardinal neighbor (Δc, Δr) given a starting point
    DIR_OFFSET = {sN: ( 0, -1), sS: ( 0,  1),
                   sE: ( 1,  0), sW: (-1,  0)}

    # ── For each site, find which slots have link neighbors. ─────────────
    # site_dc_map[i] = {slot: ("link", link_j, "this_slot", "neighbor_slot")
    #                       | ("bus_in", None, ...) | ("bus_drop", None, ...)}
    site_dc_map = {i: {} for i in range(1, n_sites + 1)}
    for i, (c, r, _, _) in enumerate(sites, start=1):
        for slot, (dc, dr) in DIR_OFFSET.items():
            j_idx = link_lookup.get((c + dc, r + dr))
            if j_idx is not None:
                neighbor_slot = SLOT_OPP[slot]
                site_dc_map[i][slot] = ("link", j_idx, neighbor_slot)

    # Pick bus slot opposite to the closest link neighbour, so the bus
    # exits the ring toward the lattice exterior (matching IQH convention).
    # Strategy:
    #   - If exactly one link DC is occupied (corner sites), put the bus at
    #     the antipodal slot (S, E, W, or N).
    #   - Otherwise, fall through to a fixed preference order.
    def pick_bus_slot(site_idx_1based):
        taken = set(site_dc_map[site_idx_1based].keys())
        link_slots = [s for s in (sN, sE, sS, sW) if s in taken]
        if len(link_slots) == 1:
            opposite = SLOT_OPP[link_slots[0]]
            if opposite not in taken:
                return opposite
        # Fallback: prefer N → S → E → W (favours exits at top/bottom edges)
        for s in (sN, sS, sE, sW):
            if s not in taken:
                return s
        return None

    bus_in_slot = pick_bus_slot(bus_in_idx)
    if bus_in_slot is None:
        raise RuntimeError(f"Cannot place bus IN on site {bus_in_idx}: no free slots")
    site_dc_map[bus_in_idx][bus_in_slot] = ("bus_in", None, None)

    if bus_drop_idx != bus_in_idx:
        bus_drop_slot = pick_bus_slot(bus_drop_idx)
        if bus_drop_slot is None:
            raise RuntimeError(f"Cannot place bus DROP on site {bus_drop_idx}: no free slots")
        site_dc_map[bus_drop_idx][bus_drop_slot] = ("bus_drop", None, None)
    else:
        bus_drop_slot = bus_in_slot   # degenerate, single-site case

    # ── For each link, find which slots have site neighbors. ─────────────
    # link_dc_map[j] = {slot: (site_i, "neighbor_slot")}
    link_dc_map = {j: {} for j in range(1, n_links + 1)}
    for j, (c, r, _, _) in enumerate(links, start=1):
        for slot, (dc, dr) in DIR_OFFSET.items():
            i_idx = site_lookup.get((c + dc, r + dr))
            if i_idx is not None:
                link_dc_map[j][slot] = (i_idx, SLOT_OPP[slot])

    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    t_J  = np.sqrt(1.0 - kappa_J ** 2)

    entries = []; src_rows = []; src_vals = []

    # Direction of perimeter slot ordering (see IQH build_template).
    dir_sign = -1 if dc_flip else +1

    # ── Site equations ───────────────────────────────────────────────────
    for i in range(1, n_sites + 1):
        site_removed = (i in removed_sites)
        slots = site_dc_map[i]
        for k in range(Nz_site):
            k_prev = (k - dir_sign) % Nz_site
            row = site_idx(i, k)
            if k in slots and not site_removed:
                action = slots[k]; kind = action[0]
                if kind == "bus_in":
                    entries.append((row, site_idx(i, k_prev), "p_site", -t_ex))
                    src_rows.append(row); src_vals.append(1j * kappa_ex)
                elif kind == "bus_drop":
                    entries.append((row, site_idx(i, k_prev), "p_site", -t_ex))
                else:  # link
                    _, link_j, neighbor_slot = action
                    if link_j in removed_links:
                        entries.append((row, site_idx(i, k_prev),
                                         "p_site", -1.0))
                    else:
                        # Self-coupling (transmitted)
                        entries.append((row, site_idx(i, k_prev),
                                         "p_site", -t_J))
                        # Cross-coupling from the link, at predecessor of
                        # the link's DC slot for this neighbor.
                        link_k_prev = (neighbor_slot - dir_sign) % Nz_link
                        entries.append((row, link_idx(link_j, link_k_prev),
                                         "p_link", -1j * kappa_J))
            else:
                entries.append((row, site_idx(i, k_prev), "p_site", -1.0))

    # ── Link equations ───────────────────────────────────────────────────
    for j in range(1, n_links + 1):
        link_removed = (j in removed_links)
        slots = link_dc_map[j]
        for k in range(Nz_link):
            k_prev = (k - dir_sign) % Nz_link
            row = link_idx(j, k)
            if k in slots and not link_removed:
                site_i, neighbor_slot = slots[k]
                if site_i in removed_sites:
                    entries.append((row, link_idx(j, k_prev),
                                     "p_link", -1.0))
                else:
                    site_k_prev = (neighbor_slot - dir_sign) % Nz_site
                    entries.append((row, link_idx(j, k_prev),
                                     "p_link", -t_J))
                    entries.append((row, site_idx(site_i, site_k_prev),
                                     "p_site", -1j * kappa_J))
            else:
                entries.append((row, link_idx(j, k_prev),
                                 "p_link", -1.0))

    # Pack arrays. kinds: 0 = p_site, 1 = p_link.
    rows = np.array([e[0] for e in entries], dtype=np.int32)
    cols = np.array([e[1] for e in entries], dtype=np.int32)
    coeffs = np.array([e[3] for e in entries], dtype=complex)
    kinds = np.array([0 if e[2] == "p_site" else 1 for e in entries], dtype=np.int8)
    extras_arr = np.zeros(len(entries), dtype=float)

    return dict(
        rows=rows, cols=cols, kinds=kinds, coeffs=coeffs,
        extras_arr=extras_arr,
        diag_rows=np.arange(state_size, dtype=np.int32),
        src_rows_arr=np.array(src_rows, dtype=np.int32),
        src_vals_arr=np.array(src_vals, dtype=complex),
        state_size=state_size,
        site_idx=site_idx, link_idx=link_idx,
        lattice_type="AQH",
        Nx=Nx, Ny=Ny, n_sites=n_sites, n_links=n_links,
        sites=sites, links=links,
        bus_in_idx=bus_in_idx, bus_drop_idx=bus_drop_idx,
        bus_in_slot=bus_in_slot, bus_drop_slot=bus_drop_slot,
        Nz_site=Nz_site, Nz_link=Nz_link,
        phi0=phi0,
        removed_sites=removed_sites, removed_links=removed_links,
        dc_flip=dc_flip,
    )


def solve_one_aqh(omega, template, beta0_eta_over_pi, kappa_ex, alpha):
    """AQH steady-state solver. Same structure as IQH solve_one but
    only 2 entry kinds (0=p_site, 1=p_link).
    """
    L_site = 1.0
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]
    dz_site = L_site / Nz_site
    dz_link = L_site / Nz_link

    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)
    beta0_eta = beta0_eta_over_pi * np.pi
    p_link = np.exp(1j * omega * dz_link
                     + 1j * beta0_eta / Nz_link
                     - alpha * dz_link / 2.0)

    rows = template["rows"]; cols = template["cols"]
    kinds = template["kinds"]; coeffs = template["coeffs"]
    diag_rows = template["diag_rows"]
    src_rows_arr = template["src_rows_arr"]
    src_vals_arr = template["src_vals_arr"]
    state_size = template["state_size"]

    vals = np.where(kinds == 0, coeffs * p_site, coeffs * p_link)
    rows_full = np.concatenate([rows, diag_rows])
    cols_full = np.concatenate([cols, diag_rows])
    vals_full = np.concatenate([vals, np.ones(state_size, dtype=complex)])

    M = csc_matrix((vals_full, (rows_full, cols_full)),
                    shape=(state_size, state_size))
    s = np.zeros(state_size, dtype=complex)
    s[src_rows_arr] = src_vals_arr

    lu = splu(M)
    E = lu.solve(s)

    # Through and drop ports
    t_ex = np.sqrt(1.0 - kappa_ex ** 2)
    site_idx = template["site_idx"]
    bus_in_idx   = template["bus_in_idx"];   bus_in_slot   = template["bus_in_slot"]
    bus_drop_idx = template["bus_drop_idx"]; bus_drop_slot = template["bus_drop_slot"]
    dir_sign = -1 if template.get("dc_flip", False) else +1
    pred_in   = (bus_in_slot   - dir_sign) % Nz_site
    pred_drop = (bus_drop_slot - dir_sign) % Nz_site
    e_at_in   = p_site * E[site_idx(bus_in_idx,   pred_in)]
    e_at_drop = p_site * E[site_idx(bus_drop_idx, pred_drop)]
    s_thru = t_ex * 1.0 + 1j * kappa_ex * e_at_in
    s_drop = 1j * kappa_ex * e_at_drop

    return E, s_drop, s_thru


def build_propagator_aqh(omega, template, beta0_eta_over_pi, alpha):
    """Return (R, s) for AQH iteration: E^(n+1) = R E^(n) + s."""
    L_site = 1.0
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]
    dz_site = L_site / Nz_site
    dz_link = L_site / Nz_link

    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)
    beta0_eta = beta0_eta_over_pi * np.pi
    p_link = np.exp(1j * omega * dz_link
                     + 1j * beta0_eta / Nz_link
                     - alpha * dz_link / 2.0)

    rows = template["rows"]; cols = template["cols"]
    kinds = template["kinds"]; coeffs = template["coeffs"]
    src_rows_arr = template["src_rows_arr"]
    src_vals_arr = template["src_vals_arr"]
    state_size = template["state_size"]

    vals = np.where(kinds == 0, -coeffs * p_site, -coeffs * p_link)
    R = csc_matrix((vals, (rows, cols)), shape=(state_size, state_size))
    s = np.zeros(state_size, dtype=complex)
    s[src_rows_arr] = src_vals_arr
    return R, s


def solve_one(omega, template, beta0_eta_over_pi, kappa_ex, alpha):
    """IQH steady-state solver: solve (I - R) E = s and return (E, s_drop, s_thru).

    Per-step propagation factors:
        p_site      = e^{iω Δz - αΔz/2}
        p_link_base = e^{iω Δz + iβ₀η/Nz_link - αΔz/2}

    PHYSICS NOTE — DO NOT confuse the *physical* length of the link ring
    with the *detuning-dependent* phase. In a real IQH-lattice device, the
    link ring is longer than the site ring by η = λ₀/(2 n_eff), which is
    half a wavelength — TINY (η/L_site ~ 10⁻⁴). Yet β₀ η = π because β₀
    is large.

    The link's round-trip phase decomposes as
        β L_link = β₀ L_link + ω L_link
                 = (β₀ L_site + β₀ η) + ω(L_site + η)
                 = (carrier wraps to 2πN) + β₀η + ω L_site + ω η
    On FSR scales (|ω| ~ 2π) the ω η piece is ~10⁻⁴·2π — negligible. The
    *only* non-negligible link-vs-site difference is the static β₀ η.

    The single physically meaningful knob is β₀η, exposed in units of π:
        beta0_eta_over_pi = 1.0  →  full anti-resonance (IQH default)
        beta0_eta_over_pi = 0.0  →  link & site degenerate (mess)
        beta0_eta_over_pi = 0.5  →  link tuned a quarter-FSR off
    Periodic mod 2 (any integer multiple of 2π is the identity).
    """
    L_site = 1.0
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]
    dz_site = L_site / Nz_site
    # Link uses L_site for the propagation length too — see docstring.
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

    # IQH entries: kind 0 = p_site, kind 1 = p_link_extra
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
    # the bus DC is at the predecessor grid index. Direction depends on dc_flip.
    dir_sign = -1 if template.get("dc_flip", False) else +1
    i_in, j_in, s_in_slot = template["bus_in"]
    pred_in = (s_in_slot - dir_sign) % Nz_site
    e_in_at_bus_in = p_site * E[site_idx(i_in, j_in, pred_in)]
    s_thru = t_ex * 1.0 + 1j * kappa_ex * e_in_at_bus_in
    i_d, j_d, s_d_slot = template["bus_drop"]
    pred_d = (s_d_slot - dir_sign) % Nz_site
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
    is_aqh = template.get("lattice_type") == "AQH"
    if is_aqh:
        R, s = build_propagator_aqh(omega, template, beta0_eta_over_pi, alpha)
    else:
        R, s = build_propagator(omega, template, beta0_eta_over_pi, alpha)

    Nz_site = template["Nz_site"]
    site_idx = template["site_idx"]
    p_site_factor = np.exp(1j * omega * (1.0 / Nz_site)
                              - alpha * (1.0 / Nz_site) / 2.0)
    dir_sign = -1 if template.get("dc_flip", False) else +1
    if is_aqh:
        # AQH: bus IN/OUT are 1-based site indices; their bus DC slots are
        # stored in the template (typically WEST = 12, depends on geometry).
        bus_in_idx = template["bus_in_idx"]
        bus_drop_idx = template["bus_drop_idx"]
        bus_in_slot = template["bus_in_slot"]
        bus_drop_slot = template["bus_drop_slot"]
        pred_in   = (bus_in_slot   - dir_sign) % Nz_site
        pred_drop = (bus_drop_slot - dir_sign) % Nz_site
        bus_in_grid = site_idx(bus_in_idx, pred_in)
        bus_drop_grid = site_idx(bus_drop_idx, pred_drop)
    else:
        i_in, j_in, s_in_slot = template["bus_in"]
        pred_in = (s_in_slot - dir_sign) % Nz_site
        i_d, j_d, s_d_slot = template["bus_drop"]
        pred_d = (s_d_slot - dir_sign) % Nz_site
        bus_in_grid = site_idx(i_in, j_in, pred_in)
        bus_drop_grid = site_idx(i_d, j_d, pred_d)
    t_ex = np.sqrt(1.0 - kappa_ex ** 2)

    # Reference steady state from direct solver
    if is_aqh:
        E_inf, _, _ = solve_one_aqh(omega, template, beta0_eta_over_pi,
                                       kappa_ex, alpha)
    else:
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
                         color=ACCENT, lw=1.2,
                         coupling_half_len=0.22, bus_bend_r=0.05,
                         tail_len=0.18, bus_gap=0.03):
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




def draw_aqh_schematic(ax, Nx, Ny, in_idx=None, out_idx=None, title="",
                          template=None, E=None, I_max=None,
                          bus_in_slot=None, bus_drop_slot=None,
                          dc_flip=False, Nz_site=None):
    """Render the AQH brick-wall lattice — site rings on (2Nx-1)×(2Ny-1)
    grid, link rings interleaved.

    If `template` and `E` are provided, color rings by intensity.
    Otherwise render an empty schematic with all rings the same neutral
    color (matching the IQH empty-lattice look).

    Nz_site: ring discretization. Read from template if present, else
        from this kwarg, else module default NZ_SITE. Affects which slots
        are used for the bus DC. Must be a multiple of 4.

    dc_flip: if True (or if template['dc_flip'] is True), the bus DC
        tangent direction is mirrored — IN port is on the right rather
        than the left, OUT drop port on the right rather than the left.
        For the empty-schematic call (no template), pass dc_flip=True
        directly.
    """
    sites = aqh_site_positions(Nx, Ny)
    links = aqh_link_positions(Nx, Ny)

    # Read Nz from template if present, else from kwarg, else module default.
    if template is not None:
        Nz_site = template["Nz_site"]
    elif Nz_site is None:
        Nz_site = NZ_SITE
    aqh_slots_local = _aqh_slots(Nz_site)
    sN = aqh_slots_local['N']; sE = aqh_slots_local['E']
    sS = aqh_slots_local['S']; sW = aqh_slots_local['W']

    # Edge case: AQH 1×1 has zero sites and zero links — no lattice exists.
    # Clear the axes and show a placeholder rather than hitting an IndexError.
    if not sites:
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a4560')
        ax.text(0.5, 0.5, f"AQH {Nx}×{Ny} has no rings —\nNx and Ny must both be ≥ 2.",
                 ha='center', va='center', color='#7a8aaa', fontsize=10,
                 transform=ax.transAxes)
        if title:
            ax.set_title(title, color=ACCENT, fontsize=9, pad=4)
        return None

    if in_idx is None or out_idx is None:
        in_idx, out_idx = aqh_default_bus_positions(Nx, Ny)

    # If template given, pull bus slots from it; otherwise pick the same
    # slots that build_template_aqh would (so empty schematic matches the
    # actual TMM bus geometry).
    if template is not None:
        in_idx = template.get("bus_in_idx", in_idx)
        out_idx = template.get("bus_drop_idx", out_idx)
        bus_in_slot = template.get("bus_in_slot", bus_in_slot)
        bus_drop_slot = template.get("bus_drop_slot", bus_drop_slot)

    # Auto-pick bus slots if not specified — use same logic as
    # build_template_aqh: place bus opposite to the unique link neighbour.
    # Guard: empty lattice or out-of-range indices fall through to a safe
    # default (W slot). This protects against calls with stale templates
    # whose bus_in_idx no longer fits the current (Nx, Ny).
    if bus_in_slot is None or bus_drop_slot is None:
        site_lookup = {(c, r): i for i, (c, r, _, _) in enumerate(sites, start=1)}
        link_lookup = {(c, r): j for j, (c, r, _, _) in enumerate(links, start=1)}
        DIRS = {sN: (0, -1), sE: (1, 0), sS: (0, 1), sW: (-1, 0)}
        OPP  = {sN: sS, sS: sN, sE: sW, sW: sE}
        def auto_slot(site_idx_1based):
            if not (1 <= site_idx_1based <= len(sites)):
                return sW   # safe fallback for out-of-range / empty
            c, r = sites[site_idx_1based - 1][0], sites[site_idx_1based - 1][1]
            link_slots = [s for s, (dc, dr) in DIRS.items()
                           if (c + dc, r + dr) in link_lookup]
            if len(link_slots) == 1:
                return OPP[link_slots[0]]
            for s in (sN, sS, sE, sW):
                if s not in link_slots:
                    return s
            return sW
        if bus_in_slot is None:    bus_in_slot   = auto_slot(in_idx)
        if bus_drop_slot is None:  bus_drop_slot = auto_slot(out_idx)

    ax.clear()
    ax.set_facecolor(PANEL_BG)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#3a4560')

    # Same ring geometry parameters as IQH plot_field_distribution
    half_side = 0.24
    corner_r  = 0.07
    lw_ring   = 2.0

    # Build adjacency for bond-line drawing: each link ring has 4 site
    # neighbours (its 4 cardinal grid neighbours).
    site_xy_lookup = {(c, r): (sx, sy) for (c, r, sx, sy) in sites}

    # ── Bond lines connecting links to their site neighbours (matches IQH) ──
    bond_color = '#1e2a40'
    for (lc, lr, lx, ly) in links:
        for (dc, dr) in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            sxy = site_xy_lookup.get((lc + dc, lr + dr))
            if sxy is not None:
                ax.plot([lx, sxy[0]], [ly, sxy[1]],
                         color=bond_color, lw=0.5, zorder=1)

    # ── Compute intensity grids (zero if no field given) ─────────────────
    if template is not None and E is not None:
        site_idx = template["site_idx"]
        link_idx = template["link_idx"]
        Nz_site = template["Nz_site"]
        Nz_link = template["Nz_link"]
        site_grids = {}
        for i in range(1, len(sites) + 1):
            idxs = [site_idx(i, k) for k in range(Nz_site)]
            site_grids[i] = np.abs(E[idxs])**2
        link_grids = {}
        for j in range(1, len(links) + 1):
            idxs = [link_idx(j, k) for k in range(Nz_link)]
            link_grids[j] = np.abs(E[idxs])**2
        if I_max is None:
            all_vals = np.concatenate(list(site_grids.values())
                                        + list(link_grids.values()))
            I_max = max(float(np.max(all_vals)), 1e-30)
    else:
        # No field — draw with all-zero intensities (matches IQH empty schematic).
        # Pick Nz_site = NZ_SITE, Nz_link = NZ_LINK as defaults.
        site_grids = {i: np.zeros(NZ_SITE) for i in range(1, len(sites) + 1)}
        link_grids = {j: np.zeros(NZ_LINK) for j in range(1, len(links) + 1)}
        if I_max is None:
            I_max = 1.0

    # ── Draw rings via the same _draw_ring helper as IQH ─────────────────
    # AQH slot conventions: slot 0 = SLOT_AQH_N. With the AQH plot's
    # ax.invert_yaxis(), slot 0 is "visually north" = top of the ring.
    #
    # Site rings come in two flavours:
    #   - V-sites: link neighbors at N and S (vertical chain). Light
    #     hops top↔bottom. Native bright arc spans the visual top↔bottom
    #     halves.
    #   - H-sites: link neighbors at E and W (horizontal chain). Light
    #     hops left↔right. Their natural "top/bottom" of the field
    #     pattern is rotated 90° from V-sites.
    # Edge sites (only one link neighbor) inherit the rotation from
    # whichever neighbor they have: N or S → V-orientation; E or W →
    # H-orientation.
    #
    # phase_offset for sites: V-sites use 0.845 (slot 0 → visually-top).
    # H-sites use 0.845 + 0.25 mod 1 = 0.095, rotating slot 0 by 90°.
    # With chirality=-1 for sites, slot index advances CCW visually.
    #
    # Links: chirality=+1 (opposite tangent rule).
    link_lookup = {(l[0], l[1]): True for l in links}

    def _classify_site(c, r):
        """Return 'V' (N/S link neighbor) or 'H' (E/W link neighbor).
        For corner/edge sites with one neighbor, it picks based on that
        neighbor's direction."""
        n_v = int((c, r-1) in link_lookup) + int((c, r+1) in link_lookup)
        n_h = int((c-1, r) in link_lookup) + int((c+1, r) in link_lookup)
        return 'V' if n_v >= n_h else 'H'

    site_phase_V = 0.845
    site_phase_H = 0.345     # slot 8 (E) at visual-right; slot 0 (N) at visual-left
    link_phase = 0.845
    for i, (c, r, sx, sy) in enumerate(sites, start=1):
        kind = _classify_site(c, r)
        ph = site_phase_V if kind == 'V' else site_phase_H
        _draw_ring(ax, (sx, sy), half_side, site_grids[i], I_max,
                    lw=lw_ring, zorder=3, phase_offset=ph,
                    chirality=-1, corner_radius=corner_r)
    for j, (c, r, lx, ly) in enumerate(links, start=1):
        _draw_ring(ax, (lx, ly), half_side, link_grids[j], I_max,
                    lw=lw_ring, zorder=2, phase_offset=link_phase,
                    chirality=+1, corner_radius=corner_r)

    # ── Bus markers (IN at lower-left, OUT at upper-left) ─────────────────
    # NOTE: ax.invert_yaxis() is in effect, so larger plot_y is visually
    # lower. _draw_horseshoe_bus(side="upper") adds +y to the coupling y,
    # which under inversion draws *visually below* the ring. To get a bus
    # visually ABOVE the ring (e.g. for OUT at the top of the lattice with
    # slot N), we need to call side="lower" so the function subtracts y,
    # putting the horseshoe at smaller plot_y (visually above under inversion).
    slot_to_side = {sN: "lower",   # visually above the ring
                     sS: "upper",   # visually below the ring
                     sE: "upper",   # fallback for E/W (rare)
                     sW: "lower"}

    def draw_bus_for_site(site_idx_1based, slot, label_in_arrow, label_out_arrow,
                            text_label, **kwargs):
        if not (1 <= site_idx_1based <= len(sites)):
            return
        _, _, sx, sy = sites[site_idx_1based - 1]
        side = slot_to_side[slot]
        ax.text(sx, sy, text_label, ha="center", va="center",
                 color="white", fontsize=6, fontweight="bold", zorder=5)
        _draw_horseshoe_bus(ax, (sx, sy), side, half_side,
                             label_in_arrow, label_out_arrow,
                             **kwargs)

    # Bus arrow direction tracks the DC tangent direction. Pull dc_flip
    # from template if present, otherwise use the dc_flip arg (used by
    # the empty-schematic preview).
    flipped = template.get("dc_flip", False) if template is not None else dc_flip
    if not flipped:
        in_left, in_right = "input", "through"
        in_kwargs   = dict(left_arrow_in=True, right_arrow_out=True)
        drop_left, drop_right = "drop", "add"
        drop_kwargs = dict(left_arrow_out=True)
    else:
        in_left, in_right = "through", "input"
        in_kwargs   = dict(right_arrow_in=True, left_arrow_out=True)
        drop_left, drop_right = "add", "drop"
        drop_kwargs = dict(right_arrow_out=True)

    if 1 <= in_idx <= len(sites):
        draw_bus_for_site(in_idx, bus_in_slot, in_left, in_right, "IN",
                            **in_kwargs)
    if 1 <= out_idx <= len(sites) and out_idx != in_idx:
        draw_bus_for_site(out_idx, bus_drop_slot, drop_left, drop_right, "OUT",
                            **drop_kwargs)

    # Axis limits — match IQH plot_field_distribution exactly. The bus
    # horseshoes and port labels extend slightly beyond pad_y so they sit
    # *outside* the panel border (matches IQH look — labels don't overlap
    # the bus DC straight section).
    pad_x = 0.4
    pad_y = 0.45
    all_x = [s[2] for s in sites] + [l[2] for l in links]
    all_y = [s[3] for s in sites] + [l[3] for l in links]
    if all_x and all_y:
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
        ax.invert_yaxis()    # match IQH/SMA convention: row 0 at top

    if title:
        ax.set_title(title, color=ACCENT, fontsize=9, pad=4)

    if template is not None and E is not None:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,
                                      norm=plt.Normalize(vmin=0, vmax=I_max))
        return sm
    return None


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

    # bus_in / bus_drop are 3-tuples (ix, iy, slot). Slot 0 = bottom; slot Nz/2 = top.
    Nz_site_t = template["Nz_site"]
    s_top_t = _iqh_slots(Nz_site_t)['TOP']
    bus_in_xy = (bus_in[0], bus_in[1])
    bus_drop_xy = (bus_drop[0], bus_drop[1])
    bus_in_side = "upper" if bus_in[2] == s_top_t else "lower"
    bus_drop_side = "upper" if bus_drop[2] == s_top_t else "lower"

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
        # Slot 0 = SLOT_BUS = south of ring (below center). The perimeter
        # parametrization starts at right-bottom corner (s_frac=0) and goes
        # CCW; bottom-left corner is at s_frac=0.75. Setting phase_offset
        # = 0.75 with chirality=+1 puts slot 0 there — close to the south
        # tangent point where the bus DC physically attaches. This is the
        # correct convention for ALL sites (IN, OUT, and bulk), since
        # build_template uses SLOT_BUS=0 = south for all sites' slot 0.
        # Earlier code used 0.25 for non-IN sites, which placed slot 0 at
        # the top-right — a half-perimeter rotation that made the chiral
        # edge-mode meander appear inverted on bottom/right edges.
        ph = 0.75
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
    # Bus arrow direction tracks the DC tangent direction (template["dc_flip"]).
    # Default (dc_flip=False): light enters IN bus from the left, exits right;
    # OUT bus drops light to the left. Flipping mirrors all bus tangents:
    # input port moves to the right, through to the left, drop to the right.
    flipped = template.get("dc_flip", False)
    if not flipped:
        in_left, in_right = "input", "through"
        in_kwargs   = dict(left_arrow_in=True, right_arrow_out=True)
        drop_left, drop_right = "drop", "add"
        drop_kwargs = dict(left_arrow_out=True)
    else:
        in_left, in_right = "through", "input"
        in_kwargs   = dict(right_arrow_in=True, left_arrow_out=True)
        drop_left, drop_right = "add", "drop"
        drop_kwargs = dict(right_arrow_out=True)
    _draw_horseshoe_bus(ax, in_pos, bus_in_side, half_side,
                         in_left, in_right, **in_kwargs)
    _draw_horseshoe_bus(ax, out_pos, bus_drop_side, half_side,
                         drop_left, drop_right, **drop_kwargs)

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
            lat_type = p.get("lattice_type", "IQH")
            if lat_type == "AQH":
                # AQH: bus_in_idx and bus_drop_idx are 1-based site indices
                template = build_template_aqh(
                    p["Nx"], p["Ny"], p["phi0"],
                    p["kappa_ex"], p["kappa_J"],
                    bus_in_idx=p.get("bus_in_idx"),
                    bus_drop_idx=p.get("bus_drop_idx"),
                    Nz_site=p.get("Nz_site", NZ_SITE),
                    Nz_link=p.get("Nz_link", NZ_LINK),
                    removed_sites=p.get("removed_sites"),
                    removed_links=p.get("removed_links"),
                    dc_flip=p.get("dc_flip", False),
                )
                solver = solve_one_aqh
            else:
                Nz_site = p.get("Nz_site", NZ_SITE)
                Nz_link = p.get("Nz_link", NZ_LINK)
                bus_in, bus_drop = default_bus_positions(p["Nx"], p["Ny"], Nz_site=Nz_site)
                template = build_template(
                    p["Nx"], p["Ny"], p["Phi0"], p["kappa_ex"], p["kappa_J"],
                    bus_in=bus_in, bus_drop=bus_drop,
                    Nz_site=Nz_site, Nz_link=Nz_link,
                    removed_sites=p.get("removed_sites"),
                    removed_links=p.get("removed_links"),
                    dc_flip=p.get("dc_flip", False),
                )
                solver = solve_one
            omegas = p["omegas"]
            N = len(omegas)
            Td = np.zeros(N); Tt = np.zeros(N)
            update_every = max(1, N // 100)
            for i, w in enumerate(omegas):
                if self._abort:
                    return
                _, sd, st = solver(w, template, p["beta0_eta_over_pi"],
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
        self._is_aqh = (self.template.get("lattice_type") == "AQH")

        self._build_ui()

    def _render_field(self, ax, E, title, I_max):
        """Lattice-type-aware field renderer. Returns the ScalarMappable
        for colorbar setup. Used by live preview, MP4 export, and GIF export.
        """
        if self._is_aqh:
            in_idx = self.template["bus_in_idx"]
            out_idx = self.template["bus_drop_idx"]
            return draw_aqh_schematic(
                ax, self.Nx, self.Ny, in_idx=in_idx, out_idx=out_idx,
                title=title, template=self.template, E=E, I_max=I_max)
        else:
            return plot_field_distribution(
                ax, E, self.Nx, self.Ny, self.template,
                title=title, I_max=I_max)

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
        is_aqh = self.template.get("lattice_type") == "AQH"
        dir_sign = -1 if self.template.get("dc_flip", False) else +1
        if is_aqh:
            bus_drop_idx = self.template["bus_drop_idx"]
            bus_drop_slot = self.template["bus_drop_slot"]
            pred_d = (bus_drop_slot - dir_sign) % Nz_site
            drop_state_idx = site_idx_fn(bus_drop_idx, pred_d)
        else:
            i_d, j_d, s_d_slot = self.template["bus_drop"]
            pred_d = (s_d_slot - dir_sign) % Nz_site
            drop_state_idx = site_idx_fn(i_d, j_d, pred_d)
        p_factor = np.exp(1j * self.omega * (1.0 / Nz_site)
                            - self.alpha * (1.0 / Nz_site) / 2)
        s_drop_ss = 1j * self.kappa_ex * p_factor * E_inf[drop_state_idx]
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
        sm = self._render_field(
            self.ax_lat, E,
            title=f'round-trip n = {n_rt:.1f}',
            I_max=self.I_max_ss)
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
        sm0 = self._render_field(
            ax_lat, self.E_hist[0],
            title=f'round-trip n = {self.steps[0]/Nz_site:.1f}',
            I_max=self.I_max_ss)
        cbar = fig.colorbar(sm0, cax=cax,
                              label=r'$|E|^2 / \max|E_\infty|^2$')
        cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
        cbar.ax.yaxis.label.set_color(TEXT_COL)
        cbar.outline.set_edgecolor('#3a4560')

        def update(i):
            ax_lat.clear()
            self._render_field(
                ax_lat, self.E_hist[i],
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
        sm0 = self._render_field(
            ax_lat, self.E_hist[sel[0]],
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
                self._render_field(
                    ax_lat, self.E_hist[i],
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
        self.setWindowTitle("Lida Xu's IQH/AQH-Lattice TMM Explorer — v1.3")
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
        t = QLabel('IQH/AQH-LATTICE TMM EXPLORER')
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
        dl = QGridLayout(gd); dl.setSpacing(4)
        dl.setColumnStretch(0, 0); dl.setColumnStretch(1, 1)
        dl.setColumnStretch(2, 0); dl.setColumnStretch(3, 1)

        # Row 0: lattice type + bus input direction
        # Lattice type — IQH (Hafezi-style) or AQH (zigzag/brick-wall with
        # central link rings per diamond plaquette). Both run full TMM
        # physics; AQH uses 4-DC link rings.
        self.cmb_lat_type = QComboBox()
        self.cmb_lat_type.addItems(['IQH', 'AQH'])
        self.cmb_lat_type.currentIndexChanged.connect(self._on_lat_type_change)
        dl.addWidget(QLabel('Type'),     0, 0)
        dl.addWidget(self.cmb_lat_type,  0, 1)

        # Bus input direction. Toggling reverses the slot ordering across
        # every ring (per-ring chirality flip), AND reverses the bus
        # waveguide arrows. For multi-ring lattices this also flips the
        # topological invariant — see THEORY.md §15.
        self.cmb_dcflip = QComboBox()
        self.cmb_dcflip.addItems(['Left', 'Right'])
        self.cmb_dcflip.setToolTip(
            'Bus DC tangent direction (which end of the bus is the input).\n'
            'Flipping reverses propagation around every ring and the chiral\n'
            'edge mode direction. The bus waveguide arrows update accordingly.')
        self.cmb_dcflip.currentIndexChanged.connect(self._on_dcflip_change)
        dl.addWidget(QLabel('Input'),    0, 2)
        dl.addWidget(self.cmb_dcflip,    0, 3)

        # Row 1: Nx, Ny side-by-side (label above each spinbox)
        self.spn_nx = QSpinBox(); self.spn_nx.setRange(1, 12); self.spn_nx.setValue(4)
        self.spn_ny = QSpinBox(); self.spn_ny.setRange(1, 12); self.spn_ny.setValue(4)
        self.spn_nx.valueChanged.connect(self._on_size_change)
        self.spn_ny.valueChanged.connect(self._on_size_change)
        dl.addWidget(QLabel('Nx'),       1, 0)
        dl.addWidget(self.spn_nx,        1, 1)
        dl.addWidget(QLabel('Ny'),       1, 2)
        dl.addWidget(self.spn_ny,        1, 3)

        # Row 2: bus position info
        self.bus_lbl = QLabel()
        self.bus_lbl.setStyleSheet('color:#7a8aaa;font-size:10px;')
        dl.addWidget(self.bus_lbl,       2, 0, 1, 4)
        self._update_bus_label()

        # Row 3: β₀η on the left, hint on the right
        # β₀η/π — the single physically meaningful link-vs-site phase knob.
        # 1.0 = full anti-resonance (IQH); 0.0 = link & site degenerate;
        # periodic mod 2 in principle, but the full 0-4 range is exposed
        # for exploring band-structure dependence.
        self.spn_beta0eta = QDoubleSpinBox()
        self.spn_beta0eta.setRange(0.0, 4.0)
        self.spn_beta0eta.setSingleStep(0.05)
        self.spn_beta0eta.setDecimals(3)
        self.spn_beta0eta.setValue(1.0)
        self.spn_beta0eta.setSuffix(' π')
        self.spn_beta0eta.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(QLabel('β₀η'),      3, 0)
        dl.addWidget(self.spn_beta0eta,  3, 1, 1, 3)

        hint = QLabel('β₀η = π  →  anti-resonance')
        hint.setStyleSheet('color:#7a8aaa;font-size:10px;')
        dl.addWidget(hint,               4, 0, 1, 4)

        # Row 5: α (loss)
        # α — round-trip intensity loss = α·L_site (with L_site=1).
        # Default α=0.01 ↔ ~1.2 GHz intrinsic linewidth at 750 GHz FSR
        # ↔ Q_int ~ 1.6e5.
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.0, 1.0); self.spn_alpha.setSingleStep(0.001)
        self.spn_alpha.setDecimals(5); self.spn_alpha.setValue(0.01)
        self.spn_alpha.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(QLabel('α (loss)'), 5, 0)
        dl.addWidget(self.spn_alpha,     5, 1, 1, 3)

        # Row 6: Nz_site — number of grid points per site ring (and link
        # ring). More points = finer z-discretization (less aliasing of
        # peripheral phase, smoother color-map around each ring), at the
        # cost of a larger sparse matrix. Default 16. Must be a multiple
        # of 16 because IQH slot allocation uses BOTTOM = 5*Nz/8.
        self.spn_nz = QSpinBox()
        self.spn_nz.setRange(16, 64); self.spn_nz.setSingleStep(16)
        self.spn_nz.setValue(NZ_SITE)
        self.spn_nz.setToolTip(
            'Number of z-discretization grid points per ring (site and link).\n'
            'Default 16. Higher Nz = finer spatial resolution but slower solve.\n'
            'Must be a multiple of 16 — the IQH slot allocation uses\n'
            'BOTTOM = 5·Nz/8 which requires this. Allowed: 16, 32, 48, 64.\n'
            'Changing this rebuilds the template and invalidates results.')
        self.spn_nz.valueChanged.connect(self._on_nz_change)
        dl.addWidget(QLabel('Nz/ring'),  6, 0)
        dl.addWidget(self.spn_nz,        6, 1, 1, 3)

        row.addWidget(gd)

        # ── Couplings group ──────────────────────────────────────────────────
        gc = QGroupBox('Couplings'); gc.setFixedWidth(360)
        cl = QGridLayout(gc); cl.setSpacing(3)

        # κ_ex slider/spinbox (range 0.001 - 0.99 mapped to 1-990).
        # Default 0.359 = κ_J / √(2π). The bus extraction rate
        # κ_ex² · FSR matches the tight-binding hopping rate
        # J = κ_J² · FSR / (2π), making the bus coupling and the
        # site-link hopping rates equal. With default κ_J = 0.9,
        # this is the "matched" choice.
        self.sld_kex = QSlider(Qt.Horizontal)
        self.sld_kex.setRange(1, 990); self.sld_kex.setValue(359)
        self.spn_kex = QDoubleSpinBox()
        self.spn_kex.setRange(0.001, 0.99); self.spn_kex.setDecimals(3)
        self.spn_kex.setSingleStep(0.005); self.spn_kex.setValue(0.359)
        self.spn_kex.setFixedWidth(70)
        self.sld_kex.valueChanged.connect(lambda v: self.spn_kex.setValue(v / 1000.))
        self.spn_kex.valueChanged.connect(lambda v: self.sld_kex.setValue(int(round(v * 1000))))
        self.sld_kex.valueChanged.connect(lambda _: self._invalidate())

        cl.addWidget(QLabel('κ_ex (bus↔site)'), 0, 0)
        cl.addWidget(self.sld_kex, 0, 1, 1, 2); cl.addWidget(self.spn_kex, 0, 3)

        # κ_J slider/spinbox.
        # Default 0.9 ↔ κ_J² = 0.81 ↔ J ≈ 96.7 GHz at 750 GHz FSR
        # (using J = κ_J²·FSR/(2π) — see THEORY.md §6 for the factor of 2π).
        # This is a strong-coupling regime that produces clear chiral
        # edge mode features even in small lattices.
        self.sld_kJ = QSlider(Qt.Horizontal)
        self.sld_kJ.setRange(1, 990); self.sld_kJ.setValue(900)
        self.spn_kJ = QDoubleSpinBox()
        self.spn_kJ.setRange(0.001, 0.99); self.spn_kJ.setDecimals(3)
        self.spn_kJ.setSingleStep(0.005); self.spn_kJ.setValue(0.9)
        self.spn_kJ.setFixedWidth(70)
        self.sld_kJ.valueChanged.connect(lambda v: self.spn_kJ.setValue(v / 1000.))
        self.spn_kJ.valueChanged.connect(lambda v: self.sld_kJ.setValue(int(round(v * 1000))))
        self.sld_kJ.valueChanged.connect(lambda _: self._invalidate())

        cl.addWidget(QLabel('κ_J  (site↔link)'), 1, 0)
        cl.addWidget(self.sld_kJ, 1, 1, 1, 2); cl.addWidget(self.spn_kJ, 1, 3)

        # Phi0 slider/spinbox (0..2 in units of pi). Only used by IQH (it's
        # the Landau-gauge plaquette flux). Hidden when AQH is selected —
        # AQH topology comes from connectivity + β₀η, not from Φ₀.
        self.sld_phi = QSlider(Qt.Horizontal)
        self.sld_phi.setRange(0, PHI_TICKS); self.sld_phi.setValue(50)  # 0.5 pi
        self.spn_phi = QDoubleSpinBox()
        self.spn_phi.setRange(0.0, 2.0); self.spn_phi.setDecimals(3)
        self.spn_phi.setSingleStep(0.05); self.spn_phi.setSuffix(' π')
        self.spn_phi.setValue(0.5); self.spn_phi.setFixedWidth(70)
        self.sld_phi.valueChanged.connect(lambda v: self.spn_phi.setValue(v / 100.))
        self.spn_phi.valueChanged.connect(lambda v: self.sld_phi.setValue(int(round(v * 100))))
        self.sld_phi.valueChanged.connect(lambda _: self._invalidate())

        self.lbl_phi = QLabel('Φ₀')
        cl.addWidget(self.lbl_phi, 2, 0)
        cl.addWidget(self.sld_phi, 2, 1, 1, 2); cl.addWidget(self.spn_phi, 2, 3)
        # Group widgets for show/hide toggling
        self._phi_row_widgets = (self.lbl_phi, self.sld_phi, self.spn_phi)

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
    def _is_aqh(self):
        """Whether the user has selected AQH lattice type."""
        return hasattr(self, 'cmb_lat_type') and self.cmb_lat_type.currentIndex() == 1

    def _dc_flip(self):
        """DC tangent direction toggle. Returns True when the user has
        selected the flipped configuration (link rings mirror-flipped).
        """
        return hasattr(self, 'cmb_dcflip') and self.cmb_dcflip.currentIndex() == 1

    def _nz_site(self):
        """Number of grid points per ring. Read from the Nz/ring spinbox;
        falls back to the module default before the spinbox is created.
        """
        if not hasattr(self, 'spn_nz'):
            return NZ_SITE
        return int(self.spn_nz.value())

    def _on_nz_change(self, _):
        """User changed Nz/ring. This rebuilds every template and the
        sparse matrix sizes change, so any cached spectrum/field is no
        longer valid. Invalidate everything and redraw the schematic.
        """
        self.btn_save.setEnabled(False)
        self.btn_tevol.setEnabled(False)
        self.state['E_at_peak'] = None
        if hasattr(self, '_scan_worker') and self._scan_worker is not None:
            try:
                if self._scan_worker.isRunning():
                    self._scan_worker.request_abort()
            except Exception:
                pass
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

    def _on_dcflip_change(self, _):
        """User toggled the DC tangent direction. This is a real physics
        change — flips the topology — so any cached spectrum/field is
        no longer valid, and we redraw the schematic.
        """
        self.btn_save.setEnabled(False)
        self.btn_tevol.setEnabled(False)
        self.state['E_at_peak'] = None
        # Stop any ongoing scan
        if hasattr(self, '_scan_worker') and self._scan_worker is not None:
            try:
                if self._scan_worker.isRunning():
                    self._scan_worker.request_abort()
            except Exception:
                pass
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

    def _update_bus_label(self):
        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        if self._is_aqh():
            n_sites = aqh_site_count(Nx, Ny)
            n_links = aqh_link_count(Nx, Ny)
            in_idx, out_idx = aqh_default_bus_positions(Nx, Ny)
            self.bus_lbl.setText(
                f'AQH: {n_sites} sites, {n_links} link rings   '
                f'IN=site {in_idx}  OUT=site {out_idx}')
            return
        bus_in, bus_drop = default_bus_positions(Nx, Ny, Nz_site=self._nz_site())
        ix_i, iy_i, sl_i = bus_in
        ix_d, iy_d, sl_d = bus_drop
        s_top_now = _iqh_slots(self._nz_site())['TOP']
        side_i = 'top' if sl_i == s_top_now else 'bot'
        side_d = 'top' if sl_d == s_top_now else 'bot'
        self.bus_lbl.setText(
            f'IN @ ({ix_i},{iy_i}) {side_i}   '
            f'OUT @ ({ix_d},{iy_d}) {side_d}')

    def _on_lat_type_change(self, _):
        """User toggled IQH/AQH.

        Φ₀ is the IQH Landau-gauge plaquette flux — meaningful only for
        IQH. In AQH mode we hide the Φ₀ slider/spinbox/label since
        topology comes from connectivity + β₀η instead.
        """
        is_aqh = self._is_aqh()
        # Show / hide the Φ₀ row depending on lattice type
        for w in self._phi_row_widgets:
            w.setVisible(not is_aqh)
        # Make sure compute / time-evolution are enabled (both lattice types
        # support full physics).
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_tevol.setEnabled(False)
        self._update_bus_label()
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

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

        if self._is_aqh():
            in_idx, out_idx = aqh_default_bus_positions(Nx, Ny)
            n_sites = aqh_site_count(Nx, Ny)
            n_links = aqh_link_count(Nx, Ny)
            draw_aqh_schematic(
                self.ax_lat, Nx, Ny, in_idx=in_idx, out_idx=out_idx,
                title=f'AQH {Nx}×{Ny} — {n_sites} sites · {n_links} link rings',
                dc_flip=self._dc_flip(),
                Nz_site=self._nz_site())
            # Hide the colorbar in empty schematic (no field to colormap)
            try:
                self.cax_lat.clear()
                self.cax_lat.set_axis_off()
            except Exception:
                pass
            self.canvas_lat.draw_idle()
            return

        # IQH path: build template and draw via plot_field_distribution.
        nz = self._nz_site()
        bus_in, bus_drop = default_bus_positions(Nx, Ny, Nz_site=nz)
        template = build_template(
            Nx, Ny, 0.0, 0.10, 0.561,
            bus_in=bus_in, bus_drop=bus_drop,
            Nz_site=nz, Nz_link=nz,
            removed_sites=self.state.get('removed_sites', set()),
            removed_links=self.state.get('removed_links', set()),
            dc_flip=self._dc_flip(),
        )
        E_zero = np.zeros(template['state_size'], dtype=complex)
        self.ax_lat.clear()
        # Re-enable colorbar axis (was hidden during AQH preview)
        self.cax_lat.set_axis_on()
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
        is_aqh = self._is_aqh()
        dc_flip = self._dc_flip()
        nz = self._nz_site()
        if is_aqh:
            in_idx, out_idx = aqh_default_bus_positions(Nx, Ny)
            params = dict(
                lattice_type='AQH', Nx=Nx, Ny=Ny, phi0=0.0,
                kappa_ex=kappa_ex, kappa_J=kappa_J,
                beta0_eta_over_pi=beta0_eta_over_pi,
                alpha=alpha, omegas=omegas,
                bus_in_idx=in_idx, bus_drop_idx=out_idx,
                Nz_site=nz, Nz_link=nz,
                # AQH ring-removal not yet implemented in UI; pass empties
                removed_sites=set(), removed_links=set(),
                dc_flip=dc_flip,
            )
        else:
            params = dict(
                lattice_type='IQH', Nx=Nx, Ny=Ny, Phi0=Phi0,
                kappa_ex=kappa_ex, kappa_J=kappa_J,
                beta0_eta_over_pi=beta0_eta_over_pi,
                alpha=alpha, omegas=omegas,
                Nz_site=nz, Nz_link=nz,
                removed_sites=set(self.state.get('removed_sites', set())),
                removed_links=set(self.state.get('removed_links', set())),
                dc_flip=dc_flip,
            )

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

        template = self.state['template']
        is_aqh = template.get("lattice_type") == "AQH"

        if is_aqh:
            E, _, _ = solve_one_aqh(omega, template,
                                      beta0_eta_over_pi, kappa_ex, alpha)
        else:
            E, _, _ = solve_one(omega, template,
                                  beta0_eta_over_pi, kappa_ex, alpha)
        self.state['E_at_peak'] = E
        self.state['last_omega'] = omega
        self.btn_tevol.setEnabled(True)

        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        self.ax_lat.clear()
        if is_aqh:
            in_idx = template['bus_in_idx']; out_idx = template['bus_drop_idx']
            sm = draw_aqh_schematic(
                self.ax_lat, Nx, Ny, in_idx=in_idx, out_idx=out_idx,
                title=f"ω/(2π) = {omega/(2*np.pi):+.5f}",
                template=template, E=E)
        else:
            sm = plot_field_distribution(
                self.ax_lat, E, Nx, Ny, template,
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
        if self._is_aqh():
            return  # AQH ring removal not yet implemented in click handler
        if event.inaxes is not self.ax_lat:
            return
        if event.xdata is None or event.ydata is None:
            return
        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        bus_in, bus_drop = default_bus_positions(Nx, Ny, Nz_site=self._nz_site())
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
        # Keep a reference so the dialog isn't garbage-collected when this
        # method returns. When it's closed, re-sync the main window's
        # button states (the user can re-open it as long as a peak is
        # selected and the spectrum is still valid).
        self._tevol_dlg = dlg
        dlg.finished.connect(self._on_tevol_dialog_closed)
        # Non-modal — lets the user keep working with the main window
        dlg.show()

    def _on_tevol_dialog_closed(self, _result=None):
        """Restore Time Evol. button enabled state after the dialog closes,
        if the prerequisites (template + selected ω) are still valid.
        """
        if (self.state.get('template') is not None
                and self.state.get('last_omega') is not None):
            self.btn_tevol.setEnabled(True)

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