"""
TMM_app.py
==========
Lida Xu's Hafezi-Lattice TMM Explorer — v1.0
Standalone desktop application matching the Linear.py UI conventions.

Run:   python TMM_app.py
Build: pyinstaller --onefile --windowed TMM_app.py

Pure z-discretized TMM throughout — no Hamiltonian, no J in the simulation
itself; only field amplitudes E(z_k) on every grid point and the round-trip
operator built from local propagation factors plus 2x2 DC scattering matrices.
Sparse-LU solve gives ~30x speedup over dense.

Bus convention (fixed):  IN at site (0, 0),  OUT at site (0, Ny - 1).
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
    QCheckBox,
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
                    bus_in, bus_drop, Nz_site=NZ_SITE, Nz_link=NZ_LINK):
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

    entries = []; src_rows = []; src_vals = []
    for (ix, iy), slots in site_neighbors.items():
        for k in range(Nz_site):
            k_prev = (k - 1) % Nz_site
            row = site_idx(ix, iy, k)
            if k in slots:
                action = slots[k]; kind = action[0]
                if kind == "bus_in":
                    entries.append((row, site_idx(ix, iy, k_prev), "p_site", -t_ex))
                    src_rows.append(row); src_vals.append(1j * kappa_ex)
                elif kind == "bus_drop":
                    entries.append((row, site_idx(ix, iy, k_prev), "p_site", -t_ex))
                elif kind == "link":
                    _, link_name, end = action
                    if end == "near":
                        link_k_dc = 0; link_k_prev = Nz_link - 1
                    else:
                        link_k_dc = half_link; link_k_prev = half_link - 1
                    extra_phase_dc = extras[link_name][link_k_dc]
                    entries.append((row, site_idx(ix, iy, k_prev), "p_site", -t_J))
                    entries.append((row, link_idx(link_name, link_k_prev),
                                     "p_link_extra", -1j * kappa_J, extra_phase_dc))
            else:
                entries.append((row, site_idx(ix, iy, k_prev), "p_site", -1.0))

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
    )


def solve_one(omega, template, eta, half_fsr_offset, kappa_ex, alpha):
    L_site = 1.0
    L_link = L_site + eta
    Nz_site = template["Nz_site"]; Nz_link = template["Nz_link"]
    dz_site = L_site / Nz_site; dz_link = L_link / Nz_link
    p_site = np.exp(1j * omega * dz_site - alpha * dz_site / 2.0)
    beta0_eta = np.pi if half_fsr_offset else 0.0
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
    i_in, j_in = template["bus_in"]
    e_in_at_bus_in = p_site * E[site_idx(i_in, j_in, Nz_site - 1)]
    s_thru = t_ex * 1.0 + 1j * kappa_ex * e_in_at_bus_in
    i_d, j_d = template["bus_drop"]
    e_in_at_bus_drop = p_site * E[site_idx(i_d, j_d, Nz_site - 1)]
    s_drop = 1j * kappa_ex * e_in_at_bus_drop

    return E, s_drop, s_thru


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
    bus_gap = 0.04
    coupling_half_len = 0.18
    bus_bend_r = 0.07
    tail_len = 0.30
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

    half_side = 0.16
    corner_r = 0.05
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

    for (ix, iy), grid_I in site_grids.items():
        ph = 0.75 if (ix, iy) == bus_in else 0.25
        _draw_ring(ax, site_pos[(ix, iy)], half_side, grid_I,
                    I_max, lw=lw_ring, zorder=3,
                    phase_offset=ph, chirality=+1, corner_radius=corner_r)

    for name, grid_I in link_grids.items():
        kind = name.split("_")[0]
        ph = 0.75 if kind == "V" else 0.5
        _draw_ring(ax, link_pos[name], half_side, grid_I,
                    I_max, lw=lw_ring, zorder=2,
                    phase_offset=ph, chirality=-1, corner_radius=corner_r)

    in_pos = site_pos[bus_in]
    out_pos = site_pos[bus_drop]
    _draw_horseshoe_bus(ax, in_pos, "lower", half_side,
                         "input", "through",
                         left_arrow_in=True, right_arrow_out=True)
    _draw_horseshoe_bus(ax, out_pos, "upper", half_side,
                         "drop", "add",
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
            template = build_template(
                p["Nx"], p["Ny"], p["Phi0"], p["kappa_ex"], p["kappa_J"],
                bus_in=(0, 0), bus_drop=(0, p["Ny"] - 1),
            )
            omegas = p["omegas"]
            N = len(omegas)
            Td = np.zeros(N); Tt = np.zeros(N)
            update_every = max(1, N // 100)
            for i, w in enumerate(omegas):
                if self._abort:
                    return
                _, sd, st = solve_one(w, template, p["eta"],
                                       p["half_fsr_offset"], p["kappa_ex"],
                                       p["alpha"])
                Td[i] = abs(sd) ** 2
                Tt[i] = abs(st) ** 2
                if (i + 1) % update_every == 0:
                    self.progress.emit(i + 1, N)
            self.progress.emit(N, N)
            self.finished_ok.emit(omegas, Td, Tt, template)
        except Exception as e:
            self.failed.emit(repr(e))


# ═════════════════════════════════════════════════════════════════════════════
#  Main window
# ═════════════════════════════════════════════════════════════════════════════

PHI_TICKS = 200   # 0..200 ticks -> 0..2 (in units of pi)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lida Xu's Hafezi-Lattice TMM Explorer — v1.0")
        self.setMinimumSize(1400, 800)
        icon_path = os.path.join(_BUNDLE_DIR, 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self._apply_dark_theme()
        self.state = dict(
            Nx=4, Ny=4, Phi0=np.pi / 2,
            kappa_ex=0.10, kappa_J=0.561,
            eta=0.5, half_fsr_offset=True, alpha=1e-4,
            omega_min=-0.1, omega_max=0.1, npts=4001,
            omegas=None, Td=None, Tt=None,
            template=None, peaks_omega=[], selected_peak_idx=-1,
            E_at_peak=None,
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
        t = QLabel('HAFEZI-LATTICE TMM EXPLORER')
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
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.05, 1.10)
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
        self.ax_lat = self.fig_lat.add_subplot(111)
        self.ax_lat.set_facecolor(PANEL_BG)
        self.ax_lat.set_xticks([]); self.ax_lat.set_yticks([])
        for sp in self.ax_lat.spines.values():
            sp.set_edgecolor('#3a4560')
        self.fig_lat.subplots_adjust(left=0.04, right=0.92, top=0.95, bottom=0.04)
        ll.addWidget(self.canvas_lat)
        self._cbar = None

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

        self.spn_nx = QSpinBox(); self.spn_nx.setRange(2, 12); self.spn_nx.setValue(4)
        self.spn_ny = QSpinBox(); self.spn_ny.setRange(2, 12); self.spn_ny.setValue(4)
        # Nx/Ny change geometry, so redraw the schematic (and invalidate spectrum)
        self.spn_nx.valueChanged.connect(self._on_size_change)
        self.spn_ny.valueChanged.connect(self._on_size_change)

        dl.addWidget(QLabel('Nx'), 0, 0); dl.addWidget(self.spn_nx, 1, 0)
        dl.addWidget(QLabel('Ny'), 0, 1); dl.addWidget(self.spn_ny, 1, 1)

        bus_lbl = QLabel('IN @ (0,0)   OUT @ (0,Ny−1)')
        bus_lbl.setStyleSheet('color:#7a8aaa;font-size:10px;')
        dl.addWidget(bus_lbl, 2, 0, 1, 2)

        # eta + anti-resonance toggle
        self.spn_eta = QDoubleSpinBox()
        self.spn_eta.setRange(0.0, 2.0); self.spn_eta.setSingleStep(0.05)
        self.spn_eta.setDecimals(3); self.spn_eta.setValue(0.5)
        self.spn_eta.setFixedWidth(70)
        self.spn_eta.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(QLabel('η'), 3, 0); dl.addWidget(self.spn_eta, 3, 1)

        self.chk_halffsr = QCheckBox('β₀η = π  (anti-resonance)')
        self.chk_halffsr.setChecked(True)
        self.chk_halffsr.toggled.connect(lambda _: self._invalidate())
        dl.addWidget(self.chk_halffsr, 4, 0, 1, 2)

        # alpha (loss)
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.0, 1.0); self.spn_alpha.setSingleStep(1e-4)
        self.spn_alpha.setDecimals(5); self.spn_alpha.setValue(1e-4)
        self.spn_alpha.setFixedWidth(70)
        self.spn_alpha.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(QLabel('α (loss)'), 5, 0); dl.addWidget(self.spn_alpha, 5, 1)

        row.addWidget(gd)

        # ── Couplings group ──────────────────────────────────────────────────
        gc = QGroupBox('Couplings'); gc.setFixedWidth(360)
        cl = QGridLayout(gc); cl.setSpacing(3)

        # κ_ex slider/spinbox (range 0.001 - 0.99 mapped to 1-990)
        self.sld_kex = QSlider(Qt.Horizontal)
        self.sld_kex.setRange(1, 990); self.sld_kex.setValue(100)
        self.spn_kex = QDoubleSpinBox()
        self.spn_kex.setRange(0.001, 0.99); self.spn_kex.setDecimals(3)
        self.spn_kex.setSingleStep(0.005); self.spn_kex.setValue(0.10)
        self.spn_kex.setFixedWidth(70)
        self.sld_kex.valueChanged.connect(lambda v: self.spn_kex.setValue(v / 1000.))
        self.spn_kex.valueChanged.connect(lambda v: self.sld_kex.setValue(int(round(v * 1000))))
        self.sld_kex.valueChanged.connect(lambda _: self._invalidate())

        cl.addWidget(QLabel('κ_ex (bus↔site)'), 0, 0)
        cl.addWidget(self.sld_kex, 0, 1, 1, 2); cl.addWidget(self.spn_kex, 0, 3)

        # κ_J slider/spinbox
        self.sld_kJ = QSlider(Qt.Horizontal)
        self.sld_kJ.setRange(1, 990); self.sld_kJ.setValue(561)
        self.spn_kJ = QDoubleSpinBox()
        self.spn_kJ.setRange(0.001, 0.99); self.spn_kJ.setDecimals(3)
        self.spn_kJ.setSingleStep(0.005); self.spn_kJ.setValue(0.561)
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
        self.spn_omin.setRange(-0.5, 0.5); self.spn_omin.setSingleStep(0.01)
        self.spn_omin.setDecimals(3); self.spn_omin.setValue(-0.1)
        self.spn_omax = QDoubleSpinBox()
        self.spn_omax.setRange(-0.5, 0.5); self.spn_omax.setSingleStep(0.01)
        self.spn_omax.setDecimals(3); self.spn_omax.setValue(0.1)
        self.spn_npts = QSpinBox()
        self.spn_npts.setRange(101, 32001); self.spn_npts.setSingleStep(500)
        self.spn_npts.setValue(4001)
        for sp in (self.spn_omin, self.spn_omax, self.spn_npts):
            sp.valueChanged.connect(lambda _: self._invalidate())

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

        # Save path row
        self.lbl_save_path = QLabel('Save to:')
        self.edit_save_path = QLineEdit(
            os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
            else os.path.dirname(os.path.abspath(__file__)))
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
        rl.addWidget(self.btn_reset, 3, 3, 1, 2)

        # row 4: save path
        rl.addWidget(self.lbl_save_path,  4, 0)
        rl.addWidget(self.edit_save_path, 4, 1, 1, 4)
        rl.addWidget(self.btn_browse,     4, 5)

        row.addWidget(gr, stretch=1)
        return w

    # ── Invalidation ─────────────────────────────────────────────────────────
    def _on_size_change(self, _):
        """Nx or Ny changed: invalidate AND redraw the lattice schematic."""
        self._invalidate()
        self._invalidate_field_only()
        self.canvas_lat.draw_idle()

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
        # Clear stored results
        self.state['omegas'] = None
        self.state['Td'] = None; self.state['Tt'] = None
        self.state['template'] = None
        self.state['peaks_omega'] = []
        self.state['selected_peak_idx'] = -1
        self.state['E_at_peak'] = None
        # Clear spectrum (always at full FSR x-axis) but leave the field
        # panel showing whatever was last drawn — don't re-render the empty
        # schematic on every keypress.
        self._clear_spectrum_axes()
        self.canvas_spec.draw_idle()
        self.status.showMessage('Parameters changed — click ▶ Compute to update.')

    def _invalidate_field_only(self):
        """Re-render an empty lattice schematic so the structure is shown."""
        Nx = self.spn_nx.value()
        Ny = self.spn_ny.value()
        # Use a dummy field of zeros so rings/buses are drawn at lowest intensity
        template = build_template(
            Nx, Ny, 0.0, 0.10, 0.561,
            bus_in=(0, 0), bus_drop=(0, Ny - 1),
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
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self._cbar = self.fig_lat.colorbar(
            sm, ax=self.ax_lat, fraction=0.04, pad=0.04,
            label=r"$|E|^2$ (norm)")
        self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self._cbar.ax.yaxis.label.set_color(TEXT_COL)
        self._cbar.outline.set_edgecolor('#3a4560')

    def _clear_spectrum_axes(self):
        for ax, col, ttl in [(self.ax_thru, '#4a9eff', 'Thru port'),
                                (self.ax_drop, '#ff4a6e', 'Drop port')]:
            ax.clear()
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=10, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor('#3a4560'); sp.set_linewidth(1.2)
            ax.grid(True, color=GRID_COL, linewidth=0.4)
            # x-axis is ALWAYS the full FSR window so changing sweep range
            # doesn't make the plot "shrink". The sweep-range spinboxes only
            # affect which slice of the FSR gets simulated.
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.05, 1.10)
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
        eta = self.spn_eta.value()
        half_fsr_offset = self.chk_halffsr.isChecked()
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
                       eta=eta, half_fsr_offset=half_fsr_offset,
                       alpha=alpha, omegas=omegas)

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
        if highlight_omega is not None:
            for ax in (self.ax_thru, self.ax_drop):
                ax.axvline(highlight_omega / (2 * np.pi),
                            color=ACCENT, ls=':', alpha=0.7, lw=1)
        self.canvas_spec.draw_idle()

    def _on_peak_changed(self, idx):
        if idx < 0 or self.state['template'] is None:
            return
        peaks = self.state['peaks_omega']
        if idx >= len(peaks):
            return
        omega = peaks[idx]
        eta = self.spn_eta.value()
        half_fsr_offset = self.chk_halffsr.isChecked()
        kappa_ex = self.spn_kex.value()
        alpha = self.spn_alpha.value()

        E, _, _ = solve_one(omega, self.state['template'], eta,
                              half_fsr_offset, kappa_ex, alpha)
        self.state['E_at_peak'] = E
        self.state['selected_peak_idx'] = idx

        Nx = self.spn_nx.value(); Ny = self.spn_ny.value()
        self.ax_lat.clear()
        sm = plot_field_distribution(
            self.ax_lat, E, Nx, Ny, self.state['template'],
            title=f"ω/(2π) = {omega/(2*np.pi):+.5f}",
        )
        self._update_colorbar(sm)
        self.canvas_lat.draw_idle()

        # Highlight peak in spectrum
        self._draw_spectra(highlight_omega=omega)

    def _on_spec_click(self, event):
        """Click on spectrum -> jump to nearest peak."""
        if event.inaxes not in (self.ax_thru, self.ax_drop):
            return
        if not self.state['peaks_omega']:
            return
        x_click = event.xdata * 2 * np.pi
        peaks = self.state['peaks_omega']
        idx = int(np.argmin([abs(w - x_click) for w in peaks]))
        self.cmb_peak.setCurrentIndex(idx)

    def _clear(self):
        self._invalidate()

    def _reset_all(self):
        self.spn_nx.setValue(4)
        self.spn_ny.setValue(4)
        self.spn_phi.setValue(0.5)        # 0.5 pi
        self.spn_kex.setValue(0.10)
        self.spn_kJ.setValue(0.561)
        self.spn_eta.setValue(0.5)
        self.chk_halffsr.setChecked(True)
        self.spn_alpha.setValue(1e-4)
        self.spn_omin.setValue(-0.1)
        self.spn_omax.setValue(0.1)
        self.spn_npts.setValue(4001)
        self._invalidate()

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
