# Hafezi-Lattice TMM Simulation

A self-contained Python implementation of the $z$-discretized transfer-matrix method (TMM)
for a Hafezi-style coupled-resonator lattice with synthetic magnetic flux. Reproduces the
chiral edge mode at $\Phi_0 = \pi/2$ on a 4×4 lattice in seconds.

## Files

- **`TMM_app.py`** — interactive PyQt5 GUI for exploring the lattice. Adjust Nx, Ny, flux,
  couplings on the left panel; spectrum and field distribution update on the right. Recommended
  way to use the simulation.
- `lattice_NxN_TMM.py` — the same simulation as a non-GUI script. Run as
  `python lattice_NxN_TMM.py` to reproduce the default 4×4 demo. Useful if you want to script
  scans or batch jobs.
- `THEORY.md` — full theory and implementation guide. Read this to understand the physics
  of anti-resonant link rings, Peierls phase, and the TMM linear system.
- `figures/` — schematic figures used in `THEORY.md`. SVG (editable) and PNG (rendered).

## Requirements

For the GUI app:
```
numpy
matplotlib
scipy
PyQt5
```

For the script-only version: just `numpy`, `matplotlib`, `scipy`.

## Quick start — interactive app

```bash
pip install numpy matplotlib scipy PyQt5
python TMM_app.py
```

The app opens with default values (4×4 lattice, Phi0 = pi/2). Click **Compute spectrum**
and after ~7 seconds you'll see all 16 site supermodes. The peak dropdown lets you pick
which one to render — try the peak nearest omega/(2pi) = 0.025 for the chiral edge mode.

### Bus convention (fixed)

Input bus is always at site **(0, 0)** (bottom-left), drop bus at site **(0, Ny-1)** (top-left).
Both buses sit on the left edge column, mirroring the standard Hafezi probe geometry.

### Parameters you can adjust

- **Lattice**: Nx, Ny (any 2-12 in each direction; computation scales as ~N^3).
- **Flux**: Phi0 in units of pi (try 0.5 for Hofstadter, 0.0 for trivial, 1.0 for half-flux).
- **Bus coupling kappa_ex**: sets resonance linewidth. 0.05-0.2 is the useful range.
- **Hopping coupling kappa_J**: sets effective hopping rate J. 0.1-0.7 is the useful range.
  kappa_J = 0.561 gives J/FSR approx 1/40 matching real Hafezi devices.
- **eta**: link-ring extra length (default 0.5; matters in combination with the next box).
- **beta0*eta = pi anti-resonance**: keep checked for normal Hafezi physics. Uncheck to
  disable anti-resonance (interesting only for sanity-checking).
- **alpha**: linear loss per unit length. 1e-4 is essentially lossless; 1e-2 noticeably damps.
- **Frequency window**: omega/(2pi) range (default [-0.1, +0.1], the central FSR cluster)
  and number of points (more points = sharper resolution but slower).

## Quick start — non-GUI script

```bash
pip install numpy matplotlib scipy
python lattice_NxN_TMM.py
```

Saves `lattice_NxN_TMM_demo.png` in the current directory. Edit the bottom of the file to
change parameters.

## Programmatic use

Both scripts expose the same core functions:

```python
import numpy as np
from lattice_NxN_TMM import scan_spectrum_fast, solve_lattice_fast

omegas = np.linspace(-0.1*2*np.pi, 0.1*2*np.pi, 4001)
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
```

## License

CC0 / public domain — use freely.
