# Hafezi-Lattice TMM Explorer

A self-contained Python application for the $z$-discretized transfer-matrix method
(TMM) applied to Hafezi-style coupled-resonator photonic lattices with synthetic
gauge fields. It simulates both the **IQH** (integer quantum Hall, Hafezi 2011)
lattice with Peierls flux and the **AQH** (anomalous Floquet, Liang–Chong 2013 /
Pasek–Chong 2014 / Mittal–Hafezi 2019) brick-wall lattice — reproducing chiral and
anomalous edge modes on small lattices in seconds.

Everything is in a single file, `TMM_app.py`. The full derivation — from
dimensional Maxwell-equation propagation down to the sparse linear solver and the
field renderer — is included below in
[Part II — Theory and Implementation](#theory-and-implementation).

## Files

- **`TMM_app.py`** — the interactive PyQt5 application. Everything lives here: IQH +
  AQH template builders, steady-state and time-domain solvers, field rendering, and
  MP4/GIF export.
- `TMM_evolution_*.mp4` — example time-evolution animations exported from the app.
- `README.md` — this file (quick start, parameter guide, and the full theory).

## Requirements

```
numpy
matplotlib
scipy
PyQt5
```

```bash
pip install numpy matplotlib scipy PyQt5
```

## Quick start

```bash
python TMM_app.py
```

The app opens on a default 6×6 IQH lattice at $\Phi_0 = \pi/2$. Choose the lattice
type (IQH / AQH) and adjust lattice size, flux, couplings, and loss on the left
panel; the transmission spectrum and field distribution update on the right. Click
**Compute spectrum**, then use the peak dropdown to render a chosen supermode — look
for one in the topological gap to see the chiral edge mode hugging the boundary.

### Default parameters (initial UI values)

| Control | Default | Meaning |
|---|---|---|
| Lattice type | IQH | IQH (Peierls flux) or AQH (brick-wall Floquet) |
| Nx × Ny | 6 × 6 | Lattice size (1–12 each; cost scales as $\sim N^3$) |
| $\Phi_0$ | 0.5 π | IQH synthetic flux per plaquette (hidden in AQH mode) |
| $\kappa_{\rm ex}$ | 0.199 | Bus coupling, $= \kappa_J/\sqrt{2\pi}$ — matched to the lattice mode |
| $\kappa_J$ | 0.5 | Hopping coupling — $J \approx 29.8$ GHz at a 750 GHz FSR |
| $\beta_0\eta/\pi$ | 1.0 | Link anti-resonance (1.0 = ideal Hafezi; sweep to explore) |
| $\alpha$ | 0.01 | Field loss per unit length (~1.2 GHz intrinsic linewidth) |
| $N_z$ | 16 | Grid points per ring (16 / 32 / 48 / 64) |
| $\omega/(2\pi)$ window | [−0.1, 0.1] | Frequency scan range in FSR units, 4001 points |

Bus convention: for IQH the input bus is at the lower-left site and the drop bus at
the upper-left site, both on the left edge column. See [§8](#8-the-iqh-lattice--geometry-and-indexing) (IQH) and [§14](#14-the-aqh-lattice--anomalous-floquet-topology-without-flux) (AQH) for the
exact slot placement, and [§15](#15-per-ring-chirality-flip-dc-tangent-direction) for the `Input: left/right` chirality flip.

## Programmatic use

The solver functions can be imported and driven without opening the GUI — importing
the module does **not** launch Qt (the window only opens when you run
`python TMM_app.py`):

```python
import numpy as np
from TMM_app import build_template, solve_one, default_bus_positions

bus_in, bus_drop = default_bus_positions(Nx=4, Ny=4)
template = build_template(
    Nx=4, Ny=4, Phi0=np.pi / 2,
    kappa_ex=0.199, kappa_J=0.5,
    bus_in=bus_in, bus_drop=bus_drop,
)

# Spectrum scan over the central FSR window
omegas = np.linspace(-0.1 * 2 * np.pi, 0.1 * 2 * np.pi, 4001)
T_drop = np.empty_like(omegas)
T_thru = np.empty_like(omegas)
for i, w in enumerate(omegas):
    E, s_drop, s_thru = solve_one(
        w, template, beta0_eta_over_pi=1.0, kappa_ex=0.199, alpha=0.01)
    T_drop[i] = abs(s_drop) ** 2
    T_thru[i] = abs(s_thru) ** 2

# Steady-state field at one detuning (e.g. an edge mode near omega/(2pi) = 0.025)
E, s_drop, s_thru = solve_one(
    0.025 * 2 * np.pi, template,
    beta0_eta_over_pi=1.0, kappa_ex=0.199, alpha=0.01)
```

For the AQH brick-wall lattice use `build_template_aqh` + `solve_one_aqh` (same call
pattern; default bus positions come from `aqh_default_bus_positions`). For transient
dynamics use `time_evolve`. Note that `kappa_ex` is passed to both `build_template`
(to set the bus DC strength) and `solve_one` (to read out the ports) — keep the two
values equal.

## License

CC0 / public domain — use freely.

---

# Theory and Implementation

A complete derivation of the simulator from physical first principles. We start with
dimensional Maxwell-equation propagation in real waveguides, introduce the
normalization scheme that gives the dimensionless quantities used in the code, and
show how the steady-state and time-domain solvers are constructed.

This document assumes familiarity with microring resonators and the basic IQH-lattice
lattice idea. It does not assume familiarity with our simulator.

---

## Table of contents

1. [Setting](#1-setting)
2. [Physical propagation in a single waveguide](#2-physical-propagation-in-a-single-waveguide)
3. [Choice of frame: the carrier wraps to identity](#3-choice-of-frame-the-carrier-wraps-to-identity)
4. [Dimensionless variables](#4-dimensionless-variables)
5. [Single ring — discretization](#5-single-ring--discretization)
6. [Single ring — adding the bus](#6-single-ring--adding-the-bus)
7. [Single ring — steady state vs time evolution](#7-single-ring--steady-state-vs-time-evolution)
8. [The IQH lattice — geometry and indexing](#8-the-iqh-lattice--geometry-and-indexing)
9. [Link rings and anti-resonance](#9-link-rings-and-anti-resonance)
10. [The Peierls phase and synthetic flux](#10-the-peierls-phase-and-synthetic-flux)
11. [Assembling the global sparse system](#11-assembling-the-global-sparse-system)
12. [Solving — sparse LU vs Ikeda iteration](#12-solving--sparse-lu-vs-ikeda-iteration)
13. [Removed rings (defects)](#13-removed-rings-defects)
14. [The AQH lattice — anomalous Floquet topology without flux](#14-the-aqh-lattice--anomalous-floquet-topology-without-flux)
15. [Per-ring chirality flip (DC tangent direction)](#15-per-ring-chirality-flip-dc-tangent-direction)
16. [Time-evolution dialog](#16-time-evolution-dialog)
17. [Field-distribution rendering](#17-field-distribution-rendering)
18. [Mapping back to experimental units](#18-mapping-back-to-experimental-units)
19. [What this simulator does NOT model](#19-what-this-simulator-does-not-model)

---

## 1. Setting

The simulator supports two photonic topological lattice models:

- **IQH** (integer quantum Hall, Hafezi 2011): a 2D array of "site" microring resonators on a square grid, every nearest-neighbor pair coupled through a "link" microring. Synthetic magnetic flux is realized via a Peierls phase pattern on the H-link rings (Landau gauge).
- **AQH** (anomalous quantum Hall, Liang–Chong 2013 / Pasek–Chong 2014): a brick-wall arrangement of site rings with central link rings inside each diamond plaquette. *Zero* synthetic flux per plaquette — instead, anomalous Floquet topology emerges from the cyclic four-DC link rings and the brick-wall geometry.

In both, link rings are designed to be **anti-resonant** with the site rings (their round-trip phase differs by π from the carrier-induced 2πN), so they don't host their own modes in the band of interest, but they mediate nearest-neighbor hopping. In the IQH case they additionally carry a directional Peierls phase. In the AQH case the topology is a property of the geometry alone.

The key quantities in a real device:

| Symbol | Meaning | Typical value | SiN device (telecom) |
|---|---|---|---|
| $\lambda_0$ | Carrier wavelength | — | 1550 nm |
| $L_{\rm site}$ | Site ring physical length | $\sim 100$ μm | $\sim 190$ μm |
| $\eta = L_{\rm link} - L_{\rm site}$ | Link extra length | $\sim \lambda_0 / (2 n_{\rm eff})$ | $\sim 440$ nm |
| $n_{\rm eff}$ | Effective refractive index | $\sim 2$ | $\sim 1.76$ |
| $n_g$ | Group index | — | $\sim 2.1$ |
| $v_g$ | Group velocity | $c/n_g \sim 10^8$ m/s | $\sim 1.43 \times 10^8$ m/s |
| $\beta(\omega)$ | Propagation constant | $\sim 10^7$ rad/m | $2\pi n_{\rm eff}/\lambda_0 \approx 7.1 \times 10^6$ rad/m |
| $\Gamma_{\rm FSR}$ | Free spectral range $= v_g/L_{\rm site}$ | hundreds of GHz | $\sim 750$ GHz |
| $\kappa_{\rm ex,exp}$ | Bus-cavity linewidth | tens of GHz | tens of GHz |
| $J$ | Tight-binding hopping rate | tens of GHz | tens of GHz |
| $\alpha$ | Intrinsic field-loss coefficient | small | $\sim$ 1–10 GHz linewidth |

The "SiN device" column corresponds to the silicon-nitride coupled-ring platform used in
the experimental work cited in [§14](#14-the-aqh-lattice--anomalous-floquet-topology-without-flux) — the same platform this simulator is built to
support. The $\eta \approx 440$ nm is the actual device value (set by lithography);
inverting the anti-resonance condition $\beta_0 \eta = \pi$ then fixes
$n_{\rm eff} = \lambda_0/(2 \eta) = 1550/880 \approx 1.76$ at telecom ([§3](#3-choice-of-frame-the-carrier-wraps-to-identity)). This is on
the lower side of the $n_{\rm eff} \approx 2$ rule of thumb because the SiN core is
relatively thin and the mode extends substantially into the cladding. The group index
$n_g \approx 2.1$ is larger than $n_{\rm eff}$ because of normal waveguide dispersion
near 1550 nm.

The simulator works in dimensionless units that absorb the carrier frequency entirely.
We derive these units carefully below.

---

## 2. Physical propagation in a single waveguide

A single-mode optical waveguide carries an envelope $A(z, t)$ around a carrier
frequency $\omega_0$. The total electric field is

$$\mathcal{E}(z, t) = \mathrm{Re}\{A(z, t)   e^{i(\beta_0 z - \omega_0 t)}\}$$

In the slowly-varying-envelope approximation, $A$ obeys (lossy free propagation):

$$\frac{\partial A}{\partial z} + \frac{1}{v_g} \frac{\partial A}{\partial t} = -\frac{\alpha}{2} A$$

where $v_g = (\partial \beta / \partial \omega)^{-1}|_{\omega_0}$ is the group velocity at
the carrier and $\alpha$ is the intensity loss per unit length.

For monochromatic (CW) excitation at detuning $\omega$ from the carrier — i.e.
$A(z, t) = E(z) e^{-i\omega t}$ — this becomes the ordinary differential equation:

$$\frac{dE}{dz} = \left( i\frac{\omega}{v_g} - \frac{\alpha}{2} \right) E$$

with solution

$$E(z) = E(0) \cdot e^{i \omega z / v_g - \alpha z / 2}$$

So in the envelope frame, propagation by distance $z$ multiplies the amplitude by

$$\boxed{\quad p(z) = e^{i \omega z / v_g - \alpha z / 2} \quad}$$

This is the **only physics in the simulator**, repeated everywhere. All the rest is
geometry: where the propagation segments connect to one another, and which DCs sit
between which segments.

Note that the *full* amplitude including the carrier is

$$E_{\rm full}(z) = E(z)   e^{i\beta_0 z} = E(0)   e^{i \beta_0 z}   e^{i\omega z/v_g - \alpha z/2}$$

We will work in the envelope frame, so the $e^{i\beta_0 z}$ part disappears
into the definition of $E$ and only the $e^{i\omega z/v_g}$ part remains. But we still
have to account for $\beta_0 z$ when comparing two waveguides of different lengths
(site vs link), because the *difference* in $\beta_0 \cdot \mathrm{length}$ between
the two doesn't trivially cancel.

---

## 3. Choice of frame: the carrier wraps to identity

The site ring has length $L_{\rm site}$. A photon's full round-trip phase is

$$\phi_{\rm site}^{\rm RT}(\omega_{\rm abs}) = \beta(\omega_{\rm abs})   L_{\rm site}$$

where $\omega_{\rm abs} = \omega_0 + \omega$ is the absolute angular frequency. Expanding:

$$\beta(\omega_{\rm abs}) = \beta_0 + \omega/v_g + O(\omega^2)$$

so

$$\phi_{\rm site}^{\rm RT} = \beta_0 L_{\rm site} + \omega L_{\rm site} / v_g$$

The **carrier is chosen** so that the device sits on a site resonance, i.e.,

$$\boxed{\quad \beta_0 L_{\rm site} = 2\pi N \quad \text{for some integer } N \quad}$$

This is a **physical fact about the device** — the laser wavelength is matched to the
ring's design. With this choice, the static carrier phase wraps cleanly to identity
and disappears from the round-trip:

$$\phi_{\rm site}^{\rm RT}(\omega) = 2\pi N + \omega L_{\rm site} / v_g  \equiv  \omega L_{\rm site} / v_g   (\text{mod } 2\pi)$$

The detuning $\omega$ is now measured *from the resonance*. Site resonances occur at
$\omega L_{\rm site}/v_g = 2\pi M$ for integer $M$, i.e., $\omega/(2\pi) \cdot \Gamma_{\rm FSR}^{-1} = M$
where $\Gamma_{\rm FSR} = v_g/L_{\rm site}$ is the free spectral range.

The carrier wavelength has been **completely absorbed** into the choice of $\omega = 0$.
$\beta_0$ no longer appears in any equation. Only the detuning $\omega$ matters.

### What about the link ring?

The link has length $L_{\rm link} = L_{\rm site} + \eta$. Its full round-trip is

$$\phi_{\rm link}^{\rm RT}(\omega) = \beta(\omega_{\rm abs})   L_{\rm link}
= \beta_0 L_{\rm site} + \beta_0 \eta + \omega(L_{\rm site} + \eta)/v_g$$

The first piece, $\beta_0 L_{\rm site}$, is $2\pi N$ — wraps to identity as before. The
second piece, $\beta_0 \eta$, **does not wrap** because $\eta$ is engineered such that

$$\boxed{\quad \beta_0 \eta = \pi \quad}$$

This is the **anti-resonance condition**: the link is exactly half a wavelength longer
than the site, so its carrier round-trip is 2πN + π — half a turn off from the site.
This is what makes the link ring not host modes in the band of interest; it's
rejecting signal from circulating in itself.

The third piece, $\omega(L_{\rm site} + \eta)/v_g$, is the detuning-dependent phase.
We separate this into the "site-like" part $\omega L_{\rm site}/v_g$ and the small
correction $\omega \eta / v_g$.

How small is the correction? For the SiN platform of [§1](#1-setting) ($\eta \approx 440$ nm,
$L_{\rm site} \approx 190$ μm), $\eta/L_{\rm site} \approx 2.3 \times 10^{-3}$. The
detuning $\omega$ ranges over the FSR, so $|\omega L_{\rm site}/v_g| \lesssim 2\pi$.
Therefore $|\omega \eta/v_g| \lesssim 2\pi \cdot 2.3 \times 10^{-3} \approx 0.015$ rad — an
order-of-magnitude smaller than any other phase. **It is dropped throughout.**

Equivalently: the link's *frequency-dependent* phase is treated as $\omega L_{\rm site}/v_g$
(same as the site), and $\eta$ enters *only* through the static $\beta_0 \eta = \pi$.

This is the source of an important earlier bug. The original code used
$\Delta z_{\rm link} = L_{\rm link} / N_z$ in the propagation factor, which gave
the link a different FSR from the site (period 2/3 in $\omega/(2\pi)$ instead of 1).
This is unphysical — it confuses the *physical length* (which $\eta$ does change) with
the *frequency-dependent phase* (which $\eta$ does not change at FSR scales). The fix
is to use $\Delta z_{\rm link} = L_{\rm site}/N_z$ for propagation, and add the static
$\beta_0\eta$ as a separate constant phase distributed over the link's grid points.

### Summary of the link's per-step factor

The link's round-trip phase, after the carrier wraps and we drop $O(\omega\eta)$, is

$$\phi_{\rm link}^{\rm RT}(\omega) = \beta_0 \eta + \omega L_{\rm site} / v_g$$

The static $\beta_0 \eta$ is distributed uniformly across the $N_z$ discretization steps:
each step contributes $\beta_0 \eta / N_z$ to the phase. (Discretization choice — see [§9](#9-link-rings-and-anti-resonance).)

---

## 4. Dimensionless variables

The simulator works entirely in dimensionless units. The transformations:

| Dimensional | Dimensionless | Convention |
|---|---|---|
| $L_{\rm site}$ | $1$ | The site ring is the unit length |
| $v_g$ | $1$ | Light travels one length unit per time unit |
| $T_R = L_{\rm site}/v_g$ | $1$ | Site round-trip time is the unit time |
| $\Gamma_{\rm FSR} = 1/T_R$ | $1$ | FSR is the unit frequency (in Hz) |
| $\omega$ (rad/s) | $\omega$ (dimensionless) | Detuning from carrier, in units of $\Gamma_{\rm FSR}\cdot 2\pi$ |
| $\alpha$ (1/m) | $\alpha$ (dimensionless) | Loss per length, with length in $L_{\rm site}$ units |
| $\eta$ (m) | $\eta_{\rm dim}$ (small) | Length difference in $L_{\rm site}$ units (typically $\ll 1$) |
| $\beta_0 \eta$ (rad) | $\pi$ | The actual physical knob for anti-resonance |

In these units:

- $L_{\rm site} = 1$, so propagation factor over the site ring is just $e^{i\omega - \alpha/2}$ per round-trip.
- The FSR in $\omega/(2\pi)$ axis units is exactly **1** — site resonances at integer $\omega/(2\pi)$.
- The link's static anti-resonance phase is $\beta_0 \eta_{\rm dim} = \pi$ regardless of how small $\eta_{\rm dim}$ is in length units. We expose this as the parameter `beta0_eta_over_pi` in the UI.
- The discretization step is $\Delta z = 1/N_z = 1/16$, both as a length and (since $v_g=1$) as a time.

### Conversion between simulator units and lab units

If your device has FSR $= \Gamma_{\rm FSR}^{\rm exp}$ (in Hz) and bus linewidth
$\kappa_{\rm ex}^{\rm exp}$ (FWHM in Hz), the dimensionless coupling parameter is

$$\kappa_{\rm ex}^{\rm sim} = \sqrt{\kappa_{\rm ex}^{\rm exp} / \Gamma_{\rm FSR}^{\rm exp}}$$

For example, $\kappa_{\rm ex}^{\rm exp} = 20$ GHz at $\Gamma_{\rm FSR}^{\rm exp} = 750$ GHz
gives $\kappa_{\rm ex}^{\rm sim} = \sqrt{0.0267} \approx 0.163$.

For the hopping $J$:

$$\kappa_J^{\rm sim} = \sqrt{2\pi \cdot J^{\rm exp} / \Gamma_{\rm FSR}^{\rm exp}}$$

The factor of $2\pi$ comes from the detailed IQH-lattice geometry and is derived
in the original Hafezi 2011 paper (which proposed this realization of the IQH lattice for photons) — see "the factor of 2" discussion in older
documentation. Briefly: a photon traversing a plaquette only sees one *arc* of each
link ring (between two DCs), not the full link round-trip, so the effective hopping
picks up an extra $2\pi$ in the relation between $\kappa_J^2$ and the band-structure $J$.

For loss:

$$\kappa_{\rm in}^{\rm exp} = \alpha^{\rm sim} \cdot \Gamma_{\rm FSR}^{\rm exp} / (2\pi)$$

So $\alpha = 0.01$ at $\Gamma_{\rm FSR} = 750$ GHz gives $\kappa_{\rm in} \approx 1.2$ GHz
intrinsic linewidth, corresponding to $Q_{\rm int} \sim 1.6 \times 10^5$ at telecom.

---

## 5. Single ring — discretization

Take a single isolated ring (no bus, no neighbors). The field around the ring is a
function of one coordinate $z \in [0, L_{\rm site})$ with periodic boundary $E(L_{\rm site}) = E(0)$.

Discretize into $N_z = 16$ equally-spaced grid points $z_k = k \Delta z$ for $k = 0, \ldots, N_z-1$,
where $\Delta z = L_{\rm site}/N_z = 1/16$. Let $E_k = E(z_k)$.

**Propagation rule.** Light at grid $k$ at time $t + \Delta t$ originates from grid $k-1$
at time $t$ (where $\Delta t = \Delta z / v_g = \Delta z$ in our units), having
picked up the per-segment factor $p$ derived in [§2](#2-physical-propagation-in-a-single-waveguide):

$$\boxed{\quad E_k(t + \Delta t) = p \cdot E_{k-1}(t) \quad}$$

with

$$p = e^{i \omega \Delta z - \alpha \Delta z / 2}$$

and the index $k-1$ taken modulo $N_z$ (so $k=0$'s predecessor is $k=N_z-1$, closing
the ring). This is the discrete-time analog of $\partial_t E + v_g \partial_z E = (i\omega - \alpha/2) v_g E$.

**Slot-ordering convention.** The choice $E_k \leftarrow E_{k-1}$ encodes one
specific direction of circulation around the ring perimeter — let's call it CCW
(counterclockwise) by convention. The opposite circulation (CW) corresponds to
$E_k \leftarrow E_{k+1}$. In an ideal (back-scatter-free) ring the two propagation
directions are degenerate — they live in independent pseudospin sectors. Our TMM
simulates **one** sector at a time. The choice of sector is set by the **DC tangent
direction** at the bus coupler (Input: left vs Input: right), which the user
controls; see [§15](#15-per-ring-chirality-flip-dc-tangent-direction) for the full story.

In matrix form, define the state vector $\vec{E}(t) = (E_0(t), E_1(t), \ldots, E_{15}(t))^T$.
Then

$$\vec{E}(t + \Delta t) = R \vec{E}(t)$$

where $R$ is a $16 \times 16$ cyclic shift matrix with $p$ on the subdiagonal
(and one $p$ in the upper-right corner for the wraparound):

$$R_{kj} = p \cdot \delta_{j, (k-1) \bmod N_z}$$

### Steady-state condition

For monochromatic excitation, the time-dependence is $e^{-i\omega t}$ — but in our
**envelope frame**, the steady-state envelope is *time-independent*. So the condition
is $\vec{E}(t + \Delta t) = \vec{E}(t)$, i.e., $\vec{E}$ is an eigenvector of $R$ with
eigenvalue 1:

$$R \vec{E} = \vec{E}$$

Equivalently, $E_k = p \cdot E_{k-1}$ for all $k$. Iterating around the ring:

$$E_0 = p \cdot E_{N_z-1} = p \cdot p \cdot E_{N_z-2} = \cdots = p^{N_z} \cdot E_0$$

So a non-trivial solution exists iff $p^{N_z} = 1$. Computing the exponent:

$$p^{N_z} = e^{i\omega \cdot N_z \Delta z  -  \alpha \cdot N_z \Delta z / 2}
= e^{i\omega L_{\rm site} - \alpha L_{\rm site}/2} = e^{i\omega - \alpha/2}$$

So in the lossless limit ($\alpha \to 0$), the resonance condition $p^{N_z} = 1$
becomes $e^{i\omega} = 1$, i.e., $\omega = 2\pi M$ for integer $M$. **One FSR per
$\Delta\omega = 2\pi$**, which in $\omega/(2\pi)$ axis units is $\Delta = 1$.

---

## 6. Single ring — adding the bus

A directional coupler at slot $k = 0$ couples the ring to a bus waveguide. The DC is
a 2×2 unitary scattering matrix:

$$\begin{pmatrix} a_{\rm thru} \\ E_0 \end{pmatrix} =
\begin{pmatrix} t_{\rm ex} & i\kappa_{\rm ex} \\ i\kappa_{\rm ex} & t_{\rm ex} \end{pmatrix}
\begin{pmatrix} a_{\rm in} \\ E_0^{\rm in} \end{pmatrix}$$

with $t_{\rm ex}^2 + \kappa_{\rm ex}^2 = 1$ (lossless DC), $\kappa_{\rm ex} \in [0, 1]$,
and the $i$ enforcing time-reversal symmetry. Here:

- $a_{\rm in}$ is the bus input amplitude (we set this to **1** — unit drive).
- $a_{\rm thru}$ is the bus output amplitude (the through port).
- $E_0^{\rm in}$ is the cavity field arriving at slot 0 from the previous grid point.
  This is $E_0^{\rm in} = p \cdot E_{N_z-1}$ (the field at $k = N_z-1$ propagated by $\Delta z$).
- $E_0$ is the cavity field *after* the DC — the new amplitude at slot 0.

The DC gives:

$$E_0 = t_{\rm ex} \cdot p \cdot E_{N_z-1} + i\kappa_{\rm ex} \cdot a_{\rm in}$$

This **replaces** the free-propagation rule for $k = 0$ only:

$$
\begin{aligned}
E_0(t + \Delta t) &= t_{\rm ex} \cdot p \cdot E_{N_z-1}(t)  +  i\kappa_{\rm ex} \\
E_k(t + \Delta t) &= p \cdot E_{k-1}(t), \quad k = 1, \ldots, N_z - 1
\end{aligned}
$$

In matrix form: $R$ keeps its sparsity pattern but row 0 has $t_{\rm ex} \cdot p$
in column $N_z-1$ instead of $p$. The source vector $\vec{s}$ has $i\kappa_{\rm ex}$
at row 0 and zero elsewhere:

$$\vec{E}(t + \Delta t) = R \vec{E}(t) + \vec{s}$$

### The thru-port amplitude

From the DC matrix:

$$a_{\rm thru} = t_{\rm ex} + i\kappa_{\rm ex} \cdot p \cdot E_{N_z-1}$$

This is the bus output: directly-transmitted bus amplitude *plus* the cross-coupled
cavity amplitude. They interfere — that interference creates the resonance dips in
transmission. At critical coupling on resonance they cancel exactly.

---

## 7. Single ring — steady state vs time evolution

There are two ways to find the field given the propagation rule.

### (a) Steady-state solver

Set $\vec{E}(t + \Delta t) = \vec{E}(t) \equiv \vec{E}_\infty$ (the field doesn't
change in the envelope frame at steady state):

$$\vec{E}_\infty = R \vec{E}_\infty + \vec{s}$$

Rearranged: $(I - R) \vec{E}_\infty = \vec{s}$. **One sparse linear solve** gives
the answer at any frequency $\omega$. This is `solve_one(omega, ...)` in the code.

For the single ring on resonance ($\omega = 0$, so $p = e^{-\alpha/(2 N_z)}$, real),
all $E_k$ are equal in steady state by translation symmetry:

$$E_\infty = t_{\rm ex} \cdot p^{N_z} \cdot E_\infty + i\kappa_{\rm ex}$$
$$E_\infty = \frac{i\kappa_{\rm ex}}{1 - t_{\rm ex} \cdot e^{-\alpha/2}}$$

For $\kappa_{\rm ex} = 0.163$ and $\alpha = 0.01$ (your defaults), this gives
$|E_\infty|^2 \approx 5.6$ — i.e., the intracavity intensity is ~5.6× the input intensity.
This is the well-known cavity buildup factor.

### (b) Time-domain iteration

Start from $\vec{E}(0) = \vec{0}$ (device dark before drive). Apply the map
$\vec{E}^{(n+1)} = R \vec{E}^{(n)} + \vec{s}$ for $n = 0, 1, 2, \ldots$. Each step
advances physical time by $\Delta t = T_R / N_z$.

This is `time_evolve(omega, ...)` in the code. The trajectory $\{\vec{E}^{(n)}\}$
shows the **transient buildup** of the field, and converges to $\vec{E}_\infty$
asymptotically.

### Convergence rate

The iteration converges if and only if the **spectral radius** $\rho(R) < 1$. The
eigenvalues of $R$ for the bus-coupled ring are related to the round-trip survival
factor: $|\lambda_{\max}|^{N_z} \approx t_{\rm ex} \cdot e^{-\alpha/2}$, so
$|\lambda_{\max}| \approx (t_{\rm ex} \cdot e^{-\alpha/2})^{1/N_z}$.

For your defaults: $t_{\rm ex} \approx 0.987$, $e^{-\alpha/2} \approx 0.995$, so
$|\lambda_{\max}|^{N_z} \approx 0.982$ per round-trip — a 1.8% decay per round-trip.
Convergence to within $10^{-3}$ of steady state takes roughly $\log(10^{-3})/\log(0.982)
\approx 380$ round-trips. To within $10^{-6}$, ~760 round-trips.

This is **slow** — high-Q cavities need long iteration times to settle. The simulator
doesn't enforce any auto-stop; the user picks `N_steps` and watches what happens.

---

## 8. The IQH lattice — geometry and indexing

This section covers the IQH (Hafezi-style) lattice. The AQH (anomalous) lattice has
its own slot-and-coupling conventions; see [§14](#14-the-aqh-lattice--anomalous-floquet-topology-without-flux).

Scale up. The IQH lattice is $N_x \times N_y$ site rings on a square grid, with link
rings on every nearest-neighbor bond (horizontal H-links between $(i_x, i_y)$ and
$(i_x+1, i_y)$; vertical V-links between $(i_x, i_y)$ and $(i_x, i_y+1)$).

### Site-ring DC slot convention

Each site ring has $N_z = 16$ grid points. We reserve specific slots for DCs:

| Slot index | Role |
|---|---|
| 0 | Bus (only at IN/OUT sites) |
| 4 | Right-link H-DC |
| 8 | Top-link V-DC |
| 10 | Bottom-link V-DC |
| 12 | Left-link H-DC |

The remaining 11 slots are pure propagation (no DC). Why these specific slot indices?
Slots 0/4/8/12 are at quarter-turn intervals around the ring (so they're at the four
sides — "south", "east", "north", "west"). Slot 10 is between TOP (8) and the going-back
to BUS (0), specifically between the TOP and BOTTOM physical positions of the ring.
This is a bit awkward — TOP and BOTTOM are physically opposite sides of the ring,
but we use adjacent slot numbers (8 and 10) because of how the visualization
works — but it does not affect the physics. The simulation only cares about
the *cumulative* phase between DCs, which is set by the segment lengths.

A bulk site has 4 link DCs (right + top + bottom + left) and 11 free-propagation steps.
An edge site has 3 link DCs; a corner site has 2. The IN site additionally has a bus DC
at slot 0.

### Link-ring DC slot convention

Each link ring also has $N_z = 16$ grid points. The two DCs (one to each adjacent site) are at:

| Slot index | Role |
|---|---|
| 0 | "Near" site DC |
| 8 (= $N_z/2$) | "Far" site DC |

So the link is split in two equal arcs of 8 grid points each between the DCs.

For an H-link `H_ix_iy` connecting $(i_x, i_y)$ and $(i_x+1, i_y)$: the "near" end
is at the right side of $(i_x, i_y)$ (slot 4 of that site), and the "far" end is at
the left side of $(i_x+1, i_y)$ (slot 12 of that site). For V-links, "near" = TOP
of the lower site (slot 8), "far" = BOTTOM of the upper site (slot 10).

### State vector layout

The total state size for an $N_x \times N_y$ lattice is

$$N_{\rm state} = N_x N_y \cdot N_z + (N_x - 1) N_y \cdot N_z + N_x (N_y - 1) \cdot N_z$$

For 4×4: $16 \cdot 16 + 12 \cdot 16 + 12 \cdot 16 = 640$ unknowns. The state vector is
laid out as: all sites first (in row-major order), then all H-links, then all V-links,
each contributing $N_z$ consecutive entries. Lookup functions `site_idx(ix, iy, k)`
and `link_idx(name, k)` return the index into the state vector.

---

## 9. Link rings and anti-resonance

In each link-ring grid step, the propagation factor is

$$p_{\rm link} = e^{i\omega \Delta z + i\beta_0\eta/N_z - \alpha \Delta z / 2}$$

The static $i\beta_0\eta/N_z$ phase is **distributed uniformly** across all $N_z$ steps
of the link ring, so they sum to $\beta_0\eta = \pi$ over a full link round-trip.
This is a discretization choice, not physics — the static phase is a property of the
*whole* ring, and we could have lumped it onto one step. Uniform spreading ensures
the field at any intra-link grid point matches what you'd get from a continuous
$\beta(z) = \beta_0 + \omega/v_g$ with uniform $\beta_0$ around the ring.

Note: $\Delta z$ here is $L_{\rm site}/N_z$ for both site and link rings — even though
the link's *physical* length is $L_{\rm site} + \eta$, we use $L_{\rm site}/N_z$ in the
$\omega$-dependent and loss terms because the additional $\eta$ contribution to those
is negligible at FSR scales (see [§3](#3-choice-of-frame-the-carrier-wraps-to-identity) discussion). The only thing $\eta$ does is contribute
the static $\beta_0\eta = \pi$ to the round-trip — and *that* is captured by the
$i\beta_0\eta/N_z$ piece.

This was the source of an earlier bug — see version history.

### Anti-resonance physics

Why $\beta_0\eta = \pi$? On a site resonance ($\omega = 0$), the site round-trip is
$2\pi N$ — identity. The link's round-trip is $2\pi N + \pi$ — minus identity. So the
field circulating in the link picks up a sign flip per round-trip, which causes
*destructive interference* with itself. The link doesn't accumulate field — it just
serves as a passive coupling channel between adjacent sites.

If we set $\beta_0\eta = 0$ (link resonant with site), both rings would host modes
in the same band and the simple "site-on-tight-binding-lattice" picture breaks down.
The simulator's `β₀η` spinbox lets you explore this regime continuously.

---

## 10. The Peierls phase and synthetic flux

To realize integer quantum Hall physics, photons must accumulate a phase $\Phi_0$
when going around any plaquette of the lattice. This is the **Peierls phase** of
the synthetic magnetic field.

The IQH-lattice mechanism: introduce a directional asymmetry on the H-links such that
photons going "up" through the link pick up a different phase than photons going
"down". Specifically, the link is implemented so that the upper arc and lower arc
have different phases — equivalent to slightly off-axis link rings.

In the code, this is implemented via the `extras_arr`: the link-ring propagation step
between grid 0 and grid 1 picks up an extra $+\theta/2$ phase, and the step between
grid $N_z/2$ and grid $N_z/2 + 1$ picks up an extra $-\theta/2$. Net round-trip
phase change is zero (so link's anti-resonance is preserved), but the *directional*
phase asymmetry is $\theta$.

For an H-link in row $i_y$, we set $\theta = -2\Phi_0 \cdot i_y$ (Landau gauge).
A photon hopping right through this link sees $-\Phi_0 \cdot i_y$ accumulated phase;
hopping left sees $+\Phi_0 \cdot i_y$. Going around a unit plaquette CCW:

- Hop right at row $i_y$: phase $-\Phi_0 i_y$
- Hop up: 0 (V-links have no Peierls phase in Landau gauge)
- Hop left at row $i_y + 1$: phase $+\Phi_0 (i_y + 1)$
- Hop down: 0

Total: $-\Phi_0 i_y + \Phi_0 (i_y + 1) = \Phi_0$. ✓ Uniform flux per plaquette.

### The factor of 2

There's a subtle factor-of-2 in the Peierls implementation. A photon traversing a
plaquette only goes through *one arc* of each link ring (between the two DCs), not
the full link round-trip. So the per-bond Peierls phase is *half* the link's
directional asymmetry. To get plaquette flux $\Phi_0$, the asymmetry $\theta$ is
set to $2\Phi_0$ — and the per-bond phase that appears in the band structure is
$\theta/2 = \Phi_0$. (See "the factor of 2" discussion in older theory notes.)

This factor-of-2 is also what gives us $\kappa_J^{\rm sim} = \sqrt{2\pi J/\Gamma_{\rm FSR}}$
in the conversion to experimental $J$ — see [§4](#4-dimensionless-variables).

---

## 11. Assembling the global sparse system

The complete propagation map for the lattice is built equation-by-equation:

For every grid point $(ring, k)$, write down the equation $E_k = (\text{stuff})$.
"Stuff" depends on whether the slot is a DC or a free segment.

### Free propagation segment

Most slots are not DCs. The equation is just:

$$E_k = p \cdot E_{k-1}$$

Contributes one entry to row `idx(ring, k)`, column `idx(ring, k-1)`, value $p$.

### Site bus_in DC

At slot 0 of the IN site:

$$E_0 = t_{\rm ex} \cdot p_{\rm site} \cdot E_{N_z-1} + i\kappa_{\rm ex} \cdot a_{\rm in}$$

Contributes: one entry to (row=site_0, col=site_{N_z-1}) with value $t_{\rm ex} \cdot p_{\rm site}$,
and a source term $i\kappa_{\rm ex}$ at row=site_0.

### Site bus_drop DC

Same as bus_in but with $a_{\rm in} = 0$ (drop port doesn't have an input):

$$E = t_{\rm ex} \cdot p_{\rm site} \cdot E_{\rm prev}$$

Just the one entry, no source term.

### Site link DC

At a slot connected to a link ring (e.g., slot 4 = right H-link):

$$E_{\rm site, k} = t_J \cdot p_{\rm site} \cdot E_{\rm site, k-1} + i\kappa_J \cdot p_{\rm link} \cdot e^{i\theta_{\rm extra}} \cdot E_{\rm link, k_{\rm link}^{\rm prev}}$$

where $k_{\rm link}^{\rm prev}$ is the predecessor of the link's DC slot (i.e. one $\Delta z$ before the DC, on the link's side).
Two entries: a self-coupling from the previous site grid (with $t_J$ instead of $1$
or $p$), and a cross-coupling from the partner link's grid just before its DC.
The extra phase $\theta_{\rm extra}$ at the link's DC slot is the Peierls $\pm\theta/2$
contribution.

### Link DC

At link slot 0 ("near") or $N_z/2$ ("far"):

$$E_{\rm link, k} = t_J \cdot p_{\rm link} \cdot e^{i\theta_{\rm extra}} \cdot E_{\rm link, k-1} + i\kappa_J \cdot p_{\rm site} \cdot E_{\rm site, k_{\rm site}^{\rm prev}}$$

where $k_{\rm site}^{\rm prev}$ is the predecessor of the site's DC slot.
Two entries: a self-coupling from the previous link grid, and a cross-coupling from
the site partner's grid just before its DC.

### Putting it all together

For each entry `(row, col, value)` we get a single nonzero element of $R$. The full
matrix is constructed once at template-build time (sparsity pattern only depends on
geometry and topology), then values are recomputed per frequency by multiplying the
template's `coeffs` array by the per-step factors.

The total number of nonzeros is roughly $N_z \cdot (\text{number of rings})$ for free
propagation, plus $2 \cdot (\text{number of DCs})$ for the cross-couplings. For a
4×4 lattice: ~640 + ~70 = ~710 nonzeros in a 640×640 matrix. Very sparse (~0.2% fill).

The steady-state matrix $M = I - R$ has the same sparsity plus the diagonal. We solve
$M \vec{E} = \vec{s}$ by `scipy.sparse.linalg.splu` — a sparse LU factorization that
exploits the structure. Solve time: ~1ms for a 4×4 lattice.

---

## 12. Solving — sparse LU vs Ikeda iteration

### Sparse LU (steady state)

`solve_one(omega, ...)`:

1. Build per-step factors $p_{\rm site}(\omega)$, $p_{\rm link}(\omega)$ from current $\omega$.
2. Compute matrix entries: `vals = -coeffs * p` (mostly).
3. Assemble $M = I - R$ as a CSC sparse matrix.
4. Factor with `splu`, back-substitute on $\vec{s}$.
5. Return $\vec{E}_\infty$, $s_{\rm thru}$, $s_{\rm drop}$.

This is the workhorse for spectrum scans. Each frequency point is independent; loop
over $\omega$ values.

### Ikeda map (time evolution)

`time_evolve(omega, ...)`:

1. Build $R$ with the same structure as `solve_one` but **without** the identity
   (the propagator itself, not $I - R$).
2. Initialize $\vec{E}^{(0)} = \vec{0}$.
3. Iterate $\vec{E}^{(n+1)} = R \vec{E}^{(n)} + \vec{s}$ for $n = 0, \ldots, N_{\rm steps}-1$.
4. Record $\vec{E}^{(n)}$, $s_{\rm thru}^{(n)}$, $s_{\rm drop}^{(n)}$ at intervals.

Each step is one sparse mat-vec (cheap, ~30 μs for 640×640). 1000 steps → ~30 ms total.

### Why Ikeda iteration when sparse LU is faster?

- **It tracks transient dynamics.** You see the field build up over time. Useful for
  understanding photon lifetimes and ring-up dynamics.
- **It generalizes to nonlinear or time-dependent systems.** Just rebuild $R$ each
  step with whatever new physics you want (Kerr from $|E|^2$, EO modulation from
  $\sin(\Omega t)$, etc.).
- **It's a cross-check on the steady-state solver.** Long-time average of the
  iteration must match `solve_one`'s output.

---

## 13. Removed rings (defects)

The simulator allows the user to "remove" any site or link ring by clicking on it.
Implementation: the corresponding ring's DC entries are zeroed out (i.e., $\kappa = 0$
at every DC touching that ring), but the ring's grid points remain in the state vector
to keep indexing stable.

Effectively: a removed ring decouples from the rest of the lattice. The field inside
it propagates with itself only (under the loss $\alpha$), so any seed decays to zero.
The neighboring rings see the removed ring as if it weren't there — they propagate
through the slot that *would* have been a DC as if it were a free segment.

This is useful for studying:

- **Topological protection**: remove a single bulk site; the chiral edge mode reroutes
  around it without back-scattering. (Visible in the field plot at a given $\omega$.)
- **Boundary engineering**: remove a chain of edge sites; see how the edge mode
  responds, whether it can still circumnavigate the lattice via a longer path.
- **Defect-induced bound states**: removing rings creates "antidots" — locations
  where photons might localize. With multiple defects you can engineer a lattice
  geometry not realizable through gauge alone.

The IN/OUT sites are protected from removal — they carry the bus DCs, removing them
would break I/O.

---

## 14. The AQH lattice — anomalous Floquet topology without flux

The IQH lattice realizes integer quantum Hall via a Peierls phase pattern that gives
each plaquette a fixed flux $\Phi_0$. The **anomalous quantum Hall (AQH)** lattice
gets topological edge modes a different way: each plaquette has *zero* net flux, but
the periodic-driving structure of the link rings (the photon's "internal time spent in
the link") creates a non-trivial Floquet bulk band. The AQH lattice was first proposed
by Liang–Chong (2013), refined into the photonic anomalous Floquet topological insulator
(AFI) of Pasek–Chong (2014), and first realized experimentally on a silicon photonic
platform by Mittal–Hafezi (2019). More recent silicon-nitride implementations of the
same brick-wall lattice — the platform this simulator is built to support — have
demonstrated nonlinear topological frequency combs [Flower, Mehrabad, Xu *et al.*,
*Science* **384**, 1356 (2024)], multi-timescale mode-locked states with independently
tunable single-ring (∼1 THz) and topological super-ring (∼3 GHz) timescales [Xu,
Mehrabad, Flower *et al.*, *Sci. Adv.* **11**, eadw7696 (2025)], and a passive
nested frequency-phase matching mechanism for wafer-scale multi-harmonic generation
[Mehrabad, Xu *et al.*, *Science* (2025), doi:10.1126/science.adu6368].

### Brick-wall geometry

The AQH lattice in the simulator is a **brick-wall** (zigzag) arrangement on a
$(2 N_x - 1) \times (2 N_y - 1)$ grid:

- **Site rings** sit on rows where, in narrow rows ($r$ even), they're at odd $c$
  (1, 3, 5, …); in wide rows ($r$ odd), they're at even $c$ (0, 2, 4, …).
- **Link rings** sit at every odd $(c, r)$ — i.e., at the centers of the diamond
  plaquettes formed between four neighboring sites.

For $N_x = N_y = 4$: 24 site rings and 9 link rings, arranged on a 7×7 grid.

```
Row 0 (narrow):   ·  ○  ·  ○  ·  ○  ·          ← sites at (1,0),(3,0),(5,0)
Row 1 (wide):     ○  □  ○  □  ○  □  ○          ← sites at even c, links at odd c
Row 2 (narrow):   ·  ○  ·  ○  ·  ○  ·
Row 3 (wide):     ○  □  ○  □  ○  □  ○
   ...
```
where ○ = site ring, □ = link ring, · = empty grid position.

Each link ring couples to **four neighboring sites** — N, E, S, W — through four
DCs (slots 0, 4, 8, 12 of the link). Each site couples to **at most two links**:
narrow-row sites have one V-link neighbor (N or S), wide-row sites have two H-link
neighbors (E and W).

### AQH slot conventions

Site rings use $N_z$ grid points per ring (default 16; configurable in the UI
to 16, 32, 48, or 64). Four cardinal slots at quarter-turns are reserved for
link DCs:

| Slot | Direction | Meaning |
|---|---|---|
| 0 (`SLOT_AQH_N`) | North | Link above the site (smaller $r$ direction) |
| $N_z/4$ (`SLOT_AQH_E`) | East | Link to the right (larger $c$) |
| $N_z/2$ (`SLOT_AQH_S`) | South | Link below the site (larger $r$) |
| $3 N_z/4$ (`SLOT_AQH_W`) | West | Link to the left (smaller $c$) |

Link rings use the **same four cardinal slots**, each pointing toward the corresponding
neighboring site. The DC pairing is **antipodal**: ring A's slot $s$ couples to ring B's
slot $(s + N_z/2) \bmod N_z$. So a site's slot 0 (N pointing toward the link) couples to
the link's slot $N_z/2$ (S pointing back toward the site). This antipodal pairing is what
encodes the geometric fact that two externally tangent rings have opposite circulation
senses at the tangent point.

### Default bus placement

For an AQH lattice with $N_x, N_y \geq 2$:

- IN bus: bottom-row narrow site at grid $(1, 2 N_y - 2)$
- OUT bus: top-row narrow site at grid $(1, 0)$

Both bus sites are narrow-row and have a single V-link neighbor (N for the OUT site,
S for the IN site). The bus DC is placed on the slot **opposite** to the link
neighbor — so for IN (link is N at slot 0), the bus sits at slot $N_z/2$ (S, visually below
the lattice); for OUT (link is S at slot $N_z/2$), the bus sits at slot 0 (N, visually above).

### Floquet-topology mechanism (sketch)

Heuristically: a photon entering a site ring spends some time (∝ link length / v_g)
inside the link before reaching a neighbor. During that time it "circulates" inside
the link ring, accumulating a phase that depends on which arc it traverses. The
brick-wall geometry forces a *cyclic* sequence of arc traversals — N → E → S → W
type — that mimics a periodic drive. With anti-resonant link rings ($\beta_0\eta = \pi$),
the resulting Floquet bulk Hamiltonian has nontrivial winding number even though the
Bloch bands themselves have zero Chern number. The edge modes are anomalous: they
exist on the boundary of *every* gap, not just the central one, and they're robust to
disorder of the link couplings.

In the simulator, the topology is set entirely by the geometric connectivity — there
is no $\Phi_0$ knob for AQH. The $\beta_0\eta$ parameter still controls anti-resonance
(set it to $\pi$ for ideal AQH; sweep across 0–4π to see Floquet phase transitions).

### State vector layout

For an AQH template:

$$N_{\rm state} = n_{\rm sites} \cdot N_z + n_{\rm links} \cdot N_z$$

with $n_{\rm sites} = N_x (N_y - 1) + N_y (N_x - 1)$ and $n_{\rm links} = (N_x - 1)(N_y - 1)$.
For $4 \times 4$: $24 \cdot 16 + 9 \cdot 16 = 528$ unknowns.

Sites and links each get their own 1-based index space (i.e., site indices and link
indices don't overlap), and the state vector is laid out as: all sites first
(in canonical row-major brick-wall order), then all links. Lookup: `site_idx(i, k)` and
`link_idx(j, k)`.

### IQH vs AQH sparse system

The AQH template assembly uses the same machinery as IQH but with different connectivity:

- For each site equation at slot $k$: if $k \in \{0, 4, 8, 12\}$ AND there's a link
  in that direction, write a 2-term equation (self-coupling + cross-coupling from the
  link's antipodal slot's predecessor). Otherwise free propagation.
- For each link equation at slot $k$: if $k \in \{0, 4, 8, 12\}$ AND there's a site
  in that direction, write a 2-term equation (self-coupling + cross-coupling from the
  site's antipodal slot's predecessor). Otherwise free propagation.

There are **no Peierls phases in the AQH template** — the synthetic gauge structure
is encoded entirely in *which slots are paired*. (Compare IQH, where two sites are
paired through a single link with a directional Peierls extra phase.)

### Connection to the experimental SiN platform

The AQH lattice in this simulator targets the silicon-nitride coupled-ring platform
used in the recent experimental work above. The mapping between simulator quantities
and experimental observables is direct: the steady-state spectrum at low pump power
gives the linear edge-band structure resolved in *Science* **384**, 1356 (2024); the
fast and slow timescales of the multi-timescale mode-locking work [*Sci. Adv.* **11**,
eadw7696 (2025)] correspond, respectively, to the single-ring round-trip time
$T_R = 1/\Gamma_{\rm FSR}$ and the topological super-ring time $\sim N_{\rm edge} T_R$
where $N_{\rm edge}$ is the number of sites along the edge (the slow GHz timescale =
fast THz timescale ÷ edge length); and the nested frequency-phase matching scheme of
the wafer-scale harmonic generation work [*Science* (2025), doi:10.1126/science.adu6368]
relies on the two-timescale density of states that this simulator's spectrum
visualizes directly. Three caveats apply when comparing simulator output to those
experiments: this simulator is **linear** (the comb formation, mode-locking, and
harmonic generation all require Kerr/$\chi^{(3)}$ or $\chi^{(2)}$ nonlinearity not
included here, see [§19](#19-what-this-simulator-does-not-model)); it simulates **one pseudospin** at a time (so does not
capture CW/CCW backscattering from sidewall roughness, which can matter at high $Q$);
and the rendered fields are **steady-state envelopes** at a single pump frequency,
not soliton or mode-locked temporal profiles. The simulator is the linear scaffold
on top of which those nonlinear effects are built.

---

## 15. Per-ring chirality flip (DC tangent direction)

The user can toggle the bus DC tangent direction through a UI control labeled
`Input: left` / `Input: right`. Internally this is a boolean parameter `dc_flip` on
the template builders. Despite its name, the flip propagates through the entire TMM —
**every** ring's slot ordering reverses, not just the bus DCs.

### Physical motivation

In a real device, each directional coupler has a tangent line where two waveguides
run side-by-side. Light injected from one end of the bus moves rightward and couples
into the ring at the DC, exciting whichever ring mode (CW or CCW pseudospin) shares
the bus's propagation direction at the tangent point. Inject from the other end and
the bus light moves leftward, coupling to the other pseudospin.

This is realized as a *physical mirror* of the bus DC layout — flipping which end of
the bus is labeled "input" is equivalent to mirror-reflecting the bus tangent. And
once you mirror-flip *one* DC, you have to flip *all* of them consistently, otherwise
the geometry doesn't close.

### Implementation: slot-reversal everywhere

In our TMM, the slot ordering $0 \to 1 \to 2 \to ... \to N_z - 1 \to 0$ encodes one
direction of circulation. The propagation rule $E_k = p \cdot E_{k-1}$ says light at
slot $k$ comes from slot $k-1$ — i.e., light flows in the +k direction. To flip the
direction, we'd want $E_k = p \cdot E_{k+1}$ instead.

Concretely, when `dc_flip = True`, we set a sign $\sigma = -1$ (and $\sigma = +1$
otherwise) and replace every

$$k_{\rm prev} = (k - 1) \bmod N_z  \longrightarrow  k_{\rm prev} = (k - \sigma) \bmod N_z$$

throughout the template. The same sign affects:

- The site-ring propagation predecessor: `(k - σ) % Nz_site`
- The link-ring propagation predecessor: `(k - σ) % Nz_link`
- The cross-coupling readout slot at every DC: `(neighbor_slot - σ) % Nz`
- The bus through-port readout slot: `(s_in_slot - σ) % Nz_site` and same for drop

The bus injection slot and the cross-coupling source slots are unchanged — the
physical bus DC sits at the same grid position regardless of which direction the
light flows. Only the **direction** of propagation past the DC changes.

### Behavior on a single ring

For an isolated bus-coupled ring at $\omega = 0$ (resonance), $\kappa_{\rm ex} = 0.163$:

- `dc_flip = False`: photon enters at slot 0, propagates 0 → 1 → 2 → … → 15 → 0.
  At step $n$ it has reached slot $n \bmod 16$.
- `dc_flip = True`: photon enters at slot 0, propagates 0 → 15 → 14 → … → 1 → 0.
  At step $n$ it has reached slot $-n \bmod 16$.

Steady-state $|E|^2$ and bus port amplitudes are **identical** between the two
configurations (this is reciprocity for a passive linear system) — but the
*transient* trajectory is the mirror image. The time-evolution dialog clearly shows
the photon racing one direction vs the other around the ring.

### Behavior on a multi-ring lattice

For multi-ring lattices (IQH or AQH), flipping every ring's slot ordering also
flips the **sign of the topological invariant**:

- IQH: the Peierls phases $\theta = -2\Phi_0 \cdot i_y$ on H-links are now traversed
  in reverse order around each ring, which is equivalent to flipping the sign of $\Phi_0$.
  The Chern number changes sign and the chiral edge mode reverses direction around
  the boundary.
- AQH: flipping the slot ordering reverses the cyclic Floquet drive's direction,
  which also flips the sign of the anomalous Chern number.

This is **physically correct**: in experiment, mirror-flipping every DC tangent
direction (which is what `dc_flip = True` represents) IS what produces the opposite
topological phase. The simulator's spectrum at flipped-vs-default is identical in
shape (by reciprocity) but mirrored about $\omega = 0$ for IQH, and similarly
reorganized for AQH.

### What does NOT change

- Total |E|² stored in the lattice at any peak frequency (reciprocity).
- The $|s_{\rm thru}|^2$, $|s_{\rm drop}|^2$ spectrum — only its $\omega$ axis is
  potentially mirrored when interpreted on top of the underlying Peierls/Floquet
  band structure.
- The schematic geometry (ring positions don't move).

### What DOES change

- Bus arrow directions and "input"/"through" port labels in the rendered lattice.
- For multi-ring lattices: the spatial pattern of the chiral edge mode at any peak
  (it now hugs the boundary going the opposite way).
- The transient propagation direction visible in time-evolution.

---

## 16. Time-evolution dialog

The simulator includes a separate window for time-resolved field evolution at a
chosen detuning $\omega$. The dialog opens after a peak is selected (or any
spectrum click) and runs the Ikeda iteration:

$$\vec{E}^{(n+1)} = R   \vec{E}^{(n)} + \vec{s}, \qquad \vec{E}^{(0)} = \vec{0}$$

over $N_{\rm steps}$ propagation steps (each step = $\Delta z = 1/N_z$ in time, so
$N_z$ steps = one round-trip).

### Outputs

- A live field-distribution plot at every recorded frame, showing the photon
  redistributing around the lattice.
- A drop-port amplitude trace $|s_{\rm drop}^{(n)}|^2$ vs round-trip number, with the
  steady-state reference line $|s_{\rm drop}^{\infty}|^2$ overlaid (computed from
  `solve_one`/`solve_one_aqh` for cross-validation).
- Optional MP4 / GIF export of the animation.

The slider at the bottom scrubs through the recorded frames. The reference dashed line
is computed using the per-step propagation factor

$$p_{\rm site}(\omega) = e^{i\omega/N_z - \alpha/(2 N_z)}$$

and the steady-state field at the drop bus's predecessor slot:

$$s_{\rm drop}^{\infty} = i \kappa_{\rm ex} \cdot p_{\rm site} \cdot E_\infty\left[i_{\rm drop},  k_{\rm pred}\right]$$

where $i_{\rm drop}$ is the drop site's index and $k_{\rm pred} = (k_{\rm slot} - \sigma) \bmod N_z$ uses the same direction
sign $\sigma$ as the rest of the template, with $k_{\rm slot}$ the drop bus DC slot.

### Convergence considerations

For high-Q cavities (small $\alpha$), the Ikeda iteration is slow to settle: the
spectral radius of $R$ is close to 1, and the field takes many round-trips to reach
steady state. The simulator's defaults ($\alpha = 0.01$, $\kappa_{\rm ex} = 0.359$,
$\kappa_J = 0.9$) give buildup times of order a few hundred round-trips for high-Q
edge modes (the matched coupling regime $\kappa_{\rm ex} = \kappa_J/\sqrt{2\pi}$
balances loading vs internal hopping). The dialog lets the user pick any $N_{\rm rt}$
to record; values of 200–2000 are typical.

### Lattice-type-aware rendering

The dialog renders fields with `plot_field_distribution` (IQH) or `draw_aqh_schematic`
(AQH), dispatched via the cached `self._is_aqh` flag set at dialog construction time.
Both renderers respect the template's `dc_flip` flag and flip bus arrows accordingly.

---

## 17. Field-distribution rendering

The state vector $\vec{E}$ has a complex amplitude $E_{i,k}$ at every (ring, slot)
position. Visualizing it requires a mapping from $(i, k)$ pairs to physical $(x, y)$
positions on the canvas. This section explains the conventions used by
`plot_field_distribution` (IQH) and `draw_aqh_schematic` (AQH).

### Ring perimeter parametrization

Each ring is drawn as a rounded square of half-side $h \approx 0.24$ and corner
radius $r_c \approx 0.07$ (in lattice grid units). The function
`_rounded_square_perimeter(cx, cy, h, rc, n)` returns $n$ equispaced points
$(x(s), y(s))$ around the perimeter, parametrized by $s \in [0, 1)$ starting at
the right edge near the **smaller-y** corner $(cx + h, cy - (h - rc))$ and
proceeding **counter-clockwise** in plot-data coordinates (math y-up).
Schematically:

| $s$ range | Position |
|---|---|
| $[0, \approx 0.19]$ | Right edge going up |
| $\approx 0.25$ | Top-right corner |
| $[0.25, \approx 0.44]$ | Top edge going left |
| $\approx 0.50$ | Top-left corner |
| $[0.50, \approx 0.69]$ | Left edge going down |
| $\approx 0.75$ | Bottom-left corner |
| $[0.75, \approx 0.94]$ | Bottom edge going right |

The midpoints of the four straight edges (in plot data) sit at:

| Edge | $s_{\rm frac}$ midpoint |
|---|---|
| Right edge (large $x$) | $\approx 0.094$ |
| Top edge (large $y$) | $\approx 0.345$ |
| Left edge (small $x$) | $\approx 0.594$ |
| Bottom edge (small $y$) | $\approx 0.845$ |

These exact values are specific to $h = 0.24$, $r_c = 0.07$ and would shift if
the corner radius changes. The renderer hardcodes 0.345 and 0.845 — if you
adjust the ring geometry, recompute these.

### `_draw_ring`: phase_offset and chirality

`_draw_ring(ax, center, half_side, intensities, I_max, ..., phase_offset,
chirality, corner_radius)` colors each segment of the perimeter by the
intensity at the corresponding slot. The mapping from segment $s_{\rm frac}$
to slot index $k$ is

$$
k_{\rm continuous} =
\begin{cases}
((s_{\rm frac} - \phi_0) \bmod 1) \cdot N_z & \chi = +1 \\
((\phi_0 - s_{\rm frac}) \bmod 1) \cdot N_z & \chi = -1
\end{cases}
$$

where $\phi_0$ is the renderer's `phase_offset` parameter and $\chi \in \{+1, -1\}$
is the `chirality` parameter. So `phase_offset` rotates which $s_{\rm frac}$
corresponds to slot 0, and `chirality` controls whether slot index advances in the
same direction as $s_{\rm frac}$ (+1) or opposite (-1).

### IQH rendering

In IQH plots the y-axis is **not inverted** — math y-up == visual y-up. Slot
conventions in the template:

| Slot | Cardinal | Position on ring |
|---|---|---|
| 0 (`SLOT_BUS`) | South | Bottom (visually) |
| $N_z/4$ (`SLOT_RIGHT`) | East | Right |
| $N_z/2$ (`SLOT_TOP`) | North | Top |
| $5 N_z/8$ (`SLOT_BOTTOM`) | South-ish | Bottom (V-link far DC) |
| $3 N_z/4$ (`SLOT_LEFT`) | West | Left |

For all sites — **IN, OUT, and bulk alike** — `_draw_ring` is called with
`phase_offset = 0.75, chirality = +1`. With this setting:

- Slot 0 lands at $s_{\rm frac} = 0.75$ = bottom-left corner ≈ visually bottom.
- Slot $N_z/4$ at $s_{\rm frac} = 0$ = right-edge bottom corner ≈ visually right.
- Slot $N_z/2$ at $s_{\rm frac} = 0.25$ = top-right corner ≈ visually top.
- Slot $3 N_z/4$ at $s_{\rm frac} = 0.5$ = top-left corner ≈ visually left.

Earlier code used `phase_offset = 0.25` for non-IN sites, which placed slot 0
at the top-right corner — a half-perimeter rotation that made the chiral edge
mode meander appear **inverted** on the bottom and right edges. Setting
`phase_offset = 0.75` for all sites fixes this so the meander reads correctly
as: site bright on the lattice-exterior side, link bright on the lattice-interior
side, alternating around the boundary.

For links:

- H-links: `phase_offset = 0.5, chirality = -1`. Slot 0 (= near DC, on the
  link's visually-left side facing the left site) lands at $s_{\rm frac} = 0.5$
  = top-left corner.
- V-links: `phase_offset = 0.75, chirality = -1`. Slot 0 (= near DC, on the
  link's visually-bottom side facing the bottom site) lands at the bottom-left
  corner.

The `chirality = -1` for links reflects the fact that two externally tangent
rings circulate with opposite chirality at the tangent point: site rings
circulate CCW and link rings circulate CW (or vice versa, depending on the
bus injection direction set by `dc_flip`).

### AQH rendering

AQH uses `ax.invert_yaxis()`: smaller plot_y is **visually upward**. This
inverts the relationship between math-orientation and visual-orientation.
Slot conventions:

| Slot | Cardinal | Direction in $(c, r)$ | Visual position |
|---|---|---|---|
| 0 (`SLOT_AQH_N`) | North | $(0, -1)$ smaller r | Top |
| $N_z/4$ (`SLOT_AQH_E`) | East | $(+1, 0)$ larger c | Right |
| $N_z/2$ (`SLOT_AQH_S`) | South | $(0, +1)$ larger r | Bottom |
| $3 N_z/4$ (`SLOT_AQH_W`) | West | $(-1, 0)$ smaller c | Left |

A subtlety arises from the y-inversion. To put slot 0 (= N) at visually-top,
we want it at small plot_y, which corresponds to $s_{\rm frac} \approx 0.845$
(the bottom edge of the perimeter parametrization in plot-data coordinates).
With `chirality = -1`, slot index advances in the $-s_{\rm frac}$ direction =
CCW visually under inversion — matching the bus injection physics, in which
light entering the bus from the left port causes the IN site's ring photon
to circulate CCW visually.

But there's a **second subtlety**: AQH sites come in two geometric flavours
based on which directions their link neighbors lie in:

- **V-sites**: link neighbors at N and/or S only. Their natural axis of light
  flow is vertical (top ↔ bottom).
- **H-sites**: link neighbors at E and/or W only. Their natural axis is
  horizontal (left ↔ right).

For both, the slot indexing 0(N) → $N_z/4$ (E) → $N_z/2$ (S) → $3N_z/4$ (W) is the same
in the template. But the renderer needs different `phase_offset` values to
align the bright halves with the physical DC positions:

| Site type | `phase_offset` | Slot 0 | Slot $N_z/4$ | Slot $N_z/2$ | Slot $3 N_z/4$ |
|---|---|---|---|---|---|
| V-site | 0.845 | top ✓ | left | bottom ✓ | right |
| H-site | 0.345 | bottom | right ✓ | top | left ✓ |

In V-sites, slot 0 (N) lands correctly at visually-top, and slot $N_z/2$ (S)
correctly at visually-bottom — the two slots that actually have link
connections. The unused E/W slots end up on visually-left/right (swapped from
what their labels suggest), but since no DC sits there for a V-site, this
doesn't matter visually.

In H-sites, slot $N_z/4$ (E) lands at visually-right and slot $3 N_z/4$ (W)
at visually-left — the two slots with link connections. The unused N/S slots
swap top/bottom, which again doesn't matter because no DC sits there.

Edge sites (with only one link neighbor) inherit the orientation from
whichever neighbor they have: N or S → V-orientation (`phase_offset = 0.845`),
E or W → H-orientation (`phase_offset = 0.345`).

For links, all use `phase_offset = 0.845, chirality = +1`. The +1 chirality is
opposite the sites' -1, again reflecting the tangent-circles rule (opposite
circulation between coupled rings).

### Visual diagnostic: the chiral edge mode

When the simulator sits at a frequency in the topological gap, with non-trivial
loss ($\alpha \gtrsim 0.05$) so the photon decays before completing a full edge
loop, the chiral edge mode becomes plainly visible as a sequence of bright
arcs hugging the boundary. The pattern is a **meander**:

- On the bottom edge: site rings bright on their bottom half, H-links between
  them bright on their top half (or vice versa, depending on `dc_flip`).
- On the right edge: site rings bright on their right half, V-links between
  them bright on their left half.
- And similarly on the top and left edges.

This pattern arises because the photon's instantaneous arc location reflects
which way it's propagating: at any DC, two externally tangent rings have
opposite-sense circulation, so the bright arcs alternate between
lattice-exterior (sites) and lattice-interior (links) as the photon traces the
boundary.

If the meander appears **inverted** (sites and links bright on the same side)
or **rotated**, suspect a `phase_offset` or `chirality` mismatch in the
renderer — the field data is generally correct (it's just the linear physics),
but visualizing it requires the right slot-to-position mapping.

---

## 18. Mapping back to experimental units

Everything in the simulator is in dimensionless units. To compare to a real device,
multiply by the right physical scale:

| Simulator quantity | Multiply by | Get |
|---|---|---|
| $\omega/(2\pi)$ | $\Gamma_{\rm FSR}^{\rm exp}$ | Detuning in Hz |
| $\kappa_{\rm ex}^2$ | $\Gamma_{\rm FSR}^{\rm exp}$ | Bus linewidth in Hz |
| $\kappa_J^2/(2\pi)$ | $\Gamma_{\rm FSR}^{\rm exp}$ | Hopping $J$ in Hz |
| $\alpha/(2\pi)$ | $\Gamma_{\rm FSR}^{\rm exp}$ | Intrinsic linewidth in Hz |
| $T_R = 1$ | $1/\Gamma_{\rm FSR}^{\rm exp}$ | Round-trip time in seconds |
| $1/\alpha$ | $T_R$ | Photon lifetime (round-trips, ÷ then × $T_R$ for sec) |

For a device with $\Gamma_{\rm FSR} = 750$ GHz and the simulator defaults
(matching the SiN platform of [§1](#1-setting)):
- $\kappa_J = 0.9$ → $J = \kappa_J^2 \Gamma_{\rm FSR}/(2\pi) \approx 96.7$ GHz
- $\kappa_{\rm ex} = 0.359 = \kappa_J / \sqrt{2\pi}$ → bus linewidth $\kappa_{\rm ex}^2 \Gamma_{\rm FSR} \approx 96.7$ GHz
  (matched to the hopping rate $J$, the "critical coupling to the lattice mode"
  condition; the factor $1/\sqrt{2\pi}$ accounts for the differing definitions
  of $\kappa$ vs $J$ in the two formulas)
- $\alpha = 0.01$ → intrinsic linewidth = 1.2 GHz, $Q_{\rm int} \approx 1.6 \times 10^5$
- $T_R = 1.33$ ps
- Photon lifetime $\sim 100 T_R = 133$ ps

---

## 19. What this simulator does NOT model

- **Nonlinearity (Kerr, FWM).** $R$ is linear and frequency-flat. Adding Kerr:
  rebuild $R$ each step with $|E_k|^2$-dependent extra phase. ~50 lines of code.
- **EO modulation, Floquet driving.** $R$ is time-independent. For sideband generation:
  switch to a frequency-comb basis where each grid point carries $(2P+1)$ amplitudes.
- **Group-velocity dispersion.** Within one FSR, $v_g$ is treated as constant. For
  multi-FSR effects (broadband combs): include $\beta_2$ in the per-step phase.
- **Spontaneous emission, vacuum noise.** Purely classical CW input.
- **Polarization.** Single-mode waveguide; one scalar amplitude per grid.
- **Pulse shape.** Source $\vec{s}$ is constant — CW excitation. For pulsed input,
  modulate $\vec{s}$ in time.
- **Backscattering between CW and CCW modes (single-pseudospin TMM).** Each ring
  carries one complex amplitude per grid point — only one pseudospin is simulated
  at a time. The `dc_flip` toggle ([§15](#15-per-ring-chirality-flip-dc-tangent-direction)) switches *which* pseudospin sector is
  simulated, but the two cannot coexist or couple. Modeling sidewall-roughness-induced
  backscattering would require **doubling the state space** to include both pseudospins
  simultaneously and adding cross-coupling at every DC.

All of these are extensions worth pursuing as subsequent layers. The linear classical
CW simulator is the foundation that everything builds on.

---

## Appendix: Variable cheat sheet

| Variable | Type | Meaning |
|---|---|---|
| $L_{\rm site}$ | dimensionless | Site ring length, set to 1 |
| $\eta$ | dimensionless (small) | Link extra length, $\sim 10^{-3}$ |
| $v_g$ | dimensionless | Group velocity, set to 1 |
| $T_R$ | dimensionless | Site round-trip time, set to 1 |
| $\Gamma_{\rm FSR}$ | dimensionless | FSR in Hz, set to 1 |
| $\Delta z$ | $1/N_z$ | Per-step length (and time) |
| $N_z^{\rm site}, N_z^{\rm link}$ | int | Discretization count per ring (site, link). Default 16, exposed as a UI spinbox; can be 16, 32, 48, 64. Site slot positions are at multiples of $N_z/16$ for IQH (BUS at 0, RIGHT at $N_z/4$, TOP at $N_z/2$, BOTTOM at $5 N_z/8$, LEFT at $3 N_z/4$) and $N_z/4$ for AQH (N at 0, E at $N_z/4$, S at $N_z/2$, W at $3 N_z/4$). |
| $\omega$ | dimensionless | Detuning from carrier (in $\omega$-units, not Hz) |
| $\omega/(2\pi)$ | dimensionless | Detuning in FSR units (axis label) |
| $p$ | complex | Per-step propagation factor |
| $\alpha$ | dimensionless | Field-loss per unit length |
| $\beta_0\eta$ | dimensionless | Static link anti-resonance phase, ideally $\pi$ |
| $\Phi_0$ | dimensionless | IQH synthetic flux per plaquette (Landau gauge) |
| $\kappa_{\rm ex}, \kappa_J$ | dimensionless | DC amplitude couplings, in $[0, 1]$ |
| $t_{\rm ex}, t_J$ | dimensionless | DC straight-through, $\sqrt{1-\kappa^2}$ |
| $\sigma$ | $\pm 1$ | Slot-ordering direction sign (+1 default, -1 if `dc_flip`) |
| $N_x, N_y$ | int | Lattice dimensions (sites per row/col for IQH; brick-wall extent for AQH) |
| $\vec{E}$ | complex array | State vector, length $N_{\rm state}$ |
| $R$ | sparse complex matrix | Propagator, size $N_{\rm state} \times N_{\rm state}$ |
| $\vec{s}$ | complex array | Source vector, length $N_{\rm state}$ |
| $\vec{E}_\infty$ | complex array | Steady-state field |
| $a_{\rm thru}, a_{\rm drop}$ | complex | Bus output amplitudes |

### Lattice-type-specific quantities

| Quantity | IQH | AQH |
|---|---|---|
| Site grid | $N_x \times N_y$ square | $(2 N_x - 1) \times (2 N_y - 1)$ brick-wall |
| Number of sites | $N_x N_y$ | $N_x(N_y - 1) + N_y(N_x - 1)$ |
| Number of links | $(N_x - 1) N_y + N_x (N_y - 1)$ | $(N_x - 1)(N_y - 1)$ |
| Site DC slots used | 0 (bus), $N_z/4$, $N_z/2$, $5 N_z/8$, $3 N_z/4$ | 0, $N_z/4$, $N_z/2$, $3 N_z/4$ (cardinal) |
| Link DC slots used | 0 (near), $N_z/2$ (far) | 0, $N_z/4$, $N_z/2$, $3 N_z/4$ (four neighbors) |
| Synthetic-gauge mechanism | Peierls phase $\theta = -2\Phi_0 i_y$ on H-links | Antipodal slot pairing in brick-wall geometry |
| Topological invariant | Chern number ($\propto \Phi_0$) | Anomalous winding number |
| Default IN bus position | site $(0, 0)$, slot 0 (south) | bottom-row site, slot $N_z/2$ (south) |
| Default OUT bus position | site $(0, N_y-1)$, slot $N_z/2$ (north) | top-row site, slot 0 (north) |
| Renderer phase_offset (sites) | 0.75 (all sites) | 0.845 (V-orientation), 0.345 (H-orientation) |
| Renderer chirality (sites) | +1 | -1 |
| Renderer phase_offset (links) | 0.5 (H), 0.75 (V) | 0.845 |
| Renderer chirality (links) | -1 | +1 |