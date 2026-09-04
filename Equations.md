Your colleague's suggestion — vorticity equation + horizontal divergence — is the classical relative vorticity framework (ζ = ∂v/∂x − ∂u/∂y), which is what pairs naturally with divergence (δ = ∂u/∂x + ∂v/∂y) in the vorticity-tendency equation. Download "vorticity" (relative), not "potential_vorticity", alongside "divergence".
Current residuals in loss.py

Geostrophic wind balance (compute_residual_geostrophic_wind, Holton eq. 6.58):
$$u_g = -\frac{1}{f_0}\frac{\partial \Phi}{\partial y}, \qquad v_g = \frac{1}{f_0}\frac{\partial \Phi}{\partial x}$$

$$R_{\text{geo}} = |u_g - u| + |v_g - v|$$

Penalizes disagreement between the model's predicted wind and the wind implied by geostrophic balance from its predicted geopotential.

QG potential vorticity conservation (compute_residual_planetary_vorticity, Holton eq. 6.65):
$$q = \frac{1}{f_0}\nabla^2\Phi + f + \frac{f_0}{\sigma}\frac{\partial^2\Phi}{\partial p^2}$$

$$R_{\text{PV}} = \frac{\partial q}{\partial t} + \mathbf{V_g}\cdot\nabla q \approx 0$$

Penalizes violation of QGPV conservation following the geostrophic wind. Because V_g is by construction non-divergent (it's a rotational wind derived from Φ), this equation structurally has no room for a divergence term — that's exactly the gap your colleague is pointing at.

Newly suggested residual: barotropic vorticity-divergence equation

Using the model's actual predicted u, v (not the geostrophic wind derived from Φ), so ageostrophic divergence can show up at all:

$$\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}, \qquad \delta = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

$$R_{\text{vort-div}} = \left|\frac{\partial \zeta}{\partial t} + u\frac{\partial \zeta}{\partial x} + v\frac{\partial \zeta}{\partial y} + (\zeta + f)\delta\right|$$

This is the horizontal (barotropic) vorticity equation: local vorticity change = advection by the actual wind + stretching of absolute vorticity (ζ+f) by horizontal divergence. It's the natural generalization of your current QG residual — same conservation-law structure (∂/∂t + advection ≈ 0), but now using the real wind and real divergence instead of the geostrophic, non-divergent approximation.

Implementation notes

ζ and δ both reuse self.gradient_helper.gradient_horizontal, already built for the geopotential gradients — no new gradient infrastructure needed.
You can compute ζ, δ two ways and cross-check them: (a) from the model's predicted u, v via finite differences, or (b) read directly from ERA5's native vorticity/divergence fields as ground truth. Comparing (a) vs (b) on real ERA5 data is a good sanity check before trusting the residual as a training signal.
This residual would slot into VorticityLoss.__init__'s residual_fns list exactly like compute_residual_geostrophic_wind does — PDE_loss.forward already loops over an arbitrary list of (fn, weight) pairs.

## Background for the horizontal vorticity equation

**Starting point: the horizontal momentum equations.**
On an isobaric surface, neglecting friction and vertical advection (i.e. treating the flow as a single "barotropic" layer), the horizontal momentum equations are (Holton eq. 4.?, same form used for $u_g, v_g$ above):

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} - fv = -\frac{\partial \Phi}{\partial x}$$

$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + fu = -\frac{\partial \Phi}{\partial y}$$

**Taking the curl.** The vorticity equation is obtained by cross-differentiating: $\partial/\partial x$ of the $v$-equation minus $\partial/\partial y$ of the $u$-equation. This eliminates $\Phi$ entirely, since $\partial^2\Phi/\partial x\partial y = \partial^2\Phi/\partial y\partial x$ — vorticity is a pressure-gradient-force-free diagnostic, which is exactly why it's a useful residual to add alongside the geostrophic and QGPV terms above (it probes a different, non-geopotential-derived constraint).

Working through the cross-differentiation term by term:

- The time-derivative terms combine directly: $\partial_x(\partial_t v) - \partial_y(\partial_t u) = \partial_t\zeta$.
- The advection terms combine, via the vector identity $k\cdot\nabla\times[(\mathbf{V}\cdot\nabla)\mathbf{V}] = \mathbf{V}\cdot\nabla\zeta + \zeta\delta$, into $u\partial_x\zeta + v\partial_y\zeta + \zeta\delta$.
- The Coriolis terms give $f\delta + v\,\partial f/\partial y$ (the $f\delta$ piece is the divergence acting on planetary vorticity; the $v\,\partial f/\partial y$ piece is meridional advection of $f$, i.e. the $\beta$-effect).

Putting these together:

$$\frac{\partial \zeta}{\partial t} + u\frac{\partial \zeta}{\partial x} + v\frac{\partial \zeta}{\partial y} + (\zeta+f)\delta + v\frac{\partial f}{\partial y} = 0$$

**Interpretation.** Writing absolute vorticity as $\eta = \zeta + f$, and noting $Df/Dt = v\,\partial f/\partial y$ since $f$ doesn't depend on $t$ or $x$, this is just the material conservation statement

$$\frac{D\eta}{Dt} = -\eta\delta \qquad\Longleftrightarrow\qquad \frac{D(\zeta+f)}{Dt} = -(\zeta+f)\delta$$

i.e. absolute vorticity changes following the flow only through stretching/shrinking of air columns by horizontal divergence — the same "spin up when squashed, spin down when stretched" mechanism as angular-momentum conservation. In the fully non-divergent (barotropic, $\delta=0$) limit this reduces to the classic $D\eta/Dt = 0$.

**Relation to $R_{\text{vort-div}}$ above.** The residual as written,

$$R_{\text{vort-div}} = \left|\frac{\partial \zeta}{\partial t} + u\frac{\partial \zeta}{\partial x} + v\frac{\partial \zeta}{\partial y} + (\zeta+f)\delta\right|,$$

keeps the relative-vorticity advection and the stretching term, but drops the $\beta$-term $v\,\partial f/\partial y$. Exactly, this residual equals $|{-v\,\partial f/\partial y}|$ rather than zero — a small but nonzero quantity except right at the equator or wherever $v\approx0$. Whether that's acceptable depends on the domain: over a limited midlatitude patch it's a minor systematic offset (an f-plane-style simplification of the advection term while still retaining $f$ itself in the stretching term); over a domain spanning a wide range of latitudes it's worth either (a) adding $v\,\partial f/\partial y$ explicitly (cheap — $\partial f/\partial y = \beta$ is a known constant per latitude on the sphere/beta-plane), or (b) replacing $\zeta$ by $\eta=\zeta+f$ throughout and using $D\eta/Dt + \eta\delta$ as the residual, which absorbs the $\beta$-term automatically.

## Geostrophic-wind residual vs. vorticity-divergence residual: what's actually different

Both residuals are physics-informed loss terms of the same generic shape — `deviation = |physical_law(model outputs)|` — but they check the model's outputs against the governing equations at two different levels of approximation. Writing them with a shared notation makes the difference explicit.

**Common setup.** The network predicts fields $u, v, \Phi$ (and their time evolution) on a lat/lon-pressure grid. $f = f(y)$ is the (known, not predicted) Coriolis parameter. Define the momentum equations the atmosphere actually obeys (no approximation yet):

$$\frac{Du}{Dt} - fv = -\frac{\partial \Phi}{\partial x}, \qquad \frac{Dv}{Dt} + fu = -\frac{\partial \Phi}{\partial y}, \qquad \frac{D}{Dt} \equiv \frac{\partial}{\partial t} + u\frac{\partial}{\partial x} + v\frac{\partial}{\partial y}$$

Both residuals are simplifications of this same pair of equations — they just simplify it differently, and that difference is exactly what makes them complementary rather than redundant.

**$R_{\text{geo}}$: drop the acceleration term entirely (diagnostic / zeroth-order balance).**
Geostrophic balance is the leading-order truncation obtained by assuming $D\mathbf{V}/Dt \approx 0$ — i.e. that the flow is in a steady, non-accelerating state where the Coriolis force exactly balances the pressure-gradient force at every instant:

$$\underbrace{\frac{D\mathbf{V}}{Dt}}_{\text{assumed} \approx 0} + f\mathbf{k}\times\mathbf{V} = -\nabla\Phi \quad\Longrightarrow\quad u_g = -\frac{1}{f_0}\frac{\partial \Phi}{\partial y}, \quad v_g = \frac{1}{f_0}\frac{\partial \Phi}{\partial x}$$

$$R_{\text{geo}} = |u_g - u| + |v_g - v|$$

This is an **algebraic, instantaneous, single-timestep constraint** — no time derivative, no advection, just "does the wind the model predicted match the wind implied by the pressure field it predicted, right now." It also uses a constant reference $f_0$ rather than the true latitude-varying $f(y)$ (the $f_0/f$ substitution is standard for the QG scaling used here). In ML terms it's closest to a **pointwise consistency loss between two output channels** ($u,v$ vs. $\Phi$) — it never looks across timesteps or across grid cells' dynamics, only across the model's own outputs at one instant.

**$R_{\text{vort-div}}$: keep the acceleration term, but only its rotational part (prognostic / evolution constraint).**
The vorticity-divergence residual does *not* assume $D\mathbf{V}/Dt \approx 0$. Instead it takes the curl of the *full, un-truncated* momentum equation above (Section "Background," this eliminates $\Phi$ rather than eliminating $D\mathbf{V}/Dt$):

$$\mathbf{k}\cdot\nabla\times\left[\frac{D\mathbf{V}}{Dt} + f\mathbf{k}\times\mathbf{V}\right] = \mathbf{k}\cdot\nabla\times(-\nabla\Phi) = 0 \quad\Longrightarrow\quad \frac{\partial \zeta}{\partial t} + u\frac{\partial \zeta}{\partial x} + v\frac{\partial \zeta}{\partial y} + (\zeta+f)\delta \;(+\, v\tfrac{\partial f}{\partial y}) = 0$$

$$R_{\text{vort-div}} = \left|\frac{\partial \zeta}{\partial t} + u\frac{\partial \zeta}{\partial x} + v\frac{\partial \zeta}{\partial y} + (\zeta+f)\delta\right|$$

This is a **differential, multi-timestep, evolution constraint**: it needs $\partial\zeta/\partial t$, so it requires the model's predictions at (at least) two adjacent timesteps, and it checks whether the *predicted trajectory* is dynamically consistent, not just whether a single frame is self-consistent. Its only approximations are (i) dropping friction and vertical (cross-isobaric) advection, and, as currently written, (ii) dropping the $\beta$-term $v\,\partial f/\partial y$ — it makes **no** balance assumption and **no** $f\to f_0$ substitution, so it is strictly a weaker set of assumptions than $R_{\text{geo}}$, applied to a full-fledged evolution equation instead of an instantaneous algebraic one.

**Side-by-side summary**

| | $R_{\text{geo}}$ | $R_{\text{vort-div}}$ |
|---|---|---|
| Derived by | truncating $D\mathbf{V}/Dt \to 0$ | taking $\mathbf{k}\cdot\nabla\times(\cdot)$ of the full momentum eq. |
| Temporal scope | single timestep (algebraic) | $\ge 2$ timesteps (needs $\partial_t$) |
| What it compares | model's $u,v$ channel vs. model's $\Phi$ channel | model's predicted vorticity evolution vs. wind-implied advection/stretching |
| Divergence ($\delta$) | structurally absent ($\mathbf{V}_g$ is non-divergent by construction) | present, as the stretching term $(\zeta+f)\delta$ |
| Coriolis treatment | constant $f_0$ | full $f(y)$, minus the $\beta$-advection term $v\,\partial f/\partial y$ |
| Failure mode it catches | ageostrophic-looking wind (model's $u,v$ disagree with its own $\Phi$) | dynamically inconsistent *evolution* (model's vorticity tendency disagrees with advection + stretching it itself implies) |

The practical upshot for training: $R_{\text{geo}}$ penalizes the model for predicting $u,v$ that don't match its own $\Phi$ at one instant, which mechanically pushes divergence toward zero (any $\delta\neq0$ shows up as $u,v$ deviating from the non-divergent $\mathbf{V}_g$). $R_{\text{vort-div}}$ is the first residual in this set that actually *uses* $\delta$ as a first-class term rather than implicitly suppressing it, and the first to constrain the model across time rather than within a single frame — which is why it's a genuine addition rather than a reweighting of the existing terms.

### Plot to be added in the paper
nstead, compute the target the same way you compute the prediction — apply gradient_helper.gradient_horizontal to ERA5's ground-truth u, v fields (which you already have) to get ζ_ERA5-FD, δ_ERA5-FD. This gives an apples-to-apples comparison using identical numerics on both sides.
Native fields as an independent diagnostic (good for the paper, not the loss): separately compare ζ_ERA5-FD (finite-difference, from case 1) against the true native vorticity/divergence you'd download. The gap between these quantifies your grid's finite-difference truncation error against ECMWF's spectral estimate — this is actually a nice, cheap analysis figure for the paper (shows how much of any residual disagreement is discretization artifact vs. genuine model error), and it's why downloading the native fields is worth doing even though you don't strictly need them for the loss.