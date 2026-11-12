# Bugs in the horizontal derivatives (`GradientHelper`)

The colleague's suspicion is correct. The geostrophic wind residual $\mathcal{R}_{geo}$ is far too large, about 25 m/s instead of a few m/s, and the cause is a scaling error. The error is not specific to $\mathcal{R}_{geo}$. It sits in the finite-difference helper that every residual uses to compute horizontal derivatives, so $\mathcal{R}_{PV}$ and the vorticity–divergence residual are affected too.

> **Status (2026-11-05): fixed.** See [section 7](#7-status-of-the-fix) for what changed and the before/after numbers on ERA5.

There are three bugs:

1. Every horizontal derivative is too small by a factor of 2.
2. The north–south derivative has the wrong sign.
3. The north–south derivative is forced to exactly 0 on the northern and southern edge rows.

---

## 1. How `GradientHelper` works

The code is in `src/pde_diff/utils.py` (`GradientHelper`), and the grid is set up in `src/pde_diff/loss.py` (`VorticityLoss.make_distance_grids`).

### Data layout

A field such as the geopotential $\Phi$ is stored as a 4-D tensor:

```
(batch, level, lon, lat)  =  (B, 3, 480, 32)
```

- **`lon`** has 480 points, spaced 0.75° and covering the full circle, so it is **periodic**.
- **`lat`** has 32 points, spaced 0.75° and covering 70°N → 46.75°N. **Index 0 is the northernmost row.** ERA5 stores latitude from north to south.

### Derivatives as a convolution

A derivative on a grid is approximated with a **central difference**:

$$
\frac{\partial f}{\partial x}\Big|_i \approx \frac{f_{i+1} - f_{i-1}}{x_{i+1} - x_{i-1}}
$$

That is the difference between the two neighbours, divided by the distance between those two neighbours. `GradientHelper` computes this in three steps:

1. **Pad** the array by one cell on each side, so the first and last points also have two neighbours.
   - In x (longitude) the padding is `circular`: the point before 0° is 359.25°, which is correct for a periodic direction.
   - In y (latitude) the padding is `reflect`, which mirrors the array at the edge. The padded value before row 0 is row 1.
2. **Convolve** with the stencil `[-1, 0, 1] / 2`. A convolution slides this small kernel along the axis and computes $\tfrac12(f_{i+1} - f_{i-1})$ at every point.
3. **Divide** by a precomputed grid-distance array, `dx` or `dy`, which has the same shape as the field. Each entry is the distance in metres used at that point.

Second derivatives work the same way with the stencil `[1, -2, 1]`, divided by `dx²` or `dy²`.

### Where `dx` and `dy` come from

`make_distance_grids` builds a lon/lat coordinate grid. For each point it computes the great-circle (**haversine**) distance between that point's two neighbours: for `dx`, the points at i−1 and i+1 along the latitude circle; for `dy`, the points at j−1 and j+1 along the meridian. The neighbours are found with `torch.roll`.

The haversine distance between two points on the same latitude circle is $R\cos\varphi\,\Delta\lambda$, so the convergence of meridians, the $\cos\varphi$ factor, **is** included correctly.

---

## 2. The bugs

| # | Topic | What it looked like before | What was wrong | Effect | How to fix |
|---|---|---|---|---|---|
| 1 | **Grid spacing counted twice** (all horizontal derivatives) | Stencil `[-1, 0, 1] / 2`, divided by `dx` = haversine distance between the neighbours i−1 and i+1. The same applies to `dy`. | The neighbours are **two** cells apart, so `dx` already equals $2\Delta x$, and the stencil divides by 2 again. The result is $\dfrac{f_{i+1}-f_{i-1}}{4\Delta x}$ instead of $\dfrac{f_{i+1}-f_{i-1}}{2\Delta x}$. Second derivatives use `[1,-2,1]/dx²` with the same 2-cell `dx`, so they are divided by $4\Delta x^2$ instead of $\Delta x^2$. | Every first derivative ($\partial/\partial x$, $\partial/\partial y$) is **½ of the true value**. Every second derivative, and therefore $\nabla^2$, is **¼ of the true value**. Measured on test fields: ratio 0.500 for $\partial/\partial x$ at all latitudes, and 0.255 for $\partial^2/\partial x^2$. | Use the distance of **one** cell: $dx = R\cos\varphi\,\Delta\lambda$ and $dy = R\,\Delta\varphi$. Keep the `/2` in the stencil. Equivalently, keep the 2-cell distances and drop the `/2`, but then the second derivatives also need fixing. Single-cell spacing fixes both. |
| 2 | **Wrong sign of the north–south derivative** | `lat = linspace(70, 46, 32)`, decreasing with index. `dy` = haversine distance, which is always positive. | The stencil computes $f_{j+1}-f_{j-1}$, and moving from j−1 to j+1 is moving **south**. Since `dy` is positive, the result is $\partial f/\partial(\text{south}) = -\partial f/\partial y$, where y points north as the equations assume. | Every $\partial/\partial y$ has the **wrong sign**, on top of the factor ½ from bug 1. In $\mathcal{R}_{geo}$, $u_g = -f_0^{-1}\partial\Phi/\partial y$ comes out as $-\tfrac12 u_g$. $v_g = f_0^{-1}\partial\Phi/\partial x$ only uses x, so it comes out as $+\tfrac12 v_g$. That is why the u and v residuals behave differently. Second derivatives are not affected by the sign, because $(-1)^2 = 1$. | Use a **signed** spacing, $dy_j = R\,(\varphi_{j+1}-\varphi_j)$ in radians. It is negative for a north-to-south grid, which fixes the sign automatically. Alternatively, `torch.gradient(f, spacing=(lat_radians * R,), dim=-1)` takes the actual coordinates and handles the sign itself. |
| 3 | **North–south derivative forced to 0 on the edge rows** | `d_in_dy` pads latitude with `mode='reflect'`. At the edges, `dy` comes from `torch.roll`, which wraps around. | With reflect padding, the value placed before row 0 is a copy of row 1, so at row 0 the stencil computes $f_1 - f_1 = 0$. The same happens at row 31. The `roll` also makes the edge `dy` the distance from 70° to 46°, about 2,600 km. | On 2 of the 32 latitude rows, i.e. **6.25 % of grid points**, $\partial\Phi/\partial y = 0$ exactly, so $u_g = 0$ exactly. Second derivatives in y at the edges are also wrong, because of the wrap-around `dy`. | Use **one-sided differences** at the two edge rows, e.g. `torch.gradient(..., edge_order=2)`, which also takes care of bug 2. Alternatively, leave the edge rows out of the residuals. Keep the circular padding in x, which is correct. *(Both are now done: one-sided stencils for the derivatives, and the edge rows left out of the residuals; see section 12.)* |

---

## 3. Smaller issues found at the same time

These are not part of the three bugs, but they are worth fixing in the same change.

| Topic | Before | Problem | Fix |
|---|---|---|---|
| Hard-coded grid coordinates | `lat = linspace(70, 46, 32)`, `lon = linspace(0, 359, 480)` | The real grid is 70°→46.75° in 0.75° steps, but the hard-coded latitudes are spaced 0.774°, about 3 % off. The hard-coded longitude spacing is 0.7495° with a 1° gap where 359° wraps to 0°, so `dx` in the first and last longitude columns is 2.33 cells instead of 2. | Read `grid_lat` and `grid_lon` from the dataset metadata. They are already saved in the precomputed `metadata.json`. |
| Constant $f_0$ in $\mathcal{R}_{geo}$ | $u_g = -f_0^{-1}\partial\Phi/\partial y$ with $f_0$ at about 58° | Across 46–70°N, $f$ varies by about ±13 % around $f_0$. Even perfectly geostrophic ERA5 data therefore has a nonzero $\mathcal{R}_{geo}$ of up to about 13 % of the wind. That is the same order as the real ageostrophic wind of 10–20 %. | Use $f(\varphi) = 2\Omega\sin\varphi$ in $\mathcal{R}_{geo}$, which is a pure balance check. Keep $f_0$ inside $\mathcal{R}_{PV}$, where quasi-geostrophic theory defines $\mathbf{V}_g$ with $f_0$. **Decision: keep $f_0$ in $\mathcal{R}_{geo}$ too.** See section 7. |

---

## 4. The bugs explain the histograms

Suppose ERA5 were perfectly geostrophic, so that $u_g = u$ and $v_g = v$. The buggy code would compute $u_g' = -\tfrac12 u$ and $v_g' = +\tfrac12 v$. The relative residuals plotted are $|u_g'-u| / (|u_g'-u| + |u_g'|)$ and the same for v:

| Quantity | Predicted from the bugs | Observed on ERA5 |
|---|---|---|
| u relative residual | $\tfrac{1.5}{1.5+0.5} = 0.75$ | **0.755** (narrow peak) |
| v relative residual | $\tfrac{0.5}{0.5+0.5} = 0.50$ | **0.493** (broad peak) |
| $\mathcal{R}_{geo}$ magnitude | $\approx 1.5\lvert u\rvert + 0.5\lvert v\rvert \approx$ 25 m/s for a midlatitude jet | **24.6 m/s** |
| Spike at exactly 1.0 in the u residual | $u_g' = 0$ on 2 of 32 rows (bug 3), giving $\lvert u\rvert/(\lvert u\rvert+0) = 1$ on **6.25 %** of points | about 640 of 10,000 values = **6.4 %** |

Notes:

- **The spike at 1.0 is not caused by clipping.** It comes from bug 3. Only u shows it, because only the y-derivative uses reflect padding.
- **The v residual is not explained by a missing $\cos\varphi$.** The x-spacing already contains $\cos\varphi$ through the haversine distance, and the measured x-derivative error is exactly ½ at every latitude. A missing $\cos\varphi$ would give an error that varies with latitude. The value $1-\cos 58° \approx 0.47$ is a coincidence; the factor ½ predicts 0.50, which is closer to the observed 0.493.
- **v is broad and u is narrow** because u at 450–550 hPa is dominated by the large, nearly geostrophic westerly flow, so it stays near the predicted 0.75. v is smaller, so the real ageostrophic part and the $f_0$ error are a much larger fraction of it, and its values spread around 0.5.

---

## 5. Effect on each residual

| Residual | Term in the equation | What the buggy code actually computed |
|---|---|---|
| $\mathcal{R}_{geo}$ | $f_0^{-1}\mathbf{k}\times\nabla_h\Phi = (u_g, v_g)$ | $(-\tfrac12 u_g,\ +\tfrac12 v_g)$, with $u_g = 0$ on the edge rows. Even perfect geostrophic balance gives a large residual. |
| $\mathcal{R}_{PV}$ | $\tfrac{1}{f_0}\nabla^2\Phi$ (relative-vorticity part of $q_{QG}$) | $\tfrac14$ of the true value, so it is too small compared with $f$ and the vertical term. The vertical derivatives in p are correct. |
| $\mathcal{R}_{PV}$ | advection $\mathbf{V}_g\cdot\nabla_h q$ | $u_g'\,\partial_x q' = (-\tfrac12)(\tfrac12) = -\tfrac14$ and $v_g'\,\partial_y q' = (\tfrac12)(-\tfrac12) = -\tfrac14$. The advection term has the **wrong sign** and ¼ of the true size. |
| Vorticity–divergence | $\zeta = \partial_x v - \partial_y u$, $\ \delta = \partial_x u + \partial_y v$ | $\zeta' = \tfrac12(\partial_x v + \partial_y u)$ and $\delta' = \tfrac12(\partial_x u - \partial_y v)$. These are not vorticity and divergence, and the $v\,\partial_y\zeta$ advection term also has the wrong sign. |

**Consequences**

- The normalisation statistics in `src/pde_diff/data/residual_stats.json` were computed with the buggy code and must be recomputed.
- All earlier residual numbers and histograms, and all models trained with a nonzero `c_residual`, were affected.
- $\mathcal{R}_{PV}$ deserves special attention. In a recent test run its gradient was about 10× larger than the data term's, and it was pushing the model towards PV advection with the wrong sign.

---

## 6. Verification after the fix

1. **Unit test with analytic fields.**
   - Take $\Phi = A\,y_{north}$. The test must give $\partial\Phi/\partial y = A$, with the correct sign, on every row including the edges.
   - Take $\Phi = \sin\lambda$. The test must give $\partial\Phi/\partial x = \cos\lambda / (R\cos\varphi)$ at every latitude.
   - The same kind of check applies to $\nabla^2$.
2. **Recompute `residual_stats.json`**, and re-plot the histograms. For the histograms, sample randomly across all timesteps; the current plots only show the first timestep.
3. **Expected result.** On ERA5, $\mathcal{R}_{geo}$ should drop to a few m/s, and the u and v relative residuals should be concentrated near small values with a long tail.

---

## 7. Status of the fix

**What changed**

- **`GradientHelper`** (`src/pde_diff/utils.py`) is now built from the latitude/longitude coordinates.
  - It uses one-cell spacings: $dx = R\cos\varphi\,\Delta\lambda$, and a **signed** $dy = R\,\Delta\varphi$. Because the spacing is signed, $\partial/\partial y$ points north even though the data is stored north to south.
  - In x it uses a periodic central difference.
  - In y it uses a central difference in the interior and **second-order one-sided** stencils on the two edge rows.
  - It checks that the grid is uniform, and that longitude covers the full circle.
- **`VorticityLoss.set_grid(grid_lat, grid_lon)`** (`src/pde_diff/loss.py`) replaces the hard-coded `linspace` grid and the haversine distances.
  - The defaults match the real training grid: 70°N → 46.75°N and −180° → 179.25°E, in 0.75° steps.
  - Training, the eval residual evaluator and the histogram script pass the dataset's own `grid_lat`/`grid_lon`.
- **$\mathcal{R}_{geo}$ uses the constant $f_0$, on purpose.** $\mathcal{R}_{PV}$ and $\mathcal{R}_{geo}$ come from the same QG equation set and share the definition $\mathbf{V}_g = f_0^{-1}\mathbf{k}\times\nabla_h\Phi$. $\mathcal{R}_{geo}$ enforces $\mathbf{V}\approx\mathbf{V}_g$ for exactly the $\mathbf{V}_g$ that advects $q$ in $\mathcal{R}_{PV}$. Using $f(\varphi)$ in only one of the two would make the constraints inconsistent.
  - The cost of $f_0$ is described at the end of this section.
- **Tests:** `tests/test_gradients.py` contains analytic tests:
  - linear and quadratic fields in y, exact on every row including the edges;
  - $\sin\lambda$ in x, checking periodicity and the $\cos\varphi$ scaling;
  - vorticity of a streamfunction flow equal to $\nabla^2\psi$;
  - $\mathcal{R}_{geo} \approx 0$ for an exactly geostrophic wind, and $= 2\lvert u_g\rvert$ for a reversed wind.
- **Histograms:** `plot_data_distribution.py` now samples randomly across all timesteps, instead of taking the first 10,000 values.
  - It also plots the vector form of the relative residual, $|\mathbf{V}-\mathbf{V}_g| / (|\mathbf{V}-\mathbf{V}_g| + |\mathbf{V}_g|)$, as an additional figure. The u and v component plots are kept.
  - It writes a balance index per residual to `reports/figures/era5_residual_balance.json`. The index is $B = \text{mean}\lvert R\rvert / \text{mean}\sum_k\lvert\text{term}_k\rvert$, between 0 and 1: 0 means the terms of the equation cancel exactly, 1 means they don't cancel at all.
- **$\mathcal{R}_{geo}$ normalisation:** the residual is now only **divided** by its ERA5 std; the mean is no longer subtracted. $\mathcal{R}_{geo} \ge 0$ and its target is 0. Subtracting the mean made the loss push the ageostrophic wind towards the ERA5 average of about 4 m/s instead of towards balance. $\mathcal{R}_{PV}$ and $\mathcal{R}_{vd}$ have means of about 0, so they are still standardised as before.

**Before and after on ERA5**

These are raw (unnormalised) residuals on 8 random 2016 timesteps, all levels and all grid points:

| Quantity | Before | After |
|---|---|---|
| $\mathcal{R}_{geo}$ mean (median) | 21.6 (18.8) m/s | **4.3 (3.3) m/s** |
| u relative residual, median | 0.76 | **0.15** |
| v relative residual, median | 0.49 | **0.16** |
| u relative residual, fraction exactly 1 | 6.3 % | **0.03 %** |
| $\mathcal{R}_{PV}$ mean $\lvert\cdot\rvert$ | 2.4e-9 | 1.1e-8 (the advection and Laplacian terms are now at full size) |
| $\mathcal{R}_{vd}$ mean $\lvert\cdot\rvert$ | 2.6e-9 | 2.8e-9 |

After the fix, u and v behave the same way, and $\mathcal{R}_{geo}$ is a few m/s, which is consistent with a 10–20 % ageostrophic wind.

**Known limitation of $f_0$ in $\mathcal{R}_{geo}$: comparison with $f(\varphi)$** (24 random 2016 timesteps, averaged over longitude, levels and time)

| Latitude | $\mathcal{R}_{geo}$ with $f_0$ (m/s) | $\mathcal{R}_{geo}$ with $f(\varphi)$ (m/s) | $\langle u-u_g\rangle$ with $f_0$ | $\langle u-u_g\rangle$ with $f(\varphi)$ |
|---|---|---|---|---|
| 70.0° | 4.42 | 4.20 | −0.47 | −0.22 |
| 61.0° | 4.58 | 4.52 | −0.31 | −0.35 |
| 58.75° (≈ $f_0$) | 4.40 | 4.39 | −0.48 | −0.48 |
| 52.0° | 3.83 | 4.23 | +0.21 | −0.38 |
| 46.75° | 5.56 | 5.81 | **+1.42** | **−1.04** |
| **All** | 4.21 | 4.29 | | |

- The domain-mean $\mathcal{R}_{geo}$ barely changes. It is slightly *higher* with $f(\varphi)$, but that does not mean $f_0$ is better.
- **With $f_0$**, the mean ageostrophic zonal wind $\langle u-u_g\rangle$ drifts steadily from −0.5 m/s in the north to +1.4 m/s in the south. That latitude-dependent bias is exactly what a wrong Coriolis parameter produces: $f_0$ is too small in the north and too large in the south.
- **With $f(\varphi)$**, the bias has the same sign everywhere: the real wind is slightly weaker than geostrophic, $u < u_g$. That is the physically expected sub-geostrophic flow around the cyclonic polar vortex (the gradient-wind effect).
- With $f_0$, the too-small $u_g$ in the south happened to partly cancel this real sub-geostrophy. That cancellation is why the $f_0$ residual looked slightly smaller there.
- **Conclusion:** QG assumes small variations of $f$ across the domain, but over 46–70°N $f$ varies by about ±13 %. Under the QG definition, ERA5 therefore carries a latitude-dependent ageostrophic bias of about ±1 m/s, and $\mathcal{R}_{geo}$ will push the model gently towards it. This is small compared with the typical $\mathcal{R}_{geo}$ of about 4 m/s, and it is accepted for consistency with $\mathcal{R}_{PV}$ and with earlier work. It should be named as a limitation.

The vector relative residual and the balance index are explained in [section 8](#8-diagnostics-relative-residuals-and-balance-index).

**Still to do**

- ~~Recompute `src/pde_diff/data/residual_stats.json`~~ Done, on the full training set (2015–2022): $\mathcal{R}_{geo}$ = 4.17 ± 3.47 m/s, $\mathcal{R}_{PV}$ = (−0.03 ± 2.97)·10⁻⁸, $\mathcal{R}_{vd}$ = (0.003 ± 5.21)·10⁻⁹.
- Re-run evaluation rollouts for any model before comparing R1–R3. Saved rollout metrics were computed with the buggy residuals.
- $\mathcal{R}_{geo}$ keeps $f_0$, the QG definition. `residual_stats.json` (computed with $f_0$) therefore stays valid.
- Investigate the weak balance of $\mathcal{R}_{PV}$ (see section 8.3) before relying on it as a loss term.

---

## 8. Diagnostics: relative residuals and balance index

All diagnostics are computed from **raw (unnormalised)** residuals, in `src/pde_diff/loss.py`. The histograms and statistics are produced by `src/pde_diff/data/plot_data_distribution.py`.

### 8.1 Relative geostrophic residual

The raw residual $\mathcal{R}_{geo} = |u_g-u| + |v_g-v|$ is in m/s. A given error is easier to judge relative to the size of the wind: 3 m/s is negligible in a 40 m/s jet but large in a 3 m/s calm. The relative residuals therefore divide the ageostrophic wind $a$ by itself plus the geostrophic wind $g$:

$$
r = \frac{a}{a+g} = \frac{\varepsilon}{1+\varepsilon}, \qquad \varepsilon = \frac{a}{g}.
$$

- $\varepsilon$ is the ageostrophic wind as a fraction of the geostrophic wind.
- $r$ is always between 0 and 1. For small values $r \approx \varepsilon$: $\varepsilon = 0.1 \to r = 0.09$, and $\varepsilon = 0.2 \to r = 0.17$.
- $r = 0.5$ means the ageostrophic wind is as large as the geostrophic wind, and $r = 1$ means $g = 0$.
- The plain ratio $a/g$ is not used, because it has no upper bound and blows up where $g \to 0$.

There are two versions. The code is `compute_residual_geostrophic_wind(..., normalize=False, relative=...)`.

| Version | `relative=` | $a$ | $g$ | Output |
|---|---|---|---|---|
| Component (two plots) | `True` | $\lvert u_g-u\rvert$, and separately $\lvert v_g-v\rvert$ | $\lvert u_g\rvert$, and separately $\lvert v_g\rvert$ | one value per component, per grid point and level |
| Vector (one plot) | `"vector"` | $\lvert\mathbf{V}_g-\mathbf{V}\rvert = \sqrt{(u_g-u)^2+(v_g-v)^2}$ | $\lvert\mathbf{V}_g\rvert = \sqrt{u_g^2+v_g^2}$ | one value per grid point and level |

Here $\mathbf{V}_g = f_0^{-1}\,\mathbf{k}\times\nabla_h\Phi$, the QG definition. A small constant (10⁻¹⁰) is added to the denominator to avoid dividing by zero.

**Why the two versions give different shapes.**

- **Near 0.**
  - A single component $\lvert u_g - u\rvert$ is close to 0 whenever that one component happens to be balanced. Values near 0 are common, so the component histograms are highest at 0, like a half-normal.
  - The vector magnitude $\lvert\mathbf{V}_g-\mathbf{V}\rvert$ is close to 0 only if **both** components are balanced at the same point, which is rare. So the vector histogram starts near 0, rises to a mode around 0.12 and then falls. This is the typical shape of the length of a 2-D random vector (Rayleigh-like).
- **Near 1.**
  - A single component $u_g$ crosses zero wherever the flow is locally meridional (and $v_g$ wherever it is zonal). There $g \to 0$ and $r \to 1$, however small the ageostrophic wind is. This produces the long tails of the component plots.
  - The full $\lvert\mathbf{V}_g\rvert$ is almost never close to zero in the midlatitude westerlies, so the vector form has a much lighter tail. Its std is 0.11, compared with 0.18–0.19 for the components.
- **Typical size.** The medians agree: 0.149 for the vector form against 0.15–0.16 for the components. Both measure the same typical ageostrophy, $\varepsilon \approx 0.18$.

The vector form is the more robust summary. The component plots remain useful for spotting direction-dependent errors, which is how the original bugs were found.

### 8.2 Balance index

Each residual is a sum of signed terms that should cancel if the equation holds:

| Residual | Terms (`return_terms=True`) | Residual built from the terms |
|---|---|---|
| $\mathcal{R}_{PV}$ | $T_1 = \partial_t q$ (1 h difference), $T_2 = u_g\,\partial_x q$, $T_3 = v_g\,\partial_y q$ | $\lvert T_1+T_2+T_3\rvert$ |
| $\mathcal{R}_{geo}$ | group u: $(u_g,\,-u)$; group v: $(v_g,\,-v)$ | $\lvert u_g-u\rvert + \lvert v_g-v\rvert$ |
| $\mathcal{R}_{vd}$ | $\partial_t\zeta$, $\ u\,\partial_x\zeta$, $\ v\,\partial_y\zeta$, $\ (\zeta+f)\,\delta$ | $\lvert\text{sum}\rvert$ |

The **balance index** compares the size of the residual with the size of the terms it is made of:

$$
B = \frac{\sum_{\text{points}} \sum_{\text{groups}} \big\lvert \sum_k T_k \big\rvert}{\sum_{\text{points}} \sum_{\text{groups}} \sum_k \lvert T_k \rvert}.
$$

- The sums run over all grid points, levels and timesteps. They are accumulated over the whole dataset, so $B$ is a ratio of two totals, not the mean of per-point ratios. This keeps it from being dominated by points where all terms are tiny.
- By the triangle inequality, $0 \le B \le 1$:
  - **$B = 0$:** the terms cancel exactly, so the equation holds.
  - **$B = 1$:** the terms all have the same sign and nothing cancels.
- **Rough reference point.** If the $n$ terms were independent, zero-mean Gaussians of equal size (no physical relationship at all), then $B = 1/\sqrt{n}$. That gives 0.71 for $n=2$, 0.58 for $n=3$ and 0.50 for $n=4$.
- The code is `balance_ratio()` in `loss.py`. The results are saved to `reports/figures/era5_residual_balance.json`.

### 8.3 Results on ERA5 (training set 2015–2022)

| Residual | $B$ | No-relationship reference | mean $\lvert R\rvert$ | mean $\sum\lvert T_k\rvert$ |
|---|---|---|---|---|
| $\mathcal{R}_{geo}$ | **0.095** | 0.71 (2 terms per group) | 4.17 m/s | 44.1 m/s |
| $\mathcal{R}_{vd}$ | **0.231** | 0.50 (4 terms) | 2.8·10⁻⁹ s⁻² | 1.2·10⁻⁸ s⁻² |
| $\mathcal{R}_{PV}$ | **0.525** | 0.58 (3 terms) | 1.1·10⁻⁸ | 2.1·10⁻⁸ |

**Geostrophic wind.** ERA5 satisfies it well. The residual is about 10 % of the size of its terms. This is expected, since geostrophic balance is the leading-order balance at midlatitudes. It is the most reliable physics constraint of the three.

**Vorticity–divergence.** Reasonably satisfied: clearly better than the no-relationship reference, but not tight. Some imbalance is expected, because the equation omits terms that ERA5 contains: vertical advection, tilting, solenoidal terms and friction.

**PV.** Barely satisfied: 0.525 is close to the 0.58 reference for three unrelated terms. Earlier, the correlation between $\partial_t q$ and the advection term was found to be only −0.2. Plausible reasons:

1. QG PV conservation assumes adiabatic, frictionless quasi-geostrophic flow.
2. $q$ contains second derivatives of $\Phi$, both horizontal ($\nabla^2$ on a 0.75° grid) and vertical (from only 3 levels, 50 hPa apart). Second derivatives amplify small-scale noise.
3. $\partial_t q$ is a 1 h difference centred at $t-\tfrac12$ h, but the advection term is evaluated at $t$ only. The time-centred version, averaging the advection over the previous and current states, may cancel better. The same applies to $\mathcal{R}_{vd}$.

This matters for training. In the first test run, $\mathcal{R}_{PV}$ supplied the largest physics gradient, roughly 10× the data term, yet ERA5 itself hardly satisfies it. It should be improved, for example with item 3 above, or down-weighted before it is relied on.

---

## 9. The log-likelihood clamp in the physics loss

### 9.1 What it does

The physics term is a Gaussian negative log-likelihood of the normalised residual $r$, with the diffusion posterior variance $\Sigma_t$ as its variance (Bastek et al., ICLR 2025):

$$
\mathcal{L}_R = c\cdot\tfrac12\, r^2/\Sigma_t .
$$

`gaussian_log_likelihood` in `src/pde_diff/loss.py` then applies

```python
log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)   # = ln(1e-12)
```

- Wherever $\tfrac12 r^2/\Sigma_t > 27.63$, i.e. $r^2 > 55\,\Sigma_t$, the loss is a constant.
- `torch.clamp` passes **zero gradient** for those elements.

### 9.2 Why it is a bad fit for ERA5

$\Sigma_t$ ranges from $5.8\cdot10^{-5}$ to $2\cdot10^{-2}$. ERA5 itself does not satisfy the residuals exactly: the normalised $\mathcal{R}_{geo}$ of the true data is about 1.2. So even **ground truth** is mostly clamped:

| Diffusion step $t$ | 1 | 50 | 100 | 250 | 499 | Average over $t$ |
|---|---|---|---|---|---|---|
| ERA5-truth elements clamped (zero gradient) | 99.6 % | 88 % | 80 % | 62 % | 45 % | **65 %** |

The gradient therefore vanishes at **low noise**, where $\hat{x}_0$ is accurate and the physics is meaningful. It survives mainly at **high noise**, where $\hat{x}_0$ is a blurry posterior mean and its residual says little. This is the opposite of what is wanted, and it explains why the $\mathcal{R}_{geo}$ loss stayed flat during training.

### 9.3 Where it came from

Bastek et al.'s general code (`denoising_utils.py`) does **not** clamp. The clamp only appears in `denoising_toy_utils.py`, which is used for the unit-circle toy problem (paper, App. A.6.1). Our function is a copy of the toy version.

The `# avoid log(0)` comment comes from Bastek code, where a *probability* is clamped before taking its log. Here the log-likelihood is computed directly in log space and can never be $-\infty$ (since $\Sigma_t > 0$), so the clamp only acts as a hard cap on the loss.

There is also a deeper mismatch. Bastek's PDEs (Darcy flow, elasticity) are satisfied **exactly** by the training data, so "residual = 0 with variance $\Sigma_t/c$" is a sensible virtual observation. Our residuals are approximate physics: on ERA5 their balance indices are 0.10 ($\mathcal{R}_{geo}$), 0.23 ($\mathcal{R}_{vd}$) and 0.53 ($\mathcal{R}_{PV}$), see section 8. Without the clamp, the same likelihood would push the model to be *more* balanced than the real atmosphere, and most strongly at small $t$: the weight at $t=1$ is 343× the weight at $t=499$.

### 9.4 Options

The mean physics term on ERA5 truth is given for $\mathcal{R}_{geo}$, before multiplying by $c$. For comparison, the data loss is about 0.15.

| Option | Loss | Mean on ERA5 | Sensible $c$ | Comment |
|---|---|---|---|---|
| 0. Keep as is | $\tfrac12 r^2/\Sigma_t$, clamped | — | — | Mostly zero gradient (9.2). Not recommended. |
| 1. Bastek, no clamp | $\tfrac12 r^2/\Sigma_t$ | 418 | $10^{-5}$–$10^{-3}$ | Faithful to the paper, but assumes exact physics. Over-constrains at small $t$. |
| 2. Soft cap | Huber or log-cosh on $r/\sqrt{\Sigma_t}$ | — | sweep | Keeps a bounded, **nonzero** gradient for outliers. A minimal change. |
| 3. Intrinsic residual variance (**implemented, default**) | $\tfrac12 r^2/(\Sigma_t + \sigma^2_\text{ERA5})$, with $\sigma^2_\text{ERA5} = $ ERA5's $E[r^2]$ (1 for every residual since section 11; was 2.44 for the old $\mathcal{R}_{geo}$) | ≈0.5 | $3\cdot10^{-2}$–$1.5\cdot10^{-1}$ | Bastek's virtual likelihood, extended so the observation noise includes the real imbalance of the atmosphere. Bounded at small $t$. A principled generalisation. |
| 4. SNR-weighted squared residual | $\min(\text{SNR}_t, 5)\cdot r^2$ | 3.6 | $4\cdot10^{-3}$–$2\cdot10^{-2}$ | Same weighting as our data term. Simple, but no likelihood interpretation. |

**Recommendation:** remove the clamp and add a config switch for options 1, 3 and 4. Then run an $\mathcal{R}_{geo}$-only sweep against a $c=0$ baseline (1 fold, about 5 epochs). Compare `val_data_loss`, the sampled $\mathcal{R}_{geo}$ against the ERA5 level (about 1.2), and the ratio of physics to data gradient norms.

### 9.4.1 Implementation (2026-11-06)

- `src/pde_diff/loss.py` has a table `RESIDUAL_WEIGHTINGS` of interchangeable weighting functions, selected with `loss.residual_weighting` in `configs/config.yaml`:

  | Name | Loss per element | Use |
  |---|---|---|
  | `intrinsic` (default) | $\tfrac12 r^2/(\Sigma_t + \sigma^2_\text{ERA5})$ | Option 3 |
  | `bastek` | $\tfrac12 r^2/\Sigma_t$ | Option 1 (paper, no clamp) |
  | `bastek_clamped` | the old clamped form | Reproduces runs from before 2026-11-06 |

- **$\sigma^2_\text{ERA5}$** is not a config option. It is ERA5's own second moment of the normalised residual, $E[r^2]$, read from `residual_stats.json` by `VorticityLoss.intrinsic_residual_variance`:
  - 1 for $\mathcal{R}_{PV}$ and $\mathcal{R}_{vd}$, which are standardised;
  - ~~$1 + (\text{mean}/\text{std})^2 \approx 2.44$ for $\mathcal{R}_{geo}$, which is only divided by its std.~~ Since the signed $\mathcal{R}_{geo}$ (section 11) this is 1 as well: $\mathcal{R}_{geo}$ is divided by its RMS, so $E[r^2] = 1$ by construction.
- **Adding option 4 later:** one function `_weight_snr(r, ctx)` returning `ctx["snr_weight"] * r**2`, registered in `RESIDUAL_WEIGHTINGS`. The context already passes the per-sample $\min(\text{SNR}_t, 5)$ weight of the data term.
- **Logging:** the clamp fraction is only logged, and only plotted, for `bastek_clamped`.
- **Note:** $\Sigma_t \le 0.02 \ll \sigma^2_\text{ERA5}$, so with `intrinsic` the weight is almost independent of $t$. High-noise steps, where $\hat{x}_0$ is a blurry mean, count about as much as low-noise ones. Combining this with an SNR factor is a natural follow-up (direction 3 in section 9.5).

### 9.5 Directions that could go into a paper

1. **Approximate physics as a likelihood (option 3).** Model the residual of real data as noise with a known, measurable variance, using the ERA5 statistics in `residual_stats.json`. This could be made spatially varying, e.g. per latitude, or learned. It generalises physics-informed diffusion to systems where the governing equations hold only approximately, which covers most real geophysical data.
2. **Residual matching, "data-anchored".** Replace $\lVert R(\hat{x}_0)\rVert^2$ with $\lVert R(\hat{x}_0) - R(x_0)\rVert^2$, where $x_0$ is the training target. This is zero for the truth, so it never conflicts with the data term, and it targets the real atmosphere's imbalance instead of zero. It acts as a physically weighted gradient-matching loss. A recent preprint (arXiv 2610.16582) reportedly proposes a related "data-anchored residual"; this needs checking before citing.
3. **Noise-level awareness.** $R(\mathbb{E}[x_0|x_t]) \neq \mathbb{E}[R(x_0)|x_t]$: at high noise, the posterior mean is smooth and violates the physics even if every individual sample satisfies it. Options:
   - weight the physics term by signal level, e.g. SNR or $\bar\alpha_t$;
   - restrict it to $t < t_\text{max}$;
   - use a few DDIM steps to estimate $\hat{x}_0$, as in Bastek's "sample estimation".
   The choice could be studied systematically against the balance index.
4. **Residual choice by balance index.** Use the balance index $B$ (section 8) to decide which equations are worth enforcing ($\mathcal{R}_{geo}$ yes, $\mathcal{R}_{PV}$ in its current form no), and report $B$ of the generated samples next to ERA5's own value as a physical-consistency metric.

---

## 10. Time discretisation of $\mathcal{R}_{PV}$ and $\mathcal{R}_{vd}$, and why $\mathcal{R}_{PV}$ is weakly balanced

### 10.1 Time-centred advection (implemented 2026-11-06)

**Problem.** Both residuals estimate the tendency over one model step $\Delta t = 1$ h,

$$
\partial_t q \approx \frac{q_c - q_p}{\Delta t},
$$

where $p$ is the previous state (the conditioning input) and $c$ is the current state (the prediction). This difference is centred at $t-\tfrac12\Delta t$. The advection and divergence terms, however, were evaluated at $t$ only, using the current state. The two halves of the equation therefore referred to different times, which makes the residual only first-order accurate in time.

**Fix (trapezoidal rule).** The spatial terms are averaged over both states, so they are also centred at $t-\tfrac12\Delta t$:

$$
\mathcal{R}_{PV} = \frac{q_c-q_p}{\Delta t} + \tfrac12\big[\mathbf{V}_g\cdot\nabla_h q\big]_p + \tfrac12\big[\mathbf{V}_g\cdot\nabla_h q\big]_c ,
$$

$$
\mathcal{R}_{vd} = \frac{\zeta_c-\zeta_p}{\Delta t} + \tfrac12 N_p + \tfrac12 N_c, \qquad N = u\,\partial_x\zeta + v\,\partial_y\zeta + (\zeta+f)\,\delta .
$$

- The previous state is the clean conditioning input, so the extra terms add no new path for the model gradient. Only the current-state half depends on the prediction.
- **Test:** `test_time_centred_residuals_symmetric_in_time`. Swapping the previous and current states must flip the sign of the tendency and leave every other term unchanged. The old forward version fails this test.
- **Code:** `compute_residual_planetary_vorticity`, `_qg_advection` and `compute_residual_vorticity_divergence` in `src/pde_diff/loss.py`.

**Effect on ERA5** (16 random 2016 timesteps):

| | Balance index before | Balance index after | Std of residual before → after |
|---|---|---|---|
| $\mathcal{R}_{PV}$ | 0.512 | 0.502 | 2.6·10⁻⁸ → 2.5·10⁻⁸ |
| $\mathcal{R}_{vd}$ | 0.231 | **0.200** | 5.3·10⁻⁹ → 4.3·10⁻⁹ |

`residual_stats.json` must be recomputed, because the $\mathcal{R}_{PV}$ and $\mathcal{R}_{vd}$ statistics changed.

### 10.2 Why $\mathcal{R}_{PV}$ stays weakly balanced

On ERA5, with the time-centred advection, the following variants were tested:

| Variant | Balance index $B$ |
|---|---|
| $q_{QG}$ advected by $\mathbf{V}_g$ (current formulation) | 0.50 |
| $q_{QG}$ advected by the actual wind $\mathbf{V}$ | 0.51 |
| Relative-vorticity part of $q$ only, $\nabla^2\Phi/f_0$ | 0.51 |
| Stretching (vertical) part of $q$ only | 0.30 |
| $\Phi$ low-pass filtered over 3×3 / **5×5** / 9×9 grid points before computing $q$ | 0.35 / **0.29** / 0.31 |
| ERA5 Ertel PV (`pv` channel) advected by $\mathbf{V}$ / by $\mathbf{V}_g$ | **0.24** / 0.25 |

- **Neither the time offset nor the choice of wind is the cause.** The poorly conserved part is the relative-vorticity term $\nabla^2\Phi/f_0$. It is a second derivative of $\Phi$ on a 0.75° grid, about 45–80 km, and it carries most of the variability of $q$.
- **QG theory only describes synoptic scales**, about 1000 km and larger. Grid-scale structure is not quasi-geostrophic, so $q_{QG}$ is not conserved there.
- **Low-pass filtering $\Phi$** over about 4° (5×5 grid points) before computing $q$ removes these scales and improves $B$ by about 40 %.
  - Such a filter is a fixed, differentiable linear operation. The model would still be constrained, but only on QG scales, and the data loss handles the small scales.
  - The latitude edges, where the filter runs off the domain, would need to be masked out of the residual.
  - **Saved for later.** The filter width should be chosen with a sweep on the full dataset.

### 10.3 Idea: conservation of Ertel PV as an alternative $\mathcal{R}_{PV}$

The model already predicts ERA5's Ertel potential vorticity $P$ (the `pv` channel, in K m² kg⁻¹ s⁻¹). For adiabatic, frictionless flow, $P$ is materially conserved:

$$
\frac{DP}{Dt} = \frac{\partial P}{\partial t} + u\frac{\partial P}{\partial x} + v\frac{\partial P}{\partial y} + \omega\frac{\partial P}{\partial p} = 0 .
$$

The vertical velocity $\omega$ is not part of the state, so a residual with horizontal advection only, time-centred and at the middle level (500 hPa), is

$$
\mathcal{R}_{Ertel}(\tilde{\mathbf{x}}_0) = \frac{P_c - P_p}{\Delta t} + \tfrac12\big[\mathbf{V}\cdot\nabla_h P\big]_p + \tfrac12\big[\mathbf{V}\cdot\nabla_h P\big]_c ,
$$

with the predicted wind $\mathbf{V} = (u, v)$ and the predicted $P$.

- **Balance on ERA5:** $B \approx 0.24$ at native resolution, without filtering. That is about twice as well balanced as $\mathcal{R}_{PV}$.
- **Advantages:**
  - It needs only first derivatives (no $\nabla^2\Phi$, no vertical second derivative), so it is much less sensitive to grid-scale noise.
  - It uses quantities the model predicts directly, with no QG approximation and no constant $f_0$.
  - It ties the predicted `pv` channel dynamically to the predicted wind; currently nothing constrains `pv`.
- **Limitations:**
  - It neglects vertical advection $\omega\,\partial_p P$, diabatic heating and friction. These are the remaining imbalance.
  - $\omega$ could be added later via continuity, $\partial\omega/\partial p = -\delta$, but with three levels this is crude.
  - It is a different equation from the original QG formulation, so it would be an extension rather than a replacement in the thesis framing.
- **Implementation sketch:** same structure as $\mathcal{R}_{vd}$ (first derivatives, trapezoidal time-centring), with its own entry in `residual_stats.json` and its own balance index.

## 11. Signed geostrophic residual (implemented 2026-11-12)

### 11.1 Change

| | Before | Now |
|---|---|---|
| Residual | $\mathcal{R}_{geo} = \lvert u_g - u\rvert + \lvert v_g - v\rvert$ (one non-negative field) | $\mathcal{R}_{geo} = (u_g - u,\; v_g - v)$ (two signed fields, shape `(b, 2, lev, lon, lat)`) |
| Squared in the loss | $(\lvert\Delta u\rvert + \lvert\Delta v\rvert)^2 = \Delta u^2 + \Delta v^2 + 2\lvert\Delta u\rvert\lvert\Delta v\rvert$ | $\Delta u^2 + \Delta v^2 = \lvert\mathbf{V}_g - \mathbf{V}\rvert^2$ |
| ERA5 statistics | mean 4.17, std 3.47 m/s (non-negative) | mean ≈ 0, RMS ≈ 3.0 m/s per component (2016 check, 48 samples) |
| Normalisation | divided by the std; $E[r^2] = 2.44$ | divided by the RMS $\sqrt{\text{mean}^2 + \text{std}^2}$, shared by both components; $E[r^2] = 1$ |

The mean is still not subtracted, so the target remains exact balance (0). With mean ≈ 0 this makes almost no difference.

### 11.2 Why

1. **Rotation-invariant.** With the old form, an ageostrophic error of 1 m/s pointing NE cost 2 and the same error pointing E cost 1, because of the cross term. Nothing in geostrophy prefers the grid axes. The new form depends only on $\lvert\mathbf{V}_g - \mathbf{V}\rvert$. The rollout drift of the data-only model (rollout balance analysis, 2026-11-10) is a large-scale ageostrophic error in every direction, so this matters for exactly the error we want to reduce.
2. **Smooth.** The old gradient $2(\lvert\Delta u\rvert + \lvert\Delta v\rvert)\,\text{sign}(\Delta u)$ jumps whenever $\Delta u$ changes sign. That happens everywhere, because each ageostrophic component is near zero over large areas.
3. **Linear before squaring.** $\Delta u$ and $\Delta v$ are affine in the predicted state ($f_0$ constant, finite differences linear, un-normalisation affine). So the residual of a roughly Gaussian $\hat{x}_0$ is Gaussian, and $\tfrac12 r^2/(\Sigma_t + \sigma^2)$ is a proper Gaussian negative log-likelihood. The absolute value made the residual a folded, non-negative distribution with mean ≠ 0, which is what required the std-only normalisation and the 2.44.
4. **Consistent with $\mathcal{R}_{PV}$ and $\mathcal{R}_{vd}$**, which are already signed.

### 11.3 Consequences

- **Weight range unchanged.** On ERA5, $(\lvert\Delta u\rvert + \lvert\Delta v\rvert)^2$ is on average 1.62× $\Delta u^2 + \Delta v^2$. But the normalisation absorbs this: in both versions the loss per element of an ERA5-like residual is $\tfrac12 E[r^2]/\sigma^2_\text{ERA5} \approx 0.5$. So the range $c \approx 3\cdot10^{-2}$–$1.5\cdot10^{-1}$ from section 9.4 still applies.
- **The convention of a mean over elements.** The loss averages over both components. Summing instead would double it, which would be equivalent to doubling $c$.
- **Diagnostics unchanged.** Relative residuals and the balance index use the components ($B$ keeps its $\lvert\cdot\rvert$ definition, section 8.2). The rollout metric `R2_geostrophic_wind` and the residual maps now use the magnitude $\lvert\mathbf{V}_g - \mathbf{V}\rvert/\text{RMS}$.
- **`residual_stats.json` must be recomputed** with `python src/pde_diff/data/plot_data_distribution.py`. It keeps the same format (mean and std); the loss computes the RMS as $\sqrt{\text{mean}^2 + \text{std}^2}$.
- `Equations.md` updated to the new definition.

## 12. Edge latitude rows left out of the residuals (implemented 2026-11-12)

### 12.1 Why

The first and last latitude rows (70°N and 46.75°N) are not physical boundaries; the grid is just cropped there. On those rows, $\partial/\partial y$ needs a one-sided stencil (section 7). That stencil is still second-order, but it is less accurate and amplifies noise more:

| Stencil | Weights | Noise variance gain | Leading error |
|---|---|---|---|
| Central $\partial/\partial y$ | $(-1, 0, 1)/2$ | 0.5 | $\tfrac16 f^{(3)}\Delta y^2$ |
| One-sided $\partial/\partial y$ | $(-3, 4, -1)/2$ | 6.5 (13×) | $\tfrac13 f^{(3)}\Delta y^2$ (2×) |
| Central $\partial^2/\partial y^2$ | $(1, -2, 1)$ | 6 | $\tfrac1{12} f^{(4)}\Delta y^2$ |
| One-sided $\partial^2/\partial y^2$ | $(2, -5, 4, -1)$ | 46 (7.7×) | $\tfrac{11}{12} f^{(4)}\Delta y^2$ (11×) |

Measured on ERA5 (2016, 48 samples), per latitude row, relative to the interior:

| | Row 0 (70°N) | Row 1 | Interior | Row 30 | Row 31 (46.75°N) |
|---|---|---|---|---|---|
| $\mathcal{R}_{vd}$ RMS | **2.4×** | 0.94× | 1 | 1.01× | **2.8×** |
| $\mathcal{R}_{vd}$ $B$ | **0.31** | 0.18 | 0.186 | 0.175 | **0.30** |
| $\mathcal{R}_{geo}$ RMS | 1.04× | 0.92× | 1 | 1.13× | 1.29× |
| $\mathcal{R}_{geo}$ $B$ | 0.113 | 0.103 | 0.098 | 0.099 | 0.107 |
| $\mathcal{R}_{PV}$ $B$ | 0.57 | 0.55 | 0.50 | 0.49 | 0.45 |

- **$\mathcal{R}_{vd}$:** a clear stencil artifact, confined to exactly the two edge rows. Row 1 is normal, even though $\partial\zeta/\partial y$ there uses the one-sided $\zeta$ of row 0. With the squared loss, these 2 of 32 rows (6 % of the grid) carried **31 % of $\mathcal{R}_{vd}$'s loss**, so the physics gradient mostly pushed on stencil error at the domain edges.
- **$\mathcal{R}_{geo}$:** a mild effect. Part of the increase towards the southern edge is the $f_0$ latitude bias (section 7), which grows towards both edges and is not a stencil effect.
- **$\mathcal{R}_{PV}$:** no clear edge artifact. Its imbalance is dominated by grid-scale noise everywhere (section 10.2).

### 12.2 Implementation

- `VorticityLoss.EDGE_ROWS = 1`. All three residuals drop that many rows at each latitude edge with `VorticityLoss.interior`, so they have 30 instead of 32 rows.
- The derivatives are still computed on the full grid (one-sided at the edges), so row 1 keeps its central difference.
- The cropping happens inside the residual functions, so it applies consistently everywhere: the loss, `residual_stats.json`, the balance index (`return_terms`), the relative residuals and the rollout metrics.
- The rollout residual maps pad the dropped rows back with NaN (`VorticityLoss.pad_edge_rows`), so they keep the full map extent; the edge rows appear blank.
- Not cropped: the spectra of the geostrophic and ageostrophic wind. The FFT needs the full rectangular grid, and its Hann window already gives the edge rows almost no weight.

### 12.3 Effect on ERA5 (2016, 48 samples)

| | $B$, full grid | $B$, without edge rows | RMS without / with edge rows |
|---|---|---|---|
| $\mathcal{R}_{PV}$ | 0.502 | 0.499 | 0.994 |
| $\mathcal{R}_{geo}$ | 0.099 | 0.098 | 0.989 |
| $\mathcal{R}_{vd}$ | 0.197 | **0.186** | **0.858** |

### 12.4 Trade-offs

- The edge rows get no physics constraint, only the data loss, so a model could in principle leave them unbalanced. Watch the rollout residual maps for imbalance growing inward from the edges.
- The full-dataset balance indices in section 8.3 include the edge rows. After recomputing the stats, $B$ for $\mathcal{R}_{vd}$ will be slightly lower (≈0.19 instead of ≈0.20).
- `residual_stats.json` must be recomputed, which is needed anyway after sections 10 and 11.
