# `syren/`: analytic matter-power base formulas

This directory contains the vendored `symbolic_pofk` formulas used by the
matter-power-spectrum emulators. A matter power spectrum
$P_{\rm lin}(k,z)$ describes the variance of the linear matter-density field
as a function of wavenumber $k$ and redshift $z$. CoCoA SONIC stores it in
$\mathrm{Mpc}^3$. The nonlinear boost

$$
B(k,z) = \frac{P_{\rm nl}(k,z)}{P_{\rm lin}(k,z)}
$$

is dimensionless. The word *base* means the deterministic analytic
approximation that the neural network may correct.

## The three target laws

A target law defines the number formed from each raw training value before
per-grid-point standardization. The natural logarithm is written `ln` below.
The configured law is stored by name in the emulator artifact.

| Target law | Allowed quantity | Law-space value $y$ | Reconstruction after the network |
|---|---|---|---|
| `none` | `pklin` or `boost` | The raw $P_{\rm lin}$ or raw $B$ | Use the decoded raw value directly. No base file is read. |
| `syren_linear` | `pklin` only | $y=\ln(P_{\rm lin}/P_{\rm base})$ | $P_{\rm lin}=P_{\rm base}\exp(y)$ |
| `syren_halofit` | `boost` only | $y=\ln(B/B_{\rm base})$ | First form $B_{\rm temp}=B_{\rm base}\exp(y)$. The MPS adapter then blends $B_{\rm temp}$ toward 1 on its low-$k$ transition. |

Here $k$ is the comoving wavenumber in $\mathrm{Mpc}^{-1}$, $z$ is redshift,
and $h=H_0/(100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$ is the reduced Hubble
constant. The Syren functions use their documented internal $h$ conventions;
`emulator/syren_base.py` owns those conversions.

For either Syren law, the data generator writes a raw dump and its row-aligned
`*_base` sibling. Training reads both files and forms the logarithmic ratio.
Inference recomputes the same type of analytic base through
`emulator/syren_base.py` and reverses the target law. Under `none`, the
validator refuses base-file keys because no base participates.

The network does not receive $y$ directly. At each $(z,k)$ coordinate,
`Grid2DGeometry` stores the training mean $\mu$ and population standard
deviation $s$, then forms

$$
t = \frac{y-\mu}{s}.
$$

The network predicts $t$. Decoding first recovers $y=t s+\mu$ and then
applies the reconstruction in the table.

**Numerical example.** Suppose one linear-power value is
$P_{\rm lin}=1200\,\mathrm{Mpc}^3$ and the Syren base is
$P_{\rm base}=1000\,\mathrm{Mpc}^3$. The law-space value is
$y=\ln(1.2)\simeq0.1823$. If the training mean is $\mu=0.10$ and the standard
deviation is $s=0.05$, the network target is
$t=(0.1823-0.10)/0.05\simeq1.646$. Decoding gives $y\simeq0.1823$, and the
adapter reconstructs $1000\exp(0.1823)\simeq1200\,\mathrm{Mpc}^3$.

The boost law uses the same ratio. For example, $B=1.50$ and
$B_{\rm base}=1.25$ also give $y=\ln(1.2)$. After reconstruction, the
`syren_halofit` serving path applies its low-$k$ blend.

## Scientific source and license

The implementation is derived from
[`DeaglanBartlett/symbolic_pofk`](https://github.com/DeaglanBartlett/symbolic_pofk)
as carried by the legacy CoCoA `emulmps` bundle. That bundle includes local
modifications that remain visible in the vendored files. The MIT license is
stored in [`LICENSE`](LICENSE). The associated formula papers are
[arXiv:2311.15865](https://arxiv.org/abs/2311.15865),
[arXiv:2402.17492](https://arxiv.org/abs/2402.17492), and
[arXiv:2410.14623](https://arxiv.org/abs/2410.14623).

Keeping these files in the repository prevents an unreviewed package-manager
upgrade from silently replacing the formulas. Vendoring does not by itself
prove byte identity with an upstream revision. This repository does not carry
an upstream source snapshot or an executable byte comparison, so this README
makes no byte-for-byte claim.

## What is here

| file | functions used by the emulator pipeline |
|---|---|
| `linear.py` | `plin_emulated`, `get_approximate_D`, `growth_correction_R`, and `As_to_sigma8` |
| `syrenhalofit.py` | `run_halofit_vec` with `return_boost=True` |

The repository consumer is `emulator/syren_base.py`. The dump generator,
`emul_mps` adapter, and gates call that owner instead of calling this package
directly.

## Local packaging choices

The current files have these directly observable import properties:

- `linear.py` imports NumPy. It does not import `warnings` or
  `scipy.integrate`.
- `syrenhalofit.py` imports the local module as
  `import syren.linear as linear`.

`linear.py` also contains local formula modifications inherited from the
CoCoA bundle. They are part of the scientific implementation used here, not
packaging-only changes.

Changing either formula changes the analytic base used to construct future
training targets and to serve predictions. Treat such a change as a
scientific model change. An emulator trained against a different base
implementation must not be served with the changed formulas without
retraining.
