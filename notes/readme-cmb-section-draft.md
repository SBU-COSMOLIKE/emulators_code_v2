# README draft: the CMB spectra emulators section (CME)

**Date:** 2026-07-11. **Status:** DRAFT — paste into the MAIN checkout's
README.md after the origin/main merge, as a new numbered section beside
the scalar-emulators section (renumber appendices as the scalar
precedent did; every anchor in the Contents). The section body below is
README-ready (define-or-drop jargon, YAML snippets, ASCII pipeline).

---

## Emulating CMB spectra (TT / TE / EE / phi-phi)

A CMB emulator maps cosmological parameters to one spectrum's C_ell
values — the angular power spectrum of the cosmic microwave background
(TT = temperature, TE = temperature-polarization cross, EE = E-mode
polarization, phi-phi = the lensing potential), on a fixed multipole
grid l = 2..lmax. One emulator learns ONE spectrum; a full set is four
artifacts. Everything rides the same training stack as the data-vector
emulators — the losses, trimming, focal weighting, EMA, fine-tuning all
compose unchanged — because the loss exposes the same per-sample chi2
interface.

The pipeline, end to end:

    dataset_generator_cmb.py            compute_cmb_covariance.py
    (CAMB through cobaya, one call      (Motloch & Hu 1709.03599 eqs 1-7,
     per sampled cosmology)              one fiducial-LCDM CAMB call)
          |                                   |
          |  dvs_*_tt.npy (+ te/ee/pp)        |  cmbcov_lcdm.npz
          |  params_*.1.txt + sidecars        |  (sigma_ell per spectrum,
          v                                   v   fiducial C_ell, provenance)
    +---------------------------------------------------+
    |  train_single_emulator_cosmic_shear.py             |
    |  with a data.cmb block: whiten each multipole by   |
    |  its error bar sigma_ell, impose the amplitude     |
    |  law, train the ResMLP trunk                       |
    +---------------------------------------------------+
          |
          |  emulator_tt_*.h5 + .emul   (the artifact)
          v
    cobaya_theory/emul_cmb.py    serves get_Cl to any cobaya likelihood

### The covariance file (why a separate script)

For cosmic shear the loss covariance comes from cosmolike. A CMB
spectrum's covariance is analytic instead: the Gaussian variance of one
measured C_ell is (Motloch & Hu 1709.03599, eq 3)

    var(C_ell) = 2 / [(2l+1) fsky] * (C_ell + N_ell)^2

where N_ell is the instrumental noise spectrum built from the detector
noise level (in muK-arcmin) and the beam width (eq 1). The script
`compute_data_vectors/compute_cmb_covariance.py` computes this once, on
a fiducial LCDM cosmology at high CAMB accuracy, and writes one .npz the
training consumes; the optional non-Gaussian lensing terms (eq 6) sit
behind a flag, off by default. What you state in its YAML is the
experiment: the noise level, the beam, the sky fraction.

### The imposed amplitude law

The primary CMB spectra scale almost exactly as A_s e^(-2 tau) (the
primordial amplitude damped by reionization). Rather than making the
network learn that known scaling, the training can impose it: with

```yaml
  cmb:
    spectrum: tt
    covariance: cmbcov_lcdm.npz
    amplitude_law: as_exp2tau
    as_name:       As
    tau_name:      tau
```

the target the network sees is C_ell * exp(2 tau) / A_s — the SHAPE
only — and the emulator multiplies the law back on the way out. A_s and
tau are read from named parameter columns of the training dump (As must
be the linear amplitude, which the generator samples directly). Set
`amplitude_law: none` to learn the raw C_ell instead (and drop the two
names). The law is stored in the artifact by name, so a saved emulator
always knows its own convention.

### The roughness penalty (optional)

CMB spectra are smooth in l; short-period wiggles in the emulator
residual are network artifacts, never physics. The optional loss term

```yaml
  loss:
    mode: sqrt
    roughness:
      lam:        0.1   # weight; absent block = the term does not exist
      period_cut: 50    # penalize residual oscillation periods below this
```

adds, per training sample, `lam` times the short-period content of the
residual (a high-pass filter at `period_cut` multipoles) to the chi2
before the usual reduction. It acts on the residual — prediction minus
truth — so a perfect prediction pays nothing, however sharp or smooth
the true peaks: it cannot bias the lensing-induced peak smoothing,
whose period (~200-300, the acoustic spacing) the filter passes
untouched.

### Serving the spectra in an MCMC

```yaml
theory:
  emul_cmb:
    python_path: ./cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - chains/emulator_tt_resmlp_ntrain50000
        - chains/emulator_ee_resmlp_ntrain50000
```

Each path root declares its own spectrum, multipole range, and units
(they are stored in the artifact — nothing is restated in the YAML).
A likelihood that requests a spectrum no artifact provides, or
multipoles beyond an artifact's training grid, fails loudly at startup.
get_Cl serves raw C_ell in the dump units (muK^2; phi-phi
dimensionless), zero below l = 2.

### Fine-tuning

A CMB emulator warm-starts from a saved CMB emulator of the same
spectrum, law, and covariance file, exactly like the data-vector
fine-tune: add

```yaml
train_args:
  finetune:
    from: chains/emulator_tt_resmlp_ntrain50000
```

and delete the `model:` block (the architecture is inherited). Epoch 0
reproduces the source exactly; training then refines it on the new
dump.

### The diagnostics pages

`--diagnostic` on a data.cmb run appends two CMB pages to the usual
PDF: per-multipole residual bands (fractional AND in error-bar units —
read the error-bar panel for TE, which crosses zero) with the
worst-cosmology overlay, and the residual's short-period wiggle content
(what the roughness term sees). The example config is
`example_yamls/cmb_emulator.yaml`; the gates are cmb-identity and
cmb-smoke on the board.

---

**Also update in the same pass:** the code-map table row for
`compute_data_vectors/` (generator_core.py + the two thin drivers +
compute_cmb_covariance.py), the Contents anchors, and the appendix
numbering if the section lands mid-document.
