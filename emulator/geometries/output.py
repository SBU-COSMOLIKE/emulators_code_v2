"""Output (data-vector) geometries and the shear angle map.

This module is the output side: it owns every transform between a raw
cosmolike data vector and the whitened, masked target the network
predicts, plus the chi2's covariance. DataVectorGeometry is the base:
squeeze to the unmasked entries, center, and whiten in the covariance
eigenbasis. Decoding reverses centering and whitening only on the kept
coordinates. ``unsqueeze`` can place those values in a full vector, but
masked coordinates are filled with zero and cannot be recovered.
DiagonalGeometry whitens by the marginal sigma only (theta order
kept, for a 1D CNN head); BlockDiagonalGeometry whitens each tomographic
bin by its own sub-block. build_shear_angle_map attaches the per-element
angle / tomography metadata (theta, source redshifts, xi+/- branch,
per-bin sizes). The only module that imports cosmolike.

PS: to whiten is to rotate into the covariance eigenbasis and scale each
coordinate to unit variance under the covariance that defines the transform.
This decorrelates the coordinates and gives them comparable numerical scale.
Learning difficulty can still differ among directions. To
squeeze is to keep only the unmasked entries of the full data vector. encode
= squeeze, center, then whiten, the form
the network predicts. The Mahalanobis distance r^T Cinv r is a squared
residual r weighted by the inverse covariance Cinv (the chi2 this geometry
owns), summed over the kept entries.

    dv (B, total_size)       raw cosmolike data vector
       │  squeeze            keep the unmasked entries (dest_idx)
       ▼
       │  - center           subtract the training-mean dv
       ▼
       │  whiten             rotate into the covariance eigenbasis,
       │                     scale each direction to unit variance
       ▼
    t  (B, n_keep)           whitened target the network predicts

(legend: B = batch rows; total_size = full data-vector length
including masked entries; n_keep = the kept/unmasked length, the
width the network emits; dest_idx = the kept entries' positions in
the full vector. ``decode(encode(dv))`` returns the kept physical
coordinates, not the original full vector. ``squeeze(unsqueeze(kept))``
returns ``kept``. ``unsqueeze(squeeze(dv))`` matches ``dv`` only at the
kept coordinates and fills masked coordinates with zero. The chi2 in
losses/core.py unwhitens the residual and contracts it with the inverse
covariance.)
"""

import os
import numpy as np
import torch
# cosmolike_lsst_y1_interface and getdist are imported at their use sites
# (from_cosmolike, build_shear_angle_map), NOT at module level: both are heavy
# optional training-path dependencies, and importing them here made a missing
# one an import-time death for every consumer of this module -- inference, the
# board, tests (25M-37). Deferred, absence is a clear failure of the one call
# that needs them, a declared disposition rather than an import that never
# returns.


class DataVectorGeometry:
  """
  Geometry and normalization of one probe's masked data
  vector. Whitening is applied to the data-vector targets.
  It decorrelates the covariance coordinates and places them
  on comparable numerical scales. Learning difficulty can
  still differ among coordinates.

  One instance owns every transform between a raw cosmolike
  dv and the vector the network sees, for one probe (xi,
  gammat, wtheta, or the full 3x2pt). It holds:

    - dest_idx: positions, in the full 3x2pt vector, of the
      entries surviving the mask. squeeze picks these columns
      out; unsqueeze scatters them back into a full-length
      zero vector.
    - evecs / sqrt_ev: the whitening basis (evecs the
      rotation, sqrt_ev the per-direction scale).
    - Cinv: the full-3x2pt masked inverse covariance the
      chi2 contracts against.
    - center: training-mean of the kept entries (the
      targets' zero-point; cancels in a residual chi2).

  Build from cosmolike at training time (from_cosmolike) or
  saved tensors at inference time (from_state); the geometry
  travels with the weights, so inference never rereads
  cosmolike.

  dtype sets the precision of the whitening basis and Cinv.
  float32 (default) gives fast GEMMs: on this xi covariance
  the float64-vs-float32 chi2 gap is ~1e-7, while float64 runs
  ~1/64 speed on a consumer GPU. Use float64 for
  cross-correlated 3x2pt if the stiff directions need it. The
  eigendecomposition is always float64 (numpy, at
  construction); only the stored result is cast to dtype.
  """

  # Keys are cosmolike possible_probes strings (passed
  # as-is to ci.init_probes), mapped to the 3x2pt blocks
  # each spans: xi (cosmic shear)=0, gammat (galaxy-galaxy
  # lensing; cosmolike's name for ggl)=1, wtheta
  # (clustering)=2. Unions like 2x2pt are [1, 2].
  PROBE_BLOCKS = {
    "xi":     [0],
    "gammat": [1],
    "wtheta": [2],
    "3x2pt":  [0, 1, 2],
  }

  def __init__(self,
               device,
               total_size,
               dest_idx,
               evecs,
               sqrt_ev,
               Cinv,
               center,
               dtype = torch.float32,
               section_sizes = None,
               probe = None,
               bin_sizes = None,
               pm_kept = None,
               head_pad_idx = None,
               head_valid_mask = None):
    """Place the geometry tensors on the device.

    Plain constructor: stores fields only; the two
    classmethods below build them. as_tensor accepts numpy
    (from cosmolike) or cpu tensors (from a saved state), so
    both paths share this code.

    Arguments:
      device     = device the tensors live on.
      total_size = length of the full 3x2pt data vector
                   (the size unsqueeze restores to).
      dest_idx   = positions, in the full 3x2pt vector, of
                   the entries that survive the mask;
                   squeeze gathers these columns, unsqueeze
                   scatters them back.
      evecs      = eigenvectors of the kept-block covariance
                   (whitening rotation; columns orthonormal).
      sqrt_ev    = square roots of that covariance's
                   eigenvalues (the whitening scale).
      Cinv       = full-3x2pt masked inverse covariance, used
                   by the chi2.
      center     = training-mean of the kept entries (the
                   targets' zero-point), already squeezed.
      dtype      = precision of evecs / sqrt_ev / Cinv
                   (float32 by default).
      section_sizes = the full 3x2pt block sizes cosmolike
                   reports (xi / gammat / wtheta lengths),
                   recorded so inference can slice this
                   geometry's own probe section out of the
                   scattered full vector. None on an older v2
                   file that predates the key (loud, no
                   fabricated default); normalized to a python
                   int list (from_cosmolike passes a list,
                   from_state a numeric tensor read back).
      probe      = the possible_probes string this geometry
                   was built for (xi / gammat / wtheta /
                   3x2pt); names which blocks the section is.
                   None on an older v2 file.
      bin_sizes  = the conv/TRF heads' per-bin kept-element
                   counts (build_shear_angle_map's attach,
                   persisted by state() when present so a head
                   artifact rebuilds without the dataset ini).
                   None (the default, and every trunk-only or
                   older file): the ATTRIBUTE stays unset —
                   hasattr(geom, "bin_sizes") is the guard the
                   head constructors and BlockDiagonalGeometry
                   check, so a None-valued attribute would
                   defeat it. Normalized to a python int list.
      pm_kept    = the per-element xi+/xi- branch flags
                   (0 = xi+, 1 = xi-) the ResCNN groups=2
                   validation reads; persisted / restored with
                   bin_sizes under the same unset-when-None
                   rule. Normalized to a numpy int array (the
                   form build_shear_angle_map attaches).
      head_pad_idx = one integer rectangular slot for every kept physical
                   value. The map preserves the original angular coordinate
                   instead of replacing it with survivor rank.
      head_valid_mask = two-dimensional Boolean array aligned with the
                   structured head's (bin, angular-coordinate) rectangle.
                   True marks physical values and False marks storage-only
                   padding. Both layout fields are absent together on a
                   trunk-only or older artifact.
    """
    self.dtype = dtype
    self.total_size = int(total_size)

    self.dest_idx = torch.as_tensor(dest_idx,
                                    dtype=torch.long,
                                    device=device)
    self.evecs = torch.as_tensor(evecs,
                                 dtype=dtype,
                                 device=device)
    self.sqrt_ev = torch.as_tensor(sqrt_ev,
                                   dtype=dtype,
                                   device=device)
    self.Cinv = torch.as_tensor(Cinv,
                                dtype=dtype,
                                device=device)
    self.center = torch.as_tensor(center,
                                  dtype=torch.float32,
                                  device=device)

    # masked sub-block of the precision (out_dim x out_dim):
    # the only part the chi2 needs, since the unsqueezed
    # residual is zero off the kept entries. Built here so
    # from_state rebuilds it too, leaving state() unchanged.
    self.Cinv_sq = self.Cinv[self.dest_idx][:,self.dest_idx]

    # section accounting (persist-resolved-values): the probe
    # string and the full 3x2pt block sizes, so inference can
    # slice this geometry's own section out of the scattered
    # vector. Normalized to python str / int-list; None stays
    # None (an older file: inference fails loudly, never a
    # fabricated default).
    self.probe = None if probe is None else str(probe)
    if section_sizes is None:
      self.section_sizes = None
    else:
      sizes_int = []
      for s in section_sizes:
        sizes_int.append(int(s))
      self.section_sizes = sizes_int

    # the heads' bin split, restored from a saved head artifact (the
    # training path attaches it later, via build_shear_angle_map).
    # Deliberately attribute-absent when None: hasattr(geom,
    # "bin_sizes") is the loud gate in ResCNN / ResTRF /
    # BlockDiagonalGeometry, and a None-valued attribute would slip
    # past it into a confusing crash. Normalized to the exact forms
    # build_shear_angle_map attaches (int list / numpy int array), so
    # a rebuilt geometry is indistinguishable from a freshly-attached
    # one.
    if bin_sizes is not None:
      bins_int = []
      for s in bin_sizes:
        bins_int.append(int(s))
      self.bin_sizes = bins_int
    if pm_kept is not None:
      # from_state hands a device tensor (the h5 reader moves every
      # numeric dataset to the run device); numpy cannot read a CUDA
      # tensor directly, so hop through cpu for that form.
      if isinstance(pm_kept, torch.Tensor):
        pm_kept = pm_kept.detach().cpu().numpy()
      self.pm_kept = np.asarray(pm_kept, dtype="int64")
    if (head_pad_idx is None) != (head_valid_mask is None):
      raise ValueError(
        "DataVectorGeometry requires head_pad_idx and head_valid_mask "
        "together")
    if head_pad_idx is not None:
      raw_idx = torch.as_tensor(head_pad_idx)
      if raw_idx.ndim != 1 or raw_idx.dtype == torch.bool \
          or raw_idx.dtype.is_floating_point \
          or raw_idx.dtype.is_complex:
        raise ValueError(
          "DataVectorGeometry head_pad_idx must be a one-dimensional "
          "integer map")
      self.head_pad_idx = raw_idx.to(dtype=torch.long, device=device)
      valid = torch.as_tensor(head_valid_mask, device=device)
      if valid.ndim != 2:
        raise ValueError(
          "DataVectorGeometry head_valid_mask must have shape "
          "(physical bins, angular coordinates)")
      if valid.dtype == torch.uint8:
        binary = torch.logical_or(valid == 0, valid == 1)
        if not bool(torch.all(binary).item()):
          raise ValueError(
            "DataVectorGeometry head_valid_mask uint8 values must be 0 or 1")
        valid = valid.to(dtype=torch.bool)
      elif valid.dtype != torch.bool:
        raise ValueError(
          "DataVectorGeometry head_valid_mask must contain booleans or "
          "persisted uint8 zeros and ones")
      self.head_valid_mask = valid
      flat_valid = valid.reshape(-1)
      if int(self.head_pad_idx.numel()) != int(self.dest_idx.numel()):
        raise ValueError(
          "DataVectorGeometry head_pad_idx must contain one slot per kept "
          "output value")
      if int(flat_valid.sum().item()) != int(self.dest_idx.numel()):
        raise ValueError(
          "DataVectorGeometry head_valid_mask must mark one slot per kept "
          "output value")
      if bool(torch.any(self.head_pad_idx < 0).item()) \
          or bool(torch.any(self.head_pad_idx >= flat_valid.numel()).item()) \
          or not bool(torch.all(flat_valid[self.head_pad_idx]).item()):
        raise ValueError(
          "DataVectorGeometry head_pad_idx must point to the physical slots "
          "in head_valid_mask")
      marked = torch.nonzero(flat_valid, as_tuple=False).reshape(-1)
      if not torch.equal(torch.sort(self.head_pad_idx).values, marked):
        raise ValueError(
          "DataVectorGeometry head_pad_idx and head_valid_mask disagree")
      if self.probe == "xi":
        if not torch.equal(self.head_pad_idx, self.dest_idx):
          raise ValueError(
            "DataVectorGeometry xi head_pad_idx must equal dest_idx: the "
            "full xi layout is the independent physical coordinate map")
        if self.section_sizes is not None \
            and int(flat_valid.numel()) != int(self.section_sizes[0]):
          raise ValueError(
            "DataVectorGeometry xi head_valid_mask must cover the complete "
            "cosmic-shear section")

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    state's keys match __init__, so cls(device, **state)
    reconstructs the geometry with no cosmolike read. cls
    (not the class name) keeps a subclass's type correct. A
    newer file also carries section_sizes / probe (and, on a
    head-model artifact, bin_sizes / pm_kept — the
    build_shear_angle_map attach, persisted so the conv/TRF
    constructors rebuild without the dataset ini); an older
    one omits them, so __init__ leaves each unset/None (loud
    at inference, never a fabricated default).

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state() (total_size, dest_idx, evecs,
               sqrt_ev, Cinv, center, dtype, and section_sizes
               / probe / bin_sizes / pm_kept when the file
               records them), splatted into __init__.

    Returns:
      a DataVectorGeometry (or subclass, via cls).
    """
    return cls(device, **state)

  @classmethod
  def from_cosmolike(cls, 
                     device, 
                     dv_center,
                     data_dir="lsst_y1",
                     dataset="lsst_y1_M1_GGL0.05.dataset",
                     probe="xi", 
                     dtype=torch.float32):
    """Build the geometry from cosmolike (training path).

    Reads the dataset's covariance, inverse covariance, mask,
    and block sizes through the cosmolike interface ci;
    selects the probe's unmasked entries; eigendecomposes the
    kept-block covariance in float64; stores the basis, Cinv,
    and squeezed center at dtype.

    Arguments:
      device    = device for the built tensors.
      dv_center = full (unsqueezed) training-mean dv; its
                  kept entries become the center.
      data_dir  = data folder under external_modules/data.
      dataset   = .dataset ini naming the cov/mask/dv files.
      probe     = one of PROBE_BLOCKS, a cosmolike
                  possible_probes string (xi, gammat,
                  wtheta, 3x2pt).
      dtype     = precision for the stored basis and Cinv.

    Returns:
      a new geometry on ``device``, carrying the probe's kept-entry
      whitening basis, Cinv, center, and the global section layout.
    """
    # deferred (25M-37): the training-path dependencies live here, at their
    # one use site, not at module import.
    import cosmolike_lsst_y1_interface as ci
    from getdist import IniFile

    if probe not in cls.PROBE_BLOCKS:
      raise ValueError(f"unknown probe: {probe}")

    RD   = os.environ["ROOTDIR"]
    path = os.path.normpath(
      os.path.join(RD, "external_modules/data", data_dir))
    ini  = IniFile(os.path.join(path, dataset))
    data_vector_file = ini.relativeFileName("data_file")
    cov_file    = ini.relativeFileName("cov_file")
    mask_file   = ini.relativeFileName("mask_file")
    lens_file   = ini.relativeFileName("nz_lens_file")
    source_file = ini.relativeFileName("nz_source_file")

    lens_ntomo   = ini.int("lens_ntomo")
    source_ntomo = ini.int("source_ntomo")
    ntheta = ini.int("n_theta")
    tmin = ini.float("theta_min_arcmin")
    tmax = ini.float("theta_max_arcmin")

    ci.initial_setup()
    ci.init_probes(possible_probes=probe)
    ci.init_binning(ntheta, tmin, tmax)
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
      lens_multihisto_file=lens_file,
      lens_ntomo=int(lens_ntomo),
      source_multihisto_file=source_file,
      source_ntomo=int(source_ntomo))
    ci.init_probes(possible_probes=probe)
    ci.init_data_real(cov_file, mask_file, data_vector_file)

    sizes = []
    for s in ci.compute_data_vector_3x2pt_real_sizes():
      sizes.append(int(s))
    total_size = int(np.sum(sizes))

    # The full 3x2pt data vector is three blocks end to end:
    # xi (0), gammat (1), wtheta (2), lengths `sizes`. Collect
    # the global positions (indices into the full vector)
    # belonging to the requested probe's blocks.
    block_ranges = []
    for block_id in cls.PROBE_BLOCKS[probe]:
      # this block starts past all earlier blocks' lengths and
      # runs contiguously for block_len.
      block_start = int(np.sum(sizes[:block_id]))
      block_len   = sizes[block_id]
      block_ranges.append(
        np.arange(block_start, block_start + block_len))
    # one flat array of every global index in this probe.
    block_global = np.concatenate(block_ranges)

    # mask is the full-vector keep/drop flag (1 = unmasked).
    # mask[block_global] picks out this probe's entries in
    # block order; nonzero(...)[0] gives the surviving offsets
    # within block_global, e.g. block_global [10,11,12,13,14,
    # 15] with 12 and 14 masked gives kept_cols [0,1,3,5].
    mask = np.asarray(ci.get_mask())
    kept_cols = np.nonzero(mask[block_global] > 0)[0]
    # dest_idx = those survivors as positions in the full 3x2pt
    # vector, the index everything downstream uses (squeeze,
    # unsqueeze, center, Cinv_sq). For xi it equals kept_cols
    # (the xi block starts at 0); for gammat/wtheta the block
    # starts further in, so only the global dest_idx is correct.
    dest_idx  = block_global[kept_cols]

    cov = np.asarray(ci.get_cov_masked(), dtype="float64")
    Cb  = cov[np.ix_(dest_idx, dest_idx)]
    lam, U = np.linalg.eigh(Cb)
    sqrt_lam = np.sqrt(lam)

    Cinv = np.asarray(ci.get_inv_cov_masked(),
                      dtype="float64")

    # center lives in the full vector, so index it by the
    # global dest_idx, not block-local kept_cols.
    center = np.asarray(dv_center)[dest_idx]

    return cls(device=device,
               total_size=total_size,
               dest_idx=dest_idx,
               evecs=U,
               sqrt_ev=sqrt_lam,
               Cinv=Cinv,
               center=center,
               dtype=dtype,
               section_sizes=sizes,
               probe=probe)

  def state(self):
    """Collect the persistable transform, keys matching __init__.

    Move everything to cpu for saving; include dtype so
    from_state rebuilds the basis and Cinv at the run's
    precision. section_sizes / probe join only when set (a
    from_cosmolike geometry): section_sizes as a small long
    tensor, a clean numeric round-trip through the h5 writer
    that __init__ normalizes back to an int list; probe as a
    string. A geometry that predates the keys has neither, so
    state() omits them and from_state leaves both None.

    Returns:
      the mapping from_state(device, state()) rebuilds the identical
      geometry from.
    """
    st = {
      "total_size": self.total_size,
      "dest_idx":   self.dest_idx.cpu(),
      "evecs":      self.evecs.cpu(),
      "sqrt_ev":    self.sqrt_ev.cpu(),
      "Cinv":       self.Cinv.cpu(),
      "center":     self.center.cpu(),
      "dtype":      self.dtype,
    }
    if self.section_sizes is not None:
      st["section_sizes"] = torch.tensor(self.section_sizes,
                                         dtype=torch.long)
    if self.probe is not None:
      st["probe"] = self.probe
    # the heads' bin split, present only after build_shear_angle_map
    # attached it (a needs_bins training run). Persisted so a saved
    # head artifact rebuilds from the files alone — rebuild_emulator
    # must never need ROOTDIR data files (the dataset ini / n(z) the
    # attach reads). A trunk-only run never has the attributes, so its
    # state() simply omits the keys; a head artifact saved without
    # them is refused loudly by _rebuild_model.
    if hasattr(self, "bin_sizes"):
      st["bin_sizes"] = torch.tensor(self.bin_sizes, dtype=torch.long)
    if hasattr(self, "pm_kept"):
      st["pm_kept"] = torch.tensor(np.asarray(self.pm_kept),
                                   dtype=torch.long)
    if hasattr(self, "head_pad_idx"):
      st["head_pad_idx"] = self.head_pad_idx.cpu()
      st["head_valid_mask"] = self.head_valid_mask.cpu().to(torch.uint8)
    return st

  # --- low-level transforms ---
  def squeeze(self, dv):
    """Keep only the unmasked entries of the full dv.

    dv has shape (B, total_size), the full 3x2pt data
    vector; the result is (B, n_keep). B = batch size
    (cosmologies in the minibatch). Indexing columns by
    dest_idx (global positions) makes a copy, not a view,
    and works for any probe block.

    Arguments:
      dv = (B, total_size) full 3x2pt data vectors.

    Returns:
      (B, n_keep) the unmasked entries, in dest_idx order.
    """
    # dv[:, dest_idx] is fancy (advanced) indexing: dest_idx is
    # a 1-D LongTensor of column numbers, so this gathers those
    # columns in that order for every row, returning a new
    # (B, n_keep) tensor.
    return dv[:, self.dest_idx]

  def unsqueeze(self, sq):
    """Scatter the unmasked entries into a full vector.

    Place the (B, n_keep) kept entries at their dest_idx slots in a fresh
    (B, total_size) zero tensor, so the full masked Cinv can
    be applied. This is a right inverse on kept coordinates because
    ``squeeze(unsqueeze(sq)) == sq``. It is not an inverse on an arbitrary
    full vector because masked-out slots stay 0.

    Arguments:
      sq = (B, n_keep) kept entries (squeeze output).

    Returns:
      (B, total_size) full vector, sq scattered to dest_idx, 0
      elsewhere.
    """
    full = torch.zeros(sq.shape[0],
                       self.total_size,
                       dtype=sq.dtype,
                       device=sq.device)
    # Fancy-index assignment: full[:, dest_idx] = sq writes
    # column sq[:, k] into full's column dest_idx[k] for every
    # row at once, scattering the n_keep kept values to their
    # global slots (other columns stay 0). NB: this method is
    # the geometry's own "unsqueeze" (scatter to the full
    # vector), not torch's tensor.unsqueeze, which only inserts
    # a size-1 axis into a shape.
    full[:, self.dest_idx] = sq
    return full

  def whiten(self, centered_sq):
    """Centered, squeezed dv -> whitened target.

    Rotate into the covariance eigenbasis (@ evecs) and
    divide by sqrt_ev: decorrelated, unit variance. Computed
    in self.dtype, returned float32 to match the model and
    keep the dv chunk single.

    Arguments:
      centered_sq = (B, n_keep) squeezed dv with the center
                    already subtracted.

    Returns:
      (B, n_keep) whitened target, float32.
    """
    y = (centered_sq.to(self.dtype) @ self.evecs)
    return (y / self.sqrt_ev).float()

  def unwhiten(self, whitened_sq):
    """Exact inverse of whiten, in self.dtype.

    Multiply by sqrt_ev and rotate back (@ evecs.T). evecs
    is orthonormal, so this inverts whiten exactly. The chi2
    contracts the result with Cinv (same dtype).

    Arguments:
      whitened_sq = (B, n_keep) whitened kept-entry vector.

    Returns:
      (B, n_keep) un-whitened (centered) residual/dv, in self.dtype.
    """
    w = whitened_sq.to(self.dtype)
    return (w * self.sqrt_ev) @ self.evecs.T

  # --- high-level: raw dv <-> network space ---
  def encode(self, dv):
    """Raw full dv -> network target.

    Squeeze to the kept entries, subtract the center, then
    whiten.

    Arguments:
      dv = (B, total_size) full 3x2pt data vectors.

    Returns:
      (B, n_keep) whitened target the network predicts.
    """
    return self.whiten(self.squeeze(dv) - self.center)

  def decode(self, whitened_sq):
    """Network output -> physical dv.

    Unwhiten, then add the center back.

    Arguments:
      whitened_sq = (B, n_keep) whitened network output.

    Returns:
      (B, n_keep) physical (kept-entry) data vector.
    """
    return self.unwhiten(whitened_sq).float() + self.center


class DiagonalGeometry(DataVectorGeometry):
  """
  DataVectorGeometry with diagonal whitening: scale each kept
  element by its marginal error bar sigma = sqrt(diag(cov)), no
  rotation. Unlike full cov-eigenbasis whitening, this keeps the
  data-vector order (theta within each bin), so an axis-aware
  model, a 1D CNN over the output (ResCNN), sees the real
  theta axis instead of a scrambled eigenbasis.

  Targets are unit-marginal-variance but not decorrelated, so
  ||pred - target||^2 is the marginal chi2, not the full one. The
  reported chi2 is unchanged, the inherited chi2 multiplies
  sigma back and contracts with the full Cinv_sq, so keep the
  explicit Cinv contraction; do not 'simplify' the loss to MSE.

  encode/decode/chi2 inherit unchanged (calling the overridden
  whiten/unwhiten). sigma is read off the stored
  eigendecomposition and cached on first use.
  """
  _sigma = None     # cached per-element sigma (lazy)

  def _diag_sigma(self):
    """Per-element marginal sigma sqrt(diag(cov)), cached (lazy).

    cov = U diag(ev) U^T, so diag_i = sum_k (U_ik * sqrt_ev_k)^2.

    Returns:
      (n_keep,) per-element sigma.
    """
    if self._sigma is None:
      self._sigma = torch.sqrt(
        ((self.evecs * self.sqrt_ev) ** 2).sum(1))
    return self._sigma

  def whiten(self, centered_sq):
    """Scale each element by 1/sigma (no rotation, theta order kept).

    Arguments:
      centered_sq = (B, n_keep) squeezed dv, center subtracted.

    Returns:
      (B, n_keep) diagonally-whitened target, float32.
    """
    return (centered_sq.to(self.dtype)
            / self._diag_sigma()).float()

  def unwhiten(self, whitened_sq):
    """Exact inverse: multiply each element by its sigma.

    Arguments:
      whitened_sq = (B, n_keep) diagonally-whitened vector.

    Returns:
      (B, n_keep) un-whitened (centered) vector, in self.dtype.
    """
    return whitened_sq.to(self.dtype) * self._diag_sigma()


class BlockDiagonalGeometry(DataVectorGeometry):
  """
  DataVectorGeometry with block-diagonal whitening: each
  tomographic bin (xi+/-, source pair) is whitened by its own
  within-bin covariance sub-block, so whiten/unwhiten never mix
  bins. This makes the whitened target per-bin separable (one
  ResMLP head per bin), keeping unit-variance, decorrelated
  outputs within each bin.

  The chi2 is unchanged: Cinv_sq stays the full kept-block
  precision, so unwhiten (per-bin) -> full physical residual ->
  full Mahalanobis keeps every cross-pair correlation. Only the
  target basis is per-bin, not the metric.

  Needs geom.bin_sizes (per-bin kept-element counts, contiguous
  in dest_idx order, summing to n_keep) from
  build_shear_angle_map(geom). The per-bin basis is built lazily
  on first whiten, from the kept-block covariance rebuilt from
  the inherited (global) evecs/sqrt_ev.
  """

  # Lazy per-bin whitening cache. Class-level None so the
  # instance starts "not built"; _build_block fills them on the
  # first whiten call. Lazy (not in __init__) because bin_sizes
  # is attached later, by build_shear_angle_map, which runs
  # after the constructor.
  _b_evecs  = None     # list: per-bin rotation matrices V
  _b_sqrt   = None     # list: per-bin sqrt(eigenvalues) (scale)
  _b_slices = None     # list: per-bin column slice into the dv

  def _build_block(self):
    """Build the per-bin whitening bases, once, on first use.

    Reconstructs the kept-block covariance from the parent's stored
    eigendecomposition, then eigendecomposes each bin's within-bin
    sub-block (numpy eigh, float64 on the CPU) and stores one
    rotation, scale, and column slice per bin on the instance. Runs
    lazily because bin_sizes is attached by build_shear_angle_map
    after the constructor.
    """
    # Guard: the per-bin split must already exist. bin_sizes is
    # set by build_shear_angle_map; fail loudly if it didn't run.
    assert hasattr(self, "bin_sizes"), (
      "run build_shear_angle_map(geom) before using a "
      "BlockDiagonalGeometry (need bin_sizes)")

    # Device the geometry tensors live on (where the per-bin
    # basis must end up so whiten avoids host<->device copies).
    dev = self.evecs.device

    # Reconstruct the kept-block covariance Cb. The parent
    # eigendecomposed Cb = U diag(eigenvalues) U^T, stored
    # evecs = U and sqrt_ev = sqrt(eigenvalues), and threw Cb
    # away; rebuild it exactly as Cb = U diag(sqrt_ev**2) U^T.
    # In numpy float64: eigh wants float64 and MPS (Apple
    # Silicon) has no on-device float64, so compute on the CPU
    # and move the results to the device afterward.
    U = self.evecs.detach().cpu().numpy().astype("float64")
    s = self.sqrt_ev.detach().cpu().numpy().astype("float64")
    # (U * s**2) scales each column k of U by eigenvalue s_k**2
    # (broadcasting the length-n vector across the columns); the
    # @ U.T sums those rank-1 pieces back into Cb, U diag(s**2)
    # U^T without materializing the diagonal.
    Cb = (U * s**2) @ U.T

    # Build one whitening basis per bin. A bin occupies a
    # contiguous block of columns [start : start+n] of the kept
    # (squeezed) vector, contiguous because dest_idx is in
    # (xi+/-, pair, theta) order and bin_sizes are that order's
    # run lengths.
    self._b_evecs, self._b_sqrt, self._b_slices = [], [], []
    start = 0
    for n in self.bin_sizes:
      # the bin's within-bin covariance: the n x n diagonal
      # block of Cb (no cross-bin entries -> no cross-bin
      # mixing in whitening).
      block = Cb[start:start + n, start:start + n]
      # eigh returns ascending eigenvalues lam and orthonormal
      # eigenvectors V (columns): V is this bin's rotation,
      # sqrt(lam) its per-direction scale.
      lam, V = np.linalg.eigh(block)
      self._b_evecs.append(torch.as_tensor(
        V, dtype=self.dtype, device=dev))
      # clip(lam, 0) guards a tiny negative eigenvalue from
      # float noise (a covariance is positive semidefinite, but
      # a near-zero eigenvalue can come out slightly negative);
      # sqrt of a negative would be nan.
      self._b_sqrt.append(torch.as_tensor(
        np.sqrt(np.clip(lam, 0.0, None)),
        dtype=self.dtype, device=dev))
      # remember where this bin sits so whiten/unwhiten write
      # back into the right columns.
      self._b_slices.append(slice(start, start + n))
      start += n   # advance to the next bin's first column

  def whiten(self, centered_sq):
    """Per-bin: rotate into the bin's eigenbasis, scale to 1.

    Arguments:
      centered_sq = (B, n_keep) squeezed dv, center subtracted.

    Returns:
      (B, n_keep) block-whitened target (each bin decorrelated
      within itself, no cross-bin mixing), float32.
    """
    # Build the per-bin basis on first use (cached afterward).
    if self._b_evecs is None:
      self._build_block()
    # Cast to the geometry's compute dtype (float32 default).
    x = centered_sq.to(self.dtype)
    # Preallocate the output (x's shape/dtype/device), filled
    # bin-by-bin.
    out = torch.empty_like(x)
    # Per bin: take its columns x[:, sl], rotate into the bin's
    # eigenbasis (@ V), divide by the scale (sqrt of eigenvalues)
    # for unit variance, write back into that bin's slice. No
    # cross-bin mixing, each bin uses only its own V / sb / sl.
    for V, sb, sl in zip(self._b_evecs, self._b_sqrt,
                         self._b_slices):
      out[:, sl] = (x[:, sl] @ V) / sb
    # Return float32 to match the model output and the loss.
    return out.float()

  def unwhiten(self, whitened_sq):
    """Exact inverse of whiten, per bin (V orthonormal).

    Arguments:
      whitened_sq = (B, n_keep) block-whitened vector.

    Returns:
      (B, n_keep) un-whitened (centered) vector, in self.dtype.
    """
    if self._b_evecs is None:
      self._build_block()
    w = whitened_sq.to(self.dtype)
    out = torch.empty_like(w)
    # Inverse of whiten, bin by bin: multiply by the scale
    # (undo the /sb), then rotate back with V transpose. V is
    # orthonormal so V @ V.T = I, an exact inverse. Returned in
    # self.dtype (not .float()) because the chi2 contracts this
    # residual with Cinv at the geometry's dtype.
    for V, sb, sl in zip(self._b_evecs, self._b_sqrt,
                         self._b_slices):
      out[:, sl] = (w[:, sl] * sb) @ V.T
    return out


def build_shear_angle_map(geom, 
                          data_dir="lsst_y1",
                          dataset="lsst_y1_M1_GGL0.05.dataset"):
  """
  Attach the cosmic-shear angle/tomography map to a geometry.

  Per kept (unmasked) element: its angular scale theta, the two
  source redshifts of its tomographic pair, and its xi+/- branch
  (pm_kept: 0 = xi_plus, 1 = xi_minus). Also stores the
  block-level metadata the matrix-layout plotting needs and the
  per-bin kept-element counts bin_sizes (bin = (xi+/-, source
  pair); contiguous in dest_idx order, summing to n_keep) that a
  per-bin BlockDiagonalGeometry / ParallelResMLP split on (the
  Returns block lists each). Reads the dataset ini and the
  source n(z) file only, no cosmolike.

  Assumes xi ordering xi_plus then xi_minus, each looping source
  pairs (i<=j) outer and theta inner. Verify against your
  cosmolike layout; reorder below if not.

  Arguments:
    geom     = DataVectorGeometry (or subclass) for probe "xi";
               geom.dest_idx gives the kept within-block
               positions.
    data_dir = data folder under external_modules/data.
    dataset  = .dataset ini naming the n(z) / binning.

  Returns:
    geom, with new attributes: theta_kept [arcmin], zsrc_i,
    zsrc_j, pm_kept (each (n_keep,)); theta_centers [arcmin]
    (ntheta,), z_src (ntomo,), ntheta, source_ntomo, xi_size,
    bin_sizes (list, len = #non-empty bins, sum = n_keep),
    head_pad_idx (the original flattened xi slot of each survivor),
    and head_valid_mask (the complete physical-bin by theta rectangle).
  """
  # Locate and parse the dataset ini (binning + n(z) file).
  # deferred (25M-37): getdist is imported here, not at module level (this
  # path reads the ini + n(z) file only, no cosmolike).
  from getdist import IniFile
  RD   = os.environ["ROOTDIR"]
  path = os.path.normpath(
    os.path.join(RD, "external_modules/data", data_dir))
  ini  = IniFile(os.path.join(path, dataset))
  ntheta = ini.int("n_theta")              # angular bins
  tmin   = ini.float("theta_min_arcmin")
  tmax   = ini.float("theta_max_arcmin")
  ns     = ini.int("source_ntomo")         # source z-bins
  source_file = ini.relativeFileName("nz_source_file")

  # theta bin centers as the geometric mean of the log-spaced
  # edges (log-spaced because xi is plotted/binned in log-theta).
  edges   = np.logspace(np.log10(tmin), np.log10(tmax),
                        ntheta + 1)
  centers = np.sqrt(edges[:-1] * edges[1:])

  # Each source bin's peak redshift: load n(z), and for source k
  # take the theta where its n(z) column is largest (the
  # delta-function source-plane approximation).
  nz    = np.loadtxt(source_file)
  zcol  = nz[:, 0]                          # the z grid
  z_src_vals = []
  for k in range(ns):
    z_src_vals.append(zcol[np.argmax(nz[:, k + 1])])
  z_src = np.array(z_src_vals)

  # Rebuild the full cosmic-shear data-vector layout, element by
  # element, in the exact order cosmolike writes it: xi_plus then
  # xi_minus (the pm loop), source pairs (i<=j) the middle loop,
  # theta innermost. Per element, record its theta, its two
  # source redshifts, and its xi+/- branch.
  pairs = []
  for i in range(ns):
    for j in range(i, ns):
      pairs.append((i, j))
  npair = len(pairs)
  th_full, zi_full, zj_full, pm_full = [], [], [], []
  for _pm in range(2):                      # 0 = xi+, 1 = xi-
    for (i, j) in pairs:
      for t in range(ntheta):
        th_full.append(centers[t])
        zi_full.append(z_src[i])
        zj_full.append(z_src[j])
        pm_full.append(_pm)
  # to numpy, so they fancy-index by the kept positions.
  th_full = np.asarray(th_full)
  zi_full = np.asarray(zi_full)
  zj_full = np.asarray(zj_full)
  pm_full = np.asarray(pm_full)

  # Full cosmic-shear block length, and the kept positions
  # (dest_idx) as plain ints. assert it really is xi-only: every
  # kept index falls inside the cosmic-shear block.
  xi_size = 2 * npair * ntheta
  keep = geom.dest_idx.cpu().numpy()
  assert keep.max() < xi_size, (
    "geometry is not cosmic-shear-sized; "
    "the analytic scaling is xi-only")

  # Pick out the per-element metadata for the kept elements only
  # (masked ones dropped), in dest_idx order.
  geom.theta_kept    = th_full[keep]   # arcmin
  geom.zsrc_i        = zi_full[keep]
  geom.zsrc_j        = zj_full[keep]
  geom.pm_kept       = pm_full[keep]   # 0 = xi+, 1 = xi-
  geom.theta_centers = centers         # arcmin
  geom.z_src         = z_src
  geom.ntheta        = ntheta
  geom.source_ntomo  = ns
  geom.xi_size       = xi_size

  # Per-bin sizes for the per-bin model/geometry. A bin =
  # (xi+/-, source pair) = a contiguous run of kept elements
  # sharing the same (pm, zsrc_i, zsrc_j), contiguous because
  # the layout above is pm/pair outer, theta inner, so one bin's
  # thetas sit together. Run-length encode: walk the kept
  # elements in order; start a new bin whenever the key changes,
  # else increment the current bin's count. .tolist() gives plain
  # Python scalars so the tuple comparison is exact.
  bkeys = list(zip(geom.pm_kept.tolist(),
                   geom.zsrc_i.tolist(),
                   geom.zsrc_j.tolist()))
  bin_sizes = []
  for k, key in enumerate(bkeys):
    if k == 0 or key != bkeys[k - 1]:
      bin_sizes.append(1)          # first element of a new bin
    else:
      bin_sizes[-1] += 1           # another theta in this bin
  # len(bin_sizes) = number of non-empty bins (a fully-masked
  # bin never appears); sum(bin_sizes) = n_keep.
  geom.bin_sizes = bin_sizes
  # The structured-head rectangle retains every physical bin, including an
  # all-masked bin. In the full xi layout each kept global index is already
  # its exact flattened (pm, source pair, theta) slot. Compacting nonempty
  # bins would shift every later physical channel after an empty row.
  geom.head_pad_idx = torch.as_tensor(keep, dtype=torch.long)
  valid = torch.zeros((2 * npair, ntheta), dtype=torch.bool)
  valid.reshape(-1)[geom.head_pad_idx] = True
  geom.head_valid_mask = valid
  return geom
