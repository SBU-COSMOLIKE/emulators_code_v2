"""Memory sizing and the regime-aware data loaders.

This module decides where each source's data lives and hands the
training loop two loader functions (rows -> whitened param inputs,
rows -> encoded targets) that hide that choice; each is a closure, a
small function that remembers the arrays it was built around.
compute_batch_byte_terms names every per-batch buffer,
compute_batch_size_bytes sums those terms, and
compute_model_size_bytes and batches_per_load plan resident and chunk
memory. _build_loaders_one picks one of three regimes against a VRAM
budget (VRAM = the GPU's own memory): pre-encode the target set on the
GPU, stream from RAM, or stream from a disk memmap; it reports the
bytes it made resident. build_loaders runs it per source (train, then
val against the reduced budget) and returns the data dict the loop
consumes.

The regime ladder, per source:

    dv rows (disk memmap or RAM)      raw params C
       │                                 │  param_geometry.encode
       │                                 ▼
       │                              C_used (n_used, Ncosmo)
       │                              resident on the GPU
       │
       │  enc_dvs + resident < 0.8 * budget ?
       │
       ├─ yes -> regime 1 (resident): encode every target once
       │         (chunked), hold (n_used, tgt_dim) on the GPU;
       │         load_dv(rows) is a pure index, no per-epoch I/O.
       ├─ no, dv is a RAM ndarray -> regime 2 (RAM stream):
       │         chunks RAM -> GPU each epoch, encoded on the
       │         fly (pinned host memory on CUDA — page-locked
       │         RAM the GPU can copy from directly).
       └─ no, dv is a disk memmap -> regime 3 (disk stream):
                 the same chunk path, reads hit the disk.

    (legend: n_used = distinct rows this source loads; Ncosmo =
     parameter count; enc_dvs = bytes of the encoded target set;
     resident = model + Cinv + encoded params, bytes pinned on
     the GPU for the whole run, where Cinv is the inverse of the
     data covariance matrix — the chi2's weight matrix; tgt_dim =
     target width, out_dim unless the loss stages a wider one via
     target_dim; budget = the VRAM bytes this source may plan
     against; 0.8 = the planning headroom factor (plan against
     0.8 * budget, leaving ~20% for allocator slack and
     fragmentation), as in batches_per_load.)

PS: a loader is a closure ``load(rows) -> tensor``. Its row numbers address
the active source array. For a compact RAM copy they are local coordinates
inside that copy. For a disk-backed source they are original dump-row
coordinates. The loader hides whether targets are resident on the GPU,
streamed from RAM, or read from a disk memmap, so the training loop uses the
same call in every regime. Two loaders per source: load_C for
whitened param inputs, load_dv for encoded targets. whitened = rotated
into the covariance eigenbasis and scaled to unit variance, so the
components are decorrelated (the form the network sees). encoded = a data
vector through the geometry's encode (keep unmasked entries, subtract the
training mean, whiten), the form trained against. resident = held in GPU
memory for the whole run, not re-loaded per batch. dump = the full
on-disk array from the data-generation run, one row per cosmology (the
dv dump is the .npy, the param dump the .txt); memmap = a NumPy array
backed by that file, read in slices so it is never loaded whole;
squeeze = keep only the unmasked dv entries (the geometry's squeeze),
the smaller vector the network emulates.
"""

import numpy as np
import torch


def compute_batch_byte_terms(
    model,
    bs,
    sample_dims,
    dv_len=3000,
    target_dim=None,
    target_dtype=None):
  """Name every GPU-memory term owned by one training batch.

  autograd is PyTorch's automatic-differentiation engine: during the
  forward pass it saves the intermediate tensors (the "activations")
  it will need to compute gradients in the backward pass, and those
  saved tensors are usually the batch's largest memory term.  This
  probe measures them on a real forward pass (a spy pair of
  saved-tensor hooks, see the body) and adds the batch input/output
  buffers and the chi2's per-batch float64 scratch. The target is a
  separate term because packed-target losses can stage more values
  than the model predicts. Only shapes matter, so the probe runs on
  zeros.

  Arguments:
    model       = the network; probed with one dummy forward.
    bs          = minibatch size (rows per gradient step) the
                  estimate is for.
    sample_dims = shape of one model input, no batch axis (the
                  cosmo param vector: (Ncosmo,)).
    dv_len      = full dv length the chi2 un-squeezes to (~3000
                  is conservative; avoids a cosmolike query).
    target_dim  = number of staged target values per row. None means
                  the target has the model output's element count.
    target_dtype = dtype used by the target-staging boundary. None
                   means the target has the model output's dtype.

  Returns:
    A dictionary whose integer values are the named byte terms. A
    separate entry for each buffer makes a wider target visible in
    reports and tests.
  """
  # model.parameters() iterates the weight tensors; next() grabs
  # the first. Its .device is where it lives; put x there too.
  dev = next(model.parameters()).device

  # dummy input batch (bs, *sample_dims). Only shapes matter for
  # memory, not values -> zeros.
  x = torch.zeros(bs, *sample_dims, device=dev)

  total = 0  # running byte count of saved tensors
  def pack(t):
    """Spy on one saved activation's byte size (autograd pack hook).

    autograd calls this the moment a tensor is saved during forward and
    stores what it returns. The size is recorded and t returned
    unchanged. (+= alone would rebind total as a local; nonlocal points
    it at the outer running count.)

    Arguments:
      t = the tensor autograd is saving for backward.

    Returns:
      t unchanged.
    """
    nonlocal total
    total += t.numel() * t.element_size()
    return t
  def unpack(stored):
    """Return the stored tensor unchanged (the required inverse hook).

    Backward never runs here, but saved_tensors_hooks requires the
    pack/unpack pair to round-trip, so the stored value is handed back.

    Arguments:
      stored = whatever pack returned for this saved tensor.

    Returns:
      stored, unchanged.
    """
    return stored

  # saved_tensors_hooks customizes how activations are stored
  # between forward and backward (to save memory): pack
  # transforms each on the way in (e.g. compress), unpack
  # reverses it out. The pair must round-trip: unpack(pack(t))
  # == t. Here the no-op pair just spies on the sizes. See the
  # PyTorch saved-tensor-hooks tutorial (URL split to fit the
  # width; rejoin with no space):
  #   https://docs.pytorch.org/tutorials/intermediate/
  #   autograd_saved_tensors_hooks_tutorial.html
  hooks = torch.autograd.graph.saved_tensors_hooks  # alias
  with hooks(pack, unpack):
    # saving happens during forward, so total is complete the
    # instant model(x) returns.
    out = model(x)

  # Device buffers tied to this batch include the input and the model
  # output. element_size() = bytes per element (float32 -> 4,
  # float64 -> 8).
  in_bytes  = x.numel() * x.element_size()
  out_bytes = out.numel() * out.element_size()

  # The ordinary target has the same number of elements and the same
  # dtype as the output. A packed-target loss passes its real width and
  # staging dtype explicitly. The staging boundary owns those values;
  # this function only converts them to a byte count.
  if target_dim is None:
    target_elements = out.numel()
  else:
    target_elements = bs * target_dim

  if target_dtype is None:
    target_element_size = out.element_size()
  else:
    target_element_size = torch.empty(
      (),
      dtype=target_dtype).element_size()
  target_bytes = target_elements * target_element_size

  # the chi2 runs outside model(x), so the hook never sees it.
  # Per batch it builds a few full-length float64 buffers (the
  # unsqueezed residual, the r @ Cinv product, the copy autograd
  # saves for backward), budget three (bs, dv_len) doubles.
  chi2 = 3 * bs * dv_len * 8

  return {
    "saved_activations": total,
    "input": in_bytes,
    "model_output": out_bytes,
    "target": target_bytes,
    "chi2_scratch": chi2,
  }


def compute_batch_size_bytes(
    model,
    bs,
    sample_dims,
    dv_len=3000,
    target_dim=None,
    target_dtype=None):
  """Return the total GPU bytes owned by one training batch.

  Keeping this integer-returning wrapper preserves the original public
  call for ordinary targets. The named-term function
  (compute_batch_byte_terms) is the single owner of the arithmetic.

  Arguments:
    model        = the live model (its saved-activation bytes are
                   measured by a probe forward).
    bs           = the batch size the bytes are computed for.
    sample_dims  = the per-sample input shape (without the batch axis).
    dv_len       = the data-vector length entering the chi2 (the Cinv
                   contraction size).
    target_dim   = the encoded target width, when it differs from
                   dv_len; None for ordinary targets.
    target_dtype = the encoded target dtype; None for float32.

  Returns:
    the batch's total byte count as an int.
  """
  terms = compute_batch_byte_terms(
    model=model,
    bs=bs,
    sample_dims=sample_dims,
    dv_len=dv_len,
    target_dim=target_dim,
    target_dtype=target_dtype)
  return sum(terms.values())


def compute_model_size_bytes(model):
  """Bytes the model keeps resident for the whole run.

  Counts weights, gradients, and the optimizer's per-parameter state,
  budgeted at the worst typical case. The optimizer is the update
  rule (SGD, Adam, ...) that turns gradients into weight changes;
  most rules keep extra running quantities per parameter, each the
  size of the parameter tensor itself. opt_state = how many such
  state tensors the rule keeps:

      SGD (plain)                    0
      SGD+momentum, Adagrad,         1
        RMSprop (default)
      Adam, AdamW, Adamax, NAdam     2
      Adam(amsgrad), RMSprop         3   <- worst typical
        (centered + momentum)

  Arguments:
    model = the network whose parameters are counted.

  Returns:
    bytes = n_params * element_size * (2 + opt_state), i.e.
    weights(1) + grads(1) + opt_state buffers.
  """
  opt_state = 3
  # total parameter elements across all weight tensors.
  p = 0
  for t in model.parameters():
    p += t.numel()
  esize = next(model.parameters()).element_size()  # bytes
  # weights(1) + grads(1) + opt_state buffers
  return p * esize * (2 + opt_state)

def batches_per_load(
    model,
    bs,
    sample_shape,
    budget,
    dv_len=3000,
    target_dim=None,
    target_dtype=None):
  """Batches per streamed chunk that fit the VRAM budget.

  Resident memory keeps the established model-plus-precision-matrix
  definition: the precision matrix is Cinv, the inverse of the data
  covariance, held in float64 as the chi2's weight. The streamed
  chunk gets what is left of 0.8 * budget, divided by one batch's
  cost.

  Arguments:
    model        = the network (sizes the resident + probe cost).
    bs           = minibatch size.
    sample_shape = shape of one model input, no batch axis.
    budget       = VRAM bytes to plan against; explicit (real
                   free VRAM in research, emulated GPU_MEM in
                   class).
    dv_len       = full dv length (sizes Cinv and the chi2
                   scratch).
    target_dim   = staged target width. None preserves the ordinary
                   output-shaped target calculation.
    target_dtype = dtype used to stage the target. None preserves the
                   ordinary output-dtype calculation.

  Returns:
    number of bs-row batches per streamed chunk.

  Raises:
    MemoryError if the planning allowance cannot hold the resident
    state and one complete batch.
  """
  cinv = dv_len * dv_len * 8
  resident = compute_model_size_bytes(model) + cinv

  # The planner reserves 20 percent of the declared budget for the
  # allocator. Keep the existing 0.8 calculation so an ordinary-target
  # run has the same chunk boundary as before this repair. The integer
  # value is used only in the human-readable error report.
  planning_allowance = 0.8 * budget
  available = int(planning_allowance)
  free = planning_allowance - resident

  terms = compute_batch_byte_terms(
    model=model,
    bs=bs,
    sample_dims=sample_shape,
    dv_len=dv_len,
    target_dim=target_dim,
    target_dtype=target_dtype)
  per_batch = sum(terms.values())
  required = resident + per_batch

  if required > planning_allowance:
    term_text = ", ".join(
      name + "=" + str(value)
      for name, value in terms.items())
    raise MemoryError(
      "streaming plan cannot hold resident state and one complete batch: "
      f"required={required}, available={available}, resident={resident}, "
      + term_text)

  return int(free // per_batch)


def _build_loaders_one(device, C, dv, idx,
                       param_geometry, chi2fn,
                       model, bs, budget,
                       dv_len=3000, CHUNK=1000):
  """Build the two data loaders for one source and place its data.

  One source is a train or val file. Both returned loaders take row
  coordinates into the active C/dv arrays; a `slots` helper maps
  those to local positions in the compact resident subset (see
  below), so the rest of the pipeline is identical wherever the data
  ended up (see Returns for the four outputs). The params are always
  encoded once and kept on the GPU (tiny, n_used x Ncosmo), so only
  the data vectors (the large array) change placement, by a memory
  ladder against `budget`:

    Regime 1 (resident gather): the encoded set fits, so
      pre-encode it once; a batch is pure on-device indexing.
    Regime 2 (RAM stream): does not fit but the dvs are an in-RAM
      ndarray; stream RAM->GPU a chunk at a time, encode on the
      fly (pinned memory on CUDA).
    Regime 3 (disk stream): the dvs exceed RAM (a np.memmap), so
      the same per-chunk path reads from disk.

  Resident memory = model + the chi2's Cinv (the inverse data
  covariance, the chi2's weight matrix) + the encoded params; the
  dvs get what is left (0.8 * budget - resident), and `fits` decides
  regime 1 versus streaming. The orchestrator subtracts the returned
  `used` before sizing the next source, so two sequential builds
  share one GPU without overrunning it.

  Works for the plain CosmolikeChi2 (encode takes the dv alone) and
  the param-aware losses (RescaledChi2 / ResidualBase /
  PCEResidualChi2 / PCERatioChi2), whose encode also takes this
  block's whitened params (the resident C_used rows) to build R or
  the PCE base; the `rescaled` flag branches encode. A loss may also
  stage a wider target via a target_dim attribute (see tgt_dim
  below).

  Arguments:
    device     = target device for the staged tensors.
    C          = active parameter array, (N, Ncosmo). It is either a
                 compact RAM copy or the full disk-backed source.
    dv         = active target array, (N, Ndv); ndarray -> regime 2,
                 np.memmap -> regime 3.
    idx        = row coordinates into the active C/dv arrays.
    param_geometry = ParamGeometry; .encode whitens raw params.
    chi2fn     = CosmolikeChi2 or RescaledChi2 (output geom).
    model      = network; read only to size resident memory.
    bs         = minibatch size; the chunk is a multiple of it.
    budget     = VRAM bytes to plan against.
    dv_len     = full dv length the chi2 unsqueezes to.
    CHUNK      = rows per block when pre-encoding (regime 1).
  Returns:
    load_C  = callable: active source rows -> whitened inputs.
    load_dv = callable: active source rows -> whitened targets.
    load    = rows per chunk chosen for this regime.
    used    = GPU bytes this source made resident.
  """
  ncosmo    = C.shape[1]
  # out_dim = model output width = the unmasked dv entries the
  # network predicts (dest_idx holds the kept positions;
  # .numel() counts them).
  out_dim   = chi2fn.dest_idx.numel()

  # tgt_dim = width of the target tensor this loader stages per
  # row. Normally just the encoded truth, one value per kept
  # entry, so tgt_dim == out_dim. One loss needs more room:
  # PCERatioChi2 forms pred = base * (1 + net_output), where
  # net_output is the model's fractional correction and base a
  # fixed reference dv (the frozen PCE). Rather than recompute
  # base every batch, it precomputes it once here and stages
  # [base ; truth] as one 2*n_keep-wide target, unpacked inside
  # the chi2.
  #
  # A loss requests that wider target via a `target_dim`
  # attribute. getattr(obj, "name", default) returns obj.name if
  # present, else `default` (never raises), so a loss without
  # target_dim falls back to out_dim and stages the plain truth
  #, the same opt-in pattern as needs_params below.
  tgt_dim   = getattr(chi2fn, "target_dim", out_dim)

  # Every target entering this loader is converted to float32 before
  # chi2fn.encode runs. Keep that dtype in one named value and pass it
  # to the memory planner, so the planner charges the same
  # representation that the loader stages.
  target_dtype = torch.float32
  target_element_size = torch.empty(
    (),
    dtype=target_dtype).element_size()

  # used_rows = the rows this source loads, sorted. idx is already a unique
  # set (stage_source refuses a duplicate upstream), so np.unique here is not
  # deduplicating; it SORTS into active storage order, which slots() assumes
  # and which makes a memmap read sequential. n_used counts them.
  used_rows = np.unique(idx)
  n_used    = len(used_rows)

  # loud (not assert: an assert is stripped under python -O, and a mismatched
  # dv width would then flow silently into encode).
  if dv.shape[1] != chi2fn.total_size:
    raise ValueError(
      "dv width " + str(dv.shape[1]) + " != loss total_size "
      + str(chi2fn.total_size) + " (the staged data vectors and the "
      "geometry disagree on the full vector length)")

  # encode the params once, resident on the GPU. For the
  # rescaled geometry these whitened params also let encode build
  # R.
  C_used = param_geometry.encode(
    torch.from_numpy(C[used_rows]).float().to(device))

  rescaled = getattr(chi2fn, "needs_params", False)

  model_bytes = compute_model_size_bytes(model)
  cinv        = dv_len * dv_len * 8
  enc_params  = n_used * ncosmo * 4
  resident    = (model_bytes + cinv + enc_params)

  def slots(rows):
    """Translate source-row numbers into resident-subset positions.

    The loaders stage the used rows into C_used / dv_used in sorted
    order, so a source coordinate can differ from its position in the
    resident subset. used_rows is the sorted set of active source
    coordinates; np.searchsorted gives each query's insertion index into
    that sorted array, and since every query row is itself in used_rows,
    the index IS its row in C_used / dv_used.

    Arguments:
      rows = numpy array of active source-row numbers.

    Returns:
      a long tensor of resident positions, on the training device.
    """
    local_pos = np.searchsorted(used_rows, rows)
    return torch.from_numpy(local_pos).to(device)

  def load_C(rows):
    """Fetch the encoded parameter rows for these source rows.

    Arguments:
      rows = numpy array of active source-row numbers.

    Returns:
      the (len(rows), ncosmo) encoded parameter tensor, resident on the
      device.
    """
    return C_used[slots(rows)]

  enc_dvs = n_used * tgt_dim * target_element_size
  fits    = enc_dvs + resident < 0.8 * budget

  if fits:
    # Regime 1: pre-encode every target, hold it on the GPU.
    dv_used = torch.empty(
      n_used,
      tgt_dim,
      dtype=target_dtype,
      device=device)
    for start in range(0, n_used, CHUNK):
      block = used_rows[start:start + CHUNK]
      # raw dvs for this block, on the device.
      dv_t = torch.from_numpy(dv[block]).to(
        dtype=target_dtype).to(device)
      if rescaled:
        # rescaled target: encode also needs this block's params.
        # C_used is in used_rows order, so the block is the local
        # slice start : start + len(block).
        params = C_used[start:start + len(block)]
        enc = chi2fn.encode(dv=dv_t, params_whitened=params)
      else:
        enc = chi2fn.encode(dv_t)
      dv_used[start:start + len(block)] = enc

    def load_dv(rows):
      """Fetch pre-encoded target rows resident on the device (regime 1).

      Arguments:
        rows = numpy array of active source-row numbers.

      Returns:
        the (len(rows), tgt_dim) encoded target tensor.
      """
      return dv_used[slots(rows)]

    bytes_per_row = (
      tgt_dim * target_element_size
      + ncosmo * C_used.element_size())
    vram_left     = 0.8 * budget - resident - enc_dvs
    fit_rows = max(bs, int(vram_left // bytes_per_row))
    load = min(len(idx), fit_rows)

  elif not isinstance(dv, np.memmap):
    # Regime 2: dvs live in CPU RAM.
    def load_dv(rows):
      """Fetch and encode target rows from CPU RAM (regime 2).

      The rows are copied to a pinned host buffer (pinned memory lets a
      CUDA transfer run asynchronously), moved to the device, then
      encoded there.

      Arguments:
        rows = numpy array of active source-row numbers.

      Returns:
        the (len(rows), tgt_dim) encoded target tensor on the device.
      """
      cpu = torch.from_numpy(dv[rows]).to(dtype=target_dtype)
      if device.type == "cuda":
        cpu = cpu.pin_memory()
      gpu = cpu.to(device)
      if rescaled:
        return chi2fn.encode(dv=gpu, params_whitened=load_C(rows))
      return chi2fn.encode(gpu)

    load = bs * batches_per_load(model=model,
                                 bs=bs,
                                 sample_shape=C.shape[1:],
                                 budget=budget,
                                 dv_len=dv_len,
                                 target_dim=tgt_dim,
                                 target_dtype=target_dtype)
  else:
    # Regime 3: dvs exceed RAM, read from the memmap.
    def load_dv(rows):
      """Read and encode target rows from the disk memmap (regime 3).

      Arguments:
        rows = numpy array of active source-row numbers.

      Returns:
        the (len(rows), tgt_dim) encoded target tensor on the device.
      """
      host = torch.from_numpy(dv[rows]).to(dtype=target_dtype)
      gpu  = host.to(device)
      if rescaled:
        return chi2fn.encode(dv=gpu, params_whitened=load_C(rows))
      return chi2fn.encode(gpu)

    load = bs * batches_per_load(model=model,
                                 bs=bs,
                                 sample_shape=C.shape[1:],
                                 budget=budget,
                                 dv_len=dv_len,
                                 target_dim=tgt_dim,
                                 target_dtype=target_dtype)

  used = enc_params + (enc_dvs if fits else 0)
  return load_C, load_dv, load, used


def build_loaders(device, train_set, val_set, param_geometry, 
                  chi2fn, model, bs, budget,
                  dv_len=3000, CHUNK=1000):
  """Build the train and val loaders and return the training data dict.

  Train and validation are passed as separate source dictionaries and
  receive separate loader closures through _build_loaders_one; the
  returned dict is what the training loop and eval_val consume.
  Separate file names do not prove that the files contain different
  cosmologies; this function does not test physical-row disjointness.
  The same training-built param_geometry and chi2fn transform both
  sources.

  Arguments:
    device     = target device.
    train_set  = training source dict:
                   "C"   active parameter array,
                   "dv"  active target array,
                   "idx" row coordinates into those active arrays.
    val_set    = validation source dict, same three keys.
    param_geometry, chi2fn, model, bs, budget, dv_len, CHUNK
               = forwarded to _build_loaders_one (see there);
                 the same geometry for both sources.
  Returns:
    data = nested dict, one sub-dict per source, both with
      the same keys:
        data["train"] = {load_C, load_dv, idx, load}
        data["val"]   = {load_C, load_dv, idx, load}
  """
  (load_C, load_dv, load,
   used_tr) = _build_loaders_one(device=device, 
                              C=train_set["C"], 
                              dv=train_set["dv"], 
                              idx=train_set["idx"],
                              param_geometry=param_geometry, 
                              chi2fn=chi2fn, 
                              model=model, 
                              bs=bs, 
                              budget=budget, 
                              dv_len=dv_len, 
                              CHUNK=CHUNK)

  # train is now resident on the GPU, so the val call plans
  # against a budget reduced by what train took. (model + Cinv
  # are shared, counted by each call.)
  (load_C_val, load_dv_val, load_val, 
   _) = _build_loaders_one(device=device, 
                           C=val_set["C"], 
                           dv=val_set["dv"], 
                           idx=val_set["idx"],
                           param_geometry=param_geometry, 
                           chi2fn=chi2fn,
                           model=model, 
                           bs=bs, 
                           budget=budget - used_tr,
                           dv_len=dv_len, 
                           CHUNK=CHUNK)

  return {
    "train": {
      "load_C": load_C,
      "load_dv": load_dv,
      "idx": train_set["idx"],
      "load": load,
    },
    "val": {
      "load_C": load_C_val,
      "load_dv": load_dv_val,
      "idx": val_set["idx"],
      "load": load_val,
    }
  }
