---
name: dev-machine-mac-m2-32gb
description: "Dev/test machine is a Mac M2 with 32 GB unified memory, MPS backend; real training runs on NVIDIA. Code must run on both."
metadata: 
  node_type: memory
  type: project
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

The notebook is developed and tested on a **Mac M2, 32 GB unified memory**,
PyTorch **MPS** backend. Real research training happens on **NVIDIA** GPUs. The
examples must run on both, so anything device-specific branches on
`device.type` (`"mps"` vs `"cuda"`).

MPS constraints that shape the code:
- **No float64 on device** — `run_sum` and the geometry `dtype` fall back to
  float32 on MPS (CUDA/CPU keep float64).
- **AMP dtype differs** — fp16 on MPS, bf16 on CUDA/CPU (fp16 may need a
  GradScaler; bf16 does not).
- **`torch.compile`** — `reduce-overhead` (CUDA graphs) is CUDA-only; on MPS
  run eager or default-mode compile.
- **No `pin_memory` / `torch.cuda.mem_get_info`** — CUDA-only; skip on MPS.
- **Unified memory** — the `build_loaders` budget is a slice of the *same*
  32 GB pool that also holds macOS, Python, and the host-side `dv0`/`C0`
  arrays; there is no separate VRAM, so don't budget near 32 GB (~16 GB is a
  safe ceiling for a real run). Related: [[locate-notebook-edits-by-context]].

## Testing when the module won't import (2026-07-06)

The Mac dev `python3` (3.14) used for Implementer-side gates has **numpy +
stdlib only** — no torch, cosmolike, matplotlib, getdist, or pyyaml — and the
package modules import those at load, so `import emulator.*` fails on the Mac.
Validate pure logic anyway by **exec-extracting** the standalone functions
from source: `ast`-parse the file, take the target `FunctionDef` / `Assign`
nodes by name, and `exec` their exact source span into a `{"np": np}`
namespace. Ran phys_cut_idx, `validate_param_cuts`, `_window_masks`, etc. this
way — each against an independent numpy reference AND an exact cross-check vs
its source-of-truth twin (e.g. plotting masks vs phys_cut_idx keep at sample
coords). For structure/style, `py_compile` (compiles without importing) and
`ast`/`tokenize` scans (caps / ` -- ` / legend / comprehension / width /
keyword-vs-signature) gate the whole tree with no imports. The rule: keep the
logic that needs Mac-side gating in a **standalone pure function** (no torch
inside), so it is exec-extractable. Runtime-import and any
cosmolike/matplotlib/getdist checks are workstation-deferred
([[test-workstation-gpus]], [[session-resume-2026-07-06]]).
