# Project purpose and durable development milestones

This note explains the project-wide decisions that still shape the emulator
library. The note does not reproduce chronological development history. Git
preserves retired implementations, while the tracked backlog records unfinished
work.

Read [`MEMORY.md`](MEMORY.md) first. Use the topic notes for detailed model,
training, artifact, data, and family requirements.

## Terms used in this note

An **emulator** is a trained, fast approximation to an expensive scientific
calculation. An **artifact** is one saved emulator result: trained weights plus
the scientific and structural facts needed to rebuild them. **Publication**
places a complete, checked result at the final path used by readers. An
**identity** is the saved set of facts or byte fingerprints used to decide
whether two datasets, runs, or artifacts are the same. A neural-network
**trunk** produces a shared internal representation; a **head** turns that
representation into one requested output.

The **reverse Huber (BerHu) loss** is a piecewise training loss defined in
`training-stack.md`. **Trimming** removes a configured fraction of the
largest-error rows before a training average. **Focus** gives harder retained
rows more weight. A **snapshot** is one saved copy of model and training state
that can be selected or restored together.

A **warm start** begins from a saved model instead of random weights.
**Fine-tuning** continues training that model. **Transfer learning** keeps a
saved base calculation and trains a correction for a related task. The
**correction target** is the numerical difference or ratio that correction
must learn; the **correction head** is the trainable network part that predicts
it.

A **parameter-name sidecar** is a small companion file that records the name
and order of each table column. A **chain root** is the common path stem for a
set of sampling files. The **path owner** is the one component responsible for
resolving that path. A **class guard** checks the saved family before reading
family-specific fields. A **predictor** rebuilds a saved artifact and returns
its scientific output. An **adapter** translates that output into the names,
shapes, and units expected by another program. A **family-shaped return** uses
the output structure documented for that family.

A **chi-square error** is a covariance-weighted squared difference between a
prediction and its reference value; larger values mean that the error matters
more for the intended scientific analysis. **Sampling temperature** is a
numerical factor that widens the parameter distribution used to create
training examples. It does not describe physical heat. **Evolving dark
energy** allows the dark-energy equation-of-state parameters to vary with
cosmic time. **Intrinsic alignment** is the non-lensing alignment of galaxy
shapes caused by the galaxies' shared environment; an emulator must preserve
its effect rather than mistake it for gravitational lensing.

A geometry **round-trips its state** when saving and then loading the state
reproduces the same stored values and meanings. A loss's **physical
contraction** is the documented mathematical operation that combines a
prediction residual with covariance or precision information to produce one
score per sample. Keeping that calculation in one owner prevents training and
diagnostics from assigning different scientific meanings to the same error.

**Multi-device execution** assigns work to more than one accelerator. **GPU
packing** lets several workers share one graphics processor under explicit
memory limits. A **journal** is the saved progress record used to restart a
study. A numerical **data type (dtype)** states how values are represented,
such as float32 or float64. In a test, a **control** is the valid case that
must pass, a **mutation** deliberately restores forbidden behavior and must
fail, and a **witness** is the concrete input and observation that distinguish
those outcomes.

## Project purpose

CoCoA is the surrounding cosmological-analysis installation that connects the
Cobaya inference program to CosmoLike likelihood calculations. Cobaya proposes
cosmological parameter values during Bayesian sampling. CosmoLike turns those
values into observables used by a likelihood.

CoCoA SONIC is this emulator library. The library uses
PyTorch, a tensor and machine-learning package, to build the approximations.
The supported outputs are:

- CosmoLike data vectors;
- derived scalar parameters;
- cosmic microwave background (CMB) angular power spectra;
- the expansion history;
- the matter-power spectrum.

The main scientific comparison is sample efficiency: the fraction of
validation cosmologies whose chi-square error exceeds the selected threshold,
measured as a function of the number of training cosmologies. Training
cosmologies are expensive when the sampling temperature is high and the
examples include evolving dark energy and intrinsic alignment. Coverage,
representation, and data identity therefore matter before extra network
capacity.

One shared training stack serves all output families. A saved artifact records
the resolved model and scientific facts required for reconstruction. Every
public capability requires an executable acceptance check that can distinguish
the required behavior from a plausible broken implementation.

## Development milestones that still shape the library

Only milestones that created a current project-wide rule belong here.

### Scientific reference and success metric

The teaching prototype established sample efficiency as the primary model
comparison. The prototype also established a durable design rule: diagnose
training coverage, target representation, and physical factorization before
adding network complexity.

The original notebook remains a scientific reference, not an executable source
of current library behavior. Current modules and tests are authoritative.

### Reusable package and CoCoA layout

Notebook logic became modules under `emulator/` plus command-line drivers that
follow CoCoA project, configuration, and output paths. `EmulatorExperiment`
became the shared owner of configuration validation and experiment setup.

Driver code must not recreate configuration or training rules already owned by
the shared experiment object.

### One training engine

Trunk/head phases, BerHu loss, trimming, focus, exponential moving averages,
absolute training and validation sizes, precision handling, model selection,
and snapshot behavior became shared contracts rather than driver-specific
features.

A family-specific driver selects a family and delegates to shared training
code. A family wrapper does not fork the training engine.

### Composable designs, geometries, and losses

Network designs, physical geometries, and losses became separate owners:

- a design owns the trainable architecture;
- a geometry owns coordinate transforms and persistent geometric facts;
- a loss owns the physical contraction and per-sample score;
- capability flags state supported behavior without scattered class-name
  checks.

This separation allows shared training features to compose across output
families without teaching a generic trainer every family formula.

### Self-describing artifacts and inference

An artifact is the saved pair of trained weights and scientific reconstruction
information. A schema is the versioned list and meaning of the saved fields.
Schema-3 artifacts record the resolved model recipe, geometry, parameter
identity, scientific fixed facts, and other values required to rebuild the
trained behavior. Schema-3 readers refuse retired schema versions rather than
guessing missing facts from current defaults.

`rebuild_emulator`, `EmulatorPredictor`, and the Cobaya adapters consume the
artifact record. A caller cannot silently replace a fact already owned by the
artifact.

### Five output families

One framework serves five kinds of output:

| Family | Output | Structural property |
|---|---|---|
| CosmoLike data vector | binned observables used by likelihoods | coordinate-aware vector geometry |
| Scalar | named derived parameters | trunk-only named outputs |
| CMB | angular spectra over multipole | spectrum geometry and covariance contraction |
| Background | distances and expansion quantities over redshift | one-dimensional redshift geometry |
| Matter power | power over wavenumber and redshift | two-dimensional grid geometry with a physical base model |

Each family supplies validation, geometry, loss behavior, prediction,
configuration examples, and acceptance checks through shared interfaces.

### Warm starts and capability symmetry

Fine-tuning applies to every family when source and target artifacts satisfy
the required identity checks. Transfer learning applies to families with a
meaningful correction target; scalar output remains excluded because named
scalars do not share the coordinate structure required by the correction
design.

Neural polynomial chaos expansion (NPCE) and coordinate-aware correction heads
are enabled by structural
capability, not by a list of favored families. Scalar output remains trunk-only
for the same coordinate reason.

### Shared drivers and executable acceptance

Family-specific command names delegate to shared implementations. Identity
checks protect save/rebuild behavior. Smoke checks exercise a real generator,
training path, saved artifact, and consumer path.

A smoke check must fail for a dead or disconnected model. A check that only
confirms that code ran does not establish scientific behavior.

### Explicit data identity and publication

Generator requests, parameter names, row coordinates, restart state, and
published members became validated records. Same-shaped files do not prove
that files belong to the same dataset.

Artifact identity is the validated record that binds every member to one
scientific dataset and run. Publication is the controlled move that makes a
complete saved result visible to later programs. Publication is complete only
when every required member is sealed, recorded, and selected through one
authoritative identity. Restart and append behavior
must preserve sampling state rather than replaying the original rows.

### Dedicated AI-development support

AI-only notes, tests, gates, and tools live under `ai/`. Permanent notes contain
current general properties. Ticket chronology remains in ignored local records.
The Architect is the only public ticket contact and controls every downstream
Implementer or Red Team handoff.

The role system is optional when one capable model has enough tokens to plan,
implement, test, and review a change. Its purpose is cost control when that is
not realistic: the Architect and optional Red Team perform the expensive
reasoning, while a less expensive and potentially simpler Implementer performs
the token-heavy reading, editing, and test work. Their instructions must
therefore resolve design choices before implementation. Authority belongs to
the role, not to a model name.

This repository owns the Python emulators, data generators, Cobaya adapters,
tests, and gates. CAMB and CosmoLike are upstream scientific programs whose
behavior this repository consumes. Ordinary emulator work may investigate and
record their behavior, but it must not turn into a Fortran CAMB port or direct
CosmoLike C modification. A required upstream change needs its own explicitly
approved scope in the repository that owns that code.

## Pattern for a new output family

A new output family follows the complete pattern below. Omitting one surface
usually creates a capability that trains in isolation but cannot be saved,
rebuilt, validated, or served safely.

### Configuration and validation

1. Add one `data.<family>` configuration block that is mutually exclusive with
   other output-family blocks.
2. Add a pure validator that checks required fields, finite values, shapes,
   supported rescaling, correction-head compatibility, fine-tuning, transfer,
   and NPCE capability before model construction.
3. Materialize defaults before saving the resolved configuration.
4. Validate warm-start fields before reading a model block whose presence may
   differ between a fresh run and a warm start.

### Geometry and physical loss

5. Add a geometry under `emulator/geometries/`. The geometry records resolved
   values. Saving and loading its state must reproduce the same bytes and
   scientific meanings.
6. Define zero-variance and constant-coordinate behavior explicitly. A
   relative threshold must remain meaningful when the reference value is zero.
7. Add a loss that exposes a per-sample chi-square-like score so trimming,
   focus, BerHu scheduling, exponential moving averages, and warm-start anchors
   compose through the shared trainer.
8. Keep the covariance- or precision-weighted residual calculation in one
   loss owner. Diagnostics and training must call that same calculation rather
   than reproduce the formula.

### Data staging and generation

9. Stage parameter columns by name through the parameter-name sidecar. Do not
   assume a fixed column position.
10. Resolve chain roots and supporting files according to the documented path
    owner.
11. Add a generation driver based on `compute_data_vectors/generator_core.py`
    so storage, checkpoint, append, failure, and publication behavior remain
    shared.

### Results and inference

12. Add the family flag and class-guarded metadata reads to results handling.
    A similarly named field in another family must not be interpreted under
    the wrong schema.
13. Add a predictor branch with a family-shaped return and explicit units.
14. Add a Cobaya adapter using the shared adapter mechanics. Both wrong-kind
    directions require refusal: the adapter must reject the wrong artifact,
    and the artifact must not be served through the wrong adapter.

### Command-line drivers

15. Add thin `<family>_<verb>_emulator.py` wrappers for training, tuning,
    training-size sweeps, and hyperparameter sweeps.
16. Delegate to the shared family-driver entry point. Preserve multi-device,
    packing, journal, restart, logging, and failure behavior through one code
    path.

### Acceptance checks

17. Add `<family>-identity` checks for configuration, geometry, save/rebuild,
    fixed facts, and every family-specific refusal.
18. Add `<family>-smoke` checks for the real generator, trainer, artifact, and
    consumer lifecycle.
19. Prove that identity checks fail when a required field, coordinate,
    scientific fact, or family tag is changed.
20. Prove that the smoke threshold rejects a dead network before treating a
    passing result as scientific evidence.

### Examples and teaching material

21. Add an example YAML file containing every required training block.
22. Add the family to the relevant README chooser and configuration example.
23. Define family-specific terms at first use and show the actual YAML block.
24. Add the family to diagnostics only through an explicit capability or
    family dispatch.

## Program-level lessons

### The current source is authoritative

Read current code and current saved schemas before porting behavior. A retired
implementation is evidence only when a current rule explicitly depends on the
retired behavior.

### Identity evidence needs the correct strength

Use byte or bit identity only when two paths perform the same computation.
Cross-path numerical checks use a documented tolerance justified by dtype and
algorithm. Do not weaken an identity check to hide an unexplained mismatch.

### Structural repairs outlast patches

Repair the owner of a broken rule. Do not preserve a patch that exists only
because a former schema or duplicated code path was flawed.

### A test must distinguish the forbidden behavior

A check harness needs a known-good witness and a mutation or control that
demonstrates failure for the broken implementation. A green command without a
discriminating witness is execution evidence, not acceptance evidence.

### Failures often reveal the next hidden layer

Classify failures by source: harness, check, configuration, library, or
scientific contract. Repairing one layer may expose another. Do not treat a new
failure after a valid repair as evidence that the repair was wrong.

### Wide changes need complete censuses

Count every target before and after a mechanical rename or family-wide change.
Truncated search output and substring collisions can hide missed paths. Use
exact path lists and longest-first replacement when names overlap.

### Detailed knowledge belongs with the owning topic

This note records only project-wide milestones and patterns. Numerical laws,
artifact fields, training state, data publication, and family-specific rules
belong in the corresponding permanent topic note.
