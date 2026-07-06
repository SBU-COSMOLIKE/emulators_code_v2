"""This subpackage holds the per-bin (grouped) emulator variant, in
which the conv stage is grouped per tomographic bin while the trunk
stays one shared ResMLP. It contains ParallelResCNN
(emulator_designs.py) and its GroupedCNNBlock
(emulator_designs_building_blocks.py), kept as the reference for the
grouped-conv batching pattern, not as a production head (see the
module docstrings for the tested-and-rejected history)."""
