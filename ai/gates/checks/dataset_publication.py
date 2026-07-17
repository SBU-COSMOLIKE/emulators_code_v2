#!/usr/bin/env python3
"""Run the CPU evidence for publishing and reading generated datasets.

The check follows one generated dataset across four visible boundaries. It
checks the small locator file that finds the current saved generation, the
generator's private work folder and final switch, Cocoa's choice of one train
and one validation generation, and the failure mask that removes bad rows
before training.

Each focused test belongs to one of the six results declared by the board. The
census is exhaustive: a new test in any covered class must be assigned here
before this gate can pass.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests.test_dataset_publication import DatasetPublicationTests
from ai.tests.test_dataset_locator import DatasetLocatorTests
from ai.tests.test_generator_publication_bridge import (
  GeneratorPublicationBridgeTests,
)
from ai.tests.test_cocoa_dataset_resolution import (
  CocoaDatasetResolutionTests,
)
from ai.tests.test_failed_row_staging import FailedRowStagingTests


LEG_AIDS = (
  "dataset-publication.slot-identity",
  "dataset-publication.exact-census",
  "dataset-publication.sealed-epoch",
  "dataset-publication.atomic-switch",
  "dataset-publication.durability-and-recovery",
  "dataset-publication.copy-on-write-continuation",
)


ARMS = {
  "dataset-publication.slot-identity": (
    (DatasetPublicationTests, (
      "test_slots_with_different_descriptors_are_isolated",
      "test_each_slot_axis_is_distinct_and_relocation_is_stable",
      "test_case_only_logical_stem_collision_is_refused",
      "test_canonical_json_is_stable_and_refuses_nonfinite_values",
      "test_canonical_and_loader_refuse_oversized_json_integer",
      "test_active_record_refuses_noncanonical_duplicate_and_unknown_fields",
      "test_manifest_refuses_noncanonical_and_unknown_fields",
    )),
    (DatasetLocatorTests, (
      "test_install_is_canonical_read_only_and_idempotent",
      "test_existing_logical_name_cannot_be_reassigned",
      "test_load_accepts_only_the_named_chains_child",
    )),
    (GeneratorPublicationBridgeTests, (
      "test_identity_hashes_canonical_yaml_and_final_uniform_support",
    )),
    (CocoaDatasetResolutionTests, (
      "test_missing_locator_refuses_even_when_legacy_flat_files_exist",
      "test_source_yaml_cannot_supply_resolver_owned_identity",
    )),
  ),
  "dataset-publication.exact-census": (
    (DatasetPublicationTests, (
      "test_identity_and_role_path_censuses_match_exactly",
      "test_declared_and_manifest_paths_cannot_traverse",
      "test_symlink_member_is_refused",
      "test_hardlinked_member_is_refused",
      "test_special_member_is_refused",
      "test_missing_and_extra_draft_members_are_refused",
      "test_extra_empty_nested_directory_is_refused",
      "test_missing_extra_and_corrupted_published_members_are_refused",
      "test_published_generation_is_read_only_and_writable_tree_is_refused",
    )),
    (DatasetLocatorTests, (
      "test_install_recomputes_slot_and_member_contract",
      "test_load_refuses_mutable_noncanonical_and_linked_records",
    )),
    (GeneratorPublicationBridgeTests, (
      "test_failed_row_mask_refuses_before_publication",
    )),
    (CocoaDatasetResolutionTests, (
      "test_chain_only_scalar_rewrites_no_payload_or_failure_mask",
      "test_cmb_selects_one_spectrum_and_requires_shared_multipoles",
      "test_background_and_grid2d_axes_and_bases_are_generation_paths",
      "test_train_and_validation_scientific_meaning_must_match",
      "test_train_and_validation_may_use_different_sampling_procedures",
      "test_axis_mismatch_and_cross_dataset_payload_are_refused",
    )),
    (FailedRowStagingTests, (
      "test_failed_rows_are_removed_before_seeded_selection",
      "test_failed_rows_reduce_the_available_pool",
      "test_failure_mask_refuses_bad_tokens_and_row_counts",
    )),
  ),
  "dataset-publication.sealed-epoch": (
    (DatasetPublicationTests, (
      "test_source_mutation_between_member_copies_refuses_whole_dataset",
      "test_retained_writable_source_fd_cannot_change_published_copy",
    )),
    (GeneratorPublicationBridgeTests, (
      "test_single_and_family_memmaps_are_closed_before_publish_call",
      "test_constructor_barrier_precedes_rank_zero_publication",
    )),
  ),
  "dataset-publication.atomic-switch": (
    (DatasetPublicationTests, (
      "test_roundtrip_authenticates_nested_members_and_checkpoints",
      "test_compare_and_swap_refuses_a_stale_writer",
      "test_compare_and_swap_detects_generation_only_active_mutation",
      "test_pinned_reader_stays_on_a_while_fresh_reader_sees_b",
      "test_callback_exception_boundaries_expose_no_in_process_hybrid",
    )),
    (DatasetLocatorTests, (
      "test_locator_resolves_each_current_generation_without_changing",
    )),
    (GeneratorPublicationBridgeTests, (
      "test_fresh_writes_only_a_private_draft_then_publishes",
      "test_fresh_refuses_an_existing_authenticated_active_before_work",
    )),
    (CocoaDatasetResolutionTests, (
      "test_cosmolike_paths_and_failure_masks_come_from_two_pins",
    )),
    (FailedRowStagingTests, (
      "test_scalar_selection_records_original_disk_rows",
      "test_saved_source_pin_binds_exact_staged_row_order",
      "test_saved_source_pin_refuses_a_partial_staged_identity",
      "test_saved_source_pin_refuses_an_equal_count_wrong_row_order",
      "test_saved_emulator_keeps_the_staged_source_identity",
    )),
  ),
  "dataset-publication.durability-and-recovery": (
    (DatasetPublicationTests, (
      "test_successful_publish_removes_mutable_source_draft",
      "test_cleanup_failure_and_warning_error_do_not_hide_commit",
      "test_oversize_manifest_refuses_before_active_switch",
      "test_draft_durable_callback_failure_cleans_sealed_not_source",
      "test_sync_order_makes_generation_durable_before_active_switch",
      "test_directory_creation_retry_resyncs_existing_parent",
      "test_durability_helpers_call_platform_primitives",
    )),
    (GeneratorPublicationBridgeTests, (
      "test_first_fresh_crash_cannot_be_mistaken_for_resumable_active",
    )),
  ),
  "dataset-publication.copy-on-write-continuation": (
    (DatasetPublicationTests, (
      "test_continuation_copies_nested_members_to_private_independent_files",
      "test_continuation_syncs_private_files_and_complete_draft_tree",
      "test_continuation_authenticates_request_before_creating_draft",
      "test_continuation_copy_failure_cleans_only_new_draft",
      "test_continuation_rechecks_every_source_after_complete_copy",
      "test_continuation_rechecks_first_member_after_last_copy",
      "test_continuation_refuses_writable_source_after_active_load",
      "test_continuation_refuses_same_byte_source_path_replacement",
      "test_continuation_requires_exact_mutable_copy_census",
      "test_continuation_refuses_nonprivate_copy_modes",
      "test_continuation_close_failure_attempts_all_and_refuses_success",
      "test_continuation_refuses_size_digest_lies_and_hardlink_copy",
      "test_continuation_rechecks_manifest_generation_census_and_modes",
      "test_continuation_keeps_pinned_token_across_concurrent_switch",
    )),
    (GeneratorPublicationBridgeTests, (
      "test_resume_copies_authenticated_active_and_publishes_with_cas",
      "test_append_authenticates_active_then_refuses_without_a_draft",
    )),
  ),
}


def _covered_classes():
  """Return every focused test class named by this check."""
  return (
    DatasetPublicationTests,
    DatasetLocatorTests,
    GeneratorPublicationBridgeTests,
    CocoaDatasetResolutionTests,
    FailedRowStagingTests,
  )


def _validate_census():
  if tuple(ARMS) != LEG_AIDS:
    raise AssertionError(
      "dataset-publication AID order differs from its declared LEG_AIDS")
  assigned = []
  for groups in ARMS.values():
    for test_class, names in groups:
      assigned.extend((test_class, name) for name in names)
  duplicates = sorted(
    test_class.__name__ + "." + name
    for test_class, name in set(assigned)
    if assigned.count((test_class, name)) > 1)
  focused = set()
  for test_class in _covered_classes():
    for name in unittest.defaultTestLoader.getTestCaseNames(test_class):
      focused.add((test_class, name))
  missing = sorted(
    test_class.__name__ + "." + name
    for test_class, name in focused - set(assigned))
  unknown = sorted(
    test_class.__name__ + "." + name
    for test_class, name in set(assigned) - focused)
  if duplicates or missing or unknown:
    raise AssertionError(
      "dataset-publication evidence census drifted: duplicates="
      + repr(duplicates) + ", unassigned=" + repr(missing)
      + ", unknown=" + repr(unknown))


def _run_arm(aid, groups):
  suite = unittest.TestSuite()
  expected = 0
  for test_class, methods in groups:
    suite.addTests(test_class(method) for method in methods)
    expected += len(methods)
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  passed = result.wasSuccessful() and result.testsRun == expected
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + " ("
        + str(result.testsRun) + "/" + str(expected) + " witnesses)")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  try:
    _validate_census()
  except Exception as exc:
    print("dataset-publication evidence census: FAIL: " + str(exc))
    for aid in ARMS:
      print("##AID " + aid + " FAIL")
    return 1

  passed = []
  for aid, methods in ARMS.items():
    passed.append(_run_arm(aid, methods))
  if not all(passed):
    print("dataset-publication: FAIL")
    return 1
  print("dataset-publication: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
