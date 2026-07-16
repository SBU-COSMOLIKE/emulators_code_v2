#!/usr/bin/env python3
"""Run the CPU evidence arms for immutable dataset-generation publication.

The production module has no Torch, MPI, Cobaya, or workstation dependency.
This check groups its focused unittest witnesses into the six exact assertions
declared by the board and emits one reserved ``##AID`` terminal for each arm.
The census below is deliberately exhaustive: adding a focused test without
assigning it to an evidence arm makes the check fail instead of silently
leaving that behavior outside the board record.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests.test_dataset_publication import DatasetPublicationTests


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
    "test_slots_with_different_descriptors_are_isolated",
    "test_each_slot_axis_is_distinct_and_relocation_is_stable",
    "test_case_only_logical_stem_collision_is_refused",
    "test_canonical_json_is_stable_and_refuses_nonfinite_values",
    "test_canonical_and_loader_refuse_oversized_json_integer",
    "test_active_record_refuses_noncanonical_duplicate_and_unknown_fields",
    "test_manifest_refuses_noncanonical_and_unknown_fields",
  ),
  "dataset-publication.exact-census": (
    "test_identity_and_role_path_censuses_match_exactly",
    "test_declared_and_manifest_paths_cannot_traverse",
    "test_symlink_member_is_refused",
    "test_hardlinked_member_is_refused",
    "test_special_member_is_refused",
    "test_missing_and_extra_draft_members_are_refused",
    "test_extra_empty_nested_directory_is_refused",
    "test_missing_extra_and_corrupted_published_members_are_refused",
    "test_published_generation_is_read_only_and_writable_tree_is_refused",
  ),
  "dataset-publication.sealed-epoch": (
    "test_source_mutation_between_member_copies_refuses_whole_dataset",
    "test_retained_writable_source_fd_cannot_change_published_copy",
  ),
  "dataset-publication.atomic-switch": (
    "test_roundtrip_authenticates_nested_members_and_checkpoints",
    "test_compare_and_swap_refuses_a_stale_writer",
    "test_compare_and_swap_detects_generation_only_active_mutation",
    "test_pinned_reader_stays_on_a_while_fresh_reader_sees_b",
    "test_callback_exception_boundaries_expose_no_in_process_hybrid",
  ),
  "dataset-publication.durability-and-recovery": (
    "test_successful_publish_removes_mutable_source_draft",
    "test_cleanup_failure_and_warning_error_do_not_hide_commit",
    "test_oversize_manifest_refuses_before_active_switch",
    "test_draft_durable_callback_failure_cleans_sealed_not_source",
    "test_sync_order_makes_generation_durable_before_active_switch",
    "test_directory_creation_retry_resyncs_existing_parent",
    "test_durability_helpers_call_platform_primitives",
  ),
  "dataset-publication.copy-on-write-continuation": (
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
  ),
}


def _focused_test_names():
  return set(unittest.defaultTestLoader.getTestCaseNames(
    DatasetPublicationTests))


def _validate_census():
  if tuple(ARMS) != LEG_AIDS:
    raise AssertionError(
      "dataset-publication AID order differs from its declared LEG_AIDS")
  assigned = [name for names in ARMS.values() for name in names]
  duplicates = sorted({name for name in assigned if assigned.count(name) > 1})
  focused = _focused_test_names()
  missing = sorted(focused - set(assigned))
  unknown = sorted(set(assigned) - focused)
  if duplicates or missing or unknown:
    raise AssertionError(
      "dataset-publication evidence census drifted: duplicates="
      + repr(duplicates) + ", unassigned=" + repr(missing)
      + ", unknown=" + repr(unknown))


def _run_arm(aid, methods):
  suite = unittest.TestSuite(
    DatasetPublicationTests(method) for method in methods)
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  passed = result.wasSuccessful() and result.testsRun == len(methods)
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + " ("
        + str(result.testsRun) + "/" + str(len(methods)) + " witnesses)")
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
