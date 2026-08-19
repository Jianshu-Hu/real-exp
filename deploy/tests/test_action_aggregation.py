import numpy as np

from deploy.action_aggregation import TemporalProposalAggregator


def test_aggregation_uses_raw_proposals_instead_of_recursive_blends() -> None:
    aggregator = TemporalProposalAggregator(decay=0.5)
    timestep = 12
    for value in (0.0, 10.0, 20.0):
        result = aggregator.add(timestep, aggregator.begin_chunk(), np.asarray([value]))

    # Raw proposal weights are [0.25, 0.5, 1.0]. A recursive 50/50 blend would be 12.5.
    np.testing.assert_allclose(result, np.asarray([100.0 / 7.0]), rtol=1e-6)
    assert aggregator.proposal_count(timestep) == 3


def test_discard_releases_executed_timestep_history() -> None:
    aggregator = TemporalProposalAggregator(decay=0.8)
    aggregator.add(3, aggregator.begin_chunk(), np.asarray([1.0, 2.0]))
    aggregator.discard(3)
    assert aggregator.proposal_count(3) == 0


def test_zero_decay_selects_latest_proposal() -> None:
    aggregator = TemporalProposalAggregator(decay=0.0)
    aggregator.add(8, aggregator.begin_chunk(), np.asarray([1.0]))
    result = aggregator.add(8, aggregator.begin_chunk(), np.asarray([9.0]))
    np.testing.assert_array_equal(result, np.asarray([9.0], dtype=np.float32))


def test_unit_decay_averages_all_raw_proposals_equally() -> None:
    aggregator = TemporalProposalAggregator(decay=1.0)
    timestep = 5
    for value in (2.0, 4.0, 12.0):
        result = aggregator.add(timestep, aggregator.begin_chunk(), np.asarray([value]))
    np.testing.assert_allclose(result, np.asarray([6.0], dtype=np.float32))


def test_continuous_gripper_dimension_is_aggregated_without_thresholding() -> None:
    aggregator = TemporalProposalAggregator(decay=1.0)
    generation = aggregator.begin_chunk()
    aggregator.add(1, generation, np.asarray([0.0, 0.2], dtype=np.float32))
    result = aggregator.add(
        1,
        aggregator.begin_chunk(),
        np.asarray([1.0, 0.8], dtype=np.float32),
    )
    np.testing.assert_allclose(result, np.asarray([0.5, 0.5], dtype=np.float32))


def test_diagnostics_report_normalized_generation_age_weights() -> None:
    aggregator = TemporalProposalAggregator(decay=0.5)
    for value in (0.0, 1.0, 2.0):
        aggregator.add(7, aggregator.begin_chunk(), np.asarray([value]))
    diagnostics = aggregator.diagnostics(7)
    assert diagnostics["proposal_count"] == 3
    assert diagnostics["max_generation_age"] == 2
    np.testing.assert_allclose(diagnostics["normalized_weights"], [1 / 7, 2 / 7, 4 / 7])
