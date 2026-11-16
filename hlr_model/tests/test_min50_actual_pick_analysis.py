"""Selection and counting regressions; no sampler or network is used."""
import numpy as np
import pandas as pd
import pytest

from hlr_model import min50_actual_pick_analysis as report


def selection_fixture():
    reasons = [
        'No strictly pregame recorded main',
        'Actual pick equals recorded main',
        'Actual pick: invalid features/history or raw benchmark',
        'Actual pick: missing or unverified external count',
        'Actual pick: external matchup games < 50',
        'Recorded main: invalid pregame features/history',
        'Recorded main: no valid verified raw matchup/count',
        'Recorded main: external matchup games < 50',
        'Actual raw matchup is not strictly higher than main',
        'Actual raw matchup is not strictly higher than main',
        'eligible',
    ]
    played = pd.DataFrame(dict(
        puuid='p', match_id=[str(i) for i in range(len(reasons))], target_row_id=range(len(reasons)),
        split='test', win=1, timestamp=200, prior_latest_end_timestamp=100,
        model_rank='BRONZE', lane='mid', opponent_champion_id=range(1000, 1000 + len(reasons)),
        champion_id=2, model_champion_id=2, prior_main_champion_id=1.,
        prior_total_games=100, prior_main_champion_games=80, prior_main_champion_wins=45,
        n_recorded=4, historical_wr_beta11=.5, mastery_group='J', log_total=np.log1p(100),
        matchup_probability=.55, source_usable_data=True, external_count_verified=True,
        external_matchup_games=50, same_role_prior_games=0,
    ))
    benchmarks = played[report.MATCHUP_KEY].assign(model_champion_id=1, matchup_probability=.50, matchup_games=50)
    played.loc[0, 'prior_main_champion_id'] = np.nan
    played.loc[1, 'champion_id'] = 1
    played.loc[2, 'source_usable_data'] = False
    played.loc[3, 'external_count_verified'] = False
    played.loc[4, 'external_matchup_games'] = 49
    played.loc[5, 'prior_latest_end_timestamp'] = 200  # Strictly earlier, not <=.
    benchmarks.loc[6, 'matchup_probability'] = np.nan
    benchmarks.loc[7, 'matchup_games'] = 49
    played.loc[8, 'matchup_probability'] = .50  # Raw ties are excluded.
    played.loc[9, 'matchup_probability'] = .49
    return played, benchmarks, reasons


def test_actual_pick_pairs_have_sequential_exclusions_and_champion_specific_main_features():
    played, benchmarks, reasons = selection_fixture()
    # Another split cannot enter the denominator, even with otherwise valid features.
    training = played.iloc[[-1]].assign(split='train', match_id='train')
    actual, main, excluded, audit = report.select_pairs(
        pd.concat([played, training], ignore_index=True), benchmarks, pd.Series({1: 'A', 2: 'J'}))
    assert audit.exclusion.tolist() == reasons
    assert len(actual) == len(main) == 1
    assert actual.match_id.tolist() == main.match_id.tolist() == ['10']
    assert excluded.excluded.sum() + len(actual) == len(played)
    assert excluded.entering.iloc[0] == len(played)
    assert excluded.remaining.iloc[-1] == 1
    assert 'win' not in actual and 'win' not in main
    assert actual.model_champion_id.iloc[0] == 2 and main.model_champion_id.iloc[0] == 1
    assert actual.n_recorded.iloc[0] == 4 and main.n_recorded.iloc[0] == 80
    assert main.historical_wr_beta11.iloc[0] == 46 / 82
    assert main.mastery_group.iloc[0] == 'A' and actual.mastery_group.iloc[0] == 'J'
    assert main.matchup_probability.iloc[0] == .50
    assert actual.external_matchup_games.iloc[0] == main.external_matchup_games.iloc[0] == 50
    # No same-role-experience screen or Rule A/B winner is imposed.
    assert actual.same_role_prior_games.iloc[0] == 0


def test_duplicate_player_games_and_matchup_contexts_fail_instead_of_pair_weighting():
    played, benchmarks, _ = selection_fixture()
    groups = pd.Series({1: 'A'})
    with pytest.raises(ValueError, match='Duplicate test player-game'):
        report.select_pairs(pd.concat([played, played.iloc[[-1]]]), benchmarks, groups)
    with pytest.raises(ValueError, match='Duplicate matchup context'):
        report.select_pairs(played, pd.concat([benchmarks, benchmarks.iloc[[-1]]]), groups)


def test_percent_uses_all_eligible_games_and_ties_are_exact():
    pairs = pd.DataFrame(dict(main_probability=[.6, .4, .5, .5 + 1e-14], actual_probability=[.5] * 4))
    summary = report.summarize('Master', pairs)
    assert summary == dict(cohort='Master', eligible_player_games=4, main_higher_mean=2,
                           main_higher_percent=50., actual_higher_mean=1, exact_mean_ties=1)
    empty = report.summarize('Master', pairs.iloc[:0])
    assert empty['eligible_player_games'] == 0 and np.isnan(empty['main_higher_percent'])
