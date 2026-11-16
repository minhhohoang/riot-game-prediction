"""Offline actual-pick comparisons from the completed min50 grouped posteriors.

Read-only: no fitting, collection, recommendation selection, or report overwrites.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hlr_model import min50_project_analysis as analysis
from hlr_model import broad_rank_modeling as broad
from hlr_model.main_pick_decision_report import KEY, file_sha, require

MATCHUP_KEY = ['model_rank', 'lane', 'opponent_champion_id', 'model_champion_id']


def verified_benchmarks(frame, cohort):
    """Apply the fitted preparation's verification rules, retaining low n for audit."""
    if cohort == 'Master':
        valid = (frame.global_filters_verified.eq(True)
                 & frame.global_verified_patch.eq('16.19')
                 & frame.global_verified_region.eq('global')
                 & frame.global_verified_rank.eq('master_plus'))
        count = frame.global_matchup_games
    else:
        valid = (frame.benchmark_validation_status.eq('verified')
                 & frame.benchmark_requested_patch.eq('16.19')
                 & frame.benchmark_requested_region.eq('global')
                 & frame.benchmark_requested_rank.eq(frame.model_rank.str.lower()))
        count = frame.benchmark_matchup_games
    result = frame[MATCHUP_KEY].copy()
    result['matchup_probability'] = frame.matchup_probability
    result['matchup_games'] = pd.to_numeric(count, errors='coerce')
    return result.loc[valid & frame.benchmark_valid_rate.eq(True)]


def load_benchmarks(root, cohort, played):
    """Union existing verified tables and prepared benchmarks, without main-pick screening."""
    source = analysis.source_path(root, cohort)
    report = source / 'reports/expanded_published_candidates_v1'
    manifest = json.loads((report / 'report_manifest.json').read_text())
    for name in ['published_matchup_lookup.parquet', 'matchup_table_coverage.parquet']:
        require(file_sha(report / name) == manifest['output_sha256'][name],
                f'Cached benchmark artifact changed: {name}')
    tables = pd.read_parquet(report / 'matchup_table_coverage.parquet')
    lookup = pd.read_parquet(report / 'published_matchup_lookup.parquet')
    require(lookup.table_id.isin(tables.loc[tables.status.eq('verified'), 'table_id']).all(),
            'Lookup contains an unverified table.')
    lookup = lookup.rename(columns={'candidate_champion_id': 'model_champion_id'})
    lookup['matchup_probability'] = lookup.raw_matchup_winrate_pct / 100

    # Original prepared alternatives include verified contexts absent from the
    # expanded recommendation cache. They are evidence, not a candidate selector.
    prep = json.loads((source / 'preparation_manifest.json').read_text())
    name = 'prepared_alternatives.parquet'
    require(file_sha(source / name) == prep['output_sha256'][name],
            'Original prepared alternatives changed.')
    alternatives = pd.read_parquet(source / name)
    sources = [lookup[MATCHUP_KEY + ['matchup_probability', 'matchup_games']],
               verified_benchmarks(alternatives, cohort), verified_benchmarks(played, cohort)]
    if cohort == 'Master':
        main = played[MATCHUP_KEY].assign(model_champion_id=played.prior_main_champion_id)
        main['matchup_probability'] = played.main_global_raw_matchup_winrate_probability
        main['matchup_games'] = pd.to_numeric(played.main_global_matchup_games, errors='coerce')
        valid = (played.main_global_valid_rate.eq(True) & played.main_global_filters_verified.eq(True)
                 & played.main_global_verified_patch.eq('16.19')
                 & played.main_global_verified_region.eq('global')
                 & played.main_global_verified_rank.eq('master_plus'))
        sources.append(main.loc[valid])
    combined = pd.concat(sources, ignore_index=True)
    n = combined.matchup_games
    combined = combined.loc[combined.model_champion_id.notna()
        & combined.matchup_probability.between(0, 1) & np.isfinite(n) & n.ge(0) & n.eq(np.floor(n))]
    # All existing sources must agree; never silently choose between conflicting evidence.
    grouped = combined.groupby(MATCHUP_KEY)
    spread = grouped.matchup_probability.max() - grouped.matchup_probability.min()
    require(spread.le(1e-12).all()
            and grouped.matchup_games.nunique().le(1).all(), 'Cached benchmark sources disagree.')
    return combined.drop_duplicates(MATCHUP_KEY).reset_index(drop=True)


def select_pairs(played, benchmarks, groups):
    """One actual/main pair per eligible test player-game; exclusions are sequential."""
    test = played.loc[played.split.eq('test')].drop(columns='win', errors='ignore').copy()
    require(not test.duplicated(KEY).any(), 'Duplicate test player-game.')
    require(not benchmarks.duplicated(MATCHUP_KEY).any(), 'Duplicate matchup context.')
    audit = test[KEY + ['target_row_id', 'champion_id', 'prior_main_champion_id']].copy()
    audit['exclusion'] = 'eligible'
    remaining = pd.Series(True, index=test.index)
    stages = []

    def exclude(reason, failed):
        nonlocal remaining
        failed = remaining & failed.fillna(True)
        stages.append(dict(exclusion=reason, entering=int(remaining.sum()), excluded=int(failed.sum()),
                           remaining=int((remaining & ~failed).sum())))
        audit.loc[failed, 'exclusion'] = reason
        remaining &= ~failed

    exclude('No strictly pregame recorded main', test.prior_main_champion_id.isna())
    exclude('Actual pick equals recorded main', test.champion_id.eq(test.prior_main_champion_id))
    exclude('Actual pick: invalid features/history or raw benchmark', ~test.source_usable_data.eq(True))
    exclude('Actual pick: missing or unverified external count', ~test.external_count_verified.eq(True))
    exclude('Actual pick: external matchup games < 50', ~test.external_matchup_games.ge(50))

    # Only champion-specific inputs change. The saved main counts/wins and latest
    # history timestamp are strictly pregame, all-role quantities from preparation.
    main = test.copy()
    main['row_kind'] = 'alternative'
    main['champion_id'] = test.prior_main_champion_id
    main['model_champion_id'] = test.prior_main_champion_id
    main['n_recorded'] = test.prior_main_champion_games
    main['prior_champion_games'] = test.prior_main_champion_games
    main['prior_champion_wins'] = test.prior_main_champion_wins
    main['historical_wr_beta11'] = (test.prior_main_champion_wins + 1) / (main.n_recorded + 2)
    main['mastery_group'] = main.model_champion_id.map(groups).fillna('Unclassified')
    main['log_total'] = np.log1p(test.prior_total_games)
    # Join by the full exact-rank / role / opponent / focal champion context.
    joined = main[MATCHUP_KEY].merge(benchmarks, on=MATCHUP_KEY, how='left', validate='many_to_one')
    main['matchup_probability'] = joined.matchup_probability.to_numpy()
    main['external_matchup_games'] = joined.matchup_games.to_numpy()
    n, w = main.n_recorded, test.prior_main_champion_wins
    valid = (np.isfinite(n) & n.gt(0) & n.eq(np.floor(n)) & n.le(test.prior_total_games)
             & np.isfinite(w) & w.between(0, n) & w.eq(np.floor(w))
             & test.prior_latest_end_timestamp.lt(test.timestamp)
             & np.isfinite(main.log_total) & main.historical_wr_beta11.between(0, 1)
             & main.lane.isin(broad.ROLES) & main.mastery_group.isin(broad.GROUPS))
    exclude('Recorded main: invalid pregame features/history', ~valid)
    exclude('Recorded main: no valid verified raw matchup/count',
            ~main.matchup_probability.between(0, 1) | main.external_matchup_games.isna())
    exclude('Recorded main: external matchup games < 50', ~main.external_matchup_games.ge(50))
    exclude('Actual raw matchup is not strictly higher than main',
            ~test.matchup_probability.gt(main.matchup_probability))
    actual = test.loc[remaining].copy()
    main = main.loc[remaining].copy()
    require(len(actual) + sum(r['excluded'] for r in stages) == len(test), 'Exclusions do not reconcile.')
    require(actual[KEY].equals(main[KEY]), 'Actual/main player-game alignment changed.')
    return actual, main, pd.DataFrame(stages), audit


def summarize(cohort, comparisons):
    n = len(comparisons)
    main = int(comparisons.main_probability.gt(comparisons.actual_probability).sum())
    actual = int(comparisons.actual_probability.gt(comparisons.main_probability).sum())
    ties = int(comparisons.main_probability.eq(comparisons.actual_probability).sum())
    require(main + actual + ties == n, 'Mean comparisons do not reconcile.')
    return dict(cohort=cohort, eligible_player_games=n, main_higher_mean=main,
                main_higher_percent=100 * main / n if n else np.nan,
                actual_higher_mean=actual, exact_mean_ties=ties)


def run(root):
    """Load completed fits; average probabilities over all 4,800 grouped draws."""
    root = Path(root).resolve()
    summaries, exclusions, comparisons, audits, provenance = [], [], {}, {}, []
    for cohort in ['Master', 'Bronze–Diamond']:
        out, prepared, fits = analysis.load_fits(root, cohort)
        played = prepared['played']
        benchmarks = load_benchmarks(root, cohort, played)
        groups = pd.read_parquet(out / 'champion_grouping_lookup.parquet').set_index('champion_id').mastery_group
        actual, main, excluded, audit = select_pairs(played, benchmarks, groups)
        result = fits['grouped']
        require('group_free' in result['samples'], 'A grouped posterior is required.')
        # Explicitly remove outcomes. Use fitted identity effects and the existing
        # identity-stable partial-pooling behavior for unseen player/champion pairs.
        module = analysis.module_for(cohort)
        if len(actual):
            actual_probability = module.posterior_mean(result, actual, mode='personalized')
            main_probability = module.posterior_mean(result, main, mode='personalized')
        else:
            actual_probability = main_probability = np.array([])
        rows = actual[KEY + ['target_row_id', 'timestamp', 'lane', 'model_rank', 'opponent_champion_id',
                             'champion_id', 'prior_main_champion_id', 'prior_total_games']].copy()
        for prefix, frame, probability in [('actual', actual, actual_probability), ('main', main, main_probability)]:
            rows[prefix + '_raw_matchup'] = frame.matchup_probability.to_numpy()
            rows[prefix + '_external_matchup_games'] = frame.external_matchup_games.to_numpy()
            rows[prefix + '_recorded_games'] = frame.n_recorded.to_numpy()
            rows[prefix + '_historical_wr_beta11'] = frame.historical_wr_beta11.to_numpy()
            rows[prefix + '_probability'] = probability
        require(np.isfinite(rows[['main_probability', 'actual_probability']]).all().all(), 'Nonfinite predictions.')
        require(not rows.duplicated(KEY).any(), 'Player-games counted more than once.')
        summaries.append(summarize(cohort, rows))
        exclusions.append(excluded.assign(cohort=cohort))
        comparisons[cohort], audits[cohort] = rows.reset_index(drop=True), audit.reset_index(drop=True)
        provenance.append(dict(cohort=cohort, min50_version=out.name, model='grouped', mode='personalized',
            retained_draws=4800, fit_manifest_sha256=file_sha(out / 'fits/grouped/fit_manifest.json'),
            chains_sha256=[file_sha(out / f'fits/grouped/chain_{i}.npz') for i in range(4)]))
    return dict(summary=pd.DataFrame(summaries), exclusions=pd.concat(exclusions, ignore_index=True),
                comparisons=comparisons, audit=audits, provenance=pd.DataFrame(provenance))
