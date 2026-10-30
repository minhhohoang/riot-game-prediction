"""Expanded published-matchup comparisons from frozen shared-model fits.

No fitting, predictive evaluation, Riot collection, or writes to familiar reports.
LoLalytics HTTP is opt-in; its existing cache/parser retain source evidence.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from hlr_model import broad_rank_modeling as broad
from hlr_model import master_cohort_modeling as master
from hlr_model import lolalytics_matchup_scraper as scraper
from hlr_model import broad_rank_matchups as exact
from hlr_model.main_pick_decision_report import COHORTS, KEY, file_sha, require

REPORT = 'expanded_published_candidates_v1'
RULES = ['Best baseline matchup', 'Best model prediction']
MODES = {'general': 'Reference player', 'personalized': 'Personalized'}
CATEGORIES = ['Alternative favored', 'Main favored over selected alternative', 'Uncertain']
HISTORY_CLASSES = ['Zero recorded champion games', 'Champion history, no same-role history', 'Prior same-role history']


def save_json(path, value):
    exact.atomic_json(Path(path), value)


def context(root, cohort):
    root = Path(root).resolve()
    folder, version, helper = COHORTS[cohort]
    source = root / 'hlr_model/data/analyses' / folder / version
    prep = json.loads((source / 'preparation_manifest.json').read_text())
    final = json.loads((source / 'final_artifact_manifest.json').read_text())
    hashes = {**final.get('output_sha256', final.get('sha256', {})), **prep['output_sha256']}
    identity = prep['identity']
    identity = json.loads(identity) if isinstance(identity, str) else identity
    require(file_sha(root / 'hlr_model' / helper) == identity['helper_sha256'], 'Fit-time helper changed.')
    for name in ['prepared_played.parquet', 'champion_grouping_lookup.parquet',
                 'fits/shared/fit_manifest.json', 'fits/shared/fit_status.json',
                 'fits/shared/sampling_diagnostics.csv']:
        require(file_sha(source / name) == hashes[name], f'Changed input: {name}')
    played = pd.read_parquet(source / 'prepared_played.parquet').drop(columns='win')
    require(not played.duplicated(KEY).any(), 'Duplicate played rows.')
    test = played.loc[played.split.eq('test')].copy()
    main = test.loc[test.prior_main_champion_id.notna() & test.champion_id.eq(test.prior_main_champion_id)].copy()
    # No same-role screening of the observed main. All other fitted-data checks remain.
    usable = main.loc[main.usable_data & main.benchmark_valid_rate & main.matchup_probability.between(0, 1)].copy()
    require(usable.n_recorded.eq(usable.prior_champion_games).all(), 'Main count mismatch.')
    out = source / 'reports' / REPORT
    out.mkdir(parents=True, exist_ok=True)
    if cohort == 'Master':
        audit = json.loads((source / 'source_audit.json').read_text())
        history_dir = root / audit['longitudinal_snapshot']
        history_hashes = {Path(n).name: v['sha256'] for n, v in audit['source_hashes'].items()
                          if str(Path(n).parent) == audit['longitudinal_snapshot']}
        old_cache = root / audit['benchmark_run']
    else:
        original = json.loads((source / 'source_metadata.json').read_text())['original_cohort']
        history_dir = root / 'hlr_model/data/runs/broad_na_16_19_seed_161926/processed' / original['snapshot']
        history_hashes = original['files_sha256']
        old_cache = root / 'hlr_model/data/lolalytics/broad_na_16_19_seed_161926_exact_tiers_v1'
    for name in ['historical_matches.parquet', 'target_matches.parquet', 'player_coverage.parquet']:
        require(file_sha(history_dir / name) == history_hashes[name], f'History source changed: {name}')
    manifest = json.loads((source / 'fits/shared/fit_manifest.json').read_text())
    status = json.loads((source / 'fits/shared/fit_status.json').read_text())
    diag = pd.read_csv(source / 'fits/shared/sampling_diagnostics.csv')
    require(manifest['model'] == 'shared' and manifest['sampling'] == dict(
        chains=4, draws=1200, warmup=600, max_tree_depth=9, train_match_limit=None), 'Not the completed shared fit.')
    require(status['status'] == 'converged' and status['diagnostic_gate_passed'], 'Fit not converged.')
    require(np.isfinite(diag[['rank_split_Rhat', 'bulk_ESS', 'tail_ESS']]).all().all()
            and diag.rank_split_Rhat.max() < 1.01 and diag.bulk_ESS.min() > 400 and diag.tail_ESS.min() > 400,
            'Saved diagnostic gate failed.')
    require(len(status['chains']) == 4 and all(c['divergences'] == 0 and c['acceptance_rate'] >= .8
                                             for c in status['chains']), 'Chain gate failed.')
    return dict(root=root, cohort=cohort, source=source, out=out, played=played, test=test,
                main=main, usable=usable, hashes=hashes, history_dir=history_dir,
                history_hashes=history_hashes, old_cache=old_cache, manifest=manifest,
                model=master if cohort == 'Master' else broad)


def table_tasks(ctx):
    rows = []
    for (rank, lane), games in ctx['usable'].groupby(['model_rank', 'lane']):
        tier = 'master_plus' if ctx['cohort'] == 'Master' else rank.lower()
        for cid, entry in sorted(scraper.CHAMPIONS.items()):
            # A table is unnecessary only if this champion is excluded in every game in this context.
            if ((games.champion_id == cid) | (games.opponent_champion_id == cid)).all():
                continue
            url = f"https://lolalytics.com/lol/{entry['slug']}/build/?lane={scraper.ROLE_SLUGS[lane]}&patch=16.19&tier={tier}"
            rows.append(dict(table_id=f'16.19:{tier}:global:ranked:{lane}:{cid}', tier=tier,
                             champion_id=cid, champion_slug=entry['slug'], lane=lane,
                             patch='16.19', region='global', queue='ranked', queue_id=420,
                             requested_region='global', source_url=url, model_rank=rank))
    return rows


def receipt_index(ctx):
    """Index existing receipts without writing to their databases or source folders."""
    index = {}
    folder = ctx['old_cache'] / ('responses' if ctx['cohort'] == 'Master' else 'receipts')
    for path in sorted(folder.glob('*.json')):
        receipt = json.loads(path.read_text())
        url = receipt.get('requested_url', '')
        if receipt.get('http_status') != 200 or not receipt.get('body_file'):
            continue
        query = parse_qs(urlparse(url).query)
        if (query.get('patch') != ['16.19'] or query.get('region', ['global']) != ['global']
                or '/vs/' in url):
            continue
        index[exact.canonical_url(url)] = (str(ctx['old_cache']), str(path))
    return index


def parse_receipt(job):
    task, base_name, receipt_name = job
    base, receipt_path = Path(base_name), Path(receipt_name)
    receipt = json.loads(receipt_path.read_text())
    require(receipt['http_status'] == 200, 'Non-success response.')
    require(exact.canonical_url(receipt['requested_url']) == exact.canonical_url(task['source_url'])
            == exact.canonical_url(receipt.get('response_url', receipt['requested_url'])), 'Redirected/wrong table.')
    body_path = (base / receipt['body_file']).resolve()
    require(body_path.is_relative_to(base.resolve()) and file_sha(body_path) == receipt['body_sha256'], 'Bad cached body.')
    html = body_path.read_text(errors='replace')
    evidence = dict(receipt_file=str(receipt_path), receipt_sha256=file_sha(receipt_path),
                    body_file=str(body_path), body_sha256=receipt['body_sha256'], source_url=task['source_url'])
    try:
        if task['tier'] == 'master_plus':
            _, dataset, _ = scraper.qwik_dataset(html)
            enemy = dataset['enemy'][scraper.ROLE_SLUGS[task['lane']]]
            # A single sentinel still verifies the table when the same-role list is empty.
            ids = [v[0] for v in enemy] or [task['champion_id']]
            tasks = [dict(task, opponent_champion_id=oid, key_id=str(oid)) for oid in ids]
            values, _ = scraper.parse_qwik_batch(html, tasks)
            rows = [dict(opponent_champion_id=int(oid), raw_matchup_winrate_pct=v['raw_matchup_winrate_pct'],
                         matchup_games=int(v['matchup_games'])) for oid, v in values.items() if v['status'] == 'ok']
        else:
            task = dict(task, tier=task['tier'].upper())
            parsed = exact.parse_table(html, task)
            rows = [dict(opponent_champion_id=int(oid), raw_matchup_winrate_pct=v['wr'], matchup_games=int(v['n']))
                    for oid, v in parsed['rows'].items()]
        return dict(table_id=task['table_id'], status='verified', rows=rows, **evidence)
    except (scraper.EvidenceError, ValueError, KeyError, TypeError, IndexError) as error:
        return dict(table_id=task['table_id'], status='validation_failed', error=str(error), rows=[], **evidence)


def coverage(ctx, *, fetch=False, workers=4):
    tasks = table_tasks(ctx)
    cache = ctx['out'] / 'cache'
    parsed_dir = cache / 'parsed_tables'
    parsed_dir.mkdir(parents=True, exist_ok=True)
    index = receipt_index(ctx)
    jobs, results = [], {}
    for task in tasks:
        file = parsed_dir / (scraper.digest(task['table_id'].encode()) + '.json')
        if file.exists():
            value = json.loads(file.read_text())
            require(file_sha(value['body_file']) == value['body_sha256']
                    and file_sha(value['receipt_file']) == value['receipt_sha256'], 'Parsed-cache evidence changed.')
            results[task['table_id']] = value
        elif exact.canonical_url(task['source_url']) in index:
            base, receipt = index[exact.canonical_url(task['source_url'])]
            jobs.append((task, base, receipt))
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, value in enumerate(pool.map(parse_receipt, jobs), 1):
                results[value['table_id']] = value
                save_json(parsed_dir / (scraper.digest(value['table_id'].encode()) + '.json'), value)
                if i % 50 == 0 or i == len(jobs):
                    print(ctx['cohort'], 'verified cached tables', i, '/', len(jobs), flush=True)
    missing = [task for task in tasks if task['table_id'] not in results]
    network = cache / 'missing_tables'
    network.mkdir(exist_ok=True)
    paused = None
    if fetch and missing:
        http = scraper.PageCache(network, scraper.Config(delay_seconds=1.0))
        pending = set()
        def retain(value):
            results[value['table_id']] = value
            if value.get('body_file'):
                save_json(parsed_dir / (scraper.digest(value['table_id'].encode()) + '.json'), value)
        try:
            # HTTP stays sequential and paced by PageCache. CPU parsing can overlap it.
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, task in enumerate(missing, 1):
                    try:
                        _, receipt = http.get(task['source_url'])
                        receipt_path = network / 'responses' / (scraper.digest(task['source_url'].encode()) + '.json')
                        if receipt['http_status'] != 200:
                            retain(dict(table_id=task['table_id'], status='unavailable', rows=[],
                                error=f"HTTP {receipt['http_status']}", source_url=task['source_url'],
                                receipt_file=str(receipt_path), receipt_sha256=file_sha(receipt_path),
                                body_file=str(network / receipt['body_file']), body_sha256=receipt['body_sha256']))
                        else:
                            pending.add(pool.submit(parse_receipt, (task, str(network), str(receipt_path))))
                        done = {future for future in pending if future.done()}
                        if len(pending) >= 2*workers and not done:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            retain(future.result())
                            pending.remove(future)
                        if i % 20 == 0:
                            print(ctx['cohort'], 'missing tables attempted', i, '/', len(missing), flush=True)
                    except (scraper.PauseCollection, scraper.EvidenceError) as error:
                        paused = str(error)
                        print(ctx['cohort'], 'coverage paused:', paused, flush=True)
                        break
                for future in pending:
                    retain(future.result())
        finally:
            http.close()
    table_rows, lookup = [], []
    for task in tasks:
        value = results.get(task['table_id'], dict(status='pending', rows=[]))
        table_rows.append({**task, **{k: v for k, v in value.items() if k != 'rows'}})
        if value['status'] == 'verified':
            for row in value['rows']:
                lookup.append(dict(table_id=task['table_id'], model_rank=task['model_rank'], lane=task['lane'],
                                   candidate_champion_id=task['champion_id'], **row))
    tables = pd.DataFrame(table_rows)
    lookup = pd.DataFrame(lookup)
    require(not lookup.duplicated(['model_rank', 'lane', 'candidate_champion_id', 'opponent_champion_id']).any(), 'Duplicate matchup.')
    tables.to_parquet(ctx['out'] / 'matchup_table_coverage.parquet', index=False)
    lookup.to_parquet(ctx['out'] / 'published_matchup_lookup.parquet', index=False)
    save_json(ctx['out'] / 'coverage_status.json', dict(required_tables=len(tables),
        counts=tables.status.value_counts().to_dict(), complete=bool(tables.status.eq('verified').all()),
        all_required_pages_attempted=bool(~tables.status.eq('pending').any()),
        pause_reason=paused, universe='eligible published same-role matchup candidates in verified tables; not draft availability'))
    print(ctx['cohort'], 'coverage', tables.status.value_counts().to_dict(), flush=True)
    return tables, lookup


def candidate_features(ctx, lookup):
    """Rebuild all-role/role counters from complete preserved pregame histories."""
    history = pd.read_parquet(ctx['history_dir'] / 'historical_matches.parquet')
    targets = pd.read_parquet(ctx['history_dir'] / 'target_matches.parquet')
    require(not pd.concat([history, targets]).duplicated(KEY).any(), 'History/target duplication.')
    pools = {p: g.sort_values(['end_timestamp', 'match_id'])
             for p, g in pd.concat([history, targets]).groupby('puuid')}
    group_table = pd.read_parquet(ctx['source'] / 'champion_grouping_lookup.parquet')
    require(file_sha(ctx['source'] / 'champion_grouping_lookup.parquet') == ctx['hashes']['champion_grouping_lookup.parquet'], 'Group mapping changed.')
    groups = group_table.set_index('champion_id').mastery_group.to_dict()
    names = {cid: entry['riot_name'] for cid, entry in scraper.CHAMPIONS.items()}
    # Explicitly distinguish unpublished opponent rows from unavailable tables.
    tables = pd.read_parquet(ctx['out'] / 'matchup_table_coverage.parquet')
    audit = ctx['usable'][KEY+['target_row_id','model_rank','lane','champion_id','opponent_champion_id','matchup_probability']].rename(
        columns={'champion_id':'main_champion_id','matchup_probability':'main_matchup_probability'}).merge(
        tables[['model_rank','lane','champion_id','table_id','status']].rename(columns={'champion_id':'candidate_champion_id','status':'table_status'}),
        on=['model_rank','lane'],validate='many_to_many')
    audit = audit.loc[audit.candidate_champion_id.ne(audit.main_champion_id) & audit.candidate_champion_id.ne(audit.opponent_champion_id)]
    audit = audit.merge(lookup.drop(columns='table_id'),on=['model_rank','lane','candidate_champion_id','opponent_champion_id'],
                       how='left',validate='many_to_one')
    audit['benchmark_status'] = np.select([
        audit.table_status.ne('verified'), audit.matchup_games.isna(), audit.matchup_games.le(0),
        audit.raw_matchup_winrate_pct.div(100).gt(audit.main_matchup_probability)],
        [audit.table_status,'not_published_in_table','published_zero_games','eligible_better_matchup'],
        default='published_not_strictly_better')
    require(not audit.duplicated(KEY+['candidate_champion_id']).any(), 'Duplicate candidate coverage row.')
    audit.to_parquet(ctx['out']/'candidate_benchmark_coverage.parquet',index=False)
    audit.groupby('benchmark_status').agg(candidate_rows=('candidate_champion_id','size'),
        player_games=('target_row_id','nunique'),players=('puuid','nunique')).to_csv(ctx['out']/'candidate_benchmark_coverage_summary.csv')
    eligible_lookup = lookup.loc[lookup.matchup_games.gt(0) & lookup.raw_matchup_winrate_pct.between(0, 100)]
    by_context = {key: g for key, g in eligible_lookup.groupby(['model_rank', 'lane', 'opponent_champion_id'])}
    required_keys = {(t['model_rank'], t['lane'], t['champion_id']) for t in table_tasks(ctx)}
    rows, main_checks = [], []
    for player, games in ctx['usable'].groupby('puuid', sort=True):
        require(player in pools, 'Missing history pool is not zero experience.')
        pool = list(pools[player].itertuples(index=False))
        counts, wins, roles = Counter(), Counter(), Counter()
        latest, role_latest = {}, {}
        cursor = total = 0
        for main in games.sort_values(['timestamp', 'match_id']).itertuples(index=False):
            while cursor < len(pool) and pool[cursor].end_timestamp < main.timestamp:
                prior = pool[cursor]
                require(prior.timestamp < main.timestamp and prior.win in (0, 1), 'Invalid prior record.')
                cid = int(prior.champion_id)
                counts[cid] += 1
                wins[cid] += int(prior.win)
                roles[(prior.lane, cid)] += 1
                latest[cid] = prior.end_timestamp
                role_latest[(prior.lane, cid)] = prior.end_timestamp
                total += 1
                cursor += 1
            require(total == main.prior_total_games, 'All-role history depth mismatch.')
            recorded_main = min(counts, key=lambda cid: (-counts[cid], cid)) if counts else None
            require(recorded_main == main.prior_main_champion_id == main.champion_id, 'Pregame main mismatch.')
            require(counts[main.champion_id] == main.n_recorded, 'Main experience mismatch.')
            np.testing.assert_allclose((wins[main.champion_id]+1)/(counts[main.champion_id]+2), main.historical_wr_beta11)
            available = by_context.get((main.model_rank, main.lane, int(main.opponent_champion_id)), eligible_lookup.iloc[:0])
            own = available.loc[available.candidate_champion_id.eq(main.champion_id)]
            if (main.model_rank, main.lane, int(main.champion_id)) in required_keys:
                require(len(own) == 1, 'Saved usable main has no matching verified full table.')
                np.testing.assert_allclose(own.raw_matchup_winrate_pct.iloc[0]/100, main.matchup_probability, rtol=0, atol=1e-12)
            # Otherwise the main's table cannot supply a candidate in this context;
            # its benchmark remains the independently hash-verified prepared value.
            options = available.loc[available.candidate_champion_id.ne(main.champion_id)
                & available.candidate_champion_id.ne(main.opponent_champion_id)
                & available.raw_matchup_winrate_pct.div(100).gt(main.matchup_probability)]
            main_checks.append(dict(puuid=player, match_id=main.match_id, prior_total_games=total,
                                    main_same_role_games=roles[(main.lane, int(main.champion_id))], candidates=len(options)))
            for candidate in options.itertuples(index=False):
                cid = int(candidate.candidate_champion_id)
                n, w, rn = counts[cid], wins[cid], roles[(main.lane, cid)]
                # Carry only player/game context; do not copy the main's champion-specific fields.
                context_columns = KEY + ['target_row_id', 'timestamp', 'lane', 'opponent_champion_id',
                    'model_rank', 'split', 'prior_total_games']
                row = {name:getattr(main,name) for name in context_columns}
                row.update(row_kind='alternative', model_champion_id=cid, candidate_champion_id=cid,
                    champion_id=cid, champion=names[cid], is_main_champion=False,
                    observed_champion_id=int(main.champion_id), main_champion_id=int(main.champion_id),
                    candidate_champion=names[cid], main_champion=names[int(main.champion_id)],
                    n_recorded=n, prior_champion_games=n, prior_champion_wins=w, prior_all_role_games=n,
                    prior_champion_wr=w/n if n else np.nan, prior_champion_wr_beta11=(w+1)/(n+2),
                    prior_champion_usage_share=n/total if total else np.nan,
                    prior_same_role_games=rn, historical_wr_beta11=(w+1)/(n+2), log_total=np.log1p(total),
                    matchup_probability=candidate.raw_matchup_winrate_pct/100,
                    benchmark_raw_matchup_winrate_pct=candidate.raw_matchup_winrate_pct,
                    benchmark_raw_matchup_winrate_probability=candidate.raw_matchup_winrate_pct/100,
                    benchmark_valid_rate=True, usable_data=True,
                    benchmark_sparse=candidate.matchup_games < 100, benchmark_matchup_games=candidate.matchup_games,
                    benchmark_table_id=candidate.table_id, mastery_group=groups.get(cid, 'Unclassified'),
                    latest_all_role_prior_end_timestamp=latest.get(cid),
                    latest_same_role_prior_end_timestamp=role_latest.get((main.lane, cid)),
                    experience_class=HISTORY_CLASSES[0 if n == 0 else 1 if rn == 0 else 2],
                    candidate_outcome=np.nan, included_in_primary_model=False)
                rows.append(row)
    candidates = pd.DataFrame(rows).sort_values(KEY + ['candidate_champion_id']).reset_index(drop=True)
    require(len(candidates) > 0 and not candidates.duplicated(KEY + ['candidate_champion_id']).any(), 'Missing/duplicate candidates.')
    require(candidates.n_recorded.ge(0).all() and candidates.n_recorded.le(candidates.prior_total_games).all(), 'Invalid experience counts.')
    require(candidates.prior_champion_wins.between(0, candidates.n_recorded).all(), 'Invalid history wins.')
    require(candidates.candidate_outcome.isna().all() and 'win' not in candidates, 'Unplayed outcome contamination.')
    expected = audit.loc[audit.benchmark_status.eq('eligible_better_matchup'),KEY+['candidate_champion_id']]
    require(set(map(tuple,expected.to_numpy())) == set(map(tuple,candidates[KEY+['candidate_champion_id']].to_numpy())), 'Candidate coverage/feature eligibility differs.')
    positive = candidates.n_recorded.gt(0)
    require(candidates.loc[positive, 'latest_all_role_prior_end_timestamp'].lt(candidates.loc[positive, 'timestamp']).all(), 'Future history.')
    zero = candidates.loc[~positive]
    require(zero.prior_champion_wins.eq(0).all() and zero.historical_wr_beta11.eq(.5).all()
            and zero.latest_all_role_prior_end_timestamp.isna().all(), 'Zero-history features incorrect.')
    candidates.to_parquet(ctx['out'] / 'expanded_candidate_features.parquet', index=False)
    pd.DataFrame(main_checks).to_parquet(ctx['out'] / 'main_history_checks.parquet', index=False)
    print(ctx['cohort'], len(candidates), 'eligible candidates in', candidates[KEY].drop_duplicates().shape[0], 'games', flush=True)
    return candidates


def frozen_fit(ctx):
    """Verify all chain archives before loading. Reuse the fitted prediction function."""
    manifest, module = ctx['manifest'], ctx['model']
    require(hashlib.sha256(inspect.getsource(module.pyro_model).encode()).hexdigest() == manifest['model_source_sha256'], 'Model source differs.')
    archives = []
    for i in range(4):
        name = f'fits/shared/chain_{i}.npz'
        info = json.loads((ctx['source'] / f'fits/shared/chain_{i}.json').read_text())
        require(file_sha(ctx['source'] / name) == info['samples_sha256'] == ctx['hashes'][name], 'Draw archive changed.')
        archives.append(ctx['source'] / name)
    train = pd.read_parquet(ctx['source'] / 'prepared_played.parquet', filters=[('split', '==', 'train')])
    design = module.Design(train.loc[train.usable_data])
    require(json.loads(exact.canonical(design.metadata())) == manifest['design'], 'Training design differs from saved design.')
    chains = []
    for path in archives:
        with np.load(path) as archive:
            chain = {key: archive[key] for key in archive.files}
        require(all(value.shape[0] == 1200 and np.isfinite(value).all() for value in chain.values()), 'Bad retained draws.')
        chains.append(chain)
    draws = {key: np.concatenate([chain[key] for chain in chains]) for key in chains[0]}
    require('group_free' not in draws, 'Grouped fit not allowed.')
    return design, draws


def score_candidates(ctx, candidates, *, batch_size=128):
    """Cache every paired comparison; reruns only score previously absent batches."""
    design, draws = frozen_fit(ctx)
    main = ctx['usable'].sort_values(KEY).reset_index(drop=True)
    indexed_main = main.set_index(KEY)
    identity = dict(schema=REPORT, candidate_sha256=file_sha(ctx['out'] / 'expanded_candidate_features.parquet'),
        fit_manifest_sha256=file_sha(ctx['source'] / 'fits/shared/fit_manifest.json'),
        draws_sha256=[file_sha(ctx['source'] / f'fits/shared/chain_{i}.npz') for i in range(4)],
        prediction_sha256=hashlib.sha256(inspect.getsource(broad.predict_draws).encode()).hexdigest(),
        master_prediction_sha256=hashlib.sha256(inspect.getsource(master.predict_draws).encode()).hexdigest(),
        seed=2026, batch_size=batch_size)
    cache_key = hashlib.sha256(exact.canonical(identity).encode()).hexdigest()[:20]
    cache = ctx['out'] / 'cache' / ('predictions_' + cache_key)
    cache.mkdir(parents=True, exist_ok=True)
    save_json(cache / 'identity.json', identity)
    outputs = []
    support = ctx['played'].loc[ctx['played'].split.eq('train') & ctx['played'].usable_data].groupby(['puuid', 'model_champion_id']).size()
    for mode in MODES:
        for start in range(0, len(candidates), batch_size):
            path = cache / f'{mode}_{start:07d}.parquet'
            part = candidates.iloc[start:start+batch_size]
            if path.exists():
                summary = pd.read_parquet(path)
                require(summary.candidate_champion_id.tolist() == part.candidate_champion_id.tolist()
                        and summary.target_row_id.tolist() == part.target_row_id.tolist(), 'Cached batch keys differ.')
            else:
                refs = indexed_main.loc[pd.MultiIndex.from_frame(part[KEY])].reset_index()
                pr = ctx['model'].predict_draws(design, refs, draws, mode=mode)
                pa = ctx['model'].predict_draws(design, part, draws, mode=mode)
                delta = 100*(pa-pr)
                require(np.isfinite(pa).all() and np.isfinite(pr).all(), 'Nonfinite prediction.')
                keep = KEY + ['target_row_id', 'timestamp', 'lane', 'opponent_champion_id', 'model_rank',
                    'main_champion_id', 'candidate_champion_id', 'main_champion', 'candidate_champion',
                    'n_recorded', 'prior_champion_wins', 'prior_same_role_games', 'prior_total_games',
                    'historical_wr_beta11', 'mastery_group', 'matchup_probability', 'benchmark_sparse',
                    'benchmark_matchup_games', 'experience_class']
                summary = part[keep].copy()
                summary['mode'] = MODES[mode]
                summary['main_recorded_games'] = refs.n_recorded.to_numpy()
                summary['main_historical_wr_beta11'] = refs.historical_wr_beta11.to_numpy()
                summary['main_matchup_probability'] = refs.matchup_probability.to_numpy()
                require(summary.matchup_probability.gt(summary.main_matchup_probability).all(), 'Alternative benchmark is not strictly better.')
                summary['main_probability'] = pr.mean(axis=0)
                summary['alternative_probability'] = pa.mean(axis=0)
                summary['difference_pp'] = delta.mean(axis=0)
                summary['low_pp'], summary['high_pp'] = np.quantile(delta, [.025, .975], axis=0)
                summary['probability_alternative_higher'] = (delta > 0).mean(axis=0)
                summary['candidate_pc_training_games'] = [int(support.get((p, int(cid)), 0)) for p, cid in zip(part.puuid, part.candidate_champion_id)]
                summary['candidate_champion_seen_training'] = part.candidate_champion_id.isin(design.champions)
                summary.to_parquet(path, index=False)
            outputs.append(summary)
            if start % (batch_size*20) == 0:
                print(ctx['cohort'], MODES[mode], min(start+batch_size, len(candidates)), '/', len(candidates), 'candidates scored/cached', flush=True)
    pairs = pd.concat(outputs, ignore_index=True)
    require(len(pairs) == 2*len(candidates) and not pairs.duplicated(KEY + ['mode', 'candidate_champion_id']).any(), 'Pair counts differ.')
    np.testing.assert_allclose(pairs.difference_pp, 100*(pairs.alternative_probability-pairs.main_probability), atol=1e-10)
    # Identity-stable pooled draws must make each main prediction independent of batch/candidate.
    require(pairs.groupby(KEY + ['mode']).main_probability.agg(lambda x: x.max()-x.min()).le(1e-12).all(), 'Unpaired main predictions.')
    pairs.to_parquet(ctx['out'] / 'all_candidate_comparisons.parquet', index=False)
    save_json(ctx['out'] / 'prediction_cache_manifest.json', dict(identity=identity, cache=str(cache.relative_to(ctx['out'])),
        output_sha256=file_sha(ctx['out'] / 'all_candidate_comparisons.parquet'), retained_draws=4800,
        note='Existing shared-model predictions only; no fit, likelihood rescoring or predictive evaluation.'))
    return pairs


def decisions_and_summaries(ctx, candidates, pairs):
    selected = []
    for rule in RULES:
        score = 'matchup_probability' if rule == RULES[0] else 'alternative_probability'
        top = pairs.sort_values(KEY + ['mode', score, 'candidate_champion_id'],
            ascending=[True, True, True, False, True], kind='stable').drop_duplicates(KEY + ['mode']).copy()
        top['selection_rule'] = rule
        top['classification'] = np.select([top.low_pp.gt(0), top.high_pp.lt(0)], CATEGORIES[:2], default=CATEGORIES[2])
        top['alternative_has_higher_mean'] = top.alternative_probability.gt(top.main_probability)
        # As in the familiar report, an exact overall mean tie uses the smallest champion ID.
        top['point_prefers_leaving_main'] = top.alternative_has_higher_mean | (
            top.alternative_probability.eq(top.main_probability) & top.candidate_champion_id.lt(top.main_champion_id))
        top['eligible_alternatives'] = top.target_row_id.map(candidates.groupby('target_row_id').size())
        selected.append(top)
    decisions = pd.concat(selected, ignore_index=True)
    require(not decisions.duplicated(KEY + ['selection_rule', 'mode']).any(), 'Duplicate selected decision.')
    eligible_keys = candidates[KEY].drop_duplicates()
    eligible = ctx['usable'].merge(eligible_keys, on=KEY, validate='one_to_one')
    missing = ctx['usable'].merge(eligible_keys.assign(has_alternative=True), on=KEY, how='left', validate='one_to_one')
    missing = missing.loc[missing.has_alternative.isna()]
    rows = []

    def add(section, measure, frame, denominator, *, rule='Both rules', mode='Both modes'):
        rows.append(dict(cohort=ctx['cohort'], selection_rule=rule, mode=mode, section=section,
            measure=measure, count=len(frame), denominator=denominator,
            percent=100*len(frame)/denominator if denominator else np.nan,
            players=frame.puuid.nunique(), unique_matches=frame.match_id.nunique()))

    for label, frame in [('Total actual main-pick games', ctx['main']),
                         ('Main-pick games with usable main features/benchmarks', ctx['usable']),
                         ('Games with eligible published better-matchup alternatives', eligible),
                         ('Usable main games without an eligible published alternative', missing)]:
        add('coverage', label, frame, len(ctx['main']))
    invalid = ctx['main'].loc[~ctx['main'].target_row_id.isin(ctx['usable'].target_row_id)]
    for reason, frame in invalid.groupby('data_exclusion_reasons'):
        add('coverage_loss', str(reason), frame, len(ctx['main']))
    require(len(invalid)+len(missing)+len(eligible) == len(ctx['main']), 'Coverage losses do not reconcile.')
    add('coverage_loss', 'No eligible published better-matchup alternative (not a confirmed main preference)', missing, len(ctx['main']))
    for (rule, mode), frame in decisions.groupby(['selection_rule', 'mode']):
        require(len(frame) == len(eligible), 'Incorrect game denominator.')
        add('point_preference', 'Alternative has strictly higher mean prediction', frame.loc[frame.alternative_has_higher_mean], len(frame), rule=rule, mode=mode)
        for label in CATEGORIES:
            add('classification', label, frame.loc[frame.classification.eq(label)], len(frame), rule=rule, mode=mode)
        for label in HISTORY_CLASSES:
            add('selected_experience', label, frame.loc[frame.experience_class.eq(label)], len(frame), rule=rule, mode=mode)
        for label in CATEGORIES:
            zero = frame.loc[frame.n_recorded.eq(0)]
            add('zero_history_classification', label, zero.loc[zero.classification.eq(label)], len(zero), rule=rule, mode=mode)
        add('selected_benchmark', 'Selected alternative has fewer than 100 matchup games (flag only)',
            frame.loc[frame.benchmark_sparse], len(frame), rule=rule, mode=mode)
    for rule in RULES:
        reference = decisions.loc[decisions.selection_rule.eq(rule) & decisions['mode'].eq('Reference player')]
        personal = decisions.loc[decisions.selection_rule.eq(rule) & decisions['mode'].eq('Personalized')]
        both = reference.merge(personal, on=KEY, suffixes=('_reference', '_personalized'), validate='one_to_one')
        for label, field in [('Stay-versus-leave point preference changed', 'point_prefers_leaving_main'),
                             ('Selected alternative changed', 'candidate_champion_id'),
                             ('Interval classification changed', 'classification')]:
            changed = both.loc[both[field+'_reference'].ne(both[field+'_personalized'])]
            add('personalization_change', label, changed, len(both), rule=rule)
    summary = pd.DataFrame(rows)
    decisions.attrs['provenance'] = dict(source_version=ctx['source'].name,
        helper_sha256=file_sha(__file__), selection='Raw benchmark or posterior mean; smallest champion ID breaks ties.',
        paired_draws='4800 retained shared-model draws; identity-stable pooled unseen effects; no refit.',
        input_sha256={name: file_sha(ctx['out']/name) for name in ['expanded_candidate_features.parquet', 'all_candidate_comparisons.parquet', 'published_matchup_lookup.parquet']},
        denominator='One actual-main-pick game per rule and mode with an eligible published better-matchup candidate; not pair-weighted.')
    decisions.to_parquet(ctx['out'] / 'game_level_decisions.parquet', index=False)
    summary.to_csv(ctx['out'] / 'summary.csv', index=False)
    return decisions, summary


def familiar_comparison(ctx, candidates, pairs, decisions):
    old_dir = ctx['source'] / 'reports/main_pick_decisions_v1'
    old = pd.read_parquet(old_dir / 'game_level_decisions.parquet')
    old_hashes = {str(p.relative_to(old_dir)): file_sha(p) for p in old_dir.iterdir() if p.is_file()}
    old_keys = old[KEY].drop_duplicates()
    new_keys = candidates[KEY].drop_duplicates()
    matched_keys = old_keys.merge(new_keys, on=KEY, validate='one_to_one')
    require(len(matched_keys) == len(old_keys), 'Expanded universe lost a familiar-eligible game.')
    # Reproduce each saved selected familiar comparison using the same frozen predictor.
    old_selected = old.merge(pairs, left_on=KEY+['mode', 'alternative_champion_id'],
        right_on=KEY+['mode', 'candidate_champion_id'], suffixes=('_old', '_new'), validate='one_to_one')
    require(len(old_selected) == len(old), 'A familiar candidate disappeared.')
    for field in ['alternative_probability', 'difference_pp', 'low_pp', 'high_pp']:
        np.testing.assert_allclose(old_selected[field+'_old'], old_selected[field+'_new'], rtol=0, atol=1e-10)
    rows, changes = [], []
    for rule in RULES:
        for mode in MODES.values():
            chosen = decisions.loc[decisions.selection_rule.eq(rule) & decisions['mode'].eq(mode)]
            matched = chosen.merge(old.loc[old['mode'].eq(mode)], on=KEY, suffixes=('_expanded', '_familiar'), validate='one_to_one')
            additional = chosen.loc[~chosen.target_row_id.isin(matched.target_row_id_expanded)]
            # Candidate expansion alone: restrict both calculations to exactly the old games.
            for label, mask in [
                ('Selected alternative changed on familiar-eligible games', matched.candidate_champion_id.ne(matched.alternative_champion_id)),
                ('Stay/leave point preference changed on familiar-eligible games', matched.point_prefers_leaving_main_expanded.ne(matched.point_prefers_leaving_main_familiar)),
                ('Interval classification changed on familiar-eligible games', matched.classification_expanded.ne(matched.classification_familiar))]:
                rows.append(dict(cohort=ctx['cohort'], selection_rule=rule, mode=mode, measure=label,
                    count=int(mask.sum()), denominator=len(matched), percent=100*mask.mean(), players=matched.loc[mask, 'puuid'].nunique()))
            for scope, frame in [('Original familiar-eligible games', matched), ('Additional expanded-eligible games', additional)]:
                classification = 'classification_expanded' if scope.startswith('Original') else 'classification'
                for category in CATEGORIES:
                    count = int(frame[classification].eq(category).sum())
                    rows.append(dict(cohort=ctx['cohort'], selection_rule=rule, mode=mode,
                        measure=scope+' — '+category, count=count, denominator=len(frame),
                        percent=100*count/len(frame) if len(frame) else np.nan,
                        players=frame.loc[frame[classification].eq(category), 'puuid'].nunique()))
            changes.append(matched)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ctx['out'] / 'familiar_matched_comparison.csv', index=False)
    pd.concat(changes, ignore_index=True).to_parquet(ctx['out'] / 'matched_familiar_decisions.parquet', index=False)
    # Separately isolate expanding the universe on the original games, using the ORIGINAL model selection rule.
    # Raw-baseline selection is a different rule, so do not attribute its entire change to candidate expansion.
    save_json(ctx['out'] / 'familiar_preservation.json', dict(original_eligible_games=len(old_keys),
        expanded_eligible_games=len(new_keys), added_eligible_games=len(new_keys)-len(old_keys),
        original_report_sha256=old_hashes,
        reproduced_selected_familiar_predictions=True,
        original_rule='Highest posterior mean among previously played same-role alternatives',
        candidate_expansion_comparison='Best model prediction on matched original games isolates the changed candidate scope.'))
    return comparison


def choose_examples(decisions):
    """Actual chronological examples, preferring the direct highest-baseline question."""
    chosen = []
    for label in ['Zero-history alternative favored', 'Main favored over zero-history alternative', 'Uncertain comparison']:
        frame = decisions.loc[decisions['mode'].eq('Personalized') & decisions.selection_rule.eq(RULES[0])]
        if label == 'Zero-history alternative favored':
            frame = frame.loc[frame.n_recorded.eq(0) & frame.classification.eq(CATEGORIES[0])]
        elif label == 'Main favored over zero-history alternative':
            frame = frame.loc[frame.n_recorded.eq(0) & frame.classification.eq(CATEGORIES[1])]
        else:
            frame = frame.loc[frame.classification.eq(CATEGORIES[2])]
        if frame.empty:
            chosen.append(dict(example_category=label, available=False))
        else:
            row = frame.sort_values(['timestamp', *KEY]).iloc[0].to_dict()
            row.update(example_category=label, available=True)
            chosen.append(row)
    return pd.DataFrame(chosen)


def build_report(root, cohort, *, fetch=False):
    ctx = context(root, cohort)
    tables, lookup = coverage(ctx, fetch=fetch)
    candidates = candidate_features(ctx, lookup)
    pairs = score_candidates(ctx, candidates)
    decisions, summary = decisions_and_summaries(ctx, candidates, pairs)
    familiar_comparison(ctx, candidates, pairs, decisions)
    choose_examples(decisions).to_parquet(ctx['out'] / 'examples.parquet', index=False)
    save_json(ctx['out'] / 'report_manifest.json', dict(schema=REPORT, cohort=cohort,
        helper_sha256=file_sha(__file__), source_version=ctx['source'].name,
        source_history_sha256={str((ctx['history_dir']/name).relative_to(ctx['root'])):ctx['history_hashes'][name]
            for name in ['historical_matches.parquet','target_matches.parquet','player_coverage.parquet']},
        saved_fit_manifest_sha256=file_sha(ctx['source']/'fits/shared/fit_manifest.json'),
        output_sha256={p.name: file_sha(p) for p in ctx['out'].iterdir() if p.is_file() and p.name != 'report_manifest.json'},
        qualification='Retrospective exploratory selected comparisons, not selection-adjusted; draft availability and unplayed outcomes unknown.',
        zero_experience='Zero in collected Season 16 history, not lifetime experience.'))
    print('Expanded report:', ctx['out'], flush=True)
    return ctx['out']


def load_report(root, cohort):
    folder, version, _ = COHORTS[cohort]
    out = Path(root).resolve() / 'hlr_model/data/analyses' / folder / version / 'reports' / REPORT
    manifest = json.loads((out / 'report_manifest.json').read_text())
    require(manifest['source_version'] == version and manifest['helper_sha256'] == file_sha(__file__), 'Report helper/provenance changed; rebuild from cache.')
    for name, expected in manifest['output_sha256'].items():
        require(file_sha(out / name) == expected, f'Report artifact changed: {name}')
    return dict(out=out, summary=pd.read_csv(out/'summary.csv'),
        decisions=pd.read_parquet(out/'game_level_decisions.parquet'),
        examples=pd.read_parquet(out/'examples.parquet'),
        comparison=pd.read_csv(out/'familiar_matched_comparison.csv'),
        coverage=json.loads((out/'coverage_status.json').read_text()),
        familiar=json.loads((out/'familiar_preservation.json').read_text()))


def summary_table(reports):
    rows = []
    for cohort, report in reports.items():
        summary = report['summary']
        for rule in RULES:
            for mode in MODES.values():
                s = summary.loc[summary.selection_rule.eq(rule) & summary['mode'].eq(mode)]
                def value(section, measure):
                    r = s.loc[s.section.eq(section) & s.measure.eq(measure)].iloc[0]
                    return f'{int(r["count"])} ({r.percent:.1f}%)'
                n = report['decisions'].loc[lambda f: f.selection_rule.eq(rule) & f['mode'].eq(mode)].shape[0]
                rows.append({'Cohort':cohort, 'Selection rule':rule, 'Prediction mode':mode, 'Eligible games':n,
                    'Alternative higher mean':value('point_preference', 'Alternative has strictly higher mean prediction'),
                    'Alternative favored':value('classification', CATEGORIES[0]),
                    'Main favored':value('classification', CATEGORIES[1]),
                    'Uncertain':value('classification', CATEGORIES[2]),
                    'Selected zero history':value('selected_experience', HISTORY_CLASSES[0])})
    return pd.DataFrame(rows)


def white_table(frame, caption):
    from IPython.display import display
    display(frame.style.hide(axis='index').set_caption(caption)
        .set_table_attributes('style="background:white; color:black; border-collapse:collapse; width:100%;"')
        .set_properties(**{'background-color':'white', 'color':'black', 'border':'none',
            'border-bottom':'1px solid #dddddd', 'padding':'10px', 'white-space':'normal', 'font-size':'14px'})
        .set_table_styles([
            {'selector':'caption', 'props':[('background','white'),('color','black'),('font-size','20px'),
                ('font-weight','bold'),('text-align','left'),('padding','12px 0')]},
            {'selector':'th', 'props':[('background','white'),('color','black'),('font-weight','bold'),
                ('border','none'),('border-bottom','2px solid black'),('padding','10px'),('white-space','normal')]}]))


def display_examples(cohort, examples):
    from IPython.display import Markdown, display
    display(Markdown('#### '+cohort+' — actual saved examples (personalized; best baseline matchup)'))
    for row in examples.itertuples(index=False):
        if not row.available:
            display(Markdown('**'+row.example_category+':** no example in these completed comparisons.'))
            continue
        date = pd.to_datetime(row.timestamp, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M UTC')
        display(Markdown(f'**{row.example_category}:** {date}, {row.lane}, '
            f'{row.main_champion} → {row.candidate_champion}. '
            f'Recorded champion games {int(row.main_recorded_games)} → {int(row.n_recorded)}; '
            f'raw matchup {100*row.main_matchup_probability:.2f}% → {100*row.matchup_probability:.2f}% '
            f'(candidate benchmark n={int(row.benchmark_matchup_games)}). '
            f'Modeled win probability {100*row.main_probability:.2f}% → {100*row.alternative_probability:.2f}%; '
            f'alternative − main **{row.difference_pp:+.2f} pp [{row.low_pp:+.2f}, {row.high_pp:+.2f}]** '
            f'(95% credible interval); **{row.classification}**.'))


def benchmark_selection_table(report):
    frame = report['decisions'].copy()
    frame['raw_100_percent'] = frame.matchup_probability.eq(1)
    return frame.groupby(['selection_rule','mode'],sort=False).agg(
        selected_games=('target_row_id','size'), raw_100_percent=('raw_100_percent','sum'),
        benchmark_n_min=('benchmark_matchup_games','min'), benchmark_n_median=('benchmark_matchup_games','median'),
        benchmark_n_max=('benchmark_matchup_games','max'), sparse_n_below_100=('benchmark_sparse','sum')).reset_index()


def display_report(root, cohort):
    from IPython.display import Markdown, display
    report = load_report(root, cohort)
    display(Markdown('### '+cohort+' — expanded eligible published matchup candidates'))
    display(Markdown('Shared models and 4,800 saved posterior draws; no refit or predictive evaluation. '
        'Highest raw matchup and highest model prediction are separate rules; smallest champion ID breaks ties. '
        'Reference-player predictions set player/player–champion effects to zero and are not population averages.'))
    coverage = report['coverage']
    display(Markdown(f"**Matchup-table coverage:** {coverage['counts']} of {coverage['required_tables']} required pages. "
        f"All pages attempted: {coverage.get('all_required_pages_attempted',False)}. "
        'Unverified, unavailable and unpublished rates stay missing; no minimum-game filter is imposed. Sparse n<100 flags are retained.'))
    s = report['summary']
    white_table(s.loc[s.section.isin(['coverage','coverage_loss']), ['measure','count','denominator','players','unique_matches']],
                'Coverage — actual main-pick games, not pairs')
    display(pd.read_csv(report['out']/'candidate_benchmark_coverage_summary.csv'))
    white_table(summary_table({cohort:report}), 'Expanded selections — counts and percentages among eligible games')
    white_table(benchmark_selection_table(report), 'Selected raw benchmarks — tiny samples can dominate the expanded search')
    display(Markdown('A published 100% raw matchup from one or a few games is an eligible input under the requested no-threshold rule, '
        'not proof of a true 100% win chance. The frozen model treats this benchmark as fixed and does not propagate its sampling error. '
        'Extreme interval-supported preferences therefore need the benchmark sample counts beside them; they do not establish reliable pick advice. '
        'When several champions tie at the maximum raw rate, the baseline rule uses the smallest champion ID.'))
    display(s.loc[s.section.isin(['selected_experience','zero_history_classification','personalization_change','selected_benchmark'])])
    familiar = report['familiar']
    display(Markdown(f"**Preserved familiar-only scope:** {familiar['original_eligible_games']} original eligible games; "
        f"{familiar['added_eligible_games']} additional games in the expanded scope. "
        'The old 17.6% Master / 2.1% broad personalized alternative-favored figures apply only to familiar same-role alternatives. '
        'Best-model results on matched original games isolate candidate expansion; the baseline rule also changes the selection criterion.'))
    display(report['comparison'])
    display_examples(cohort, report['examples'])
    display(Markdown('Zero means zero in recorded Season 16 history, not lifetime experience. All-role counts, '
        'Beta(1,1) historical WR and total prior history are reconstructed strictly before kickoff; zero experience '
        'does not zero the learned coefficient, matchup benchmark or champion baseline. Unseen identities retain '
        'identity-stable pooled predictive uncertainty. These are retrospective exploratory conditional comparisons. '
        'Unplayed outcomes and draft availability are unknown; selected-comparison credible intervals are not selection-adjusted '
        'and omit external benchmark sampling/transport error.'))
    return report['out']


def display_presentation(root):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from IPython.display import Markdown, display
    reports = {cohort:load_report(root, cohort) for cohort in ['Master','Bronze–Diamond']}
    white_table(summary_table(reports), 'Expanded main-versus-alternative comparisons — saved shared models')
    for cohort, report in reports.items():
        s = report['summary']
        c = s.loc[s.section.eq('coverage')]
        display(Markdown('**'+cohort+':** '+ '; '.join(f'{r.measure}: {int(r.count)} ({int(r.players)} players)' for r in c.itertuples())+'. '
            f"Table coverage: {report['coverage']['counts']}; all pages attempted: {report['coverage'].get('all_required_pages_attempted',False)}."))
        white_table(benchmark_selection_table(report).loc[lambda f:f['mode'].eq('Personalized')],
                    cohort+' — selected matchup benchmark sample counts')
    display(Markdown('**Why an extreme result can appear:** the expanded no-threshold search can select published 100% matchup '
        'rates from one or a few games. Such a rate is not evidence of a true 100% chance of winning. '
        'The unchanged model treats the benchmark as fixed, so paired posterior intervals omit its sampling error '
        'as well as selection adjustment. Keep these sample counts visible when presenting the results. '
        'Equal highest raw benchmarks use the smallest champion ID; they do not establish a uniquely strongest champion.'))
    with plt.rc_context({'figure.facecolor':'white','axes.facecolor':'white','text.color':'black',
                         'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','font.size':12,
                         'savefig.transparent':False}):
        fig, axes = plt.subplots(2,2,figsize=(15,10),sharey=True,facecolor='white')
        for j,(cohort,report) in enumerate(reports.items()):
            for i,rule in enumerate(RULES):
                ax = axes[j,i]
                ax.set_facecolor('white')
                s = report['summary'].loc[lambda f:f.selection_rule.eq(rule) & f.section.eq('classification')]
                bottom = np.zeros(2)
                small = []
                for category,color in zip(CATEGORIES,['#2563A6','#C58B2A','#AAB2BD']):
                    f = s.loc[s.measure.eq(category)].set_index('mode').reindex(list(MODES.values()))
                    ax.bar(range(2),f.percent,bottom=bottom,color=color,width=.55,label=category)
                    for k,row in enumerate(f.itertuples()):
                        label = f'{int(row.count)} ({row.percent:.1f}%)'
                        if row.percent >= 7:
                            ax.text(k,bottom[k]+row.percent/2,label,ha='center',va='center',color='white' if category==CATEGORIES[0] else 'black',fontsize=11)
                        elif row.count:
                            small.append(('Ref' if k==0 else 'Personal')+' '+('main' if category==CATEGORIES[1] else 'alt' if category==CATEGORIES[0] else 'uncertain')+': '+label)
                    bottom += f.percent.to_numpy()
                ax.set(xticks=range(2),xticklabels=['Reference player','Personalized'],ylim=(0,100),
                    ylabel='Eligible main-pick games (%)',title=cohort+' — '+rule)
                ax.yaxis.set_major_formatter(PercentFormatter(100))
                if small: ax.text(.5,-.14,'; '.join(small),transform=ax.transAxes,ha='center',va='top',fontsize=9,wrap=True)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=11)
        fig.suptitle('Expanded candidates: interval-based selected comparisons',fontsize=20,color='black')
        fig.subplots_adjust(top=.91,bottom=.11,hspace=.43,wspace=.23)
        plt.show()
    for cohort,report in reports.items():
        white_table(report['summary'].loc[lambda f:f.section.eq('personalization_change'),
            ['selection_rule','measure','count','denominator','percent']],cohort+' — personalization changes')
        familiar = report['familiar']
        display(Markdown(f"**{cohort}, familiar-only versus expanded:** {familiar['original_eligible_games']} original eligible games "
            f"and {familiar['added_eligible_games']} newly eligible games. The following changes use matched original games; "
            'only the best-model row holds the original model-based selection rule fixed.'))
        white_table(report['comparison'].loc[lambda f:f['mode'].eq('Personalized') & f.measure.str.contains('changed on familiar'),
            ['selection_rule','measure','count','denominator','percent']],cohort+' — changes on original eligible games')
        white_table(report['summary'].loc[lambda f:f['mode'].eq('Personalized') & f.section.eq('selected_experience'),
            ['selection_rule','measure','count','denominator','percent']],cohort+' — selected alternatives’ recorded history')
        display_examples(cohort,report['examples'])
    display(Markdown('**Interpretation:** the highest published raw matchup is a theoretical benchmark pick; '
        'the model additionally conditions on recorded experience, historical champion WR and fitted champion/identity effects. '
        'The highest model-mean choice answers a different question. A higher mean is weaker evidence than a paired interval '
        'excluding zero. Sparse published matchup counts can produce extreme raw benchmarks, and their sampling error is '
        'outside these posterior intervals. The tables retain n<100 flags without filtering such candidates. '
        'Zero recorded experience is bounded Season 16 history. Draft availability and alternative outcomes are unknown; '
        'these retrospective exploratory selected-comparison intervals are not selection-adjusted. '
        '**The earlier 17.6% Master / 2.1% broad findings remain the preserved familiar-only comparison.**'))
    return reports
