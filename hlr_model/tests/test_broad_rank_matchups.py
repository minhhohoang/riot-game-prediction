"""Offline fixtures only. Even accidental real requests are forbidden."""
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import requests

from hlr_model import broad_rank_matchups as b


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, 'request',
                        lambda *a, **kw: pytest.fail('Live HTTP is forbidden in these tests'))


class Clock:
    def __init__(self):
        self.now = 2000000000.
        self.waits = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class Response:
    def __init__(self, task, *, code=200, html=None, headers=None):
        self.status_code = code
        self.content = (html if html is not None else table_html(task)).encode()
        self.url = task['source_url']
        self.headers = headers or {}


class HTTP:
    def __init__(self, plan, handler=None):
        self.tasks = {r['source_url']: r for r in plan['tables'].to_dict('records')}
        self.handler = handler
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        task = self.tasks[url]
        return self.handler(task) if self.handler else Response(task)

    def close(self):
        pass


def base36(n):
    out = ''
    while True:
        n, r = divmod(n, 36)
        out = '0123456789abcdefghijklmnopqrstuvwxyz'[r] + out
        if not n:
            return out


def table_html(task, *, tier=None, patch='16.19', region='all', queue=420, rows=None,
               heading=None, opposing_lane=None, columns=None, route_overrides=None, header_overrides=None):
    """Synthetic Qwik reference container following the preserved named schema."""
    lane = b.source.ROLE_SLUGS[task['lane']]
    tier = tier or task['tier'].lower()
    route = dict(path=f"/lol/{task['champion_slug']}/build/", page='build', cid=int(task['champion_id']),
                 champion=task['champion_slug'], patch=patch, tier=tier, mode='ranked', status=200,
                 region=region, lane=lane)
    enemy = {k: [] for k in b.source.ROLE_SLUGS.values()}
    enemy[opposing_lane or lane] = rows if rows is not None else [[103, 52.41, 1.2, .36, 3.5, 685]]
    dataset = dict(header=dict(cid=int(task['champion_id']), lane=lane, patch=patch, queue=queue,
                               wr=53.22, avgWr=50.55, avgWrDelta=2.67),
                   enemy_h=columns or ['id', 'wr', 'd1', 'd2', 'pr', 'n'], enemy=enemy,
                   response={'valid': True})
    route.update(route_overrides or {})
    dataset['header'].update(header_overrides or {})
    objects = []

    def encode(value):
        if isinstance(value, dict):
            value = {k: encode(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = [encode(v) for v in value]
        objects.append(value)
        return base36(len(objects) - 1)

    encode(route)
    encode(dataset)
    rank_label = tier.replace('_plus', '+').upper()
    region_label = 'GLOBAL' if region == 'all' else 'NA'
    name = heading or task['champion'] + ' ' + lane
    return f'''<html><h1>{name}</h1>
    <button aria-pressed="true" data-filter="rank">{rank_label}</button>
    <button aria-pressed="true" data-filter="patch">Patch {patch}</button>
    <button aria-pressed="true" data-filter="region">{region_label}</button>
    <button aria-pressed="true" data-filter="queue">{'Ranked Solo/Duo' if queue==420 else 'Ranked Flex'}</button>
    <p>Counter Win Rate Pick Rate Games: the champion wins against the matchup</p>
    <p>{b.SOURCE_DEFINITIONS['d1']}</p><p>{b.SOURCE_DEFINITIONS['d2']}</p>
    <p>Average {rank_label} Win Rate: 49.5 %</p>
    <script type="qwik/json">{json.dumps({'objs':objects})}</script></html>'''


def make_source(root):
    cfg = replace(b.Config(), players_per_tier=1, source_export_fingerprint='fixture')
    path = root / 'hlr_model/data/runs/broad_na_16_19_seed_161926/processed' / cfg.source_snapshot
    path.mkdir(parents=True)
    cohorts, histories, targets = [], [], []
    complete = dict(target_discovery_complete=True, history_discovery_complete=True,
                    history_incomplete=False, missing_required_payloads=0)

    def record(uid, mid, cid, lane, start, end, opponent=103, missing=False):
        return dict(puuid=uid, match_id=mid, champion_id=cid, champion=b.source.CHAMPIONS[cid]['riot_name'],
                    lane=lane, timestamp=start, end_timestamp=end, opponent_champion_id=opponent,
                    opponent_champion=b.source.CHAMPIONS[opponent]['riot_name'] if opponent else None,
                    matchup_missing=missing, patch='16.19', queue=420, win=1)

    for tier in b.TIERS:
        uid = 'p-' + tier
        cohorts.append(dict(puuid=uid, seed_tier=tier, rank=tier, **complete))
        histories.extend([record(uid, 'h1', 1, 'mid', 50, 100),
                          record(uid, 'h2', 3, 'top', 60, 110),
                          record(uid, 'h3', 103, 'mid', 70, 120)])
        targets.extend([record(uid, 't1', 105, 'mid', 200, 250),
                        record(uid, 'overlap', 99, 'mid', 240, 260),
                        record(uid, 't2', 105, 'mid', 300, 350),
                        record(uid, 'unknown', 105, None, 400, 450, opponent=None, missing=True)])
    cohort = pd.DataFrame(cohorts)
    target = pd.DataFrame(targets)
    model = target.merge(cohort[['puuid', 'seed_tier']], on='puuid', validate='many_to_one')
    frames = dict(player_cohort=cohort, historical_matches=pd.DataFrame(histories),
                  target_matches=target, longitudinal_modeling=model,
                  player_coverage=cohort[['puuid', *complete]])
    for name, frame in frames.items():
        frame.to_parquet(path / (name + '.parquet'), index=False)
    metadata = dict(cohort_id='broad_na_16_19', export_fingerprint='fixture', players_with_incomplete_collection=0,
                    output_sha256={p.name: b.sha(p.read_bytes()) for p in path.glob('*.parquet')})
    (path / 'collection_metadata.json').write_text(json.dumps(metadata))
    bundle = b.load_snapshot(root, cfg)
    return cfg, bundle, b.build_plan(bundle, cfg)


@pytest.fixture
def prepared(tmp_path):
    return make_source(tmp_path)


def collector(tmp_path, prepared, *, http=None, clock=None, config=None):
    cfg, bundle, plan = prepared
    clock = clock or Clock()
    return b.Collector(tmp_path, bundle, plan, config or cfg, session=http or HTTP(plan),
                       clock=clock.time, sleep=clock.sleep, status=lambda x: None)


@pytest.mark.parametrize('tier', b.TIERS)
def test_exact_tier_content_and_returned_data(tier, prepared):
    task = next(t for t in prepared[2]['tables'].to_dict('records') if t['tier'] == tier)
    value = b.parse_table(table_html(task), task)
    assert value['rows'][103]['wr'] == 52.41
    assert value['header']['wr'] == 53.22 and value['tier_average'] == 49.5
    assert value['opponent_lane'] == task['lane']


@pytest.mark.parametrize('kwargs', [dict(tier='diamond_plus'), dict(tier='emerald_plus'),
    dict(tier='master_plus'), dict(tier='silver'), dict(patch='16.20'), dict(region='na'),
    dict(queue=440), dict(heading='WrongChampion middle'), dict(columns=['id','wr','unknown','d2','pr','n'])])
def test_wrong_tier_plus_patch_region_queue_champion_or_schema_rejected(kwargs, prepared):
    task = prepared[2]['tables'].iloc[0].to_dict()
    with pytest.raises(b.source.EvidenceError):
        b.parse_table(table_html(task, **kwargs), task)


def test_selected_filters_and_returned_route_must_both_match(prepared):
    task = prepared[2]['tables'].iloc[0].to_dict()
    html = table_html(task, tier='emerald_plus').replace('>EMERALD+<', '>' + task['tier'] + '<')
    with pytest.raises(b.source.EvidenceError):
        b.parse_table(html, task)
    with pytest.raises(b.source.EvidenceError):
        b.parse_table('<h1>Fizz middle Patch 16.19 Bronze Global Solo/Duo</h1>', task)


@pytest.mark.parametrize('overrides', [dict(tier='emerald_plus'),dict(region='na'),dict(patch='16.20'),dict(lane='top'),dict(mode='flex')])
def test_correct_visible_controls_cannot_override_conflicting_returned_route(overrides, prepared):
    task = prepared[2]['tables'].iloc[0].to_dict()
    with pytest.raises(b.source.EvidenceError):
        b.parse_table(table_html(task, route_overrides=overrides), task)


@pytest.mark.parametrize('overrides', [dict(queue=440),dict(cid=999),dict(patch='16.20'),dict(lane='top')])
def test_correct_selected_filters_and_route_cannot_override_wrong_header(overrides, prepared):
    task = prepared[2]['tables'].iloc[0].to_dict()
    with pytest.raises(b.source.EvidenceError):
        b.parse_table(table_html(task, header_overrides=overrides), task)


def test_conflicting_selected_role_is_not_ignored(prepared):
    task = prepared[2]['tables'].iloc[0].to_dict()
    html = table_html(task) + '<button aria-pressed="true" data-filter="lane">Top</button>'
    with pytest.raises(b.source.EvidenceError):
        b.parse_table(html, task)


def test_correct_opposing_role_only(prepared):
    task = prepared[2]['tables'].iloc[0].to_dict()
    parsed = b.parse_table(table_html(task, opposing_lane='top'), task)
    assert parsed['rows'] == {}  # Never use the top-opponent row for a mid matchup.


def test_exact_candidate_eligibility_and_shared_plan(prepared):
    _, bundle, plan = prepared
    mapping = plan['mapping']
    p = mapping.loc[mapping.puuid.eq('p-BRONZE')]
    assert set(p.loc[p.match_id.eq('t1') & p.request_kind.eq('alternative'), 'candidate_champion_id']) == {1}
    assert set(p.loc[p.match_id.eq('overlap') & p.request_kind.eq('alternative'), 'candidate_champion_id']) == {1}
    assert set(p.loc[p.match_id.eq('t2') & p.request_kind.eq('alternative'), 'candidate_champion_id']) == {1, 99}
    assert not mapping.candidate_champion_id.eq(103).any()  # Opponent excluded.
    assert not mapping.loc[mapping.request_kind.eq('alternative'), 'candidate_champion_id'].eq(3).any()  # Wrong role.
    assert mapping.loc[mapping.request_kind.eq('alternative'), 'latest_same_role_end'].lt(
        mapping.loc[mapping.request_kind.eq('alternative'), 'timestamp']).all()
    assert len(plan['decisions']) == len(bundle['frames']['target_matches'])
    assert len(plan['tables']) < len(mapping) and plan['lookups'].lookup_key.is_unique
    assert plan['tables'].source_url.str.contains('queue=').sum() == 0
    assert not plan['tables'].source_url.str.contains('plus|master').any()


def test_offline_switch_then_one_request_per_shared_table_and_unique_joins(tmp_path, prepared):
    http = HTTP(prepared[2]);run = collector(tmp_path, prepared, http=http)
    run.collect(enabled=False)
    assert not http.calls
    run.collect(enabled=True)
    assert len(http.calls) == len(prepared[2]['tables'])
    lookup, played, alternatives = run.joined()
    assert lookup.valid_rate.all() and played.win.eq(1).all()
    assert len(played) == len(prepared[1]['frames']['longitudinal_modeling'])
    assert played.loc[played.match_id.eq('unknown'), 'benchmark_status'].str.contains('unknown').all()
    assert 'win' not in alternatives and alternatives.candidate_outcome.isna().all()
    assert not alternatives.duplicated(b.KEY + ['candidate_champion_id']).any()
    assert lookup.matchup_wins.isna().all() and lookup.matchup_losses.isna().all()
    assert (lookup.raw_matchup_winrate_probability == lookup.raw_matchup_winrate_pct / 100).all()
    assert (lookup.derived_tier_centered_rate_pct == lookup.raw_matchup_winrate_pct - 49.5 + 50).all()
    out, _, meta = run.export()
    assert meta['collection_complete'] and (out / 'collection_metadata.json').is_file()
    assert meta['row_counts']['played_before'] == meta['row_counts']['played_after']
    assert meta['row_counts']['alternatives_before'] == meta['row_counts']['alternatives_after']
    calls = len(http.calls);run.close()
    resumed = collector(tmp_path, prepared, http=http)
    resumed.collect(enabled=True)
    assert len(http.calls) == calls and resumed.progress().status.eq('validated').all()
    second, _, _ = resumed.export()
    assert second != out and (out / 'collection_metadata.json').is_file()


def test_missing_sparse_and_failed_are_distinct_without_dropping_rows(tmp_path, prepared):
    http = HTTP(prepared[2], lambda task: Response(task, html=table_html(task, rows=[])))
    run = collector(tmp_path, prepared, http=http);run.collect(enabled=True)
    lookup, played, alternatives = run.joined()
    assert lookup.status.eq('not_published_in_table').all()
    assert lookup.matchup_games.isna().all() and lookup.raw_matchup_winrate_pct.isna().all()
    assert len(played) == 24 and len(alternatives) == 24
    assert run.export()[2]['collection_complete']  # Successfully checked, explicitly unavailable.
    tid = next(iter(run.tasks));task = run.tasks[tid]
    file = tmp_path / 'sparse.html';file.write_text(table_html(task, rows=[[103, 80., 1., 2., 3., 5]]))
    assert run.import_page(tid, file, source_url=task['source_url'], retrieved_at='2026-10-27T12:00:00+00:00')
    found = run.lookup().loc[lambda f:f.table_id.eq(tid)]
    assert found['sparse'].all() and found.valid_rate.all() and found.matchup_games.eq(5).all()
    assert found.raw_matchup_winrate_pct.eq(80).all()  # Not imputed or discarded.


def test_failure_is_not_no_data_or_complete_and_retry_preserves_old_evidence(tmp_path, prepared):
    http = HTTP(prepared[2], lambda task: Response(task, code=404, html='page missing'))
    run = collector(tmp_path, prepared, http=http);run.collect(enabled=True)
    assert len(http.calls) == run.config.max_consecutive_failed_tables
    lookup, played, _ = run.joined()
    assert lookup.status.eq('request_failed').any() and not lookup.status.eq('not_published_in_table').any()
    assert len(played) == 24 and not run.export()[2]['collection_complete']
    bodies = {p:p.read_bytes() for p in (run.run_dir / 'raw').glob('*.html')}
    http.handler = None;run.close()
    resumed = collector(tmp_path, prepared, http=http);resumed.collect(enabled=True)
    assert resumed.progress().status.eq('validated').all()
    assert all(p.read_bytes() == content for p, content in bodies.items())


def test_rate_limit_retry_after_is_persisted_honored_and_bounded(tmp_path, prepared):
    clock = Clock();attempt = [0]
    def handler(task):
        attempt[0] += 1
        return Response(task, code=429, html='rate limited', headers={'Retry-After':'7'}) if attempt[0] == 1 else Response(task)
    http = HTTP(prepared[2], handler);run = collector(tmp_path, prepared, http=http, clock=clock)
    run.collect(enabled=True)
    assert sum(clock.waits) >= 8 and len(http.calls) == len(prepared[2]['tables']) + 1
    assert run.progress().status.eq('validated').all()
    assert all(kwargs['allow_redirects'] is False for _, kwargs in http.calls)


def test_persistent_rate_limit_has_capped_attempts_and_no_false_completion(tmp_path, prepared):
    http = HTTP(prepared[2], lambda task:Response(task, code=429, html='limited', headers={'Retry-After':'0'}))
    run = collector(tmp_path, prepared, http=http);run.collect(enabled=True)
    assert len(http.calls) == run.config.max_attempts
    assert run.progress().status.eq('request_failed').sum() == 1
    assert not run.export()[2]['collection_complete']


def test_champion_and_exact_tier_values_join_to_the_correct_played_and_alternative_rows(tmp_path, prepared):
    def handler(task):
        rate = 40 + b.TIERS.index(task['tier']) + task['champion_id'] % 10
        return Response(task, html=table_html(task, rows=[[103, rate, 1., 2., 3., 685]]))
    run = collector(tmp_path, prepared, http=HTTP(prepared[2], handler));run.collect(enabled=True)
    _, played, alternatives = run.joined()
    known = played.loc[played.lookup_key.notna()]
    assert known.benchmark_champion_id.eq(known.champion_id).all()
    assert known.benchmark_tier.eq(known.seed_tier).all()
    assert alternatives.benchmark_champion_id.eq(alternatives.candidate_champion_id).all()
    assert alternatives.benchmark_tier.eq(alternatives.tier).all()
    for frame, cid, tier in ((known,'champion_id','seed_tier'),(alternatives,'candidate_champion_id','tier')):
        expected = 40 + frame[tier].map(dict(zip(b.TIERS,range(6)))) + frame[cid] % 10
        assert frame.benchmark_raw_matchup_winrate_pct.eq(expected).all()


def test_published_zero_game_row_is_retained_without_usable_probability(tmp_path, prepared):
    http = HTTP(prepared[2], lambda task:Response(task, html=table_html(task, rows=[[103,0.,0.,0.,0.,0]])))
    run = collector(tmp_path, prepared, http=http);run.collect(enabled=True)
    lookup = run.lookup()
    assert lookup.status.eq('reported_zero_games').all() and lookup.matchup_games.eq(0).all()
    assert lookup.raw_matchup_winrate_pct.eq(0).all()  # Literal source value, not an imputation.
    assert lookup.raw_matchup_winrate_probability.isna().all() and not lookup.valid_rate.any()
    assert len(run.joined()[1]) == 24


def test_one_complete_table_serves_different_opponents_without_extra_request(tmp_path, prepared):
    cfg, old, _ = prepared
    frames = {k:v.copy() for k,v in old['frames'].items()}
    for name in ('target_matches','longitudinal_modeling'):
        extra = frames[name].loc[frames[name].puuid.eq('p-BRONZE') & frames[name].match_id.eq('t1')].copy()
        extra['match_id'] = 'another-opponent';extra['opponent_champion_id'] = 101;extra['opponent_champion'] = 'Xerath'
        if 'target_row_id' in extra:
            extra['target_row_id'] = 'another-target-id'
        frames[name] = pd.concat([frames[name],extra],ignore_index=True)
    bundle = {**old, 'frames':frames};plan = b.build_plan(bundle,cfg)
    http = HTTP(plan, lambda task:Response(task,html=table_html(task,rows=[[103,52.,1.,2.,3.,500],[101,61.,2.,3.,4.,200]])))
    run = collector(tmp_path,(cfg,bundle,plan),http=http);run.collect(enabled=True)
    table = next(t for t in run.tasks.values() if t['tier']=='BRONZE' and t['champion_id']==105)
    assert sum(url == table['source_url'] for url,_ in http.calls) == 1
    rows = run.lookup().loc[lambda f:f.table_id.eq(table['table_id'])]
    assert rows.set_index('opponent_champion_id').raw_matchup_winrate_pct.to_dict() == {103:52.,101:61.}


def test_long_retry_after_and_blocking_pause_without_rapid_requests(tmp_path, prepared):
    clock = Clock();http = HTTP(prepared[2], lambda t:Response(t, code=429, html='limited', headers={'Retry-After':'900'}))
    run = collector(tmp_path, prepared, http=http, clock=clock);run.collect(enabled=True)
    assert len(http.calls) == 1 and run._get('network')['retry_not_before'] >= clock.time() + 900
    run.close();resumed = collector(tmp_path, prepared, http=http, clock=clock)
    resumed.collect(enabled=True)
    assert len(http.calls) == 1 and not resumed.export()[2]['collection_complete']


@pytest.mark.parametrize('code,html,status', [(403,'denied','blocked'),(200,'<h1>Verify you are human</h1>','blocked'),
    (200,'<h1>Historical data is no longer available</h1>','unavailable_patch'),(200,'<h1>unreviewed</h1>','validation_failed')])
def test_block_or_schema_error_stops_and_does_not_claim_unavailable_matchup(tmp_path, prepared, code, html, status):
    http = HTTP(prepared[2], lambda t:Response(t, code=code, html=html))
    run = collector(tmp_path, prepared, http=http);run.collect(enabled=True)
    assert len(http.calls) == 1 and run.progress().status.eq(status).sum() == 1
    assert not run.export()[2]['collection_complete']
    assert not run.lookup().status.eq('not_published_in_table').any()


def test_interrupt_restarts_at_pending_table_without_redownloading_success(tmp_path, prepared):
    count = [0]
    def handler(task):
        count[0] += 1
        if count[0] == 2:
            raise KeyboardInterrupt
        return Response(task)
    http = HTTP(prepared[2], handler);run = collector(tmp_path, prepared, http=http)
    with pytest.raises(KeyboardInterrupt):
        run.collect(enabled=True)
    first_url = http.calls[0][0]
    assert run.progress().status.eq('validated').sum() == 1
    run.close();http.handler = None
    resumed = collector(tmp_path, prepared, http=http);resumed.collect(enabled=True)
    assert sum(u == first_url for u, _ in http.calls) == 1
    assert resumed.progress().status.eq('validated').all()


def test_atomic_receipt_recovers_crash_before_index_or_database_commit(tmp_path, prepared, monkeypatch):
    run = collector(tmp_path, prepared);tid = sorted(run.tasks)[0]
    original = b.atomic_json
    def interrupt(path, value):
        if Path(path) == run._index(tid):
            raise KeyboardInterrupt
        return original(path, value)
    monkeypatch.setattr(b, 'atomic_json', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run.collect(enabled=True)
    assert not run._index(tid).exists()
    run.close();monkeypatch.setattr(b, 'atomic_json', original)
    http = HTTP(prepared[2]);resumed = collector(tmp_path, prepared, http=http)
    assert tid in resumed.parsed and resumed._index(tid).exists()
    resumed.collect(enabled=True)
    assert all(u != resumed.tasks[tid]['source_url'] for u, _ in http.calls)
    assert resumed.progress().status.eq('validated').all()


def test_cache_corruption_and_changed_identity_cannot_silently_resume(tmp_path, prepared):
    run = collector(tmp_path, prepared);run.collect(enabled=True)
    tid = next(iter(run.tasks));receipt = json.loads(run._index(tid).read_text())
    body = run.run_dir / receipt['body_file'];body.write_bytes(b'truncated')
    run.close();http = HTTP(prepared[2]);resumed = collector(tmp_path, prepared, http=http)
    assert resumed.progress().set_index('table_id').loc[tid, 'status'] == 'cache_invalid'
    resumed.collect(enabled=True)
    assert not http.calls and not resumed.export()[2]['collection_complete']
    resumed.close()
    with pytest.raises(ValueError, match='Pinned source/configuration/plan changed'):
        collector(tmp_path, prepared, config=replace(prepared[0], sparse_below_games=20))
    changed = dict(prepared[1], identity={**prepared[1]['identity'], 'snapshot':'different'})
    with pytest.raises(ValueError, match='Pinned source/configuration/plan changed'):
        collector(tmp_path, (prepared[0], changed, prepared[2]))


def test_browser_import_and_compatible_cache_reuse_are_offline_and_read_only(tmp_path, prepared):
    run = collector(tmp_path, prepared);tid = next(iter(run.tasks));task = run.tasks[tid]
    html = tmp_path / 'browser.html';html.write_text(table_html(task))
    assert run.import_page(tid, html, source_url=task['source_url'], retrieved_at='2026-10-27T12:00:00-07:00')
    original_cache = run.run_dir
    before = {p:p.read_bytes() for p in original_cache.rglob('*') if p.is_file() and p.suffix in ('.json','.html')}
    cfg = replace(prepared[0], run_name='another_exact_tier_run')
    other = collector(tmp_path, prepared, config=cfg)
    assert other.reuse_compatible([original_cache]) == 1
    assert tid in other.parsed and all(p.read_bytes() == v for p, v in before.items())
    with pytest.raises(ValueError):
        other.import_page(tid, html, source_url=task['source_url'].replace('tier=bronze','tier=master_plus'), retrieved_at='2026-10-27T12:00:00-07:00')


def test_pinned_source_hash_changes_fail_without_touching_checkpoint(tmp_path, prepared):
    cfg, bundle, _ = prepared
    file = bundle['path'] / 'target_matches.parquet'
    file.write_bytes(file.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Source hash mismatch'):
        b.load_snapshot(tmp_path, cfg)


def test_preserved_real_master_fixture_is_not_reused_as_exact_tier(prepared):
    root = Path(__file__).resolve().parents[2]
    diagnostics = root / 'hlr_model/data/lolalytics/diagnostics/20261020_url_repair'
    for file in (diagnostics / 'responses').glob('*.json'):
        receipt = json.loads(file.read_text())
        if receipt['http_status'] == 200 and '/fizz/build/' in receipt['requested_url'] and 'lane=middle' in receipt['requested_url'] and 'queue=' not in receipt['requested_url']:
            html = (diagnostics / receipt['body_file']).read_text()
            route, dataset, _ = b.source.qwik_dataset(html)
            assert route['tier'] == 'master_plus' and dataset['enemy_h'] == ['id','wr','d1','d2','pr','n']
            task = next(t for t in prepared[2]['tables'].to_dict('records') if t['champion_id'] == 105)
            with pytest.raises(b.source.EvidenceError):
                b.parse_table(html, task)
            return
    pytest.fail('Expected preserved project diagnostic fixture is absent.')
