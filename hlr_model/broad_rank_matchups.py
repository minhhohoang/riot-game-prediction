"""Exact-tier, retrospective LoLalytics tables for a pinned broad-cohort export.

Import and planning are offline. HTTP is possible only via collect(enabled=True).
The Master collector is imported only for its pure registry/plan/Qwik helpers;
its configuration, run preparation, cache writer and collector are never used.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlparse

import pandas as pd
import requests

from hlr_model import lolalytics_matchup_scraper as source
from hlr_model.broad_rank_pipeline import writer_lock

TIERS = ('BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND')
KEY = ['puuid', 'match_id']
SCHEMA = 'broad-exact-tier-tables-v1'
SOURCE_FILES = ('player_cohort', 'historical_matches', 'target_matches',
                'longitudinal_modeling', 'player_coverage')
RATE_FIELDS = ['raw_matchup_winrate_pct', 'raw_matchup_winrate_probability', 'matchup_games',
               'matchup_wins', 'matchup_losses', 'champion_overall_winrate_pct',
               'tier_average_winrate_pct', 'source_header_avgWr_pct', 'source_header_avgWrDelta_pp',
               'source_delta1_pp', 'website_delta2_pp', 'source_pick_rate_pct',
               'derived_tier_centered_rate_pct']
SOURCE_DEFINITIONS = {
    'd1': 'Delta 1 = Counter Matchup Win Rate - (100 - Counter Champions Overall Average Win Rate)',
    'd2': 'Delta 2 = Counter Matchup Win Rate - Normalised Champions Win Rate in Matchup',
}
LIMITS = [
    'Player tier was checked at collection time, not verified historically during patch 16.19.',
    'Full-patch benchmarks are retrospective aggregates, not necessarily available before each target game.',
    'Prior same-role play establishes candidate eligibility, not draft availability or a causal switching effect.',
    'Roles follow postgame Riot teamPosition inference; no rank, region, patch or 50% fallback is used.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def stamp(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Config:
    source_snapshot: str = '20261027T153245042167Z'
    source_export_fingerprint: str = '352a75c162ac4f3ac573385b4c1b5cc9fe90e23b6c7d3af7b6e78ca1da4c9b44'
    patch: str = '16.19'
    region: str = 'global'
    queue: str = 'ranked'
    queue_id: int = 420
    tiers: tuple = TIERS
    players_per_tier: int = 50
    run_name: str = 'broad_na_16_19_seed_161926_exact_tiers_v1'
    delay_seconds: float = 3.0
    max_attempts: int = 3
    backoff_seconds: float = 5.0
    backoff_cap_seconds: float = 60.0
    max_wait_seconds: float = 300.0
    max_consecutive_failed_tables: int = 3
    timeout_connect: float = 10.0
    timeout_read: float = 45.0
    # A reporting flag only: never excludes a benchmark or a player-game.
    sparse_below_games: int = 100

    def __post_init__(self):
        require((self.patch, self.region, self.queue, self.queue_id) == ('16.19', 'global', 'ranked', 420),
                'This study requires patch 16.19, Global, Ranked Solo/Duo (420).')
        require(tuple(self.tiers) == TIERS, 'Use all six exact tiers, Bronze through Diamond; no plus brackets.')
        require(self.players_per_tier > 0 and self.sparse_below_games >= 0, 'Invalid cohort/sparse configuration.')
        require(bool(re.fullmatch(r'\d{8}T\d{12}Z', self.source_snapshot)), 'Pin an explicit export snapshot.')
        require(bool(re.fullmatch(r'[a-zA-Z0-9_-]+', self.run_name)), 'Use a simple separate run folder name.')
        require(self.delay_seconds >= 1 and self.max_attempts >= 1 and self.max_consecutive_failed_tables >= 1,
                'Use conservative sequential pacing and bounded retries.')
        require(min(self.timeout_connect, self.timeout_read, self.max_wait_seconds) > 0
                and min(self.backoff_seconds, self.backoff_cap_seconds) >= 0, 'Invalid timeout/backoff.')


def load_snapshot(project_root, config):
    """Read only this explicit, completed snapshot. Never search for 'latest'."""
    root = Path(project_root).resolve()
    path = root / 'hlr_model/data/runs/broad_na_16_19_seed_161926/processed' / config.source_snapshot
    metadata_path = path / 'collection_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    require(metadata.get('cohort_id') == 'broad_na_16_19', 'Not a broad-cohort export.')
    require(metadata.get('export_fingerprint') == config.source_export_fingerprint, 'Pinned export fingerprint differs.')
    frames, hashes = {}, {}
    for name in SOURCE_FILES:
        file = path / (name + '.parquet')
        hashes[file.name] = sha(file.read_bytes())
        require(metadata.get('output_sha256', {}).get(file.name) == hashes[file.name], f'Source hash mismatch: {file.name}')
        frames[name] = pd.read_parquet(file)
    cohort = frames['player_cohort']
    require(not cohort.puuid.duplicated().any(), 'Duplicate cohort player.')
    require(cohort.groupby('seed_tier').size().to_dict() == {t: config.players_per_tier for t in TIERS},
            'Completed cohort must have the configured quota in every exact tier.')
    require(cohort['rank'].eq(cohort.seed_tier).all(), 'Rank and frozen tier disagree.')
    coverage = frames['player_coverage']
    require(not coverage.puuid.duplicated().any() and set(coverage.puuid) == set(cohort.puuid), 'Coverage/cohort mismatch.')
    for frame in (cohort, coverage):
        require(frame.target_discovery_complete.all() and frame.history_discovery_complete.all()
                and not frame.history_incomplete.any() and frame.missing_required_payloads.eq(0).all(),
                'Source collection is incomplete; finish it before benchmark collection.')
    require(metadata.get('players_with_incomplete_collection') == 0, 'Source metadata reports incomplete collection.')
    for name in ('historical_matches', 'target_matches', 'longitudinal_modeling'):
        frame = frames[name]
        required = set(KEY + ['timestamp', 'end_timestamp', 'champion_id', 'champion', 'lane',
                              'opponent_champion_id', 'opponent_champion', 'matchup_missing', 'patch', 'queue', 'win'])
        require(required <= set(frame), f'{name}: missing record columns.')
        require(not frame.duplicated(KEY).any(), f'{name}: duplicate player-game.')
        require(set(frame.puuid) <= set(cohort.puuid), f'{name}: player outside the selected cohort.')
        require(frame.queue.eq(420).all() and frame.end_timestamp.ge(frame.timestamp).all(), f'{name}: queue/time mismatch.')
    targets, history, model = [frames[n] for n in ('target_matches', 'historical_matches', 'longitudinal_modeling')]
    require(set(targets.puuid) == set(cohort.puuid) and targets.patch.eq(config.patch).all(), 'Missing players/wrong target patch.')
    pd.testing.assert_frame_equal(model.set_index(KEY).sort_index()[targets.drop(columns=KEY).columns],
                                  targets.set_index(KEY).sort_index(), check_dtype=False)
    require(not set(map(tuple, history[KEY].values)) & set(map(tuple, targets[KEY].values)), 'History/target overlap.')
    anchors = targets.groupby('puuid').timestamp.min()
    require(history.end_timestamp.lt(history.puuid.map(anchors)).all(), 'History overlaps initial target kickoff.')
    ranks = cohort.set_index('puuid').seed_tier
    require(model.seed_tier.eq(model.puuid.map(ranks)).all(), 'Model tier disagrees with cohort.')
    frames['longitudinal_modeling'] = model.assign(source_target_row=range(len(model)),
        target_row_id=[sha(canonical(list(k)).encode()) for k in model[KEY].itertuples(index=False, name=None)])
    return {'path': path, 'frames': frames, 'identity': {
        'snapshot': config.source_snapshot, 'export_fingerprint': config.source_export_fingerprint,
        'files_sha256': hashes, 'metadata_sha256': sha(metadata_path.read_bytes())}}


def build_plan(bundle, config):
    """Reuse the project's exact strictly prior, same-role candidate algorithm."""
    f = bundle['frames']
    model, history, targets = [f[n] for n in ('longitudinal_modeling', 'historical_matches', 'target_matches')]
    tier_by_player = f['player_cohort'].set_index('puuid').seed_tier
    plans, mappings, decisions = [], [], []
    for tier in TIERS:
        players = set(tier_by_player[tier_by_player.eq(tier)].index)
        t = targets.loc[targets.puuid.isin(players)]
        cfg = SimpleNamespace(patch=config.patch, rank=tier.lower(), region='global', queue='ranked',
                              candidate_roles=tuple(source.ROLE_SLUGS), include_current_pick_in_candidates=False)
        known = ~t.matchup_missing & t.lane.isin(source.ROLE_SLUGS) & t.opponent_champion_id.notna() & t.opponent_champion.notna()
        if not (known & t.champion_id.ne(t.opponent_champion_id)).any():
            decisions.append(t[KEY + ['lane']].assign(tier=tier, known_lane_opponent=known,
                unexpected_observed_mirror=known & t.champion_id.eq(t.opponent_champion_id)))
            continue
        plan, mapping, decision = source._build_base_plan(model, history.loc[history.puuid.isin(players)], t, cfg)
        prefix = f'{config.patch}:{tier.lower()}:global:ranked:'
        plan['lookup_key'] = prefix + plan.pop('key_id')
        mapping['lookup_key'] = prefix + mapping.pop('key_id')
        plan['tier'] = mapping['tier'] = decision['tier'] = tier
        plan['table_id'] = [prefix + f'{r.lane}:{int(r.champion_id)}' for r in plan.itertuples()]
        plan['opponent_lane'] = plan.lane
        mapping = mapping.merge(plan[['lookup_key', 'table_id']], on='lookup_key', validate='many_to_one')
        plans.append(plan.drop(columns=['page_url']))  # No directional fallback.
        mappings.append(mapping)
        decisions.append(decision)
    require(bool(plans), 'No known, non-mirror target lookups; no collection can be planned.')
    plan = pd.concat(plans, ignore_index=True).sort_values('lookup_key').reset_index(drop=True)
    mapping = pd.concat(mappings, ignore_index=True)
    decision = pd.concat(decisions, ignore_index=True)
    require(not plan.lookup_key.duplicated().any(), 'Duplicate benchmark lookup key.')
    require(not mapping.duplicated(KEY + ['candidate_champion_id']).any(), 'Duplicate candidate within a player-game.')
    require(len(decision) == len(targets) and not decision.duplicated(KEY).any(), 'Decision row loss/duplication.')
    mapping = mapping.merge(model[KEY + ['target_row_id']], on=KEY, validate='many_to_one')
    task_columns = ['table_id', 'tier', 'lane', 'champion_id', 'champion', 'champion_slug', 'batch_url']
    tables = plan[task_columns].drop_duplicates('table_id').rename(columns={'batch_url': 'source_url'})
    tables['patch'] = config.patch
    tables['region'] = config.region
    tables['queue'] = config.queue
    tables['queue_id'] = config.queue_id
    return {'lookups': plan, 'mapping': mapping, 'decisions': decision, 'tables': tables.reset_index(drop=True)}


def canonical_url(url):
    parsed = urlparse(url)
    require(parsed.scheme == 'https' and parsed.netloc == 'lolalytics.com' and not parsed.fragment,
            'Only ordinary HTTPS lolalytics.com page URLs are allowed.')
    return parsed._replace(query=urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))).geturl()


def verify_exact_filters(soup, task):
    """Require selected content, then independently cross-check returned Qwik data."""
    contexts = source.evidence_contexts(soup)
    predicates = {
        'rank': lambda x: x.upper() == task['tier'],
        'patch': lambda x: bool(re.fullmatch(r'(?:Patch\s+)?' + re.escape(task['patch']), x, re.I)),
        'region': lambda x: x.upper() in {'GLOBAL', 'WORLD', 'WORLDWIDE'},
        'queue': lambda x: bool(re.fullmatch(r'Ranked\s+Solo\s*/\s*Duo', x, re.I)),
    }
    evidence = {}
    for name, predicate in predicates.items():
        label = r'rank|tier' if name == 'rank' else name
        selected = [c for c in contexts if c['selected'] and re.search(label, c['label'], re.I)]
        if not selected or not all(predicate(c['text'].strip()) for c in selected):
            raise source.EvidenceError('validation_failed', f'Returned selected {name} does not confirm the exact requested filter.')
        evidence[name] = selected
    roles = [c for c in contexts if c['selected'] and re.search(r'lane|role', c['label'], re.I)
             and not re.search(r'vs|opponent|opposing', c['label'], re.I)]
    if any(c['text'].strip().lower() not in {n.lower() for n in source.ROLE_NAMES[task['lane']]} for c in roles):
        raise source.EvidenceError('validation_failed', 'Selected focal role conflicts with the requested dataset.')
    if roles:
        evidence['lane'] = roles
    return evidence


def parse_table(html, task):
    """Read the complete named enemy table; never infer unnamed numeric columns."""
    soup, visible = source.parse_document(html)
    evidence = verify_exact_filters(soup, task)
    try:
        route, dataset, index = source.qwik_dataset(html)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise source.EvidenceError('validation_failed', 'Named Qwik dataset is absent or has an unreviewed schema.') from error
    lane = source.ROLE_SLUGS[task['lane']]
    expected = {'path': f"/lol/{task['champion_slug']}/build/", 'page': 'build',
                'cid': int(task['champion_id']), 'champion': task['champion_slug'],
                'patch': task['patch'], 'tier': task['tier'].lower(), 'mode': 'ranked',
                'status': 200, 'region': 'all', 'lane': lane}
    if any(route.get(k) != v for k, v in expected.items()):
        raise source.EvidenceError('validation_failed', 'Returned Qwik route conflicts with requested exact-tier table.')
    header = dataset.get('header')
    if (not isinstance(header, dict) or any(header.get(k) != v for k, v in {
            'cid': int(task['champion_id']), 'lane': lane, 'patch': task['patch'], 'queue': 420}.items())
            or dataset.get('response', {}).get('valid') is not True):
        raise source.EvidenceError('validation_failed', 'Returned champion/role/patch/queue header or dataset validity differs.')
    heading = source.name_pattern(task['champion_id'])
    if not any(re.search(r'\b' + heading + r'\b', source.text_of(h).replace('’', "'"), re.I)
               and re.search(r'\b' + re.escape(lane) + r'\b', source.text_of(h), re.I) for h in soup.select('h1')):
        raise source.EvidenceError('validation_failed', 'Focal champion and role heading not confirmed.')
    columns = dataset.get('enemy_h')
    if columns != ['id', 'wr', 'd1', 'd2', 'pr', 'n']:
        raise source.EvidenceError('validation_failed', 'Unreviewed matchup columns; no numeric positions were guessed.')
    if not all(text in visible for text in ('Counter Win Rate', 'Pick Rate Games', 'champion wins against the matchup')):
        raise source.EvidenceError('validation_failed', 'Rendered counter-table perspective/column semantics missing.')
    if not all(definition in visible for definition in SOURCE_DEFINITIONS.values()):
        raise source.EvidenceError('validation_failed', 'Source Delta 1/Delta 2 meanings are missing or changed; no adjusted metric was inferred.')
    enemy = dataset.get('enemy')
    if not isinstance(enemy, dict) or set(enemy) != set(source.ROLE_SLUGS.values()) or not isinstance(enemy[lane], list):
        raise source.EvidenceError('validation_failed', 'Opposing-role table is incomplete or unreviewed.')
    rows = {}
    for values in enemy[lane]:
        if not isinstance(values, list) or len(values) != len(columns):
            raise source.EvidenceError('validation_failed', 'Malformed opponent row.')
        row = dict(zip(columns, values))
        if (type(row['id']) is not int or row['id'] <= 0 or row['id'] in rows
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= row['wr'] <= 100 or not 0 <= row['pr'] <= 100
                or row['n'] < 0 or int(row['n']) != row['n']):
            raise source.EvidenceError('validation_failed', 'Invalid/duplicate opponent ID, percentage or sample count.')
        rows[row['id']] = row
    pattern = r'Average\s+' + re.escape(task['tier']) + r'\s+Win\s+Rate\s*:?\s*([\d.]+)\s*%'
    averages = {float(x) for x in re.findall(pattern, visible, re.I)}
    if len(averages) > 1 or any(not 0 <= x <= 100 for x in averages):
        raise source.EvidenceError('validation_failed', 'Conflicting/invalid exact-bracket average win rate.')
    for field in ('wr', 'avgWr', 'avgWrDelta'):
        value = header.get(field)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                  or field != 'avgWrDelta' and not 0 <= value <= 100):
            raise source.EvidenceError('validation_failed', 'Invalid overall champion/source header rate.')
    evidence.update(dataset_header=header, dataset_object_index=index, columns=columns,
                    source_metric_definitions=SOURCE_DEFINITIONS,
                    opposing_lane=lane, route={k: route[k] for k in expected},
                    perspective='focal champion wins against opponent; named enemy table, never synergy or reverse rate')
    return {'rows': rows, 'header': header, 'tier_average': next(iter(averages), None),
            'evidence': canonical(evidence), 'opponent_lane': task['lane']}


class Collector:
    """Single writer, durable table receipts and explicit opt-in sequential HTTP."""
    def __init__(self, project_root, bundle, plan, config=None, *, run_dir=None,
                 session=None, clock=time.time, sleep=time.sleep, status=print):
        self.config = config or Config()
        self.bundle, self.plan = bundle, plan
        self.root = Path(project_root).resolve()
        self.run_dir = Path(run_dir) if run_dir is not None else self.root / 'hlr_model/data/lolalytics' / self.config.run_name
        self.run_dir = self.run_dir.resolve()
        protected = [bundle['path'].resolve(), (self.root / 'hlr_model/data/runs').resolve(),
                     (self.root / 'hlr_model/data/lolalytics/runs').resolve(),
                     (self.root / 'lpm_model/data').resolve()]
        require(not any(self.run_dir == p or self.run_dir.is_relative_to(p) for p in protected), 'Use a separate matchup cache folder.')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.run_dir / '.writer.lock'
        self.clock, self.sleep, self.status = clock, sleep, status
        self.http = session or requests.Session()
        if session is None:
            self.http.headers.update({'User-Agent': 'BroadCohortMatchupResearch/1.0 (sequential cached collection)',
                                      'Accept-Language': 'en-US,en;q=0.9'})
        self.rng = random.Random()
        self.metrics = Counter()
        self.parsed = {}
        self.tasks = {r['table_id']: r for r in plan['tables'].to_dict('records')}
        identity = {'schema': SCHEMA, 'config': asdict(self.config), 'source': bundle['identity'],
                    'plan_sha256': {n: sha(f.to_json(orient='split', index=False).encode()) for n, f in plan.items()},
                    'candidate_definition': 'strict prior end < kickoff; all same-role champions; exclude played/opponent; no threshold'}
        with writer_lock(self.lock_path):
            self.db = sqlite3.connect(self.run_dir / 'checkpoint.sqlite')
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS tables (table_id TEXT PRIMARY KEY,status TEXT NOT NULL,receipt TEXT,error TEXT)')
            self.db.commit()
            saved = self._get('identity')
            if saved is not None and saved != json.loads(canonical(identity)):
                self.db.close()
                raise ValueError('Pinned source/configuration/plan changed. Restore original settings or choose a separate run folder; nothing was reset.')
            if saved is None:
                self._put('identity', identity)
                with self.db:
                    self.db.executemany('INSERT INTO tables VALUES (?,?,NULL,NULL)', [(k, 'pending') for k in self.tasks])
            require(set(r[0] for r in self.db.execute('SELECT table_id FROM tables')) == set(self.tasks), 'Checkpoint table set differs.')
            self._reconcile()

    def _get(self, key, default=None):
        row = self.db.execute('SELECT value FROM kv WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _put(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)', (key, canonical(value)))

    def _update(self, tid, status, receipt=None, error=None):
        with self.db:
            self.db.execute('UPDATE tables SET status=?,receipt=?,error=? WHERE table_id=?',
                            (status, canonical(receipt) if receipt else None, error, tid))

    def _index(self, tid):
        return self.run_dir / 'receipts' / (sha(tid.encode()) + '.json')

    def _save_receipt(self, task, body, **receipt):
        receipt.update(table_id=task['table_id'], requested_url=task['source_url'],
                       request_identity={k: task[k] for k in ('patch', 'region', 'queue', 'queue_id', 'tier', 'champion_id', 'lane')})
        if body is not None:
            digest = sha(body)
            file = self.run_dir / 'raw' / (digest + '.html')
            if not file.exists():
                atomic_bytes(file, body)
            require(sha(file.read_bytes()) == digest, 'Content-addressed body differs.')
            receipt.update(body_file=str(file.relative_to(self.run_dir)), body_sha256=digest)
        # Immutable per-attempt receipt survives a torn index/database update.
        archive = self.run_dir / 'receipts' / (sha(task['table_id'].encode()) + '-' + uuid.uuid4().hex + '.json')
        atomic_json(archive, receipt)
        atomic_json(self._index(task['table_id']), receipt)
        self._update(task['table_id'], 'received', receipt)
        return receipt

    def _accept(self, task, receipt):
        tid = task['table_id']
        require(receipt['table_id'] == tid and receipt['request_identity'] == {
            k: task[k] for k in ('patch', 'region', 'queue', 'queue_id', 'tier', 'champion_id', 'lane')}, 'Receipt identity differs.')
        require(canonical_url(receipt['requested_url']) == canonical_url(task['source_url']), 'Receipt URL differs.')
        require(canonical_url(receipt.get('response_url', receipt['requested_url'])) == canonical_url(task['source_url']), 'Redirected/default page refused.')
        if receipt.get('http_status') != 200:
            status = 'blocked' if receipt.get('http_status') in (401, 403) else 'request_failed'
            self._update(tid, status, receipt, receipt.get('error_code') or f"HTTP_{receipt.get('http_status')}")
            return False
        require(datetime.fromisoformat(receipt['retrieved_at']).tzinfo is not None, 'Receipt requires a timezone-aware retrieval timestamp.')
        body = (self.run_dir / receipt['body_file']).resolve()
        require(body.is_relative_to(self.run_dir / 'raw') and body.is_file(), 'Cached response path is invalid/missing.')
        data = body.read_bytes()
        require(sha(data) == receipt['body_sha256'], 'Cached source-body checksum differs.')
        try:
            parsed = parse_table(data.decode('utf-8', errors='replace'), task)
        except source.EvidenceError as error:
            status = error.status if error.status in ('blocked', 'unavailable_patch') else 'validation_failed'
            self._update(tid, status, receipt, str(error))
            return False
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            self._update(tid, 'validation_failed', receipt, 'Unreviewed response schema; inspect the saved body or import a verified page.')
            return False
        self.parsed[tid] = parsed
        self._update(tid, 'validated', receipt)
        return True

    def _reconcile(self):
        self.parsed.clear()
        for tid, task in self.tasks.items():
            index = self._index(tid)
            if not index.exists():
                # Crash after the immutable receipt but before index/DB commit:
                # recover only complete receipts, never a .tmp body or index.
                receipts = list((self.run_dir / 'receipts').glob(sha(tid.encode()) + '-*.json'))
                if receipts:
                    newest = max(receipts, key=lambda p: (p.stat().st_mtime_ns, p.name))
                    try:
                        receipt = json.loads(newest.read_text())
                        self._accept(task, receipt)
                        atomic_json(index, receipt)
                    except (OSError, ValueError, TypeError, KeyError):
                        self._update(tid, 'cache_invalid', error='Uncommitted receipt failed validation; evidence preserved.')
                    continue
                if task['source_url'] is None:
                    self._update(tid, 'unsupported_mapping', error='Focal champion has no reviewed website slug.')
                else:
                    self._update(tid, 'pending')
                continue
            try:
                self._accept(task, json.loads(index.read_text()))
            except (OSError, ValueError, TypeError, KeyError):
                self._update(tid, 'cache_invalid', error='Receipt/body identity or checksum failed; original evidence preserved.')

    def progress(self):
        rows = [dict(table_id=k, status=s, error=e, **{c: self.tasks[k][c] for c in ('tier', 'lane')},
                     origin=json.loads(r).get('origin') if r else None)
                for k, s, r, e in self.db.execute('SELECT table_id,status,receipt,error FROM tables')]
        return pd.DataFrame(rows)

    def show(self):
        progress = self.progress()
        counts = progress.status.value_counts().to_dict()
        pending = progress.status.isin(['pending', 'requesting', 'received']).sum()
        failed = (~progress.status.isin(['validated', 'pending', 'requesting', 'received'])).sum()
        self.status(f'Required tables={len(progress):,}; cached/validated={counts.get("validated", 0):,}; pending={pending:,}; failed={failed:,}; downloaded this session={self.metrics["downloaded"]:,}.')
        self.status(f'Table statuses: {counts}; retry/cooldown: {self._get("network", {})}')
        self.status(f'Requested unique matchups: {self.lookup().status.value_counts().to_dict()}')
        return progress.groupby(['tier', 'lane', 'status']).size().rename('tables').reset_index()

    def _wait_for_slot(self):
        network = self._get('network', {})
        until = max(network.get('next_request_at', 0), network.get('retry_not_before', 0))
        remaining = until - self.clock()
        if remaining > self.config.max_wait_seconds:
            self.status(f'Saved Retry-After cooldown has {remaining:.0f}s remaining. Paused; reopen after it expires.')
            return False
        if remaining > 0:
            self.status(f'Retry/pacing cooldown: {remaining:.1f}s; interrupt safely to pause.')
        while until > self.clock():
            self.sleep(min(30, until - self.clock()))
        self._put('network', {'next_request_at': self.clock() + self.config.delay_seconds, 'retry_not_before': 0})
        return True

    def _fetch(self, task):
        for attempt in range(1, self.config.max_attempts + 1):
            if not self._wait_for_slot():
                return False
            self._update(task['table_id'], 'requesting')
            body, code, retry_after, error = None, None, None, None
            response_url = task['source_url']
            try:
                response = self.http.get(task['source_url'], allow_redirects=False,
                                         timeout=(self.config.timeout_connect, self.config.timeout_read))
                body, code = response.content, response.status_code
                response_url = response.url
                retry_after = response.headers.get('Retry-After')
            except requests.RequestException as exception:
                error = type(exception).__name__  # Exception text can contain unrelated sensitive data.
            delay = min(self.config.backoff_cap_seconds, self.config.backoff_seconds * 2 ** (attempt - 1)) + self.rng.uniform(0, 1)
            retry_seconds = 0
            if retry_after is not None:
                try:
                    retry_seconds = float(retry_after)
                except (TypeError, ValueError):
                    try:
                        retry_seconds = parsedate_to_datetime(retry_after).timestamp() - self.clock()
                    except (TypeError, ValueError, OverflowError):
                        retry_seconds = 60
                if not math.isfinite(retry_seconds):
                    retry_seconds = 60
            retryable = code is None or code in (408, 429) or code is not None and 500 <= code <= 599
            network = {'next_request_at': self.clock() + self.config.delay_seconds,
                       'retry_not_before': self.clock() + max(0, retry_seconds) + 1 if retry_after else 0}
            if retryable:
                network['retry_not_before'] = max(network['retry_not_before'], self.clock() + delay)
            self._put('network', network)
            receipt = self._save_receipt(task, body, http_status=code, retrieved_at=stamp(self.clock()),
                response_url=response_url, origin='download', error_code=error,
                attempt=attempt, retry_not_before=network['retry_not_before'])
            self.metrics['http_attempts'] += 1
            if code == 200:
                self.metrics['downloaded'] += 1
                try:
                    return self._accept(task, receipt)
                except (ValueError, TypeError, KeyError):
                    self._update(task['table_id'], 'validation_failed', receipt, 'Returned response URL/receipt identity failed validation.')
                    return False
            self._accept(task, receipt)
            if not retryable or attempt == self.config.max_attempts:
                return False
            self.status(f'Table {task["table_id"]}: HTTP {code} / {error or "retryable"}; bounded retry {attempt + 1}/{self.config.max_attempts}.')
        return False

    def collect(self, *, enabled=False):
        if not enabled:
            self.status('RUN_COLLECTION=False: offline planning/cache checks only; no HTTP requests.')
            return
        with writer_lock(self.lock_path):
            self._reconcile()
            failed = 0
            for tid, state, receipt, _ in list(self.db.execute('SELECT table_id,status,receipt,error FROM tables ORDER BY table_id')):
                if state == 'validated':
                    self.metrics['cache_hits'] += 1
                    continue
                task = self.tasks[tid]
                if state in ('unsupported_mapping', 'cache_invalid', 'validation_failed', 'unavailable_patch') or state == 'blocked' and receipt and json.loads(receipt).get('http_status') == 200:
                    self.status(f'Paused at {tid}: {state}. Inspect saved evidence; use import_page() for a normally accessible verified HTML page. No source/filter fallback.')
                    break
                if self._fetch(task):
                    failed = 0
                else:
                    failed += 1
                    current = self.db.execute('SELECT status FROM tables WHERE table_id=?', (tid,)).fetchone()[0]
                    if current != 'request_failed' or failed >= self.config.max_consecutive_failed_tables or self._get('network', {}).get('retry_not_before', 0) > self.clock():
                        self.status(f'Paused after {current}; required work remains. Saved evidence is available for inspection/import.')
                        break
                if (self.metrics['downloaded'] + failed) % 20 == 0:
                    self.show()
            self.show()

    def import_page(self, table_id, html_path, *, source_url, retrieved_at):
        """Import an ordinary browser Save Page Source HTML; no HTTP or bypass.

        The HTML must contain the named Qwik table AND selected exact filters.
        A URL, screenshot, or manually asserted filter manifest alone is insufficient.
        """
        task = self.tasks[table_id]
        require(canonical_url(source_url) == canonical_url(task['source_url']), 'Import URL differs from this requested table.')
        require(datetime.fromisoformat(retrieved_at).tzinfo is not None, 'Include timezone in retrieved_at.')
        body = Path(html_path).read_bytes()
        with writer_lock(self.lock_path):
            receipt = self._save_receipt(task, body, http_status=200, retrieved_at=retrieved_at,
                                         response_url=source_url, origin='browser_import')
            return self._accept(task, receipt)

    def reuse_compatible(self, roots):
        """Read legacy receipts without editing them; accept only exact URL+content identity."""
        with writer_lock(self.lock_path):
            wanted = {canonical_url(t['source_url']): t for tid, t in self.tasks.items()
                      if tid not in self.parsed and t['source_url'] is not None}
            used = 0
            for root in map(Path, roots):
                if not root.is_dir():
                    continue
                candidates = set(root.rglob('responses/*.json')) | set(root.rglob('receipts/*.json'))
                for file in sorted(candidates):
                    try:
                        receipt = json.loads(file.read_text())
                        url = canonical_url(receipt.get('requested_url', ''))
                        if url not in wanted or receipt.get('http_status') != 200:
                            continue
                        task = wanted[url]
                        base = file.parent.parent
                        body_path = (base / receipt['body_file']).resolve()
                        require(body_path.is_relative_to(base.resolve()), 'External receipt path escapes its cache.')
                        body = body_path.read_bytes()
                        require(sha(body) == receipt.get('body_sha256'), 'External checksum mismatch.')
                        require(canonical_url(receipt.get('response_url', url)) == url, 'External page redirected.')
                        parse_table(body.decode('utf-8', errors='replace'), task)
                        date = receipt.get('retrieved_at') or receipt.get('collected_at_utc')
                        require(datetime.fromisoformat(date).tzinfo is not None, 'External retrieval timestamp missing.')
                        local = self._save_receipt(task, body, http_status=200, retrieved_at=date,
                            response_url=url, origin='compatible_cache', original_receipt=str(file.resolve()))
                        if self._accept(task, local):
                            used += 1
                            del wanted[url]
                    except (OSError, ValueError, TypeError, KeyError, source.EvidenceError):
                        continue
            self.metrics['compatible_cache_tables'] += used
            return used

    def lookup(self):
        states = {tid: (status, json.loads(receipt) if receipt else {}, error)
                  for tid, status, receipt, error in self.db.execute('SELECT table_id,status,receipt,error FROM tables')}
        records = []
        for requested in self.plan['lookups'].to_dict('records'):
            tid = requested['table_id']
            state, receipt, error = states[tid]
            row = {**requested, **{k: None for k in RATE_FIELDS}, 'status': state,
                   'validation_status': state, 'availability_status': 'unknown', 'valid_rate': False,
                   'sparse': False, 'source_url': self.tasks[tid]['source_url'],
                   'retrieved_at': receipt.get('retrieved_at'), 'response_sha256': receipt.get('body_sha256'),
                   'filter_evidence_json': None, 'error': error,
                   'matchup_evidence_json': None,
                   'rate_units': 'percent: *_pct; probability 0–1: *_probability; differences: *_pp',
                   'matchup_wins_basis': 'not exposed by reviewed table; never reconstructed from rounded percentages',
                   'derived_rate_definition': 'raw_matchup_winrate_pct - tier_average_winrate_pct + 50; percentage units, not a probability; not clipped',
                   'website_delta2_definition': SOURCE_DEFINITIONS['d2'],
                   'source_delta1_definition': SOURCE_DEFINITIONS['d1'],
                   'source_header_avgWr_definition': 'Source header.avgWr; not substituted for the rendered bracket average'}
            if not requested['mapping_verified_locally']:
                row.update(status='unsupported_mapping', availability_status='unknown')
            elif state == 'validated':
                table = self.parsed[tid]
                header = table['header']
                row.update(validation_status='verified', filter_evidence_json=table['evidence'],
                           champion_overall_winrate_pct=header.get('wr'),
                           tier_average_winrate_pct=table['tier_average'],
                           source_header_avgWr_pct=header.get('avgWr'), source_header_avgWrDelta_pp=header.get('avgWrDelta'))
                value = table['rows'].get(int(requested['opponent_champion_id']))
                if value is None:
                    row.update(status='not_published_in_table', availability_status='not_published',
                               error='Successfully verified complete same-role table; opponent row absent; count and rate unknown.')
                else:
                    n, raw = int(value['n']), value['wr']
                    row.update(status='ok' if n > 0 else 'reported_zero_games', availability_status='reported',
                        raw_matchup_winrate_pct=raw, raw_matchup_winrate_probability=raw / 100 if n > 0 else None,
                        matchup_games=n, valid_rate=n > 0, sparse=n < self.config.sparse_below_games,
                        source_delta1_pp=value['d1'], website_delta2_pp=value['d2'], source_pick_rate_pct=value['pr'],
                        matchup_evidence_json=canonical({'enemy_h': ['id', 'wr', 'd1', 'd2', 'pr', 'n'],
                                                         'row': value, 'opposing_lane': table['opponent_lane']}),
                        derived_tier_centered_rate_pct=raw - table['tier_average'] + 50 if n > 0 and table['tier_average'] is not None else None)
            records.append(row)
        result = pd.DataFrame(records)
        require(not result.lookup_key.duplicated().any(), 'Lookup join would multiply rows.')
        return result

    def joined(self):
        lookup = self.lookup()
        mapping = self.plan['mapping']
        model = self.bundle['frames']['longitudinal_modeling'].copy()
        observed = mapping.loc[mapping.request_kind.eq('observed'), KEY + ['lookup_key']]
        require(not observed.duplicated(KEY).any(), 'Duplicate played lookup mapping.')
        benchmark = lookup.rename(columns={c: 'benchmark_' + c for c in lookup if c != 'lookup_key'})
        played = model.merge(observed, on=KEY, how='left', validate='one_to_one').merge(
            benchmark, on='lookup_key', how='left', validate='many_to_one')
        played['benchmark_status'] = played.benchmark_status.fillna('unknown_opponent_or_role_or_observed_mirror')
        alternatives = mapping.loc[mapping.request_kind.eq('alternative')].copy()
        context_fields = [c for c in ('puuid', 'match_id', 'champion_id', 'champion', 'opponent_champion_id',
                                     'opponent_champion', 'rank_observed_at_utc', 'rank_limitation') if c in model]
        context = model[context_fields].rename(columns={'champion_id': 'observed_champion_id', 'champion': 'observed_champion'})
        alternatives = alternatives.merge(context, on=KEY, how='left', validate='many_to_one')
        alternatives['is_eligible_alternative'] = True
        alternatives['opponent_lane'] = alternatives.lane
        alternatives['candidate_outcome'] = pd.array([pd.NA] * len(alternatives), dtype='Int8')
        alternatives = alternatives.merge(benchmark, on='lookup_key', how='left', validate='many_to_one')
        for frame in (played, alternatives):
            for column in ('benchmark_valid_rate', 'benchmark_sparse'):
                frame[column] = frame[column].astype('boolean').fillna(False).astype(bool)
        require(len(played) == len(model) and not played.duplicated(KEY).any(), 'Played join lost/multiplied targets.')
        require(not alternatives.duplicated(KEY + ['candidate_champion_id']).any(), 'Alternative join multiplied rows.')
        require(len(alternatives) == mapping.request_kind.eq('alternative').sum(), 'Alternative join changed row count.')
        require('win' not in alternatives and alternatives.candidate_outcome.isna().all(), 'Unplayed alternative received an observed outcome.')
        return lookup, played, alternatives

    def coverage(self, played, alternatives):
        reports = []
        for name, frame, tier_column in [('played', played, 'seed_tier'), ('alternative', alternatives, 'tier')]:
            report = frame.groupby([tier_column, 'lane', 'benchmark_status'], dropna=False).agg(
                rows=('benchmark_status', 'size'), valid_rows=('benchmark_valid_rate', 'sum'),
                sparse_rows=('benchmark_sparse', 'sum')).reset_index().rename(columns={tier_column: 'tier', 'benchmark_status': 'status'})
            report['dataset'] = name
            report['denominator_rows'] = report.groupby(['tier', 'lane'], dropna=False).rows.transform('sum')
            report['fraction'] = report.rows / report.denominator_rows
            reports.append(report)
        return pd.concat(reports, ignore_index=True)

    def export(self):
        with writer_lock(self.lock_path):
            self._reconcile()  # Revalidate bodies; checkpoint labels alone are insufficient.
            lookup, played, alternatives = self.joined()
            coverage = self.coverage(played, alternatives)
            progress = self.progress()
            complete = bool(progress.status.eq('validated').all() and self.plan['lookups'].mapping_verified_locally.all())
            out = self.run_dir / 'processed' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8])
            out.mkdir(parents=True, exist_ok=False)
            frames = {'matchup_benchmark_lookup': lookup, 'played_target_games': played,
                      'eligible_alternatives': alternatives, 'coverage_missingness': coverage,
                      'table_statuses': progress, 'required_lookups': self.plan['lookups'],
                      'decision_request_mapping': self.plan['mapping']}
            for name, frame in frames.items():
                temporary = out / ('.' + name + '.parquet.tmp')
                frame.to_parquet(temporary, index=False)
                os.replace(temporary, out / (name + '.parquet'))
            coverage.to_csv(out / 'coverage_missingness.csv', index=False)
            atomic_json(out / 'collection_configuration.json', asdict(self.config))
            atomic_json(out / 'source_metadata.json', self.bundle['identity'])
            metadata = {'schema': SCHEMA, 'created_at': stamp(), 'collection_complete': complete,
                        'source_snapshot': self.bundle['identity'], 'study_limits': LIMITS,
                        'table_status_counts': progress.status.value_counts().to_dict(),
                        'lookup_status_counts': lookup.status.value_counts().to_dict(),
                        'row_counts': {'played_before': len(self.bundle['frames']['longitudinal_modeling']),
                                       'played_after': len(played),
                                       'alternatives_before': int(self.plan['mapping'].request_kind.eq('alternative').sum()),
                                       'alternatives_after': len(alternatives)},
                        'sparse_below_games': self.config.sparse_below_games, 'minimum_game_exclusion': None,
                        'access_verification': 'Offline fixtures only during implementation; exact-tier live access/historical availability must be verified by returned content when enabled.',
                        'output_sha256': {p.name: sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
            atomic_json(out / 'collection_metadata.json', metadata)  # Last file is the export completion marker.
            self.status(f'{"COMPLETE" if complete else "PARTIAL — unresolved required work remains"}: versioned export {out}')
            return out, coverage, metadata

    def close(self):
        self.db.close()
        self.http.close()
