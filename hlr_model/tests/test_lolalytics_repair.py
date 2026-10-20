"""Offline regression tests using the preserved live diagnostic receipts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from hlr_model import lolalytics_matchup_scraper as scraper

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / 'hlr_model/data/lolalytics/diagnostics/20261020_url_repair'
OLD = ROOT / 'hlr_model/data/lolalytics/runs/20261020T163330214361Z'


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan = pd.read_parquet(OLD / 'request_plan.parquet')
        cls.tasks = plan.loc[plan.observed & plan.champion.eq('Fizz')
                             & plan.opponent_champion.eq('Ahri') & plan.lane.eq('mid')].to_dict('records')
        cls.html = {}
        for f in (DIAG / 'responses').glob('*.json'):
            r = json.loads(f.read_text())
            u = r['requested_url']
            if '/fizz/build/' in u and 'lane=middle' in u and 'queue=' not in u:
                cls.html[r['requested_region']] = (DIAG/r['body_file']).read_text()

    def test_real_both_regions_with_samples_and_tier_average(self):
        for task in self.tasks:
            rates, _ = scraper.parse_batch_table(self.html[task['region']], [task])
            r = rates[task['key_id']]
            expected = (52.41, 685, 52.45) if task['region'] == 'global' else (54.55, 66, 52.64)
            self.assertEqual((r['raw_matchup_winrate_pct'], r['matchup_games'], r['tier_average_winrate_pct']), expected)
            self.assertTrue(r['filters_verified'])
            self.assertEqual(r['verified_region'], task['region'])

    def test_wrong_region_and_fallback_filters_rejected(self):
        task = next(t for t in self.tasks if t['region'] == 'na')
        with self.assertRaises(scraper.EvidenceError):
            scraper.parse_batch_table(self.html['global'], [task])
        for before, after in [('MASTER+', 'EMERALD+'), ('16.19', '16.20'),
                              ('Solo/Duo', 'Flex')]:
            self.assertTrue(before in self.html['na'], f'Missing fixture token: {before}')
            with self.assertRaises(scraper.EvidenceError):
                scraper.parse_batch_table(self.html['na'].replace(before, after), [task])

    def test_explicit_data_schema_and_header_are_required(self):
        task = self.tasks[0]
        html = self.html[task['region']]
        with self.assertRaises(scraper.EvidenceError):
            scraper.parse_batch_table(html.replace('"enemy_h":', '"unreviewed_enemy_h":'), [task])
        # Requested query values alone cannot establish a returned dataset.
        with self.assertRaises(scraper.EvidenceError):
            scraper.parse_batch_table('<h1>Fizz middle Patch 16.19 Master+</h1>', [task])

    def test_missing_table_opponent_keeps_unknown_sample(self):
        task = dict(self.tasks[0], opponent_champion_id=999999)
        rows, _ = scraper.parse_batch_table(self.html[task['region']], [task])
        r = rows[task['key_id']]
        self.assertEqual(r['status'], 'not_published_in_table')
        self.assertNotIn('raw_matchup_winrate_pct', r)
        self.assertNotIn('matchup_games', r)
        self.assertTrue(r['filters_verified'])

    def test_extracted_record_passes_receipt_provenance(self):
        task = self.tasks[0].copy()
        region = task['region']
        r = next(json.loads(f.read_text()) for f in (DIAG/'responses').glob('*.json')
                 if json.loads(f.read_text())['requested_url'].find('/fizz/build/?lane=middle') >= 0
                 and json.loads(f.read_text())['requested_region'] == region)
        task['batch_url'] = r['requested_url']
        rows, _ = scraper.parse_batch_table(self.html[region], [task])
        context = dict(run_dir=DIAG, run_id='diagnostic', input_snapshot='test')
        record = scraper.result_record(task, rows[task['key_id']], context, r)
        scraper.validate_checkpoint_provenance(record, task, context)
        record['response_sha256'] = 'tampered'
        with self.assertRaises(ValueError):
            scraper.validate_checkpoint_provenance(record, task, context)

    def test_url_and_impossible_candidate(self):
        url = scraper.build_url('xerath', 'adc', scraper.Config(), 'varus', region='na')
        self.assertNotIn('queue=', url)
        self.assertIn('lane=bottom', url)
        self.assertIn('vslane=bottom', url)
        self.assertIn('region=na', url)
        # Prior Ahri is ineligible against Ahri, but a prior Annie qualifies.
        history = pd.DataFrame([
            dict(puuid='p', match_id='h1', timestamp=1, end_timestamp=2, champion_id=103, champion='Ahri', lane='mid'),
            dict(puuid='p', match_id='h2', timestamp=3, end_timestamp=4, champion_id=1, champion='Annie', lane='mid')])
        targets = pd.DataFrame([dict(puuid='p', match_id='t', timestamp=10, end_timestamp=20,
            champion_id=105, champion='Fizz', lane='mid', matchup_missing=False,
            opponent_champion_id=103, opponent_champion='Ahri')])
        plan, mapping, decisions = scraper.build_plan(pd.DataFrame(), history, targets, scraper.Config())
        self.assertFalse(plan.champion_id.eq(plan.opponent_champion_id).any())
        self.assertIn(1, mapping.candidate_champion_id.values)
        targets.loc[0, ['champion_id', 'champion']] = [103, 'Ahri']
        # Add a non-mirror target so planning remains nonempty.
        targets = pd.concat([targets, pd.DataFrame([dict(targets.iloc[0], match_id='t2', champion_id=105, champion='Fizz')])])
        plan, _, decisions = scraper.build_plan(pd.DataFrame(), history, targets, scraper.Config())
        self.assertEqual(decisions.unexpected_observed_mirror.sum(), 1)
        self.assertFalse(plan.champion_id.eq(plan.opponent_champion_id).any())

    def test_scoped_404_retry_preserves_old_body_and_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = scraper.PageCache(root, scraper.Config(max_attempts=1))
            url = scraper.build_url('fizz', 'mid', scraper.Config())
            pid = scraper.digest(url.encode())
            body = cache.cache/'old.html'
            body.write_bytes(b'old 404')
            receipt = dict(requested_url=url, requested_region='global', http_status=404,
                           body_file='responses/old.html', body_sha256=scraper.digest(body.read_bytes()), encoding='utf-8')
            scraper.atomic_json(cache.cache/f'{pid}.json', receipt)
            response = Mock(status_code=200, content=b'new 200', encoding='utf-8', url=url,
                            headers={'Content-Type':'text/html'}, request=Mock(headers={'User-Agent':'test'}))
            cache.session.get = Mock(return_value=response)
            self.assertEqual(cache.get(url)[1]['http_status'], 404)
            cache.session.get.assert_not_called()
            self.assertEqual(cache.get(url, retry_cached_error=True)[1]['http_status'], 200)
            self.assertEqual(body.read_bytes(), b'old 404')
            self.assertEqual(len(list(cache.cache.glob('*-receipt-*.json'))), 1)
            cache.close()


if __name__ == '__main__':
    unittest.main()
