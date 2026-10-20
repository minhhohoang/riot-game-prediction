"""Offline page lifecycle tests. Every HTTP response is mocked."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests
from hlr_model import lolalytics_matchup_scraper as s

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'hlr_model/data/lolalytics/runs/20261020T164500_repaired'
FAILED_URL = 'https://lolalytics.com/lol/volibear/build/?lane=bottom&patch=16.19&region=na&tier=master_plus'


def parsed_tables(html, tasks):
    result = {}
    for t in tasks:
        verified = {f'verified_{k}': t[f'requested_{k}'] for k in ['patch','rank','region','queue']}
        verified.update(verified_queue_id=420, verified_lane=t['lane'], verified_opponent_lane=t['lane'],
                        filters_verified=True, filter_evidence_json='{"offline_fixture":true}')
        result[t['key_id']] = {**verified, **s.finish_rate(60., 10, dict(tier_average_winrate_pct=52.,
            tier_average_missing=False, tier_average_source='fixture', tier_average_context='fixture'))}
        if t['opponent_champion_id'] == 999999:
            result[t['key_id']] = {**verified, 'status':'not_published_in_table', 'error':'absent'}
    return result, ['offline fixture']


class PageFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = pd.read_parquet(RUN/'request_plan.parquet')

    def context(self, directory, na_pages=3):
        p = self.plan
        urls = p.loc[p.region.eq('na'), 'batch_url'].drop_duplicates().tolist()
        others = [u for u in urls if u != FAILED_URL][:max(na_pages-1, 0)]
        ordered = [p.loc[p.region.eq('global')].iloc[0].batch_url]
        ordered += others[:1] + [FAILED_URL] + others[1:]
        groups = []
        for url in ordered:
            group = p.loc[p.batch_url.eq(url)]
            groups.append(group if url == FAILED_URL else group.head(1))
        plan = pd.concat(groups, ignore_index=True)
        directory.mkdir(exist_ok=True)
        return dict(run_dir=directory, run_id='offline_test', input_snapshot='fixture',
            config=s.Config(max_attempts=1, export_every_pages=100), plan=plan,
            mapping=pd.DataFrame(columns=['request_kind','region','puuid','match_id','key_id','matchup_key_id','lane']),
            decisions=pd.DataFrame({'puuid':['p'], 'match_id':['m'], 'known_lane_opponent':[False], 'lane':['mid']}), manifest={})

    def run_collection(self, context, statuses, **kwargs):
        calls = []
        def get(session, url, **request_kwargs):
            calls.append(url)
            code = statuses.get(url, 200)
            if isinstance(code, Exception):
                raise code
            response = requests.Response()
            response.status_code = code
            response.url = url
            response.encoding = 'utf-8'
            response._content = b'<html><h1>offline response</h1></html>'
            response.headers['Content-Type'] = 'text/html'
            response.request = requests.Request('GET', url).prepare()
            return response
        with patch.object(requests.Session, 'get', get), patch.object(s, 'parse_batch_table', parsed_tables), \
             patch.object(s.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            result = s.collect(context, **kwargs)
        return result, calls

    def test_isolated_404_records_every_key_and_continues_then_reuses_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp))
            (lookup, _, quality), calls = self.run_collection(c, {FAILED_URL:404})
            failed = lookup.loc[lookup.batch_url.eq(FAILED_URL)]
            self.assertEqual(len(failed), 2)
            self.assertTrue(failed.status.eq(s.PAGE_UNRESOLVED).all())
            self.assertTrue(failed[s.RATE_FIELDS].isna().all().all())
            self.assertFalse(failed.filters_verified.any())
            self.assertEqual(calls[-1], c['plan'].iloc[-1].batch_url)
            self.assertTrue(quality['all_planned_pages_attempted'])
            self.assertFalse(quality['all_keys_valid'])
            self.assertEqual(quality['page_summary_by_region']['na']['unresolved_failed_pages'], 1)
            self.assertEqual(quality['page_summary_by_region']['na']['successful_pages'], 2)
            before = (c['run_dir']/'matchup_checkpoint.jsonl').read_bytes()
            (again, _, _), new_calls = self.run_collection(c, {})
            self.assertEqual(new_calls, [])
            self.assertEqual((c['run_dir']/'matchup_checkpoint.jsonl').read_bytes(), before)
            pd.testing.assert_frame_equal(lookup, again, check_like=True)

    def test_bounded_retry_only_unresolved_never_pending_or_successful(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp), na_pages=4)
            # Leave one pending page while collecting a prefix in the same temporary run.
            full_plan = c['plan']
            c['plan'] = full_plan.iloc[:-1]
            self.run_collection(c, {FAILED_URL:404})
            c['plan'] = full_plan
            old = s.load_results(c)
            successful = {k:v.copy() for k,v in old.items() if v['status']=='ok'}
            receipt = c['run_dir']/'responses'/f'{s.digest(FAILED_URL.encode())}.json'
            old_receipt = json.loads(receipt.read_text())
            body = c['run_dir']/old_receipt['body_file']
            old_body = body.read_bytes()
            (lookup, _, quality), calls = self.run_collection(c, {}, retry_max_pages=1, retry_regions=('na',))
            self.assertEqual(calls, [FAILED_URL])
            self.assertEqual(lookup.status.eq('pending').sum(), 1)
            self.assertFalse(quality['all_planned_pages_attempted'])
            self.assertTrue(lookup.loc[lookup.batch_url.eq(FAILED_URL), 'status'].eq('ok').all())
            self.assertEqual(body.read_bytes(), old_body)
            archives = list((c['run_dir']/'responses').glob('*-receipt-*.json'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text()), old_receipt)
            after = s.load_results(c)
            for key, row in successful.items():
                self.assertEqual(after[key], row)

    def test_repeated_failures_pause_with_pending_pages_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp), na_pages=8)
            statuses = {u:404 for u in c['plan'].loc[c['plan'].region.eq('na'), 'batch_url']}
            (lookup, _, quality), calls = self.run_collection(c, statuses)
            self.assertEqual(len(calls), 6)  # Global success + five consecutive NA failures.
            self.assertGreater(lookup.status.eq('pending').sum(), 0)
            state = json.loads((c['run_dir']/'run_state.json').read_text())
            self.assertIn('Repeated/systemic', state['pause_reason'])
            self.assertFalse(quality['all_planned_pages_attempted'])

    def test_parse_failures_also_continue_and_pause_at_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp), na_pages=8)
            original = parsed_tables
            def parser(html, tasks):
                if tasks[0]['region']=='na':
                    raise s.EvidenceError('unverified_filters', 'offline wrong region')
                return original(html, tasks)
            # run_collection patches parser; intercept it with a side effect here.
            with patch.dict(globals(), parsed_tables=parser):
                (lookup, _, _), calls = self.run_collection(c, {})
            self.assertEqual(len(calls), 6)
            failed = lookup.loc[lookup.status.eq(s.PAGE_UNRESOLVED)]
            self.assertFalse(failed.filters_verified.any())
            self.assertTrue(failed[s.RATE_FIELDS].isna().all().all())

    def test_access_block_and_persistent_429_stop_immediately(self):
        for code in [403, 429]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                c = self.context(Path(tmp))
                first_na = c['plan'].loc[c['plan'].region.eq('na')].iloc[0].batch_url
                (lookup, _, _), calls = self.run_collection(c, {first_na:code})
                self.assertEqual(len(calls), 2)
                self.assertIn('blocked', lookup.status.values)
                self.assertGreater(lookup.status.eq('pending').sum(), 0)

    def test_network_failure_is_not_missing_matchup_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp))
            (lookup, _, _), calls = self.run_collection(c, {FAILED_URL:requests.ConnectionError('offline')})
            self.assertEqual(len(calls), 4)
            failed = lookup.loc[lookup.batch_url.eq(FAILED_URL)]
            self.assertTrue(failed.status.eq(s.PAGE_UNRESOLVED).all())
            self.assertTrue(failed[s.RATE_FIELDS].isna().all().all())

    def test_missing_opponents_are_separate_from_failed_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp))
            original = parsed_tables
            def parser(html, tasks):
                rows, notes = original(html, tasks)
                if tasks[0]['region']=='na' and tasks[0]['batch_url'] != FAILED_URL:
                    for row in rows.values():
                        row.update(status='not_published_in_table', error='absent')
                        for field in s.RATE_FIELDS:
                            row.pop(field, None)
                return rows, notes
            with patch.dict(globals(), parsed_tables=parser):
                (_, _, quality), _ = self.run_collection(c, {FAILED_URL:404})
            report = quality['page_summary_by_region']['na']
            self.assertEqual(report['successful_pages'], 2)
            self.assertEqual(report['verified_tables_missing_requested_opponents'], 2)
            self.assertEqual(report['missing_requested_opponent_keys'], 2)
            self.assertEqual(report['unresolved_failed_keys'], 2)

    def test_bounded_retry_limit_leaves_other_unresolved_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp), na_pages=3)
            failures = dict.fromkeys(c['plan'].loc[c['plan'].region.eq('na'), 'batch_url'], 404)
            self.run_collection(c, failures)
            (_, _, quality), calls = self.run_collection(c, {}, retry_max_pages=1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(quality['page_summary_by_region']['na']['unresolved_failed_pages'], 2)
            with self.assertRaises(ValueError):
                s.retry_unresolved_pages(c, max_pages=0)

    def test_rolling_systemic_failures_pause_without_consecutive_streak(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp), na_pages=12)
            urls = c['plan'].loc[c['plan'].region.eq('na'), 'batch_url'].drop_duplicates().tolist()
            failures = {u:404 for i,u in enumerate(urls) if i % 2 == 1}
            (lookup, _, _), calls = self.run_collection(c, failures)
            self.assertEqual(len(calls), 11)  # One Global + ten NA pages.
            self.assertGreater(lookup.status.eq('pending').sum(), 0)
            state = json.loads((c['run_dir']/'run_state.json').read_text())
            self.assertIn('5/10', state['pause_reason'])

    def test_offline_reconciliation_checkpoints_cached_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.context(Path(tmp))
            source = RUN/'responses'/f'{s.digest(FAILED_URL.encode())}.json'
            receipt = json.loads(source.read_text())
            responses = c['run_dir']/'responses'
            responses.mkdir()
            (c['run_dir']/receipt['body_file']).write_bytes((RUN/receipt['body_file']).read_bytes())
            (responses/source.name).write_bytes(source.read_bytes())
            latest = s.reconcile_cached_page_failures(c)
            self.assertEqual(len(latest), 2)
            self.assertTrue(all(r['status']==s.PAGE_UNRESOLVED for r in latest.values()))
            self.assertEqual(s.reconcile_cached_page_failures(c), latest)


if __name__=='__main__':
    unittest.main()
