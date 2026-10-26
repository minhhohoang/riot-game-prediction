"""Offline only: the network is disabled globally; HTTP is injected explicitly."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from hlr_model.broad_rank_pipeline import (
    BroadConfig,BroadCollector,DiskState,DurableClient,ContractError,FrameExhausted,
    writer_lock,hashed,TIERS,QUEUE,valid_entries,atomic_write,
)
from hlr_model.longitudinal_pipeline import Config
from hlr_model.riot_api_helpers import AuthenticationError

@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    monkeypatch.setattr(requests.Session,'get',lambda *a,**kw:pytest.fail('Real HTTP forbidden in offline tests'))

class Clock:
    def __init__(self):self.now=2000000000.;self.waits=[]
    def time(self):return self.now
    def sleep(self,s):self.waits.append(s);self.now+=s

class Response:
    def __init__(self,data=None,status=200,headers=None):self.data=data;self.status_code=status;self.headers=headers or {}
    def json(self):return self.data

class HTTP:
    def __init__(self,handler):self.handler=handler;self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append((url,kwargs.get('params')))
        value=self.handler(url,kwargs.get('params') or {})
        if isinstance(value,BaseException):raise value
        return value if isinstance(value,Response) else Response(value)

def config(**kw):
    return BroadConfig(tiers=('BRONZE',),per_tier=1,league_pages_per_division=1,
        study=Config(player_count=1,historical_depth=2,page_size=2),**kw)

def entry(uid,tier='BRONZE',division='I',queue=QUEUE):
    return {'puuid':uid,'tier':tier,'rank':division,'queueType':queue}

def payload(mid,uid,start,end=None,patch='16.19.1',extra=()):
    ids=list(dict.fromkeys([uid,*extra]+[f'other{i}' for i in range(10)]))[:10]
    participants=[]
    for i,p in enumerate(ids):
        participants.append({'puuid':p,'championName':'Annie' if i==0 else 'Galio','championId':1 if i==0 else 3,
            'win':i<5,'teamId':100 if i<5 else 200,'teamPosition':['TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY'][i%5]})
    return {'metadata':{'matchId':mid,'participants':ids},'info':{'participants':participants,'queueId':420,
        'gameVersion':patch,'gameStartTimestamp':start,'gameEndTimestamp':end or start+30000}}

def collector(tmp_path,http,*,cfg=None,clock=None,enabled=True,legacy=None,excluded=()):
    clock=clock or Clock()
    return BroadCollector(tmp_path,tmp_path/'run',cfg or config(),enabled=enabled,api_key='SENTINEL_NEVER_LOG',
        session=http,excluded_players=excluded,legacy_roots=legacy or [],clock=clock.time,sleep=clock.sleep,startup_delay=0,status=lambda x:None)

class FixtureAPI:
    def __init__(self,cfg=None,players=None,histories=None):
        self.cfg=cfg or config();self.players=players or {'BRONZE':['p']};self.history=histories
        self.anchor=self.cfg.study.target_search_start*1000+100000
        self.payloads={}
        for tier,ids in self.players.items():
            for uid in ids:
                number=len(self.payloads)+1
                t=f'NA1_{number}';h1=f'NA1_{number+1}';h2=f'NA1_{number+2}'
                self.payloads[t]=payload(t,uid,self.anchor)
                self.payloads[h1]=payload(h1,uid,self.anchor-100000)
                self.payloads[h2]=payload(h2,uid,self.anchor-200000)
    def ids_for(self,uid):return [mid for mid,p in self.payloads.items() if p['metadata']['participants'][0]==uid]
    def __call__(self,url,params):
        if '/entries/by-puuid/' in url:
            uid=url.split('/')[-1]
            return [entry(uid,tier) for tier,ids in self.players.items() if uid in ids]
        if '/league/v4/entries/' in url:
            tier,division=url.split('/')[-2:]
            return [entry(uid,tier,division) for uid in self.players.get(tier,[])] if division=='I' else []
        if '/by-puuid/' in url:
            uid=url.split('/by-puuid/')[1].split('/')[0];ids=self.ids_for(uid)
            ids=ids[:1] if params['startTime']==self.cfg.study.target_search_start else (ids[1:] if self.history is None else self.history)
            return ids[params['start']:params['start']+params['count']]
        return self.payloads[url.split('/')[-1]]

def test_allowed_tiers_and_default_definitions():
    c=BroadConfig()
    assert c.study.player_count==300 and c.per_tier==50 and c.tiers==TIERS
    assert c.study.historical_depth==170 and c.study.queue==420 and c.study.target_patch=='16.19'
    with pytest.raises(ValueError):replace(c,tiers=('IRON',))
    with pytest.raises(ValueError):replace(c,tiers=('MASTER',))
    assert valid_entries([]) and not valid_entries([{'summonerId':'outdated schema'}])

def test_full_offline_run_short_history_export_and_zero_requests_after_done(tmp_path):
    api=FixtureAPI(histories=[]);http=HTTP(api)
    run=collector(tmp_path,http);run.collect()
    assert run.done and len(run.completed())==1
    p=run.completed()[0]
    assert p['history_complete'] and p['history_exhausted'] and p['history_ids']==[]
    out=run.export();model=pd.read_parquet(out/'longitudinal_modeling.parquet')
    assert model.genuinely_short_history.all() and model.prior_champion_games.eq(0).all()
    assert model.cohort_id.eq('broad_na_16_19').all()
    calls=len(http.calls);run.close()
    resumed=collector(tmp_path,http,enabled=None)
    resumed.collect();assert len(http.calls)==calls and resumed.done
    assert resumed.export()==out
    assert resumed.progress().completed.sum()==1
    assert all('SENTINEL_NEVER_LOG' not in p.read_text() for p in (tmp_path/'run').rglob('*.json'))

def test_all_rank_quotas_and_master_exclusion(tmp_path):
    cfg=BroadConfig(per_tier=1,league_pages_per_division=1,study=Config(player_count=6,historical_depth=1,page_size=2))
    api=FixtureAPI(cfg,{t:[t.lower()] for t in TIERS});http=HTTP(api)
    run=collector(tmp_path,http,cfg=cfg,excluded=('old_master',))
    run.collect()
    assert run.done and set(p['tier'] for p in run.completed())==set(TIERS)
    assert all(len(run.completed(t))==1 for t in TIERS)
    assert len(run.state['selected'])==6

def test_exact_query_cache_survives_response_checkpoint_crash(tmp_path,monkeypatch):
    clock=Clock();disk=DiskState(tmp_path/'state.sqlite');http=HTTP(lambda *a:[])
    client=DurableClient(disk,'secret',enabled=True,session=http,clock=clock.time,sleep=clock.sleep,startup_delay=0,status=lambda x:None)
    original=disk.save_response
    def crash(*a):original(*a);raise KeyboardInterrupt
    monkeypatch.setattr(disk,'save_response',crash)
    with pytest.raises(KeyboardInterrupt):client.request('americas','/lol/match/v5/matches/by-puuid/p/ids',method='match_ids',params={'count':2},validate=lambda x:isinstance(x,list))
    disk.close();disk=DiskState(tmp_path/'state.sqlite')
    client=DurableClient(disk,'refreshed',enabled=False,session=http,clock=clock.time,sleep=clock.sleep,startup_delay=0)
    assert client.request('americas','/lol/match/v5/matches/by-puuid/p/ids',method='match_ids',params={'count':2})[0]==[]
    assert len(http.calls)==1
    with pytest.raises(Exception):client.request('americas','/lol/match/v5/matches/by-puuid/p/ids',method='match_ids',params={'count':1})

def test_interrupted_target_pagination_resumes_exact_page(tmp_path):
    api=FixtureAPI();t=api.ids_for('p')[0]
    extra=['NA1_100','NA1_101'];api.payloads.update({mid:payload(mid,'p',api.anchor+1000) for mid in extra})
    interrupted=[False]
    def handler(url,p):
        if '/by-puuid/p/ids' in url and p['startTime']==api.cfg.study.target_search_start:
            if p['start']==0:return [t,extra[0]]
            if not interrupted[0]:interrupted[0]=True;raise KeyboardInterrupt
            return [extra[1]]
        return api(url,p)
    http=HTTP(handler);run=collector(tmp_path,http)
    with pytest.raises(KeyboardInterrupt):run.collect()
    assert run.state['players']['p']['target_query']['offset']==2
    run.close();resumed=collector(tmp_path,http,enabled=None);resumed.collect();assert resumed.done
    zero_pages=[p for u,p in http.calls if '/by-puuid/p/ids' in u and p['startTime']==api.cfg.study.target_search_start and p['start']==0]
    assert len(zero_pages)==1

def test_raw_match_saved_before_player_checkpoint_is_reused(tmp_path):
    api=FixtureAPI();http=HTTP(api);run=collector(tmp_path,http)
    mid=api.ids_for('p')[0]
    run.record(mid,'p',download=True)
    calls=len(http.calls);run.close()
    resumed=collector(tmp_path,http,enabled=None)
    assert resumed.record(mid,'p',download=True)['match_id']==mid
    assert len(http.calls)==calls

def test_shared_match_downloaded_once_and_legacy_read_only(tmp_path):
    mid='NA1_999';start=config().study.target_search_start*1000+50000
    match=payload(mid,'p',start,extra=('q',));http=HTTP(lambda *a:match)
    run=collector(tmp_path,http)
    assert run.record(mid,'p',download=True) and run.record(mid,'q',download=True)
    assert len(http.calls)==1
    legacy=tmp_path/'legacy';legacy.mkdir();atomic_write(legacy/(mid+'.json'),match)
    before=(legacy/(mid+'.json')).read_bytes()
    second=BroadCollector(tmp_path,tmp_path/'second',config(),enabled=True,api_key='x',session=http,
        excluded_players=[],legacy_roots=[legacy],startup_delay=0,status=lambda x:None)
    assert second.record(mid,'p',download=True) and len(http.calls)==1
    assert (legacy/(mid+'.json')).read_bytes()==before
    assert not (tmp_path/'second/raw/matches'/f'{mid}.json').exists()

def test_temporary_failure_never_replaces_or_completes_player(tmp_path):
    api=FixtureAPI(players={'BRONZE':['p','q']})
    http=HTTP(api);run=collector(tmp_path,http);run.discover()
    first=run.state['candidate_order']['BRONZE'][0]['puuid']
    stage=[0]
    def handler(url,p):
        if '/by-puuid/'+first+'/ids' in url:
            stage[0]+=1
            if stage[0]==1:return requests.ConnectionError('SENSITIVE MUST NOT PERSIST')
            raise KeyboardInterrupt
        return api(url,p)
    http.handler=handler
    with pytest.raises(KeyboardInterrupt):run.collect()
    assert not run.done and list(run.state['players'])==[first]
    assert run.state['players'][first]['status']=='incomplete'
    assert run.state['cursor']['BRONZE']==0
    assert 'SENSITIVE' not in json.dumps(run.state)
    http.handler=api;run.collect();assert run.done and first in [p['puuid'] for p in run.completed()]

def test_history_failure_does_not_become_genuinely_short(tmp_path):
    api=FixtureAPI();http=HTTP(api);run=collector(tmp_path,http)
    history_id=api.ids_for('p')[1]
    http.handler=lambda u,p: (_ for _ in ()).throw(KeyboardInterrupt()) if u.endswith('/'+history_id) else api(u,p)
    with pytest.raises(KeyboardInterrupt):run.collect()
    p=run.state['players']['p'];assert not p.get('history_complete',False) and p['status']=='incomplete'
    assert p['history_ids']==[] and run.state['cursor']['BRONZE']==0
    http.handler=api;run.collect();assert run.done and len(run.completed()[0]['history_ids'])==2

def test_only_definitive_exclusions_advance_frozen_reserve(tmp_path):
    api=FixtureAPI(players={'BRONZE':['p','q']});http=HTTP(api);run=collector(tmp_path,http);run.discover()
    first=run.state['candidate_order']['BRONZE'][0]['puuid']
    def handler(u,p):
        if '/by-puuid/'+first+'/ids' in u and p['startTime']==api.cfg.study.target_search_start:return []
        return api(u,p)
    http.handler=handler;run.collect()
    assert run.done and run.state['players'][first]['excluded_reason']=='no_valid_patch_targets'
    assert run.progress().excluded.sum()==1
    assert first not in run.state['selected']['BRONZE']

def test_wrong_rank_is_definitive_but_auth_is_not(tmp_path):
    api=FixtureAPI(players={'BRONZE':['p','q']});http=HTTP(api);run=collector(tmp_path,http);run.discover()
    first=run.state['candidate_order']['BRONZE'][0]['puuid']
    http.handler=lambda u,p:[entry(first,'IRON')] if u.endswith('/entries/by-puuid/'+first) else api(u,p)
    run.collect();assert run.done and run.state['players'][first]['status']=='excluded'

def test_429_authentication_persisted_cooldown_and_refreshed_key(tmp_path):
    clock=Clock();disk=DiskState(tmp_path/'state.sqlite')
    responses=iter([Response(status=429,headers={'Retry-After':'7'}),Response([])])
    http=HTTP(lambda *a:next(responses))
    client=DurableClient(disk,'secret',enabled=True,session=http,clock=clock.time,sleep=clock.sleep,startup_delay=0,status=lambda x:None)
    client.request('na1','/lol/league/v4/entries/by-puuid/p',method='rank')
    assert sum(clock.waits)>=8 and len(http.calls)==2
    auth=HTTP(lambda *a:Response(status=403));before=clock.now
    client=DurableClient(disk,'expired',enabled=True,session=auth,clock=clock.time,sleep=clock.sleep,startup_delay=0)
    with pytest.raises(AuthenticationError):client.request('na1','/lol/league/v4/entries/by-puuid/q',method='rank')
    assert len(auth.calls)==1 and disk.get('network')['status']=='authentication_error'
    assert 'expired' not in json.dumps(disk.get('network'))
    # Restart protection is a real wait in production, a fake-clock advance here.
    resumed=DurableClient(disk,'new',enabled=True,session=HTTP(lambda *a:[]),clock=clock.time,sleep=clock.sleep,startup_delay=121)
    resumed.request('na1','/lol/league/v4/entries/by-puuid/q',method='rank')
    assert clock.now>=before+121

def test_cooldown_survives_interrupt_and_restart(tmp_path):
    clock=Clock();disk=DiskState(tmp_path/'state.sqlite');http=HTTP(lambda *a:Response(status=429,headers={'Retry-After':'60'}))
    def interrupted_sleep(s):raise KeyboardInterrupt
    c=DurableClient(disk,'x',enabled=True,session=http,clock=clock.time,sleep=interrupted_sleep,startup_delay=0,status=lambda x:None)
    with pytest.raises(KeyboardInterrupt):c.request('na1','/lol/league/v4/entries/by-puuid/p',method='rank')
    blocked=disk.get('rates')['na1:app']['blocked_until'];assert blocked>clock.now
    disk.close();disk=DiskState(tmp_path/'state.sqlite')
    resumed=DurableClient(disk,'new',enabled=True,session=HTTP(lambda *a:[]),clock=clock.time,sleep=clock.sleep,startup_delay=0)
    resumed.request('na1','/lol/league/v4/entries/by-puuid/p',method='rank')
    assert clock.now>=blocked

def test_config_changes_rejected_and_competing_lock(tmp_path):
    run=collector(tmp_path,HTTP(FixtureAPI()));before=run.disk.get('state')
    with pytest.raises(ValueError):collector(tmp_path,HTTP(FixtureAPI()),cfg=replace(config(),seed=999))
    assert run.disk.get('state')==before
    with writer_lock(run.lock_path):
        with pytest.raises(RuntimeError):collector(tmp_path,HTTP(FixtureAPI()))

def test_corrupt_completed_payload_retried_not_replaced(tmp_path):
    api=FixtureAPI();http=HTTP(api);run=collector(tmp_path,http);run.collect();assert run.done
    uid=run.completed()[0]['puuid'];mid=run.completed()[0]['history_ids'][0]
    (run.roots[0]/(mid+'.json')).write_text('{truncated')
    before=len(http.calls);run.collect()
    assert run.done and [p['puuid'] for p in run.completed()]==[uid]
    new_details=[u for u,p in http.calls[before:] if u.endswith('/'+mid)]
    assert len(new_details)==1

def test_offline_disabled_never_uses_http_and_atomic_temp_is_ignored(tmp_path):
    http=HTTP(lambda *a:pytest.fail('Network should be disabled'))
    run=collector(tmp_path,http,enabled=None);run.collect();assert not http.calls
    assert not run.done
    run.roots[0].mkdir(parents=True);(run.roots[0]/'.NA1_123.json.interrupted.tmp').write_text('{')
    assert run.record('NA1_123','p') is None

def test_frame_excludes_existing_master_wrong_queue_and_uses_saved_order(tmp_path):
    api=FixtureAPI();http=HTTP(api)
    def handler(u,p):
        if u.endswith('/'+QUEUE+'/BRONZE/I'):
            return [entry('old_master'),entry('wrong_queue',queue='RANKED_FLEX_SR'),entry('iron','IRON'),entry('p')]
        return api(u,p)
    http.handler=handler;run=collector(tmp_path,http,excluded=['old_master']);run.discover()
    assert [r['puuid'] for r in run.state['candidate_order']['BRONZE']]==['p']
    original=run.state['candidate_order'];calls=len(http.calls);run.close()
    resumed=collector(tmp_path,http,enabled=None);resumed.discover()
    assert resumed.state['candidate_order']==original and len(http.calls)==calls

def test_history_overlap_and_expanding_completed_targets(tmp_path):
    api=FixtureAPI();anchor=api.anchor
    mids=['NA1_1001','NA1_1002','NA1_1003']
    api.payloads.update({mids[0]:payload(mids[0],'p',anchor,anchor+5000),
        mids[1]:payload(mids[1],'p',anchor+1000,anchor+2000),
        mids[2]:payload(mids[2],'p',anchor+6000,anchor+10000),
        'NA1_1004':payload('NA1_1004','p',anchor-1000,anchor+1000),
        'NA1_1005':payload('NA1_1005','p',anchor-100000),
        'NA1_1006':payload('NA1_1006','p',anchor-200000)})
    hist=['NA1_1004','NA1_1005','NA1_1006']
    def handler(u,p):
        if '/by-puuid/p/ids' in u:
            ids=mids if p['startTime']==api.cfg.study.target_search_start else hist
            return ids[p['start']:p['start']+p['count']]
        return api(u,p)
    http=HTTP(handler);run=collector(tmp_path,http);run.collect()
    assert run.done
    player=run.completed()[0]
    assert set(player['history_ids'])=={'NA1_1005','NA1_1006'}
    assert player['history_exclusions']['not_completed_before_anchor']==1
    out=run.export();model=pd.read_parquet(out/'longitudinal_modeling.parquet').set_index('match_id')
    assert list(model.loc[mids,'prior_total_games'])==[2,2,4]
    assert model.prior_latest_end_timestamp.lt(model.timestamp).all()
    history_requests=[p for u,p in http.calls if '/by-puuid/p/ids' in u and p['startTime']==api.cfg.study.season_start]
    assert [(p['start'],p['count']) for p in history_requests]==[(0,2),(2,1)]

def test_corrupt_complete_status_reconciled_on_preview_restart(tmp_path):
    api=FixtureAPI();http=HTTP(api);run=collector(tmp_path,http);run.collect()
    mid=run.completed()[0]['target_ids'][0];(run.roots[0]/(mid+'.json')).unlink()
    run.close();before=len(http.calls)
    preview=collector(tmp_path,http,enabled=False)
    assert not preview.done and preview.progress().incomplete.sum()==1
    preview.collect();assert len(http.calls)==before

def test_method_limits_and_external_counts_are_persisted(tmp_path):
    clock=Clock();disk=DiskState(tmp_path/'state.sqlite')
    headers={'X-App-Rate-Limit':'20:1,100:120','X-App-Rate-Limit-Count':'20:1,100:120',
             'X-Method-Rate-Limit':'1:10','X-Method-Rate-Limit-Count':'1:10'}
    responses=iter([Response([],headers=headers),Response([])])
    http=HTTP(lambda *a:next(responses))
    c=DurableClient(disk,'x',enabled=True,session=http,clock=clock.time,sleep=clock.sleep,startup_delay=0)
    c.request('na1','/lol/league/v4/entries/by-puuid/p',method='rank')
    start=clock.now
    c.request('na1','/lol/league/v4/entries/by-puuid/q',method='rank')
    assert clock.now-start>=120
    assert disk.get('rates')['na1:rank']['limits']==[[1,10.0]]

def test_authentication_mid_history_preserves_player_and_resumes(tmp_path):
    api=FixtureAPI();http=HTTP(api);run=collector(tmp_path,http)
    mid=api.ids_for('p')[1]
    http.handler=lambda u,p:Response(status=401) if u.endswith('/'+mid) else api(u,p)
    with pytest.raises(AuthenticationError):run.collect()
    assert run.progress().incomplete.sum()==1 and run.progress().excluded.sum()==0 and not run.done
    assert run.state['cursor']['BRONZE']==0
    run.close();http.handler=api
    resumed=collector(tmp_path,http,enabled=None);resumed.client.key='REFRESHED_KEY'
    resumed.collect();assert resumed.done and [p['puuid'] for p in resumed.completed()]==['p']

def test_corrupt_export_gets_new_snapshot_without_overwrite(tmp_path):
    api=FixtureAPI();run=collector(tmp_path,HTTP(api));run.collect()
    first=run.export();bad=first/'target_matches.parquet';bad.write_bytes(b'truncated')
    second=run.export()
    assert second!=first and bad.read_bytes()==b'truncated'
    assert len(pd.read_parquet(second/'target_matches.parquet'))>0

def test_multi_player_cohort_order_and_export_survive_restart(tmp_path):
    cfg=BroadConfig(tiers=('BRONZE',),per_tier=2,league_pages_per_division=1,
        study=Config(player_count=2,historical_depth=1,page_size=2))
    api=FixtureAPI(cfg,players={'BRONZE':['z_player','a_player']});http=HTTP(api)
    run=collector(tmp_path,http,cfg=cfg);run.collect();assert run.done
    expected=[p['puuid'] for p in run.state['candidate_order']['BRONZE']]
    assert [p['puuid'] for p in run.completed()]==expected
    first=run.export();run.close()
    resumed=collector(tmp_path,http,cfg=cfg,enabled=None)
    assert [p['puuid'] for p in resumed.completed()]==expected
    assert resumed.export()==first
    assert pd.read_parquet(first/'player_cohort.parquet').puuid.tolist()==expected
