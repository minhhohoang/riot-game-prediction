"""Opt-in, single-writer broad-rank collection; importing performs no I/O.

SQLite FULL transactions checkpoint state and exact-parameter HTTP page caches.
Raw matches are atomically written only to this run; legacy caches are read-only.
Rank quotas describe a bounded collection-time API frame, not NA population shares.
"""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from hlr_model.longitudinal_pipeline import Config, RECORD_COLUMNS, player_record, expanding_features, quality_report
from hlr_model.riot_api_helpers import AuthenticationError, CollectionDisabled

TIERS=('BRONZE','SILVER','GOLD','PLATINUM','EMERALD','DIAMOND')
DIVISIONS=('I','II','III','IV')
QUEUE='RANKED_SOLO_5x5'
SCHEMA=1

class ContractError(RuntimeError):
    """Schema/request mismatch: stop with work pending, never exclude a player."""

class MatchContractError(ContractError):
    """Only allowlisted field paths, types and conditions; never response values."""
    def __init__(self,mid,issues):
        self.diagnostics={'match_id':mid,'issues':issues}
        fields='; '.join(f"{x['field']}: {x['actual_type']}"+
            (f" length={x['length']}" if 'length' in x else '')+f" (expected {x['expected']})" for x in issues)
        super().__init__(f'Match {mid} is unusable: {fields}. Work remains pending.')

class AbortedMatch(RuntimeError):
    """A verified pre-start abort has no player outcome; never a failed fetch."""

class FrameExhausted(RuntimeError):pass

def utc(ts=None):return datetime.fromtimestamp(time.time() if ts is None else ts,timezone.utc).isoformat()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def hashed(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def match_id(x):return isinstance(x,str) and re.fullmatch(r'NA1_\d+',x) is not None

def atomic_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)

@contextlib.contextmanager
def writer_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another cooperating collector/exporter holds the repository collection lock.') from None
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

@dataclass(frozen=True)
class BroadConfig:
    tiers:tuple=TIERS
    per_tier:int=50
    seed:int=161926
    league_pages_per_division:int=3
    study:Config=field(default_factory=lambda:Config(player_count=300,allow_unverified_masters=False,random_seed=161926))
    def __post_init__(self):
        if not self.tiers or len(set(self.tiers))!=len(self.tiers) or not set(self.tiers)<=set(TIERS):raise ValueError('Only Bronze through Diamond are allowed.')
        if self.per_tier<1 or self.league_pages_per_division<1:raise ValueError('Positive quota/frame required.')
        if self.study.player_count!=self.per_tier*len(self.tiers):raise ValueError('Study player_count must equal rank quotas.')
    @property
    def identity(self):return json.loads(canonical({'schema':SCHEMA,'configuration':asdict(self)}))

class DiskState:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, request TEXT NOT NULL, value TEXT NOT NULL, received REAL NOT NULL)')
        self.db.commit()
    def get(self,key,default=None):
        r=self.db.execute('SELECT value FROM kv WHERE key=?',(key,)).fetchone()
        return json.loads(r[0]) if r else default
    def put(self,key,value):
        with self.db:self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',(key,canonical(value)))
    def response(self,request):
        row=self.db.execute('SELECT request,value,received FROM responses WHERE key=?',(hashed(request),)).fetchone()
        if row is None:return None
        try:
            if row[0]!=canonical(request):return None
            return json.loads(row[1]),row[2]
        except (ValueError,TypeError):return None
    def save_response(self,request,value,now):
        with self.db:self.db.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?,?)',(hashed(request),canonical(request),canonical(value),now))
    def close(self):self.db.close()

class DurableClient:
    """Key is header-only and memory-only. All persisted errors are sanitized codes.

    Infinite capped transient retries are interruptible; 401/403 stop immediately.
    New processes wait >=121s before an uncached request, on top of persisted
    app/method/service events and Retry-After. Other noncooperating collectors
    cannot be controlled: stop the original Master collector before enabling.
    """
    def __init__(self,disk,key,*,enabled=False,session=None,clock=time.time,sleep=time.sleep,startup_delay=121,status=print):
        self.disk=disk;self.enabled=enabled;self.key=key;self.http=session or requests.Session()
        self.clock=clock;self.sleep=sleep;self.startup_until=clock()+startup_delay;self.status=status
        self.metrics=Counter();self.rng=random.Random()
    def _wait(self,until):
        while until>self.clock():self.sleep(min(30,until-self.clock()))
    @staticmethod
    def limits(value):
        try:return [[int(x.split(':')[0]),float(x.split(':')[1])] for x in value.split(',') if x]
        except (ValueError,IndexError):raise ContractError('Malformed advertised rate-limit header.') from None
    def reserve(self,host,method):
        while True:
            now=self.clock();rates=self.disk.get('rates',{})
            wait=self.startup_until
            for name in ['app',method,'service']:
                key=host+':'+name;entry=rates.setdefault(key,{'events':[],'limits':[[20,1],[100,120]] if name=='app' else [],'blocked_until':0})
                longest=max([w for _,w in entry['limits']]+[120])
                entry['events']=[t for t in entry['events'] if t>now-longest-1]
                wait=max(wait,entry['blocked_until'])
                for cap,window in entry['limits']:
                    recent=[t for t in entry['events'] if t>now-window-.2]
                    if len(recent)>=cap:wait=max(wait,recent[-cap]+window+.2)
            if wait<=now:
                for name in ['app',method,'service']:rates[host+':'+name]['events'].append(now)
                self.disk.put('rates',rates);return
            self.disk.put('rates',rates)
            self.disk.put('network',{'status':'cooldown','until':wait})
            if wait-now>=5:self.status(f'Rate/startup cooldown: {wait-now:.0f}s; progress is saved.')
            self._wait(wait)
    def observe(self,host,method,headers):
        rates=self.disk.get('rates',{});now=self.clock()
        for prefix,name in [('App','app'),('Method',method),('Service','service')]:
            entry=rates.setdefault(host+':'+name,{'events':[],'limits':[],'blocked_until':0})
            limits=self.limits(headers.get('X-'+prefix+'-Rate-Limit',''))
            if limits:entry['limits']=limits
            for count,window in self.limits(headers.get('X-'+prefix+'-Rate-Limit-Count','')):
                caps=[cap for cap,w in entry['limits'] if w==window]
                # Header counts include calls made elsewhere with this key.
                if caps:
                    entry['events'].extend([now]*max(0,count-len([t for t in entry['events'] if t>now-window])))
                    if count>=min(caps):entry['blocked_until']=max(entry['blocked_until'],now+window+.2)
        self.disk.put('rates',rates)
    def request(self,host,path,*,method,params=None,validate=lambda x:True,cache=True):
        if host not in ('na1','americas') or not path.startswith(('/lol/league/v4/entries/','/lol/match/v5/matches/')):raise ContractError('Unexpected collector endpoint.')
        request={'host':host,'path':path,'params':params or {}}
        if cache:
            cached=self.disk.response(request)
            if cached is not None and validate(cached[0]):self.metrics['query_cache_hits']+=1;return cached
        if not self.enabled:raise CollectionDisabled('Collection is disabled; no Riot requests were made.')
        if not self.key:raise AuthenticationError('Set RIOT_API_KEY in the repository-root .env, then Run All again.')
        retry=self.disk.get('retry',{});attempt=int(retry.get('attempt',0)) if retry.get('request')==hashed(request) else 0
        while True:
            self.reserve(host,method);self.metrics['new_http_requests']+=1
            reason=None;headers={}
            try:
                response=self.http.get(f'https://{host}.api.riotgames.com{path}',params=params,
                    headers={'X-Riot-Token':self.key},timeout=(10,45))
                headers=response.headers;self.observe(host,method,headers);code=response.status_code
                if code in (401,403):
                    self.disk.put('network',{'status':'authentication_error','http_status':code})
                    raise AuthenticationError('Riot rejected the key (401/403). Refresh RIOT_API_KEY in root .env and Run All; saved work is intact.')
                if code==200:
                    try:value=response.json()
                    except ValueError:raise ContractError('Riot returned invalid JSON; work remains pending.') from None
                    if not validate(value):raise ContractError('Riot response does not match the documented schema; work remains pending.')
                    received=self.clock()
                    if cache:self.disk.save_response(request,value,received)
                    self.disk.put('retry',{});self.disk.put('network',{'status':'ok','at':received})
                    return value,received
                if code in (404,408,429) or 500<=code<=599:reason='HTTP_'+str(code)
                else:raise ContractError(f'HTTP {code}; fix endpoint/configuration before retrying. Player remains pending.')
            except (requests.Timeout,requests.ConnectionError,requests.exceptions.ChunkedEncodingError):reason='temporary_network_failure'
            attempt+=1;delay=min(300,2**min(attempt,8))+self.rng.uniform(0,2)
            if reason=='HTTP_429':
                raw=headers.get('Retry-After')
                try:delay=max(1,float(raw))+1
                except (TypeError,ValueError):
                    try:delay=max(1,parsedate_to_datetime(raw).timestamp()-self.clock())+1
                    except (ValueError,TypeError,AttributeError):delay=max(10,delay)
            until=self.clock()+delay
            rates=self.disk.get('rates',{});entry=rates.setdefault(host+':app',{'events':[],'limits':[[20,1],[100,120]],'blocked_until':0})
            entry['blocked_until']=max(entry['blocked_until'],until)
            self.disk.put('rates',rates)
            self.disk.put('retry',{'request':hashed(request),'attempt':attempt,'reason':reason,'next_retry_at':until})
            self.disk.put('network',{'status':'retryable','reason':reason,'until':until})
            self.metrics['transient_retries']+=1
            if attempt<=3 or attempt%10==0:self.status(f'{reason}: saved progress; retry {attempt}, cooldown {delay:.0f}s (interrupt to pause).')
            self._wait(until)

def valid_entries(value):
    return isinstance(value,list) and all(isinstance(x,dict) and isinstance(x.get('puuid'),str) and bool(x['puuid'])
        and isinstance(x.get('queueType'),str) and isinstance(x.get('tier'),str) and isinstance(x.get('rank'),str) for x in value)
def valid_ids(value,count):return isinstance(value,list) and len(value)<=count and all(match_id(x) for x in value)

def aborted_before_start(payload,mid,uid):
    """Recognize only the observed Match-V5 Abort_Unexpected non-game shape.

    Empty/partial normal games, other abort reasons, missing fields and mismatched
    identity remain contract failures. Never infer an outcome or timestamp.
    """
    if not isinstance(payload,dict):return False
    meta=payload.get('metadata');info=payload.get('info')
    if not isinstance(meta,dict) or not isinstance(info,dict):return False
    ids=meta.get('participants')
    return (meta.get('matchId')==mid and isinstance(ids,list) and len(ids)==10
        and all(isinstance(x,str) and bool(x) for x in ids) and len(set(ids))==10 and uid in ids
        and info.get('endOfGameResult')=='Abort_Unexpected'
        and info.get('participants')==[] and info.get('gameVersion')==''
        and type(info.get('gameStartTimestamp')) is int and info['gameStartTimestamp']==0
        and type(info.get('gameDuration')) is int and info['gameDuration']==0
        and type(info.get('gameEndTimestamp')) is int and info['gameEndTimestamp']>0
        and type(info.get('queueId')) is int)

def match_issues(payload,mid,uid=None):
    """Explain the match/record contract without logging values, IDs or headers.

    Diagnostics must not include arbitrary JSON keys, exception messages, URLs,
    or response snippets: even a malformed response could echo credentials.
    """
    issues=[]
    def check(ok,field,value,expected,*,present=True):
        if not ok:
            issue={'field':field,'actual_type':type(value).__name__ if present else 'missing','expected':expected}
            if isinstance(value,(str,list,dict)):issue['length']=len(value)
            issues.append(issue)
        return ok
    def field(obj,name,path,predicate,expected):
        value=obj.get(name)
        return check(predicate(value),path,value,expected,present=name in obj)
    def number(x):return type(x) is int or (type(x) is float and math.isfinite(x))
    if not check(isinstance(payload,dict),'response',payload,'object'):return issues
    for name in ('metadata','info'):
        field(payload,name,name,lambda x:isinstance(x,dict),'object')
    if issues:return issues
    metadata=payload['metadata'];info=payload['info']
    field(metadata,'matchId','metadata.matchId',lambda x:x==mid,'requested match ID')
    field(info,'gameVersion','info.gameVersion',lambda x:isinstance(x,str) and bool(x),'nonempty string')
    if not field(info,'participants','info.participants',lambda x:isinstance(x,list),'list'):return issues
    ps=info['participants']
    check(len(ps)==10,'info.participants',ps,'10 participants')
    for i,p in enumerate(ps):
        if check(isinstance(p,dict),f'info.participants[{i}]',p,'object'):
            field(p,'puuid',f'info.participants[{i}].puuid',lambda x:isinstance(x,str) and bool(x),'nonempty string')
    ids=[p.get('puuid') for p in ps if isinstance(p,dict)]
    ids_valid=len(ids)==len(ps) and all(isinstance(x,str) and x for x in ids)
    if ids_valid:check(len(set(ids))==10,'info.participants[].puuid',ids,'10 unique PUUIDs')
    mids=metadata.get('participants')
    if field(metadata,'participants','metadata.participants',lambda x:isinstance(x,list) and all(isinstance(i,str) for i in x),'list of PUUID strings') and ids_valid:
        check(set(mids)==set(ids),'metadata.participants',mids,'same PUUID set as info.participants')
    if uid is None:return issues
    start=info.get('gameStartTimestamp');end=info.get('gameEndTimestamp')
    start_ok=field(info,'gameStartTimestamp','info.gameStartTimestamp',lambda x:number(x) and x>0,'positive finite timestamp')
    if end is None:
        field(info,'gameDuration','info.gameDuration',lambda x:number(x) and x>0,'positive finite duration when end timestamp is absent')
    else:
        field(info,'gameEndTimestamp','info.gameEndTimestamp',lambda x:number(x) and (not start_ok or x>=start),'finite timestamp >= gameStartTimestamp')
    if not ids_valid:return issues
    focal=[(i,p) for i,p in enumerate(ps) if p['puuid']==uid]
    if not check(len(focal)==1,'info.participants[].puuid',ids,'exactly one focal player'):return issues
    i,p=focal[0];prefix=f'info.participants[{i}].'
    field(p,'championName',prefix+'championName',lambda x:isinstance(x,str) and bool(x),'nonempty string')
    field(p,'championId',prefix+'championId',lambda x:type(x) is int,'integer')
    field(p,'win',prefix+'win',lambda x:isinstance(x,bool),'boolean')
    field(p,'teamId',prefix+'teamId',lambda x:type(x) is int and x in (100,200),'100 or 200')
    field(info,'queueId','info.queueId',lambda x:type(x) is int,'integer')
    if not issues:
        # Optional nested fields can also make projection unusable. Do not echo
        # exception text, which may contain data controlled by the response.
        try:record=player_record(payload,uid)
        except (TypeError,ValueError,KeyError,AttributeError,OverflowError):record=None
        check(record is not None,'player_record',payload,'convertible player record (including optional role/perks fields)')
    return issues

def master_ids(project):
    checkpoint=Path(project)/'hlr_model/data/runs/patch_16_19_seed_1619/metadata/checkpoint.json'
    if not checkpoint.is_file():raise ValueError('Cannot locate the existing Master cohort checkpoint; supply explicit excluded_players for a fixture or locate it before enabling collection.')
    with checkpoint.open() as f:payload=json.load(f)
    ids=[row['puuid'] for row in payload['identity']['cohort']]
    if not ids:raise ValueError('Master cohort is empty; inspect the existing checkpoint.')
    return sorted(set(ids))

class BroadCollector:
    def __init__(self,project,run_dir,config=None,*,enabled=None,api_key=None,session=None,
                 excluded_players=None,legacy_roots=None,clock=time.time,sleep=time.sleep,startup_delay=121,status=print):
        self.project=Path(project).resolve();self.run_dir=Path(run_dir).resolve();self.config=config or BroadConfig()
        self.lock_path=self.project/'hlr_model/data/.riot_collection.lock'
        self.clock=clock;self.status=status;self.metrics=Counter()
        default_roots=[self.project/'hlr_model/data/runs/patch_16_19_seed_1619/raw/matches',
            self.project/'data/otp_raw_matches',self.project/'lpm_model/data/otp_raw_matches']
        self.roots=[self.run_dir/'raw/matches']+list(map(Path,default_roots if legacy_roots is None else legacy_roots))
        with writer_lock(self.lock_path):
            self.disk=DiskState(self.run_dir/'metadata/checkpoint.sqlite')
            self.state=self.disk.get('state')
            if self.state is None:
                self.state={'identity':self.config.identity,'created_at':utc(clock()),'enabled':False,
                    'excluded_master_ids':master_ids(self.project) if excluded_players is None else sorted(set(excluded_players)),
                    'frame_pages':{},'candidate_order':{},'cursor':{t:0 for t in self.config.tiers},'selected':{t:[] for t in self.config.tiers},
                    'players':{},'failures':[],'stage':'not_started','exports':[]}
            elif self.state['identity']!=self.config.identity:
                self.disk.close();raise ValueError('Saved schema/configuration differs. Restore the original settings or choose a new run folder; nothing was reset or merged.')
            for player in self.completed():self.reconcile_player(player)
            if enabled is not None:self.state['enabled']=bool(enabled)
            self.save()
        self.client=DurableClient(self.disk,api_key,enabled=self.state['enabled'],session=session,clock=clock,sleep=sleep,startup_delay=startup_delay,status=status)
    def save(self):self.disk.put('state',self.state)
    def close(self):self.disk.close()
    def completed(self,tier=None):
        players=[p for p in self.state['players'].values() if p['status']=='completed' and (tier is None or p['assigned_tier']==tier)]
        return sorted(players,key=lambda p:(self.config.tiers.index(p['assigned_tier']),p.get('candidate_order',0),p['puuid']))
    @property
    def done(self):return all(len(self.completed(t))==self.config.per_tier for t in self.config.tiers)
    def progress(self):
        rows=[]
        for tier in self.config.tiers:
            players=[p for p in self.state['players'].values() if p['assigned_tier']==tier]
            counts=Counter(p['status'] for p in players)
            rows.append({'tier':tier,'goal':self.config.per_tier,'attempted':len(players),
                'excluded':counts['excluded'],'pending':counts['pending'],'incomplete':counts['incomplete'],'completed':counts['completed'],
                'reserves':max(0,len(self.state['candidate_order'].get(tier,[]))-self.state['cursor'][tier]-(1 if any(p['status'] in ('pending','incomplete') for p in players) else 0))})
        return pd.DataFrame(rows)
    def show(self):
        self.status(f"Completed eligible players: {len(self.completed())} / {self.config.study.player_count}; stage={self.state['stage']}; remaining={self.config.study.player_count-len(self.completed())}")
        self.status(' | '.join(f'{t} {len(self.completed(t))}/{self.config.per_tier}' for t in self.config.tiers))
        self.status(f'HTTP/cache session: {dict(self.client.metrics)}; raw cache: {dict(self.metrics)}; network: {self.disk.get("network",{})}')
    def discover(self):
        if self.state['candidate_order']:return
        self.state['stage']='rank-frame discovery';self.save()
        excluded=set(self.state['excluded_master_ids']);seen=set();buckets={}
        for tier in self.config.tiers:
            for division in DIVISIONS:
                label=tier+':'+division;rows=[]
                for page in range(1,self.config.league_pages_per_division+1):
                    key=f'{label}:{page}'
                    if key in self.state['frame_pages']:envelope=self.state['frame_pages'][key]
                    else:
                        value,received=self.client.request('na1',f'/lol/league/v4/entries/{QUEUE}/{tier}/{division}',method='league_entries',params={'page':page},validate=valid_entries)
                        envelope={'entries':[{'puuid':r['puuid'],'tier':r['tier'],'rank':r['rank'],'queueType':r['queueType']} for r in value],
                                  'received_at':utc(received),'page':page}
                        # HTTP response already durable; crash here reuses exact cached query.
                        self.state['frame_pages'][key]=envelope;self.save()
                    entries=envelope['entries']
                    if not entries:break
                    for r in entries:
                        uid=r['puuid']
                        if uid in seen or uid in excluded or r['tier']!=tier or r['rank']!=division or r['queueType']!=QUEUE:continue
                        seen.add(uid);rows.append({'puuid':uid,'assigned_tier':tier,'frame_division':division,
                            'frame_page':page,'frame_observed_at_utc':envelope['received_at']})
                rows.sort(key=lambda r:hashed([self.config.seed,tier,division,r['puuid']]))
                buckets[label]=rows
        order={}
        for tier in self.config.tiers:
            divisions=list(DIVISIONS);random.Random(str(self.config.seed)+tier).shuffle(divisions)
            order[tier]=[];i=0
            while any(i<len(buckets[tier+':'+d]) for d in divisions):
                for d in divisions:
                    if i<len(buckets[tier+':'+d]):order[tier].append(buckets[tier+':'+d][i])
                i+=1
        for tier,entries in order.items():
            for index,entry in enumerate(entries):entry['candidate_order']=index
        self.state['candidate_order']=order;self.state['frame_frozen_at']=utc(self.clock());self.save()
    @staticmethod
    def valid_match(payload,mid):
        return not match_issues(payload,mid)
    @staticmethod
    def validate_record(payload,mid,uid):
        if aborted_before_start(payload,mid,uid):return True
        issues=match_issues(payload,mid,uid)
        if issues:raise MatchContractError(mid,issues)
        return True
    def record(self,mid,uid,download=False):
        if not match_id(mid):raise ContractError('Invalid NA match ID.')
        for root in self.roots:
            path=root/f'{mid}.json'
            if not path.is_file():continue
            try:
                with path.open() as f:payload=json.load(f)
                if aborted_before_start(payload,mid,uid):
                    self.metrics['raw_cache_hits']+=1
                    if not download:return None
                    raise AbortedMatch(mid)
                if not match_issues(payload,mid,uid):
                    record=player_record(payload,uid)
                    if record is not None:self.metrics['raw_cache_hits']+=1;return record
            except (OSError,ValueError,TypeError,KeyError):pass
        if not download:return None
        payload,_=self.client.request('americas',f'/lol/match/v5/matches/{mid}',method='match_detail',
            validate=lambda p:self.validate_record(p,mid,uid),cache=False)
        path=self.roots[0]/f'{mid}.json'
        if path.exists():path.rename(path.with_name(path.name+f'.invalid-{time.time_ns()}'))
        atomic_write(path,payload);self.metrics['new_raw_matches']+=1
        if aborted_before_start(payload,mid,uid):raise AbortedMatch(mid)
        return player_record(payload,uid)
    def query(self,player,stage):
        config=self.config.study
        q=player.setdefault(stage+'_query',{'offset':0,'ids':[],'done':False,'page_index':0,'request_count':None})
        count=q.get('request_count')
        if count is None:
            count=config.page_size if stage=='target' else min(config.page_size,config.historical_depth-len(player.get('history_ids',[])))
            q['request_count']=count;self.save()
        params={'queue':config.queue,'start':q['offset'],'count':count,
            'startTime':config.target_search_start if stage=='target' else config.season_start,
            'endTime':config.target_search_end if stage=='target' else player['anchor_timestamp']//1000+1}
        page,_=self.client.request('americas',f'/lol/match/v5/matches/by-puuid/{quote(player["puuid"],safe="")}/ids',method='match_ids',params=params,
            validate=lambda x:valid_ids(x,count))
        if page and not set(page)-set(q['ids']) and q['page_index']==0:raise ContractError('Repeated match-ID page made no progress; player stays pending.')
        return q,page,count
    def targets(self,p):
        if p.get('target_complete'):return
        self.state['stage']='target discovery';self.save()
        q=p.setdefault('target_query',{'offset':0,'ids':[],'done':False,'page_index':0,'request_count':None})
        while not q['done']:
            q,page,count=self.query(p,'target')
            q['ids']=list(dict.fromkeys(q['ids']+page));q['offset']+=len(page)
            q['done']=len(page)<count;q['request_count']=None;self.save()
        p.setdefault('target_ids',[]);p.setdefault('target_checked',[]);p.setdefault('target_exclusions',{})
        for mid in q['ids']:
            # Reconcile even checked files: never trust statuses over missing/corrupt payloads.
            try:r=self.record(mid,p['puuid'],download=True)
            except AbortedMatch:
                reason='aborted_before_start'
                self.status(f'Match {mid}: verified Abort_Unexpected before start; participants=list length=0, gameVersion=str length=0, gameStartTimestamp=int zero, gameDuration=int zero; excluded as a non-game.')
            else:
                if r['queue']!=self.config.study.queue:reason='wrong_queue'
                elif r['patch']!=self.config.study.target_patch:reason='wrong_patch'
                elif not self.config.study.target_search_start*1000<=r['timestamp']<self.config.study.target_search_end*1000:reason='outside_target_window'
                else:reason=None
            if mid not in p['target_checked']:
                p['target_checked'].append(mid)
                if reason:p['target_exclusions'][reason]=p['target_exclusions'].get(reason,0)+1
            if reason is not None and mid in p['target_ids']:p['target_ids'].remove(mid)
            if reason is None and mid not in p['target_ids']:p['target_ids'].append(mid)
            self.save()
        p['target_complete']=True
        if not p['target_ids']:self.exclude(p,'no_valid_patch_targets')
        else:
            p['anchor_timestamp']=min(self.record(m,p['puuid'])['timestamp'] for m in p['target_ids']);self.save()
    def histories(self,p):
        if p.get('history_complete'):return
        self.state['stage']='initial history';self.save()
        p.setdefault('history_ids',[]);p.setdefault('history_exclusions',{})
        q=p.setdefault('history_query',{'offset':0,'ids':[],'done':False,'page_index':0,'request_count':None})
        stop=None
        while len(p['history_ids'])<self.config.study.historical_depth and not q['done']:
            q,page,count=self.query(p,'history')
            while q['page_index']<len(page):
                mid=page[q['page_index']]
                try:r=self.record(mid,p['puuid'],download=True)
                except AbortedMatch:
                    self.status(f'Match {mid}: verified Abort_Unexpected before start; participants=list length=0, gameVersion=str length=0, gameStartTimestamp=int zero, gameDuration=int zero; excluded as a non-game.')
                    p['history_exclusions']['aborted_before_start']=p['history_exclusions'].get('aborted_before_start',0)+1
                    q['page_index']+=1;self.save();continue
                if r['timestamp']<self.config.study.season_start*1000:stop='season_boundary_reached'
                elif r['queue']!=self.config.study.queue:reason='wrong_queue'
                elif r['timestamp']>=p['anchor_timestamp'] or r['end_timestamp']>=p['anchor_timestamp']:reason='not_completed_before_anchor'
                else:
                    reason=None
                    if mid not in p['history_ids']:p['history_ids'].append(mid)
                if stop:break
                if reason:p['history_exclusions'][reason]=p['history_exclusions'].get(reason,0)+1
                q['page_index']+=1;self.save()
                if len(p['history_ids'])>=self.config.study.historical_depth:stop='historical_depth_reached';break
            if stop:break
            q['ids']=list(dict.fromkeys(q['ids']+page));q['offset']+=len(page)
            q['done']=len(page)<count;q['page_index']=0;q['request_count']=None;self.save()
        records=[self.record(m,p['puuid']) for m in p['history_ids']]
        if any(r is None for r in records):raise ContractError('A required history payload is missing; resume retrieval before completing.')
        records.sort(key=lambda r:(r['timestamp'],r['match_id']),reverse=True)
        p['history_ids']=[r['match_id'] for r in records[:self.config.study.historical_depth]]
        p['history_complete']=True;p['history_stop_reason']=stop or 'api_exhausted_within_season'
        p['history_exhausted']=stop is None and q['done']
        self.save()
    def exclude(self,p,reason):
        p['status']='excluded';p['excluded_reason']=reason
        selected=self.state['selected'][p['assigned_tier']]
        if p['puuid'] in selected:selected.remove(p['puuid'])
        self.save()
    def process(self,p):
        uid=p['puuid'];p['status']='pending';self.save()
        if not p.get('rank_verified'):
            self.state['stage']='rank verification';self.save()
            entries,received=self.client.request('na1',f'/lol/league/v4/entries/by-puuid/{quote(uid,safe="")}',method='rank_by_puuid',validate=valid_entries)
            solo=[r for r in entries if r['puuid']==uid and r['queueType']==QUEUE]
            if len(solo)>1:raise ContractError('Ambiguous Solo/Duo rank response.')
            if not solo or solo[0]['tier']!=p['assigned_tier'] or solo[0]['rank'] not in DIVISIONS:
                self.exclude(p,'no_allowed_rank_in_frozen_tier');return
            r=solo[0];p.update(rank_verified=True,tier=r['tier'],division=r['rank'],queue=QUEUE,rank_checked_at_utc=utc(received))
            if uid not in self.state['selected'][p['assigned_tier']]:self.state['selected'][p['assigned_tier']].append(uid)
            self.save()
        self.targets(p)
        if p['status']=='excluded':return
        self.histories(p)
        self.reconcile_player(p)
        if p.get('target_complete') and p.get('history_complete'):
            p['status']='completed';p['completed_at_utc']=utc(self.clock());self.save()
    def reconcile_player(self,p):
        if p.get('target_complete') and any(self.record(m,p['puuid']) is None for m in p.get('target_ids',[])):
            p['target_complete']=False;p['status']='incomplete'
        if p.get('history_complete') and any(self.record(m,p['puuid']) is None for m in p.get('history_ids',[])):
            # Reprocess the same history query from cached pages, not older substitutes.
            p['history_complete']=False;p['history_ids']=[];p.pop('history_query',None);p['status']='incomplete'
    def collect(self):
        if not self.state['enabled']:self.status('Collection disabled; no Riot API calls. Set RUN_COLLECTION=True once to enable.');return
        with writer_lock(self.lock_path):
            # Reload under the long-lived lock so a second instance cannot overwrite newer state.
            self.state=self.disk.get('state')
            if not self.state['enabled']:return
            self.client.enabled=True
            for p in self.completed():self.reconcile_player(p)
            self.save()
            if self.done:self.state['stage']='complete';self.save();self.show();return
            current=None
            try:
                self.discover()
                for tier in self.config.tiers:
                    order=self.state['candidate_order'][tier]
                    while len(self.completed(tier))<self.config.per_tier:
                        i=self.state['cursor'][tier]
                        # A corrupt completed payload must resume that selected player, not consume a reserve.
                        unresolved=[p for p in self.state['players'].values() if p['assigned_tier']==tier and p['status'] in ('pending','incomplete')]
                        if unresolved:current=unresolved[0]
                        else:
                            if i>=len(order):raise FrameExhausted(f'{tier} frozen frame exhausted before quota. No completion claimed. A larger frame needs a separate configured run; existing data stay intact.')
                            entry=order[i];uid=entry['puuid']
                            current=self.state['players'].setdefault(uid,dict(entry,status='pending'))
                        if current['status'] not in ('completed','excluded'):self.process(current)
                        if i<len(order) and order[i]['puuid']==current['puuid'] and current['status'] in ('completed','excluded'):
                            self.state['cursor'][tier]=i+1;self.save()
                        self.show()
                if self.done:self.state['stage']='complete';self.save();self.show()
            except BaseException as error:
                if current is not None and current['status'] not in ('excluded','completed'):current['status']='incomplete'
                # Never serialize exception text: requests exceptions may contain URLs/headers.
                failure={'type':type(error).__name__,'stage':self.state['stage'],'at_utc':utc(self.clock())}
                if isinstance(error,MatchContractError):failure['diagnostics']=error.diagnostics
                self.state['failures'].append(failure)
                self.save();raise
    def export(self):
        with writer_lock(self.lock_path):
            self.state=self.disk.get('state')
            for p in self.completed():self.reconcile_player(p)
            self.save()
            if not self.done:raise CollectionDisabled('Exports require all six quotas and all required payloads complete; no incomplete dataset exported.')
            rows=[];history=[];targets=[];coverage=[]
            for p in self.completed():
                uid=p['puuid'];h=[self.record(m,uid) for m in p['history_ids']];t=[self.record(m,uid) for m in p['target_ids']]
                history.extend(h);targets.extend(t)
                rows.append({'puuid':uid,'cohort_id':'broad_na_16_19','selection_source':'bounded_league_v4_frame',
                    'cohort_order':len(rows),'seed_tier':p['tier'],'rank':p['tier'],'division':p['division'],'rank_queue':QUEUE,
                    'rank_observed_at_utc':p['rank_checked_at_utc'],'rank_evidence_source':'league-v4/entries/by-puuid',
                    'historical_masters_verified':False,'rank_limitation':'Collection-time allowed rank; patch-16.19 historical rank unverified',
                    'selection_seed':self.config.seed,'target_patch':self.config.study.target_patch})
                coverage.append({'puuid':uid,'target_discovery_complete':True,'history_discovery_complete':True,
                    'available_historical_matches':len(h),'history_shortfall':self.config.study.historical_depth-len(h),
                    'history_exhausted':p.get('history_exhausted',False),'history_stop_reason':p['history_stop_reason'],
                    'history_exclusions_json':canonical(p['history_exclusions']),
                    'genuinely_short_history':len(h)<self.config.study.historical_depth,'history_incomplete':False,
                    'target_matches':len(t),'missing_required_payloads':0,'candidate_matches':len(p['target_query']['ids']),
                    'anchor_timestamp':p['anchor_timestamp'],'target_exclusions_json':canonical(p['target_exclusions'])})
            cohort=pd.DataFrame(rows);h=pd.DataFrame(history,columns=RECORD_COLUMNS).drop_duplicates(['puuid','match_id'])
            t=pd.DataFrame(targets,columns=RECORD_COLUMNS).drop_duplicates(['puuid','match_id']);cover=pd.DataFrame(coverage)
            cohort=cohort.merge(cover,on='puuid',validate='one_to_one')
            model=expanding_features(h,t,self.config.study).merge(cohort,on='puuid',validate='many_to_one')
            # Pandas otherwise leaves an entirely empty-history timestamp column as object/None.
            model['prior_latest_end_timestamp']=pd.to_numeric(model['prior_latest_end_timestamp'],errors='raise')
            summary=quality_report(cohort,h,t,model,cover,self.config.study)
            fingerprint=hashed({'players':[(p['puuid'],p['target_ids'],p['history_ids']) for p in self.completed()], 'identity':self.config.identity})
            # An interrupted export has no completion marker and is never reused.
            for path in (self.run_dir/'processed').glob('*/collection_metadata.json'):
                try:
                    value=json.loads(path.read_text())
                    if value.get('export_fingerprint')==fingerprint and value.get('output_sha256'):
                        if all((path.parent/name).is_file() and hashlib.sha256((path.parent/name).read_bytes()).hexdigest()==digest for name,digest in value['output_sha256'].items()):return path.parent
                except (OSError,ValueError):continue
            out=self.run_dir/'processed'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            out.mkdir(parents=True,exist_ok=False)
            for name,frame in [('player_cohort',cohort),('historical_matches',h),('target_matches',t),('longitudinal_modeling',model),('player_coverage',cover)]:
                frame.to_parquet(out/(name+'.parquet'),index=False)
            output_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in out.glob('*.parquet')}
            atomic_write(out/'collection_metadata.json',dict(summary,config=asdict(self.config.study),broad_config=asdict(self.config),output_sha256=output_sha256,
                search_end=self.config.study.target_search_end,cohort_id='broad_na_16_19',export_fingerprint=fingerprint,
                sampling_frame='First configured pages per allowed tier/division; seeded hash within division; round-robin divisions. Not population representative.',
                rank_limitation='Collection-time rank, not verified patch-period rank',failures=self.state['failures'],
                run_created_at_utc=self.state['created_at'],api_metrics_this_session=dict(self.client.metrics),cache_metrics_this_session=dict(self.metrics)))
            self.state['exports'].append(str(out));self.save();return out
