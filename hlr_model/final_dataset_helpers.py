"""Pinned longitudinal preparation. No EDA, modeling, thresholds or predictor matrix.

Only optional_collection_cleanup may use the network, and only through the
existing bounded unresolved-NA retry helper. All preparation functions are offline.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pandas as pd

from hlr_model import lolalytics_matchup_scraper as scraper

SNAPSHOT = '20261018T195205790466Z'
COLLECTION = '20261020T164500_repaired'
KEY = ['puuid', 'match_id']
MATCH_KEY = ['lane', 'champion_id', 'opponent_champion_id']
REGIONS = ('global', 'na')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def paths(root):
    root = Path(root).resolve()
    return (root/'hlr_model/data/runs/patch_16_19_seed_1619/processed'/SNAPSHOT,
            root/'hlr_model/data/lolalytics/runs'/COLLECTION)


def optional_collection_cleanup(root):
    """One authorized pass, max 12 NA unresolved pages, never a full collection."""
    _, run = paths(root)
    if (run/'collector.lock').exists():
        raise RuntimeError('Collection is running; preparation needs a stable source run')
    sessions = scraper.read_events(run/'sessions.jsonl')
    previous = [r for r in sessions if r.get('mode') == 'retry_unresolved_only'
                and 'na' in r.get('retry_regions', [])]
    unresolved = pd.read_csv(run/'unresolved_pages.csv')
    unresolved = unresolved.loc[unresolved.region.eq('na')]
    if unresolved.empty or previous:
        return {'action':'no_unresolved_na_pages' if unresolved.empty else 'skipped_previously_retried',
                'unresolved_na_pages_before':len(unresolved),
                'last_retry_session':previous[-1] if previous else None}
    page_summary = pd.read_csv(run/'page_summary_by_region.csv')
    if not page_summary.all_planned_pages_attempted.all():
        raise ValueError('Full collection is incomplete; preparation will not broaden collection')
    with redirect_stdout(io.StringIO()):
        context = scraper.prepare_collection(root, scraper.Config(), run_id=COLLECTION)
        before_events = scraper.read_events(run/'request_events.jsonl')
        prior_success = {r['key_id']:scraper.canonical(r) for r in scraper.load_results(context).values()
                         if r['status'] in scraper.TERMINAL}
        scraper.retry_unresolved_pages(context, max_pages=12, regions=('na',))
        after = scraper.load_results(context)
    assert all(scraper.canonical(after[k]) == value for k,value in prior_success.items())
    events = scraper.read_events(run/'request_events.jsonl')[len(before_events):]
    attempts = [r for r in events if r.get('event')=='attempt_started']
    assert set(r['requested_url'] for r in attempts) <= set(unresolved.batch_url)
    assert len({r['requested_url'] for r in attempts}) <= 12 and len(attempts) <= 12*context['config'].max_attempts
    remaining = pd.read_csv(run/'unresolved_pages.csv')
    return {'action':'one_bounded_na_retry', 'max_pages':12,
            'unresolved_na_pages_before':len(unresolved),
            'unresolved_na_pages_after':int(remaining.region.eq('na').sum()),
            'actual_http_attempts':len(attempts), 'successful_records_preserved':len(prior_success)}


def load_inputs(root):
    root = Path(root).resolve()
    source, run = paths(root)
    names = ['longitudinal_modeling','historical_matches','target_matches','player_coverage','player_cohort']
    frames = {n:pd.read_parquet(source/f'{n}.parquet') for n in names}
    for n in ['matchup_lookup','request_plan','decision_request_mapping']:
        frames[n] = pd.read_parquet(run/f'{n}.parquet')
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['snapshot']==SNAPSHOT and manifest['run_id']==COLLECTION
    assert manifest['identity']['filters']=={'patch':'16.19','rank':'master_plus','queue':'ranked'}
    assert manifest['identity']['regions']==list(REGIONS)
    for n, expected in manifest['identity']['input_sha256'].items():
        assert sha256(source/f'{n}.parquet')==expected, f'Pinned input hash differs: {n}'
    input_files = [*(source.glob('*.parquet')), source/'collection_metadata.json',
                   *(run/f'{n}.parquet' for n in ['matchup_lookup','request_plan','decision_request_mapping']),
                   *(run/n for n in ['manifest.json','quality_summary.json','page_summary_by_region.csv',
                                     'unresolved_pages.csv','request_events.jsonl','sessions.jsonl','matchup_checkpoint.jsonl'])]
    provenance = {str(p.relative_to(root)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in input_files}
    frames.update(root=root, source_dir=source, run_dir=run, manifest=manifest, input_files=provenance,
                  collection_metadata=json.loads((source/'collection_metadata.json').read_text()))
    return frames


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def validate_sources(data):
    model, history, target = [data[n] for n in ['longitudinal_modeling','historical_matches','target_matches']]
    for name in ['longitudinal_modeling','historical_matches','target_matches']:
        frame=data[name]
        require(not frame.duplicated(KEY).any(), f'Duplicate keys: {name}')
        require(frame.queue.eq(420).all() and frame.end_timestamp.ge(frame.timestamp).all(), f'Queue/times: {name}')
        require(frame.win.isin([0,1]).all(), f'Outcome domain: {name}')
    pd.testing.assert_frame_equal(model.set_index(KEY)[target.columns.drop(KEY)].sort_index(),
                                  target.set_index(KEY).sort_index(), check_dtype=False)
    require(target.patch.eq('16.19').all(), 'Wrong target patch')
    require(not set(map(tuple,history[KEY].values)) & set(map(tuple,target[KEY].values)), 'History/target overlap')
    anchors=target.groupby('puuid').timestamp.min()
    require(history.end_timestamp.lt(history.puuid.map(anchors)).all(), 'Initial history reaches targets')
    for field in data['player_coverage'].columns.drop('puuid'):
        comparison=model[['puuid',field]].drop_duplicates().merge(data['player_coverage'][['puuid',field]],
                    on='puuid',validate='one_to_one',suffixes=('_model','_coverage'))
        a,b=comparison[field+'_model'],comparison[field+'_coverage']
        require((a.eq(b) | (a.isna() & b.isna())).all(), f'Coverage metadata disagrees: {field}')
    plan, lookup=data['request_plan'],data['matchup_lookup']
    require(not lookup.key_id.duplicated().any() and set(lookup.key_id)==set(plan.key_id), 'Lookup/plan keys differ')
    require(not lookup.duplicated(['region',*MATCH_KEY]).any(), 'Duplicate regional matchup IDs')
    for field in ['patch','rank','region','queue']:
        require(lookup[field].eq(lookup['requested_'+field]).all(), f'Lookup requested filter conflict: {field}')
    valid=scraper.valid_rate_mask(lookup)
    require(valid.eq(lookup.status.eq('ok')).all(), 'Published success is not fully verified')
    require(lookup.loc[valid,'verified_queue_id'].eq(420).all(), 'Wrong verified queue ID')
    require(lookup.loc[~valid,[c for c in scraper.RATE_FIELDS if c != 'tier_average_winrate_pct']].isna().all().all(), 'Invalid statuses contain numeric statistics')
    context={'run_dir':data['run_dir'],'run_id':COLLECTION,'input_snapshot':SNAPSHOT}
    tasks={r['key_id']:r for r in plan.to_dict('records')}
    for row in lookup.to_dict('records'):
        row={k:None if pd.isna(v) else v for k,v in row.items()}
        scraper.validate_checkpoint_provenance(row,tasks[row['key_id']],context)
    # Named source-table rows independently confirm raw matchup values/counts.
    for row in lookup.loc[valid].itertuples():
        evidence=json.loads(row.matchup_evidence)
        require(evidence['opposing_lane']==scraper.ROLE_SLUGS[row.lane], 'Opposing lane evidence differs')
        require(evidence['row']['id']==row.opponent_champion_id, 'Wrong opponent evidence')
        require(evidence['row']['wr']==row.raw_matchup_winrate_pct and evidence['row']['n']==row.matchup_games,
                'Rate/sample differs from source row')
    require(lookup.snapshot_id.eq(COLLECTION).all() and lookup.input_snapshot.eq(SNAPSHOT).all(), 'Mixed source runs')
    require(lookup.loc[valid,'rate_units'].str.startswith('percent').all(), 'Wrong population rate units')
    data['valid_lookup']=lookup.assign(valid_rate=valid)
    return {'source_keys_outcomes_queues_and_timing':'PASS', 'regional_lookup_filters_units_and_cached_provenance':'PASS',
            'lookup_valid_keys_by_region':lookup.loc[valid].groupby('region').size().to_dict()}


def regional_join(frame, lookup, champion_column, prefix=''):
    """Only exact lane/candidate/opponent IDs; one verified row per region/key."""
    out=frame.copy()
    for region in REGIONS:
        source=lookup.loc[lookup.region.eq(region)].copy()
        # Preserve every source field, including identities, status and evidence.
        columns=[c for c in source if c not in ['region',*MATCH_KEY]]
        renamed={c:prefix+region+'_'+c for c in columns}
        right=source[MATCH_KEY+columns].rename(columns={'champion_id':champion_column,**renamed})
        before=len(out)
        out=out.merge(right,on=['lane',champion_column,'opponent_champion_id'],how='left',validate='many_to_one',sort=False)
        require(len(out)==before,'Regional join multiplied rows')
        name=prefix+region+'_'
        valid=out[name+'valid_rate'].eq(True)
        out[name+'valid_rate']=valid
        known=out.lane.isin(scraper.ROLE_SLUGS) & out.opponent_champion_id.notna() & ~out.matchup_missing
        unmatched=out[name+'status'].isna()
        out.loc[unmatched,name+'status']=np.where(known[unmatched], 'not_in_collection_plan','unidentified_lane_opponent')
        out[name+'raw_matchup_winrate_probability']=out[name+'raw_matchup_winrate_pct'].div(100).where(valid)
        # Numeric statistics must never be promoted from unresolved/missing rows.
        require(out.loc[~valid,name+'raw_matchup_winrate_probability'].isna().all(), 'Invalid probability filled')
    return out


def experience(counts,wins,role_counts,role_wins,cid,lane,total,latest,role_latest):
    n,w=counts[cid],wins[cid]
    rn,rw=role_counts[(lane,cid)],role_wins[(lane,cid)]
    return {'all_role_prior_games':n,'all_role_prior_wins':w,
        'all_role_prior_wr':w/n if n else np.nan,'all_role_prior_wr_beta11':(w+1)/(n+2),
        'all_role_prior_usage_share':n/total if total else np.nan,
        'same_role_prior_games':rn,'same_role_prior_wins':rw,
        'same_role_prior_wr':rw/rn if rn else np.nan,'same_role_prior_wr_beta11':(rw+1)/(rn+2),
        'zero_all_role_experience':n==0,'zero_same_role_experience':rn==0,
        'latest_all_role_prior_end_timestamp':latest.get(cid),
        'latest_same_role_prior_end_timestamp':role_latest.get((lane,cid))}


def build_candidates(data):
    model,history,target=[data[n] for n in ['longitudinal_modeling','historical_matches','target_matches']]
    mapping=data['decision_request_mapping']
    require(set(mapping.region)==set(REGIONS),'Candidate mapping lacks a region')
    nonregional=[c for c in mapping if c not in ['region','key_id']]
    a=mapping.loc[mapping.region.eq('global'),nonregional].sort_values(nonregional[:1]+['match_id','candidate_champion_id','request_kind']).reset_index(drop=True)
    b=mapping.loc[mapping.region.eq('na'),nonregional].sort_values(nonregional[:1]+['match_id','candidate_champion_id','request_kind']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    require(not a.duplicated(KEY+['candidate_champion_id']).any(), 'Mapping repeats a candidate within a decision')
    mapped={k:g for k,g in a.groupby(KEY,sort=False)}
    histories={p:g for p,g in history.groupby('puuid')}
    original=model.set_index(KEY)
    names=pd.concat([history[['champion_id','champion']],target[['champion_id','champion']]]).drop_duplicates()
    require(not names.champion_id.duplicated().any(),'Champion ID has inconsistent names')
    names=names.set_index('champion_id').champion.to_dict()
    rows=[]; decisions=[]
    for player,group in target.groupby('puuid',sort=True):
        pool=pd.concat([histories.get(player,history.iloc[:0]),group]).sort_values(['end_timestamp','match_id'])
        require(not pool.duplicated(KEY).any(),'Duplicate preserved history records')
        available=list(pool.itertuples(index=False))
        counts,wins,rc,rw=Counter(),Counter(),Counter(),Counter()
        latest,role_latest={},{}
        cursor=total=total_wins=0
        for t in group.sort_values(['timestamp','match_id']).itertuples(index=False):
            while cursor<len(available) and available[cursor].end_timestamp<t.timestamp:
                prior=available[cursor]
                require(prior.timestamp<t.timestamp,'Future history entered counters')
                counts[prior.champion_id]+=1; wins[prior.champion_id]+=prior.win
                if prior.lane in scraper.ROLE_SLUGS:
                    rc[(prior.lane,prior.champion_id)]+=1; rw[(prior.lane,prior.champion_id)]+=prior.win
                latest[prior.champion_id]=prior.end_timestamp
                if prior.lane in scraper.ROLE_SLUGS:
                    role_latest[(prior.lane,prior.champion_id)]=prior.end_timestamp
                total+=1; total_wins+=prior.win; cursor+=1
            main=min(counts,key=lambda c:(-counts[c],c)) if counts else None
            role_ids={cid for (lane,cid),n in rc.items() if lane==t.lane and n>0}
            role_main=min(role_ids,key=lambda c:(-rc[(t.lane,c)],c)) if role_ids else None
            saved=original.loc[(player,t.match_id)]
            require(total==saved.prior_total_games and total_wins==saved.prior_total_wins,'Pregame total counts differ')
            require(main==saved.prior_main_champion_id,'Existing all-role main differs')
            observed_exp=experience(counts,wins,rc,rw,t.champion_id,t.lane,total,latest,role_latest)
            for new,old in [('all_role_prior_games','prior_champion_games'),('all_role_prior_wins','prior_champion_wins'),
                            ('all_role_prior_wr','prior_champion_wr'),('all_role_prior_wr_beta11','prior_champion_wr_beta11'),
                            ('all_role_prior_usage_share','prior_champion_usage_share')]:
                v,w=observed_exp[new],saved[old]
                require((pd.isna(v) and pd.isna(w)) or np.isclose(v,w),'Existing chosen proficiency differs')
            known=not t.matchup_missing and t.lane in scraper.ROLE_SLUGS and pd.notna(t.opponent_champion_id)
            expected=role_ids-{t.champion_id,int(t.opponent_champion_id)} if known else set()
            evidence=mapped.get((player,t.match_id),a.iloc[:0])
            alternatives=set(evidence.loc[evidence.request_kind.eq('alternative'),'candidate_champion_id'])
            require(alternatives==expected,'Saved alternatives differ from strictly prior same-role eligibility')
            observed_mapping=evidence.loc[evidence.request_kind.eq('observed')]
            require(set(observed_mapping.candidate_champion_id)==({t.champion_id} if known else set()),'Observed mapping differs')
            d={'puuid':player,'match_id':t.match_id,'pregame_same_role_main_id':role_main,
               'pregame_same_role_main_champion':names.get(role_main),'reconstructed_prior_total_games':total}
            for label,cid in [('main',main),('same_role_main',role_main)]:
                exp=experience(counts,wins,rc,rw,cid,t.lane,total,latest,role_latest) if cid is not None else {k:np.nan for k in observed_exp}
                d.update({label+'_'+k:v for k,v in exp.items()})
            for new,old in [('main_all_role_prior_games','prior_main_champion_games'),
                            ('main_all_role_prior_wins','prior_main_champion_wins'),
                            ('main_all_role_prior_wr','prior_main_champion_wr'),
                            ('main_all_role_prior_usage_share','prior_main_champion_usage_share')]:
                require(np.isclose(d[new],saved[old],equal_nan=True),'Reconstructed main experience differs')
            decisions.append(d)
            for cid in sorted(alternatives|{t.champion_id}):
                exp=experience(counts,wins,rc,rw,cid,t.lane,total,latest,role_latest)
                e=evidence.loc[evidence.candidate_champion_id.eq(cid)]
                if len(e):
                    e=e.iloc[0]
                    require(e.prior_same_role_games==exp['same_role_prior_games'] and e.prior_all_role_games==exp['all_role_prior_games'], 'Mapping experience differs')
                    require((pd.isna(e.latest_same_role_end) and pd.isna(exp['latest_same_role_prior_end_timestamp'])) or e.latest_same_role_end==exp['latest_same_role_prior_end_timestamp'],'Mapping completion time differs')
                observed=cid==t.champion_id
                rows.append({'puuid':player,'match_id':t.match_id,'timestamp':t.timestamp,
                    'candidate_champion_id':cid,'candidate_champion':names[cid],
                    'observed_champion_id':t.champion_id,'opponent_champion_id':t.opponent_champion_id,
                    'opponent_champion':t.opponent_champion,'lane':t.lane,'matchup_missing':t.matchup_missing,
                    'is_observed_pick':observed,'observed_pick_win':t.win if observed else None,
                    'is_eligible_alternative':not observed,'draft_availability_verified':False,
                    'is_existing_pregame_main':cid==main,'is_pregame_same_role_main':cid==role_main,
                    'prior_total_games':total,'prior_total_wins':total_wins,
                    'pregame_main_champion_id':main,'pregame_same_role_main_id':role_main,**exp})
    return pd.DataFrame(rows),pd.DataFrame(decisions)


REFERENCE_FIELDS = ['valid_rate','status','raw_matchup_winrate_pct','raw_matchup_winrate_probability',
    'matchup_games','tier_average_winrate_pct','website_delta2_pp','derived_tier_centered_rate_pct',
    'key_id','source_url','collected_at_utc','response_body_file','response_sha256',
    'filters_verified','verified_patch','verified_rank','verified_region','verified_queue','verified_lane',
    'verified_opponent_lane','extraction_strategy']


def assemble(data, candidates, decision_features):
    model=data['longitudinal_modeling']
    observed=model.merge(decision_features,on=KEY,how='left',validate='one_to_one')
    observed=regional_join(observed,data['valid_lookup'],'champion_id')
    # Fetch main rates for the target's exact role and opponent, without requiring
    # that this player's main was itself an eligible saved alternative.
    references=regional_join(model[KEY+['lane','matchup_missing','opponent_champion_id','prior_main_champion_id']],
                             data['valid_lookup'],'prior_main_champion_id',prefix='main_')
    main_fields=[f'main_{r}_{f}' for r in REGIONS for f in REFERENCE_FIELDS]
    observed=observed.merge(references[KEY+main_fields],on=KEY,how='left',validate='one_to_one')
    candidates=regional_join(candidates,data['valid_lookup'],'candidate_champion_id')
    main_experience=[c for c in decision_features if c.startswith('main_')]
    candidate_context=observed[KEY+['end_timestamp','side','patch','queue','history_incomplete',
        'genuinely_short_history','historical_masters_verified','rank_limitation']+main_experience+main_fields].copy()
    for region in REGIONS:
        for field in REFERENCE_FIELDS:
            candidate_context['observed_pick_'+region+'_'+field]=observed[region+'_'+field]
    selected=candidates.loc[candidates.is_observed_pick,KEY+[
        'all_role_prior_games','all_role_prior_wins','all_role_prior_wr','all_role_prior_wr_beta11',
        'same_role_prior_games','same_role_prior_wins','same_role_prior_wr','same_role_prior_wr_beta11']]
    selected=selected.rename(columns={c:'observed_pick_'+c for c in selected if c not in KEY})
    candidate_context=candidate_context.merge(selected,on=KEY,validate='one_to_one')
    candidates=candidates.merge(candidate_context,on=KEY,how='left',validate='many_to_one')
    candidates['observed_pick_win']=candidates.observed_pick_win.astype('Int8')
    for reference in ['main','observed_pick']:
        for role in ['all_role','same_role']:
            for method in ['wr','wr_beta11']:
                field=f'{role}_prior_{method}'
                candidates[f'candidate_minus_{reference}_{field}_pp']=100*(candidates[field]-candidates[f'{reference}_{field}'])
        for region in REGIONS:
            valid=candidates[region+'_valid_rate'] & candidates[f'{reference}_{region}_valid_rate']
            candidates[f'candidate_minus_{reference}_{region}_raw_matchup_pp']=(
                candidates[region+'_raw_matchup_winrate_pct']-candidates[f'{reference}_{region}_raw_matchup_winrate_pct']).where(valid)
    for frame in [observed,candidates]:
        frame['has_known_lane_opponent']=(~frame.matchup_missing & frame.lane.isin(scraper.ROLE_SLUGS)
                                           & frame.opponent_champion_id.notna())
        frame['has_valid_global_observed_matchup_rate']=(frame.global_valid_rate if frame is observed else frame.observed_pick_global_valid_rate)
        frame['has_valid_na_observed_matchup_rate']=(frame.na_valid_rate if frame is observed else frame.observed_pick_na_valid_rate)
        frame['has_both_observed_regional_rates']=frame.has_valid_global_observed_matchup_rate & frame.has_valid_na_observed_matchup_rate
        frame['has_incomplete_history']=frame.history_incomplete
    for region in REGIONS:
        candidates[f'has_usable_main_vs_alternative_{region}']=(
            candidates.is_eligible_alternative & ~candidates.is_existing_pregame_main
            & candidates.main_same_role_prior_games.gt(0)
            & candidates[region+'_valid_rate'] & candidates['main_'+region+'_valid_rate']
            & candidates.candidate_minus_main_all_role_prior_wr_pp.notna())
        usable=candidates.groupby(KEY)[f'has_usable_main_vs_alternative_{region}'].any()
        observed[f'has_usable_main_vs_alternative_{region}']=pd.MultiIndex.from_frame(observed[KEY]).map(usable).astype(bool)
    observed['included_global_analysis']=(observed.has_known_lane_opponent
        & observed.global_valid_rate & ~observed.has_incomplete_history)
    observed['included_paired_region_analysis']=observed.included_global_analysis & observed.na_valid_rate
    def reasons(row, paired=False):
        out=[]
        if not row.has_known_lane_opponent:out.append('unidentified_lane_opponent')
        if not row.global_valid_rate:out.append('global_observed_rate_unavailable:'+row.global_status)
        if row.has_incomplete_history:out.append('incomplete_initial_history')
        if paired and not row.na_valid_rate:out.append('na_observed_rate_unavailable:'+row.na_status)
        return ';'.join(out)
    observed['global_analysis_exclusion_reasons']=[reasons(r) for r in observed.itertuples()]
    observed['paired_region_exclusion_reasons']=[reasons(r,True) for r in observed.itertuples()]
    exclusions=observed.loc[~observed.included_paired_region_analysis].copy()
    return {'observed_player_games':observed,'candidate_champion_comparisons':candidates,
        'observed_global_analysis':observed.loc[observed.included_global_analysis].copy(),
        'observed_paired_region_analysis':observed.loc[observed.included_paired_region_analysis].copy(),
        'observed_exclusions':exclusions}


def validate_outputs(data, tables):
    observed=tables['observed_player_games']; candidates=tables['candidate_champion_comparisons']
    pd.testing.assert_frame_equal(observed.set_index(KEY)[data['longitudinal_modeling'].columns.drop(KEY)].sort_index(),
        data['longitudinal_modeling'].set_index(KEY).sort_index(),check_dtype=False)
    require(len(observed)==len(data['target_matches']),'Original target observations lost')
    require(not observed.duplicated(KEY).any(),'Duplicate observed key')
    require(not candidates.duplicated(KEY+['candidate_champion_id']).any(),'Duplicate candidate key')
    chosen=candidates.loc[candidates.is_observed_pick]
    require(len(chosen)==len(observed) and chosen.groupby(KEY).size().eq(1).all(),'Observed pick not exactly once')
    check=chosen[KEY+['observed_pick_win']].merge(observed[KEY+['win']],on=KEY,validate='one_to_one')
    require(check.observed_pick_win.eq(check.win).all(),'Selected outcome changed')
    require(candidates.loc[~candidates.is_observed_pick,'observed_pick_win'].isna().all(),'Outcome assigned to unplayed candidate')
    require('win' not in candidates,'Target outcome duplicated as candidate label')
    alt=candidates.loc[candidates.is_eligible_alternative]
    require(alt.same_role_prior_games.gt(0).all(),'Alternative has no prior same-role play')
    require(alt.candidate_champion_id.ne(alt.opponent_champion_id).all(),'Alternative equals opponent')
    require(not candidates.draft_availability_verified.any(),'Draft availability inferred')
    for col in ['latest_all_role_prior_end_timestamp','latest_same_role_prior_end_timestamp']:
        require((candidates[col].isna() | candidates[col].lt(candidates.timestamp)).all(),'History completion leakage')
    # Independent exact as-of counts/wins across every candidate row. Unlike the
    # expanding-counter builder, this uses per-champion sorted arrays and searchsorted.
    pool=pd.concat([data['historical_matches'],data['target_matches']],ignore_index=True)
    sources={(p,int(cid)):g.sort_values(['end_timestamp','match_id']) for (p,cid),g in pool.groupby(['puuid','champion_id'])}
    for (player,cid),group in candidates.groupby(['puuid','candidate_champion_id']):
        prior=sources.get((player,int(cid)),pool.iloc[:0])
        for role in ['all_role','same_role']:
            by_lane=[(None,group)] if role=='all_role' else [(lane,q) for lane,q in group.groupby('lane',dropna=False)]
            for lane,q in by_lane:
                records=prior if role=='all_role' else prior.loc[prior.lane.eq(lane)] if pd.notna(lane) else prior.iloc[:0]
                clock=records.end_timestamp.to_numpy()
                index=np.searchsorted(clock,q.timestamp.to_numpy(),side='left')
                cumulative=np.r_[0,records.win.to_numpy().cumsum()]
                np.testing.assert_array_equal(q[role+'_prior_games'],index)
                np.testing.assert_array_equal(q[role+'_prior_wins'],cumulative[index])
                raw=np.divide(cumulative[index],index,out=np.full(len(index),np.nan),where=index>0)
                np.testing.assert_allclose(q[role+'_prior_wr'],raw,equal_nan=True)
                np.testing.assert_allclose(q[role+'_prior_wr_beta11'],(cumulative[index]+1)/(index+2))
    for frame in [observed,candidates]:
        for region in REGIONS:
            valid=frame[region+'_valid_rate']
            require(frame.loc[valid,region+'_raw_matchup_winrate_probability'].between(0,1).all(),'Probability outside domain')
            np.testing.assert_allclose(frame.loc[valid,region+'_raw_matchup_winrate_probability'],
                frame.loc[valid,region+'_raw_matchup_winrate_pct']/100)
            require(frame.loc[~valid,region+'_raw_matchup_winrate_probability'].isna().all(),'Unverified rate used')
            good=valid & frame[region+'_tier_average_winrate_pct'].notna()
            np.testing.assert_allclose(frame.loc[good,region+'_derived_tier_centered_rate_pct'],
                frame.loc[good,region+'_raw_matchup_winrate_pct']-frame.loc[good,region+'_tier_average_winrate_pct']+50)
    require(not tables['observed_global_analysis'].has_incomplete_history.any(),'Incomplete history in primary subset')
    require(tables['observed_global_analysis'].global_valid_rate.all(),'Invalid primary rate included')
    require(tables['observed_paired_region_analysis'].na_valid_rate.all(),'Invalid paired NA rate included')
    require(set(map(tuple,tables['observed_paired_region_analysis'][KEY].values)) <=
            set(map(tuple,tables['observed_global_analysis'][KEY].values)), 'Paired subset not drawn from primary')
    for reference in ['main','observed_pick']:
        for region in REGIONS:
            result=candidates[f'candidate_minus_{reference}_{region}_raw_matchup_pp']
            expected=(candidates[region+'_raw_matchup_winrate_pct']-candidates[f'{reference}_{region}_raw_matchup_winrate_pct']).where(
                candidates[region+'_valid_rate'] & candidates[f'{reference}_{region}_valid_rate'])
            np.testing.assert_allclose(result,expected,equal_nan=True)
            # key IDs encode the same lane/opponent and explicitly separate regions.
            for prefix in ['',reference+'_']:
                valid=candidates[prefix+region+'_valid_rate']
                evidence=candidates.loc[valid,prefix+region+'_key_id'].str.split(':')
                require(evidence.str[2].eq(region).all(),'Comparison crosses region')
                require(evidence.str[4].eq(candidates.loc[valid,'lane']).all(),'Comparison crosses role')
                require(pd.to_numeric(evidence.str[6]).eq(candidates.loc[valid,'opponent_champion_id']).all(),'Comparison crosses opponent')
    # Disjoint inclusion/exclusion reconciles every master target; no thresholds.
    require(observed.included_global_analysis.eq(observed.global_analysis_exclusion_reasons.eq('')).all(),'Primary exclusion mismatch')
    require(observed.included_paired_region_analysis.eq(observed.paired_region_exclusion_reasons.eq('')).all(),'Paired exclusion mismatch')
    return {name:'PASS' for name in [
        'all_original_observed_keys_and_outcomes_preserved','no_join_row_multiplication_or_duplicate_keys',
        'each_observed_pick_occurs_once','unplayed_candidates_have_no_outcome',
        'strict_completion_before_kickoff','all_candidate_counts_wins_rates_independently_reconstructed',
        'saved_mapping_matches_prior_role_eligibility','main_identity_matches_existing_pipeline',
        'regional_raw_probability_and_derived_units_separate','comparisons_match_role_opponent_region',
        'incomplete_history_excluded_without_short_history_or_experience_threshold',
        'primary_paired_subsets_and_exclusions_reconcile']}


ORIGINAL_DEFINITIONS = {
 'puuid':('identifier','Player PUUID; repeated players require explicit dependence handling.'),
 'match_id':('identifier','Riot match identifier; all focal rows sharing a match must stay together in later splits.'),
 'timestamp':('milliseconds since Unix epoch','Target kickoff; chronological indexing and completion-before-kickoff cutoff.'),
 'end_timestamp':('milliseconds since Unix epoch','Postgame completion time retained for audit, not a pregame predictor.'),
 'win':('binary 0/1','Observed target outcome: 1 win, 0 loss; outcome table response, never a predictor.'),
 'champion':('name','Actually selected champion, unchanged from target record.'),
 'champion_id':('Riot champion ID','Actually selected champion ID, unchanged from target record.'),
 'opponent_champion':('name','Same-role opponent inferred from completed match; missing if assignment ambiguous.'),
 'opponent_champion_id':('Riot champion ID','Same-role opponent ID inferred from completed match.'),
 'opponent_puuid':('identifier','Inferred opposing player PUUID; audit identity, not a supplied predictor.'),
 'lane':('role label','Focal role inferred from completed-match teamPosition; not verified pregame intent.'),
 'team_position':('Riot role label','Original completed-match teamPosition; role-inference metadata.'),
 'side':('blue/red','Observed team side.'),'team_id':('100/200','Original Riot team ID.'),
 'patch':('patch label','Target Match-V5 major.minor patch 16.19.'),'queue':('Riot queue ID','Target queue 420, Ranked Solo/Duo.'),
 'matchup_missing':('boolean','Original inability to identify a unique same-role lane opponent.'),
 'summoner_spell_1':('spell ID','Selected summoner spell 1.'),'summoner_spell_2':('spell ID','Selected summoner spell 2.'),
 'perks_json':('JSON','Selected rune/stat-perk IDs; original helper stripped postgame rune counters.'),
 'platform':('platform label','Original source platform NA1.'),
 'prior_overall_wr':('probability 0–1','Prior total wins / prior total games; missing if no prior games.'),
 'prior_champion_wr':('probability 0–1','Prior selected-champion wins / games; missing at zero experience.'),
 'prior_main_champion_wr':('probability 0–1','Prior all-role main champion wins / games.'),
 'prior_champion_usage_share':('proportion 0–1','Prior chosen-champion games / all prior games.'),
 'prior_main_champion_usage_share':('proportion 0–1','Prior all-role main games / all prior games.'),
 'prior_champion_wr_beta11':('probability 0–1','(Prior chosen-champion wins+1)/(games+2); separately named Beta(1,1) mean.'),
 'prior_overall_wr_beta11':('probability 0–1','(All prior wins+1)/(all prior games+2); Beta(1,1) mean.'),
 'prior_champion_wr_minus_overall':('probability difference','Raw chosen-champion prior WR minus raw overall prior WR; missing when unsupported.'),
 'prior_latest_end_timestamp':('milliseconds since Unix epoch','Latest included prior completion; strictly before kickoff; audit field.'),
 'is_main_champion':('boolean','Observed pick equals existing all-role pregame main.'),
 'is_otp':('boolean; legacy operational flag','Original pipeline flag preserved unchanged (main pick and share >= saved configuration threshold); no new cutoff selected.'),
 'history_incomplete':('boolean; audit metadata','Initial history incomplete due to failed discovery/payloads; excludes player from analysis subsets.'),
 'genuinely_short_history':('boolean; audit metadata','Bounded history exhausted with a complete but short initial history; retained without a minimum-depth screen.'),
 'historical_masters_verified':('boolean; rank metadata','Whether dated historical Master membership was verified; not a supplied predictor.'),
 'rank_limitation':('text; audit metadata','Original limitation concerning dated study-period rank evidence.'),
 'rank_evidence_source':('text; audit metadata','Original dated historical rank evidence source, if any.'),
 'rank_observed_at_utc':('UTC time; audit metadata','Original historical rank observation time, if available.'),
 'seed_tier':('tier label; audit metadata','Original sampling rank label; not proof of patch-period membership.'),
 'selection_source':('source path; audit metadata','Original cohort sampling source.'),
 'selection_seed':('integer; audit metadata','Original reproducible sampling seed.'),
 'cohort_order':('integer; audit metadata','Original cohort ordering.'),
 'target_patch':('patch label; audit metadata','Pinned target patch for cohort collection.'),
 'target_discovery_complete':('boolean; audit metadata','Whether bounded target discovery completed; not a pregame predictor.'),
 'history_discovery_complete':('boolean; audit metadata','Whether initial-history discovery completed.'),
 'available_historical_matches':('games; audit metadata','Number of preserved initial historical games for the player.'),
 'history_shortfall':('games; audit metadata','Requested initial depth minus available initial games.'),
 'history_exhausted':('boolean; audit metadata','Whether bounded history discovery reached exhaustion.'),
 'history_stop_reason':('text; audit metadata','Saved collection stop reason.'),
 'history_exclusions_json':('JSON; audit metadata','Saved historical exclusion counters.'),
 'target_exclusions_json':('JSON; audit metadata','Saved target exclusion counters.'),
 'target_matches':('games; post-collection metadata','Final collected target-game count for player; not a pregame predictor.'),
 'missing_required_payloads':('count; audit metadata','Collection payload shortfall.'),
 'candidate_matches':('games; post-collection metadata','Original number of matches considered during target discovery; unrelated to candidate-champion rows.'),
 'anchor_timestamp':('milliseconds since Unix epoch; audit metadata','Earliest target anchor for history collection.'),
 'initial_history_games':('games; audit metadata','Initial preserved history size before first target.'),
 'initial_history_shortfall':('games; audit metadata','Configured requested initial depth minus initial preserved history size.'),
}

EXTERNAL_DEFINITIONS = {
 'raw_matchup_winrate_pct':('percent 0–100','Raw observed focal-champion same-role matchup win rate; not overall champion WR.'),
 'raw_matchup_winrate_probability':('probability 0–1','Valid verified raw matchup percent / 100; missing for unavailable/unverified rates.'),
 'matchup_games':('games','Published directional same-role matchup count, not overall champion games.'),
 'matchup_wins':('wins','Explicit published matchup wins only; not reconstructed from rounded rates.'),
 'matchup_losses':('losses','Explicit published matchup losses only; not reconstructed from rounded rates.'),
 'tier_average_winrate_pct':('percent 0–100','Labeled regional Average Master+ Win Rate; distinct from matchup WR and may exist even when an opponent row is absent.'),
 'derived_tier_centered_rate_pct':('derived percentage-unit value; not necessarily 0–100','raw_matchup_winrate_pct - tier_average_winrate_pct + 50; never clipped or treated as a probability.'),
 'website_delta2_pp':('percentage points','Website Delta 2 normalization statistic; distinct from raw rate and derived tier-centered value.'),
 'website_delta2_definition':('text','Saved website normalization definition.'),
 'status':('status label','Source status, or unidentified_lane_opponent/not_in_collection_plan for an unjoinable key. 404 remains page_unavailable_unresolved.'),
 'valid_rate':('boolean','status=ok, verified exact filters/roles, positive integer matchup games and raw rate in 0–100.'),
 'filters_verified':('boolean','Saved affirmative filter verification; no requested-URL inference.'),
 'key_id':('full regional lookup key','patch:rank:region:queue:lane:focal champion ID:opponent ID.'),
 'matchup_key_id':('nonregional lookup key','lane:focal champion ID:opponent ID.'),
 'source_url':('URL; provenance','Actual source page requested for the regional rate.'),
 'page_url':('URL; provenance','Planned directional URL; not necessarily the fetched representation.'),
 'batch_url':('URL; provenance','Planned champion/lane table URL.'),
 'collected_at_utc':('UTC ISO timestamp','Original regional page collection time; retained on reuse.'),
 'parsed_at_utc':('UTC ISO timestamp','Source parsing timestamp.'),
 'response_body_file':('run-relative path','Immutable cached source body relative to the pinned LoLalytics run.'),
 'response_sha256':('SHA-256','Cached response-body hash verified during preparation.'),
 'filter_evidence_json':('JSON; audit provenance','Saved current-control evidence and returned dataset header; not predictors.'),
 'matchup_evidence':('JSON; audit provenance','Named enemy-table row and opposing-lane evidence.'),
 'snapshot_id':('run ID','Pinned LoLalytics collection run ID.'),
 'input_snapshot':('snapshot ID','Pinned longitudinal snapshot backing the collection plan.'),
 'rank':('tier label','Requested Master+ (master_plus).'),
 'patch':('patch label','Requested 16.19.'),'queue':('queue label','Requested ranked; independently verified queue ID 420.'),
 'error':('text; audit metadata','Source error or table-omission context.'),
 'extraction_strategy':('adapter label','Verified page representation/adapter used.'),
 'tier_average_missing':('boolean','Whether labeled regional Master+ average is absent.'),
 'tier_average_source':('text; provenance','Labeled regional tier-average source.'),
 'tier_average_context':('text; provenance','Original rendered labeled regional tier-average context.'),
 'rate_units':('text','Original source rate and difference units.'),
 'derived_rate_definition':('text','Original formula distinguishing derived values from probabilities.'),
 'sample_size_context':('text','Original explanation that sample is matchup-specific.'),
 'champion':('name','Source focal champion identity for this lookup key.'),
 'opponent_champion':('name','Source opposing champion identity for this lookup key.'),
 'champion_slug':('URL slug','Reviewed source focal champion mapping.'),
 'opponent_slug':('URL slug','Reviewed source opponent mapping.'),
 'observed':('boolean; plan metadata','Whether this deduplicated source key supports any observed target in the collection plan; not a candidate outcome label.'),
 'alternative':('boolean; plan metadata','Whether this source key supports any saved alternative in the collection plan.'),
 'mapping_verified_locally':('boolean','Explicit champion ID/name/slug mapping was locally verified.'),
 'http_status':('HTTP code','Saved response status; 404 does not establish matchup sample or rate.'),
 'request_attempt_id':('attempt ID','Original request attempt when available.'),
 'page_failure_kind':('failure label','Source unresolved request/parsing failure classification.'),
 'diagnostic_evidence_json':('JSON; audit provenance','Source failure evidence when available.'),
 'website_lane':('website role label','Explicit source role slug corresponding to the focal role.'),
}


def describe_field(name):
    if name.endswith('exclusion_reasons'):
        return 'semicolon-separated reasons','All applicable exclusions for the named analysis subset; empty means included. Reasons may overlap.'
    if name in ORIGINAL_DEFINITIONS:return ORIGINAL_DEFINITIONS[name]
    # Prefixes describe references; every reference retains the same units.
    for prefix,description in [('observed_pick_','Selected-pick reference: '),('main_','Existing all-role pregame main reference: ')]:
        for region in REGIONS:
            marker=prefix+region+'_'
            if name.startswith(marker):
                units,text=describe_field(region+'_'+name[len(marker):])
                return units,description+text
    for region in REGIONS:
        if name.startswith(region+'_'):
            suffix=name[len(region)+1:]
            if suffix in EXTERNAL_DEFINITIONS:
                units,text=EXTERNAL_DEFINITIONS[suffix];return units,region.upper()+' retrospective Master+ patch-16.19: '+text
            if suffix.startswith(('verified_','requested_')):
                return 'source filter metadata',region.upper()+' '+suffix.replace('_',' ')+': exact requested/verified filter or role value; queue ID is 420.'
            raise ValueError('Undocumented external field: '+name)
    if name.startswith('candidate_minus_'):
        return 'percentage points','Candidate minus '+name[len('candidate_minus_'):].replace('_',' ')+'. Population percentages subtract directly; historical 0–1 win-rate differences multiply by 100. Requires both reference values; missing stays missing.'
    if name.endswith('exclusion_reasons'):
        return 'semicolon-separated reasons','All applicable exclusions for the named analysis subset; empty means included. Reasons may overlap.'
    if name.startswith('included_'):
        return 'boolean','Membership in named analysis subset. Primary: known opponent, verified Global rate, complete initial history. Paired additionally needs verified NA.'
    special={
        'is_observed_pick':('boolean','Candidate is the actually selected champion; exactly once per target.'),
        'observed_pick_win':('nullable binary 0/1','Actual outcome only on the observed-pick candidate row; missing on all unplayed alternatives. Candidate table is not an expanded outcome dataset.'),
        'is_eligible_alternative':('boolean','Unplayed saved candidate with strictly prior same-role play; known-opponent mirror excluded.'),
        'draft_availability_verified':('boolean; always false','Historical eligibility does not establish bans, pick order, teammate selections or draft availability.'),
        'is_existing_pregame_main':('boolean','Candidate ID equals original all-role prior main; ties use smallest champion ID.'),
        'is_pregame_same_role_main':('boolean','Candidate equals prior main within focal role; ties use smallest champion ID.'),
        'candidate_champion_id':('Riot champion ID','Candidate choice identity; observed pick or eligible unplayed alternative.'),
        'candidate_champion':('name','Candidate champion name from preserved records.'),
        'observed_champion_id':('Riot champion ID','Actually selected champion, for candidate comparison reference.'),
        'pregame_main_champion_id':('Riot champion ID','Existing all-role main recomputed and checked against original pipeline.'),
        'pregame_same_role_main_id':('Riot champion ID','Most-used prior same-role champion, smallest-ID tie rule; missing for no same-role history.'),
        'pregame_same_role_main_champion':('name','Prior same-role main champion name.'),
        'reconstructed_prior_total_games':('games','Independently reconstructed number of completed prior records; equals original prior_total_games.'),
        'has_known_lane_opponent':('boolean','Known valid focal role and identified same-role opponent; no guess when missing.'),
        'has_incomplete_history':('boolean; audit metadata','Alias of original history_incomplete, not a predictor.'),
        'has_valid_global_observed_matchup_rate':('boolean','Actual selected pick has a valid verified Global raw matchup rate.'),
        'has_valid_na_observed_matchup_rate':('boolean','Actual selected pick has a valid verified NA raw matchup rate.'),
        'has_both_observed_regional_rates':('boolean','Both separate observed-pick regional rates pass verification.'),
    }
    if name in special:return special[name]
    if name.startswith('has_usable_main_vs_alternative_'):
        return 'boolean','A distinct eligible unplayed alternative and the existing main both have valid rates for the named region, main has prior same-role play, and both raw all-role proficiency values exist; decision table means any such candidate.'
    for prefix,meaning in [('same_role_main_','Prior same-role main: '),('main_','Existing all-role main: '),('observed_pick_','Selected-pick historical reference: ')]:
        if name.startswith(prefix):
            units,text=describe_field(name[len(prefix):]);return units,meaning+text
    if name.startswith('zero_'):
        return 'boolean','No prior games for the named scope (all roles or focal role). Raw WR remains missing; separately named Beta mean may be 0.5.'
    if name.startswith('latest_') and name.endswith('_end_timestamp'):
        return 'milliseconds since Unix epoch','Latest included completion for the named candidate/scope, strictly before target kickoff; missing at zero experience.'
    if name.endswith('_wr_beta11'):
        return 'probability 0–1','Beta(1,1) posterior mean (prior wins+1)/(prior games+2) in the named scope; not fitted and distinct from raw WR.'
    if name.endswith('_wr'):
        return 'probability 0–1','Preserved strictly pregame wins/games in the named scope; missing when games=0.'
    if name.endswith('_usage_share'):
        return 'proportion 0–1','Prior candidate games in named scope divided by all-role prior games; missing if denominator=0.'
    if name.endswith(('_games','_wins')) and ('prior_' in name or name.startswith('prior_')):
        return 'count','Count from preserved initial history plus earlier targets completed strictly before kickoff, for the named champion/scope.'
    if name in ['prior_main_champion_id','prior_main_champion']:
        return 'champion identity','Original pregame all-role main by prior counts, smallest champion ID breaks ties.'
    raise ValueError('Undocumented output field: '+name)


def summary_counts(data,tables):
    o=tables['observed_player_games'];c=tables['candidate_champion_comparisons']
    counts={name:{'rows':len(frame),'columns':len(frame.columns),'players':int(frame.puuid.nunique()),
                  'unique_matches':int(frame.match_id.nunique())} for name,frame in tables.items()}
    counts.update(observed_global_valid_rows=int(o.global_valid_rate.sum()),observed_na_valid_rows=int(o.na_valid_rate.sum()),
        observed_both_regions_valid_rows=int(o.has_both_observed_regional_rates.sum()),
        unidentified_opponent_rows=int((~o.has_known_lane_opponent).sum()),
        incomplete_history_players=int(o.loc[o.history_incomplete,'puuid'].nunique()),
        incomplete_history_target_rows=int(o.history_incomplete.sum()),
        incomplete_history_initial_games=data['historical_matches'].loc[data['historical_matches'].puuid.isin(o.loc[o.history_incomplete,'puuid'])].shape[0],
        candidate_observed_pick_rows=int(c.is_observed_pick.sum()),candidate_unplayed_alternative_rows=int(c.is_eligible_alternative.sum()),
        genuinely_short_history_target_rows=int(o.genuinely_short_history.sum()),
        zero_chosen_champion_experience_rows=int(o.prior_champion_games.eq(0).sum()),
        global_exclusion_combinations=o.loc[~o.included_global_analysis,'global_analysis_exclusion_reasons'].value_counts().to_dict(),
        paired_exclusion_combinations=o.loc[~o.included_paired_region_analysis,'paired_region_exclusion_reasons'].value_counts().to_dict())
    return counts


def save_outputs(data,tables,validation,cleanup):
    """Export a new immutable preparation snapshot only after critical validation."""
    # Normalize index-free storage and object inference before documenting dtypes.
    # Joins may leave complete Boolean columns as object; Arrow restores bool.
    tables={name:frame.reset_index(drop=True).copy() for name,frame in tables.items()}
    for name,frame in tables.items():
        for column in frame.select_dtypes(include='object'):
            frame[column]=frame[column].where(frame[column].notna(),None)
        tables[name]=frame.infer_objects(copy=False)
    definitions=[]
    for name,frame in tables.items():
        for field,dtype in frame.dtypes.items():
            units,description=describe_field(field)
            definitions.append({'dataset':name,'field':field,'dtype':str(dtype),'units':units,'definition':description})
    # Source reads must not change under a running preparation.
    for relative,info in data['input_files'].items():
        require(sha256(data['root']/relative)==info['sha256'],'Source changed during preparation: '+relative)
    output=data['root']/'hlr_model/data/final'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    (output/'PREPARATION_INCOMPLETE').write_text('Export in progress; use only after this marker is removed.\n')
    output_hashes={}
    for name,frame in tables.items():
        serialized=frame
        for ext in ['parquet','csv']:
            p=output/f'{name}.{ext}'
            if ext=='parquet':serialized.to_parquet(p,index=False)
            else:serialized.to_csv(p,index=False,chunksize=5000)
            output_hashes[p.name]={'sha256':sha256(p),'bytes':p.stat().st_size}
        pd.testing.assert_frame_equal(pd.read_parquet(output/f'{name}.parquet'),serialized,check_exact=True)
    dictionary=pd.DataFrame(definitions)
    dictionary.to_csv(output/'data_dictionary.csv',index=False)
    (output/'data_dictionary.json').write_text(json.dumps(definitions,indent=2)+'\n')
    counts=summary_counts(data,tables)
    metadata={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'preparation_snapshot':output.name,
        'inputs':data['input_files'],'input_longitudinal_snapshot':str(data['source_dir'].relative_to(data['root'])),
        'input_lolalytics_run':str(data['run_dir'].relative_to(data['root'])),
        'implementation_sha256':sha256(Path(__file__)),
        'input_schemas':{name:{c:str(t) for c,t in data[name].dtypes.items()} for name in
            ['longitudinal_modeling','historical_matches','target_matches','player_coverage','player_cohort',
             'matchup_lookup','request_plan','decision_request_mapping']},
        'configuration':{'patch':'16.19','external_rank':'master_plus','queue_id':420,'primary_region':'global',
                         'sensitivity_region':'na','exclude_incomplete_history_player':True,
                         'minimum_external_matchup_games':None,'otp_cutoff_selected':None,
                         'candidate_definition':'Saved prior same-role candidates plus actual pick; exclude alternative equal to opponent',
                         'new_riot_api_requests':0,'full_scrape_started':False},
        'definitions':{'prior_window':'Initial preserved history plus earlier targets with end_timestamp strictly < current kickoff',
            'main':'Most prior all-role games; smallest champion ID breaks ties; existing main preserved',
            'same_role_main':'Most prior games in focal role; smallest champion ID breaks ties',
            'raw_historical_wr':'wins/games; missing at zero games',
            'beta11_wr':'(wins+1)/(games+2), probability units; 0.5 at zero historical experience is smoothing, not external imputation',
            'external_raw_probability':'Valid verified raw_matchup_winrate_pct / 100',
            'tier_centered':'raw_matchup_winrate_pct - tier_average_winrate_pct + 50; percentage-unit derived value, not necessarily a probability; never clipped',
            'candidate_outcome':'Only is_observed_pick=True has observed_pick_win; unplayed alternatives have missing labels',
            'descriptive_differences':'Candidate minus named reference. External percent values subtract directly in pp; historical probability differences multiply by 100.',
            'observed_exclusions':'Rows excluded from either primary or paired subset; separate inclusion flags and all applicable reasons retained',
            'predictor_matrix':'None created. Explicit later feature selection must exclude outcomes, collection/postgame metadata and audit fields.'},
        'cleanup':cleanup,'counts':counts,'validation':validation,
        'source_page_summary_by_region':pd.read_csv(data['run_dir']/'page_summary_by_region.csv').to_dict('records'),
        'unresolved_pages':pd.read_csv(data['run_dir']/'unresolved_pages.csv').to_dict('records'),
        'original_collection_definition':data['collection_metadata'],
        'limitations':['External rates are retrospective patch-16.19 benchmarks, not guaranteed available at target kickoff.',
            'Time-safe player histories do not make retrospective external rates time-safe.',
            'Role and opponent assignments are inferred from completed matches, not verified pregame lane intent.',
            'Historical Master membership flags are preserved; seed labels without dated evidence do not prove study-period rank.',
            'Global includes NA; benchmark populations overlap.',
            'Historical candidate play does not establish draft availability or causal effects of switching champions.',
            'No minimum experience/sample thresholds or best-counterpick optimization applied.'],
        'output_files':output_hashes}
    (output/'preparation_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    readme=f'''# Longitudinal OTP-versus-counterpick preparation {output.name}

Use `observed_global_analysis.parquet` for the primary observed-outcome analysis and
`observed_paired_region_analysis.parquet` for a Global-versus-NA sensitivity analysis
on identical observations. `win` is the observed binary response. These are audit-rich
data tables, not ready-made predictor matrices; choose features explicitly later.
No models, thresholds, EDA or presentation graphics were produced here.

`observed_player_games.parquet` preserves every original target player-game, including
missing opponents/rates and incomplete initial history. `observed_exclusions.parquet`
contains rows excluded from either named analysis subset, with separate inclusion flags
and reasons. Genuinely short histories and zero chosen-champion experience are retained.

`candidate_champion_comparisons.parquet` is a descriptive candidate comparison table,
NOT an additional outcome dataset. Each target contains the actual pick exactly once.
Only that row has `observed_pick_win`; unplayed alternatives have no observed outcome.
Do not repeat the target response across alternatives or fit an outcome model to them
as if they were played. All candidates' experience uses records completed strictly
before kickoff. Alternatives follow the saved same-role design and omit the known
opponent's champion. Draft availability remains unverified.

Main references use the existing all-role pregame main; same-role main identity and
experience are separate. Both mains break count ties with the smallest champion ID.
Historical raw rates use wins/games (missing at zero); separately named Beta(1,1)
means use (wins+1)/(games+2). Raw external rates are percentages; the explicitly named
probability divides valid raw values by 100. Website Delta 2 is separate. Derived
tier-centered values are raw percent minus regional tier-average percent plus 50;
they are never clipped and are not necessarily valid probabilities. All comparison
fields are candidate minus the named reference, in documented percentage points.

Global is primary; NA is sensitivity. Regions are never substituted and missing
external rates are never filled with 50%. Unresolved pages retain explicit status,
not zero games. Every successful source body, filter/role evidence and source time
was verified; original collection sources/caches remain in place.

External rates are retrospective patch-16.19 Master+ Ranked Solo/Duo benchmarks.
Time-safe player histories do not make these retrospective external rates time-safe.
Roles/opponents are inferred from completed matches, not verified pregame lane intent.
Keep outcome, collection flags, postgame completion fields, sampling/rank metadata and
audit provenance out of a predictor matrix unless later explicitly justified.
Group shared match rows together in any later chronological evaluation.

Counts: observed master {counts['observed_player_games']['rows']:,}; candidates
{counts['candidate_champion_comparisons']['rows']:,}; primary
{counts['observed_global_analysis']['rows']:,}; paired
{counts['observed_paired_region_analysis']['rows']:,}.
All five datasets also have CSV equivalents. See `data_dictionary.csv` (or JSON) for
every field and `preparation_metadata.json` for input/output hashes, definitions,
validation, exclusions and unresolved-page evidence. The timestamped sources are pinned;
re-execution creates another preparation snapshot instead of overwriting one.
'''
    (output/'README.md').write_text(readme)
    (output/'PREPARATION_INCOMPLETE').unlink()
    return output,metadata
