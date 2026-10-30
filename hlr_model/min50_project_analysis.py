"""Project-wide external-n>=50 reanalysis using the unchanged Pyro models.

Original histories/features, split assignments, fits and reports are read-only.
Preparation, new fitting, frozen evaluation and reporting are explicit stages.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hlr_model import broad_rank_modeling as broad
from hlr_model import master_cohort_modeling as master
from hlr_model import expanded_main_pick_report as expanded
from hlr_model import supported_main_pick_report as supported
from hlr_model.broad_rank_matchups import atomic_json, canonical, require
from hlr_model.main_pick_decision_report import COHORTS, KEY, file_sha

SCHEMA = 'project-wide-min50-refit-v1'
EXPECTED = {'Master':{'train':4462,'validation':895,'test':943},
            'Bronze–Diamond':{'train':3710,'validation':797,'test':874}}


def project_dir(root):
    return Path(root).resolve()/'hlr_model/data/analyses/min50_project_v1'


def module_for(cohort):
    return master if cohort == 'Master' else broad


def source_path(root,cohort):
    folder,version,_ = COHORTS[cohort]
    return Path(root).resolve()/'hlr_model/data/analyses'/folder/version


def environment():
    import torch,pyro
    require(pyro.__version__=='1.9.2' and torch.__version__.split('+')[0]=='2.15.0','Preserve established Pyro/Torch backend versions.')
    return dict(interpreter=sys.executable,python=sys.version,base_prefix=sys.base_prefix,
        dependencies={name:metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn',
            'matplotlib','pyarrow','torch','pyro-ppl','nbformat','nbclient','ipykernel']},
        inference='Unchanged exact Bernoulli Pyro NUTS; float64; no inference substitute')


def prepare_project(root):
    root=Path(root).resolve();project=project_dir(root);project.mkdir(parents=True,exist_ok=True)
    pointer=project/'versions.json'
    if pointer.exists():
        versions=json.loads(pointer.read_text())
        for cohort,name in versions.items():load_prepared(root,cohort,Path(name))
        return versions
    versions={};env=environment()
    for cohort in ['Master','Bronze–Diamond']:
        source=source_path(root,cohort)
        prep=json.loads((source/'preparation_manifest.json').read_text())
        original=pd.read_parquet(source/'prepared_played.parquet')
        require(file_sha(source/'prepared_played.parquet')==prep['output_sha256']['prepared_played.parquet'],'Original prepared data changed.')
        previous=supported.load_report(root,cohort)
        candidates=pd.read_parquet(previous['out']/'expanded_candidate_features.parquet')
        require(candidates.benchmark_matchup_games.ge(50).all(),'Candidate benchmark support changed.')
        count_col='global_matchup_games' if cohort=='Master' else 'benchmark_matchup_games'
        n=pd.to_numeric(original[count_col],errors='coerce')
        valid=np.isfinite(n)&n.ge(0)&n.eq(np.floor(n))
        if cohort=='Master':
            valid &= original.global_filters_verified.fillna(False)&original.global_verified_patch.eq('16.19')
            valid &= original.global_verified_region.eq('global')&original.global_verified_rank.eq('master_plus')
        else:
            valid &= original.benchmark_validation_status.eq('verified')&original.benchmark_requested_patch.eq('16.19')
            valid &= original.benchmark_requested_region.eq('global')&original.benchmark_requested_rank.eq(original.model_rank.str.lower())
        passed=valid&n.ge(50)
        played=original.copy()
        played['source_usable_data']=original.usable_data
        played['source_included_in_primary_model']=original.included_in_primary_model
        played['source_data_exclusion_reasons']=original.data_exclusion_reasons
        played['external_matchup_games']=n
        played['external_count_verified']=valid
        played['external_min50_pass']=passed
        played['usable_data']=original.usable_data&passed
        played['included_in_primary_model']=original.included_in_primary_model&passed
        for i in played.index[~passed]:
            reason='external_matchup_games_below_50' if valid.loc[i] else 'missing_or_invalid_external_matchup_count'
            old=original.loc[i,'data_exclusion_reasons']
            played.loc[i,'data_exclusion_reasons']=reason if old=='none' else old+'|'+reason
        # No low-support game is removed from a history or used to reconstruct a smaller history.
        unchanged=[c for c in original if c not in ['usable_data','included_in_primary_model','data_exclusion_reasons']]
        pd.testing.assert_frame_equal(original[unchanged],played[unchanged])
        require(played.split.equals(original.split) and played.partition.equals(original.partition),'Chronological partitions changed.')
        require(not played.duplicated(KEY).any(),'Duplicate observed game.')
        counts=played.loc[played.included_in_primary_model].groupby('split').size().to_dict()
        require(counts==EXPECTED[cohort],f'Investigate split discrepancy: {cohort}: {counts}')
        require(not played.loc[played.split.str.startswith('purged'),'included_in_primary_model'].any(),'Boundary purge changed.')
        identity=dict(schema=SCHEMA,cohort=cohort,minimum_external_games=50,
            parent_source=str(source.relative_to(root)),original_played_sha256=file_sha(source/'prepared_played.parquet'),
            candidates_sha256=file_sha(previous['out']/'expanded_candidate_features.parquet'),
            group_mapping_sha256=file_sha(source/'champion_grouping_lookup.parquet'),
            model_helper_sha256=file_sha(Path(module_for(cohort).__file__)),
            broad_helper_sha256=file_sha(Path(broad.__file__)),splits=counts,
            history_policy='Original strictly pregame features copied unchanged; histories never filtered.',
            split_policy='Original match-level boundaries and purges copied unchanged.')
        token=hashlib.sha256(canonical(identity).encode()).hexdigest()[:8]
        version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+token
        folder='master_cohort_mastery_min50_v1' if cohort=='Master' else 'broad_rank_mastery_min50_v1'
        out=root/'hlr_model/data/analyses'/folder/version;out.mkdir(parents=True,exist_ok=False)
        (out/'figures').mkdir()
        played.to_parquet(out/'prepared_played.parquet',index=False)
        candidates.to_parquet(out/'expanded_candidate_features.parquet',index=False)
        grouping=pd.read_parquet(source/'champion_grouping_lookup.parquet')
        grouping.to_parquet(out/'champion_grouping_lookup.parquet',index=False)
        exclusions=[]
        for split in ['train','validation','test']:
            before=original.loc[original.split.eq(split)&original.included_in_primary_model]
            after=played.loc[played.split.eq(split)&played.included_in_primary_model]
            exclusions.append(dict(cohort=cohort,split=split,original_included=len(before),retained=len(after),
                removed=len(before)-len(after),removed_percent=100*(len(before)-len(after))/len(before),
                retained_players=after.puuid.nunique(),retained_matches=after.match_id.nunique()))
        pd.DataFrame(exclusions).to_csv(out/'external_support_exclusions.csv',index=False)
        parent=json.loads((source/'reports/expanded_published_candidates_v1/report_manifest.json').read_text())
        for name,sha in parent['source_history_sha256'].items():require(file_sha(root/name)==sha,'History changed.')
        atomic_json(out/'execution_environment.json',env)
        atomic_json(out/'preparation_manifest.json',dict(identity=identity,created_at=datetime.now(timezone.utc).isoformat(),
            history_sha256=parent['source_history_sha256'],output_sha256={name:file_sha(out/name) for name in
                ['prepared_played.parquet','expanded_candidate_features.parquet','champion_grouping_lookup.parquet','external_support_exclusions.csv','execution_environment.json']}))
        atomic_json(out/'model_configuration.json',dict(specification=broad.model_specifications(),
            parent_model_helper=str(Path(module_for(cohort).__file__).relative_to(root)),
            master_rank='Absorbed in intercept; no rank coefficient' if cohort=='Master' else 'Bronze reference; all six protocol ranks',
            full_sampling=dict(chains=4,warmup=600,draws=1200,target_accept=.9,max_tree_depth=9,torch_threads=1),
            prior_outcome_exposure=True,primary='shared',secondary='grouped',minimum_external_games=50))
        versions[cohort]=str(out.relative_to(root))
        print(cohort,'new project-wide min50 version:',out,counts,flush=True)
    atomic_json(pointer,versions)
    return versions


def load_prepared(root,cohort,out=None):
    root=Path(root).resolve()
    if out is None:out=Path(json.loads((project_dir(root)/'versions.json').read_text())[cohort])
    out=out if out.is_absolute() else root/out
    manifest=json.loads((out/'preparation_manifest.json').read_text())
    for name,sha in manifest['output_sha256'].items():require(file_sha(out/name)==sha,f'Prepared artifact changed: {name}')
    for name,sha in manifest['history_sha256'].items():require(file_sha(root/name)==sha,'Original history changed.')
    require(file_sha(module_for(cohort).__file__)==manifest['identity']['model_helper_sha256'],'Frozen model helper changed.')
    played=pd.read_parquet(out/'prepared_played.parquet')
    candidates=pd.read_parquet(out/'expanded_candidate_features.parquet')
    counts=played.loc[played.included_in_primary_model].groupby('split').size().to_dict()
    require(counts==EXPECTED[cohort],'Prepared retained counts differ from audit.')
    return out,dict(played=played,alternatives=candidates)


def fit_models(root,cohort,*,smoke=False):
    out,prepared=load_prepared(root,cohort);module=module_for(cohort)
    config=module.Config(profile='smoke' if smoke else 'full')
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data].copy()
    if smoke:
        matches=train.sort_values(['timestamp','match_id']).match_id.drop_duplicates().head(config.sampling['train_match_limit'])
        train=train.loc[train.match_id.isin(matches)].copy()
    design=module.Design(train)
    for name,value in [('continuous',list(broad.CONTINUOUS)),('training_rows',len(train)),('players',len(design.players)),
                       ('champions',len(design.champions)),('player_champions',len(design.pcs))]:
        print(cohort,'SMOKE' if smoke else 'FULL',name,value,flush=True)
    target=out/'smoke' if smoke else out
    results={}
    for grouped in [False,True]:
        name='grouped' if grouped else 'shared'
        results[name]=module.fit_one(train,design,target,config,grouped=grouped)
        if not smoke:require(results[name]['status']=='converged',f'{cohort} {name}: diagnostic gate failed; do not evaluate/interpret.')
    atomic_json(target/'fit_stage_status.json',dict(cohort=cohort,profile=config.profile,
        models={name:r['status'] for name,r in results.items()},training_rows=len(train),
        completed_at=datetime.now(timezone.utc).isoformat()))
    return results


def load_fits(root,cohort):
    out,prepared=load_prepared(root,cohort);module=module_for(cohort)
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data].copy()
    design=module.Design(train);results={}
    training_sha=hashlib.sha256(train.to_json(orient='split',index=False).encode()).hexdigest()
    for name in ['shared','grouped']:
        folder=out/'fits'/name
        manifest=json.loads((folder/'fit_manifest.json').read_text())
        status=json.loads((folder/'fit_status.json').read_text())
        require(manifest['training_sha256']==training_sha and manifest['design']==json.loads(canonical(design.metadata())), 'Wrong new training/design cache.')
        require(manifest['model_source_sha256']==hashlib.sha256(inspect.getsource(module.pyro_model).encode()).hexdigest(),'Model equation changed.')
        require(status['status']=='converged' and status['diagnostic_gate_passed'],'Full diagnostic gate must pass.')
        diag=pd.read_csv(folder/'sampling_diagnostics.csv')
        require(np.isfinite(diag[['rank_split_Rhat','bulk_ESS','tail_ESS']]).all().all()
            and diag.rank_split_Rhat.max()<1.01 and diag.bulk_ESS.min()>400 and diag.tail_ESS.min()>400,'Diagnostic gate failed.')
        chains=[]
        for i in range(4):
            info=json.loads((folder/f'chain_{i}.json').read_text())
            require(info['divergences']==0 and info['acceptance_rate']>=.8 and file_sha(folder/f'chain_{i}.npz')==info['samples_sha256'],'Chain integrity/gate failed.')
            with np.load(folder/f'chain_{i}.npz') as archive:chains.append({key:archive[key] for key in archive.files})
        samples={key:np.stack([c[key] for c in chains]) for key in chains[0]}
        require(all(v.shape[:2]==(4,1200) and np.isfinite(v).all() for v in samples.values()),'Incomplete retained samples.')
        results[name]=dict(model=name,design=design,samples=samples,status='converged',diagnostics=diag,chain_info=status['chains'])
    return out,prepared,results


def evaluate_all(root,cohort,prepared,results,out,*,split):
    """Use the established evaluator; add identical-row baselines for broad."""
    module=module_for(cohort)
    identity=dict(prepared_sha256=file_sha(out/'prepared_played.parquet'),split=split,
        fit_manifest_sha256={name:file_sha(out/f'fits/{name}/fit_manifest.json') for name in results},
        evaluator_helper_sha256=file_sha(module.__file__))
    combined_manifest=out/f'{split}_all_evaluation.json'
    if combined_manifest.exists():
        cached=json.loads(combined_manifest.read_text())
        require(cached['identity']==identity,'Frozen combined evaluation inputs changed.')
        for name,sha in cached['output_sha256'].items():require(file_sha(out/name)==sha,'Combined evaluation output changed.')
        return pd.read_csv(out/f'{split}_all_metrics.csv'),pd.read_csv(out/f'{split}_all_calibration.csv')
    saved=module.evaluate_split(prepared,results,out,split=split)
    metrics=saved['metrics'].copy();calibration=saved['calibration'].copy();predictions=saved['predictions'].copy()
    frame=prepared['played'].loc[prepared['played'].split.eq(split)&prepared['played'].usable_data]
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data]
    if cohort=='Bronze–Diamond':
        benchmark_scores=[];benchmark_cal=[];benchmark_predictions=[]
        for name,probability in [('training_prevalence',np.full(len(frame),train.win.mean())),
                                 ('raw_matchup',frame.matchup_probability.to_numpy())]:
            score,cal=broad.evaluate(frame,probability,name,'benchmark',split)
            benchmark_scores.append(score);benchmark_cal.append(cal)
            f=frame[KEY+['target_row_id']].copy();f['probability']=probability
            f['model']=name;f['mode']='benchmark';f['split']=split;benchmark_predictions.append(f)
        metrics=pd.concat([metrics,pd.DataFrame(benchmark_scores)],ignore_index=True)
        calibration=pd.concat([calibration,*benchmark_cal],ignore_index=True)
        predictions=pd.concat([predictions,*benchmark_predictions],ignore_index=True)
    require(len(metrics)==6 and metrics.rows.eq(EXPECTED[cohort][split]).all(),'Evaluate every method on identical filtered rows.')
    for _,g in predictions.groupby(['model','mode']):
        require(len(g)==len(frame) and set(map(tuple,g[KEY].to_numpy()))==set(map(tuple,frame[KEY].to_numpy())), 'Evaluation row sets differ.')
    metrics['cohort']=cohort;calibration['cohort']=cohort;predictions['cohort']=cohort
    metrics.to_csv(out/f'{split}_all_metrics.csv',index=False)
    calibration.to_csv(out/f'{split}_all_calibration.csv',index=False)
    predictions.to_parquet(out/f'{split}_all_predictions.parquet',index=False)
    atomic_json(out/f'{split}_all_evaluation.json',dict(identity=identity,split=split,rows=len(frame),prior_outcome_exposure=True,
        baseline_prevalence=float(train.win.mean()),training_rows=len(train),
        qualification='Previously examined outcomes; retrospective benchmark-assisted reanalysis, not fresh confirmation.',
        output_sha256={n:file_sha(out/n) for n in [f'{split}_all_metrics.csv',f'{split}_all_calibration.csv',f'{split}_all_predictions.parquet']}))
    return metrics,calibration


def parameter_outputs(cohort,prepared,results,out):
    rows=[];groups=[];scaling=[]
    def row(model,name,values,units,positive=True):
        low,median,high=np.quantile(values,[.025,.5,.975])
        return dict(cohort=cohort,model=model,parameter=name,units=units,median=median,low_95=low,high_95=high,
            probability_positive=float(np.mean(values>0)) if positive else np.nan)
    for name,result in results.items():
        s=broad.flatten_draws(result);design=result['design']
        rows.append(row(name,'Intercept',s['alpha'],'Log-odds; centered continuous covariates; n=0; top/reference rank; zero identity effects'))
        for j,label in enumerate(['Raw matchup','log(1 + prior total games)','Historical champion WR Beta(1,1)']):
            rows.append(row(name,label,s['beta'][:,j],'Log-odds per filtered-training SD'))
        rows.append(row(name,'Raw matchup: +2 percentage points',s['beta'][:,0]*.02/design.scale[0],'Log-odds per +0.02 raw matchup probability'))
        for j,role in enumerate(broad.ROLES[1:]):rows.append(row(name,role+' versus top',s['beta_role'][:,j],'Log-odds contrast'))
        if 'beta_rank' in s:
            for j,rank in enumerate(broad.RANKS[1:]):rows.append(row(name,rank+' versus Bronze',s['beta_rank'][:,j],'Log-odds contrast'))
        rows.append(row(name,'Experience amplitude',s['beta_exp'],'Asymptotic log-odds; common amplitude'))
        rows.append(row(name,'Half-saturation k',np.exp(s['log_k']),'Recorded prior champion games',False))
        for key,label in [('log_sigma_player','Player SD'),('log_sigma_pc','Player–champion SD'),('log_sigma_champion','Champion contrast SD')]:
            rows.append(row(name,label,np.exp(s[key]),'Log-odds scale; exp(log SD)',False))
        if name=='grouped':
            rows.append(row(name,'Group-deviation contrast SD',np.exp(s['log_sigma_group']),'Log-odds amplitude scale; exp(log SD)',False))
            amplitude=s['beta_exp'][:,None]+np.exp(s['log_sigma_group'])[:,None]*(s['group_free']@design.group_basis.T)
            for j,group in enumerate(broad.GROUPS):groups.append(row(name,group,amplitude[:,j],'Actual total group amplitude, log-odds'))
        if name=='shared':
            for j,col in enumerate(broad.CONTINUOUS):scaling.append(dict(cohort=cohort,predictor=col,training_mean=design.mean[j],training_SD=design.scale[j]))
    pd.DataFrame(rows).to_csv(out/'parameter_summary.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'group_amplitude_summary.csv',index=False)
    pd.DataFrame(scaling).to_csv(out/'training_standardization.csv',index=False)
    # Conditional curve slices share the same summary type and retain original recorded-game support.
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data]
    curves=[]
    for name,result in results.items():
        s=broad.flatten_draws(result);design=result['design']
        context=train[list(broad.CONTINUOUS)].median().to_numpy(dtype=float)
        base=s['alpha']+s['beta']@((context-design.mean)/design.scale)
        amplitude=np.repeat(s['beta_exp'][:,None],len(broad.GROUPS),axis=1)
        if 'group_free' in s:amplitude+=np.exp(s['log_sigma_group'])[:,None]*(s['group_free']@design.group_basis.T)
        for j,group in enumerate(broad.GROUPS):
            support=train.loc[train.mastery_group.eq(group)]
            if support.empty:continue
            lower=float(support.n_recorded.min());upper=min(float(support.n_recorded.quantile(.95)),float(np.expm1(context[1])))
            if upper<lower:continue
            grid=np.linspace(lower,upper,60) if upper>lower else np.array([lower])
            probabilities=broad.expit(base[:,None]+amplitude[:,j,None]*grid/(grid+np.exp(s['log_k'])[:,None]))
            low,median,high=np.quantile(probabilities,[.025,.5,.975],axis=0)
            for k,n in enumerate(grid):curves.append(dict(cohort=cohort,model=name,group=group,n_recorded=n,
                median=median[k],low=low[k],high=high[k],training_games=len(support),training_players=support.puuid.nunique(),
                support_min=lower,support_cap=upper,context_matchup=context[0],context_log_total=context[1],context_historical_wr=context[2],
                context_role='top',context_rank='Master constant' if cohort=='Master' else 'Bronze',identity_effects='Zero champion/player/PC adjustments; not population average'))
    pd.DataFrame(curves).to_parquet(out/'group_experience_curves.parquet',index=False)


def recommendation_outputs(root,cohort,prepared,results,out):
    """Score expanded n>=50 candidates with NEW shared and grouped posterior draws."""
    module=module_for(cohort);p=prepared['played'];candidates=prepared['alternatives']
    main=p.loc[p.split.eq('test')&p.prior_main_champion_id.notna()&p.champion_id.eq(p.prior_main_champion_id)].drop(columns='win').copy()
    usable=main.loc[main.usable_data].copy()
    require(usable.external_matchup_games.ge(50).all(),'Unsupported main.')
    candidates=candidates.merge(usable[KEY],on=KEY,validate='many_to_one')
    require(candidates.benchmark_matchup_games.ge(50).all() and candidates.candidate_outcome.isna().all(),'Unsupported or labeled alternative.')
    zero=candidates.loc[candidates.n_recorded.eq(0)]
    require(zero.prior_champion_wins.eq(0).all() and zero.historical_wr_beta11.eq(.5).all(),'Zero-experience inputs changed.')
    train=p.loc[p.split.eq('train')&p.usable_data]
    pc_counts=train.groupby(['puuid','model_champion_id']).size()
    all_summaries=[]
    for model,result in results.items():
        report=out/'reports'/('expanded_min50_'+model+'_v1');report.mkdir(parents=True,exist_ok=True)
        candidates.to_parquet(report/'expanded_candidate_features.parquet',index=False)
        original_supported=supported.load_report(root,cohort)
        lookup=pd.read_parquet(original_supported['out']/'published_matchup_lookup.parquet')
        lookup.to_parquet(report/'published_matchup_lookup.parquet',index=False)
        identity=dict(candidate_sha256=file_sha(report/'expanded_candidate_features.parquet'),model=model,
            fit_manifest_sha256=file_sha(out/f'fits/{model}/fit_manifest.json'),
            chains_sha256=[file_sha(out/f'fits/{model}/chain_{i}.npz') for i in range(4)],batch_size=128,seed=2026)
        cache=report/'cache'/hashlib.sha256(canonical(identity).encode()).hexdigest()[:20]
        cache.mkdir(parents=True,exist_ok=True);atomic_json(cache/'identity.json',identity)
        samples=broad.flatten_draws(result);indexed=usable.set_index(KEY);parts=[]
        for mode in ['general','personalized']:
            for start in range(0,len(candidates),128):
                part=candidates.iloc[start:start+128];path=cache/f'{mode}_{start:07d}.parquet'
                if path.exists():
                    batch=pd.read_parquet(path)
                    require(batch.target_row_id.tolist()==part.target_row_id.tolist() and batch.candidate_champion_id.tolist()==part.candidate_champion_id.tolist(),'Prediction-cache key mismatch.')
                else:
                    refs=indexed.loc[pd.MultiIndex.from_frame(part[KEY])].reset_index()
                    pr=module.predict_draws(result['design'],refs,samples,mode=mode)
                    pa=module.predict_draws(result['design'],part,samples,mode=mode)
                    delta=100*(pa-pr)
                    keep=KEY+['target_row_id','timestamp','lane','opponent_champion_id','model_rank','main_champion_id',
                        'candidate_champion_id','main_champion','candidate_champion','n_recorded','prior_champion_wins',
                        'prior_same_role_games','prior_total_games','historical_wr_beta11','mastery_group','matchup_probability',
                        'benchmark_sparse','benchmark_matchup_games','experience_class']
                    batch=part[keep].copy();batch['mode']=expanded.MODES[mode];batch['model']=model
                    batch['source_analysis_version']=out.name
                    batch['main_recorded_games']=refs.n_recorded.to_numpy()
                    batch['main_historical_wr_beta11']=refs.historical_wr_beta11.to_numpy()
                    batch['main_matchup_probability']=refs.matchup_probability.to_numpy()
                    batch['main_external_matchup_games']=refs.external_matchup_games.to_numpy()
                    batch['main_probability']=pr.mean(axis=0);batch['alternative_probability']=pa.mean(axis=0)
                    batch['difference_pp']=delta.mean(axis=0);batch['low_pp'],batch['high_pp']=np.quantile(delta,[.025,.975],axis=0)
                    batch['probability_alternative_higher']=(delta>0).mean(axis=0)
                    batch['candidate_pc_training_games']=[int(pc_counts.get((player,int(cid)),0)) for player,cid in zip(part.puuid,part.candidate_champion_id)]
                    batch['candidate_champion_seen_training']=part.candidate_champion_id.isin(result['design'].champions)
                    require(np.isfinite(batch[['main_probability','alternative_probability','difference_pp','low_pp','high_pp']]).all().all(),'Nonfinite candidate predictions.')
                    require(batch.matchup_probability.gt(batch.main_matchup_probability).all(),'Alternative benchmark is not better.')
                    batch.to_parquet(path,index=False)
                parts.append(batch)
                if start%(128*30)==0:print(cohort,model,expanded.MODES[mode],min(start+128,len(candidates)),'/',len(candidates),'new-fit candidates scored/cached',flush=True)
        pairs=pd.concat(parts,ignore_index=True)
        require(len(pairs)==2*len(candidates) and not pairs.duplicated(KEY+['mode','candidate_champion_id']).any(),'Candidate prediction count mismatch.')
        require(pairs.groupby(KEY+['mode']).main_probability.agg(lambda s:s.max()-s.min()).le(1e-12).all(),'Main draws are not identity-stable.')
        pairs.to_parquet(report/'all_candidate_comparisons.parquet',index=False)
        ctx=dict(root=Path(root),source=out,out=report,cohort=cohort,main=main,usable=usable,played=p)
        decisions,summary=expanded.decisions_and_summaries(ctx,candidates,pairs)
        decisions.attrs['provenance'].update(specification=model,
            paired_draws=f'4800 retained NEW project-wide min50 {model} posterior draws; no original-fit posterior reuse.',
            fitted_chain_sha256=identity['chains_sha256'])
        decisions.to_parquet(report/'game_level_decisions.parquet',index=False)
        expanded.choose_examples(decisions).to_parquet(report/'examples.parquet',index=False)
        summary['model']=model;summary.to_csv(report/'summary.csv',index=False);all_summaries.append(summary)
        atomic_json(report/'report_manifest.json',dict(model=model,new_analysis_version=out.name,
            posterior_provenance=identity,minimum_external_games=50,
            output_sha256={name:file_sha(report/name) for name in ['expanded_candidate_features.parquet','published_matchup_lookup.parquet','all_candidate_comparisons.parquet','game_level_decisions.parquet','examples.parquet','summary.csv']}))
    pd.concat(all_summaries,ignore_index=True).to_csv(out/'recommendation_summary.csv',index=False)


def finish_project(root):
    root=Path(root).resolve();loaded={cohort:load_fits(root,cohort) for cohort in ['Master','Bronze–Diamond']}
    metrics=[];calibrations=[]
    # Freeze and assess BOTH validation cohorts before opening either filtered test period.
    for cohort,(out,prepared,results) in loaded.items():
        m,c=evaluate_all(root,cohort,prepared,results,out,split='validation');metrics.append(m);calibrations.append(c)
    for cohort,(out,prepared,results) in loaded.items():
        decision=dict(primary='shared',secondary='grouped',selection='Predeclared; retain both models. No parameter/threshold tuning.',
            prior_outcome_exposure=True,validation_metrics_sha256=file_sha(out/'validation_all_metrics.csv'),
            frozen_fit_manifest_sha256={name:file_sha(out/f'fits/{name}/fit_manifest.json') for name in results})
        path=out/'validation_decision.json'
        if path.exists():require(json.loads(path.read_text())==decision,'Frozen validation decision changed.')
        else:atomic_json(path,decision)
    for cohort,(out,prepared,results) in loaded.items():
        m,c=evaluate_all(root,cohort,prepared,results,out,split='test');metrics.append(m);calibrations.append(c)
        parameter_outputs(cohort,prepared,results,out)
        recommendation_outputs(root,cohort,prepared,results,out)
        atomic_json(out/'analysis_complete.json',dict(status='all_full_fits_converged; frozen validation then test evaluated; expanded recommendations completed',
            prior_outcome_exposure=True,minimum_external_games=50,
            output_sha256={p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_complete.json'}))
    project=project_dir(root)
    pd.concat(metrics,ignore_index=True).to_csv(project/'cohort_metrics.csv',index=False)
    pd.concat(calibrations,ignore_index=True).to_csv(project/'cohort_calibration.csv',index=False)
    for path,sha in json.loads((project/'protected_original_sha256.json').read_text()).items():require(file_sha(root/path)==sha,f'Original artifact changed: {path}')
    atomic_json(project/'completion.json',dict(status='complete',versions=json.loads((project/'versions.json').read_text()),
        helper_sha256=file_sha(__file__),completed_at=datetime.now(timezone.utc).isoformat(),
        qualification='Retrospective reanalysis. Intervals omit external benchmark uncertainty and selection adjustment; alternative outcomes/draft availability unknown; no causal switching claims.'))
