"""Separate, offline Master-labeled longitudinal analysis; never mutates broad inputs.

Model differences: constant seed rank is absorbed in the intercept. Everything
else follows the pinned broad Bernoulli/NUTS specification. Master+ benchmark
coverage and prior test exposure are explicit, not silently harmonized.
"""
from __future__ import annotations
import hashlib
import inspect
import io
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from hlr_model import broad_rank_modeling as broad
from hlr_model import final_dataset_helpers as source_helpers
from hlr_model.broad_rank_matchups import atomic_bytes, atomic_json, canonical, require, sha, stamp

KEY = broad.KEY
GROUPS = broad.GROUPS
ROLES = broad.ROLES
CONTINUOUS = broad.CONTINUOUS
SCHEMA = 'master-separate-grouped-nuts-v1'
BROAD_VERSION = '20261028T193830350096Z_5a469939'
LIMITS = [s for s in broad.LIMITS if not s.startswith('Ranks were checked')] + [
    'Master labels are inherited V1 seed labels; none has dated historical or collection-time rank verification.',
    'Master uses Global Master+ benchmarks; broad uses exact Bronze–Diamond tiers. No tier/region substitution occurs.',
    'These validation/test outcomes were already examined by hlr_modeling.ipynb; this is a previously exposed retrospective reanalysis.',
    'Separate cohort estimates do not test a rank interaction. No pooled analysis is fitted here.',
]

@dataclass(frozen=True)
class Config(broad.Config):
    snapshot: str = '20261020T180254240509Z'
    metadata_sha256: str = '81c65671542095e25a3883e882075d587bfb8d5a20df477e0c0f0fefd94c5fd7'


def file_sha(path):
    return source_helpers.sha256(Path(path))


def load_audit(root, config, *, validate=True):
    """Discover sources from the pinned preparation manifest; no collector runs."""
    root=Path(root).resolve()
    found=[]
    for file in (root/'hlr_model/data/final').glob('*/preparation_metadata.json'):
        meta=json.loads(file.read_text())
        if meta.get('preparation_snapshot')==config.snapshot:found.append((file,meta))
    require(len(found)==1,'Expected one authoritative Master preparation manifest.')
    file,meta=found[0]
    require(file_sha(file)==config.metadata_sha256,'Master preparation manifest changed; do not repin silently.')
    require(meta['configuration']['external_rank']=='master_plus' and meta['configuration']['primary_region']=='global',
            'Incompatible Master benchmark tier/region.')
    for name,info in meta['inputs'].items():require(file_sha(root/name)==info['sha256'],f'Source changed: {name}')
    for name,info in meta['output_files'].items():require(file_sha(file.parent/name)==info['sha256'],f'Prepared source changed: {name}')
    data=source_helpers.load_inputs(root)
    require(str(data['source_dir'].relative_to(root))==meta['input_longitudinal_snapshot']
            and str(data['run_dir'].relative_to(root))==meta['input_lolalytics_run'],'Manifest lineage differs from offline validator.')
    names=('observed_player_games','candidate_champion_comparisons','observed_global_analysis',
           'observed_paired_region_analysis','observed_exclusions')
    tables={n:pd.read_parquet(file.parent/(n+'.parquet')) for n in names}
    checks=dict(source_helpers.validate_sources(data),**source_helpers.validate_outputs(data,tables)) if validate else meta['validation']
    cohort=data['player_cohort'];played=tables['observed_player_games'];candidates=tables['candidate_champion_comparisons']
    require(cohort.seed_tier.str.lower().eq('master').all(),'Mixed seed ranks in Master cohort.')
    require(played.platform.eq('NA1').all() and played.queue.eq(420).all() and played.patch.eq('16.19').all(),'Wrong platform/queue/patch.')
    exposures=[]
    for name in ('hlr_modeling.ipynb','tradeoff_modeling.ipynb','hlr_EDAfinal.ipynb'):
        path=root/'hlr_model'/name;n=json.loads(path.read_text())
        cells=[i for i,c in enumerate(n['cells']) if c['cell_type']=='code' and c.get('execution_count') is not None
               and any(s in ''.join(c['source']) for s in ('test_scores=','test_labels=','validation_predictions=','val_scores='))]
        exposures.append(dict(notebook=name,sha256=file_sha(path),executed_outcome_analysis_cells=cells))
    audit=dict(preparation_manifest=str(file.relative_to(root)),preparation_sha256=file_sha(file),
        longitudinal_snapshot=meta['input_longitudinal_snapshot'],benchmark_run=meta['input_lolalytics_run'],
        source_hashes=meta['inputs'],prepared_hashes=meta['output_files'],checks=checks,
        selected_players=len(cohort),target_players=played.puuid.nunique(),usable_source_players=tables['observed_global_analysis'].puuid.nunique(),
        target_rows=len(played),target_unique_matches=played.match_id.nunique(),candidate_rows=len(candidates),
        unplayed_alternatives=int((~candidates.is_observed_pick).sum()),
        cohort_definition='200 deterministic seed-1619 draws from legacy Master-labeled seeds; no OTP-only filter; 15 have no target games.',
        dated_rank_verified_players=int(cohort.historical_masters_verified.sum()),
        collection_time_rank='Not independently verified; legacy Master labels only.',patch='16.19',queue=420,platform='NA1',
        benchmark='Global Master+ raw matchup percent / 100; NOT exact Master-only, NOT NA fallback, NOT tier-centered probability',
        benchmark_quality=json.loads((data['run_dir']/'quality_summary.json').read_text()),
        source_exclusion_counts=meta['counts']['global_exclusion_combinations'],
        initial_history_actual_target_players=played[['puuid','initial_history_games']].drop_duplicates().initial_history_games.describe().to_dict(),
        prior_outcome_exposure=True,prior_analysis=exposures,
        previous_models='Ordinary and hierarchical logistic/VI; nonlinear B/C VI, independent player-only B NUTS check. Not the new shared/grouped champion-baseline specification.',
        limits=LIMITS)
    return data,tables,audit


def prepare(root, config):
    data,tables,audit=load_audit(root,config)
    groups,study_rows,study_meta=broad.load_groups(root,config)
    require(study_meta['modeling_ready'],'Pinned workbook grouping unavailable.')
    p=tables['observed_player_games'].copy();c=tables['candidate_champion_comparisons']
    chosen=c.loc[c.is_observed_pick,KEY+['same_role_prior_games']]
    p=p.merge(chosen,on=KEY,how='left',validate='one_to_one')
    a=c.loc[c.is_eligible_alternative].copy().reset_index(drop=True)
    require(a.observed_pick_win.isna().all(),'Unplayed outcome contamination.')
    a=a.drop(columns='observed_pick_win');a['candidate_outcome']=np.nan
    p['target_row_id']=p.puuid+':'+p.match_id
    split=broad.split_assignments(p,config)
    p=p.merge(split[KEY+['partition','split']],on=KEY,validate='one_to_one')
    a=a.merge(p[KEY+['target_row_id','partition','split']],on=KEY,how='left',validate='many_to_one')
    for frame,kind in [(p,'played'),(a,'alternative')]:
        frame['row_kind']=kind;frame['model_rank']='MASTER';frame['seed_tier']='MASTER'
        frame['model_champion_id']=frame.champion_id if kind=='played' else frame.candidate_champion_id
        frame['n_recorded']=frame.prior_champion_games if kind=='played' else frame.all_role_prior_games
        frame['historical_wr_beta11']=frame.prior_champion_wr_beta11 if kind=='played' else frame.all_role_prior_wr_beta11
        frame['log_total']=np.log1p(frame.prior_total_games)
        frame['matchup_probability']=frame.global_raw_matchup_winrate_probability.where(frame.global_valid_rate)
        frame['benchmark_valid_rate']=frame.global_valid_rate
        frame['benchmark_raw_matchup_winrate_probability']=frame.matchup_probability
        frame['benchmark_sparse']=frame.global_valid_rate & frame.global_matchup_games.lt(100)
        frame['benchmark_status']=frame.global_status
        if kind=='alternative':
            frame['prior_all_role_games']=frame.all_role_prior_games;frame['prior_same_role_games']=frame.same_role_prior_games
        group_map=groups.set_index('champion_id')
        for col in ('mastery_group','classification_reason','raw_gain_pp','grouping_score'):
            frame[col]=frame.model_champion_id.map(group_map[col])
        require(frame.mastery_group.notna().all(),'Unmapped champion group.')
        frame['data_exclusion_reasons']=[
            '|'.join(s for s,flag in [('incomplete_initial_history',row.has_incomplete_history),
                ('unidentified_lane_opponent',not row.has_known_lane_opponent),
                ('missing_or_unverified_global_matchup',not row.global_valid_rate)] if flag) or 'none'
            for row in frame.itertuples()]
        valid=np.isfinite(frame[['n_recorded',*CONTINUOUS]].to_numpy(dtype=float)).all(axis=1)
        valid &= frame.n_recorded.ge(0)&frame.n_recorded.le(frame.prior_total_games)&frame.lane.isin(ROLES)
        invalid=frame.data_exclusion_reasons.eq('none') & ~valid
        frame.loc[invalid,'data_exclusion_reasons']='invalid_pregame_feature'
        frame['usable_data']=frame.data_exclusion_reasons.eq('none')
        frame['included_in_primary_model']=frame.usable_data & frame.split.isin(['train','validation','test']) if kind=='played' else False
        frame['experience_bin']=pd.cut(frame.n_recorded,[-.5,.5,4.5,9.5,19.5,49.5,np.inf],labels=broad.BINS)
    require(p.usable_data.equals(p.included_global_analysis),'New usable definitions differ from the audited primary source.')
    require('win' not in a and a.candidate_outcome.isna().all(),'Alternative contains outcomes.')
    counts=p.loc[p.usable_data & p.split.isin(['train','validation','test'])].groupby('split').size().to_dict()
    require(counts==dict(train=5086,validation=1049,test=1065),f'Investigate unexpected comparable splits: {counts}')
    return dict(played=p,alternatives=a,split=split),groups,study_rows,study_meta,audit


def support_tables(prepared):
    p=prepared['played'];train=p.loc[p.split.eq('train')&p.usable_data]
    pc=train.groupby(['puuid','model_champion_id']).size()
    support=pd.DataFrame([dict(training_games=len(train),players=train.puuid.nunique(),player_champions=len(pc),
        median_games_per_pc=pc.median(),fraction_pc_once=pc.eq(1).mean(),fraction_pc_lt5=pc.lt(5).mean(),
        players_n_ge50=train.loc[train.n_recorded.ge(50)].puuid.nunique(),players_n_ge100=train.loc[train.n_recorded.ge(100)].puuid.nunique())])
    group=train.groupby('mastery_group').agg(games=('match_id','size'),players=('puuid','nunique'),
        champions=('model_champion_id','nunique'),median_experience=('n_recorded','median')).reindex(GROUPS).reset_index()
    for cutoff in (50,100):group[f'players_n_ge{cutoff}']=group.mastery_group.map(train.loc[train.n_recorded.ge(cutoff)].groupby('mastery_group').puuid.nunique()).fillna(0).astype(int)
    splits=p.groupby('split').agg(rows=('match_id','size'),players=('puuid','nunique'),matches=('match_id','nunique'),usable_rows=('usable_data','sum')).reset_index()
    used=p.loc[p.usable_data & p.split.isin(['train','validation','test'])].groupby('split').agg(rows=('match_id','size'),players=('puuid','nunique'),matches=('match_id','nunique')).reset_index()
    exclusions=p.groupby(['split','data_exclusion_reasons']).size().rename('rows').reset_index()
    return dict(training_support=support,training_group_support=group,split_support=splits,usable_split_support=used,exclusions=exclusions)


def prepare_outputs(root,prepared,groups,study_rows,study_meta,audit,config):
    identity=json.loads(canonical(dict(schema=SCHEMA,source_preparation=audit['preparation_sha256'],config=asdict(config),
        helper_sha256=file_sha(__file__),broad_helper_sha256=file_sha(broad.__file__),source_helper_sha256=file_sha(source_helpers.__file__),
        study_sha256=study_meta['sha256'])))
    base=Path(root)/'hlr_model/data/analyses/master_cohort_mastery_v1'
    for path in sorted(base.glob('*/preparation_manifest.json'),reverse=True):
        saved=json.loads(path.read_text())
        if saved['identity']==identity:
            require(all(file_sha(path.parent/n)==v for n,v in saved['output_sha256'].items()),'Prepared Master artifact changed.')
            return path.parent
    out=base/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    (out/'figures').mkdir()
    for name,frame in dict(prepared_played=prepared['played'],prepared_alternatives=prepared['alternatives'],
                           champion_grouping_lookup=groups,external_study_rows=study_rows,**support_tables(prepared)).items():
        frame.to_parquet(out/(name+'.parquet'),index=False)
        if not name.startswith('prepared_'):frame.to_csv(out/(name+'.csv'),index=False)
    atomic_json(out/'source_audit.json',audit);atomic_json(out/'grouping_metadata.json',study_meta)
    specs=broad.model_specifications();specs.pop('changes_from_master',None)
    specs.update(rank='Constant Master seed label absorbed in intercept; no rank coefficient/contrast sampled.',
                 eta=specs['eta'].replace(' + rank',''),cohort='Separate longitudinal Master-labeled cohort; Global Master+ benchmark')
    atomic_json(out/'model_configuration.json',dict(config=asdict(config),sampling=config.sampling,specifications=specs,limits=LIMITS))
    hashes={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file()}
    atomic_json(out/'preparation_manifest.json',dict(identity=identity,output_sha256=hashes,created_at=stamp(),status='prepared_only'))
    return out


class Design(broad.Design):
    def __init__(self,train):
        require(train.model_rank.eq('MASTER').all(),'Design only accepts the Master-labeled cohort.')
        super().__init__(train)
    def matrix(self,frame):
        require(frame.model_rank.eq('MASTER').all(),'Cross-cohort prediction is forbidden.')
        # Reuse continuous/role/identity maps without changing broad global levels.
        arrays=super().matrix(frame.assign(model_rank='BRONZE'))
        arrays['rank']=np.empty((len(frame),0))
        return arrays
    def metadata(self):
        data=super().metadata();data['ranks']=['MASTER'];data['rank_effect']='absorbed in intercept; zero sampled contrasts'
        data['group_basis']=self.group_basis.tolist();data['champion_basis']=self.champion_basis.tolist()
        return data


def pyro_model(data,dimensions,grouped=False):
    import torch
    import pyro
    import pyro.distributions as dist
    def normal(name,loc=0.,scale=1.,size=None):
        shape=() if size is None else (size,)
        distribution=dist.Normal(torch.full(shape,float(loc),dtype=torch.float64),torch.full(shape,float(scale),dtype=torch.float64))
        return pyro.sample(name,distribution.to_event(1) if size is not None else distribution)
    alpha=normal('alpha',scale=2.);beta=normal('beta',scale=2.,size=len(CONTINUOUS))
    role=normal('beta_role',size=len(ROLES)-1);beta_exp=normal('beta_exp');k=normal('log_k',np.log(30),1.).exp()
    sigma_player=normal('log_sigma_player',np.log(.3),.7).exp();sigma_pc=normal('log_sigma_pc',np.log(.15),.5).exp()
    sigma_champion=normal('log_sigma_champion',np.log(.3),.7).exp()
    player=normal('player_z',size=dimensions['players'])*sigma_player;pc=normal('pc_z',size=dimensions['pcs'])*sigma_pc
    champion=data['champion_basis']@normal('champion_free',size=dimensions['champions']-1)*sigma_champion if dimensions['champions']>1 else torch.zeros(1,dtype=torch.float64)
    slope=beta_exp.expand(len(GROUPS))
    if grouped:slope=slope+normal('log_sigma_group',np.log(.3),.7).exp()*(data['group_basis']@normal('group_free',size=len(GROUPS)-1))
    eta=(alpha+data['X']@beta+data['role']@role+champion[data['champion']]+player[data['player']]+pc[data['pc']]
         +slope[data['group']]*data['n']/(data['n']+k))
    pyro.sample('outcomes',dist.Bernoulli(logits=eta).to_event(1),obs=data['y'])


def fit_one(train,design,out,config,*,grouped):
    import torch,pyro
    from pyro.infer import MCMC,NUTS
    from hlr_model.tradeoff_helpers import rank_mcmc_diagnostics
    require(pyro.__version__=='1.9.2','Exact Pyro 1.9.2 required; no inference substitute.')
    model='grouped' if grouped else 'shared';settings=config.sampling;folder=Path(out)/'fits'/model;folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(canonical(dict(schema=SCHEMA,model=model,backend_version=pyro.__version__,torch_version=torch.__version__,
        sampling=settings,seed=config.seed,target_accept=config.target_accept,torch_threads=config.torch_threads,
        training_sha256=sha(train.to_json(orient='split',index=False).encode()),design=design.metadata(),
        model_source_sha256=sha(inspect.getsource(pyro_model).encode()),sampler_source_sha256=sha(inspect.getsource(fit_one).encode()),
        helper_sha256=file_sha(__file__),broad_helper_sha256=file_sha(broad.__file__))))
    file=folder/'fit_manifest.json'
    if file.exists():require(json.loads(file.read_text())==manifest,'Master fit cache configuration mismatch.')
    else:atomic_json(file,manifest)
    data=broad.torch_training_data(design,train);dimensions=dict(players=len(design.players),pcs=len(design.pcs),champions=len(design.champions))
    samples=[];infos=[];previous=torch.get_num_threads();torch.set_num_threads(config.torch_threads)
    try:
        for chain in range(settings['chains']):
            sample_file=folder/f'chain_{chain}.npz';info_file=sample_file.with_suffix('.json')
            print(f'{model}: chain {chain+1}/{settings["chains"]}',flush=True)
            if sample_file.exists() and info_file.exists():
                info=json.loads(info_file.read_text());require(file_sha(sample_file)==info['samples_sha256'],'Chain checksum mismatch.')
            else:
                pyro.set_rng_seed(config.seed+chain+(1000 if grouped else 0))
                small=['alpha','beta','beta_role','beta_exp','log_k','log_sigma_player','log_sigma_pc','log_sigma_champion']
                if grouped:small+=['log_sigma_group','group_free']
                kernel=NUTS(pyro_model,target_accept_prob=config.target_accept,max_tree_depth=settings['max_tree_depth'],
                            full_mass=[tuple(small)],adapt_mass_matrix=config.profile=='full',jit_compile=False)
                sampler=MCMC(kernel,num_samples=settings['draws'],warmup_steps=settings['warmup'],num_chains=1,disable_progbar=False)
                tick=time.perf_counter();sampler.run(data,dimensions,grouped=grouped)
                values={k:v.detach().cpu().numpy() for k,v in sampler.get_samples().items()}
                buffer=io.BytesIO();np.savez_compressed(buffer,**values);atomic_bytes(sample_file,buffer.getvalue())
                d=sampler.diagnostics();info=dict(chain=chain,seconds=time.perf_counter()-tick,divergences=len(d['divergences']['chain 0']),
                    acceptance_rate=float(d['acceptance rate']['chain 0']),samples_sha256=file_sha(sample_file))
                atomic_json(info_file,info)
            with np.load(sample_file) as a:values={k:a[k] for k in a.files}
            require('beta_rank' not in values and all(v.shape[0]==settings['draws'] and np.isfinite(v).all() for v in values.values()),'Invalid retained draws.')
            samples.append(values);infos.append(info)
            print(f'{model}: saved/verified chain {chain+1}; divergences={info["divergences"]}, acceptance={info["acceptance_rate"]:.3f}',flush=True)
    finally:torch.set_num_threads(previous)
    samples={k:np.stack([s[k] for s in samples]) for k in samples[0]}
    diagnostic=rank_mcmc_diagnostics(samples)
    passed=bool(np.isfinite(diagnostic[['rank_split_Rhat','bulk_ESS','tail_ESS']]).all().all()
        and diagnostic.rank_split_Rhat.max()<config.max_rhat and diagnostic.bulk_ESS.min()>config.min_ess
        and diagnostic.tail_ESS.min()>config.min_ess and sum(x['divergences'] for x in infos)==0
        and min(x['acceptance_rate'] for x in infos)>=.8)
    status='smoke_only' if config.profile=='smoke' else 'converged' if passed else 'unconverged'
    diagnostic.to_csv(folder/'sampling_diagnostics.csv',index=False)
    atomic_json(folder/'fit_status.json',dict(status=status,diagnostic_gate_passed=passed,chains=infos,
        thresholds=dict(max_rhat=config.max_rhat,min_bulk_tail_ess=config.min_ess,max_divergences=0,min_acceptance=.8),
        unavailable=['tree-depth saturation','BFMI']))
    return dict(model=model,design=design,samples=samples,status=status,diagnostics=diagnostic,chain_info=infos)


def predict_draws(design,frame,samples,*,mode):
    # Empty rank contribution is deterministic padding for the reusable predictor,
    # not a sampled coefficient or a change to the Master likelihood.
    return broad.predict_draws(design,frame,dict(samples,beta_rank=np.zeros((len(samples['alpha']),0))),mode=mode)


def posterior_mean(result,frame,mode):
    draws=broad.flatten_draws(result)
    return np.concatenate([predict_draws(result['design'],frame.iloc[i:i+128],draws,mode=mode).mean(axis=0) for i in range(0,len(frame),128)])


def run_fits(prepared,out,config,*,enabled=False):
    if not enabled:print('Preparation only: no sampling or evaluation.');return None
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data].copy()
    if config.sampling['train_match_limit'] is not None:
        ids=train.sort_values(['timestamp','match_id']).match_id.drop_duplicates().head(config.sampling['train_match_limit'])
        train=train.loc[train.match_id.isin(ids)].copy()
    design=Design(train);results={}
    for grouped in (False,True):
        name='grouped' if grouped else 'shared';print(name,len(train),'training rows',config.sampling,flush=True)
        results[name]=fit_one(train,design,out,config,grouped=grouped)
        if config.profile=='full':require(results[name]['status']=='converged',f'{name}: full diagnostic gate failed; evaluation forbidden.')
    return results


def evaluate_split(prepared,results,out,*,split):
    require(split in ('validation','test') and set(results)=={'shared','grouped'},'Invalid split/model set.')
    require(all(r['status']=='converged' for r in results.values()),'Only full converged models authorize scoring.')
    out=Path(out)
    require(split!='test' or (out/'validation_decision.json').exists(),'Record validation decision before test scoring.')
    frame=prepared['played'].loc[prepared['played'].split.eq(split)&prepared['played'].usable_data]
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data]
    identity=json.loads(canonical(dict(helper_sha256=file_sha(__file__),rows_sha256=sha(frame.to_json(orient='split',index=False).encode()),
        fits={name:file_sha(out/'fits'/name/'fit_manifest.json') for name in results},
        chains={name:[file_sha(out/'fits'/name/f'chain_{i}.npz') for i in range(4)] for name in results})))
    mf=out/f'{split}_evaluation.json';files=[f'{split}_metrics.csv',f'{split}_calibration.csv',f'{split}_predictions.parquet']
    if mf.exists():
        saved=json.loads(mf.read_text());require(saved['identity']==identity,'Evaluation configuration changed.')
        require(all(file_sha(out/n)==h for n,h in saved['output_sha256'].items()),'Evaluation checksum mismatch.')
        return dict(metrics=pd.read_csv(out/files[0]),calibration=pd.read_csv(out/files[1]),predictions=pd.read_parquet(out/files[2]))
    predictions={('training_prevalence','benchmark'):np.full(len(frame),train.win.mean()),('raw_matchup','benchmark'):frame.matchup_probability.to_numpy()}
    for name,result in results.items():
        for mode in ('general','personalized'):predictions[(name,mode)]=posterior_mean(result,frame.drop(columns='win'),mode)
    scores=[];calibrations=[];rows=[]
    for (name,mode),probability in predictions.items():
        score,cal=broad.evaluate(frame,probability,name,mode,split);scores.append(score);calibrations.append(cal)
        f=frame[KEY+['target_row_id']].copy();f['probability']=probability;f['model']=name;f['mode']=mode;f['split']=split;rows.append(f)
    metrics=pd.DataFrame(scores);calibration=pd.concat(calibrations,ignore_index=True);predictions=pd.concat(rows,ignore_index=True)
    metrics.to_csv(out/files[0],index=False);calibration.to_csv(out/files[1],index=False);predictions.to_parquet(out/files[2],index=False)
    atomic_json(mf,dict(created_at=stamp(),identity=identity,prior_outcome_exposure=True,output_sha256={n:file_sha(out/n) for n in files}))
    return dict(metrics=metrics,calibration=calibration,predictions=predictions)


def eda_outputs(prepared,out):
    import matplotlib.pyplot as plt
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data].copy()
    train['usage_share']=train.n_recorded/train.prior_total_games.where(train.prior_total_games.gt(0))
    usage=['0–20%','>20–40%','>40–60%','>60–80%','>80–100%']
    train['usage_bin']=pd.cut(train.usage_share,[0,.2,.4,.6,.8,1],labels=usage,include_lowest=True)
    train['choice']=np.where(train.champion_id.eq(train.prior_main_champion_id),'Recorded main','Another champion')
    tables={};out=Path(out)
    for col,levels,title,name in [('experience_bin',broad.BINS,'Recorded champion experience','experience'),
            ('usage_bin',usage,'OTP rate (prior champion share)','usage'),
            ('choice',['Another champion','Recorded main'],'Recorded main versus another champion','main')]:
        f=train.groupby(col,observed=True).agg(wins=('win','sum'),games=('win','size'),players=('puuid','nunique'),baseline=('matchup_probability','mean')).reindex(levels)
        f[['games','players','wins']]=f[['games','players','wins']].fillna(0).astype(int)
        n=f.games.where(f.games.gt(0));p=f.wins/n;z=1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        f['observed_wr']=p;f['ci_low']=center-half;f['ci_high']=center+half
        f.to_csv(out/f'eda_{name}.csv');tables[name]=f
        fig,ax=plt.subplots(figsize=(11,6));x=np.arange(len(f))
        ax.errorbar(x,100*p,yerr=100*np.array([p-f.ci_low,f.ci_high-p]),fmt='o-',color='#2563A6',capsize=5,label='Observed WR; approximate 95% Wilson')
        ax.plot(x,100*f.baseline,'s--',color='#C58B2A',label='Mean raw Global Master+ benchmark')
        ax.set(xticks=x,xticklabels=[f'{v}\n{r.games:,} games\n{r.players} players' for v,r in zip(levels,f.itertuples())],
               ylabel='Win rate (%)',xlabel=title,title=title+' — matched training player-games',ylim=(0,100))
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
        fig.text(.5,.015,'Identical rows for observed/benchmark rates. Wilson intervals ignore repeated-player/match dependence.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'figures'/f'eda_{name}.png',dpi=140);plt.close(fig)
    coverage=train.groupby(['mastery_group','experience_bin'],observed=False).agg(games=('match_id','size'),players=('puuid','nunique')).reset_index()
    coverage.to_csv(out/'eda_group_experience.csv',index=False);tables['group_experience']=coverage
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,value in zip(axes,['games','players']):
        grid=coverage.pivot(index='mastery_group',columns='experience_bin',values=value).reindex(index=GROUPS,columns=broad.BINS).fillna(0)
        ax.imshow(np.log1p(grid.to_numpy(dtype=float)),aspect='auto',cmap='Blues')
        for i in range(len(GROUPS)):
            for j in range(len(broad.BINS)):ax.text(j,i,str(int(grid.iloc[i,j])),ha='center',va='center',fontsize=8)
        ax.set(xticks=range(6),xticklabels=broad.BINS,yticks=range(11),yticklabels=GROUPS,title='Training '+value,xlabel='Recorded prior champion games')
    fig.tight_layout();fig.savefig(out/'figures/eda_group_support.png',dpi=140);plt.close(fig)
    return tables


def recommendations(prepared,results):
    pairs,reference,alternative=broad.comparison_pairs(prepared);output=[]
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data]
    pc=train.groupby(['puuid','model_champion_id']).size();cc=train.groupby('model_champion_id').size();gp=train.groupby('mastery_group').puuid.nunique()
    for name,result in results.items():
        samples=broad.flatten_draws(result)
        for mode in ('general','personalized'):
            for i in range(0,len(pairs),128):
                r=reference.iloc[i:i+128];a=alternative.iloc[i:i+128]
                pr=predict_draws(result['design'],r,samples,mode=mode);pa=predict_draws(result['design'],a,samples,mode=mode)
                delta=100*(pa-pr);low,high=np.quantile(delta,[.025,.975],axis=0)
                batch=pairs.iloc[i:i+128].copy();batch['model']=name;batch['mode']=mode
                batch['reference_probability']=pr.mean(axis=0);batch['alternative_probability']=pa.mean(axis=0)
                batch['difference_pp']=delta.mean(axis=0);batch['low_pp']=low;batch['high_pp']=high
                batch['probability_alternative_higher']=(delta>0).mean(axis=0)
                batch['preference']=np.select([low>0,high<0],['Alternative','Reference'],default='Uncertain')
                batch['point_prefers_alternative']=batch.difference_pp.gt(0)
                for prefix,choice in [('reference',r),('alternative',a)]:
                    batch[prefix+'_pc_training_games']=[int(pc.get((p,int(c)),0)) for p,c in zip(choice.puuid,choice.model_champion_id)]
                    batch[prefix+'_pc_training_supported']=batch[prefix+'_pc_training_games'].gt(0)
                    batch[prefix+'_champion_training_games']=[int(cc.get(int(c),0)) for c in choice.model_champion_id]
                    batch[prefix+'_group_training_players']=[int(gp.get(g,0)) for g in choice.mastery_group]
                    batch[prefix+'_sparse_training_support']=(batch[prefix+'_pc_training_games'].lt(5)|batch[prefix+'_champion_training_games'].lt(20)|batch[prefix+'_group_training_players'].lt(5))
                output.append(batch)
    if not output:return pd.DataFrame(),pd.DataFrame()
    frame=pd.concat(output,ignore_index=True);summary=[]
    for key,f in frame.groupby(['model','mode','reference']):
        for event,v in [('point_prefers_alternative',f.point_prefers_alternative),('interval_favors_alternative',f.preference.eq('Alternative')),('uncertain',f.preference.eq('Uncertain'))]:
            z=f[KEY].copy();z['value']=v.astype(float).to_numpy();targets=z.groupby(KEY).value.mean()
            summary.append(dict(model=key[0],mode=key[1],reference=key[2],event=event,pairs=len(f),targets=len(targets),players=f.puuid.nunique(),
                                pair_weighted=z.value.mean(),equal_target=targets.mean(),equal_player=targets.groupby(level='puuid').mean().mean()))
    return frame,pd.DataFrame(summary)


def curve_outputs(train,results,out):
    import matplotlib.pyplot as plt
    rows=[]
    for name,result in results.items():
        samples=broad.flatten_draws(result);design=result['design'];reference=train[list(CONTINUOUS)].median().to_numpy(dtype=float)
        base=samples['alpha']+samples['beta']@((reference-design.mean)/design.scale)
        amplitude=np.repeat(samples['beta_exp'][:,None],11,axis=1)
        if 'group_free' in samples:amplitude+=np.exp(samples['log_sigma_group'])[:,None]*(samples['group_free']@design.group_basis.T)
        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for j,g in enumerate(GROUPS):
            support=train.loc[train.mastery_group.eq(g)];ax=axes.flat[j]
            if support.empty:ax.axis('off');continue
            low=float(support.n_recorded.min());cap=min(float(support.n_recorded.quantile(.95)),float(np.expm1(reference[1])))
            grid=np.linspace(low,cap,60) if cap>low else np.array([low])
            prob=broad.expit(base[:,None]+amplitude[:,j,None]*grid/(grid+np.exp(samples['log_k'])[:,None]))
            lower,median,upper=np.quantile(prob,[.025,.5,.975],axis=0)
            ax.plot(grid,median);ax.fill_between(grid,lower,upper,alpha=.2)
            rug=support.loc[support.n_recorded.between(low,cap),'n_recorded'];ax.plot(rug,np.full(len(rug),.02),'|',transform=ax.get_xaxis_transform(),alpha=.15)
            ax.set(title=f'{g}: {len(support)} games / {support.puuid.nunique()} players',xlabel='Recorded prior champion games',ylabel='Win probability',ylim=(0,1))
            for i,n in enumerate(grid):rows.append(dict(model=name,group=g,n_recorded=n,median=median[i],low=lower[i],high=upper[i],
                training_games=len(support),training_players=support.puuid.nunique(),support_95th=float(support.n_recorded.quantile(.95)),context_cap=cap,
                context='Master-labeled/top; training-median continuous features; zero champion/player/PC baselines'))
        axes.flat[-1].axis('off');fig.suptitle(name+' — conditional associations; supported reference slice, not causal practice gains')
        fig.tight_layout();fig.savefig(Path(out)/'figures'/f'{name}_group_curves.png',dpi=140);plt.close(fig)
    frame=pd.DataFrame(rows);frame.to_parquet(Path(out)/'group_experience_curves.parquet',index=False)
    return frame


def posterior_summary(results,out):
    rows=[];groups=[]
    def row(model,label,v,sign=False):
        low,median,high=np.quantile(v,[.025,.5,.975])
        return dict(model=model,quantity=label,median=median,low_95=low,high_95=high,probability_positive=np.mean(v>0) if sign else np.nan)
    for name,result in results.items():
        s=broad.flatten_draws(result)
        for label,v,sign in [('matchup_logodds_per_training_SD',s['beta'][:,0],True),('shared_experience_amplitude_logodds',s['beta_exp'],True),
                             ('k_recorded_games',np.exp(s['log_k']),False),*[(k[4:],np.exp(s[k]),False) for k in ('log_sigma_player','log_sigma_pc','log_sigma_champion')]]:
            rows.append(row(name,label,v,sign))
        rows.append(row(name,'matchup_logodds_per_2_raw_pp',s['beta'][:,0]*.02/result['design'].scale[0],True))
        if name=='grouped':
            rows.append(row(name,'sigma_group_contrast',np.exp(s['log_sigma_group'])))
            amps=s['beta_exp'][:,None]+np.exp(s['log_sigma_group'])[:,None]*(s['group_free']@result['design'].group_basis.T)
            for j,g in enumerate(GROUPS):groups.append(row(name,g,amps[:,j],True))
    frame=pd.DataFrame(rows);frame.to_csv(Path(out)/'posterior_summary.csv',index=False)
    pd.DataFrame(groups).to_csv(Path(out)/'group_amplitude_summary.csv',index=False)
    return frame


def cohort_comparison(root,prepared,results,out,metrics):
    """Read broad saved fits/results; derived baseline scores go ONLY to Master outputs."""
    base=Path(root)/'hlr_model/data/analyses/broad_rank_mastery_v1'/BROAD_VERSION
    prep=json.loads((base/'preparation_manifest.json').read_text());audit=json.loads((base/'final_artifact_manifest.json').read_text())['sha256']
    for file in ['prepared_played.parquet','evaluation_metrics.csv','paired_recommendations.parquet','personalization_summary.csv']:
        expected=prep['output_sha256'].get(file,audit.get(file));require(file_sha(base/file)==expected,f'Broad input changed: {file}')
    broad_played=pd.read_parquet(base/'prepared_played.parquet')
    bm=pd.read_csv(base/'evaluation_metrics.csv');bm['cohort']='Bronze–Diamond';mm=metrics.copy();mm['cohort']='Master-labeled'
    extra=[];supports=[];associations=[]
    for cohort,frame in [('Bronze–Diamond',broad_played),('Master-labeled',prepared['played'])]:
        train=frame.loc[frame.split.eq('train')&frame.usable_data];pcs=train.groupby(['puuid','model_champion_id']).size()
        supports.append(dict(cohort=cohort,training_rows=len(train),players=train.puuid.nunique(),matches=train.match_id.nunique(),pc_pairs=len(pcs),median_games_per_pc=pcs.median(),fraction_pc_once=pcs.eq(1).mean(),fraction_pc_lt5=pcs.lt(5).mean(),players_n_ge100=train.loc[train.n_recorded.ge(100)].puuid.nunique()))
        if cohort=='Bronze–Diamond':
            for split in ['validation','test']:
                f=frame.loc[frame.split.eq(split)&frame.usable_data]
                for name,prob in [('training_prevalence',np.full(len(f),train.win.mean())),('raw_matchup',f.matchup_probability.to_numpy())]:
                    score,_=broad.evaluate(f,prob,name,'benchmark',split);score['cohort']=cohort;extra.append(score)
    for name in ('shared','grouped'):
        folder=base/'fits'/name;status=json.loads((folder/'fit_status.json').read_text());manifest=json.loads((folder/'fit_manifest.json').read_text())
        require(file_sha(folder/'fit_manifest.json')==audit[f'fits/{name}/fit_manifest.json'] and file_sha(folder/'fit_status.json')==audit[f'fits/{name}/fit_status.json'],'Broad fit manifest/status changed.')
        require(status['status']=='converged' and status['diagnostic_gate_passed'],'Broad comparison requires converged saved fits.')
        saved=[]
        for i in range(4):
            path=folder/f'chain_{i}.npz';info=json.loads(path.with_suffix('.json').read_text())
            require(file_sha(path)==info['samples_sha256'],'Broad posterior checksum mismatch.')
            with np.load(path) as a:saved.append({k:a[k] for k in ('beta','beta_exp','log_k','log_sigma_player','log_sigma_pc','log_sigma_champion')})
        samples={k:np.concatenate([s[k] for s in saved]) for k in saved[0]}
        for cohort,s,scale in [('Bronze–Diamond',samples,manifest['design']['scale'][0]),('Master-labeled',broad.flatten_draws(results[name]),results[name]['design'].scale[0])]:
            for quantity,v in [('matchup_logodds_per_2_raw_pp',s['beta'][:,0]*.02/scale),('shared_experience_amplitude_logodds',s['beta_exp']),('k_recorded_games',np.exp(s['log_k'])),*[(k[4:],np.exp(s[k])) for k in ('log_sigma_player','log_sigma_pc','log_sigma_champion')]]:
                q=np.quantile(v,[.025,.5,.975]);associations.append(dict(cohort=cohort,model=name,quantity=quantity,median=q[1],low_95=q[0],high_95=q[2]))
    pd.concat([bm,mm,pd.DataFrame(extra)],ignore_index=True).to_csv(Path(out)/'cohort_performance_comparison.csv',index=False)
    pd.DataFrame(supports).to_csv(Path(out)/'cohort_support_comparison.csv',index=False)
    pd.DataFrame(associations).to_csv(Path(out)/'cohort_association_comparison.csv',index=False)
    overlap=dict(shared_players=len(set(broad_played.puuid)&set(prepared['played'].puuid)),
        shared_matches=len(set(broad_played.match_id)&set(prepared['played'].match_id)),
        duplicate_player_match_rows=len(set(map(tuple,broad_played[KEY].to_numpy()))&set(map(tuple,prepared['played'][KEY].to_numpy()))))
    recommendation_rows=[]
    for cohort,pairs in [('Bronze–Diamond',pd.read_parquet(base/'paired_recommendations.parquet')),
                         ('Master-labeled',pd.read_parquet(Path(out)/'paired_recommendations.parquet'))]:
        keys=['model',*KEY,'reference','reference_champion_id','alternative_champion_id']
        ref=pairs.loc[pairs['mode'].eq('general')];per=pairs.loc[pairs['mode'].eq('personalized')]
        joined=ref.merge(per,on=keys,suffixes=('_ref','_per'),validate='one_to_one')
        for key,g in joined.groupby(['model','reference']):
            recommendation_rows.append(dict(cohort=cohort,model=key[0],reference=key[1],pairs=len(g),
                point_preference_changed=g.point_prefers_alternative_ref.ne(g.point_prefers_alternative_per).mean(),
                reference_uncertain=g.preference_ref.eq('Uncertain').mean(),personalized_uncertain=g.preference_per.eq('Uncertain').mean(),
                alternative_sparse=g.alternative_sparse_training_support_per.mean(),
                median_personalized_interval_width_pp=(g.high_pp_per-g.low_pp_per).median()))
    pd.DataFrame(recommendation_rows).to_csv(Path(out)/'cohort_recommendation_comparison.csv',index=False)
    atomic_json(Path(out)/'cohort_comparison_scope.json',dict(broad_version=BROAD_VERSION,broad_models_refitted=False,broad_files_modified=False,
        broad_baselines='Derived descriptively here on original frozen broad rows; stored only in Master outputs.',formal_rank_interaction_test=False,overlap=overlap,ready_for_confirmatory_pooled_rank_analysis=False,
        limits=['Unverified Master seed ranks versus collection-time broad ranks','Global Master+ versus exact Bronze–Diamond benchmark tiers','Both test sets previously examined','Cross-cohort overlap counts are reported; shared matches must remain together in any pooled analysis']))


def finalize(prepared,results,out,config,*,decision,root):
    require(config.profile=='full' and bool(decision.strip()),'Full fit and written validation decision required.')
    out=Path(out);validation=evaluate_split(prepared,results,out,split='validation')
    record=json.loads(canonical(dict(decision=decision,validation_sha256=file_sha(out/'validation_evaluation.json'),config=asdict(config),
        prior_outcome_exposure=True,model_changes_after_test=False)))
    file=out/'validation_decision.json'
    if file.exists():require(json.loads(file.read_text())['record']==record,'The recorded validation decision is frozen.')
    else:atomic_json(file,dict(created_at=stamp(),record=record))
    test=evaluate_split(prepared,results,out,split='test')
    metrics=pd.concat([validation['metrics'],test['metrics']],ignore_index=True);metrics.to_csv(out/'evaluation_metrics.csv',index=False)
    calibration=pd.concat([validation['calibration'],test['calibration']],ignore_index=True);calibration.to_csv(out/'calibration.csv',index=False)
    pd.concat([validation['predictions'],test['predictions']],ignore_index=True).to_parquet(out/'heldout_predictions.parquet',index=False)
    pairs,summary=recommendations(prepared,results);pairs.to_parquet(out/'paired_recommendations.parquet',index=False);summary.to_csv(out/'recommendation_summary.csv',index=False)
    train=prepared['played'].loc[prepared['played'].split.eq('train')&prepared['played'].usable_data]
    curve_outputs(train,results,out);posterior_summary(results,out)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for key,g in calibration.loc[calibration.split.eq('test')].groupby(['model','mode']):ax.plot(g.predicted,g.observed,'o-',label=' / '.join(key))
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Predicted probability',ylabel='Observed win fraction',title='Previously exposed final-test calibration; bin counts in calibration.csv');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'figures/test_calibration.png',dpi=140);plt.close(fig)
    cohort_comparison(root,prepared,results,out,metrics)
    diagnostic=pd.DataFrame([dict(model=name,status=r['status'],max_Rhat=r['diagnostics'].rank_split_Rhat.max(),min_bulk_ESS=r['diagnostics'].bulk_ESS.min(),
        min_tail_ESS=r['diagnostics'].tail_ESS.min(),divergences=sum(c['divergences'] for c in r['chain_info']),min_acceptance=min(c['acceptance_rate'] for c in r['chain_info'])) for name,r in results.items()])
    diagnostic.to_csv(out/'diagnostic_summary.csv',index=False)
    atomic_json(out/'modeling_complete.json',dict(status='both_full_models_converged_and_evaluated',created_at=stamp(),
        prior_outcome_exposure=True,specifications='Frozen shared primary and grouped secondary; rank absorbed in intercept',limits=LIMITS))
    return dict(metrics=metrics,calibration=calibration,recommendation_summary=summary,diagnostics=diagnostic)
