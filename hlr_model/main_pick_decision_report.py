"""Read-only shared-model decision reporting; no model/backend/evaluation imports.

Outputs are three derived report artifacts per cohort. Existing prepared data,
fit manifests, predictions and evaluated metrics are never written.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['puuid','match_id']
MODES={'general':'Reference player','personalized':'Personalized'}
CATEGORIES=('Alternative favored','Main favored over selected alternative','Uncertain')
COHORTS={
    'Master':('master_cohort_mastery_v1','20261028T214327398991Z_4beeb7d4','master_cohort_modeling.py'),
    'Bronze–Diamond':('broad_rank_mastery_v1','20261028T193830350096Z_5a469939','broad_rank_modeling.py'),
}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def require(value,message):
    if not bool(value):raise ValueError(message)


def verified_inputs(root,cohort):
    folder,version,helper=COHORTS[cohort];root=Path(root).resolve()
    source=root/'hlr_model/data/analyses'/folder/version
    preparation=json.loads((source/'preparation_manifest.json').read_text())
    final=json.loads((source/'final_artifact_manifest.json').read_text())
    hashes=final.get('output_sha256',final.get('sha256'))
    identity=preparation['identity'];identity=json.loads(identity) if isinstance(identity,str) else identity
    require(file_sha(root/'hlr_model'/helper)==identity['helper_sha256'],'Fit-time helper changed.')
    names=['prepared_played.parquet','prepared_alternatives.parquet','champion_grouping_lookup.parquet',
           'paired_recommendations.parquet','fits/shared/fit_manifest.json','fits/shared/fit_status.json',
           'fits/shared/sampling_diagnostics.csv','test_evaluation.json','modeling_complete.json']
    verified={}
    for name in names:
        expected=preparation['output_sha256'].get(name,hashes.get(name))
        require(expected is not None and file_sha(source/name)==expected,f'Artifact checksum mismatch: {name}')
        verified[name]=expected
    manifest=json.loads((source/'fits/shared/fit_manifest.json').read_text())
    status=json.loads((source/'fits/shared/fit_status.json').read_text())
    require(manifest['model']=='shared' and manifest['backend_version']=='1.9.2','Wrong saved model/backend.')
    require(manifest['sampling']==dict(chains=4,draws=1200,warmup=600,max_tree_depth=9,train_match_limit=None),'Wrong full-fit profile.')
    require(status['status']=='converged' and status['diagnostic_gate_passed'],'Shared full diagnostic gate did not pass.')
    diagnostic=pd.read_csv(source/'fits/shared/sampling_diagnostics.csv')
    require(np.isfinite(diagnostic[['rank_split_Rhat','bulk_ESS','tail_ESS']]).all().all()
        and diagnostic.rank_split_Rhat.max()<1.01 and diagnostic.bulk_ESS.min()>400 and diagnostic.tail_ESS.min()>400,'Saved diagnostics fail.')
    require(len(status['chains'])==4 and all(c['divergences']==0 and c['acceptance_rate']>=.8 for c in status['chains']),'Chain diagnostics fail.')
    # Inspect parquet schemas before reading; observed outcomes are not loaded.
    import pyarrow.parquet as pq
    observed_columns=KEY+['target_row_id','timestamp','split','champion_id','prior_main_champion_id',
        'prior_champion_games','prior_total_games','n_recorded','historical_wr_beta11','mastery_group',
        'matchup_probability','benchmark_valid_rate','usable_data','data_exclusion_reasons',
        'included_in_primary_model','same_role_prior_games','lane','opponent_champion_id']
    alternative_columns=KEY+['target_row_id','timestamp','split','candidate_champion_id','n_recorded',
        'historical_wr_beta11','mastery_group','matchup_probability','benchmark_valid_rate','usable_data',
        'data_exclusion_reasons','prior_same_role_games','prior_all_role_games','prior_total_games',
        'candidate_outcome','lane','opponent_champion_id','latest_all_role_prior_end_timestamp']
    a_schema=set(pq.read_schema(source/'prepared_alternatives.parquet').names)
    role_time='latest_same_role_end' if 'latest_same_role_end' in a_schema else 'latest_same_role_prior_end_timestamp'
    alternative_columns.append(role_time)
    require(set(observed_columns)<=set(pq.read_schema(source/'prepared_played.parquet').names),'Observed schema incompatible.')
    require(set(alternative_columns)<=a_schema,'Alternative schema incompatible.')
    pair_schema=set(pq.read_schema(source/'paired_recommendations.parquet').names)
    require(set(KEY+['model','mode','reference','reference_champion_id','alternative_champion_id','reference_probability',
        'alternative_probability','difference_pp','low_pp','high_pp','probability_alternative_higher'])<=pair_schema,'Pair schema incompatible.')
    observed=pd.read_parquet(source/'prepared_played.parquet',columns=observed_columns,filters=[('split','==','test')])
    alternatives=pd.read_parquet(source/'prepared_alternatives.parquet',columns=alternative_columns,filters=[('split','==','test')])
    pairs=pd.read_parquet(source/'paired_recommendations.parquet',filters=[('model','==','shared'),('reference','==','main')])
    names=pd.read_parquet(source/'champion_grouping_lookup.parquet',columns=['champion_id','canonical_champion'])
    require(not observed.duplicated(KEY).any() and observed.target_row_id.is_unique,'Duplicate observed game.')
    require(not alternatives.duplicated(KEY+['candidate_champion_id']).any(),'Duplicate recorded alternative.')
    require(not pairs.duplicated(KEY+['mode','alternative_champion_id']).any(),'Duplicate saved main comparison.')
    require(set(pairs['mode'])==set(MODES),'Missing or unknown saved prediction mode.')
    require(alternatives.candidate_outcome.isna().all(),'Unplayed alternative has an outcome.')
    require(names.champion_id.is_unique,'Champion names are not unique by ID.')
    return source,observed,alternatives,pairs,names.set_index('champion_id').canonical_champion.to_dict(),verified,role_time


def select_top_alternatives(pairs):
    """Select by saved posterior mean only; never select using the interval."""
    counts=pairs.groupby(KEY+['mode']).alternative_champion_id.nunique().rename('eligible_alternatives')
    top=pairs.sort_values(KEY+['mode','alternative_probability','alternative_champion_id'],
        ascending=[True,True,True,False,True],kind='stable').drop_duplicates(KEY+['mode']).copy()
    top=top.merge(counts,on=KEY+['mode'],validate='one_to_one')
    top['classification']=np.select([top.low_pp.gt(0),top.high_pp.lt(0)],CATEGORIES[:2],default=CATEGORIES[2])
    top['alternative_has_higher_mean']=top.alternative_probability.gt(top.reference_probability)
    # Apply smallest champion-ID tie-breaking also to an exact main/alternative mean tie.
    leave=top.alternative_has_higher_mean | (top.alternative_probability.eq(top.reference_probability)
        &top.alternative_champion_id.lt(top.reference_champion_id))
    top['highest_mean_choice_id']=np.where(leave,top.alternative_champion_id,top.reference_champion_id).astype(int)
    top['point_prefers_leaving_main']=leave
    return top


def build_report(root,cohort):
    source,o,a,pairs,names,verified,role_time=verified_inputs(root,cohort)
    known=o.loc[o.prior_main_champion_id.notna()]
    main=known.loc[known.champion_id.eq(known.prior_main_champion_id)].copy()
    require(main.n_recorded.eq(main.prior_champion_games).all() and main.n_recorded.le(main.prior_total_games).all(),'Main experience inconsistent.')
    main_rate=main.loc[main.benchmark_valid_rate & main.matchup_probability.between(0,1)].copy()
    main_supported=main_rate.loc[main_rate.usable_data & main_rate.same_role_prior_games.gt(0)].copy()
    # Reconstruct the source's candidate universe using saved pregame features,
    # solely to reconcile coverage; no probabilities or intervals are recomputed.
    candidates=a.merge(main_supported[KEY+['matchup_probability','opponent_champion_id','lane']],on=KEY,
        suffixes=('','_main'),validate='many_to_one')
    usable=candidates.loc[candidates.usable_data & candidates.benchmark_valid_rate
        &candidates.prior_same_role_games.gt(0)&candidates.matchup_probability.between(0,1)].copy()
    expected=usable.loc[usable.matchup_probability.gt(usable.matchup_probability_main)].copy()
    require(expected.candidate_champion_id.ne(expected.opponent_champion_id_main).all(),'Alternative equals opponent.')
    require(expected.lane.eq(expected.lane_main).all(),'Alternative crosses roles.')
    require(expected[role_time].lt(expected.timestamp).all()
        and expected.latest_all_role_prior_end_timestamp.lt(expected.timestamp).all(),'Alternative chronology invalid.')
    require(expected.n_recorded.eq(expected.prior_all_role_games).all()
        and expected.n_recorded.le(expected.prior_total_games).all(),'Alternative experience inconsistent.')
    selected_pairs=pairs.merge(main[KEY],on=KEY,how='inner',validate='many_to_one')
    expected_keys=set(map(tuple,expected[KEY+['candidate_champion_id']].to_numpy()))
    for mode,g in selected_pairs.groupby('mode'):
        actual=set(map(tuple,g[KEY+['alternative_champion_id']].to_numpy()))
        require(actual==expected_keys,f'Saved main comparisons do not reconcile with recorded eligibility in {mode}.')
    require(selected_pairs.reference.eq('main').all() and selected_pairs.model.eq('shared').all(),'Mixed reference/model types.')
    main_columns=KEY+['target_row_id','timestamp','champion_id','n_recorded','historical_wr_beta11',
        'mastery_group','matchup_probability','same_role_prior_games','prior_total_games','lane','opponent_champion_id']
    checked=selected_pairs.merge(main[main_columns],on=KEY,suffixes=('','_observed'),validate='many_to_one')
    require(checked.reference_champion_id.eq(checked.champion_id).all()
        and checked.target_row_id.eq(checked.target_row_id_observed).all(),'Reference is not the actual recorded main pick.')
    for left,right in [('reference_experience','n_recorded'),('reference_historical_wr','historical_wr_beta11'),
                       ('reference_raw_matchup_probability','matchup_probability')]:
        np.testing.assert_allclose(checked[left],checked[right],rtol=0,atol=1e-12)
    require(checked.reference_group.eq(checked.mastery_group).all(),'Main group mismatch.')
    require(checked.alternative_champion_id.ne(checked.reference_champion_id).all(),'Alternative equals the main.')
    expected_features=expected[KEY+['candidate_champion_id','n_recorded','historical_wr_beta11','mastery_group','matchup_probability']].rename(columns={
        'candidate_champion_id':'alternative_champion_id','n_recorded':'alternative_n_check','historical_wr_beta11':'alternative_wr_check',
        'mastery_group':'alternative_group_check','matchup_probability':'alternative_benchmark_check'})
    checked=checked.merge(expected_features,on=KEY+['alternative_champion_id'],validate='many_to_one')
    for left,right in [('alternative_experience','alternative_n_check'),('alternative_historical_wr','alternative_wr_check'),
                       ('alternative_raw_matchup_probability','alternative_benchmark_check')]:
        np.testing.assert_allclose(checked[left],checked[right],rtol=0,atol=1e-12)
    require(checked.alternative_group.eq(checked.alternative_group_check).all(),'Alternative group mismatch.')
    np.testing.assert_allclose(checked.difference_pp,100*(checked.alternative_probability-checked.reference_probability),rtol=0,atol=1e-10)
    require(np.isfinite(checked[['reference_probability','alternative_probability','difference_pp','low_pp','high_pp','probability_alternative_higher']]).all().all(),'Nonfinite saved predictions.')
    require(checked.probability_alternative_higher.between(0,1).all() and checked.low_pp.le(checked.high_pp).all(),'Invalid saved paired intervals/probabilities.')
    spread=checked.groupby(KEY+['mode']).reference_probability.agg(lambda s:s.max()-s.min())
    require(spread.le(1e-12).all(),'Main prediction varies with alternative within a mode.')
    top=select_top_alternatives(selected_pairs)
    keep=KEY+['target_row_id','mode','reference_champion_id','alternative_champion_id','reference_experience','alternative_experience',
        'reference_historical_wr','alternative_historical_wr','reference_group','alternative_group','reference_raw_matchup_probability',
        'alternative_raw_matchup_probability','reference_probability','alternative_probability','difference_pp','low_pp','high_pp',
        'probability_alternative_higher','eligible_alternatives','classification','alternative_has_higher_mean',
        'highest_mean_choice_id','point_prefers_leaving_main','reference_sparse_training_support','alternative_sparse_training_support',
        'reference_pc_training_games','alternative_pc_training_games']
    decisions=top[keep].rename(columns={
        'reference_champion_id':'main_champion_id','reference_experience':'main_recorded_games',
        'alternative_experience':'alternative_recorded_games','reference_historical_wr':'main_historical_wr_beta11',
        'alternative_historical_wr':'alternative_historical_wr_beta11','reference_group':'main_group',
        'reference_raw_matchup_probability':'main_raw_matchup_probability','reference_probability':'main_predicted_probability',
        'reference_sparse_training_support':'main_sparse_training_support','reference_pc_training_games':'main_pc_training_games'}).copy()
    decisions=decisions.merge(main[KEY+['timestamp','lane','opponent_champion_id','prior_total_games']],on=KEY,validate='many_to_one')
    decisions['cohort']=cohort;decisions['source_version']=source.name
    decisions['main_champion']=decisions.main_champion_id.map(names)
    decisions['alternative_champion']=decisions.alternative_champion_id.map(names)
    require(decisions[['main_champion','alternative_champion']].notna().all().all(),'Unknown champion name.')
    labels={p:f'Player {i+1:03d}' for i,p in enumerate(sorted(o.puuid.unique()))}
    decisions['anonymous_player']=decisions.puuid.map(labels)
    decisions['mode']=decisions['mode'].map(MODES)
    decisions=decisions.sort_values(['timestamp',*KEY,'mode']).reset_index(drop=True)
    n=len(expected[KEY].drop_duplicates());require(len(decisions)==2*n and n>0,'Expected one selected row per game/mode.')
    require(not decisions.duplicated(KEY+['mode']).any(),'Duplicate game/mode decision.')
    selected_keys=expected[KEY].drop_duplicates();eligible_games=main.merge(selected_keys,on=KEY,validate='one_to_one')
    summary=[]
    def add(section,measure,frame,denominator,mode='Both modes'):
        count=len(frame);summary.append(dict(cohort=cohort,section=section,mode=mode,measure=measure,count=count,
            denominator=denominator,percent=100*count/denominator if denominator else np.nan,
            players=frame.puuid.nunique(),unique_matches=frame.match_id.nunique()))
    for label,frame in [('Retained chronological test player-games',o),('Identifiable pregame main',known),
        ('Actual main-pick games',main),('Main-pick games with usable main benchmark',main_rate),
        ('Main-pick games with at least one eligible better-matchup alternative',eligible_games)]:
        add('coverage',label,frame,len(o))
    add('context','Test rows used in original predictive evaluation',o.loc[o.included_in_primary_model],len(o))
    # Disjoint reasons reconcile the last coverage loss, without discarding games silently.
    remaining=main_rate.copy()
    mask=~remaining.usable_data
    add('coverage_loss','Main history/features unusable',remaining.loc[mask],len(main_rate));remaining=remaining.loc[~mask]
    mask=remaining.same_role_prior_games.le(0)
    add('coverage_loss','Main lacks prior same-role support required by saved main-reference comparisons',remaining.loc[mask],len(main_rate));remaining=remaining.loc[~mask]
    for label,pool in [('No recorded same-role alternative',candidates),('No usable same-role alternative benchmark/history',usable),
                       ('No usable alternative with strictly better raw matchup',expected)]:
        present=set(map(tuple,pool[KEY].to_numpy()));mask=pd.Series([k not in present for k in remaining[KEY].itertuples(index=False,name=None)],index=remaining.index)
        add('coverage_loss',label,remaining.loc[mask],len(main_rate));remaining=remaining.loc[~mask]
    require(set(map(tuple,remaining[KEY].to_numpy()))==set(map(tuple,eligible_games[KEY].to_numpy())),'Coverage loss reasons do not reconcile.')
    for mode,frame in decisions.groupby('mode'):
        for category in CATEGORIES:add('classification',category,frame.loc[frame.classification.eq(category)],n,mode)
        add('point_preference','Alternative has strictly higher mean prediction',frame.loc[frame.alternative_has_higher_mean],n,mode)
        add('context','Exact main/selected-alternative mean ties',frame.loc[frame.alternative_probability.eq(frame.main_predicted_probability)],n,mode)
    r=decisions.loc[decisions['mode'].eq('Reference player')];p=decisions.loc[decisions['mode'].eq('Personalized')]
    matched=r.merge(p,on=KEY,suffixes=('_reference','_personalized'),validate='one_to_one')
    require(len(matched)==n and matched.main_champion_id_reference.eq(matched.main_champion_id_personalized).all(),'Mode denominators/main identities differ.')
    for label,left,right in [('Selected alternative changed','alternative_champion_id_reference','alternative_champion_id_personalized'),
        ('Highest-mean choice champion changed','highest_mean_choice_id_reference','highest_mean_choice_id_personalized'),
        ('Point preference switched between main and any alternative','point_prefers_leaving_main_reference','point_prefers_leaving_main_personalized'),
        ('Uncertainty classification changed','classification_reference','classification_personalized')]:
        add('personalization_change',label,matched.loc[matched[left].ne(matched[right])],n)
    summary=pd.DataFrame(summary)
    out=source/'reports/main_pick_decisions_v1';out.mkdir(parents=True,exist_ok=True)
    provenance=dict(source_version=source.name,input_sha256=verified,report_helper_sha256=file_sha(__file__),
        denominator='Prepared split==test, then actual recorded main pick with usable saved main-reference comparison; not pair-weighted.',
        alternative_selection='Maximum saved alternative_probability within game/mode; smallest champion ID on exact ties.',
        overall_choice_ties='Smallest champion ID for exact main/alternative mean ties.',
        labels='Reference-player mode retains recorded histories, sets learned player/PC effects to zero; not population averages.',
        caveats=['Selected-comparison intervals; not simultaneous or selection-adjusted.','Main favored concerns only the selected alternative in that mode.',
                 'Draft availability is not verified.','Retrospective benchmark-assisted modeled preferences, not observed alternative outcomes or causal switching gains.',
                 ('Master labels lack dated rank verification; Global Master+ benchmarks; previously exposed outcomes.' if cohort=='Master'
                  else 'Broad ranks are collection-time labels, not verified patch-period ranks; Global exact-tier benchmarks; previously examined outcomes.')])
    decisions.attrs['provenance']=provenance
    decisions.to_parquet(out/'game_level_decisions.parquet',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    plot_classifications(summary,cohort,n,out/'classifications.png')
    return dict(cohort=cohort,decisions=decisions,summary=summary,output=out,
                examples=example_rows(decisions),provenance=provenance)


def plot_classifications(summary,cohort,n,path):
    import matplotlib.pyplot as plt
    modes=['Reference player','Personalized'];colors=['#2563A6','#C58B2A','#AAB2BD']
    fig,ax=plt.subplots(figsize=(9,5.5));bottom=np.zeros(2);small=[]
    for category,color in zip(CATEGORIES,colors):
        f=summary.loc[summary.section.eq('classification')&summary.measure.eq(category)].set_index('mode').reindex(modes)
        heights=f.percent.to_numpy();ax.bar(modes,heights,bottom=bottom,label=category,color=color,width=.55)
        for i,row in enumerate(f.itertuples()):
            if row.percent>=3:ax.text(i,bottom[i]+row.percent/2,f'{row.count} ({row.percent:.1f}%)',ha='center',va='center',color='white' if category==CATEGORIES[0] else '#18212B',fontsize=11)
            elif row.count>0:
                short='main' if category==CATEGORIES[1] else 'alternative' if category==CATEGORIES[0] else 'uncertain'
                small.append(f'{modes[i]} {short}: {row.count} ({row.percent:.1f}%)')
        bottom+=heights
    if small:fig.text(.5,.075,'Small segments: '+'; '.join(small),ha='center',fontsize=8)
    title_cohort='Master-labeled cohort' if cohort=='Master' else cohort
    ax.set(ylim=(0,100),ylabel='Main-pick games with eligible alternatives (%)',title=f'{title_cohort}: selected alternative versus recorded main\n{n} matched main-pick player-games per mode')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False,fontsize=9)
    qualification=('Undated Master labels; Global Master+ benchmarks; outcomes previously examined.' if cohort=='Master'
                   else 'Collection-time ranks, not verified patch-period ranks; Global exact-tier benchmarks; outcomes previously examined.')
    fig.text(.5,.045,qualification,ha='center',fontsize=8)
    fig.text(.5,.015,'Saved shared-model predictions. Selected-comparison intervals are not selection-adjusted; draft availability unverified.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(path,dpi=140);plt.close(fig)


def example_rows(decisions):
    """First chronological distinct game exhibiting each category in either mode.

    This allows a reference-only main-favored case to be shown honestly when no
    personalized comparison is main-favored; labels state the qualifying mode.
    """
    rows=[];used=set()
    for category in CATEGORIES:
        unused=pd.Series([k not in used for k in decisions[KEY].itertuples(index=False,name=None)],index=decisions.index)
        matching=decisions.loc[decisions.classification.eq(category)&unused].sort_values(['timestamp',*KEY,'mode'])
        if matching.empty:continue
        first=matching.iloc[0];used.add((first.puuid,first.match_id))
        frame=decisions.loc[decisions.puuid.eq(first.puuid)&decisions.match_id.eq(first.match_id)].copy()
        frame['illustrative_category']=category
        frame['category_present_in_modes']=' / '.join(frame.loc[frame.classification.eq(category),'mode'].tolist())
        rows.append(frame)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def display_examples(report):
    """Compact side-by-side examples with anonymous players; never show outcomes."""
    from IPython.display import display
    examples=report['examples']
    for category in CATEGORIES:
        frame=examples.loc[examples.illustrative_category.eq(category)]
        if frame.empty:print(f'No {category.lower()} game exists in either mode.');continue
        first=frame.iloc[0]
        print(f'Illustrative {category}: {first.anonymous_player}, {pd.to_datetime(first.timestamp,unit="ms",utc=True)}, '
              f'{first.lane}; category present in: {first.category_present_in_modes}.')
        values={}
        for row in frame.itertuples():
            values[row.mode]={
                'Main':f'{row.main_champion} ({row.main_champion_id})',
                'Selected alternative':f'{row.alternative_champion} ({row.alternative_champion_id})',
                'Eligible better-matchup alternatives':row.eligible_alternatives,
                'Recorded games: main / alternative':f'{row.main_recorded_games:g} / {row.alternative_recorded_games:g}',
                'Historical Beta(1,1) WR: main / alternative':f'{row.main_historical_wr_beta11:.1%} / {row.alternative_historical_wr_beta11:.1%}',
                'Group: main / alternative':f'{row.main_group} / {row.alternative_group}',
                'Raw matchup: main / alternative':f'{row.main_raw_matchup_probability:.2%} / {row.alternative_raw_matchup_probability:.2%}',
                'Predicted probability: main / alternative':f'{row.main_predicted_probability:.2%} / {row.alternative_probability:.2%}',
                'Alternative − main (probability pp)':f'{row.difference_pp:+.2f} [{row.low_pp:+.2f}, {row.high_pp:+.2f}]',
                'P(alternative higher)':f'{row.probability_alternative_higher:.3f}',
                'Classification':row.classification,
                'PC likelihood games: main / alternative':f'{row.main_pc_training_games} / {row.alternative_pc_training_games}',
                'Sparse training: main / alternative':f'{row.main_sparse_training_support} / {row.alternative_sparse_training_support}',
            }
        display(pd.DataFrame(values).reindex(columns=['Reference player','Personalized']))
    personal=report['decisions'].loc[report['decisions']['mode'].eq('Personalized')]
    for category in CATEGORIES:
        if not personal.classification.eq(category).any():print(f'No personalized game is classified as {category.lower()}; any displayed example in that category is explicitly reference-player only.')
