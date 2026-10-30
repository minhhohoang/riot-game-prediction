"""Fixed n>=50 external-matchup reporting, reusing expanded prediction caches.

Only cached source bodies and frozen predictions are read. No HTTP, fitting,
predictive evaluation, smoothing, or changes to the unfiltered reports.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hlr_model import expanded_main_pick_report as expanded
from hlr_model import lolalytics_matchup_scraper as scraper
from hlr_model.broad_rank_matchups import canonical_url
from hlr_model.main_pick_decision_report import KEY, file_sha, require

MIN_GAMES = 50
REPORT = 'expanded_published_candidates_min50_v1'


def raw_matchup_rows(job):
    """Audit named enemy-row n against cached counts, not header n or row totals."""
    task, evidence, expected = job
    receipt_path, body_path = Path(evidence['receipt_file']), Path(evidence['body_file'])
    require(file_sha(receipt_path) == evidence['receipt_sha256'], 'Receipt changed.')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['http_status'] == 200 and file_sha(body_path) == evidence['body_sha256']
            == receipt['body_sha256'], 'Missing or changed raw table.')
    require(canonical_url(receipt['requested_url']) == canonical_url(task['source_url'])
            == canonical_url(receipt.get('response_url', receipt['requested_url'])), 'Wrong table URL/redirect.')
    route, dataset, _ = scraper.qwik_dataset(body_path.read_text(errors='replace'))
    lane = scraper.ROLE_SLUGS[task['lane']]
    for key, value in dict(cid=task['champion_id'], champion=task['champion_slug'], patch='16.19',
                           tier=task['tier'], region='all', mode='ranked', page='build', status=200).items():
        require(route.get(key) == value, f'Raw table filter mismatch: {key}')
    require(route.get('lane') in {lane, 'all'} and dataset['response'].get('valid') is True, 'Invalid role/table.')
    header = dataset['header']
    require(all(header.get(k) == v for k,v in dict(cid=task['champion_id'],lane=lane,patch='16.19',queue=420).items()), 'Wrong raw header.')
    columns = dataset['enemy_h']
    require(columns == ['id','wr','d1','d2','pr','n'], 'Unreviewed raw matchup-column schema.')
    values = dataset['enemy'][lane]
    rows = {row[columns.index('id')]:dict(zip(columns,row)) for row in values}
    require(len(rows) == len(values), 'Duplicate raw opponent row.')
    result = []
    for item in expected:
        opponent = int(item['opponent_champion_id'])
        require(opponent in rows, 'A saved valid matchup is absent from its raw table.')
        row = rows[opponent]
        n = row['n']
        require(type(n) in (int,float) and np.isfinite(n) and n >= 0 and int(n) == n, 'Unverified raw external count.')
        require(np.isfinite(item['matchup_games']) and n == item['matchup_games'], 'PARSE ERROR: cached count differs from raw enemy-row n.')
        np.testing.assert_allclose(row['wr']/100,item['matchup_probability'],rtol=0,atol=1e-12)
        result.append(dict(table_id=task['table_id'], model_rank=task['model_rank'], lane=task['lane'],
            champion_id=int(task['champion_id']), opponent_champion_id=opponent,
            matchup_games=int(n), matchup_probability=row['wr']/100, header_champion_games=header.get('n'),
            enemy_table_rows=len(values), raw_count_field='enemy[verified_role][opponent].n',
            patch='16.19',region='global',tier=task['tier'],queue_id=420,
            source_url=task['source_url'],body_sha256=evidence['body_sha256'],receipt_sha256=evidence['receipt_sha256']))
    return result


def verified_support(root, cohort, *, workers=4):
    """Verify every cached eligible candidate and every usable main before filtering."""
    ctx = expanded.context(root,cohort)
    old = expanded.load_report(root,cohort)
    candidates = pd.read_parquet(old['out']/'expanded_candidate_features.parquet')
    pairs = pd.read_parquet(old['out']/'all_candidate_comparisons.parquet')
    tables = pd.read_parquet(old['out']/'matchup_table_coverage.parquet').set_index('table_id')
    expected = candidates[['model_rank','lane','candidate_champion_id','opponent_champion_id',
        'benchmark_matchup_games','matchup_probability']].rename(columns={
            'candidate_champion_id':'champion_id','benchmark_matchup_games':'matchup_games'})
    main = ctx['usable'].copy()
    count_column = 'global_matchup_games' if cohort == 'Master' else 'benchmark_matchup_games'
    # Counts without a verified raw row cannot pass; do not fill them from other counts.
    main['external_matchup_games'] = pd.to_numeric(main[count_column],errors='coerce')
    require(np.isfinite(main.external_matchup_games).all(), 'Usable saved main has a missing count; investigate raw provenance.')
    expected = pd.concat([expected,main[['model_rank','lane','champion_id','opponent_champion_id',
        'external_matchup_games','matchup_probability']].rename(columns={'external_matchup_games':'matchup_games'})],ignore_index=True)
    keys = ['model_rank','lane','champion_id','opponent_champion_id']
    require(expected.groupby(keys).matchup_games.nunique().le(1).all()
            and expected.groupby(keys).matchup_probability.nunique().le(1).all(), 'Inconsistent saved matchup values.')
    expected = expected.drop_duplicates(keys)
    index = expanded.receipt_index(ctx)
    jobs = []
    for (rank,lane,cid), frame in expected.groupby(['model_rank','lane','champion_id'],sort=True):
        tier = 'master_plus' if cohort == 'Master' else rank.lower()
        entry = scraper.CHAMPIONS[int(cid)]
        url = f"https://lolalytics.com/lol/{entry['slug']}/build/?lane={scraper.ROLE_SLUGS[lane]}&patch=16.19&tier={tier}"
        table_id = f'16.19:{tier}:global:ranked:{lane}:{int(cid)}'
        task = dict(table_id=table_id,model_rank=rank,lane=lane,champion_id=int(cid),
                    champion_slug=entry['slug'],tier=tier,source_url=url)
        if table_id in tables.index:
            record = tables.loc[table_id]
            require(record.status == 'verified', 'Unverified table claimed to supply a benchmark.')
            evidence = record[['receipt_file','receipt_sha256','body_file','body_sha256']].to_dict()
        else:
            # Expanded collection omitted main-only tables that could never be candidates.
            base,receipt_file = index[canonical_url(url)]
            receipt = json.loads(Path(receipt_file).read_text())
            evidence = dict(receipt_file=receipt_file,receipt_sha256=file_sha(receipt_file),
                body_file=str(Path(base)/receipt['body_file']),body_sha256=receipt['body_sha256'])
        jobs.append((task,evidence,frame.to_dict('records')))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(raw_matchup_rows,jobs),1):
            rows.extend(result)
            if i % 250 == 0 or i == len(jobs):print(cohort,'raw-count tables verified',i,'/',len(jobs),flush=True)
    audit = pd.DataFrame(rows)
    require(len(audit) == len(expected) and not audit.duplicated(keys).any(), 'Raw verification incomplete.')
    main = main.merge(audit[keys+['matchup_games']],on=keys,validate='many_to_one')
    np.testing.assert_array_equal(main.external_matchup_games,main.matchup_games)
    matched = pairs.merge(candidates[KEY+['candidate_champion_id','benchmark_matchup_games','matchup_probability']],
        on=KEY+['candidate_champion_id'],suffixes=('','_feature'),validate='many_to_one')
    require(len(matched)==2*len(candidates) and not pairs.duplicated(KEY+['mode','candidate_champion_id']).any(), 'Prediction cache incomplete.')
    np.testing.assert_array_equal(matched.benchmark_matchup_games,matched.benchmark_matchup_games_feature)
    np.testing.assert_allclose(matched.matchup_probability,matched.matchup_probability_feature,rtol=0,atol=0)
    return ctx,old,candidates,pairs,main,audit


def supported_report(verified):
    ctx,old,candidates,pairs,main,audit = verified
    out = ctx['source']/'reports'/REPORT
    out.mkdir(parents=True,exist_ok=True)
    main_supported = main.loc[main.external_matchup_games.ge(MIN_GAMES)].copy()
    # This is an external-data restriction, not a player/champion-experience rule.
    supported = candidates.loc[candidates.benchmark_matchup_games.ge(MIN_GAMES)].merge(
        main_supported[KEY+['external_matchup_games']],on=KEY,validate='many_to_one')
    selected_pairs = pairs.merge(supported[KEY+['candidate_champion_id','external_matchup_games']],
        on=KEY+['candidate_champion_id'],validate='many_to_one')
    require(len(selected_pairs)==2*len(supported), 'Missing supported cached predictions; do not invent predictions.')
    selected_pairs = selected_pairs.rename(columns={'external_matchup_games':'main_external_matchup_games'})
    require(selected_pairs.benchmark_matchup_games.ge(50).all() and selected_pairs.main_external_matchup_games.ge(50).all(), 'External support filter failed.')
    require(selected_pairs.matchup_probability.gt(selected_pairs.main_matchup_probability).all(), 'Raw rate is not strictly better.')
    zero = supported.loc[supported.n_recorded.eq(0)]
    require(zero.prior_champion_wins.eq(0).all() and zero.historical_wr_beta11.eq(.5).all(), 'Zero-experience feature changed.')
    supported.to_parquet(out/'expanded_candidate_features.parquet',index=False)
    selected_pairs.to_parquet(out/'all_candidate_comparisons.parquet',index=False)
    pd.read_parquet(old['out']/'published_matchup_lookup.parquet').loc[lambda f:f.matchup_games.ge(50)].to_parquet(out/'published_matchup_lookup.parquet',index=False)
    audit.to_parquet(out/'raw_count_verification.parquet',index=False)
    main_out = ctx['main'].copy()
    failed = main.loc[~main.external_matchup_games.ge(50),KEY].assign(unsupported_external_main=True)
    main_out = main_out.merge(failed,on=KEY,how='left',validate='one_to_one')
    main_out.loc[main_out.unsupported_external_main.eq(True),'data_exclusion_reasons']='main_external_matchup_games_below_50'
    subctx = dict(ctx,out=out,main=main_out,usable=main_supported)
    decisions,summary = expanded.decisions_and_summaries(subctx,supported,selected_pairs)
    summary['measure'] = summary.measure.replace({'Main-pick games with usable main features/benchmarks':
        'Usable main features and verified external main matchup n >= 50'})
    # The reused selector retains exact cached means/paired intervals and smallest-ID ties.
    summary.to_csv(out/'summary.csv',index=False)
    main_counts = main[KEY+['external_matchup_games']]
    main_counts.to_parquet(out/'main_external_support.parquet',index=False)
    provenance = dict(schema=REPORT,minimum_external_matchup_games=50,cohort=ctx['cohort'],
        helper_sha256=file_sha(__file__),expanded_helper_sha256=file_sha(expanded.__file__),
        parent_report_manifest_sha256=file_sha(old['out']/'report_manifest.json'),
        parent_report=str(old['out']),cached_candidate_mode_predictions_reused=len(selected_pairs),
        new_predictions=0,raw_tables_verified=audit.table_id.nunique(),raw_matchup_keys_verified=len(audit),
        count_definition='Named n field of the exact focal champion / opposing champion enemy row for the verified role, tier, Global region, patch 16.19 and queue 420; not header n or row count.',
        qualification='Both main and alternative require n >= 50. Raw rates/coefficients unchanged; no smoothing. Paired Bayesian credible intervals omit external benchmark uncertainty and selection adjustment; retrospective modeled comparisons, not proven switching benefits.',
        output_sha256={p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='report_manifest.json'})
    expanded.save_json(out/'report_manifest.json',provenance)
    print(ctx['cohort'],'MIN50:',len(ctx['main']),'actual main games;',len(main_supported),'supported mains;',
          decisions[KEY].drop_duplicates().shape[0],'eligible games;',len(supported),'candidates;',len(selected_pairs),'cached predictions reused; 0 new predictions',flush=True)
    return out


def load_report(root,cohort):
    folder,version,_ = expanded.COHORTS[cohort]
    out = Path(root)/'hlr_model/data/analyses'/folder/version/'reports'/REPORT
    manifest = json.loads((out/'report_manifest.json').read_text())
    require(manifest['helper_sha256']==file_sha(__file__) and manifest['minimum_external_matchup_games']==50,'Supported report provenance changed.')
    for name,sha in manifest['output_sha256'].items():require(file_sha(out/name)==sha,f'Supported output changed: {name}')
    return dict(out=out,summary=pd.read_csv(out/'summary.csv'),decisions=pd.read_parquet(out/'game_level_decisions.parquet'),manifest=manifest)


def presentation_tables(root):
    from IPython.display import HTML,display
    reports = {c:load_report(root,c) for c in ['Master','Bronze–Diamond']}
    labels = [('point_preference','Alternative has strictly higher mean prediction','Alternative had the higher average prediction.'),
        ('classification',expanded.CATEGORIES[0],'Alternative favored: paired 95% interval entirely above zero.'),
        ('classification',expanded.CATEGORIES[1],'Main favored: paired 95% interval entirely below zero.'),
        ('classification',expanded.CATEGORIES[2],'Comparison remained uncertain.')]
    for rule in expanded.RULES:
        table = {'Result':[label for _,_,label in labels]}
        for cohort,r in reports.items():
            s = r['summary'].loc[lambda f:f.selection_rule.eq(rule)&f['mode'].eq('Personalized')]
            classes = s.loc[s.section.eq('classification')]
            denominator = int(classes.denominator.iloc[0])
            require(classes['count'].sum()==denominator,'Classification counts do not reconcile.')
            values = []
            for section,measure,_ in labels:
                row = s.loc[s.section.eq(section)&s.measure.eq(measure)].iloc[0]
                values.append(f'{int(row["count"])} ({row.percent:.1f}%)')
            table[f'{cohort}: {denominator} games'] = values
        frame = pd.DataFrame(table)
        numeric_columns = list(frame.columns[1:])
        display(frame.style.hide(axis='index')
            .set_caption('Minimum 50 LoLalytics matchup games — Personalized — '+rule)
            .set_table_attributes('style="background:white; color:black; border-collapse:collapse; width:100%; max-width:1200px;"')
            .set_properties(**{'background-color':'white','color':'black','border':'none','border-bottom':'1px solid #dddddd',
                'padding':'14px 16px','font-size':'16px','white-space':'normal','line-height':'1.5','text-align':'left'})
            .set_properties(subset=numeric_columns,**{'font-weight':'bold','text-align':'center','min-width':'160px'})
            .set_properties(subset=['Result'],**{'max-width':'550px'})
            .set_table_styles([
                {'selector':'caption','props':[('background','white'),('color','black'),('text-align','left'),('font-size','21px'),('font-weight','bold'),('padding','12px 0 18px')]},
                {'selector':'th','props':[('background','white'),('color','black'),('font-weight','bold'),('border','none'),('border-bottom','2px solid black'),('padding','14px 16px'),('white-space','normal'),('text-align','left')]}]))
    display(HTML('<p style="background:white;color:black;padding:12px 16px;font:14px/1.5 Arial,sans-serif;">'
        'Both main and alternative benchmarks require at least 50 external LoLalytics matchup games. This does not require any player experience on the alternative. '
        'The higher-average row overlaps the other categories. The 95% intervals are paired Bayesian credible intervals and still omit external benchmark uncertainty and selection adjustment. '
        'These retrospective modeled comparisons do not establish switching benefits. Familiar-only and unfiltered expanded results above remain preserved.</p>'))
    return reports
