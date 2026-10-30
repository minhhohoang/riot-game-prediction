"""Inline figures/tables for completed NEW project-wide min50 posterior fits."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,PercentFormatter
from IPython.display import Markdown,display

from hlr_model import min50_project_analysis as analysis
from hlr_model import broad_rank_modeling as broad
from hlr_model import expanded_main_pick_report as expanded
from hlr_model.main_pick_decision_report import file_sha,require

COLORS={'observed':'#2563A6','benchmark':'#C58B2A','personalized':'#7656A4','red':'#C84444','uncertain':'#AAB2BD'}
EXPERIENCE=['0','1–4','5–9','10–19','20–49','50+']
USAGE=['0–20%','>20–40%','>40–60%','>60–80%','>80–100%']


def load(root):
    root=Path(root).resolve();completion=json.loads((analysis.project_dir(root)/'completion.json').read_text())
    require(completion['status']=='complete','Refitted min50 analysis is not complete; do not display old-fit results as new.')
    data={}
    for cohort in ['Bronze–Diamond','Master']:
        out,prepared=analysis.load_prepared(root,cohort)
        complete=json.loads((out/'analysis_complete.json').read_text())
        for name,sha in complete['output_sha256'].items():require(file_sha(out/name)==sha,f'Completed artifact changed: {name}')
        diagnostics=[]
        for model in ['shared','grouped']:
            status=json.loads((out/f'fits/{model}/fit_status.json').read_text())
            diag=pd.read_csv(out/f'fits/{model}/sampling_diagnostics.csv')
            require(status['status']=='converged' and status['diagnostic_gate_passed'],'Do not interpret failed fit.')
            require(diag.rank_split_Rhat.max()<1.01 and diag.bulk_ESS.min()>400 and diag.tail_ESS.min()>400,'Diagnostic gate failed.')
            diagnostics.append(dict(cohort=cohort,model=model,chains=4,warmup=600,draws_per_chain=1200,
                max_Rhat=diag.rank_split_Rhat.max(),min_bulk_ESS=diag.bulk_ESS.min(),min_tail_ESS=diag.tail_ESS.min(),
                divergences=sum(c['divergences'] for c in status['chains']),min_acceptance=min(c['acceptance_rate'] for c in status['chains'])))
        reports={}
        for model in ['shared','grouped']:
            path=out/'reports'/('expanded_min50_'+model+'_v1')
            manifest=json.loads((path/'report_manifest.json').read_text())
            require(manifest['new_analysis_version']==out.name and manifest['model']==model,'Wrong recommendation posterior version.')
            for name,sha in manifest['output_sha256'].items():require(file_sha(path/name)==sha,'Recommendation artifact changed.')
            for i,sha in enumerate(manifest['posterior_provenance']['chains_sha256']):
                require(file_sha(out/f'fits/{model}/chain_{i}.npz')==sha,'Recommendation draws differ from new fit.')
            reports[model]=dict(summary=pd.read_csv(path/'summary.csv'),decisions=pd.read_parquet(path/'game_level_decisions.parquet'),examples=pd.read_parquet(path/'examples.parquet'))
        played=prepared['played'];train=played.loc[played.split.eq('train')&played.usable_data].copy()
        train['usage_share']=train.n_recorded/train.prior_total_games.where(train.prior_total_games.gt(0))
        train['usage_bin']=pd.cut(train.usage_share,[0,.2,.4,.6,.8,1],labels=USAGE,include_lowest=True)
        train['main_choice']=np.where(train.champion_id.eq(train.prior_main_champion_id),'Recorded main','Another champion')
        data[cohort]=dict(out=out,played=played,train=train,reports=reports,diagnostics=diagnostics,
            parameters=pd.read_csv(out/'parameter_summary.csv'),amplitudes=pd.read_csv(out/'group_amplitude_summary.csv'),
            scaling=pd.read_csv(out/'training_standardization.csv'),curves=pd.read_parquet(out/'group_experience_curves.parquet'),
            metrics=pd.concat([pd.read_csv(out/f'{split}_all_metrics.csv') for split in ['validation','test']],ignore_index=True),
            calibration=pd.read_csv(out/'test_all_calibration.csv'),exclusions=pd.read_csv(out/'external_support_exclusions.csv'))
    return data


def white_table(frame,caption,*,formats=None,bold_cols=None):
    styled=(frame.style.hide(axis='index').set_caption(caption)
        .set_table_attributes('style="background:white;color:black;border-collapse:collapse;width:100%;max-width:1400px;"')
        .set_properties(**{'background-color':'white','color':'black','border':'none','border-bottom':'1px solid #dddddd',
            'padding':'10px 12px','font-size':'15px','white-space':'normal','line-height':'1.4'})
        .set_table_styles([
            {'selector':'caption','props':[('background','white'),('color','black'),('font-weight','bold'),('font-size','20px'),('text-align','left'),('padding','14px 0')]},
            {'selector':'th','props':[('background','white'),('color','black'),('font-weight','bold'),('border','none'),('border-bottom','2px solid black'),('padding','10px 12px'),('white-space','normal')]}]))
    if formats:styled=styled.format(formats,na_rep='')
    if bold_cols:styled=styled.set_properties(subset=bold_cols,**{'font-weight':'bold','text-align':'center'})
    display(styled)


def overview(data):
    display(Markdown('**These are new project-wide min50 fits**, not recommendation-only filtering of the original posteriors. '
        'Original histories and recorded experience features remain intact. Original chronological boundaries, match partitions and purges are unchanged. '
        'Pyro 1.9.2 / Torch 2.15.0; four sequential chains per model, 600 warmup + 1,200 retained draws, target acceptance .9, depth 9, one Torch thread. '
        'All four full diagnostic gates passed. Master rank is constant in its intercept; broad retains rank contrasts. '
        'Master labels are unverified legacy seeds with Global Master+ benchmarks; broad uses exact tiers and collection-time rank labels.'))
    for cohort,d in data.items():print(cohort,'NEW version:',d['out'])
    white_table(pd.concat([d['exclusions'] for d in data.values()],ignore_index=True),'External support exclusions — histories and purged rows preserved',
        formats={'removed_percent':'{:.2f}%'})
    diagnostics=pd.DataFrame([r for d in data.values() for r in d['diagnostics']])
    white_table(diagnostics,'Completed refit diagnostics — unchanged R-hat/ESS/divergence/acceptance gates',
        formats={'max_Rhat':'{:.5f}','min_bulk_ESS':'{:.0f}','min_tail_ESS':'{:.0f}','min_acceptance':'{:.3f}'})
    display(Markdown('Stable numerical sampling does not establish predictive usefulness. Tree-depth saturation and BFMI remain unavailable from the established wrapper. '
        'Validation was evaluated before test, with parameters frozen and shared primary/grouped secondary predeclared. These outcomes were previously examined: **retrospective reanalysis, not fresh confirmation**.'))


def rate_summary(frame,category,levels):
    f=frame.groupby(category,observed=True).agg(wins=('win','sum'),games=('win','size'),players=('puuid','nunique'),baseline=('matchup_probability','mean')).reindex(levels)
    f[['wins','games','players']]=f[['wins','games','players']].fillna(0).astype(int)
    n=f.games.where(f.games.gt(0));p=f.wins/n;z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    f['observed']=p;f['low']=center-half;f['high']=center+half
    return f


def limits(*values):
    v=np.concatenate([np.asarray(x,dtype=float).ravel() for x in values]);v=v[np.isfinite(v)]
    return (max(0,float(v.min())-5),min(100,float(v.max())+5)) if len(v) else (0,100)


def rate_axis(ax,ylim):
    ax.set_ylim(*ylim);ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6,steps=[1,2,2.5,5,10]));ax.grid(axis='y',alpha=.2)


def eda(data):
    for category,levels,title,xlabel in [
        ('experience_bin',EXPERIENCE,'Recorded champion experience','Recorded prior champion games'),
        ('usage_bin',USAGE,'OTP rate','Selected champion share of recorded prior games'),
        ('main_choice',['Another champion','Recorded main'],'Recorded main versus another champion','Observed champion choice'),
        ('mastery_group',list(broad.GROUPS),'External mastery groups','A–J / Unclassified')]:
        summaries={};excluded={}
        for cohort,d in data.items():
            frame=d['train']
            if category=='main_choice':frame=frame.loc[frame.prior_main_champion_id.notna()]
            if category=='usage_bin':frame=frame.loc[frame.usage_bin.notna()]
            excluded[cohort]=len(d['train'])-len(frame)
            summaries[cohort]=rate_summary(frame,category,levels)
        ylim=limits(*(100*f[['observed','baseline','low','high']] for f in summaries.values()))
        fig,axes=plt.subplots(1,2,figsize=(18,6),sharey=True,facecolor='white')
        for ax,(cohort,f) in zip(axes,summaries.items()):
            x=np.arange(len(f),dtype=float)
            if levels[-1]=='Unclassified':x[-1]+=.7
            ax.errorbar(x,100*f.observed,yerr=100*np.array([f.observed-f.low,f.high-f.observed]),fmt='o-',capsize=4,color=COLORS['observed'],label='Observed WR; approximate 95% Wilson')
            ax.plot(x,100*f.baseline,'s--',color=COLORS['benchmark'],label='Mean raw matchup benchmark')
            ticks=[f'{level}\n{r.games:,}g\n{r.players}p' for level,r in zip(levels,f.itertuples())]
            ax.set(xticks=x,xticklabels=ticks,title=cohort+' — filtered training',xlabel=xlabel,ylabel='Win rate (%) — zoomed y-axis')
            rate_axis(ax,ylim);ax.legend(fontsize=9)
        fig.suptitle(title+' — new project-wide min50 training rows',fontsize=18)
        fig.tight_layout();plt.show()
        display(Markdown('Matched observed/benchmark rows; g = likelihood player-games, p = distinct players. '
            'Wilson intervals ignore repeated-player/shared-match dependence; no benchmark interval is invented. '
            f'Undefined-category exclusions: {excluded}. Recorded histories were not filtered. These are descriptive associations.'))
    f=rate_summary(data['Bronze–Diamond']['train'],'model_rank',list(broad.RANKS))
    fig,ax=plt.subplots(figsize=(13,5),facecolor='white');x=np.arange(6)
    ax.errorbar(x,100*f.observed,yerr=100*np.array([f.observed-f.low,f.high-f.observed]),fmt='o-',capsize=4,label='Observed; approximate 95% Wilson')
    ax.plot(x,100*f.baseline,'s--',color=COLORS['benchmark'],label='Exact-tier raw benchmark')
    ax.set(xticks=x,xticklabels=[f'{rank.title()}\n{r.games} games / {r.players} players' for rank,r in zip(broad.RANKS,f.itertuples())],title='Broad min50 training: WR by collection-time rank',ylabel='Win rate (%) — zoomed y-axis')
    rate_axis(ax,limits(100*f[['observed','baseline','low','high']]));ax.legend();fig.tight_layout();plt.show()
    # Preserve the requested presentation style: side bars have labels, without error bars.
    side_rows=[]
    for cohort,d in data.items():
        frame=d['train'];require(frame.side.eq(frame.team_id.map({100:'blue',200:'red'})).all(),'Side mapping changed.')
        ranks=list(broad.RANKS) if cohort=='Bronze–Diamond' else ['MASTER']
        for rank in ranks:
            f=rate_summary(frame.loc[frame.model_rank.eq(rank)],'side',['blue','red']).reset_index()
            f['rank']=rank;side_rows.append(f)
    side=pd.concat(side_rows,ignore_index=True);rank_order=list(broad.RANKS)+['MASTER']
    fig,ax=plt.subplots(figsize=(15,6),facecolor='white');x=np.arange(7)
    for name,offset,color in [('blue',-.17,COLORS['observed']),('red',.17,COLORS['red'])]:
        f=side.loc[side.side.eq(name)].set_index('rank').reindex(rank_order)
        bars=ax.bar(x+offset,100*f.observed,.34,color=color,label=name.title())
        ax.bar_label(bars,labels=[f'{100*v:.1f}%' for v in f.observed],padding=4)
    ax.set(xticks=x,xticklabels=[r.title() for r in rank_order],ylabel='Observed win rate (%) — zoomed y-axis',title='New min50 training: blue/red rates by sampled cohort rank')
    rate_axis(ax,limits(100*side[['observed','low','high']]));ax.legend();fig.tight_layout();plt.show()
    display(Markdown('Side bars omit error bars; their shared zoom includes underlying Wilson bounds plus five percentage points. '
        'Rank labels describe sampled cohorts; Master uses unverified legacy seed labels. No causal side or rank effect is established.'))


def parameters(data,*,grouped=False):
    model='grouped' if grouped else 'shared'
    for cohort,d in data.items():
        frame=d['parameters'].loc[d['parameters'].model.eq(model),['parameter','units','median','low_95','high_95','probability_positive']]
        white_table(frame,cohort+' — NEW min50 '+model+' parameter estimates (95% credible intervals)',
            formats={c:'{:.3f}' for c in ['median','low_95','high_95','probability_positive']})
        if grouped:
            white_table(d['amplitudes'][['parameter','median','low_95','high_95','probability_positive']],
                cohort+' — reconstructed TOTAL group amplitudes, not contrast coordinates',
                formats={c:'{:.3f}' for c in ['median','low_95','high_95','probability_positive']})
    if not grouped:
        white_table(pd.concat([d['scaling'] for d in data.values()],ignore_index=True),'Recalculated filtered-training standardization',
            formats={'training_mean':'{:.6f}','training_SD':'{:.6f}'})
    display(Markdown('Coefficients are log-odds, not win-probability percentage points. The common +2 raw matchup pp row equals β × .02 / the NEW training SD. '
        'Experience is b × n/(n+k); n is recorded prior champion games, and k is half-saturation in recorded games. '
        'k and SDs are exponentiated from their log parameters, with blank sign probabilities; posterior sign probabilities are not p-values. '
        'Shared uses one experience amplitude; grouped totals add the shared amplitude and the reconstructed zero-sum group deviation. '
        'The original priors/equations/group definitions are unchanged. No A–J ordering is imposed.'))


def curves(data):
    ylim=limits(*(100*d['curves'][['median','low','high']] for d in data.values()))
    fig,axes=plt.subplots(1,2,figsize=(16,5.5),sharey=True,facecolor='white')
    contexts=[]
    for ax,(cohort,d) in zip(axes,data.items()):
        all_curves=d['curves'].loc[d['curves'].model.eq('shared')]
        caps=all_curves.groupby('group').n_recorded.max();floor=all_curves.groupby('group').n_recorded.min().max()
        f=all_curves.loc[all_curves.group.eq(caps.idxmin())&all_curves.n_recorded.ge(floor)].sort_values('n_recorded')
        ax.plot(f.n_recorded,100*f['median'],color=COLORS['observed'],label='Posterior median')
        ax.fill_between(f.n_recorded,100*f.low,100*f.high,color=COLORS['observed'],alpha=.2,label='95% credible interval')
        ax.set(title=cohort+' — refitted shared experience curve',xlabel='Recorded prior champion games',ylabel='Conditional win probability (%)')
        rate_axis(ax,ylim);ax.legend(fontsize=9)
        r=f.iloc[0];contexts.append(dict(cohort=cohort,matchup_percent=100*r.context_matchup,historical_WR_percent=100*r.context_historical_wr,
            total_prior_games=float(np.expm1(r.context_log_total)),role='Top',rank=r.context_rank,max_displayed_experience=f.n_recorded.max()))
    fig.tight_layout();plt.show();white_table(pd.DataFrame(contexts),'Explicit curve reference contexts — zero champion/player/PC adjustments')
    display(Markdown('Conditional reference-player illustrations, not population averages or total causal benefits of practice. '
        'Curves control historical champion WR and other covariates. Displayed experience stops at saved training-support/context caps; counts never exceed total prior games. '
        'Shared/grouped curve panels use one common zoom including all pointwise credible bands.'))
    for cohort,d in data.items():
        saved=d['curves'].loc[d['curves'].model.eq('grouped')]
        fig,axes=plt.subplots(3,4,figsize=(17,10),sharey=True,facecolor='white')
        for j,group in enumerate(broad.GROUPS):
            ax=axes.flat[j];f=saved.loc[saved.group.eq(group)].sort_values('n_recorded')
            if f.empty:ax.axis('off');continue
            ax.plot(f.n_recorded,100*f['median'],color=COLORS['observed'])
            ax.fill_between(f.n_recorded,100*f.low,100*f.high,color=COLORS['observed'],alpha=.2)
            support=d['train'].loc[d['train'].mastery_group.eq(group)]
            rug=support.loc[support.n_recorded.between(f.n_recorded.min(),f.n_recorded.max()),'n_recorded']
            ax.plot(rug,np.full(len(rug),.02),'|',transform=ax.get_xaxis_transform(),alpha=.15,color=COLORS['benchmark'])
            ax.set(title=f'{group}: {len(support)} games / {support.puuid.nunique()} players',xlabel='Recorded prior champion games',ylabel='Conditional win probability (%)')
            rate_axis(ax,ylim)
        axes.flat[-1].axis('off');fig.suptitle(cohort+' — NEW min50 grouped curves, 95% credible bands and filtered-training rugs',fontsize=17)
        fig.tight_layout();plt.show()


def predictive_results(data):
    metrics=pd.concat([d['metrics'] for d in data.values()],ignore_index=True)
    frame=metrics.loc[metrics.split.eq('test'),['cohort','model','mode','rows','players','matches','log_loss','brier','roc_auc','equal_player_log_loss','equal_player_brier']]
    white_table(frame,'Refitted min50 test performance — identical rows for every model and fixed benchmark',
        formats={c:'{:.5f}' for c in ['log_loss','brier','roc_auc','equal_player_log_loss','equal_player_brier']})
    order=[('training_prevalence','benchmark'),('raw_matchup','benchmark'),('shared','general'),('shared','personalized'),('grouped','general'),('grouped','personalized')]
    fig,axes=plt.subplots(1,2,figsize=(17,6),sharex=True,facecolor='white');deltas=[]
    for cohort in data:
        f=frame.loc[frame.cohort.eq(cohort)].set_index(['model','mode']);base=f.loc[('training_prevalence','benchmark'),'log_loss']
        deltas.extend([f.loc[key,'log_loss']-base for key in order])
    pad=max(.001,max(abs(v) for v in deltas)*.3);extent=max(abs(v) for v in deltas)+pad
    for ax,(cohort,d) in zip(axes,data.items()):
        f=frame.loc[frame.cohort.eq(cohort)].set_index(['model','mode']);base=f.loc[('training_prevalence','benchmark'),'log_loss']
        values=np.array([f.loc[key,'log_loss']-base for key in order]);ax.barh(range(6),values,color=[COLORS['red'] if v>0 else COLORS['observed'] for v in values])
        for j,v in enumerate(values):ax.annotate(f'{v:+.5f}',(v,j),xytext=(4 if v>=0 else -4,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=9)
        ax.axvline(0,color='gray',lw=1);ax.set(yticks=range(6),yticklabels=[a+' / '+b for a,b in order],xlim=(-extent,extent),title=cohort,xlabel='Test log loss minus training-prevalence benchmark (lower is better)');ax.invert_yaxis()
    fig.tight_layout();plt.show()
    fig,axes=plt.subplots(1,2,figsize=(17,6),sharex=True,sharey=True,facecolor='white')
    for ax,(cohort,d) in zip(axes,data.items()):
        for (model,mode),g in d['calibration'].groupby(['model','mode']):
            ax.plot(100*g.predicted,100*g.observed,'o',ls='--' if mode=='personalized' else '-',ms=4,label=model+' / '+mode)
        ax.plot([0,100],[0,100],':',color='gray');ax.set(xlim=(0,100),ylim=(0,100),title=cohort+' — filtered test calibration',xlabel='Mean predicted win probability (%)',ylabel='Observed WR (%)')
        ax.xaxis.set_major_formatter(PercentFormatter(100));ax.yaxis.set_major_formatter(PercentFormatter(100));ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();plt.show()
    display(Markdown('These held-out outcomes were previously examined. Test was evaluated only after validation, with fitted parameters and the predeclared shared/grouped roles frozen. '
        'No test tuning occurred. Calibration retains 0–100% axes; sparse extreme bins are unstable. Loss/AUC/equal-player values measure prediction, not switching benefits.'))


def recommendations(data):
    labels=[('point_preference','Alternative has strictly higher mean prediction','Alternative has the higher mean prediction.'),
        ('classification',expanded.CATEGORIES[0],'Alternative favored: paired 95% interval above zero.'),
        ('classification',expanded.CATEGORIES[1],'Main favored: paired 95% interval below zero.'),
        ('classification',expanded.CATEGORIES[2],'Uncertain: paired interval includes zero.')]
    for model in ['shared','grouped']:
        for rule in expanded.RULES:
            table={'Result':[label for _,_,label in labels]}
            for cohort,d in data.items():
                r=d['reports'][model];s=r['summary'].loc[lambda f:f.selection_rule.eq(rule)&f['mode'].eq('Personalized')]
                n=int(s.loc[s.section.eq('classification'),'denominator'].iloc[0])
                require(s.loc[s.section.eq('classification'),'count'].sum()==n,'Classification sum mismatch.')
                values=[]
                for section,measure,_ in labels:
                    row=s.loc[s.section.eq(section)&s.measure.eq(measure)].iloc[0]
                    values.append(f'{int(row["count"])} ({row.percent:.1f}%)')
                table[cohort+f': {n} games']=values
            cols=['Result',*['Master'+c[len('Master'):] for c in table if c.startswith('Master')],*[c for c in table if c.startswith('Bronze')]]
            frame=pd.DataFrame(table)[cols]
            white_table(frame,'REFITTED project-wide min50 — '+model+' — personalized — '+rule,bold_cols=list(frame.columns[1:]))
    display(Markdown('The higher-mean row overlaps the three interval categories. Both choices require ≥50 verified external matchup games. '
        'Zero recorded experience and no prior same-role play remain eligible. Reference-player results are retained separately in the saved reports and do not represent population averages. '
        'Intervals are paired Bayesian credible intervals from the NEW fits; they omit external benchmark uncertainty and selection adjustment.'))
    fig,axes=plt.subplots(2,2,figsize=(15,10),sharey=True,facecolor='white')
    for i,model in enumerate(['shared','grouped']):
        for j,(cohort,d) in enumerate(data.items()):
            ax=axes[i,j];s=d['reports'][model]['summary'];s=s.loc[s['mode'].eq('Personalized')&s.section.eq('classification')];bottom=np.zeros(2)
            for category,color in zip(expanded.CATEGORIES,[COLORS['observed'],COLORS['benchmark'],COLORS['uncertain']]):
                f=s.loc[s.measure.eq(category)].set_index('selection_rule').reindex(expanded.RULES)
                ax.bar(range(2),f.percent,bottom=bottom,color=color,width=.55,label=category)
                for k,row in enumerate(f.itertuples()):
                    if row.percent>=7:ax.text(k,bottom[k]+row.percent/2,f'{int(row.count)} ({row.percent:.1f}%)',ha='center',va='center',color='white' if category==expanded.CATEGORIES[0] else 'black',fontsize=11)
                bottom+=f.percent.to_numpy()
            ax.set(xticks=range(2),xticklabels=['Highest raw matchup','Highest model mean'],ylim=(0,100),title=cohort+' — '+model,ylabel='Eligible main-pick games (%)');ax.yaxis.set_major_formatter(PercentFormatter(100))
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.suptitle('NEW project-wide min50 fits — personalized selected comparisons',fontsize=19)
    fig.subplots_adjust(bottom=.10,top=.91,hspace=.28,wspace=.25);plt.show()
    for cohort,d in data.items():
        for model,r in d['reports'].items():
            s=r['summary'];coverage=s.loc[s.section.isin(['coverage','coverage_loss']),['measure','count','denominator','players','unique_matches']]
            white_table(coverage,cohort+' / '+model+' — coverage (no-comparison games are not confirmed main preferences)')
            details=s.loc[(s.section.eq('selected_experience')&s['mode'].eq('Personalized'))|s.section.eq('personalization_change'),
                ['selection_rule','mode','section','measure','count','denominator','percent','players']]
            white_table(details,cohort+' / '+model+' — recorded experience and matched personalization changes',formats={'percent':'{:.2f}%'})
            expanded.display_examples(cohort+' / NEW '+model,r['examples'])


def interpretation(data):
    display(Markdown('**Experience and matchup in the new shared fits:**'))
    for cohort,d in data.items():
        p=d['parameters'].loc[d['parameters'].model.eq('shared')].set_index('parameter')
        experience=p.loc['Experience amplitude'];k=p.loc['Half-saturation k'];matchup=p.loc['Raw matchup: +2 percentage points']
        display(Markdown(f'**{cohort}:** experience amplitude **{experience["median"]:.3f}** '
            f'(95% credible interval {experience.low_95:.3f} to {experience.high_95:.3f}) log-odds; '
            f'half-saturation **{k["median"]:.1f} recorded games** ({k.low_95:.1f} to {k.high_95:.1f}). '
            f'A +2 raw percentage-point matchup increase corresponds to **{matchup["median"]:.3f}** '
            f'({matchup.low_95:.3f} to {matchup.high_95:.3f}) log-odds. '
            'Both conditional associations are positive, while the wide k interval leaves the rate of saturation uncertain. '
            'These are not percentage-point changes in a player’s win probability or causal benefits of practice.'))
        counts=d['train'].groupby(['puuid','model_champion_id']).size()
        display(Markdown(f'The filtered likelihood contains **{len(counts):,} player–champion combinations**, '
            f'with median **{counts.median():.0f} fitted games** per combination; '
            f'**{100*counts.le(2).mean():.1f}%** have at most two. '
            'This limited likelihood support remains distinct from the much longer recorded histories used for experience. '
            'The fitted identity-scale intervals and pooled unseen-combination uncertainty limit how precisely the model can personalize a comparison.'))
    display(Markdown('**What grouping changes in the new analysis:**'))
    for cohort,d in data.items():
        test=d['metrics'].loc[d['metrics'].split.eq('test')].set_index(['model','mode'])
        pieces=[]
        for mode in ['general','personalized']:
            shared=test.loc[('shared',mode)];grouped=test.loc[('grouped',mode)]
            pieces.append(f'{mode}: grouped − shared log loss {grouped.log_loss-shared.log_loss:+.5f}, AUC {grouped.roc_auc-shared.roc_auc:+.4f}')
        display(Markdown('**'+cohort+':** '+'; '.join(pieces)+'. These point differences do not establish a rank interaction or a causal benefit.'))
        for model in ['shared','grouped']:
            s=d['reports'][model]['summary']
            for rule in expanded.RULES:
                f=s.loc[s.selection_rule.eq(rule)&s['mode'].eq('Personalized')]
                favored=f.loc[f.section.eq('classification')&f.measure.eq(expanded.CATEGORIES[0])].iloc[0]
                uncertain=f.loc[f.section.eq('classification')&f.measure.eq('Uncertain')].iloc[0]
                zero=f.loc[f.section.eq('selected_experience')&f.measure.eq(expanded.HISTORY_CLASSES[0])].iloc[0]
                change=s.loc[s.selection_rule.eq(rule)&s.section.eq('personalization_change')&s.measure.eq('Stay-versus-leave point preference changed')].iloc[0]
                display(Markdown(f'{model}, {rule}: alternative interval-favored in **{int(favored["count"])}/{int(favored.denominator)} ({favored.percent:.1f}%)**, '
                    f'uncertain **{uncertain.percent:.1f}%**; selected zero-history alternative **{zero.percent:.1f}%**; personalization changes staying/leaving **{change.percent:.1f}%**.'))
    display(Markdown('Recorded games mean bounded collected Season 16 history, not lifetime practice. History events with low-support external benchmarks still contribute to experience, WR and main definitions. '
        'Sparse fitted player–champion likelihood support is distinct from long recorded histories. Learned identity effects and pooled unseen-identity uncertainty are retained in paired predictions. '
        'Stable chains and small predictive differences do not establish reliable pick advice. These retrospective conditional comparisons have unknown draft availability and unplayed outcomes; '
        'credible intervals omit external benchmark uncertainty and selection adjustment, and cannot establish causal switching benefits.'))
