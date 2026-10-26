"""Train-only design, noncentered nonlinear variational logistic model, and scoring.

The ELBO uses exact Gaussian KLs and conditional-normal likelihood integration.
Hyperparameter expectations use fixed scrambled Sobol normal draws; the remaining
one-dimensional logistic-normal integral uses Gauss-Hermite quadrature.
"""
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, ndtri
from scipy.stats import qmc
from numpy.polynomial.hermite import hermgauss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

torch.set_num_threads(1)
DT=torch.float64

def keys_pc(f):
    return list(zip(f.puuid, f.champion_id.astype(int)))

class Design:
    def __init__(self, f, usage=False):
        self.cols=['log_total','matchup','usage','wr'] if usage else ['log_total','matchup','wr']
        self.scaler=StandardScaler().fit(f[self.cols])
        self.encoder=OneHotEncoder(drop='first',sparse_output=False,handle_unknown='error').fit(f[['lane']])
        self.names=['Intercept']+self.cols+list(self.encoder.get_feature_names_out(['lane']))
    def matrix(self,f):
        return np.column_stack([np.ones(len(f)),self.scaler.transform(f[self.cols]),self.encoder.transform(f[['lane']])])

class NonlinearVI:
    """Mean-field Gaussian q on noncentered latents (not exact joint inference)."""
    def __init__(self, train, rich=False):
        self.design=Design(train); self.rich=rich
        self.players={p:i for i,p in enumerate(sorted(train.puuid.unique()))}
        self.pcs={p:i for i,p in enumerate(sorted(set(keys_pc(train))))} if rich else {}
        self.p=len(self.design.names)
        self.i_exp=self.p; self.i_k=self.p+1; self.i_sigma=self.p+2
        self.z_start=self.p+3; self.z_stop=self.z_start+len(self.players)
        self.i_tau=self.z_stop if rich else None
        self.pc_start=self.z_stop+1 if rich else self.z_stop
        self.dim=self.pc_start+len(self.pcs)
        self.prior_m=np.zeros(self.dim); self.prior_s=np.ones(self.dim)
        self.prior_s[:self.p]=2.
        self.prior_m[self.i_k]=np.log(30); self.prior_s[self.i_k]=1.
        self.prior_m[self.i_sigma]=np.log(.3);self.prior_s[self.i_sigma]=.7
        if rich:self.prior_m[self.i_tau]=np.log(.15);self.prior_s[self.i_tau]=.5
        self.X=torch.tensor(self.design.matrix(train),dtype=DT)
        self.y=torch.tensor(train.win.to_numpy(),dtype=DT)
        self.n=torch.tensor(train.n.to_numpy(),dtype=DT)
        self.pi=torch.tensor([self.players[p] for p in train.puuid],dtype=torch.long)
        self.ci=torch.tensor([self.pcs[p] for p in keys_pc(train)],dtype=torch.long) if rich else None
        self.pm=torch.tensor(self.prior_m,dtype=DT);self.ps=torch.tensor(self.prior_s,dtype=DT)
        nodes,weights=hermgauss(12)
        self.nodes=torch.tensor(nodes,dtype=DT);self.weights=torch.tensor(weights/np.sqrt(np.pi),dtype=DT)
        self.set_integration(6,1619)
    def set_integration(self,power=6,seed=1619):
        z=ndtri(qmc.Sobol(3 if self.rich else 2,scramble=True,seed=seed).random_base2(power))
        # Antithetic points reduce the asymmetric tail error of lognormal moments.
        self.hyper=torch.tensor(np.concatenate([z,-z]),dtype=DT)
    def objective(self,theta):
        t=torch.tensor(theta,dtype=DT,requires_grad=True)
        mu=t[:self.dim]; sd=t[self.dim:].exp()
        k=(mu[self.i_k]+sd[self.i_k]*self.hyper[:,0]).exp()
        sig=(mu[self.i_sigma]+sd[self.i_sigma]*self.hyper[:,1]).exp()
        saturation=self.n[None,:]/(self.n[None,:]+k[:,None])
        mean=(self.X@mu[:self.p])[None,:]+mu[self.i_exp]*saturation+sig[:,None]*mu[self.z_start+self.pi][None,:]
        var=(self.X.square()@sd[:self.p].square())[None,:]+sd[self.i_exp].square()*saturation.square()+sig[:,None].square()*sd[self.z_start+self.pi][None,:].square()
        if self.rich:
            tau=(mu[self.i_tau]+sd[self.i_tau]*self.hyper[:,2]).exp()
            mean=mean+tau[:,None]*mu[self.pc_start+self.ci][None,:]
            var=var+tau[:,None].square()*sd[self.pc_start+self.ci][None,:].square()
        # Avoid a third large tensor: accumulate a smooth, accurate 1D quadrature.
        softplus=0.
        root=(2*var).sqrt()
        for node,weight in zip(self.nodes,self.weights):
            softplus=softplus+weight*torch.nn.functional.softplus(mean+root*node)
        loglik=(self.y[None,:]*mean-softplus).sum(dim=1).mean()
        kl=.5*(((sd/self.ps).square()+((mu-self.pm)/self.ps).square()-1+2*torch.log(self.ps/sd)).sum())
        loss=-loglik+kl
        loss.backward()
        return float(loss.detach()),t.grad.detach().numpy().copy()
    def fit(self,start=None,power=6,maxiter=800,seed=1619):
        self.set_integration(power,seed)
        if start is None:
            start=np.r_[self.prior_m,np.log(np.r_[np.full(self.pc_start,.3),np.full(len(self.pcs),.9)])]
            start[self.dim+self.z_start:self.dim+self.z_stop]=np.log(.9)
        tick=time.perf_counter()
        # Bounds only guard numerical overflow, far outside the fitted posterior.
        bounds=[(None,None)]*self.dim+[(-8,2)]*self.dim
        self.result=minimize(self.objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
             options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-4,'maxls':40,'maxcor':20})
        self.seconds=time.perf_counter()-tick
        self.mu=self.result.x[:self.dim];self.sd=np.exp(self.result.x[self.dim:])
        return self
    def diagnostics(self,label):
        return {'model':label,'seconds':self.seconds,'success':bool(self.result.success),
            'iterations':self.result.nit,'negative_ELBO':self.result.fun,
            'max_gradient':float(np.abs(self.result.jac).max()),'message':str(self.result.message)}
    def draws(self,size=1200,seed=2310):
        rng=np.random.default_rng(seed)
        return self.mu+rng.normal(size=(size,self.dim))*self.sd
    def predict_draws(self,f,draws,seed=50):
        """Reuse a supplied draw matrix; cache unseen group z's consistently by identity."""
        D=len(draws);out=np.empty((D,len(f)))
        def group_normal(key):
            value=int.from_bytes(hashlib.sha256(repr((seed,key)).encode()).digest()[:8],'little')
            return np.random.default_rng(value).normal(size=D)
        new_players={p:group_normal(('player',p)) for p in sorted(set(f.puuid)-set(self.players))}
        new_pcs={p:group_normal(('pc',p)) for p in sorted(set(keys_pc(f))-set(self.pcs))} if self.rich else {}
        X=self.design.matrix(f)
        for a in range(0,len(f),256):
            b=min(a+256,len(f));g=f.iloc[a:b]
            linear=draws[:,:self.p]@X[a:b].T
            k=np.exp(draws[:,self.i_k]);sig=np.exp(draws[:,self.i_sigma])
            linear+=draws[:,self.i_exp,None]*g.n.to_numpy()[None,:]/(g.n.to_numpy()[None,:]+k[:,None])
            z=np.column_stack([draws[:,self.z_start+self.players[p]] if p in self.players else new_players[p] for p in g.puuid])
            linear+=sig[:,None]*z
            if self.rich:
                zc=np.column_stack([draws[:,self.pc_start+self.pcs[p]] if p in self.pcs else new_pcs[p] for p in keys_pc(g)])
                linear+=np.exp(draws[:,self.i_tau,None])*zc
            out[:,a:b]=expit(linear)
        return out

def observed_frame(f):
    out=f.copy()
    out['n']=out.prior_champion_games
    out['wr']=out.prior_champion_wr_beta11
    out['usage']=out.prior_champion_usage_share
    out['matchup']=out.global_raw_matchup_winrate_probability
    out['log_total']=np.log1p(out.prior_total_games)
    return out

def candidate_frame(f):
    out=f.copy()
    out['champion_id']=out.candidate_champion_id
    out['n']=out.all_role_prior_games
    out['wr']=out.all_role_prior_wr_beta11
    out['usage']=out.all_role_prior_usage_share
    out['matchup']=out.global_raw_matchup_winrate_probability
    out['log_total']=np.log1p(out.prior_total_games)
    return out

def metrics(f,predictions):
    rows=[];y=f.win.to_numpy()
    for name,p in predictions.items():
        p=np.clip(p,1e-10,1-1e-10)
        per=pd.DataFrame({'player':f.puuid.to_numpy(),'ll':-(y*np.log(p)+(1-y)*np.log(1-p)),'bs':(y-p)**2}).groupby('player')[['ll','bs']].mean().mean()
        rows.append({'model':name,'n':len(f),'log_loss':log_loss(y,p),'Brier':brier_score_loss(y,p),
            'AUC':roc_auc_score(y,p),'equal_player_log_loss':per['ll'],'equal_player_Brier':per['bs']})
    return pd.DataFrame(rows).set_index('model')


def pyro_player_model(X, n, player_index, n_players, y=None):
    """Independent Pyro implementation of B's exact generative specification."""
    import pyro
    import pyro.distributions as dist
    beta=pyro.sample('beta',dist.Normal(X.new_zeros(X.shape[1]),X.new_full((X.shape[1],),2.)).to_event(1))
    beta_exp=pyro.sample('beta_exp',dist.Normal(X.new_tensor(0.),X.new_tensor(1.)))
    log_k=pyro.sample('log_k',dist.Normal(X.new_tensor(np.log(30.)),X.new_tensor(1.)))
    log_sigma=pyro.sample('log_sigma',dist.Normal(X.new_tensor(np.log(.3)),X.new_tensor(.7)))
    z=pyro.sample('z',dist.Normal(X.new_zeros(n_players),X.new_ones(n_players)).to_event(1))
    logits=X@beta+beta_exp*n/(n+log_k.exp())+log_sigma.exp()*z[player_index]
    pyro.sample('outcomes',dist.Bernoulli(logits=logits).to_event(1),obs=y)


def mcmc_parameter_draws(template, samples):
    """Map Pyro joint draws into the original B predictor's parameter layout."""
    flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in samples.items()}
    out=np.zeros((len(flat['log_k']),template.dim))
    out[:,:template.p]=flat['beta']
    out[:,template.i_exp]=flat['beta_exp']
    out[:,template.i_k]=flat['log_k']
    out[:,template.i_sigma]=flat['log_sigma']
    out[:,template.z_start:template.z_stop]=flat['z']
    return out


def preference_labels(f, prefix):
    if prefix=='A':
        return np.select([f.A_difference_pp.lt(0),f.A_difference_pp.gt(0)],['Main','Alternative'],default='Tied')
    return np.select([f[prefix+'_high_pp'].lt(0),f[prefix+'_low_pp'].gt(0)],['Main','Alternative'],default='Uncertain')


def weighted_pair_mean(f, value):
    """Pair-weighted; equal-target; equal-player (first average within target)."""
    values=np.asarray(value,dtype=float)
    data=f[['puuid','match_id']].copy();data['value']=values
    targets=data.groupby(['puuid','match_id']).value.mean()
    return {'pair_weighted':float(values.mean()),'equal_target':float(targets.mean()),
            'equal_player':float(targets.groupby(level='puuid').mean().mean())}


def preference_summary(f, label):
    rows=[]
    for prefix in ['A','B','C']:
        labels=preference_labels(f,prefix)
        for choice in (['Main','Alternative','Tied'] if prefix=='A' else ['Main','Alternative','Uncertain']):
            rows.append({'scope':label,'model':prefix,'preference':choice,
                         **weighted_pair_mean(f,labels==choice)})
    return pd.DataFrame(rows)


def point_agreement_summary(f, label):
    signs={p:np.sign(f[p+'_difference_pp'].to_numpy()) for p in ['A','B','C']}
    rows=[]
    for text,mask in [('All three agree',(signs['A']==signs['B'])&(signs['B']==signs['C'])),
        ('A vs B disagree',signs['A']!=signs['B']),('A vs C disagree',signs['A']!=signs['C']),
        ('B vs C disagree',signs['B']!=signs['C'])]:
        rows.append({'scope':label,'comparison':text,**weighted_pair_mean(f,mask)})
    return pd.DataFrame(rows)


def select_supported_examples(pairs):
    """Prefer BOTH >=20 same-role games, then >=10, then >=5; never extremes.

    Within each preference category, select the stable sorted median target/pair.
    Different pairs are selected for each category and support tiers are reported.
    """
    chosen=[];used=set()
    main=preference_labels(pairs,'B')=='Main'
    alt=preference_labels(pairs,'B')=='Alternative'
    uncertain=(preference_labels(pairs,'B')=='Uncertain')|(preference_labels(pairs,'B')!=preference_labels(pairs,'C'))
    for label,condition in [('Main favored',main),('Alternative favored',alt),('Uncertain / disagreement',uncertain)]:
        found=False
        for threshold in [20,10,5]:
            subset=pairs.loc[condition & pairs.main_same_role_prior_games.ge(threshold)
                              &pairs.alternative_same_role_prior_games.ge(threshold)&~pairs.index.isin(used)]
            # The main-favored example should match the genuine opposing-signal question.
            if label=='Main favored':subset=subset.loc[subset.signal_pattern.eq('Main more games AND higher WR')]
            if len(subset):
                ordered=subset.sort_values(['puuid','match_id','alternative_candidate_champion','main_candidate_champion'])
                # Equal-target median first, to avoid selecting through repertoire size.
                target_order=ordered[['puuid','match_id']].drop_duplicates()
                target=target_order.iloc[len(target_order)//2]
                target_pairs=ordered.loc[ordered.puuid.eq(target.puuid)&ordered.match_id.eq(target.match_id)]
                index=target_pairs.index[len(target_pairs)//2]
                chosen.append((index,label,threshold));used.add(index);found=True;break
        if not found:chosen.append((None,label,None))
    return chosen


def rank_mcmc_diagnostics(samples):
    """Rank/folded split R-hat and bulk/tail ESS, computed using Pyro stats."""
    from scipy.stats import rankdata
    from pyro.ops.stats import split_gelman_rubin, effective_sample_size
    rows=[]
    for name,value in samples.items():
        arr=np.asarray(value);flat=arr.reshape(arr.shape[0],arr.shape[1],-1)
        pool=flat.reshape(-1,flat.shape[-1]);N=len(pool)
        normal=ndtri((rankdata(pool,axis=0)-.375)/(N+.25)).reshape(flat.shape)
        folded=np.abs(pool-np.median(pool,axis=0))
        fold_normal=ndtri((rankdata(folded,axis=0)-.375)/(N+.25)).reshape(flat.shape)
        rh=np.maximum(split_gelman_rubin(torch.tensor(normal)).numpy(),split_gelman_rubin(torch.tensor(fold_normal)).numpy())
        bulk=effective_sample_size(torch.tensor(normal)).numpy()
        low=effective_sample_size(torch.tensor((flat<=np.quantile(pool,.05,axis=0)).astype(float))).numpy()
        high=effective_sample_size(torch.tensor((flat>=np.quantile(pool,.95,axis=0)).astype(float))).numpy()
        for i in range(flat.shape[-1]):
            rows.append({'parameter':name+(f'[{i}]' if flat.shape[-1]>1 else ''),
                'rank_split_Rhat':float(rh[i]),'bulk_ESS':float(bulk[i]),'tail_ESS':float(min(low[i],high[i]))})
    return pd.DataFrame(rows)


def run_player_mcmc(template, cache_dir, data_sha256, chains=4, warmup=600, samples=1200):
    """Reproducible independent NUTS check; cached chains avoid repeated large fits.

    Requires optional pyro-ppl==1.9.2. No held-out responses are accepted.
    ``template`` holds only observed training X/n/player codes/y.
    """
    from pathlib import Path
    import inspect,json
    import pyro
    from pyro.infer import MCMC,NUTS
    from pyro.infer.autoguide.initialization import init_to_value
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    manifest={'model_source_sha256':hashlib.sha256(inspect.getsource(pyro_player_model).encode()).hexdigest(),
        'data_sha256':data_sha256,'n_train':len(template.y),'warmup':warmup,
        'samples_per_chain':samples,'chains':chains,'framework':'Pyro '+pyro.__version__,
        'model':'B matched priors, exact Bernoulli likelihood, noncentered player intercept','seed':20261023}
    manifest_path=cache_dir/'manifest.json'
    if manifest_path.is_file():
        old= json.loads(manifest_path.read_text())
        assert all(old.get(key)==manifest[key] for key in ['model_source_sha256','data_sha256','n_train','warmup','samples_per_chain','chains']), 'Incompatible MCMC cache; use a new cache directory.'
    else:manifest_path.write_text(json.dumps(manifest,indent=2))
    for chain in range(chains):
        path=cache_dir/f'chain_{chain}.npz'
        if path.is_file() and (cache_dir/f'chain_{chain}.json').is_file():continue
        pyro.set_rng_seed(20261023+chain)
        values={'beta':torch.zeros(template.p,dtype=DT),'beta_exp':torch.tensor(.3,dtype=DT),
            'log_k':torch.tensor(np.log(10+10*chain),dtype=DT),
            'log_sigma':torch.tensor(np.log(.15+.05*chain),dtype=DT),
            'z':torch.zeros(len(template.players),dtype=DT)}
        kernel=NUTS(pyro_player_model,target_accept_prob=.9,max_tree_depth=9,
            full_mass=[('beta','beta_exp','log_k','log_sigma')],init_strategy=init_to_value(values=values),
            jit_compile=True,ignore_jit_warnings=True)
        sampler=MCMC(kernel,num_samples=samples,warmup_steps=warmup,num_chains=1,disable_progbar=True)
        tick=time.perf_counter()
        sampler.run(template.X,template.n,template.pi,len(template.players),template.y)
        np.savez_compressed(path,**{key:value.detach().numpy() for key,value in sampler.get_samples().items()})
        diagnostic=sampler.diagnostics()
        info={'seconds':time.perf_counter()-tick,'divergences':len(diagnostic['divergences']['chain 0']),
            'acceptance_rate':diagnostic['acceptance rate']['chain 0']}
        (cache_dir/f'chain_{chain}.json').write_text(json.dumps(info))
    return manifest
