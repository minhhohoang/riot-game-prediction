# League of Legends One-Trick Trade-off Project

**Presentation:** [View the final presentation](https://docs.google.com/presentation/d/13uFz-BaLjYhYm6tWFDpwUoj5fvxoheXM0Qj-PshEiqU/edit?usp=sharing)

## Overview

When does champion familiarity outweigh a stronger matchup alternative in League of Legends? This completed project compares a player's recorded main with champions offering better published matchup rates. The presentation provides the detailed analysis.

## Approach

Riot API Ranked Solo/Duo player-games from patch 16.19 are joined with LoLalytics Global matchup benchmarks. The Master-labeled cohort uses Master+ benchmarks; the Bronze–Diamond cohort uses exact-tier benchmarks.

Bayesian hierarchical logistic regression accounts for prior champion experience and win rate, player and champion differences, role, and rank where applicable. The final reanalysis requires at least 50 verified external games per observed matchup and per compared choice, preserving chronological training, validation, and test splits.

A main is the most-played champion in strictly prior, all-role recorded history. Eligible alternatives face the same opponent champion in the same role and have a strictly higher raw matchup rate; zero recorded champion experience remains eligible.

## Key findings

Using the **grouped model in personalized mode**, “favors” means a higher posterior mean win probability. Rule B selects the eligible alternative with the highest model-predicted win probability.

| Actual champion chosen | Model favors | Master | Bronze–Diamond |
| --- | --- | --- | --- |
| Main | Best eligible alternative (Rule B) | 60.8% | 54.2% |
| Alternative | Recorded main | 36.0% | 58.8% |

Percentages are conditional on actual choice and cohort among eligible test player-games. Rule B applies only to the first row; the second compares the actual non-main pick with the recorded main. These are model-based preference rates, not predicted win probabilities or causal effects.

## Explore the project

- [Final reanalysis](hlr_model/min50_modeling.ipynb): filtered models, results, and actual-pick comparisons.
- [Presentation figure gallery](hlr_model/riot_v2_presentation.ipynb): figures and supporting tables.
- Original cohort analyses: [Master](hlr_model/master_cohort_modeling.ipynb) and [Bronze–Diamond](hlr_model/broad_rank_modeling.ipynb).

## Limitations and future work

- Observational champion selection, bounded histories, and sparse player–champion data limit causal and personalized conclusions.
- Rank labels lack patch-period verification; full-patch benchmarks and previously examined test outcomes make this retrospective.
- Validate prospectively across more patches, verify dated ranks, and incorporate draft availability and external-benchmark uncertainty.
