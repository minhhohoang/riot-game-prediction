# League of Legends One-Trick Trade-off Project

**Presentation:** [View the final presentation](https://docs.google.com/presentation/d/13uFz-BaLjYhYm6tWFDpwUoj5fvxoheXM0Qj-PshEiqU/edit?usp=sharing)

## Overview

When does champion familiarity outweigh a stronger matchup alternative in League of Legends? This completed project compares a player's recorded main with champions offering better published matchup rates, while quantifying uncertainty. The presentation provides the detailed analysis.

## Approach

Riot API Ranked Solo/Duo player-games from patch 16.19 are joined with LoLalytics Global matchup benchmarks. The Master-labeled cohort uses Master+ benchmarks; the Bronze–Diamond cohort uses exact-tier benchmarks.

Bayesian hierarchical logistic regression accounts for prior champion experience and win rate, player and champion differences, role, and rank where applicable. Shared and mastery-group experience specifications are compared. The final reanalysis requires at least 50 verified external games per observed matchup and per compared choice, preserving chronological training, validation, and test splits.

A main is the most-played champion in strictly prior, all-role recorded history. Eligible alternatives face the same opponent champion in the same role and have a strictly higher raw matchup rate; zero recorded champion experience remains eligible.

## Key findings

For the primary shared model's personalized predictions, the table compares each main with the eligible alternative having the highest raw matchup rate. Denominators are eligible test-period player-games where the player actually picked their main:

| Cohort | Alternative has higher mean predicted win probability | Uncertain comparison |
| --- | --- | --- |
| Master-labeled | 139/339 (41.0%) | 333/339 (98.2%) |
| Bronze–Diamond | 121/264 (45.8%) | 263/264 (99.6%) |

Uncertain means the paired 95% credible interval for the predicted difference includes zero. Player-games are focal-player records, not unique matches. These predictions do not establish that switching champions improves outcomes.

## Explore the project

- [Final reanalysis](hlr_model/min50_modeling.ipynb): filtered models, results, and actual-pick comparisons.
- [Presentation figure gallery](hlr_model/riot_v2_presentation.ipynb): figures and supporting tables.
- Original cohort analyses: [Master](hlr_model/master_cohort_modeling.ipynb) and [Bronze–Diamond](hlr_model/broad_rank_modeling.ipynb).

## Limitations and future work

- Observational champion selection, bounded histories, and sparse player–champion data limit causal and personalized conclusions.
- Rank labels lack patch-period verification; full-patch benchmarks and previously examined test outcomes make this retrospective.
- Validate prospectively across more patches, verify dated ranks, and incorporate draft availability and external-benchmark uncertainty.
