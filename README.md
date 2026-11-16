# League of Legends One-Trick Trade-off Project

**Presentation:** [View the final presentation](https://docs.google.com/presentation/d/13uFz-BaLjYhYm6tWFDpwUoj5fvxoheXM0Qj-PshEiqU/edit?usp=sharing)

## Overview

When does champion familiarity outweigh a stronger matchup alternative in League of Legends? This completed project compares a player's recorded main with champions offering better published matchup rates, while quantifying uncertainty. The presentation provides the detailed analysis.

## Approach

Riot API Ranked Solo/Duo player-games from patch 16.19 are joined with LoLalytics Global matchup benchmarks. The Master-labeled cohort uses Master+ benchmarks; the Bronze–Diamond cohort uses exact-tier benchmarks.

Bayesian hierarchical logistic regression accounts for prior champion experience and win rate, player and champion differences, role, and rank where applicable. Shared and mastery-group experience specifications are compared. The final reanalysis requires at least 50 verified external games per observed matchup and per compared choice, preserving chronological training, validation, and test splits.

A main is the most-played champion in strictly prior, all-role recorded history. Eligible alternatives face the same opponent champion in the same role and have a strictly higher raw matchup rate; zero recorded champion experience remains eligible.

## Key findings

Using the **grouped model in personalized mode**, “favors” means a higher posterior mean win probability. Rule B selects the eligible alternative with the highest model-predicted win probability.

| Actual champion chosen | Model favors | Master | Bronze–Diamond |
| --- | --- | --- | --- |
| Main | Best eligible alternative (Rule B) | 60.8% (206/339) | 54.2% (143/264) |
| Alternative | Recorded main | 36.0% (71/197) | 58.8% (124/211) |

Percentages are conditional on each row's actual choice and cohort; fractions count eligible test player-games. Both options require valid pregame features and at least 50 verified external matchup games, and the alternative must have a strictly better raw matchup. The second row compares the actual non-main pick directly with the recorded main, without Rule B selection. These are model-based preference rates, not champions' predicted win probabilities or evidence that switching improves outcomes.

## Explore the project

- [Final reanalysis](hlr_model/min50_modeling.ipynb): filtered models, results, and actual-pick comparisons.
- [Presentation figure gallery](hlr_model/riot_v2_presentation.ipynb): figures and supporting tables.
- Original cohort analyses: [Master](hlr_model/master_cohort_modeling.ipynb) and [Bronze–Diamond](hlr_model/broad_rank_modeling.ipynb).

## Limitations and future work

- Observational champion selection, bounded histories, and sparse player–champion data limit causal and personalized conclusions.
- Rank labels lack patch-period verification; full-patch benchmarks and previously examined test outcomes make this retrospective.
- Validate prospectively across more patches, verify dated ranks, and incorporate draft availability and external-benchmark uncertainty.
