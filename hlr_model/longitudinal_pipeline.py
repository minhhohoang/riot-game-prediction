"""V2 cohort, resumable collection, and expanding pregame features.

No module-level collection or filesystem writes. No LoLalytics integration.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lpm_model.otp_match_pipeline import ROLE_MAP, patch_from_match
from hlr_model.riot_api_helpers import (
    AuthenticationError, CollectionDisabled, MatchStore, RiotRequestError,
    atomic_json, read_json,
)


@dataclass(frozen=True)
class Config:
    player_count: int = 200
    historical_depth: int = 170
    random_seed: int = 1619
    target_patch: str = "16.19"
    queue: int = 420
    # NA season start: Jan 8, 2026, noon PST (Riot's local-server season start).
    season_start: int = int(datetime.fromisoformat("2026-01-08T12:00:00-08:00").timestamp())
    # V1's verified pre-deployment lower bound; target discovery is patch-sized.
    target_search_start: int = 1790164800
    # 16.20 is scheduled Oct 7. Include that entire NA calendar day to avoid
    # assuming an exact deployment hour, then validate every gameVersion.
    target_search_end: int = int(datetime.fromisoformat("2026-10-08T00:00:00-07:00").timestamp())
    page_size: int = 100
    allow_unverified_masters: bool = True
    otp_share_threshold: float = 0.50

    def __post_init__(self):
        if self.player_count < 1 or self.historical_depth < 1 or not 1 <= self.page_size <= 100:
            raise ValueError("Invalid player count, history depth, or API page size.")
        if self.queue != 420:
            raise ValueError("This study collects Ranked Solo/Duo queue 420 only.")
        if not self.season_start <= self.target_search_start < self.target_search_end:
            raise ValueError("Season and patch search boundaries must be ordered and bounded.")
        if not 0 <= self.otp_share_threshold <= 1:
            raise ValueError("Invalid OTP definition.")


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "lpm_model" / "otp_match_pipeline.py").is_file():
            return path
    raise FileNotFoundError("Open the notebook from within the repository.")


def candidate_players(project_root, config, evidence_path=None):
    """Never infer rank from match participants, win rate, or missing tier labels.

    Optional evidence CSV columns: puuid, tier, patch, observed_at_utc,
    evidence_source. Supply only documented study-period snapshots; this
    collector cannot authenticate external historical rank evidence itself.
    """
    frames = []
    for root in [project_root / "data" / "processed", project_root / "lpm_model" / "data" / "processed"]:
        for name in ["seed_players.csv", "player_match_index.csv", "player_match_index_raw.csv"]:
            path = root / name
            if not path.exists():
                continue
            frame = pd.read_csv(path, usecols=lambda c: c in {"puuid", "seed_tier"})
            if not {"puuid", "seed_tier"}.issubset(frame.columns):
                continue  # V1's fallback silently assigns Master; V2 must not.
            frame = frame.loc[frame.seed_tier.astype(str).str.lower().eq("master")].dropna(subset=["puuid"])
            frame["selection_source"] = str(path.relative_to(project_root))
            frames.append(frame)
    candidates = (pd.concat(frames, ignore_index=True).drop_duplicates("puuid")
                  if frames else pd.DataFrame(columns=["puuid", "seed_tier", "selection_source"]))
    candidates["historical_masters_verified"] = False
    candidates["rank_observed_at_utc"] = None
    candidates["rank_evidence_source"] = None
    if evidence_path is not None:
        evidence = pd.read_csv(evidence_path)
        required = {"puuid", "tier", "patch", "observed_at_utc", "evidence_source"}
        if not required.issubset(evidence.columns):
            raise ValueError(f"Historical rank evidence needs columns {sorted(required)}.")
        dated = pd.to_datetime(evidence.observed_at_utc, utc=True, errors="coerce")
        evidence = evidence.loc[
            evidence.tier.astype(str).str.lower().eq("master")
            & evidence.patch.astype(str).eq(config.target_patch) & dated.notna()
            & evidence.evidence_source.fillna("").astype(str).str.strip().ne("")
        ].dropna(subset=["puuid"]).drop_duplicates("puuid")
        # Evidence may introduce players absent from V1, without fetching today's league.
        for row in evidence.itertuples(index=False):
            mask = candidates.puuid.eq(row.puuid)
            if not mask.any():
                candidates.loc[len(candidates)] = [row.puuid, "master", "historical_rank_evidence", False, None, None]
                mask = candidates.puuid.eq(row.puuid)
            candidates.loc[mask, "historical_masters_verified"] = True
            candidates.loc[mask, "rank_observed_at_utc"] = str(row.observed_at_utc)
            candidates.loc[mask, "rank_evidence_source"] = str(row.evidence_source)
    candidates["rank_limitation"] = candidates.historical_masters_verified.map({
        True: "Study-period membership supported by user-supplied rank evidence",
        False: "V1 Master label has no dated snapshot; patch-16.19 membership unverified",
    })
    return candidates


def select_cohort(candidates, config):
    verified = sorted(candidates.loc[candidates.historical_masters_verified, "puuid"].tolist())
    unverified = sorted(candidates.loc[~candidates.historical_masters_verified, "puuid"].tolist())
    rng = random.Random(config.random_seed)
    selected = rng.sample(verified, min(config.player_count, len(verified)))
    needed = config.player_count - len(selected)
    if needed:
        if not config.allow_unverified_masters:
            raise ValueError("Insufficient historically verified Masters. Supply rank evidence or explicitly allow flagged V1 seeds.")
        if len(unverified) < needed:
            raise ValueError("Insufficient Master-labeled seeds for the requested cohort; no silent current-rank replacement.")
        selected.extend(rng.sample(unverified, needed))
    cohort = candidates.set_index("puuid").loc[selected].reset_index()
    cohort["selection_seed"] = config.random_seed
    cohort["target_patch"] = config.target_patch
    cohort["cohort_order"] = range(len(cohort))
    return cohort


class CollectionRun:
    """Single-writer V2 state, pinned search end and cohort. Resume same folder.

    Each match is cached before the next request; every ID page is checkpointed.
    Partial or failed players are retried, never marked successfully complete.
    A new cohort/configuration requires a new run folder.
    """

    def __init__(self, project_root, run_dir, config, cohort, client=None, enabled=False):
        self.project_root = Path(project_root)
        self.run_dir = Path(run_dir)
        self.config = config
        self.enabled = enabled
        self.client = client
        self.cohort = cohort.copy()
        self.store = MatchStore(self.run_dir / "raw" / "matches", [
            self.project_root / "data" / "otp_raw_matches",
            self.project_root / "lpm_model" / "data" / "otp_raw_matches",
        ], client)
        self.path = self.run_dir / "metadata" / "checkpoint.json"
        identity = {"config": asdict(config), "cohort": cohort.fillna("").to_dict("records")}
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state["identity"] != identity:
                raise ValueError("Run configuration/cohort changed. Use a new OUTPUT_RUN directory.")
        else:
            self.state = {"identity": identity, "search_end": min(int(time.time()), config.target_search_end),
                          "players": {}, "failures": [], "created_at_utc": datetime.now(timezone.utc).isoformat()}
        # Existing patch IDs are extra evidence, not proof of exhaustive discovery.
        self.legacy_targets = {}
        for root in [self.project_root / "data" / "processed", self.project_root / "lpm_model" / "data" / "processed"]:
            path = root / f"player_match_index_{config.target_patch.replace('.', '_')}.csv"
            if path.exists():
                frame = pd.read_csv(path, usecols=["puuid", "match_id"])
                for puuid, group in frame.groupby("puuid"):
                    self.legacy_targets.setdefault(puuid, set()).update(group.match_id.dropna())

    def require_enabled(self):
        if not self.enabled or self.client is None:
            raise CollectionDisabled("Set RUN_COLLECTION = True before collection stages.")
        self.client.require_enabled()

    def save(self):
        self.require_enabled()
        atomic_json(self.path, self.state)

    def failure(self, puuid, stage, match_id, error):
        self.state["failures"].append({"puuid": puuid, "stage": stage,
            "match_id": match_id, "error": str(error), "at_utc": datetime.now(timezone.utc).isoformat()})
        self.save()

    def pages(self, puuid, *, stage, end_time, start_time, remaining=None):
        """Bounded, cached pages; history requests only the remaining ID count."""
        offset = 0
        seen = set()
        while True:
            count = min(self.config.page_size, remaining()) if remaining is not None else self.config.page_size
            if count <= 0:
                return
            params = {"queue": self.config.queue, "count": count,
                      "start": offset, "endTime": end_time}
            if start_time is not None:
                params["startTime"] = start_time
            digest = hashlib.sha256(json.dumps([puuid, params], sort_keys=True).encode()).hexdigest()
            path = self.run_dir / "raw" / "match_ids" / stage / f"{digest}.json"
            if path.exists():
                page = read_json(path)["ids"]
            else:
                self.require_enabled()
                page = self.client.match_ids(puuid, **params)
                if not isinstance(page, list) or any(not isinstance(x, str) for x in page) or len(page) > count:
                    raise RiotRequestError("Invalid match-ID page.")
                atomic_json(path, {"puuid": puuid, "params": params, "ids": page})
            unique = [x for x in dict.fromkeys(page) if x not in seen]
            if page and not unique:
                raise RiotRequestError("Pagination made no progress; discovery incomplete.")
            seen.update(unique)
            yield unique
            if len(page) < count:
                break
            offset += count

    def collect_targets(self):
        """Fetch all candidate metadata to validate patch and obtain the anchor.

        Metadata is required before history discovery (notebook section 5).
        Search only the configured patch window, never the entire season.
        """
        self.require_enabled()
        # A legacy JSON may contain another focal player who was never indexed
        # in V1. Recover those appearances without duplicating any JSON files.
        if "cached_target_ids" not in self.state:
            cached_targets = {p: set() for p in self.cohort.puuid}
            for root in self.store.roots:
                for path in root.glob("NA1_*.json"):
                    try:
                        match = read_json(path)
                        match_id = match["metadata"]["matchId"]
                        if (not self.store.valid(match, match_id)
                                or match["info"].get("queueId") != self.config.queue
                                or patch_from_match(match) != self.config.target_patch):
                            continue
                        for participant in match["info"]["participants"]:
                            puuid = participant.get("puuid")
                            if puuid in cached_targets:
                                cached_targets[puuid].add(match_id)
                    except (OSError, ValueError, KeyError, TypeError, AttributeError):
                        continue  # Invalid cached files will not establish eligibility.
            self.state["cached_target_ids"] = {p: sorted(ids) for p, ids in cached_targets.items()}
            self.save()
        for number, puuid in enumerate(self.cohort.puuid, 1):
            player = self.state["players"].setdefault(puuid, {})
            if player.get("target_complete"):
                continue
            try:
                ids = set(self.legacy_targets.get(puuid, ()))
                ids.update(self.state["cached_target_ids"].get(puuid, ()))
                for page in self.pages(puuid, stage="target", end_time=self.state["search_end"],
                                       start_time=self.config.target_search_start):
                    ids.update(page)
                player["candidate_ids"] = sorted(ids)
                self.save()
                targets, exclusions = [], Counter()
                failed = False
                for match_id in sorted(ids):
                    try:
                        match = self.store.get(match_id, download=True)
                        record = player_record(match, puuid)
                        if match["info"].get("queueId") != self.config.queue:
                            exclusions["wrong_queue"] += 1
                        elif patch_from_match(match) != self.config.target_patch:
                            exclusions["wrong_patch"] += 1
                        elif record is None:
                            exclusions["invalid_player_record"] += 1
                            failed = True
                            self.failure(puuid, "target_participant", match_id,
                                         RiotRequestError("Incomplete patch-target record; earliest target anchor is uncertain."))
                        elif not (self.config.target_search_start * 1000 <= record["timestamp"]
                                  < self.state["search_end"] * 1000):
                            exclusions["outside_target_search_boundaries"] += 1
                        else:
                            targets.append(match_id)
                    except AuthenticationError:
                        raise
                    except (RiotRequestError, OSError, ValueError, KeyError, TypeError) as error:
                        failed = True
                        self.failure(puuid, "target_detail", match_id, error)
                player.update(target_ids=targets, target_complete=not failed, target_exclusions=dict(exclusions))
                self.save()
                print(f"Targets {number}/{len(self.cohort)}: {len(targets)} eligible; complete={not failed}")
            except AuthenticationError:
                raise
            except (RiotRequestError, OSError, ValueError) as error:
                self.failure(puuid, "target_discovery", None, error)

    def collect_histories(self):
        """Most recent 170 usable queue-420 games before earliest target kickoff.

        Failures cannot be filled with older substitute games. Page checkpoints
        and cached details permit retries from the same immutable cutoff.
        """
        self.require_enabled()
        for number, puuid in enumerate(self.cohort.puuid, 1):
            player = self.state["players"].get(puuid, {})
            if not player.get("target_complete") or not player.get("target_ids") or player.get("history_complete"):
                continue
            records = {}
            exhausted = False
            stop_reason = "not_started"
            exclusions = Counter()
            try:
                target_records = [player_record(self.store.get(m, download=True), puuid)
                                  for m in player["target_ids"]]
                if any(r is None for r in target_records):
                    raise RiotRequestError("Target anchor payload incomplete; resolve before collecting history.")
                anchor = min(r["timestamp"] for r in target_records)
                player["anchor_timestamp"] = anchor
                # endTime is whole seconds: include the anchor second, then
                # filter exact millisecond timestamps and completion locally.
                for page in self.pages(puuid, stage="history", end_time=anchor // 1000 + 1,
                                       start_time=self.config.season_start,
                                       remaining=lambda: self.config.historical_depth - len(records)):
                    for match_id in page:
                        match = self.store.get(match_id, download=True)
                        record = player_record(match, puuid)
                        if record is None:
                            raise RiotRequestError("Incomplete historical participant record; do not replace it with older history.")
                        if record["timestamp"] < self.config.season_start * 1000:
                            # Defensive guard if Riot returns an out-of-bound ID.
                            stop_reason = "season_boundary_reached"
                            break
                        if record["queue"] != self.config.queue:
                            exclusions["wrong_queue"] += 1
                        elif record["timestamp"] >= anchor or record["end_timestamp"] >= anchor:
                            exclusions["not_completed_before_anchor"] += 1
                        else:
                            records[match_id] = record
                        if len(records) >= self.config.historical_depth:
                            stop_reason = "historical_depth_reached"
                            break
                    partial = sorted(records.values(), key=lambda r: (r["timestamp"], r["match_id"]),
                                     reverse=True)[:self.config.historical_depth]
                    player.update(history_ids=[r["match_id"] for r in partial], history_complete=False,
                                  history_stop_reason="in_progress", history_exclusions=dict(exclusions))
                    self.save()
                    # ID pages are newest first; sort actual timestamps before selection.
                    if stop_reason in {"season_boundary_reached", "historical_depth_reached"}:
                        break
                else:
                    exhausted = True
                    stop_reason = "api_exhausted_within_season"
                selected = sorted(records.values(), key=lambda r: (r["timestamp"], r["match_id"]), reverse=True)[:self.config.historical_depth]
                player.update(history_ids=[r["match_id"] for r in selected], history_complete=True,
                    history_exhausted=exhausted, history_shortfall=self.config.historical_depth - len(selected),
                    history_stop_reason=stop_reason, history_exclusions=dict(exclusions))
                self.save()
                print(f"History {number}/{len(self.cohort)}: {len(selected)}/{self.config.historical_depth}")
            except AuthenticationError as error:
                partial = sorted(records.values(), key=lambda r: (r["timestamp"], r["match_id"]),
                                 reverse=True)[:self.config.historical_depth]
                player.update(history_ids=[r["match_id"] for r in partial], history_complete=False,
                              history_stop_reason="collection_error", history_exclusions=dict(exclusions))
                self.failure(puuid, "history_authentication", None, error)
                raise
            except (RiotRequestError, OSError, ValueError, KeyError, TypeError) as error:
                partial = sorted(records.values(), key=lambda r: (r["timestamp"], r["match_id"]),
                                 reverse=True)[:self.config.historical_depth]
                player["history_ids"] = [r["match_id"] for r in partial]
                player["history_complete"] = False
                player["history_stop_reason"] = "collection_error"
                player["history_exclusions"] = dict(exclusions)
                self.failure(puuid, "history_discovery_or_detail", None, error)

    def download_required(self):
        self.require_enabled()
        ids = {m for p in self.state["players"].values()
               for field in ["target_ids", "history_ids"] for m in p.get(field, [])}
        for match_id in sorted(ids):
            self.store.get(match_id, download=True)
        self.save()
        return len(ids)

    def build_records(self):
        historical, targets, coverage = [], [], []
        for puuid in self.cohort.puuid:
            player = self.state["players"].get(puuid, {})
            missing = Counter()
            for field, output in [("history_ids", historical), ("target_ids", targets)]:
                for match_id in player.get(field, []):
                    match = self.store.get(match_id)
                    record = player_record(match, puuid) if match else None
                    if record is None:
                        missing[field] += 1
                    else:
                        output.append(record)
            coverage.append({"puuid": puuid,
                "target_discovery_complete": player.get("target_complete", False),
                "history_discovery_complete": player.get("history_complete", False),
                "available_historical_matches": len(player.get("history_ids", [])) - missing["history_ids"],
                "history_shortfall": self.config.historical_depth - len(player.get("history_ids", [])) + missing["history_ids"],
                "history_exhausted": player.get("history_exhausted", False),
                "history_stop_reason": player.get("history_stop_reason", "not_started"),
                "history_exclusions_json": json.dumps(player.get("history_exclusions", {}), sort_keys=True),
                "genuinely_short_history": (player.get("history_complete", False)
                    and len(player.get("history_ids", [])) < self.config.historical_depth
                    and missing["history_ids"] == 0),
                "history_incomplete": (bool(player.get("target_ids"))
                    and (not player.get("history_complete", False) or missing["history_ids"] > 0)),
                "target_matches": len(player.get("target_ids", [])) - missing["target_ids"],
                "missing_required_payloads": sum(missing.values()),
                "candidate_matches": len(player.get("candidate_ids", [])),
                "anchor_timestamp": player.get("anchor_timestamp"),
                "target_exclusions_json": json.dumps(player.get("target_exclusions", {}), sort_keys=True)})
        columns = RECORD_COLUMNS
        history = pd.DataFrame(historical, columns=columns).drop_duplicates(["puuid", "match_id"])
        target = pd.DataFrame(targets, columns=columns).drop_duplicates(["puuid", "match_id"])
        return history, target, pd.DataFrame(coverage)


RECORD_COLUMNS = ["puuid", "match_id", "timestamp", "end_timestamp", "patch", "queue",
    "champion", "champion_id", "lane", "team_position", "opponent_champion",
    "opponent_champion_id", "opponent_puuid", "matchup_missing", "win", "team_id", "side",
    "summoner_spell_1", "summoner_spell_2", "perks_json", "platform"]


def player_record(match, puuid):
    """Keep every valid player outcome, including missing/ambiguous opponents.

    teamPosition is Riot's postgame role inference, useful for matchup indexing;
    it is not verified pregame lane intent. End time is a leakage audit field.
    Spell/rune choices are pregame choices. No KDA/gold/damage predictors.
    """
    info = match["info"]
    participants = info.get("participants", [])
    focal = [p for p in participants if p.get("puuid") == puuid]
    if len(focal) != 1:
        return None
    p = focal[0]
    if (not p.get("championName") or not isinstance(p.get("championId"), int)
            or not isinstance(p.get("win"), bool) or p.get("teamId") not in (100, 200)):
        return None
    start = info.get("gameStartTimestamp")
    end = info.get("gameEndTimestamp")
    if not isinstance(start, (int, float)) or start <= 0:
        return None
    if end is None:
        # Modern 16.x Match-V5 gameDuration is in seconds.
        duration = info.get("gameDuration")
        if not isinstance(duration, (int, float)) or duration <= 0:
            return None
        end = start + duration * 1000
    if not isinstance(end, (int, float)) or end < start:
        return None
    role = p.get("teamPosition")
    own = [x for x in participants if x.get("teamId") == p["teamId"] and x.get("teamPosition") == role]
    other = [x for x in participants if x.get("teamId") == 300 - p["teamId"] and x.get("teamPosition") == role]
    opponent = other[0] if role in ROLE_MAP and len(own) == len(other) == 1 else {}
    return {"puuid": puuid, "match_id": match["metadata"]["matchId"],
        "timestamp": int(start), "end_timestamp": int(end), "patch": patch_from_match(match),
        "queue": info.get("queueId"), "champion": p["championName"], "champion_id": p["championId"],
        "lane": ROLE_MAP.get(role), "team_position": role,
        "opponent_champion": opponent.get("championName"), "opponent_champion_id": opponent.get("championId"),
        "opponent_puuid": opponent.get("puuid"), "matchup_missing": not bool(opponent.get("championName")),
        "win": int(p["win"]), "team_id": p["teamId"], "side": "blue" if p["teamId"] == 100 else "red",
        "summoner_spell_1": p.get("summoner1Id"), "summoner_spell_2": p.get("summoner2Id"),
        # Riot's rune var1/var2/var3 values are postgame counters, not choices.
        "perks_json": json.dumps({
            "statPerks": p.get("perks", {}).get("statPerks", {}),
            "styles": [{"style": style.get("style"),
                        "perk_ids": [selection.get("perk") for selection in style.get("selections", [])]}
                       for style in p.get("perks", {}).get("styles", [])],
        }, sort_keys=True), "platform": "NA1"}


def expanding_features(history, targets, config):
    """Initial history plus completed earlier targets. Never roll the window.

    Strict end_timestamp < target timestamp prevents same-time or overlapping
    matches contributing outcomes. Zero experience has raw WR missing, share
    zero when the player has experience; Beta(1,1) smoothing is separate.
    """
    result = []
    histories = {p: g for p, g in history.groupby("puuid")}
    for puuid, group in targets.groupby("puuid", sort=True):
        group = group.sort_values(["timestamp", "match_id"])
        base = histories.get(puuid, pd.DataFrame(columns=RECORD_COLUMNS))
        all_prior = pd.concat([base, group], ignore_index=True).drop_duplicates(["puuid", "match_id"])
        available = sorted(all_prior.to_dict("records"), key=lambda r: (r["end_timestamp"], r["match_id"]))
        counts, wins, names = Counter(), Counter(), {}
        cursor, total, total_wins, last_end = 0, 0, 0, None
        for row in group.to_dict("records"):
            while cursor < len(available) and available[cursor]["end_timestamp"] < row["timestamp"]:
                prior = available[cursor]
                if prior["timestamp"] >= row["timestamp"]:
                    raise ValueError("Invalid temporal record.")
                cid = prior["champion_id"]
                counts[cid] += 1
                wins[cid] += prior["win"]
                names[cid] = prior["champion"]
                total += 1
                total_wins += prior["win"]
                last_end = prior["end_timestamp"]
                cursor += 1
            cid = row["champion_id"]
            n, w = counts[cid], wins[cid]
            main = min(counts, key=lambda c: (-counts[c], c)) if counts else None
            mn, mw = (counts[main], wins[main]) if main is not None else (0, 0)
            overall = total_wins / total if total else None
            raw = w / n if n else None
            out = dict(row)
            out.update(prior_total_games=total, prior_total_wins=total_wins,
                prior_overall_wr=overall, prior_champion_games=n, prior_champion_wins=w,
                prior_champion_wr=raw, prior_champion_usage_share=n / total if total else None,
                prior_main_champion=names.get(main), prior_main_champion_id=main,
                prior_main_champion_games=mn, prior_main_champion_wins=mw,
                prior_main_champion_wr=mw / mn if mn else None,
                prior_main_champion_usage_share=mn / total if total else None,
                is_main_champion=(cid == main),
                is_otp=(cid == main and total > 0 and mn / total >= config.otp_share_threshold),
                prior_other_champion_games=total - n, prior_other_champion_wins=total_wins - w,
                prior_champion_wr_beta11=(w + 1) / (n + 2),
                prior_overall_wr_beta11=(total_wins + 1) / (total + 2),
                prior_champion_wr_minus_overall=(raw - overall if raw is not None else None),
                prior_latest_end_timestamp=last_end,
                initial_history_games=len(base), initial_history_shortfall=config.historical_depth - len(base))
            result.append(out)
    features = ["prior_total_games", "prior_total_wins", "prior_overall_wr", "prior_champion_games",
        "prior_champion_wins", "prior_champion_wr", "prior_champion_usage_share", "prior_main_champion",
        "prior_main_champion_id", "prior_main_champion_games", "prior_main_champion_wins", "prior_main_champion_wr",
        "prior_main_champion_usage_share", "is_main_champion", "is_otp", "prior_other_champion_games",
        "prior_other_champion_wins", "prior_champion_wr_beta11", "prior_overall_wr_beta11",
        "prior_champion_wr_minus_overall", "prior_latest_end_timestamp", "initial_history_games", "initial_history_shortfall"]
    return pd.DataFrame(result, columns=RECORD_COLUMNS + features)


def quality_report(cohort, history, targets, final, coverage, config):
    """Called by the notebook later; these checks have not been run by Codex."""
    for frame in [history, targets, final]:
        if frame.duplicated(["puuid", "match_id"]).any():
            raise ValueError("Duplicate player/match rows.")
        if not frame.empty and not frame.queue.eq(config.queue).all():
            raise ValueError("Non-Solo/Duo record.")
    if not targets.patch.eq(config.target_patch).all() or len(final) != len(targets):
        raise ValueError("Target patch or row conservation failed.")
    if not final.empty:
        if not (final.prior_latest_end_timestamp.isna()
                | final.prior_latest_end_timestamp.lt(final.timestamp)).all():
            raise ValueError("Future/current outcome leakage.")
        if not (final.prior_champion_games.le(final.prior_total_games)
                & final.prior_total_wins.le(final.prior_total_games)
                & final.prior_champion_wins.le(final.prior_champion_games)).all():
            raise ValueError("Invalid historical counts.")
        for _, g in final.sort_values(["timestamp", "match_id"]).groupby("puuid"):
            if not g.prior_total_games.diff().dropna().ge(0).all():
                raise ValueError("Expanding count decreased.")
    anchors = targets.groupby("puuid").timestamp.min().to_dict()
    if set(zip(history.puuid, history.match_id)) & set(zip(targets.puuid, targets.match_id)):
        raise ValueError("Initial history overlaps target observations.")
    if any(r.end_timestamp >= anchors.get(r.puuid, float("inf")) for r in history.itertuples()):
        raise ValueError("Initial history reaches target period.")
    if not history.timestamp.ge(config.season_start * 1000).all():
        raise ValueError("Historical match predates the 2026 season boundary.")
    if not (targets.timestamp.ge(config.target_search_start * 1000)
            & targets.timestamp.lt(config.target_search_end * 1000)).all():
        raise ValueError("Target lies outside patch search boundaries.")
    if history.groupby("puuid").size().gt(config.historical_depth).any():
        raise ValueError("Historical collection exceeded its requested depth.")
    return {"selected_players": len(cohort),
        "historically_verified_masters": int(cohort.historical_masters_verified.sum()),
        "historical_rows": len(history), "target_rows": len(targets), "modeling_rows": len(final),
        "unique_required_match_ids": len(set(history.match_id) | set(targets.match_id)),
        "missing_lane_matchups": int(targets.matchup_missing.sum()),
        "players_with_history_shortfall": int(coverage.history_shortfall.gt(0).sum()),
        "players_with_genuinely_short_history": int(coverage.genuinely_short_history.sum()),
        "players_with_incomplete_history": int(coverage.history_incomplete.sum()),
        "players_with_incomplete_collection": int((~coverage.target_discovery_complete
            | (coverage.target_matches.gt(0) & ~coverage.history_discovery_complete)).sum()),
        "target_search_start": config.target_search_start,
        "season_start": config.season_start,
        "target_search_end": config.target_search_end,
        "discovery_scope": "bounded patch discovery plus cached patch IDs, validated against metadata and timestamps",
        "role_limitation": "teamPosition is inferred from play, not verified pregame lane intent",
        "otp_definition": f"Current champion is prior main, share >= {config.otp_share_threshold}; no minimum-game threshold"}


def save_outputs(run, cohort, history, targets, final, coverage, summary, *, enabled=False, csv=False):
    if not enabled:
        raise CollectionDisabled("Set SAVE_OUTPUTS = True to create an output snapshot.")
    # Every export is a new directory; no existing datasets are overwritten.
    output = run.run_dir / "processed" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [("player_cohort", cohort), ("historical_matches", history),
                        ("target_matches", targets), ("longitudinal_modeling", final), ("player_coverage", coverage)]:
        frame.to_parquet(output / f"{name}.parquet", index=False)
        if csv:
            frame.to_csv(output / f"{name}.csv", index=False)
    metadata = dict(summary, config=asdict(run.config), search_end=run.state["search_end"],
                    cache_metrics_this_session=dict(run.store.metrics),
                    api_metrics_this_session=dict(run.client.metrics) if run.client else {},
                    failures=run.state["failures"], run_created_at_utc=run.state["created_at_utc"])
    atomic_json(output / "collection_metadata.json", metadata)
    return output
