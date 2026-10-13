"""Resumable collection and processing utilities for OTP match data.

This module deliberately handles Match-V5 match payloads only.  It does not
request or depend on timeline data.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


PLATFORM = "na1"
ROUTING = "americas"
RANKED_QUEUE = "RANKED_SOLO_5x5"
TARGET_QUEUE = 420
ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "adc",
    "UTILITY": "support",
}
INDEX_COLUMNS = ["puuid", "match_id", "seed_tier"]
DATASET_COLUMNS = [
    "match_id", "puuid", "seed_tier", "patch", "side", "lane", "champion",
    "opponent_champion", "opponent_puuid", "win", "game_duration",
    "game_start_timestamp", "champion_games", "player_games", "champion_share",
    "otp_measure",
]
HISTORY_COLUMNS = ["puuid", "match_id", "history_position", "history_exhausted"]
UNIQUE_MATCHUP_COLUMNS = ["lane", "champion", "opponent_champion"]
OPPONENT_INDEX_COLUMNS = ["opponent_puuid", "match_id"]
OPPONENT_SPECIALIZATION_COLUMNS = [
    "opponent_puuid", "opponent_champion", "opponent_champion_games",
    "opponent_player_games", "opponent_champion_share",
]
EXPECTED_MATCHUP_COLUMNS = [
    "lane", "champion", "opponent_champion", "matchup_rank",
    "expected_winrate", "matchup_games",
]


def patch_from_match(match_data: dict) -> str:
    """Return the major.minor patch from a Match-V5 payload."""
    parts = str(match_data["info"]["gameVersion"]).split(".")
    return ".".join(parts[:2])


def riot_get(url: str, headers: dict, params: dict | None = None) -> dict | list:
    """Request Riot's API and wait out rate limits without exposing credentials."""
    while True:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 429:
            wait_seconds = int(response.headers.get("Retry-After", 10)) + 1
            print(f"Rate limited; waiting {wait_seconds} seconds.")
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _normalise_seed_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["puuid", "seed_tier"])
    result = frame.loc[:, [column for column in ["puuid", "seed_tier"] if column in frame]].copy()
    if "puuid" not in result:
        return pd.DataFrame(columns=["puuid", "seed_tier"])
    if "seed_tier" not in result:
        result["seed_tier"] = "master"
    result = result.dropna(subset=["puuid"])
    result["seed_tier"] = result["seed_tier"].fillna("master").str.lower()
    return result.drop_duplicates("puuid", keep="first").reset_index(drop=True)


def _normalise_index(frame: pd.DataFrame, seed_players: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    required = {"puuid", "match_id"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=INDEX_COLUMNS)
    result = frame.copy()
    if "seed_tier" not in result:
        result = result.merge(seed_players, on="puuid", how="left")
    else:
        result = result.merge(
            seed_players.rename(columns={"seed_tier": "known_seed_tier"}),
            on="puuid", how="left",
        )
        result["seed_tier"] = result["seed_tier"].fillna(result["known_seed_tier"])
        result = result.drop(columns="known_seed_tier")
    result["seed_tier"] = result["seed_tier"].fillna("master").str.lower()
    return result.loc[:, INDEX_COLUMNS].dropna(subset=["puuid", "match_id"]).drop_duplicates(
        ["puuid", "match_id"], keep="first"
    ).reset_index(drop=True)


def load_seed_players(processed_dir: Path, legacy_index_path: Path) -> pd.DataFrame:
    """Load seeds, migrating legacy sampled players as Master seeds once."""
    seed_path = processed_dir / "seed_players.csv"
    seeds = _normalise_seed_table(_read_csv(seed_path, ["puuid", "seed_tier"]))
    if seeds.empty and legacy_index_path.exists():
        legacy = _read_csv(legacy_index_path, ["puuid", "match_id"])
        if "puuid" in legacy:
            seeds = pd.DataFrame({"puuid": legacy["puuid"].dropna().unique(), "seed_tier": "master"})
    return seeds


def save_seed_players(seed_players: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    result = _normalise_seed_table(seed_players)
    result.to_csv(processed_dir / "seed_players.csv", index=False)
    return result


def get_league_puuids(tier: str, headers: dict) -> list[str]:
    endpoint = f"{tier.lower()}leagues"
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/{endpoint}/by-queue/{RANKED_QUEUE}"
    payload = riot_get(url, headers)
    return [entry["puuid"] for entry in payload.get("entries", []) if entry.get("puuid")]


def extend_seed_players(
    seed_players: pd.DataFrame,
    desired_counts: dict[str, int],
    headers: dict,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Preserve existing seeds and add enough random players for each tier."""
    rng = rng or random.Random()
    result = _normalise_seed_table(seed_players)
    existing = set(result["puuid"])
    additions: list[dict[str, str]] = []
    for tier, desired in desired_counts.items():
        tier = tier.lower()
        have = int((result["seed_tier"] == tier).sum())
        needed = max(0, desired - have)
        if not needed:
            print(f"{tier.title()} seeds retained: {have}")
            continue
        pool = [puuid for puuid in get_league_puuids(tier, headers) if puuid not in existing]
        selected = rng.sample(pool, min(needed, len(pool)))
        additions.extend({"puuid": puuid, "seed_tier": tier} for puuid in selected)
        existing.update(selected)
        print(f"{tier.title()} seeds: retained {have}, added {len(selected)}, total {have + len(selected)}")
    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True)
    return _normalise_seed_table(result)


def load_player_match_index(
    processed_dir: Path,
    seed_players: pd.DataFrame,
    stem: str = "player_match_index",
) -> pd.DataFrame:
    """Load a persistent index, migrating the old raw index when appropriate."""
    current_path = processed_dir / f"{stem}.csv"
    legacy_path = processed_dir / "player_match_index_raw.csv"
    source = current_path if current_path.exists() else legacy_path if stem == "player_match_index" else current_path
    return _normalise_index(_read_csv(source, INDEX_COLUMNS), seed_players)


def save_player_match_index(
    index: pd.DataFrame,
    processed_dir: Path,
    stem: str = "player_match_index",
) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    result = index.loc[:, INDEX_COLUMNS].drop_duplicates(["puuid", "match_id"], keep="first")
    csv_path = processed_dir / f"{stem}.csv"
    parquet_path = processed_dir / f"{stem}.parquet"
    csv_temp = processed_dir / f".{stem}.csv.tmp"
    parquet_temp = processed_dir / f".{stem}.parquet.tmp"
    result.to_csv(csv_temp, index=False)
    csv_temp.replace(csv_path)
    result.to_parquet(parquet_temp, index=False)
    parquet_temp.replace(parquet_path)
    return result


def append_match_histories(
    index: pd.DataFrame,
    seed_players: pd.DataFrame,
    headers: dict,
    matches_per_player: int,
    start_time: int | None = None,
    end_time: int | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_stem: str = "player_match_index",
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """Fetch filtered history-ID pages for each seed; checkpoint when requested.

    A Match-V5 history page contains at most 100 IDs. Continue until a page
    is short so active players are not truncated to their newest 100 games.
    IDs are not downloaded match payloads; a request failure can leave gaps.
    """
    if not 1 <= matches_per_player <= 100:
        raise ValueError("matches_per_player must be between 1 and 100 (one API page)")
    if checkpoint_dir is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    result = index.loc[:, INDEX_COLUMNS].copy()
    new_rows: list[dict[str, str]] = []
    for number, seed in enumerate(seed_players.itertuples(index=False), start=1):
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{seed.puuid}/ids"
        offset = 0
        player_ids = 0
        while True:
            params = {
                "queue": TARGET_QUEUE,
                "start": offset,
                "count": matches_per_player,
            }
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time
            try:
                match_ids = riot_get(url, headers, params)
            except requests.RequestException as error:
                print(f"History page {offset} skipped for seed {number}/{len(seed_players)}: {error}")
                break
            new_rows.extend(
                {"puuid": seed.puuid, "match_id": match_id, "seed_tier": seed.seed_tier}
                for match_id in match_ids
            )
            player_ids += len(match_ids)
            if len(match_ids) < matches_per_player:
                break
            offset += matches_per_player
            time.sleep(0.1)
        print(f"Seed {number}/{len(seed_players)}: {player_ids} history IDs")
        time.sleep(0.1)
        if checkpoint_dir is not None and number % checkpoint_every == 0:
            result = pd.concat(
                [result, pd.DataFrame(new_rows, columns=INDEX_COLUMNS)], ignore_index=True
            ).drop_duplicates(["puuid", "match_id"], keep="first")
            result = save_player_match_index(result, checkpoint_dir, checkpoint_stem)
            new_rows.clear()
    result = pd.concat(
        [result, pd.DataFrame(new_rows, columns=INDEX_COLUMNS)], ignore_index=True
    ).drop_duplicates(["puuid", "match_id"], keep="first")
    return result.reset_index(drop=True)


def cached_patch_history_coverage(
    patch_index: pd.DataFrame,
    raw_match_dir: Path,
    target_patch: str,
    history_page_size: int = 100,
) -> tuple[pd.DataFrame, set[str]]:
    """Compare indexed IDs since patch start with usable cached patch games.

    Returns player coverage plus the distinct cached patch/queue match IDs.
    The index is an upper-bound denominator if the configured start time
    precedes the patch; a one-page history request may also miss older games.
    """
    relationships = patch_index.loc[:, ["puuid", "match_id"]].dropna().drop_duplicates()
    ids_by_match = relationships.groupby("match_id")["puuid"].agg(set).to_dict()
    cached_patch_ids: set[str] = set()
    covered_pairs: list[tuple[str, str]] = []
    for match_id, expected_puuids in ids_by_match.items():
        path = raw_match_dir / f"{match_id}.json"
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as file:
                match = json.load(file)
            if patch_from_match(match) != target_patch or match["info"].get("queueId") != TARGET_QUEUE:
                continue
            participants = match["info"].get("participants", [])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        cached_patch_ids.add(match_id)
        covered_pairs.extend(
            (participant["puuid"], match_id)
            for participant in participants
            if participant.get("puuid") in expected_puuids and participant.get("championName")
        )

    indexed = relationships.groupby("puuid").size().rename("indexed_since_start")
    covered = pd.DataFrame(covered_pairs, columns=["puuid", "match_id"])
    if covered.empty:
        coverage_counts = pd.Series(dtype="int64", name="cached_usable_patch_games")
    else:
        coverage_counts = covered.drop_duplicates().groupby("puuid").size().rename("cached_usable_patch_games")
    summary = indexed.to_frame().join(coverage_counts).fillna({"cached_usable_patch_games": 0})
    summary["cached_usable_patch_games"] = summary["cached_usable_patch_games"].astype(int)
    summary["indexed_cache_coverage"] = summary["cached_usable_patch_games"] / summary["indexed_since_start"]
    summary["at_or_above_page_size"] = summary["indexed_since_start"] >= history_page_size
    return summary.reset_index(), cached_patch_ids


def download_missing_matches(
    index: pd.DataFrame,
    raw_match_dir: Path,
    headers: dict,
    max_new_downloads: int,
    target_patch: str | None = None,
    selection_seed: int | None = None,
    progress_every: int = 1,
) -> tuple[int, int, int]:
    """Save only missing Match-V5 payloads in a round-robin tier order.

    Existing files never count toward the cap. If target_patch is provided,
    only newly saved Solo/Duo matches on that patch count toward it. Other
    responses are still cached so they will not be repeatedly downloaded.
    A match associated with multiple seed tiers is fetched only once.
    When a selection seed is supplied, shuffle candidate IDs within each
    tier before round-robin scheduling. This includes newly discovered games
    without systematically favoring IDs from an older saved index.
    """
    raw_match_dir.mkdir(parents=True, exist_ok=True)
    match_ids = index["match_id"].dropna().drop_duplicates().tolist()
    cached_ids = {path.stem for path in raw_match_dir.glob("*.json")}
    missing_ids = [match_id for match_id in match_ids if match_id not in cached_ids]
    print(f"Cached matches found: {len(cached_ids):,}")
    print(f"Match IDs requested: {len(match_ids):,}")
    print(f"New matches remaining: {len(missing_ids):,}")

    missing_set = set(missing_ids)
    tiered_index = index.loc[:, ["match_id", "seed_tier"]].dropna(subset=["match_id"]).copy()
    tiered_index["seed_tier"] = tiered_index["seed_tier"].fillna("unknown").str.lower()
    tier_order = ["master", "grandmaster", "challenger"]
    other_tiers = [tier for tier in tiered_index["seed_tier"].unique() if tier not in tier_order]
    tier_order.extend(other_tiers)
    rng = random.Random(selection_seed) if selection_seed is not None else None
    tier_queues = {}
    for tier in tier_order:
        tier_ids = [
            match_id for match_id in tiered_index.loc[
                tiered_index["seed_tier"] == tier, "match_id"
            ].drop_duplicates()
            if match_id in missing_set
        ]
        if rng is not None:
            rng.shuffle(tier_ids)
        tier_queues[tier] = deque(tier_ids)

    # One pass selects at most one new match from each tier. Shared matches are
    # skipped in later queues, so every Match-V5 payload still has one download.
    scheduled_ids: list[str] = []
    scheduled_set: set[str] = set()
    while True:
        scheduled_this_round = False
        for tier in tier_order:
            queue = tier_queues[tier]
            while queue and queue[0] in scheduled_set:
                queue.popleft()
            if queue:
                match_id = queue.popleft()
                scheduled_ids.append(match_id)
                scheduled_set.add(match_id)
                scheduled_this_round = True
        if not scheduled_this_round:
            break

    # `missing_ids` is derived from this same index, but retain this fallback so
    # a malformed tier value cannot ever make an otherwise valid match disappear.
    scheduled_ids.extend(match_id for match_id in missing_ids if match_id not in scheduled_set)

    downloads = 0
    for match_id in scheduled_ids:
        if downloads >= max_new_downloads:
            break
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        try:
            payload = riot_get(url, headers)
        except requests.RequestException as error:
            print(f"Match skipped ({match_id}): {error}")
            continue
        final_path = raw_match_dir / f"{match_id}.json"
        temporary_path = raw_match_dir / f".{match_id}.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)
        temporary_path.replace(final_path)
        if target_patch is None or (
            payload.get("info", {}).get("queueId") == TARGET_QUEUE
            and patch_from_match(payload) == target_patch
        ):
            downloads += 1
            if downloads % progress_every == 0 or downloads == max_new_downloads:
                print(f"New qualifying downloads this run: {downloads}/{max_new_downloads} ({match_id})")
        else:
            print(f"Cached {match_id}, but excluded from the {target_patch} Solo/Duo target")
        time.sleep(0.1)
    return len(cached_ids), len(missing_ids), downloads


def build_otp_dataset(index: pd.DataFrame, raw_match_dir: Path, target_patch: str) -> tuple[pd.DataFrame, int]:
    """Build one role-matched observation for each indexed seed-player appearance.

    Build candidate patch observations. Attach pre-game champion usage separately,
    after retrieving the ordered prior-game IDs and their match payloads.
    """
    index = index.loc[:, INDEX_COLUMNS].drop_duplicates(["puuid", "match_id"], keep="first")
    seeds_by_match = index.groupby("match_id")[["puuid", "seed_tier"]].apply(
        lambda group: group.to_dict("records")
    ).to_dict()
    observations: list[dict] = []
    unreadable_files = 0
    for match_id, seed_rows in seeds_by_match.items():
        path = raw_match_dir / f"{match_id}.json"
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as file:
                match = json.load(file)
            info = match["info"]
            patch = patch_from_match(match)
        except (OSError, ValueError, KeyError, TypeError):
            unreadable_files += 1
            continue
        if patch != target_patch or info.get("queueId") != TARGET_QUEUE:
            continue
        participants = info.get("participants", [])
        by_puuid = {participant.get("puuid"): participant for participant in participants}
        for seed in seed_rows:
            player = by_puuid.get(seed["puuid"])
            champion = player.get("championName") if player else None
            role = ROLE_MAP.get(player.get("teamPosition", "")) if player else None
            if not player or not champion or not role or player.get("teamId") not in (100, 200):
                continue
            opponents = [
                participant for participant in participants
                if participant.get("teamId") != player["teamId"]
                and ROLE_MAP.get(participant.get("teamPosition", "")) == role
            ]
            if len(opponents) != 1:
                continue
            opponent = opponents[0]
            observations.append({
                "match_id": match_id,
                "puuid": seed["puuid"],
                "seed_tier": seed["seed_tier"],
                "patch": patch,
                "side": "blue" if player["teamId"] == 100 else "red",
                "lane": role,
                "champion": champion,
                "opponent_champion": opponent.get("championName"),
                "opponent_puuid": opponent.get("puuid"),
                "win": bool(player.get("win")),
                "game_duration": info.get("gameDuration"),
                "game_start_timestamp": info.get("gameStartTimestamp"),
            })
    dataset = pd.DataFrame(observations)
    if dataset.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS), unreadable_files
    for column in ("champion_games", "player_games", "champion_share", "otp_measure"):
        dataset[column] = pd.NA
    return dataset.loc[:, DATASET_COLUMNS].sort_values(["game_start_timestamp", "match_id", "puuid"]).reset_index(drop=True), unreadable_files


def load_pregame_history_index(processed_dir: Path) -> pd.DataFrame:
    """Ordered, completed history requests; positions run newest to oldest."""
    path = processed_dir / "pregame_history_index_16_19.csv"
    frame = _read_csv(path, HISTORY_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    missing = set(HISTORY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Pre-game history index is missing columns: {sorted(missing)}")
    frame = frame.loc[:, HISTORY_COLUMNS].copy()
    frame["history_position"] = pd.to_numeric(frame["history_position"], errors="raise").astype(int)
    frame["history_exhausted"] = frame["history_exhausted"].astype(str).str.lower().eq("true")
    if frame.duplicated(["puuid", "match_id"]).any() or frame.duplicated(["puuid", "history_position"]).any():
        raise ValueError("The pre-game history index contains duplicate player IDs or positions")
    return frame


def save_pregame_history_index(frame: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "pregame_history_index_16_19.csv"
    temporary = processed_dir / ".pregame_history_index_16_19.csv.tmp"
    frame.loc[:, HISTORY_COLUMNS].to_csv(temporary, index=False)
    temporary.replace(path)
    return frame


def append_pregame_histories(
    index: pd.DataFrame,
    observations: pd.DataFrame,
    headers: dict,
    processed_dir: Path,
    prior_games: int = 50,
    page_size: int = 100,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """Get ordered Solo/Duo IDs through 50 games before each player's earliest observation.

    Save only completed requests for each player. On reruns reuse a completed
    history if it still covers every current observation and its prior window.
    An interrupted player's old index is retained and can be refreshed later.
    """
    if not 1 <= page_size <= 100 or prior_games < 1 or checkpoint_every < 1:
        raise ValueError("Invalid history page size, window, or checkpoint interval")
    required = observations.groupby("puuid")["match_id"].agg(set).to_dict()
    saved = {puuid: group.copy() for puuid, group in index.groupby("puuid", sort=False)}
    changed = 0
    for number, (puuid, target_ids) in enumerate(required.items(), 1):
        old = saved.get(puuid)
        if old is not None and not old.empty:
            old = old.sort_values("history_position")
            locations = dict(zip(old["match_id"], old["history_position"]))
            if target_ids.issubset(locations) and (
                len(old) > max(locations[match_id] for match_id in target_ids) + prior_games
                or bool(old["history_exhausted"].iloc[0])
            ):
                continue
        match_ids: list[str] = []
        completed = False
        exhausted = False
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        for offset in range(0, 10000, page_size):
            try:
                page = riot_get(url, headers, {"queue": TARGET_QUEUE, "start": offset, "count": page_size})
            except requests.RequestException as error:
                print(f"Pre-game history skipped for player {number}/{len(required)}: {error}")
                break
            if not isinstance(page, list) or len(page) > page_size or len(set(page)) != len(page):
                raise ValueError(f"Unexpected match-history response for player {number}")
            match_ids.extend(page)
            if len(set(match_ids)) != len(match_ids):
                raise ValueError(f"Duplicate IDs across match-history pages for player {number}")
            exhausted = len(page) < page_size
            positions = {match_id: position for position, match_id in enumerate(match_ids)}
            if target_ids.issubset(positions) and (
                len(match_ids) > max(positions[match_id] for match_id in target_ids) + prior_games
                or exhausted
            ):
                completed = True
                break
            if exhausted:
                # Missing target IDs cannot be used to construct an exact window.
                break
            time.sleep(0.1)
        if not completed:
            print(f"Incomplete pre-game history for player {number}/{len(required)}; retry on resume")
            continue
        saved[puuid] = pd.DataFrame({
            "puuid": puuid, "match_id": match_ids,
            "history_position": range(len(match_ids)), "history_exhausted": exhausted,
        })
        changed += 1
        if changed % checkpoint_every == 0:
            save_pregame_history_index(pd.concat(saved.values(), ignore_index=True), processed_dir)
            print(f"Pre-game histories checkpointed: {changed:,}/{len(required):,} players refreshed")
        time.sleep(0.1)
    result = pd.concat(saved.values(), ignore_index=True) if saved else pd.DataFrame(columns=HISTORY_COLUMNS)
    return save_pregame_history_index(result, processed_dir)


def pregame_required_ids(
    observations: pd.DataFrame, history_index: pd.DataFrame, prior_games: int = 50,
) -> set[str]:
    """Return only the prior match IDs needed for observed player-games."""
    required: set[str] = set()
    targets = observations.groupby("puuid")["match_id"].agg(set).to_dict()
    for puuid, group in history_index.groupby("puuid", sort=False):
        if puuid not in targets:
            continue
        ids = group.sort_values("history_position")["match_id"].tolist()
        positions = {match_id: position for position, match_id in enumerate(ids)}
        for match_id in targets[puuid]:
            position = positions.get(match_id)
            if position is not None and len(ids) >= position + prior_games + 1:
                required.update(ids[position + 1:position + prior_games + 1])
    return required


def attach_pregame_usage(
    observations: pd.DataFrame,
    history_index: pd.DataFrame,
    raw_match_dir: Path,
    prior_games: int = 50,
) -> pd.DataFrame:
    """Count current champion in the 50 completed Solo/Duo games before each match.

    A missing or unreadable earlier game invalidates the entire window. No
    current game or future game is ever counted in champion_share.
    """
    if observations.empty:
        return observations.copy()
    result = observations.drop(columns=["champion_games", "player_games", "champion_share", "otp_measure"]).copy()
    ordered = {
        puuid: group.sort_values("history_position")["match_id"].tolist()
        for puuid, group in history_index.groupby("puuid", sort=False)
    }
    positions = {puuid: {match_id: i for i, match_id in enumerate(ids)} for puuid, ids in ordered.items()}
    targets = observations.groupby("puuid")["match_id"].agg(set).to_dict()
    needed_by_id: dict[str, set[str]] = {}
    for puuid, target_ids in targets.items():
        ids = ordered.get(puuid, [])
        for target in target_ids:
            position = positions.get(puuid, {}).get(target)
            if position is not None and len(ids) >= position + prior_games + 1:
                for older_id in ids[position + 1:position + prior_games + 1]:
                    needed_by_id.setdefault(older_id, set()).add(puuid)
    matches: dict[tuple[str, str], tuple[str, int, int]] = {}
    for match_id, puuids in needed_by_id.items():
        try:
            with (raw_match_dir / f"{match_id}.json").open(encoding="utf-8") as file:
                info = json.load(file)["info"]
            if info.get("queueId") != TARGET_QUEUE:
                continue
            start = int(info["gameStartTimestamp"])
            end = int(info.get("gameEndTimestamp") or (start + int(info["gameDuration"]) * 1000))
            for player in info["participants"]:
                if player.get("puuid") in puuids and player.get("championName"):
                    matches[(player["puuid"], match_id)] = (player["championName"], start, end)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    champion_games: list[int | None] = []
    for row in result.itertuples(index=False):
        ids = ordered.get(row.puuid, [])
        position = positions.get(row.puuid, {}).get(row.match_id)
        if position is None or len(ids) < position + prior_games + 1 or pd.isna(row.game_start_timestamp):
            champion_games.append(None)
            continue
        window = [matches.get((row.puuid, match_id)) for match_id in ids[position + 1:position + prior_games + 1]]
        # Require all 50 payloads, correct player, and completion before kickoff.
        if any(game is None or game[2] > int(row.game_start_timestamp) for game in window):
            champion_games.append(None)
        else:
            champion_games.append(sum(game[0] == row.champion for game in window))
    result["champion_games"] = pd.array(champion_games, dtype="Int64")
    result["player_games"] = pd.array(
        [prior_games if games is not None else None for games in champion_games], dtype="Int64"
    )
    result["champion_share"] = result["champion_games"] / result["player_games"]
    result["otp_measure"] = "previous_50_ranked_solo"
    return result.loc[:, DATASET_COLUMNS]


def save_otp_dataset(dataset: pd.DataFrame, processed_dir: Path, target_patch: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"otp_matches_{target_patch.replace('.', '_')}"
    dataset.to_csv(processed_dir / f"{stem}.csv", index=False)
    dataset.to_parquet(processed_dir / f"{stem}.parquet", index=False)


def build_unique_matchups(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return the distinct usable lane/champion/opponent combinations."""
    if dataset.empty:
        return pd.DataFrame(columns=UNIQUE_MATCHUP_COLUMNS)
    missing = set(UNIQUE_MATCHUP_COLUMNS).difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing matchup columns: {sorted(missing)}")
    return dataset.loc[:, UNIQUE_MATCHUP_COLUMNS].dropna().drop_duplicates().sort_values(
        UNIQUE_MATCHUP_COLUMNS
    ).reset_index(drop=True)


def save_unique_matchups(matchups: pd.DataFrame, processed_dir: Path, target_patch: str) -> None:
    """Persist unique matchups in matching CSV and Parquet outputs."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"unique_matchups_{target_patch.replace('.', '_')}"
    matchups.to_csv(processed_dir / f"{stem}.csv", index=False)
    matchups.to_parquet(processed_dir / f"{stem}.parquet", index=False)


def load_preferred_table(processed_dir: Path, stem: str) -> pd.DataFrame:
    """Load a processed Parquet table, falling back to its CSV counterpart."""
    parquet_path = processed_dir / f"{stem}.parquet"
    csv_path = processed_dir / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists.")


def prepare_expected_matchups(expected_matchups: pd.DataFrame, target_patch: str) -> pd.DataFrame:
    """Return a one-row-per-matchup successful LoLalytics lookup for a patch."""
    required = {
        "patch", "rank", "lane", "champion", "opponent_champion",
        "expected_winrate", "matchup_games", "status",
    }
    missing = required.difference(expected_matchups.columns)
    if missing:
        raise ValueError(f"Expected-winrate table is missing columns: {sorted(missing)}")
    lookup = expected_matchups.copy()
    lookup["patch"] = lookup["patch"].astype(str)
    lookup = lookup.loc[
        lookup["patch"].eq(str(target_patch)) & lookup["status"].eq("ok")
    ].rename(columns={"rank": "matchup_rank"})
    lookup["expected_winrate"] = pd.to_numeric(lookup["expected_winrate"], errors="coerce")
    lookup["matchup_games"] = pd.to_numeric(lookup["matchup_games"], errors="coerce")
    lookup = lookup.loc[:, EXPECTED_MATCHUP_COLUMNS].drop_duplicates()
    matchup_key = ["lane", "champion", "opponent_champion"]
    duplicates = lookup.duplicated(matchup_key, keep=False)
    if duplicates.any():
        duplicate_rows = lookup.loc[duplicates, matchup_key + ["matchup_rank"]]
        raise ValueError(
            "Expected-winrate lookup has multiple successful ranks for a matchup; "
            f"choose one rank before merging:\n{duplicate_rows.to_string(index=False)}"
        )
    return lookup.reset_index(drop=True)


def merge_expected_matchups(otp_matches: pd.DataFrame, expected_lookup: pd.DataFrame) -> pd.DataFrame:
    """Merge LoLalytics fields without changing the Riot-side ``patch`` column."""
    matchup_key = ["lane", "champion", "opponent_champion"]
    return otp_matches.merge(expected_lookup, on=matchup_key, how="left", validate="many_to_one")


def build_missing_matchups(otp_matches: pd.DataFrame, expected_lookup: pd.DataFrame) -> pd.DataFrame:
    """Find distinct Riot matchups that have no successful expected-winrate row."""
    matchup_key = ["lane", "champion", "opponent_champion"]
    otp_matchups = otp_matches.loc[:, matchup_key].dropna().drop_duplicates()
    known_matchups = expected_lookup.loc[:, matchup_key].drop_duplicates()
    return otp_matchups.merge(known_matchups, on=matchup_key, how="left", indicator=True).loc[
        lambda frame: frame["_merge"].eq("left_only"), matchup_key
    ].sort_values(matchup_key).reset_index(drop=True)


def save_missing_matchups(matchups: pd.DataFrame, processed_dir: Path, target_patch: str) -> None:
    """Save only unresolved matchup combinations for a future resumable scrape."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(processed_dir / f"missing_matchups_{target_patch.replace('.', '_')}.csv", index=False)


def _normalise_opponent_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(OPPONENT_INDEX_COLUMNS).issubset(frame.columns):
        return pd.DataFrame(columns=OPPONENT_INDEX_COLUMNS)
    return frame.loc[:, OPPONENT_INDEX_COLUMNS].dropna().drop_duplicates(
        OPPONENT_INDEX_COLUMNS, keep="first"
    ).reset_index(drop=True)


def load_opponent_match_index(processed_dir: Path) -> pd.DataFrame:
    """Load a persistent opponent history index, preferring Parquet."""
    try:
        return _normalise_opponent_index(load_preferred_table(processed_dir, "opponent_match_index"))
    except FileNotFoundError:
        return pd.DataFrame(columns=OPPONENT_INDEX_COLUMNS)


def save_opponent_match_index(index: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """Persist a deduplicated opponent history index in both requested formats."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    result = _normalise_opponent_index(index)
    result.to_csv(processed_dir / "opponent_match_index.csv", index=False)
    result.to_parquet(processed_dir / "opponent_match_index.parquet", index=False)
    return result


def add_seed_histories_to_opponent_index(
    opponent_index: pd.DataFrame,
    opponent_puuids: Iterable[str],
    player_match_index: pd.DataFrame,
) -> pd.DataFrame:
    """Reuse seed-player histories when an observed opponent is a known seed."""
    opponents = set(opponent_puuids)
    seed_rows = player_match_index.loc[
        player_match_index["puuid"].isin(opponents), ["puuid", "match_id"]
    ].rename(columns={"puuid": "opponent_puuid"})
    return _normalise_opponent_index(pd.concat([opponent_index, seed_rows], ignore_index=True))


def append_unindexed_opponent_histories(
    opponent_index: pd.DataFrame,
    opponent_puuids: Iterable[str],
    headers: dict,
    matches_per_opponent: int,
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Fetch histories only for opponents with no persisted index entries.

    The index is saved after each successful history response when
    ``processed_dir`` is supplied, so a later notebook run continues safely.
    Returns the updated index, the number of opponents that needed histories,
    and the number of successful history requests.
    """
    result = _normalise_opponent_index(opponent_index)
    existing_opponents = set(result["opponent_puuid"])
    unindexed = [puuid for puuid in dict.fromkeys(opponent_puuids) if puuid not in existing_opponents]
    successful_requests = 0
    for number, puuid in enumerate(unindexed, start=1):
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        try:
            match_ids = riot_get(
                url,
                headers,
                {"queue": TARGET_QUEUE, "start": 0, "count": matches_per_opponent},
            )
        except requests.HTTPError as error:
            print(f"Opponent history skipped {number}/{len(unindexed)}: {error}")
            continue
        new_rows = pd.DataFrame(
            [{"opponent_puuid": puuid, "match_id": match_id} for match_id in match_ids],
            columns=OPPONENT_INDEX_COLUMNS,
        )
        result = _normalise_opponent_index(pd.concat([result, new_rows], ignore_index=True))
        if processed_dir is not None:
            result = save_opponent_match_index(result, processed_dir)
        successful_requests += 1
        print(f"Opponent history {number}/{len(unindexed)}: {len(match_ids)} match IDs")
        time.sleep(0.1)
    return result, len(unindexed), successful_requests


def build_opponent_specialization(
    opponent_puuids: Iterable[str], raw_match_dir: Path, target_patch: str
) -> tuple[pd.DataFrame, int]:
    """Calculate opponent champion usage from every usable cached target-patch game."""
    opponents = set(opponent_puuids)
    usable_games: list[dict[str, str]] = []
    unreadable_files = 0
    for path in sorted(raw_match_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as file:
                match = json.load(file)
            info = match["info"]
            if patch_from_match(match) != target_patch or info.get("queueId") != TARGET_QUEUE:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            unreadable_files += 1
            continue
        for participant in info.get("participants", []):
            puuid = participant.get("puuid")
            champion = participant.get("championName")
            if puuid in opponents and champion:
                usable_games.append({
                    "opponent_puuid": puuid,
                    "match_id": path.stem,
                    "opponent_champion": champion,
                })
    usage = pd.DataFrame(usable_games)
    if usage.empty:
        return pd.DataFrame(columns=OPPONENT_SPECIALIZATION_COLUMNS), unreadable_files
    usage = usage.drop_duplicates(["opponent_puuid", "match_id"], keep="first")
    usage["opponent_champion_games"] = usage.groupby(
        ["opponent_puuid", "opponent_champion"]
    )["match_id"].transform("size")
    usage["opponent_player_games"] = usage.groupby("opponent_puuid")["match_id"].transform("size")
    usage["opponent_champion_share"] = (
        usage["opponent_champion_games"] / usage["opponent_player_games"]
    )
    return usage.loc[:, OPPONENT_SPECIALIZATION_COLUMNS].drop_duplicates(
        ["opponent_puuid", "opponent_champion"], keep="first"
    ).sort_values(["opponent_puuid", "opponent_champion"]).reset_index(drop=True), unreadable_files


def save_opponent_specialization(
    specialization: pd.DataFrame, processed_dir: Path, target_patch: str
) -> None:
    """Persist the reusable opponent specialization lookup in CSV and Parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"opponent_specialization_{target_patch.replace('.', '_')}"
    specialization.to_csv(processed_dir / f"{stem}.csv", index=False)
    specialization.to_parquet(processed_dir / f"{stem}.parquet", index=False)


def merge_opponent_specialization(
    analysis_df: pd.DataFrame, opponent_specialization: pd.DataFrame
) -> pd.DataFrame:
    """Attach opponent usage without discarding rows that lack cached coverage."""
    return analysis_df.merge(
        opponent_specialization,
        on=["opponent_puuid", "opponent_champion"],
        how="left",
        validate="many_to_one",
    )


def finalize_analysis_dataframe(
    otp_matches: pd.DataFrame,
    expected_lookup: pd.DataFrame,
    opponent_specialization: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all analysis inputs and add probability and specialization contrast."""
    analysis = merge_expected_matchups(otp_matches, expected_lookup)
    analysis = merge_opponent_specialization(analysis, opponent_specialization)
    analysis["expected_winrate_prob"] = analysis["expected_winrate"] / 100
    analysis["specialization_diff"] = analysis["champion_share"] - analysis["opponent_champion_share"]
    if analysis.duplicated(["match_id", "puuid"]).any():
        raise ValueError("Analysis merge produced duplicate seed-player match rows.")
    return analysis


def save_analysis_dataframe(analysis_df: pd.DataFrame, processed_dir: Path, target_patch: str) -> None:
    """Save the enriched analysis dataframe without touching its OTP source table."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"otp_analysis_{target_patch.replace('.', '_')}"
    analysis_df.to_csv(processed_dir / f"{stem}.csv", index=False)
    analysis_df.to_parquet(processed_dir / f"{stem}.parquet", index=False)
