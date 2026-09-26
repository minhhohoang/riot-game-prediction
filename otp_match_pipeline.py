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
]
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


def load_player_match_index(processed_dir: Path, seed_players: pd.DataFrame) -> pd.DataFrame:
    """Load the persistent index, migrating the old raw index if necessary."""
    current_path = processed_dir / "player_match_index.csv"
    legacy_path = processed_dir / "player_match_index_raw.csv"
    source = current_path if current_path.exists() else legacy_path
    return _normalise_index(_read_csv(source, INDEX_COLUMNS), seed_players)


def save_player_match_index(index: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    result = index.loc[:, INDEX_COLUMNS].drop_duplicates(["puuid", "match_id"], keep="first")
    result.to_csv(processed_dir / "player_match_index.csv", index=False)
    result.to_parquet(processed_dir / "player_match_index.parquet", index=False)
    return result


def append_match_histories(
    index: pd.DataFrame,
    seed_players: pd.DataFrame,
    headers: dict,
    matches_per_player: int,
) -> pd.DataFrame:
    """Fetch history IDs and append only new player-match relationships."""
    new_rows: list[dict[str, str]] = []
    for number, seed in enumerate(seed_players.itertuples(index=False), start=1):
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{seed.puuid}/ids"
        try:
            match_ids = riot_get(url, headers, {"queue": TARGET_QUEUE, "start": 0, "count": matches_per_player})
        except requests.HTTPError as error:
            print(f"History skipped for seed {number}/{len(seed_players)}: {error}")
            continue
        new_rows.extend({"puuid": seed.puuid, "match_id": match_id, "seed_tier": seed.seed_tier} for match_id in match_ids)
        print(f"Seed {number}/{len(seed_players)}: {len(match_ids)} history IDs")
        time.sleep(0.1)
    new_index = pd.DataFrame(new_rows, columns=INDEX_COLUMNS)
    return pd.concat([index, new_index], ignore_index=True).drop_duplicates(
        ["puuid", "match_id"], keep="first"
    ).reset_index(drop=True)


def download_missing_matches(
    index: pd.DataFrame,
    raw_match_dir: Path,
    headers: dict,
    max_new_downloads: int,
) -> tuple[int, int, int]:
    """Save only missing Match-V5 payloads in a round-robin tier order.

    Existing files never count toward the cap. A match associated with multiple
    seed tiers can appear in multiple tier queues, but is scheduled and fetched
    only once.
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
    tier_queues = {
        tier: deque(
            match_id for match_id in tiered_index.loc[
                tiered_index["seed_tier"] == tier, "match_id"
            ].drop_duplicates()
            if match_id in missing_set
        )
        for tier in tier_order
    }

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
        except requests.HTTPError as error:
            print(f"Match skipped ({match_id}): {error}")
            continue
        final_path = raw_match_dir / f"{match_id}.json"
        temporary_path = raw_match_dir / f".{match_id}.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)
        temporary_path.replace(final_path)
        downloads += 1
        print(f"New downloads this run: {downloads}/{max_new_downloads} ({match_id})")
        time.sleep(0.1)
    return len(cached_ids), len(missing_ids), downloads


def build_otp_dataset(index: pd.DataFrame, raw_match_dir: Path, target_patch: str) -> tuple[pd.DataFrame, int]:
    """Build one role-matched observation for each indexed seed-player appearance.

    Champion usage is calculated from every usable cached target-patch game for
    a seed player before any game is excluded for a missing or ambiguous role
    matchup. This makes specialization independent of role-data quality.
    """
    index = index.loc[:, INDEX_COLUMNS].drop_duplicates(["puuid", "match_id"], keep="first")
    seeds_by_match = index.groupby("match_id")[["puuid", "seed_tier"]].apply(
        lambda group: group.to_dict("records")
    ).to_dict()
    usable_games: list[dict] = []
    observations: list[dict] = []
    unreadable_files = 0
    for path in sorted(raw_match_dir.glob("*.json")):
        match_id = path.stem
        seed_rows = seeds_by_match.get(match_id)
        if not seed_rows:
            continue
        try:
            with path.open(encoding="utf-8") as file:
                match = json.load(file)
            info = match["info"]
            patch = patch_from_match(match)
        except (OSError, ValueError, KeyError, TypeError):
            unreadable_files += 1
            continue
        if patch != target_patch:
            continue
        participants = info.get("participants", [])
        by_puuid = {participant.get("puuid"): participant for participant in participants}
        for seed in seed_rows:
            player = by_puuid.get(seed["puuid"])
            champion = player.get("championName") if player else None
            if player and champion:
                usable_games.append({
                    "match_id": match_id,
                    "puuid": seed["puuid"],
                    "champion": champion,
                })
            role = ROLE_MAP.get(player.get("teamPosition", "")) if player else None
            if not player or not role or player.get("teamId") not in (100, 200):
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
    if dataset.empty or not usable_games:
        return pd.DataFrame(columns=DATASET_COLUMNS), unreadable_files
    usage = pd.DataFrame(usable_games).drop_duplicates(["puuid", "match_id"], keep="first")
    usage["champion_games"] = usage.groupby(["puuid", "champion"])["match_id"].transform("size")
    usage["player_games"] = usage.groupby("puuid")["match_id"].transform("size")
    usage["champion_share"] = usage["champion_games"] / usage["player_games"]
    dataset = dataset.merge(
        usage[["match_id", "puuid", "champion_games", "player_games", "champion_share"]],
        on=["match_id", "puuid"],
        how="left",
        validate="one_to_one",
    )
    return dataset.loc[:, DATASET_COLUMNS].sort_values(["game_start_timestamp", "match_id", "puuid"]).reset_index(drop=True), unreadable_files


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
