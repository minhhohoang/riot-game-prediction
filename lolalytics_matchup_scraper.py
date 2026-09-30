"""Direct, resumable LoLalytics matchup collector for the OTP analysis.

This version does not use ``lolalytics-api``. That package relies on a fixed
XPath that no longer matches the current site. Instead, this collector requests
the selected LoLalytics page directly and extracts its visible headline text.
Every attempt is appended to a JSONL checkpoint before the next request, so an
interrupted run resumes without repeating completed lookups.
"""

from __future__ import annotations

import csv
import html
import json
import random
import re
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests


BASE_URL = "https://lolalytics.com/lol"
LANE_SLUGS = {
    "top": "top",
    "jungle": "jungle",
    "mid": "middle",
    "adc": "bottom",
    "support": "support",
}
CHAMPION_SLUG_EXCEPTIONS = {"MonkeyKing": "wukong"}
KEY_COLUMNS = [
    "patch",
    "rank",
    "region",
    "lane",
    "champion",
    "opponent_champion",
]
OUTPUT_COLUMNS = KEY_COLUMNS + [
    "raw_matchup_winrate",
    "tier_average_winrate",
    "lolalytics_delta2",
    "expected_winrate",
    "matchup_games",
    "status",
    "source_url",
    "error",
    "scraped_at_utc",
    "snapshot_id",
]
TERMINAL_STATUSES = {"ok", "no_data"}


class _VisibleTextParser(HTMLParser):
    """Collect visible page text without depending on fragile HTML paths."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", html.unescape(" ".join(self.parts))).strip()


def champion_slug(champion: str) -> str:
    """Convert a Riot championName value to its LoLalytics URL slug."""
    if champion in CHAMPION_SLUG_EXCEPTIONS:
        return CHAMPION_SLUG_EXCEPTIONS[champion]
    return re.sub(r"[^a-z0-9]", "", champion.lower())


def build_matchup_url(
    champion: str,
    opponent_champion: str,
    lane: str,
    patch: str,
    rank: str,
    region: str,
) -> tuple[str, dict[str, str]]:
    """Build the exact LoLalytics page and query parameters."""
    if lane not in LANE_SLUGS:
        raise ValueError(f"Unsupported lane: {lane}")
    if region not in {"global", "na"}:
        raise ValueError(f"Unsupported region: {region}")

    url = (
        f"{BASE_URL}/{quote(champion_slug(champion))}/vs/"
        f"{quote(champion_slug(opponent_champion))}/build/"
    )
    params = {
        "lane": LANE_SLUGS[lane],
        "vslane": LANE_SLUGS[lane],
        "tier": rank,
        "patch": str(patch),
    }
    if region != "global":
        params["region"] = region
    return url, params


def _float_match(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_matchup_page(page_html: str) -> dict[str, float | int | str | None]:
    """Extract headline win rate, sample size, tier average, and Delta 2."""
    parser = _VisibleTextParser()
    parser.feed(page_html)
    text = parser.text()

    raw_match = re.search(
        r"wins against .+? ([0-9]+(?:\.[0-9]+)?)% of the time",
        text,
        flags=re.IGNORECASE,
    )
    tier_average = _float_match(
        r"Average\s+D2\+\s+Win\s+Rate:\s*([0-9]+(?:\.[0-9]+)?)%",
        text,
    )
    delta2 = _float_match(
        r"After normalising both champions win rates .+? "
        r"([+-]?[0-9]+(?:\.[0-9]+)?)% (?:more|less|different)",
        text,
    )

    if not raw_match:
        no_data_markers = (
            "not enough data",
            "no games",
            "0 games",
            "matchup requires a minimum",
        )
        if any(marker in text.lower() for marker in no_data_markers):
            return {
                "raw_matchup_winrate": None,
                "tier_average_winrate": tier_average,
                "lolalytics_delta2": delta2,
                "expected_winrate": None,
                "matchup_games": None,
                "status": "no_data",
                "error": "LoLalytics did not publish this matchup.",
            }
        raise ValueError("Could not find the headline matchup win rate.")

    raw_winrate = float(raw_match.group(1))
    nearby_text = text[raw_match.start() : raw_match.start() + 2500]
    games_match = re.search(
        r"[0-9]+(?:\.[0-9]+)?%\s+Win\s+Rate\s+([0-9,]+)\s+Games",
        nearby_text,
        flags=re.IGNORECASE,
    )
    if not games_match:
        games_match = re.search(
            r"([0-9,]+)\s+Games",
            nearby_text,
            flags=re.IGNORECASE,
        )
    if not games_match:
        raise ValueError("Could not find the headline matchup game count.")

    matchup_games = int(games_match.group(1).replace(",", ""))
    expected_winrate = (
        raw_winrate - tier_average + 50
        if tier_average is not None
        else raw_winrate
    )
    return {
        "raw_matchup_winrate": raw_winrate,
        "tier_average_winrate": tier_average,
        "lolalytics_delta2": delta2,
        "expected_winrate": expected_winrate,
        "matchup_games": matchup_games,
        "status": "ok",
        "error": "",
    }


def load_unique_matchups(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"lane", "champion", "opponent_champion"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Unique-matchup file is missing columns: {sorted(missing)}")
    frame = frame.loc[:, ["lane", "champion", "opponent_champion"]].dropna()
    return (
        frame.drop_duplicates()
        .sort_values(["lane", "champion", "opponent_champion"])
        .reset_index(drop=True)
    )


def _row_key(row: dict | pd.Series) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in KEY_COLUMNS)


def load_checkpoint(path: Path) -> dict[tuple[str, ...], dict]:
    latest: dict[tuple[str, ...], dict] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                latest[_row_key(row)] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"Ignoring malformed checkpoint line {line_number}.")
    return latest


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def export_results(
    rows: Iterable[dict], output_csv: Path, output_parquet: Path
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    if frame.empty:
        frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        frame = (
            frame.loc[:, OUTPUT_COLUMNS]
            .sort_values(KEY_COLUMNS)
            .reset_index(drop=True)
        )
    frame.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    try:
        frame.to_parquet(output_parquet, index=False)
    except ImportError:
        print(
            "CSV export succeeded, but Parquet export requires pyarrow. "
            "Run: python -m pip install pyarrow"
        )
    return frame


def _request_page(
    session: requests.Session,
    url: str,
    params: dict[str, str],
    max_attempts: int = 4,
) -> str:
    """Request one page with bounded handling for rate limits/server errors."""
    for attempt in range(1, max_attempts + 1):
        response = session.get(url, params=params, timeout=45)
        if response.status_code == 429:
            wait_seconds = int(response.headers.get("Retry-After", 60)) + 1
            print(f"LoLalytics rate limited the request; waiting {wait_seconds}s.")
            time.sleep(wait_seconds)
            continue
        if response.status_code >= 500 and attempt < max_attempts:
            wait_seconds = 2 ** attempt
            print(f"LoLalytics returned {response.status_code}; retrying in {wait_seconds}s.")
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.text
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {url}")


def create_session() -> requests.Session:
    """Create a browser-like session shared across sequential lookups."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/153.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def scrape_one_matchup(
    champion: str,
    opponent_champion: str,
    lane: str,
    patch: str = "16.19",
    rank: str = "d2_plus",
    region: str = "global",
    session: requests.Session | None = None,
) -> tuple[str, dict[str, float | int | str | None]]:
    """Fetch and parse one matchup, used by both preflight and collection."""
    session = session or create_session()
    url, params = build_matchup_url(
        champion=champion,
        opponent_champion=opponent_champion,
        lane=lane,
        patch=patch,
        rank=rank,
        region=region,
    )
    source_url = requests.Request("GET", url, params=params).prepare().url
    page_html = _request_page(session, url, params)
    return source_url, parse_matchup_page(page_html)


def collect_expected_matchups(
    unique_matchups_path: Path,
    processed_dir: Path,
    patch: str = "16.19",
    rank: str = "d2_plus",
    regions: tuple[str, ...] = ("global",),
    delay_seconds: float = 1.25,
    refresh_snapshot: str | None = None,
) -> pd.DataFrame:
    """Collect missing lookups, optionally refreshing each key once per snapshot.

    Pass an explicit YYYY-MM-DD ``refresh_snapshot`` to fetch even previously
    completed keys. Each successful/no-data result gets that snapshot ID in the
    append-only checkpoint, so rerunning after interruption skips those keys.
    Without a snapshot ID, the original resume/skip behavior is preserved.
    """
    if refresh_snapshot is not None:
        try:
            valid_date = date.fromisoformat(refresh_snapshot)
        except (TypeError, ValueError) as error:
            raise ValueError("refresh_snapshot must be YYYY-MM-DD or None.") from error
        if valid_date.isoformat() != refresh_snapshot:
            raise ValueError("refresh_snapshot must be YYYY-MM-DD or None.")

    matchups = load_unique_matchups(unique_matchups_path)
    stem = f"expected_matchups_{str(patch).replace('.', '_')}"
    checkpoint_path = processed_dir / f"{stem}_checkpoint.jsonl"
    output_csv = processed_dir / f"{stem}.csv"
    output_parquet = processed_dir / f"{stem}.parquet"
    latest = load_checkpoint(checkpoint_path)

    tasks: list[dict[str, str]] = []
    for region in regions:
        if region not in {"global", "na"}:
            raise ValueError(f"Unsupported region: {region}")
        for matchup in matchups.itertuples(index=False):
            tasks.append(
                {
                    "patch": str(patch),
                    "rank": rank,
                    "region": region,
                    "lane": matchup.lane,
                    "champion": matchup.champion,
                    "opponent_champion": matchup.opponent_champion,
                }
            )

    required_keys = {_row_key(task) for task in tasks}
    if refresh_snapshot is not None:
        newer = [
            key for key in required_keys
            if key in latest and str(latest[key].get("snapshot_id") or "") > refresh_snapshot
        ]
        if newer:
            raise ValueError(
                f"{len(newer):,} required lookups have a newer snapshot than "
                f"{refresh_snapshot}. Choose that date or a later date."
            )
    completed = {
        key for key in required_keys
        if key in latest
        and latest[key].get("status") in TERMINAL_STATUSES
        and (
            refresh_snapshot is None
            or latest[key].get("snapshot_id") == refresh_snapshot
        )
    }
    remaining = [task for task in tasks if _row_key(task) not in completed]
    print(f"Required matchup-region lookups: {len(tasks):,}")
    print(f"Refresh snapshot: {refresh_snapshot or 'off (new/error keys only)'}")
    print(f"Completed for this run: {len(completed):,}")
    print(f"Requests planned: {len(remaining):,}")

    session = create_session()

    try:
        for number, task in enumerate(remaining, start=1):
            base_row = {
                **task,
                "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": refresh_snapshot,
            }
            try:
                source_url, parsed = scrape_one_matchup(
                    **task,
                    session=session,
                )
                row = {**base_row, "source_url": source_url, **parsed}
            except (requests.RequestException, RuntimeError, ValueError) as error:
                row = {
                    **base_row,
                    "source_url": "",
                    "raw_matchup_winrate": None,
                    "tier_average_winrate": None,
                    "lolalytics_delta2": None,
                    "expected_winrate": None,
                    "matchup_games": None,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }

            append_checkpoint(checkpoint_path, row)
            latest[_row_key(row)] = row
            print(
                f"Lookup {number:,}/{len(remaining):,} | {task['region']} | "
                f"{task['lane']} {task['champion']} vs "
                f"{task['opponent_champion']} | {row['status']}"
            )
            if number % 25 == 0:
                export_results(latest.values(), output_csv, output_parquet)
            time.sleep(delay_seconds + random.uniform(0.0, 0.35))
    except KeyboardInterrupt:
        print("Collection interrupted; completed checkpoint rows are safe.")
    finally:
        result = export_results(latest.values(), output_csv, output_parquet)
        if refresh_snapshot is not None:
            unfinished = sum(
                key not in latest
                or latest[key].get("status") not in TERMINAL_STATUSES
                or latest[key].get("snapshot_id") != refresh_snapshot
                for key in required_keys
            )
            print(f"Snapshot {refresh_snapshot}: {unfinished:,} lookups unfinished.")
            if unfinished:
                print("Resume this snapshot before running the final merge.")

    return result


if __name__ == "__main__":
    project_dir = Path.cwd()
    result = collect_expected_matchups(
        unique_matchups_path=(
            project_dir / "data" / "processed" / "unique_matchups_16_19.csv"
        ),
        processed_dir=project_dir / "data" / "processed",
    )
    print(result["status"].value_counts(dropna=False))
