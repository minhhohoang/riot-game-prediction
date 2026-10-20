"""Independent, retrospective Global/NA Master+ longitudinal matchup collection.

Import is inert. Only collect() makes requests, when explicitly called by the
companion notebook. No V1 code/data or Riot APIs are used. Rates are percentages;
NA-minus-Global differences and Delta 2 use percentage points. Derived
tier-centered rates remain separate percentage-unit values, not probabilities.

The website representation was inspected on 2026-10-20. Parsing is
intentionally conservative: missing content evidence is a failure, never proof
that a query parameter was honored. Raw responses are retained for diagnosis.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://lolalytics.com/lol"
SCHEMA_VERSION = "hlr-master-matchups-v3-qwik"
REGIONS = ('global', 'na')
LOOKUP_KEYS = ['patch', 'rank', 'region', 'queue', 'lane', 'champion', 'opponent_champion']
PAIR_KEYS = [c for c in LOOKUP_KEYS if c != 'region']
# Explicit website role names; no reverse-rate or cross-role inference.
ROLE_SLUGS = {"top": "top", "jungle": "jungle", "mid": "middle", "adc": "bottom", "support": "support"}
ROLE_NAMES = {"top": ("Top",), "jungle": ("Jungle",), "mid": ("Mid", "Middle"),
              "adc": ("ADC", "Bottom"), "support": ("Support", "Utility")}
TEAM_ROLES = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "adc", "UTILITY": "support"}
# Reviewed, explicit Riot-ID : Riot-name : website-slug registry. No automatic
# punctuation-stripping fallback for unregistered champions. Page identity must
# still confirm each mapping at runtime (including newer champions).
_CHAMPIONS = """
1 Annie annie
2 Olaf olaf
3 Galio galio
4 TwistedFate twistedfate
5 XinZhao xinzhao
6 Urgot urgot
7 Leblanc leblanc
8 Vladimir vladimir
9 FiddleSticks fiddlesticks
10 Kayle kayle
11 MasterYi masteryi
12 Alistar alistar
13 Ryze ryze
14 Sion sion
15 Sivir sivir
16 Soraka soraka
17 Teemo teemo
18 Tristana tristana
19 Warwick warwick
20 Nunu nunu
21 MissFortune missfortune
22 Ashe ashe
23 Tryndamere tryndamere
24 Jax jax
25 Morgana morgana
26 Zilean zilean
27 Singed singed
28 Evelynn evelynn
29 Twitch twitch
30 Karthus karthus
31 Chogath chogath
32 Amumu amumu
33 Rammus rammus
34 Anivia anivia
35 Shaco shaco
36 DrMundo drmundo
37 Sona sona
38 Kassadin kassadin
39 Irelia irelia
40 Janna janna
41 Gangplank gangplank
42 Corki corki
43 Karma karma
44 Taric taric
45 Veigar veigar
48 Trundle trundle
50 Swain swain
51 Caitlyn caitlyn
53 Blitzcrank blitzcrank
54 Malphite malphite
55 Katarina katarina
56 Nocturne nocturne
57 Maokai maokai
58 Renekton renekton
59 JarvanIV jarvaniv
60 Elise elise
61 Orianna orianna
62 MonkeyKing wukong
63 Brand brand
64 LeeSin leesin
67 Vayne vayne
68 Rumble rumble
69 Cassiopeia cassiopeia
72 Skarner skarner
74 Heimerdinger heimerdinger
75 Nasus nasus
76 Nidalee nidalee
77 Udyr udyr
78 Poppy poppy
79 Gragas gragas
80 Pantheon pantheon
81 Ezreal ezreal
82 Mordekaiser mordekaiser
83 Yorick yorick
84 Akali akali
85 Kennen kennen
86 Garen garen
89 Leona leona
90 Malzahar malzahar
91 Talon talon
92 Riven riven
96 KogMaw kogmaw
98 Shen shen
99 Lux lux
101 Xerath xerath
102 Shyvana shyvana
103 Ahri ahri
104 Graves graves
105 Fizz fizz
106 Volibear volibear
107 Rengar rengar
110 Varus varus
111 Nautilus nautilus
112 Viktor viktor
113 Sejuani sejuani
114 Fiora fiora
115 Ziggs ziggs
117 Lulu lulu
119 Draven draven
120 Hecarim hecarim
121 Khazix khazix
122 Darius darius
126 Jayce jayce
127 Lissandra lissandra
131 Diana diana
133 Quinn quinn
134 Syndra syndra
136 AurelionSol aurelionsol
141 Kayn kayn
142 Zoe zoe
143 Zyra zyra
145 Kaisa kaisa
147 Seraphine seraphine
150 Gnar gnar
154 Zac zac
157 Yasuo yasuo
161 Velkoz velkoz
163 Taliyah taliyah
164 Camille camille
166 Akshan akshan
200 Belveth belveth
201 Braum braum
202 Jhin jhin
203 Kindred kindred
221 Zeri zeri
222 Jinx jinx
223 TahmKench tahmkench
233 Briar briar
234 Viego viego
235 Senna senna
236 Lucian lucian
238 Zed zed
240 Kled kled
245 Ekko ekko
246 Qiyana qiyana
254 Vi vi
266 Aatrox aatrox
267 Nami nami
268 Azir azir
350 Yuumi yuumi
360 Samira samira
412 Thresh thresh
420 Illaoi illaoi
421 RekSai reksai
427 Ivern ivern
429 Kalista kalista
432 Bard bard
497 Rakan rakan
498 Xayah xayah
516 Ornn ornn
517 Sylas sylas
518 Neeko neeko
523 Aphelios aphelios
526 Rell rell
555 Pyke pyke
711 Vex vex
777 Yone yone
799 Ambessa ambessa
800 Mel mel
804 Yunara yunara
805 Locke locke
875 Sett sett
876 Lillia lillia
887 Gwen gwen
888 Renata renata
893 Aurora aurora
895 Nilah nilah
897 KSante ksante
901 Smolder smolder
902 Milio milio
904 Zaahen zaahen
910 Hwei hwei
950 Naafiri naafiri
"""
CHAMPIONS = {int(cid): {"riot_name": name, "slug": slug}
             for cid, name, slug in (line.split() for line in _CHAMPIONS.splitlines() if line.strip())}
# Display aliases are explicit too; the ID registry remains the authority.
DISPLAY_ALIASES = {
    4: ("Twisted Fate",), 5: ("Xin Zhao",), 9: ("Fiddlesticks",), 11: ("Master Yi",),
    20: ("Nunu & Willump", "Nunu and Willump"), 21: ("Miss Fortune",), 31: ("Cho'Gath",),
    36: ("Dr. Mundo", "Dr Mundo"), 59: ("Jarvan IV",), 62: ("Wukong",), 64: ("Lee Sin",),
    96: ("Kog'Maw",), 121: ("Kha'Zix",), 136: ("Aurelion Sol",), 145: ("Kai'Sa",),
    161: ("Vel'Koz",), 200: ("Bel'Veth",), 223: ("Tahm Kench",), 421: ("Rek'Sai",),
    888: ("Renata Glasc",), 897: ("K'Sante",),
}
FILTERS = {"patch": "16.19", "rank": "master_plus", "region": "global", "queue": "ranked"}
TERMINAL = {"ok", "no_data", "unsupported_combination", "unsupported_mapping", "not_published_in_table"}
PAGE_UNRESOLVED = 'page_unavailable_unresolved'
SKIP_ON_RESUME = TERMINAL | {PAGE_UNRESOLVED}
RATE_FIELDS = ["raw_matchup_winrate_pct", "matchup_games", "matchup_wins", "matchup_losses",
               "tier_average_winrate_pct", "derived_tier_centered_rate_pct", "website_delta2_pp"]


@dataclass(frozen=True)
class Config:
    patch: str = "16.19"
    rank: str = "master_plus"
    region: str = "global"  # Primary benchmark; retained for v1 plan-hash compatibility.
    regions: tuple[str, ...] = REGIONS
    queue: str = "ranked"
    delay_seconds: float = 2.0
    timeout_connect: float = 10.0
    timeout_read: float = 45.0
    max_attempts: int = 3
    backoff_seconds: float = 5.0
    max_retry_after_seconds: float = 300.0
    max_consecutive_failures: int = 5
    export_every_pages: int = 20
    inspect_batch_tables: bool = True
    # Table and directional endpoints disagreed in diagnostics. Do not silently
    # mix the two representations or spend thousands of fallback requests.
    directional_fallback: bool = False
    include_current_pick_in_candidates: bool = False
    # All observed prior same-role champions are used: no experience threshold.
    candidate_roles: tuple[str, ...] = ("top", "jungle", "mid", "adc", "support")

    def __post_init__(self):
        if self.directional_fallback or not self.inspect_batch_tables:
            raise ValueError('Use verified champion/lane tables; directional fallback remains disabled.')
        for k, v in FILTERS.items():
            if getattr(self, k) != v:
                raise ValueError(f"This collector requires {k}={v!r}; no filter fallback.")
        if tuple(self.regions) != REGIONS:
            raise ValueError('Collect both regions in primary/sensitivity order: (global, na).')
        if not self.candidate_roles or not set(self.candidate_roles) <= set(ROLE_SLUGS):
            raise ValueError("Unknown/empty candidate role selection")
        if (self.delay_seconds < 1 or self.max_attempts < 1 or self.max_consecutive_failures < 1
                or self.export_every_pages < 1 or min(self.timeout_connect, self.timeout_read) <= 0
                or self.backoff_seconds < 0 or self.max_retry_after_seconds < 0):
            raise ValueError("Invalid pacing / retry configuration")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_event(path, value):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open('rb') as tail:
            tail.seek(-1, os.SEEK_END)
            terminated = tail.read(1) == b'\n'
        if not terminated:
            read_events(path)  # Preserve/repair a torn append before appending.
            if path.stat().st_size:
                with path.open('ab') as f:
                    f.write(b'\n')
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_events(path):
    """Recover a torn FINAL append only; never silently skip interior corruption."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise ValueError(f"Corrupt checkpoint at {path}:{i+1}")
            # Repair only the torn trailing append before another append can occur.
            # All complete events and the original damaged bytes are preserved.
            original = Path(path).read_bytes()
            backup = Path(path).with_name(Path(path).name + f".torn-{time.time_ns()}")
            backup.write_bytes(original)
            with Path(path).open("w", encoding="utf-8") as f:
                for event in events:
                    f.write(canonical(event)+"\n")
            print(f"Recovered torn final checkpoint line; original saved at {backup.name}")
    return events


def find_project_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for root in [p, *p.parents]:
        if (root/"hlr_model/longitudinal_pipeline.py").is_file():
            return root
    raise FileNotFoundError("Open from repository root or hlr_model/.")


def load_snapshot(root, pinned=None):
    """Read actual schemas; reject inconsistent snapshots, not incomplete players."""
    processed = Path(root)/"hlr_model/data/runs/patch_16_19_seed_1619/processed"
    paths = [processed/pinned] if pinned else sorted(processed.iterdir(), reverse=True)
    rejected = []
    for path in paths:
        if not path.is_dir() or not re.fullmatch(r"\d{8}T\d{12}Z", path.name):
            continue
        try:
            names = ["longitudinal_modeling", "historical_matches", "target_matches"]
            frames = {n: pd.read_parquet(path/f"{n}.parquet") for n in names}
            required = {"puuid", "match_id", "timestamp", "end_timestamp", "patch", "queue", "lane",
                        "champion", "champion_id", "opponent_champion", "opponent_champion_id", "matchup_missing"}
            for n, f in frames.items():
                if not required <= set(f):
                    raise ValueError(f"{n}: missing columns {required-set(f)}")
                if f.duplicated(["puuid", "match_id"]).any():
                    raise ValueError(f"{n}: duplicate player/match")
                if not f.queue.eq(420).all() or not f.end_timestamp.ge(f.timestamp).all():
                    raise ValueError(f"{n}: invalid queue/times")
                for field in ['champion_id', 'opponent_champion_id']:
                    ids = f[field].dropna()
                    if not (ids.gt(0) & ids.mod(1).eq(0)).all():
                        raise ValueError(f'{n}: invalid {field}')
            model, history, targets = [frames[n] for n in names]
            if targets.empty or not targets.patch.eq("16.19").all():
                raise ValueError("Empty/wrong-patch targets")
            keys = ["puuid", "match_id"]
            pd.testing.assert_frame_equal(model.set_index(keys).sort_index()[targets.drop(columns=keys).columns],
                                          targets.set_index(keys).sort_index(), check_dtype=False)
            if set(map(tuple, history[keys].values)) & set(map(tuple, targets[keys].values)):
                raise ValueError("Initial/target overlap")
            anchors = targets.groupby("puuid").timestamp.min()
            if not history.end_timestamp.lt(history.puuid.map(anchors)).all():
                raise ValueError("History not strictly before first target")
            combined = pd.concat([history, targets])
            if combined.groupby("champion_id").champion.nunique().gt(1).any():
                raise ValueError("Inconsistent champion ID/name pairs")
            hashes = {n: digest((path/f"{n}.parquet").read_bytes()) for n in names}
            print("Selected longitudinal snapshot:", path.relative_to(root))
            for n, f in frames.items():
                print(f"  {n}: {len(f):,} rows x {len(f.columns)} columns")
            if rejected:
                print("Rejected newer snapshots:", rejected)
            return path, model, history, targets, hashes
        except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
            rejected.append({"snapshot": path.name, "error": str(exc)[:400]})
            if pinned:
                raise ValueError(f"Pinned snapshot no longer valid: {rejected[-1]}") from exc
    raise FileNotFoundError(f"No valid longitudinal snapshot; rejected={rejected}")


def registry_entry(cid, original_name):
    if pd.isna(cid) or pd.isna(original_name) or float(cid) != int(cid):
        return None
    entry = CHAMPIONS.get(int(cid))
    if entry is None or original_name not in (entry["riot_name"], *DISPLAY_ALIASES.get(int(cid), ())):
        return None
    return entry


def matchup_key_id(lane, cid, oid):
    return f"{lane}:{int(cid)}:{int(oid)}"


def key_id(lane, cid, oid, region, config):
    return f'{config.patch}:{config.rank}:{region}:{config.queue}:{matchup_key_id(lane, cid, oid)}'


def url_region(url):
    if not url:
        return 'unknown'
    values = parse_qs(urlparse(url).query).get('region', [])
    if not values:
        return 'global'
    return values[0] if len(values) == 1 and values[0] in REGIONS else 'unknown'


def build_url(champion_slug, lane, config, opponent_slug=None, region='global'):
    # queue=ranked causes HTTP 404. Ranked Solo/Duo is the default and must
    # still be confirmed by returned content and dataset header.queue == 420.
    path = (f"/{champion_slug}/vs/{opponent_slug}/build/" if opponent_slug
            else f"/{champion_slug}/build/")
    params = {"lane": ROLE_SLUGS[lane], "tier": config.rank, "patch": config.patch}
    if region not in REGIONS:
        raise ValueError('Unsupported region; no fallback')
    if region != 'global':
        params['region'] = region  # V1 website convention: NA is region=na.
    if opponent_slug:
        params["vslane"] = ROLE_SLUGS[lane]
    return BASE_URL + path + "?" + urlencode(sorted(params.items()))


def _build_base_plan(model, history, targets, config):
    """Preserved history + strictly completed earlier targets, same focal role.

    Candidates are previously PLAYED champions, not known available draft picks.
    Global pregame main uses the pipeline's all-role count/tie rule; a main is
    eligible only if it also has prior games in the target role. No min cutoff.
    """
    requests_by_key, mapping, decisions = {}, [], []
    hist = {p: f for p, f in history.groupby("puuid")}
    for p, group in targets.groupby("puuid", sort=True):
        pool = pd.concat([hist.get(p, history.iloc[:0]), group], ignore_index=True)
        pool = pool.drop_duplicates(["puuid", "match_id"]).sort_values(["end_timestamp", "match_id"])
        available = list(pool.itertuples(index=False))
        counts, role_counts, names = Counter(), Counter(), {}
        latest_role_end = {}
        cursor = 0
        for r in group.sort_values(["timestamp", "match_id"]).itertuples(index=False):
            while cursor < len(available) and available[cursor].end_timestamp < r.timestamp:
                prior = available[cursor]
                if prior.timestamp >= r.timestamp:
                    raise ValueError("Invalid prior time ordering")
                cid = int(prior.champion_id)
                counts[cid] += 1
                names[cid] = prior.champion
                if prior.lane in ROLE_SLUGS:
                    role_counts[(prior.lane, cid)] += 1
                    latest_role_end[(prior.lane, cid)] = int(prior.end_timestamp)
                cursor += 1
            main = min(counts, key=lambda cid: (-counts[cid], cid)) if counts else None
            known = (not r.matchup_missing and r.lane in ROLE_SLUGS
                     and pd.notna(r.opponent_champion_id) and pd.notna(r.opponent_champion))
            decision = {"puuid": p, "match_id": r.match_id, "timestamp": int(r.timestamp), "lane": r.lane,
                        "champion_id": int(r.champion_id), "opponent_champion_id": r.opponent_champion_id,
                        "known_lane_opponent": bool(known), "pregame_main_id": main,
                        "candidate_generation_in_scope": bool(known and r.lane in config.candidate_roles)}
            decisions.append(decision)
            if not known:
                continue
            candidates = {cid for (lane, cid), n in role_counts.items() if lane == r.lane and n > 0}
            if r.lane not in config.candidate_roles:
                candidates = set()
            if not config.include_current_pick_in_candidates:
                candidates.discard(int(r.champion_id))
            candidates.discard(int(r.opponent_champion_id))
            decision['unexpected_observed_mirror'] = int(r.champion_id) == int(r.opponent_champion_id)
            if decision['unexpected_observed_mirror']:
                # Preserve the target in decisions and a separate investigation
                # artifact, but never issue impossible ranked mirror requests.
                continue
            supports = [(int(r.champion_id), r.champion, "observed")]
            supports += [(cid, names[cid], "alternative") for cid in sorted(candidates)]
            for cid, name, kind in supports:
                oid, lane = int(r.opponent_champion_id), r.lane
                kid = matchup_key_id(lane, cid, oid)
                c, o = registry_entry(cid, name), registry_entry(oid, r.opponent_champion)
                if kid not in requests_by_key:
                    requests_by_key[kid] = {"key_id": kid, "lane": lane, "champion_id": cid, "champion": name,
                        "opponent_champion_id": oid, "opponent_champion": r.opponent_champion,
                        "champion_slug": c["slug"] if c else None, "opponent_slug": o["slug"] if o else None,
                        "website_lane": ROLE_SLUGS[lane], "observed": False, "alternative": False,
                        "requested_patch": config.patch, "requested_rank": config.rank,
                        "requested_region": config.region, "requested_queue": config.queue,
                        "requested_queue_id": 420, "mapping_verified_locally": bool(c and o)}
                request = requests_by_key[kid]
                request[kind] = True
                request["page_url"] = build_url(c["slug"], lane, config, o["slug"]) if c and o else None
                request["batch_url"] = build_url(c["slug"], lane, config) if c and o else None
                mapping.append({"puuid": p, "match_id": r.match_id, "timestamp": int(r.timestamp),
                    "key_id": kid, "request_kind": kind, "candidate_champion_id": cid,
                    "candidate_champion": name, "lane": lane, "pregame_main_id": main,
                    "is_pregame_main": cid == main, "prior_same_role_games": role_counts[(lane, cid)],
                    "prior_all_role_games": counts[cid], "latest_same_role_end": latest_role_end.get((lane, cid)),
                    "availability_in_draft_verified": False})
    request_columns = ["key_id", "lane", "champion_id", "champion", "opponent_champion_id", "opponent_champion",
                       "champion_slug", "opponent_slug", "website_lane", "observed", "alternative", "requested_patch",
                       "requested_rank", "requested_region", "requested_queue", "requested_queue_id",
                       "mapping_verified_locally", "page_url", "batch_url"]
    plan = pd.DataFrame(requests_by_key.values(), columns=request_columns).sort_values("key_id").reset_index(drop=True)
    mapping_columns = ['puuid', 'match_id', 'timestamp', 'key_id', 'request_kind', 'candidate_champion_id',
                       'candidate_champion', 'lane', 'pregame_main_id', 'is_pregame_main', 'prior_same_role_games',
                       'prior_all_role_games', 'latest_same_role_end', 'availability_in_draft_verified']
    mapping = pd.DataFrame(mapping, columns=mapping_columns).drop_duplicates(["puuid", "match_id", "key_id", "request_kind"])
    if plan.empty:
        raise ValueError('No known-opponent requests can be built; collection will not start')
    if plan.key_id.duplicated().any():
        raise ValueError("Duplicate request keys")
    return plan, mapping, pd.DataFrame(decisions)


def regionalize_plan(base_plan, base_mapping, config):
    """Replicate the SAME observed/candidate scope; never change eligibility."""
    plans, mappings = [], []
    for region in config.regions:
        plan = base_plan.copy()
        plan['matchup_key_id'] = plan.key_id
        plan['key_id'] = [key_id(r.lane, r.champion_id, r.opponent_champion_id, region, config)
                          for r in plan.itertuples()]
        plan['requested_region'] = region
        for field in ['patch', 'rank', 'queue']:
            plan[field] = getattr(config, field)
        plan['region'] = region
        for field, opposing in [('page_url', True), ('batch_url', False)]:
            plan[field] = [build_url(r.champion_slug, r.lane, config,
                                    r.opponent_slug if opposing else None, region=region)
                           if r.mapping_verified_locally else None for r in plan.itertuples()]
        mapping = base_mapping.rename(columns={'key_id': 'matchup_key_id'}).copy()
        mapping = mapping.merge(plan[['matchup_key_id', 'key_id']], on='matchup_key_id',
                                how='left', validate='many_to_one')
        mapping['region'] = region
        plans.append(plan)
        mappings.append(mapping)
    plan = pd.concat(plans, ignore_index=True)
    if plan.key_id.duplicated().any() or plan.duplicated(LOOKUP_KEYS).any():
        raise ValueError('Duplicate full regional request identities')
    if not plan.groupby('matchup_key_id').region.nunique().eq(len(REGIONS)).all():
        raise ValueError('Both regions must be planned for every key')
    return plan, pd.concat(mappings, ignore_index=True)


def build_plan(model, history, targets, config):
    base_plan, base_mapping, decisions = _build_base_plan(model, history, targets, config)
    plan, mapping = regionalize_plan(base_plan, base_mapping, config)
    return plan, mapping, decisions


class EvidenceError(ValueError):
    def __init__(self, status, message, evidence=None):
        super().__init__(message)
        self.status, self.evidence = status, evidence or {}


def text_of(node):
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def name_pattern(cid):
    names = [CHAMPIONS[int(cid)]["riot_name"], *DISPLAY_ALIASES.get(int(cid), ())]
    # Normalize curly apostrophes only in the HTML text; aliases themselves are explicit.
    return "(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + ")"


def evidence_contexts(soup):
    """Content only. Query URLs, arbitrary scripts, menus and links aren't proof.

    Selected option/active controls, rendered headings and compact currently
    displayed filter buttons are admissible. Hidden/menu choices are excluded.
    No arbitrary recursive JSON search: a request-query object can merely echo
    requested filters and is not authoritative returned dataset metadata.
    """
    out = []
    # Reviewed Qwik navigation representation: the compact scroller contains
    # current values, while choices are rendered elsewhere after interaction.
    # Scope by the actual rank badge and region icon; never scan menu text.
    for badge in soup.select('img[src*="/emblem40/"]'):
        nav = next((a for a in badge.parents if a.name == 'div'
                    and any('_scroller_' in c for c in a.get('class', []))), None)
        if nav is None or not nav.select_one('img[src*="/region35/"]'):
            continue
        for child in nav.find_all(recursive=False):
            value = text_of(child)
            label = None
            if child.select_one('img[src*="/emblem40/"]'):
                label = 'rank'
            elif child.select_one('img[src*="/region35/"]'):
                label = 'region'
            elif re.fullmatch(r'Patch\s+\d+\.\d+', value, re.I):
                label = 'patch'
            elif re.fullmatch(r'Ranked Solo/Duo|Ranked Flex|ARAM|Arena', value, re.I):
                label = 'queue'
            if label:
                out.append({'tag':'div', 'text':value, 'label':label,
                            'selected':True, 'representation':'qwik_current_navigation'})
    for node in soup.select('select option[selected], [aria-selected="true"], [aria-pressed="true"], '
                            '[role="radio"][aria-checked="true"], [role="combobox"], h1, h2, h3, button'):
        ancestors = [node, *node.parents]
        if any(getattr(a, "attrs", {}).get("hidden") is not None
               or getattr(a, "attrs", {}).get("aria-hidden") == "true"
               or re.search(r"display\s*:\s*none", str(getattr(a, "attrs", {}).get("style", ""))) for a in ancestors):
            continue
        if node.name == "button" and node.get("role") in {"option", "menuitem", "radio"}:
            if node.get("aria-selected") != "true" and node.get("aria-checked") != "true":
                continue
        if node.name == 'button' and not (node.get('aria-selected') == 'true'
                or node.get('aria-pressed') == 'true' or node.get('aria-checked') == 'true'
                or node.get('aria-haspopup') in {'listbox', 'menu', 'true'}):
            continue  # An arbitrary menu option/button isn't a selected filter.
        if any(getattr(a, "attrs", {}).get("role") in {"menu", "listbox"} for a in node.parents):
            if node.get("aria-selected") != "true" and not node.has_attr("selected"):
                continue
        text = text_of(node).replace("’", "'")
        if text and len(text) <= 350:
            label_node = node.parent if node.name == 'option' else node
            label = ' '.join(str(label_node.get(a, '')) for a in ['name', 'id', 'aria-label', 'data-filter'])
            out.append({"tag": node.name, "text": text, 'label': label,
                        'selected': node.name == 'option' or node.get('aria-selected') == 'true'
                        or node.get('aria-pressed') == 'true' or node.get('aria-checked') == 'true'
                        or node.get('role') == 'combobox' or node.get('aria-haspopup') in {'listbox', 'menu', 'true'}})
    return out


def verify_filters(soup, lane, opponent_role_required, region):
    if region not in REGIONS:
        raise EvidenceError('unverified_filters', 'Unsupported requested region')
    contexts = evidence_contexts(soup)
    # Every filter needs affirmative, unambiguous rendered CONTENT evidence.
    # A menu listing all patches/ranks does not qualify as a selected context.
    patterns = {
        "patch": r"\b(?:Patch\s*)?16\.19\b",
        "rank": r"\bMaster\s*\+",
        "region": r"\b(?:Global|Worldwide|World)\b" if region == 'global' else r"\b(?:North America|NA)\b",
        "queue": r"\bRanked\s+Solo(?:\s*/\s*Duo)?\b|\bSolo\s*/\s*Duo\b",
    }
    evidence, missing = {}, []
    # A requested patch in a heading cannot overrule a conflicting selected
    # control. Treat such disagreements as unverified, never silently prefer it.
    conflicts = []
    for context in contexts:
        text, label = context['text'], context['label']
        if not context['selected']:
            continue
        if re.search(r'patch', label, re.I) or re.fullmatch(r'(?:Patch\s*)?\d{1,2}\.\d{1,2}', text, re.I):
            versions = set(re.findall(r'\b\d{1,2}\.\d{1,2}\b', text))
            if versions and versions != {'16.19'}:
                conflicts.append({'filter': 'patch', 'text': text})
        rank_control = re.search(r'tier|rank', label, re.I) or re.fullmatch(
            r'(?:Iron|Bronze|Silver|Gold|Platinum|Emerald|Diamond|Master|Grandmaster|Challenger|All Ranks)(?:\s+[IV1-4]+)?\s*\+?', text, re.I)
        if rank_control and not re.search(patterns['rank'], text, re.I):
            conflicts.append({'filter': 'rank', 'text': text})
        region_control = re.search(r'region', label, re.I) or re.fullmatch(
            r'Global|Worldwide|World|North America|NA|Europe|Korea|EUW|EUNE|KR|BR|OCE|Japan|JP', text, re.I)
        if region_control and not re.search(patterns['region'], text, re.I):
            conflicts.append({'filter': 'region', 'text': text})
        if re.search(r'queue', label, re.I) and not re.search(patterns['queue'], text, re.I):
            conflicts.append({'filter': 'queue', 'text': text})
        if re.search(r'lane|role', label, re.I):
            role_pattern = r'\b(?:'+'|'.join(map(re.escape,ROLE_NAMES[lane]))+r')\b'
            if not re.search(role_pattern,text,re.I):
                conflicts.append({'filter':'lane/opponent_lane','text':text})
    if conflicts:
        raise EvidenceError('unverified_filters', 'Selected controls conflict with requested filters', {'conflicts': conflicts})
    for name, pattern in patterns.items():
        hits = []
        for context in contexts:
            text = context["text"]
            if not re.search(pattern, text, re.I):
                continue
            if name == "patch" and len(set(re.findall(r"\b\d{1,2}\.\d{1,2}\b", text))) != 1:
                continue
            if name == "rank" and re.search(r"Diamond|Emerald|Grandmaster|Challenger|D2\+|All Ranks", text, re.I):
                continue
            if name == 'region':
                other_regions = (r'\b(?:North America|Europe|Korea|NA|EUW|KR)\b' if region == 'global'
                                 else r'\b(?:Global|Worldwide|World|Europe|Korea|EUW|KR)\b')
                if re.search(other_regions, text, re.I):
                    continue
                if not context['selected'] and not (context['tag']=='h1' and
                        (re.search(patterns['rank'],text,re.I) or re.search(patterns['patch'],text,re.I))):
                    continue  # A sidebar/article merely mentioning NA is not dataset-region evidence.
            if name == "queue" and re.search(r"Flex|ARAM|Arena", text, re.I):
                continue
            hits.append(context)
        if not hits:
            missing.append(name)
        else:
            evidence[name] = hits[:4]
    # Verify focal lane from a displayed control/heading, not lane= in URL.
    role_pattern = r"\b(?:" + "|".join(map(re.escape, ROLE_NAMES[lane])) + r")\b"
    role_hits = [c for c in contexts if re.search(role_pattern, c["text"], re.I)
                 and (c['tag'] == 'h1' or c['selected'] or re.fullmatch(role_pattern, c['text'], re.I))]
    if not role_hits:
        missing.append("lane")
    else:
        evidence["lane"] = role_hits[:4]
    # Directional page also needs opposing-role evidence: an explicit vs-role
    # heading or a selected opposing-role control with semantic label.
    if opponent_role_required:
        opposing_hits = []
        for node in soup.select('select, [aria-selected="true"], button, h1, h2, h3'):
            label = " ".join(str(node.get(a, "")) for a in ["name", "id", "aria-label", "data-filter"])
            if node.name == "select":
                selected = node.select('option[selected]')
                value = " ".join(text_of(o) for o in selected)
            else:
                value = text_of(node)
            if len(value) > 350:
                continue
            if (re.search(r"vs.?lane|opponent.*(?:role|lane)|opposing.*(?:role|lane)", label, re.I)
                    and re.search(role_pattern, value, re.I)):
                if any(c["text"] == value for c in contexts) or node.name == "select" and selected:
                    opposing_hits.append(value)
            # A heading must spell out both role occurrences: "Mid ... vs ... Mid".
            if node.name in {"h1", "h2", "h3"} and re.search(role_pattern+r".*?\b(?:vs\.?|versus)\b.*?"+role_pattern, value, re.I):
                opposing_hits.append(value)
        if not opposing_hits:
            missing.append("opponent_lane")
        else:
            evidence["opponent_lane"] = opposing_hits[:4]
    if missing:
        raise EvidenceError("unverified_filters", "Returned content does not confirm: " + ", ".join(missing), evidence)
    return {"verified_patch": "16.19", "verified_rank": "master_plus", "verified_region": region,
            "verified_queue": "ranked", "verified_queue_id": 420, "verified_lane": lane,
            "verified_opponent_lane": lane if opponent_role_required else None,
            "filters_verified": True, "filter_evidence_json": canonical(evidence)}


def parse_document(html):
    soup = BeautifulSoup(html, "html.parser")
    for n in soup(["script", "style", "noscript", "template"]):
        n.decompose()
    visible = text_of(soup).replace("’", "'")
    if re.search(r"verify you are human|checking your browser|just a moment|captcha|access denied", visible, re.I):
        raise EvidenceError("blocked", "Challenge/block page; no bypass attempted")
    if re.search(r"(?:patch|historical data).{0,60}(?:no longer available|not supported|unavailable)", visible, re.I):
        raise EvidenceError("unavailable_patch", "Page reports unavailable historical patch")
    return soup, visible


def parse_tier_average(visible):
    pattern = r"Average\s+Master\s*\+\s+Win\s+Rate\s*:?\s*([\d.]+)\s*%"
    hits = list(re.finditer(pattern, visible, re.I))
    values = {float(m.group(1)) for m in hits}
    if len(values) > 1:
        raise EvidenceError("parsing_error", "Conflicting Average Master+ win rates")
    value = next(iter(values), None)
    if value is not None and not 0 <= value <= 100:
        raise EvidenceError("validation_error", "Invalid Master+ tier average")
    return {"tier_average_winrate_pct": value, "tier_average_missing": value is None,
            "tier_average_source": "Rendered 'Average Master+ Win Rate' label" if value is not None else None,
            "tier_average_context": hits[0].group(0) if hits else None}


def validate_numbers(raw, games, wins=None, losses=None):
    if not math.isfinite(raw) or not 0 <= raw <= 100:
        raise EvidenceError("validation_error", "Matchup rate outside [0,100] percent")
    if not math.isfinite(games) or games <= 0 or int(games) != games:
        raise EvidenceError("validation_error", "Invalid matchup sample size")
    if (wins is None) != (losses is None):
        raise EvidenceError("validation_error", "Only one of explicit wins/losses present")
    if wins is not None and (wins < 0 or losses < 0 or wins+losses != games
                            or abs(100*wins/games-raw) > .11):
        raise EvidenceError("validation_error", "Explicit wins/losses inconsistent with rate/sample")


def finish_rate(raw, games, tier, wins=None, losses=None, delta=None, definition=None):
    validate_numbers(raw, games, wins, losses)
    average = tier["tier_average_winrate_pct"]
    return {"raw_matchup_winrate_pct": raw, "matchup_games": int(games),
            "matchup_wins": wins, "matchup_losses": losses, **tier,
            "derived_tier_centered_rate_pct": raw-average+50 if average is not None else None,
            "derived_rate_definition": "raw_matchup_winrate_pct - tier_average_winrate_pct + 50; derived, not a website probability",
            "website_delta2_pp": delta, "website_delta2_definition": definition,
            "rate_units": "percent (0–100); differences / Delta 2 are percentage points",
            "sample_size_context": "requested directional same-role matchup, not overall champion games",
            "status": "ok", "error": ""}


def parse_individual(html, task):
    soup, visible = parse_document(html)
    verified = verify_filters(soup, task["lane"], opponent_role_required=True, region=task['requested_region'])
    c, o = name_pattern(task["champion_id"]), name_pattern(task["opponent_champion_id"])
    role = '(?:' + '|'.join(map(re.escape, ROLE_NAMES[task['lane']])) + ')'
    # Exact direction must be in a rendered matchup heading AND a directional
    # summary. This avoids picking a different matchup or overall champion WR.
    heading_node = next((h for h in soup.select('h1,h2')
                    if re.search(r"\b"+c+r"(?:\s+"+role+r")?\s+(?:vs\.?|versus)\s+"+o+r"\b", text_of(h).replace("’", "'"), re.I)), None)
    if heading_node is None:
        raise EvidenceError("unverified_matchup", "No rendered heading confirming requested champion direction", verified)
    heading = text_of(heading_node).replace("’", "'")
    directional = re.search(r"\b"+c+r"\s+wins against\s+"+o+r"\s+([\d.]+)%\s+of the time", visible, re.I)
    if directional is None:
        # No-data phrases are accepted only with identity and filters verified.
        # No-data labels elsewhere (items, another matchup) do not establish
        # no-data for this key. Require a compact matchup-specific container.
        scope = heading_node.find_parent(['section', 'article']) or heading_node.parent
        scope_text = text_of(scope).replace("’", "'")
        if len(scope_text) > 2200 or not re.search(r'\bmatchup\b', scope_text, re.I):
            raise EvidenceError('parsing_error', 'No unambiguous matchup-scoped no-data container', verified)
        message = re.search(r"(?:not enough data|no games|0 games|matchup requires a minimum)", scope_text, re.I)
        unsupported = re.search(r"(?:matchup|combination).{0,60}(?:not supported|not available for this lane)", scope_text, re.I)
        if message or unsupported:
            return {**verified, **parse_tier_average(visible), "status": "unsupported_combination" if unsupported else "no_data",
                    "error": (unsupported or message).group(0), "matchup_heading": heading,
                    "matchup_games": 0 if re.search(r"\b(?:0 games|no games)\b", (message or unsupported).group(0), re.I) else None}
        raise EvidenceError("parsing_error", "No directional matchup summary or verified no-data message", verified)
    raw = float(directional.group(1))
    # Bound to the immediate matchup summary; no fallback to an arbitrary game
    # count elsewhere on the page. Require the repeated SAME rate beside Games.
    summary = visible[directional.start():directional.end()+1800]
    boundary = re.search(r"\b(?:Recommended Items|Item Build|Skill Order|Overall Champion|Other Matchups)\b", summary, re.I)
    if boundary:
        summary = summary[:boundary.start()]
    samples = list(re.finditer(r"([\d.]+)%\s+Win\s+Rate\s+([\d,]+)\s+Games\b", summary, re.I))
    matching = [m for m in samples if float(m.group(1)) == raw]
    if len(matching) != 1:
        raise EvidenceError("parsing_error", "Need one matching-rate matchup Games label in directional summary", verified)
    games = int(matching[0].group(2).replace(",", ""))
    # Overall champion sample text invalidates this scope rather than being reused.
    before_sample = summary[:matching[0].end()]
    if re.search(r"\b(?:overall|total champion|all opponents|champion win rate)\b", before_sample, re.I):
        raise EvidenceError("unverified_matchup", "Sample label may belong to overall champion", verified)
    if games == 0:
        return {**verified, **parse_tier_average(visible), 'status':'no_data', 'matchup_games':0,
                'error':'Verified matchup reports zero games; any printed rate is not retained',
                'matchup_heading':heading, 'matchup_evidence':before_sample}
    wl = re.search(r"\b([\d,]+)\s+Wins\s+([\d,]+)\s+Losses\b", before_sample, re.I)
    wins, losses = (int(wl.group(1).replace(",", "")), int(wl.group(2).replace(",", ""))) if wl else (None, None)
    delta_match = re.search(r"After normali[sz]ing both champions win rates.{0,500}?([+-]?[\d.]+)%\s+(more|less|different)", summary, re.I)
    delta, definition = None, None
    if delta_match:
        magnitude = float(delta_match.group(1))
        delta = -abs(magnitude) if delta_match.group(2).lower() == "less" else magnitude
        definition = delta_match.group(0)
    return {**verified, **finish_rate(raw, games, parse_tier_average(visible), wins, losses, delta, definition),
            "matchup_heading": heading, "matchup_evidence": before_sample[:1800], "extraction_strategy": "individual_headline"}


def qwik_dataset(html):
    """Decode only reviewed named Qwik objects, with bounded reference traversal.

    Qwik container values are base-36 references; primitive strings in objs are
    literal values. Re-interpreting those strings as references corrupts data.
    No script execution, recursive metadata search, or guessed numeric columns.
    """
    soup = BeautifulSoup(html, 'html.parser')
    scripts = soup.select('script[type="qwik/json"]')
    if len(scripts) != 1:
        raise EvidenceError('parsing_error', 'Expected one Qwik state container')
    state = json.loads(scripts[0].string or scripts[0].text)
    objects = state['objs']

    def decode(value, depth=0):
        if depth > 16:
            raise EvidenceError('parsing_error', 'Qwik reference depth exceeded')
        def reference(ref):
            if not isinstance(ref, str):
                return ref
            if not re.fullmatch(r'[0-9a-z]+!?', ref):
                raise EvidenceError('parsing_error', 'Unsupported Qwik reference')
            index = int(ref.rstrip('!'), 36)
            if index >= len(objects):
                raise EvidenceError('parsing_error', 'Qwik reference outside object pool')
            return decode(objects[index], depth + 1)
        if isinstance(value, dict):
            return {k: reference(v) for k, v in value.items()}
        if isinstance(value, list):
            return [reference(v) for v in value]
        return value

    datasets = [(i, v) for i, v in enumerate(objects) if isinstance(v, dict)
                and {'header', 'enemy_h', 'enemy', 'response'} <= v.keys()]
    routes = [v for v in objects if isinstance(v, dict)
              and {'path', 'cid', 'champion', 'page', 'tier', 'region', 'status', 'mode'} <= v.keys()]
    if len(datasets) != 1 or len(routes) != 1:
        raise EvidenceError('parsing_error', 'Ambiguous/missing named Qwik dataset or route state')
    index, raw = datasets[0]
    dataset = decode({k: raw[k] for k in ['header', 'enemy_h', 'enemy', 'response']})
    return decode(routes[0]), dataset, index


def parse_qwik_batch(html, tasks):
    first = tasks[0]
    if len({(t['champion_id'], t['lane'], t['requested_region']) for t in tasks}) != 1:
        raise ValueError('Batch tasks must share focal champion, lane and region')
    soup, visible = parse_document(html)
    verified = verify_filters(soup, first['lane'], False, first['requested_region'])
    route, dataset, index = qwik_dataset(html)
    header = dataset['header']
    expected = {'page':'build', 'cid':int(first['champion_id']), 'champion':first['champion_slug'],
                'patch':'16.19', 'tier':'master_plus', 'mode':'ranked', 'status':200,
                'region':'all' if first['requested_region'] == 'global' else 'na'}
    if any(route.get(k) != v for k, v in expected.items()):
        raise EvidenceError('unverified_filters', 'Qwik route conflicts with requested dataset', verified)
    lane = ROLE_SLUGS[first['lane']]
    if (header.get('cid') != int(first['champion_id']) or header.get('lane') != lane
            or header.get('patch') != '16.19' or header.get('queue') != 420
            or route.get('lane') not in {lane, 'all'}
            or dataset['response'].get('valid') is not True):
        raise EvidenceError('unverified_filters', 'Returned dataset header/validity mismatch', verified)
    champion = name_pattern(first['champion_id'])
    if not any(re.search(r'\b'+champion+r'\b', text_of(h).replace('’', "'"), re.I)
               and re.search(r'\b'+re.escape(lane)+r'\b', text_of(h), re.I) for h in soup.select('h1')):
        raise EvidenceError('unverified_matchup', 'Focal champion/lane heading missing', verified)
    # enemy_h is supplied alongside enemy by the server. Source renderer passes
    # enemy to type="counter" and indexes by opposing lane (diagnostic assets).
    columns = dataset['enemy_h']
    if columns != ['id', 'wr', 'd1', 'd2', 'pr', 'n']:
        raise EvidenceError('parsing_error', 'Unreviewed matchup column schema', verified)
    if ('Counter Win Rate' not in visible or 'Pick Rate Games' not in visible
            or 'champion wins against the matchup' not in visible):
        raise EvidenceError('unverified_matchup', 'Rendered counter column semantics missing', verified)
    enemy = dataset['enemy']
    if set(enemy) != set(ROLE_SLUGS.values()):
        raise EvidenceError('parsing_error', 'Incomplete/unreviewed opposing-lane table schema', verified)
    rows = {}
    for values in enemy[lane]:
        if not isinstance(values, list) or len(values) != len(columns):
            raise EvidenceError('parsing_error', 'Matchup row length mismatch', verified)
        row = dict(zip(columns, values))
        if (not isinstance(row['id'], int) or row['id'] in rows
                or any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) for k in columns)):
            raise EvidenceError('parsing_error', 'Duplicate/non-numeric matchup row', verified)
        validate_numbers(row['wr'], row['n'])
        rows[row['id']] = row
    evidence = {**json.loads(verified['filter_evidence_json']),
                'dataset_header':header, 'dataset_object_index':index, 'columns':columns,
                'opposing_lane':lane, 'region_route_crosscheck':route['region'],
                'perspective':'focal champion counter win rate; enemy table, not synergy or header.wr'}
    verified.update(verified_opponent_lane=first['lane'], filter_evidence_json=canonical(evidence))
    tier = parse_tier_average(visible)
    results = {}
    for task in tasks:
        row = rows.get(int(task['opponent_champion_id']))
        if row is None:
            results[task['key_id']] = {**verified, **tier, 'status':'not_published_in_table',
                'error':'Opponent absent from returned same-role enemy table; sample and rate unknown',
                'extraction_strategy':'qwik_enemy_table_v1', 'matchup_evidence':canonical(evidence)}
            continue
        results[task['key_id']] = {**verified,
            **finish_rate(row['wr'], row['n'], tier, delta=row['d2'],
                          definition='Delta 2 = Counter Matchup Win Rate - Normalised Champions Win Rate in Matchup'),
            'extraction_strategy':'qwik_enemy_table_v1',
            'matchup_evidence':canonical({'dataset_object_index':index, 'columns':columns,
                                        'opposing_lane':lane, 'row':row})}
    return results, [f'Named Qwik enemy[{lane}] table: {len(rows)} rows; only requested opponents extracted.']


def parse_batch_table(html, tasks):
    """Extract ONLY requested opponents, never scrape an entire global key grid.

    Prefer the reviewed named Qwik dataset. A legacy semantic HTML table needs
    champion/lane content verified above;
    table caption explicitly says 'same lane/role' OR data-lane/data-vslane both
    match; named Opponent / Win Rate / Games columns. Unlabeled JS arrays are
    inspected as raw cached source but never assigned guessed field meanings.
    Unsupported representations pause by default; directional fallback is opt-in.
    """
    if 'type="qwik/json"' in html:
        return parse_qwik_batch(html, tasks)
    first = tasks[0]
    soup, visible = parse_document(html)
    if len({t['requested_region'] for t in tasks}) != 1:
        raise ValueError('Batch groups cannot mix regions')
    verified = verify_filters(soup, first["lane"], opponent_role_required=False, region=first['requested_region'])
    champion = name_pattern(first["champion_id"])
    if not any(re.search(r"\b"+champion+r"\b", text_of(h).replace("’", "'"), re.I) for h in soup.select('h1,h2')):
        raise EvidenceError("unverified_matchup", "Batch page lacks focal champion heading", verified)
    results, notes = {}, []
    by_slug = {t["opponent_slug"]: t for t in tasks}
    tier = parse_tier_average(visible)
    for table in soup.select('table'):
        caption = " ".join(text_of(n) for n in table.select('caption'))
        same_role = (bool(re.search(r"same\s+(?:lane|role)", caption, re.I)) or
                     table.get('data-lane') == table.get('data-vslane') == ROLE_SLUGS[first['lane']])
        if not same_role:
            continue
        # An Opponent/Win Rate table may show the opponent's win rate. Require
        # explicit focal-champion perspective instead of assuming direction.
        focal_perspective = (table.get('data-rate-perspective') == 'focal' or bool(re.search(
            r'(?:win rates?\s+(?:for|of)\s+'+champion+r'|'+champion+r"(?:'s)?\s+win rates?)", caption, re.I)))
        if not focal_perspective:
            continue
        headers = [text_of(n).lower() for n in table.select('thead th')]
        # Explicit header names, not numeric offsets inferred from minified JS.
        choices = {'opponent': ['opponent', 'enemy champion'], 'rate': ['win rate', 'matchup win rate'],
                   'games': ['games', 'matchup games']}
        indices = {}
        for field, aliases in choices.items():
            hits = [i for i, h in enumerate(headers) if h in aliases]
            if len(hits) == 1:
                indices[field] = hits[0]
        if len(indices) != 3:
            continue
        for row in table.select('tbody tr'):
            cells = row.find_all(['td', 'th'], recursive=False)
            if len(cells) != len(headers):
                continue
            opponent = cells[indices['opponent']]
            slugs = set()
            for link in opponent.select('a[href]'):
                m = re.fullmatch(r'/lol/([a-z0-9]+)/build/?', urlparse(link['href']).path)
                if m:
                    slugs.add(m.group(1))
            if len(slugs) != 1:
                continue
            slug = next(iter(slugs))
            if slug not in by_slug:
                continue
            task = by_slug[slug]
            if not re.fullmatch(name_pattern(task['opponent_champion_id']), text_of(opponent).replace("’", "'"), re.I):
                continue
            rate_text, games_text = text_of(cells[indices['rate']]), text_of(cells[indices['games']])
            if not re.fullmatch(r'[\d.]+%', rate_text) or not re.fullmatch(r'[\d,]+', games_text):
                continue
            raw, games = float(rate_text[:-1]), int(games_text.replace(',', ''))
            if games == 0:
                continue  # A printed 0% / 0 is not usable; confirm no-data individually.
            if task['key_id'] in results:
                raise EvidenceError('parsing_error', 'Duplicate requested opponent rows in batch table', verified)
            wins = losses = None
            if 'wins' in headers and 'losses' in headers:
                win_text, loss_text = text_of(cells[headers.index('wins')]), text_of(cells[headers.index('losses')])
                if not re.fullmatch(r'[\d,]+', win_text) or not re.fullmatch(r'[\d,]+', loss_text):
                    raise EvidenceError('validation_error', 'Non-numeric explicit table wins/losses', verified)
                wins, losses = int(win_text.replace(',', '')), int(loss_text.replace(',', ''))
            results[task['key_id']] = {**verified, 'verified_opponent_lane': task['lane'],
                **finish_rate(raw, games, tier, wins, losses), 'extraction_strategy': 'verified_same_role_table',
                'matchup_evidence': canonical({'caption': caption, 'headers': headers, 'row': text_of(row)}),
                'filter_evidence_json': canonical({'page': json.loads(verified['filter_evidence_json']),
                    'table_same_role': caption or {'data-lane': table.get('data-lane'), 'data-vslane': table.get('data-vslane')}})}
    notes.append(f'{len(soup.select("table"))} HTML tables inspected; {len(results)} requested keys accepted.')
    notes.append('Embedded scripts cached; no unverified JSON column/schema assumptions used.')
    return results, notes


def representation_inventory(html):
    """Record what the runtime source actually exposes, without guessed indices.

    Named JSON structures are inventoried for diagnosis; without an established
    rate scale, role/direction and dataset-filter contract they are not accepted
    as matchup data. Individual pages remain the conservative fallback.
    """
    soup = BeautifulSoup(html, 'html.parser')
    scripts = []
    for node in soup.select('script[type="application/json"], script#__NEXT_DATA__'):
        record = {'id': node.get('id'), 'type': node.get('type')}
        try:
            payload = json.loads(node.string or node.get_text())
            record['root_type'] = type(payload).__name__
            if isinstance(payload, dict):
                record['root_keys'] = sorted(payload)[:40]
                props = payload.get('props')
                if isinstance(props, dict) and isinstance(props.get('pageProps'), dict):
                    record['pageProps_keys'] = sorted(props['pageProps'])[:60]
        except (ValueError, TypeError):
            record['error'] = 'Not valid JSON; no guessed JavaScript evaluation'
        scripts.append(record)
    return {'html_tables': len(soup.select('table')), 'json_scripts': scripts,
            'policy': 'Accept only verified semantic HTML tables; embedded arrays with unverified semantics require a reviewed adapter, otherwise use individual pages.'}


class PauseCollection(RuntimeError):
    pass


class PageCache:
    """Cache by fully specified URL; cache each HTTP attempt before parsing.

    Only relevant nonsecret response headers are saved. No browser or challenge
    bypass is implemented. Valid 200 pages are reused even if parsing fails.
    """
    def __init__(self, run_dir, config):
        self.run_dir, self.config = Path(run_dir), config
        self.cache = self.run_dir/'responses'
        self.cache.mkdir(exist_ok=True)
        self.log = self.run_dir/'request_events.jsonl'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LongitudinalMatchupResearch/1.0 (sequential cached collection)',
                                     'Accept-Language': 'en-US,en;q=0.9'})
        self.last_request = 0.0
        self.cache_hits = 0
        self.paused = False
        self.last_receipt = None
        self.actual_attempts = 0
        self.attempts_by_region = Counter()

    def _wait(self, seconds):
        print(f'  Waiting {seconds:.1f}s (pacing / bounded retry; interrupt safely if needed).', flush=True)
        time.sleep(seconds)

    def get(self, url, *, retry_cached_error=False, refresh_unresolved=False):
        page_id = digest(url.encode())
        index_path = self.cache/f'{page_id}.json'
        if index_path.exists():
            receipt = json.loads(index_path.read_text())
            self.last_receipt = receipt
            if receipt['requested_url'] != url:
                raise ValueError('Page cache URL hash mismatch')
            if receipt.get('requested_region', url_region(url)) != url_region(url):
                raise ValueError('Page cache region mismatch')
            retry_error = refresh_unresolved or (retry_cached_error and receipt.get('http_status') in {404, 410})
            if retry_error:
                # Scoped deliberate retry: preserve the old index as well as
                # its immutable body and original request event.
                archive = self.cache/f'{page_id}-receipt-{time.time_ns()}.json'
                atomic_json(archive, receipt)
                append_event(self.log, {'event':'scoped_cached_error_retry', 'requested_url':url,
                                       'old_receipt_file':str(archive.relative_to(self.run_dir)),
                                       'at_utc':now_utc()})
            if not retry_error and receipt.get('http_status') in {200, 404, 410} and receipt.get('body_file'):
                body = self.run_dir/receipt['body_file']
                if not body.exists() or digest(body.read_bytes()) != receipt['body_sha256']:
                    raise ValueError('Cached body missing/changed; preserve artifacts and use a new run')
                self.cache_hits += 1
                self.last_receipt = receipt
                return body.read_bytes().decode(receipt.get('encoding') or 'utf-8', errors='replace'), receipt
            not_before = receipt.get('retry_not_before_epoch')
            if not_before and not_before > time.time():
                remaining = not_before - time.time()
                if remaining > self.config.max_retry_after_seconds:
                    raise PauseCollection(f'Previous Retry-After still active for {remaining:.0f}s; resume later')
                self._wait(remaining + 1)
        for attempt in range(1, self.config.max_attempts+1):
            elapsed = time.monotonic()-self.last_request
            if elapsed < self.config.delay_seconds:
                time.sleep(self.config.delay_seconds-elapsed)
            start = time.monotonic()
            stamp = now_utc()
            receipt = {'requested_url': url, 'requested_at_utc': stamp, 'collected_at_utc': None, 'attempt_in_session': attempt,
                       'requested_region': url_region(url),
                       'page_id': page_id, 'http_status': None, 'body_file': None,
                       'attempt_id': f'{page_id}-{time.time_ns()}', 'event': 'attempt_finished'}
            response = None
            append_event(self.log, {'event':'attempt_started','attempt_id':receipt['attempt_id'],
                                   'requested_url':url,'requested_region':url_region(url),'at_utc':stamp})
            self.actual_attempts += 1
            self.attempts_by_region[url_region(url)] += 1
            try:
                # Do not follow redirects to current-patch/default/challenge pages.
                # Record them for diagnosis; accepting a redirected page is unsafe.
                response = self.session.get(url, timeout=(self.config.timeout_connect, self.config.timeout_read),
                                            allow_redirects=False)
                self.last_request = time.monotonic()
                body = response.content
                body_path = self.cache/f'{page_id}-{time.time_ns()}.html'
                body_path.write_bytes(body)
                receipt.update(http_status=response.status_code, response_url=response.url,
                    collected_at_utc=now_utc(),
                    encoding=response.encoding or 'utf-8', body_file=str(body_path.relative_to(self.run_dir)),
                    body_sha256=digest(body),
                    request_headers={k:v for k,v in response.request.headers.items()
                                     if k.lower() not in {'cookie', 'authorization', 'proxy-authorization'}},
                    headers={k:v for k,v in response.headers.items()
                             if k.lower() not in {'set-cookie', 'authorization', 'proxy-authorization'}})
            except requests.RequestException as exc:
                self.last_request = time.monotonic()
                receipt['request_error'] = f'{type(exc).__name__}: {exc}'
            receipt['request_seconds'] = round(time.monotonic()-start, 3)
            retry_after = receipt.get('headers', {}).get('Retry-After')
            retry_seconds = None
            if retry_after:
                try:
                    retry_seconds = float(retry_after)
                except ValueError:
                    try:
                        retry_seconds = (parsedate_to_datetime(retry_after)-datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        retry_seconds = 60.0
                if not math.isfinite(retry_seconds):
                    retry_seconds = 60.0
                receipt['retry_not_before_epoch'] = time.time()+max(0,retry_seconds)
            # Every actual attempt is durable, including failed/block responses.
            append_event(self.log, receipt)
            atomic_json(index_path, receipt)
            self.last_receipt = receipt
            status = receipt['http_status']
            if status in {200, 404, 410}:
                return body.decode(receipt['encoding'], errors='replace'), receipt
            if status in {401, 403}:
                self.paused = True
                raise PauseCollection(f'HTTP {status}: persistent-access/challenge risk; stopped without bypass')
            if status is not None and 300 <= status < 400:
                raise EvidenceError('request_failure', f'HTTP redirect {status} refused: {receipt.get("headers",{}).get("Location")}')
            retryable = status is None or status in {408, 429, 500, 502, 503, 504}
            if not retryable:
                raise EvidenceError('request_failure', f'HTTP {status}')
            if attempt == self.config.max_attempts:
                if status == 429:
                    self.paused = True
                    raise PauseCollection('Persistent HTTP 429 after bounded retries; resume later')
                raise EvidenceError('request_failure', f'Exhausted {attempt} attempts; last HTTP {status}')
            delay = self.config.backoff_seconds * (2**(attempt-1))
            if retry_seconds is not None:
                seconds = retry_seconds
                if seconds > self.config.max_retry_after_seconds:
                    self.paused = True
                    raise PauseCollection(f'Retry-After {seconds:.0f}s exceeds wait budget; pause and resume later')
                delay = max(delay, seconds+1)
            self._wait(max(0, delay))
        raise AssertionError('Unreachable request loop')

    def close(self):
        self.session.close()


def _write_frame(frame, path):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    if path.suffix == '.parquet':
        frame.to_parquet(temp, index=False)
    else:
        frame.to_csv(temp, index=False)
    temp.replace(path)


def prepare_collection(root=None, config=None, run_id=None, fresh=False, pinned_snapshot=None):
    """Plan both regions, pin inputs, and safely extend a matching HLR v1 run.

    The legacy manifest/plan/mapping are backed up BEFORE any runtime upgrade.
    Legacy checkpoints and cached Global URLs remain in place. Only matching,
    content-verified records with intact response provenance can be reused.
    This function makes no website requests; collect() starts collection.
    """
    root = find_project_root(root)
    config = config or Config()
    base = root/'hlr_model/data/lolalytics'
    active = base/'active_run.json'
    if fresh and run_id is not None:
        raise ValueError('Use fresh=True OR an explicit run_id, not both')
    if run_id is None and active.exists() and not fresh:
        run_id = json.loads(active.read_text())['run_id']
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    run_dir = base/'runs'/run_id
    if (run_dir/'collector.lock').exists():
        raise RuntimeError('Collector lock exists; stop the other collector or verify a stale lock before resuming.')
    manifest_path = run_dir/'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and pinned_snapshot and previous['snapshot'] != pinned_snapshot:
        raise ValueError('Requested snapshot differs from the pinned run')
    snapshot, model, history, targets, hashes = load_snapshot(root, previous['snapshot'] if previous else pinned_snapshot)
    base_plan, base_mapping, decisions = _build_base_plan(model, history, targets, config)
    plan, mapping = regionalize_plan(base_plan, base_mapping, config)
    common_identity = {'snapshot':snapshot.name, 'input_sha256':hashes,
        'candidate_roles':list(config.candidate_roles),
        'include_current_pick_in_candidates':config.include_current_pick_in_candidates,
        'champion_registry_sha256':digest(canonical(CHAMPIONS).encode())}
    identity = {**common_identity, 'schema_version':SCHEMA_VERSION,
        'plan_sha256':digest(plan.to_json(orient='records').encode()),
        'mapping_sha256':digest(mapping.to_json(orient='records').encode()),
        'filters':{k:v for k,v in FILTERS.items() if k!='region'}, 'regions':list(config.regions)}
    legacy_identity = {**common_identity, 'schema_version':'hlr-master-matchups-v1',
        'plan_sha256':digest(base_plan.to_json(orient='records').encode()),
        'mapping_sha256':digest(base_mapping.to_json(orient='records').encode()), 'filters':FILTERS}
    migrate = bool(previous and previous['identity'].get('schema_version') == 'hlr-master-matchups-v1')
    if previous and previous['identity'] != (legacy_identity if migrate else identity):
        raise ValueError('Pinned input/provenance/candidate definitions differ. Use original settings or a NEW run; no Global/D2+ substitution.')
    run_dir.mkdir(parents=True, exist_ok=True)
    if migrate:
        backup = run_dir/'migration_v1'
        backup.mkdir(exist_ok=True)
        for name in ['manifest.json', 'request_plan.parquet', 'request_plan.csv',
                     'decision_request_mapping.parquet', 'decision_request_mapping.csv',
                     'target_decisions.parquet', 'target_decisions.csv', 'strategy.json']:
            source, dest = run_dir/name, backup/name
            if source.exists() and not dest.exists():
                temp = dest.with_name(dest.name+'.tmp')
                with temp.open('wb') as handle:
                    handle.write(source.read_bytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                temp.replace(dest)
        if digest((backup/'manifest.json').read_bytes()) != digest(manifest_path.read_bytes()):
            raise ValueError('Existing migration backup differs from the original manifest; inspect before resuming.')
        print('Extending a matching HLR Global run; original manifest/plans preserved in migration_v1/.')
    manifest = dict(previous) if previous else {
        'run_id':run_id, 'created_at_utc':now_utc(), 'snapshot':snapshot.name,
        'initial_config':asdict(config),
        'schemas':{n:{c:str(t) for c,t in f.dtypes.items()} for n,f in
                   [('modeling',model),('history',history),('targets',targets)]},
        'candidate_definition':'Preserved previously played same-role champions, end < kickoff; exclude candidate equal to opponent; no experience cutoff; observed mirrors retained separately; draft availability unverified',
        'rate_units':'Percent; NA minus Global and Delta 2 in percentage points',
        'website_assumptions':'Reviewed 2026-10-20: omit queue parameter (queue=ranked returns 404); Global omits region; NA uses region=na. Explicit lane. Verify displayed filters and dataset header.queue=420; named Qwik enemy_h/enemy tables. Directional endpoint discrepancy preserved separately; fallback disabled by default.'}
    manifest.update(identity=identity, collection_regions=list(config.regions),
        primary_benchmark='global', sensitivity_benchmark='na',
        interpretation='Retrospectively collected patch-16.19 overlapping benchmarks; Global includes NA. Not necessarily available before target kickoff.')
    if migrate:
        manifest['legacy_global_reuse'] = {'enabled':True, 'original_identity':legacy_identity,
            'backup_dir':'migration_v1', 'upgraded_at_utc':now_utc(),
            'original_manifest_sha256':digest((run_dir/'migration_v1/manifest.json').read_bytes())}
    # Commit the upgraded manifest LAST so an interrupted migration can repeat
    # from its still-valid v1 identity. Never rewrite the checkpoint or responses.
    for name, frame in [('request_plan',plan),('decision_request_mapping',mapping),('target_decisions',decisions)]:
        for ext in ['parquet','csv']:
            path = run_dir/f'{name}.{ext}'
            if previous is None or migrate or not path.exists():
                _write_frame(frame,path)
    mirrors = decisions.loc[decisions.get('unexpected_observed_mirror', pd.Series(False, index=decisions.index)).eq(True)]
    if previous is None:
        mirrors.to_csv(run_dir/'unexpected_observed_mirrors.csv', index=False)
    atomic_json(manifest_path,manifest)
    atomic_json(active, {'run_id':run_id})
    context = {'root':root, 'run_dir':run_dir, 'run_id':run_id, 'config':config,
               'plan':plan, 'mapping':mapping, 'decisions':decisions, 'input_snapshot':snapshot.name,
               'manifest':manifest}
    print_plan(context)
    return context


def print_plan(context):
    plan, config = context['plan'], context['config']
    print('Run:',context['run_id'],'| artifacts:',context['run_dir'])
    print('Configuration:',canonical(asdict(config)))
    maximum = 0
    for region in config.regions:
        p = plan.loc[plan.region.eq(region)]
        fetchable = p.loc[p.mapping_verified_locally]
        directional = fetchable.page_url.nunique()
        batch = fetchable.batch_url.nunique() if config.inspect_batch_tables else 0
        maximum += directional+batch
        print(f'{region}: observed={p.observed.sum():,}, alternatives={p.alternative.sum():,}, '
              f'both purposes={(p.observed & p.alternative).sum():,}, region keys={len(p):,}; '
              f'unique directional URLs={directional:,}, candidate batch URLs={batch:,}, '
              f'unique-page upper bound={directional+batch:,}; unknown mappings={(~p.mapping_verified_locally).sum():,}')
    print(f'Unique full regional keys: {len(plan):,}; base matchup combinations: {plan.matchup_key_id.nunique():,}')
    table_pages = plan.loc[plan.mapping_verified_locally, 'batch_url'].nunique()
    print(f'Champion/lane table strategy: {table_pages:,} requests, ~{table_pages*(config.delay_seconds+2)/3600:.2f} h '
          '(assumed 2s response + pacing; representation failures pause).')
    print(f'Directional fallback enabled: {config.directional_fallback}')
    print(f'FALLBACK UPPER SCENARIO ONLY, before cache savings: ~{maximum*(config.delay_seconds+2)/3600:.2f} h '
          '(pacing + assumed 2s/page; retry/block delays excluded).')
    ceiling_pages = maximum if config.directional_fallback else table_pages
    print(f'Bounded-retry attempt ceiling for selected strategy: {ceiling_pages*config.max_attempts:,}; verified completed regional pages are reused.')
    print('Global is primary; NA is sensitivity. Global includes NA: these are overlapping benchmarks.')


def validate_checkpoint_provenance(row, task, context):
    """Never promote old D2+, wrong-region or untraceable data to valid rates."""
    for field in ['patch','rank','region','queue']:
        if row.get('requested_'+field) != task['requested_'+field]:
            raise ValueError(f'Checkpoint requested {field} mismatch')
        if field in row and row[field] != task[field]:
            raise ValueError(f'Checkpoint output {field} mismatch')
    for field in ['champion_id','opponent_champion_id','champion','opponent_champion','lane',
                  'champion_slug','opponent_slug','page_url','batch_url']:
        if row.get(field) != task[field]:
            raise ValueError(f'Checkpoint matchup field mismatch: {field}')
    if row.get('snapshot_id') != context['run_id'] or row.get('input_snapshot') != context['input_snapshot']:
        raise ValueError('Checkpoint snapshot/run mismatch')
    if row.get('status') == 'unsupported_mapping':
        if task['mapping_verified_locally']:
            raise ValueError('Unsupported mapping is inconsistent with current registry')
        return
    if row.get('status') == PAGE_UNRESOLVED:
        if row.get('filters_verified') or any(row.get(f) is not None for f in RATE_FIELDS):
            raise ValueError('An unresolved page cannot establish filters, rates or sample counts')
        if row.get('source_url') != task['batch_url'] or url_region(row['source_url']) != task['region']:
            raise ValueError('Unresolved page source/region mismatch')
        if row.get('response_body_file'):
            root = context['run_dir'].resolve()
            body = (root/row['response_body_file']).resolve()
            if not body.is_relative_to(root) or not body.is_file() or digest(body.read_bytes()) != row.get('response_sha256'):
                raise ValueError('Unresolved cached body missing or changed')
        return
    if row.get('status') not in {'ok','no_data','unsupported_combination','not_published_in_table'}:
        return
    if not row.get('filters_verified'):
        raise ValueError('Terminal website result has unverified filters')
    for field in ['patch','rank','region','queue']:
        if row.get('verified_'+field) != task['requested_'+field]:
            raise ValueError(f'Checkpoint verified {field} mismatch')
    if any(row.get('verified_'+field) != task['lane'] for field in ['lane','opponent_lane']):
        raise ValueError('Checkpoint role evidence mismatch')
    if not row.get('filter_evidence_json') or not row.get('collected_at_utc'):
        raise ValueError('Missing original filter evidence or collection timestamp')
    url = row.get('source_url')
    if url not in {task['page_url'],task['batch_url']} or url_region(url) != task['region']:
        raise ValueError('Checkpoint source URL/region mismatch')
    run_dir = context['run_dir'].resolve()
    if not row.get('response_body_file') or not row.get('response_sha256'):
        raise ValueError('No cached response provenance')
    body = (run_dir/row['response_body_file']).resolve()
    if not body.is_relative_to(run_dir) or not body.is_file():
        raise ValueError('Missing/out-of-run cached response')
    hashes = context.setdefault('_validated_response_hashes',{})
    if body not in hashes:
        hashes[body] = digest(body.read_bytes())
    if hashes[body] != row['response_sha256']:
        raise ValueError('Cached response hash mismatch')
    receipt_path = run_dir/'responses'/f'{digest(url.encode())}.json'
    if not receipt_path.is_file():
        raise ValueError('Missing URL-specific response receipt')
    receipts = context.setdefault('_validated_response_receipts',{})
    if receipt_path not in receipts:
        receipts[receipt_path] = json.loads(receipt_path.read_text())
    receipt = receipts[receipt_path]
    if (receipt.get('requested_url') != url or receipt.get('http_status') != 200
            or receipt.get('body_sha256') != row['response_sha256']
            or receipt.get('collected_at_utc') != row['collected_at_utc']
            or receipt.get('requested_region',url_region(url)) != task['region']):
        raise ValueError('Response receipt does not match checkpoint provenance')
    if row['status'] == 'ok':
        raw, games = row.get('raw_matchup_winrate_pct'), row.get('matchup_games')
        if raw is None or games is None:
            raise ValueError('Missing rate/sample in completed checkpoint')
        validate_numbers(raw,games,row.get('matchup_wins'),row.get('matchup_losses'))
        average = row.get('tier_average_winrate_pct')
        centered = row.get('derived_tier_centered_rate_pct')
        if (average is None and centered is not None) or (average is not None and
                (not 0 <= average <= 100 or centered is None or not math.isclose(centered,raw-average+50))):
            raise ValueError('Inconsistent region-specific tier-centered rate')
        if bool(row.get('tier_average_missing')) != (average is None):
            raise ValueError('Inconsistent tier-average missingness flag')


def load_results(context):
    run_dir = context['run_dir']
    tasks = {t['key_id']:t for t in context['plan'].to_dict('records')}
    global_by_base = {t['matchup_key_id']:t for t in tasks.values() if t['region']=='global'}
    latest, rejected = {}, []
    legacy_allowed = context['manifest'].get('legacy_global_reuse',{}).get('enabled',False)
    for row in read_events(run_dir/'matchup_checkpoint.jsonl'):
        old_key = row['key_id']
        legacy = old_key not in tasks
        task = global_by_base.get(old_key) if legacy and legacy_allowed else tasks.get(old_key)
        if task is None:
            raise ValueError(f'Checkpoint key outside pinned regional plan: {old_key}')
        try:
            validate_checkpoint_provenance(row,task,context)
            merged = {**row,**task}  # New identity/flags, ORIGINAL statistics and timestamps.
            if legacy:
                merged['reused_legacy_global_key'] = old_key
            latest[task['key_id']] = merged
        except (ValueError,TypeError,KeyError,OSError) as exc:
            rejected.append({'key_id':old_key,'region':task['region'],'reason':str(exc)})
            latest[task['key_id']] = result_record(task,{'status':'provenance_error',
                'error':str(exc)},context)
    if rejected:
        _write_frame(pd.DataFrame(rejected),run_dir/'checkpoint_provenance_issues.csv')
        print(f'{len(rejected)} checkpoint records lack matching provenance; reparse cached source, never substitute rates.')
    return latest


def result_record(task, parsed, context, receipt=None):
    record = {**task, **{f: None for f in RATE_FIELDS}, 'filters_verified': False,
              'verified_patch': None, 'verified_rank': None, 'verified_region': None,
              'verified_queue': None, 'verified_queue_id': None, 'verified_lane': None, 'verified_opponent_lane': None,
              'filter_evidence_json': None, 'website_delta2_definition': None,
              'rate_units': 'percent (0–100); differences / Delta 2 are percentage points',
              'derived_rate_definition': 'raw_matchup_winrate_pct - tier_average_winrate_pct + 50; derived, not a website probability',
              'sample_size_context': 'requested directional same-role matchup, not overall champion games',
              'tier_average_missing': True, 'tier_average_source': None, 'tier_average_context': None,
              'snapshot_id': context['run_id'], 'input_snapshot': context['input_snapshot'],
              'source_url': receipt['requested_url'] if receipt else task.get('page_url'),
              'collected_at_utc': receipt['collected_at_utc'] if receipt else None,
              'response_body_file': receipt.get('body_file') if receipt else None,
              'response_sha256': receipt.get('body_sha256') if receipt else None,
              'http_status': receipt.get('http_status') if receipt else None,
              'request_attempt_id': receipt.get('attempt_id') if receipt else None,
              'parsed_at_utc': now_utc(), **parsed}
    if record['status'] == 'ok':
        if not record['filters_verified']:
            raise ValueError('Cannot accept a rate with unverified filters')
        for requested, verified in [('patch','patch'),('rank','rank'),('region','region'),('queue','queue')]:
            if record.get('verified_'+verified) != task['requested_'+requested]:
                raise ValueError('Verified filter mismatch')
        if record['verified_lane'] != task['lane'] or record['verified_opponent_lane'] != task['lane']:
            raise ValueError('Verified focal/opponent role mismatch')
    return record


def _coverage(frame, ok_keys):
    if frame.empty:
        return {'total': 0, 'covered': 0, 'coverage': None}
    covered = frame.key_id.isin(ok_keys)
    return {'total': len(frame), 'covered': int(covered.sum()), 'coverage': float(covered.mean())}


def valid_rate_mask(frame):
    valid = frame.status.eq('ok') & frame.filters_verified.eq(True)
    valid &= frame.raw_matchup_winrate_pct.between(0,100) & frame.matchup_games.gt(0)
    valid &= frame.matchup_games.mod(1).eq(0)
    for field in ['patch','rank','region','queue']:
        valid &= frame['verified_'+field].eq(frame['requested_'+field])
    valid &= frame.verified_lane.eq(frame.lane) & frame.verified_opponent_lane.eq(frame.lane)
    return valid.fillna(False)


def paired_tables(context, lookup):
    """One pair per full nonregional key. Differences are percentage POINTS."""
    base_cols = ['matchup_key_id', *PAIR_KEYS, 'champion_id', 'opponent_champion_id',
                 'champion_slug', 'opponent_slug', 'observed', 'alternative']
    paired = lookup[base_cols].drop_duplicates()
    if paired.duplicated(PAIR_KEYS).any() or paired.matchup_key_id.duplicated().any():
        raise ValueError('Ambiguous nonregional pairing keys')
    source_fields = {'raw_matchup_winrate_pct':'raw_winrate', 'matchup_games':'matchup_games',
        'tier_average_winrate_pct':'tier_average_winrate', 'tier_average_missing':'tier_average_missing',
        'derived_tier_centered_rate_pct':'derived_tier_centered_rate', 'website_delta2_pp':'website_delta2_pp',
        'website_delta2_definition':'website_delta2_definition', 'status':'status', 'error':'error',
        'filters_verified':'filters_verified', 'source_url':'source_url',
        'collected_at_utc':'collected_at_utc', 'snapshot_id':'snapshot_id',
        'filter_evidence_json':'filter_evidence_json'}
    for region in REGIONS:
        regional = lookup.loc[lookup.region.eq(region)].copy()
        regional['valid'] = valid_rate_mask(regional)
        columns = [*PAIR_KEYS,*source_fields,'valid']
        regional = regional[columns].rename(columns={k:f'{region}_{v}' for k,v in source_fields.items()}
                                              | {'valid':f'{region}_valid'})
        paired = paired.merge(regional,on=PAIR_KEYS,how='left',validate='one_to_one')
    paired['both_regions_valid'] = paired.global_valid.eq(True) & paired.na_valid.eq(True)
    # Raw rates are already percentages. Do NOT multiply this difference by 100.
    paired['na_minus_global_pp'] = (paired.na_raw_winrate-paired.global_raw_winrate).where(paired.both_regions_valid)
    paired['smaller_regional_sample'] = paired[['global_matchup_games','na_matchup_games']].min(axis=1).where(paired.both_regions_valid)
    observed = context['mapping'].loc[context['mapping'].request_kind.eq('observed')
                                      & context['mapping'].region.eq('global'),
                                      ['puuid','match_id','matchup_key_id']].drop_duplicates()
    target_pairs = context['decisions'].merge(observed,on=['puuid','match_id'],how='left',validate='one_to_one')
    values = ['matchup_key_id','global_raw_winrate','global_matchup_games','na_raw_winrate','na_matchup_games',
              'global_valid','na_valid','both_regions_valid','na_minus_global_pp','smaller_regional_sample']
    target_pairs = target_pairs.merge(paired[values],on='matchup_key_id',how='left',validate='many_to_one')
    for flag in ['global_valid','na_valid','both_regions_valid']:
        target_pairs[flag] = target_pairs[flag].eq(True)
    return paired, target_pairs


def network_summary(context, events):
    started = [e for e in events if e.get('event')=='attempt_started']
    finished = [e for e in events if e.get('event')=='attempt_finished']
    rows = []
    for region in REGIONS:
        plan = context['plan'].loc[context['plan'].region.eq(region) & context['plan'].mapping_verified_locally]
        begins = [e for e in started if url_region(e.get('requested_url',''))==region]
        ends = [e for e in finished if url_region(e.get('requested_url',''))==region]
        directional = plan.page_url.nunique()
        batch = plan.batch_url.nunique() if context['config'].inspect_batch_tables else 0
        rows.append({'region':region, 'planned_unique_directional_pages':directional,
            'possible_unique_batch_pages':batch, 'planned_unique_page_upper_bound':directional+batch,
            'actual_unique_network_urls':len({e['requested_url'] for e in begins}),
            'actual_http_attempts':len(begins),
            'unfinished_http_attempts':len({e['attempt_id'] for e in begins}-{e['attempt_id'] for e in ends}),
            'actual_request_seconds':sum(e.get('request_seconds',0) for e in ends)})
    return pd.DataFrame(rows)


def record_page_failure(context, latest, group, error, receipt=None,
                        status=PAGE_UNRESOLVED, failure_kind='request_failure'):
    """Append a durable outcome for EVERY unresolved key without touching successes."""
    affected = []
    for task in group:
        if latest.get(task['key_id'], {}).get('status') in TERMINAL:
            continue
        row = result_record(task, {'status':status, 'error':str(error),
                                   'page_failure_kind':failure_kind,
                                   'diagnostic_evidence_json':canonical(error.evidence)
                                   if isinstance(error, EvidenceError) else None}, context, receipt)
        row['source_url'] = task['batch_url']
        append_event(context['run_dir']/'matchup_checkpoint.jsonl', row)
        latest[task['key_id']] = row
        affected.append(task['key_id'])
    append_event(context['run_dir']/'page_events.jsonl', {
        'at_utc':now_utc(), 'region':group[0]['region'], 'url':group[0]['batch_url'],
        'status':status, 'failure_kind':failure_kind, 'error':str(error),
        'http_status':receipt.get('http_status') if receipt else None,
        'response_body_file':receipt.get('body_file') if receipt else None,
        'affected_keys':affected})


def reconcile_cached_page_failures(context, latest=None):
    """Offline: checkpoint old cached 404/410 pages that previously halted early."""
    if (context['run_dir']/'collector.lock').exists():
        raise RuntimeError('Cannot reconcile while a collector holds the run lock')
    latest = load_results(context) if latest is None else latest
    groups = defaultdict(list)
    for task in context['plan'].to_dict('records'):
        if task['mapping_verified_locally'] and latest.get(task['key_id'], {}).get('status') not in SKIP_ON_RESUME:
            groups[task['batch_url']].append(task)
    for url, group in groups.items():
        index = context['run_dir']/'responses'/f'{digest(url.encode())}.json'
        if not index.exists():
            continue
        receipt = json.loads(index.read_text())
        if receipt.get('http_status') not in {404, 410}:
            continue
        if receipt.get('requested_url') != url or receipt.get('requested_region') != group[0]['region']:
            raise ValueError('Cached failure URL/region mismatch')
        root = context['run_dir'].resolve()
        body = (root/receipt['body_file']).resolve()
        if not body.is_relative_to(root) or digest(body.read_bytes()) != receipt['body_sha256']:
            raise ValueError('Cached failure body mismatch')
        # An HTTP error can still be a challenge page. Never downgrade a block.
        parse_document(body.read_bytes().decode(receipt.get('encoding') or 'utf-8', errors='replace'))
        record_page_failure(context, latest, group,
            f'HTTP {receipt["http_status"]}: page unavailable; matchup availability and sample unknown',
            receipt, failure_kind='http_unavailable')
    return latest


def page_reports(context, lookup, events):
    """Page attempts, table completeness and data validity are separate concepts."""
    attempted = {e.get('requested_url') for e in events
                 if e.get('event') in {'attempt_started', 'attempt_finished'}}
    rows = []
    for (region, url), group in lookup.loc[lookup.mapping_verified_locally].groupby(['region','batch_url'], sort=False):
        valid = valid_rate_mask(group)
        missing = group.status.eq('not_published_in_table') & group.filters_verified.eq(True)
        verified = bool(group.status.isin(TERMINAL).all() and group.filters_verified.eq(True).all())
        attempted_page = url in attempted or bool(group.collected_at_utc.notna().any())
        unresolved = group.status.eq(PAGE_UNRESOLVED)
        state = ('verified_table' if verified else PAGE_UNRESOLVED if unresolved.any()
                 else 'attempted_failed_or_incomplete' if attempted_page else 'pending')
        rows.append({'region':region, 'batch_url':url, 'page_status':state,
                     'attempted':attempted_page, 'successful_page':verified,
                     'planned_keys':len(group), 'valid_keys':int(valid.sum()),
                     'missing_requested_opponent_keys':int(missing.sum()),
                     'unresolved_failed_keys':int(unresolved.sum()),
                     'pending_keys':int(group.status.eq('pending').sum()),
                     'all_keys_valid':bool(valid.all()),
                     'http_status':next((x for x in group.http_status if pd.notna(x)), None),
                     'error':next((str(x) for x in group.error if pd.notna(x) and str(x)), '')})
    pages = pd.DataFrame(rows)
    summary = []
    for region in REGIONS:
        p = pages.loc[pages.region.eq(region)]
        summary.append({'region':region, 'planned_pages':len(p),
            'successful_pages':int(p.successful_page.sum()),
            'verified_tables_missing_requested_opponents':int(p.missing_requested_opponent_keys.gt(0).sum()),
            'missing_requested_opponent_keys':int(p.missing_requested_opponent_keys.sum()),
            'unresolved_failed_pages':int(p.page_status.eq(PAGE_UNRESOLVED).sum()),
            'unresolved_failed_keys':int(p.unresolved_failed_keys.sum()),
            'other_attempted_failed_pages':int(p.page_status.eq('attempted_failed_or_incomplete').sum()),
            'attempted_pages':int(p.attempted.sum()), 'pending_pages':int((~p.attempted).sum()),
            'valid_keys':int(p.valid_keys.sum()),
            'all_planned_pages_attempted':bool(p.attempted.all()),
            'all_keys_valid':bool(p.all_keys_valid.all())})
    return pages, pd.DataFrame(summary)


def export_results(context, latest, runtime_seconds=0.0):
    """Long regional lookup, paired comparison, and explicit coverage denominators."""
    run_dir, plan, mapping = context['run_dir'], context['plan'], context['mapping']
    rows = [latest.get(task['key_id'], result_record(task,{'status':'pending','error':'Not attempted'},context))
            for task in plan.to_dict('records')]
    lookup = pd.DataFrame(rows)
    for field in RATE_FIELDS:
        lookup[field] = pd.to_numeric(lookup[field],errors='raise')
    if (lookup.key_id.duplicated().any() or lookup.duplicated(LOOKUP_KEYS).any()
            or set(lookup.key_id) != set(plan.key_id)):
        raise ValueError('Regional lookup uniqueness / complete request-key preservation failed')
    for ext in ['csv','parquet']:
        _write_frame(lookup,run_dir/f'matchup_lookup.{ext}')
    _write_frame(lookup.loc[~lookup.status.eq('ok')],run_dir/'failure_no_data_report.csv')
    valid = valid_rate_mask(lookup)
    decisions = context['decisions']
    summary, thresholds, samples = [], [], []
    for region in REGIONS:
        region_mask = lookup.region.eq(region)
        ok_keys = set(lookup.loc[valid & region_mask,'key_id'])
        regional_plan = plan.loc[plan.region.eq(region)]
        observed = mapping.loc[mapping.request_kind.eq('observed') & mapping.region.eq(region)].drop_duplicates(['puuid','match_id'])
        covered = int(observed.key_id.isin(ok_keys).sum())
        entries = [
            {'metric':'observed_target_rows_all','total':len(decisions),'covered':covered,
             'coverage':covered/len(decisions) if len(decisions) else None},
            {'metric':'observed_known_opponent_rows',**_coverage(observed,ok_keys)},
            {'metric':'observed_unique_keys',**_coverage(regional_plan.loc[regional_plan.observed],ok_keys)},
            {'metric':'alternative_unique_keys',**_coverage(regional_plan.loc[regional_plan.alternative],ok_keys)}]
        for lane in ROLE_SLUGS:
            lane_rows = observed.loc[observed.lane.eq(lane)]
            lane_all = int(decisions.lane.eq(lane).sum())
            lane_covered = int(lane_rows.key_id.isin(ok_keys).sum())
            entries.extend([
                {'metric':f'lane_{lane}_observed_all_rows','total':lane_all,'covered':lane_covered,
                 'coverage':lane_covered/lane_all if lane_all else None},
                {'metric':f'lane_{lane}_observed_keys',**_coverage(regional_plan.loc[regional_plan.observed & regional_plan.lane.eq(lane)],ok_keys)},
                {'metric':f'lane_{lane}_alternative_keys',**_coverage(regional_plan.loc[regional_plan.alternative & regional_plan.lane.eq(lane)],ok_keys)}])
        summary += [dict(region=region,**entry) for entry in entries]
        for n in [0,10,25,50,100,200,500]:
            retained = set(lookup.loc[valid & region_mask & lookup.matchup_games.ge(n),'key_id'])
            count = int(observed.key_id.isin(retained).sum())
            obs_keys = regional_plan.loc[regional_plan.observed,'key_id']
            alt_keys = regional_plan.loc[regional_plan.alternative,'key_id']
            thresholds.append({'region':region,'minimum_matchup_games':n,'lookup_keys':len(retained),
                'observed_target_rows':count,'all_target_row_coverage':count/len(decisions) if len(decisions) else None,
                'observed_keys':int(obs_keys.isin(retained).sum()),'alternative_keys':int(alt_keys.isin(retained).sum()),
                'observed_key_coverage':float(obs_keys.isin(retained).mean()) if len(obs_keys) else None,
                'alternative_key_coverage':float(alt_keys.isin(retained).mean()) if len(alt_keys) else None})
        size = lookup.loc[valid & region_mask,'matchup_games']
        stats = size.describe(percentiles=[.1,.25,.5,.75,.9,.95])
        samples += [{'region':region,'statistic':stat,'matchup_games':value} for stat,value in stats.items()]
    summary = pd.DataFrame(summary)
    _write_frame(summary,run_dir/'coverage_summary.csv')
    _write_frame(pd.DataFrame(thresholds),run_dir/'sample_threshold_coverage.csv')
    _write_frame(pd.DataFrame(samples),run_dir/'matchup_sample_distribution.csv')
    paired, target_pairs = paired_tables(context,lookup)
    for stem,frame in [('regional_comparison',paired),('paired_target_observations',target_pairs)]:
        for ext in ['csv','parquet']:
            _write_frame(frame,run_dir/f'{stem}.{ext}')
    distributions = []
    populations = [('all_requested_unique_keys',paired),('observed_unique_keys',paired.loc[paired.observed]),
                   ('observed_player_games',target_pairs)]
    for name,frame in populations:
        differences = frame.loc[frame.both_regions_valid,'na_minus_global_pp']
        stats = differences.describe(percentiles=[.1,.25,.5,.75,.9,.95])
        distributions += [{'population':name,'statistic':stat,'na_minus_global_pp':value} for stat,value in stats.items()]
    _write_frame(pd.DataFrame(distributions),run_dir/'regional_difference_distribution.csv')
    extremes = paired.loc[paired.both_regions_valid].assign(absolute_difference_pp=lambda x:x.na_minus_global_pp.abs())
    _write_frame(extremes.nlargest(30,'absolute_difference_pp'),run_dir/'largest_regional_differences.csv')
    paired_thresholds = []
    for n in [0,10,25,50,100,200,500]:
        keep = target_pairs.both_regions_valid & target_pairs.global_matchup_games.ge(n) & target_pairs.na_matchup_games.ge(n)
        keys_kept = paired.both_regions_valid & paired.global_matchup_games.ge(n) & paired.na_matchup_games.ge(n)
        paired_thresholds.append({'minimum_games_in_each_region':n,'target_rows':int(keep.sum()),
            'all_target_share':float(keep.mean()) if len(keep) else None,
            'observed_unique_keys':int((keys_kept & paired.observed).sum()),
            'alternative_unique_keys':int((keys_kept & paired.alternative).sum())})
    _write_frame(pd.DataFrame(paired_thresholds),run_dir/'both_regions_sample_coverage.csv')
    events = read_events(run_dir/'request_events.jsonl')
    pages, page_summary = page_reports(context, lookup, events)
    _write_frame(pages, run_dir/'page_status.csv')
    _write_frame(page_summary, run_dir/'page_summary_by_region.csv')
    _write_frame(pages.loc[pages.page_status.eq(PAGE_UNRESOLVED)], run_dir/'unresolved_pages.csv')
    network = network_summary(context,events)
    _write_frame(network,run_dir/'network_summary_by_region.csv')
    sessions = read_events(run_dir/'sessions.jsonl')
    region_quality = {}
    for region in REGIONS:
        mask = lookup.region.eq(region)
        region_quality[region] = {
            'status_counts':lookup.loc[mask,'status'].value_counts().to_dict(),
            'successful_rates_missing_tier_average':int((valid & mask & lookup.tier_average_missing).sum()),
            'unverified_filter_keys':int((mask & ~lookup.filters_verified).sum())}
    quality = {'updated_at_utc':now_utc(),'run_id':context['run_id'],
        'status_counts':lookup.status.value_counts().to_dict(),'by_region':region_quality,
        'page_summary_by_region':page_summary.set_index('region').to_dict(orient='index'),
        'all_planned_pages_attempted':bool(page_summary.all_planned_pages_attempted.all()),
        'all_keys_valid':bool(valid.all()),
        'unresolved_failed_keys':int(lookup.status.eq(PAGE_UNRESOLVED).sum()),
        'pending_keys':int(lookup.status.eq('pending').sum()),
        'unidentified_opponent_target_rows':int((~decisions.known_lane_opponent).sum()),
        'target_rows_with_unknown_lane':int(decisions.lane.isna().sum()),
        'successful_rates_missing_tier_average':int((valid & lookup.tier_average_missing).sum()),
        'unverified_filter_keys':int((~lookup.filters_verified).sum()),
        'actual_http_attempts':int(network.actual_http_attempts.sum()),
        'actual_unique_network_urls':int(network.actual_unique_network_urls.sum()),
        'unfinished_http_attempts':int(network.unfinished_http_attempts.sum()),
        'actual_request_seconds':float(network.actual_request_seconds.sum()),
        'recorded_prior_session_seconds':sum(e.get('wall_seconds',0) for e in sessions),
        'current_session_seconds':round(runtime_seconds,3),
        'remaining_retryable_keys':int((~lookup.status.isin(TERMINAL)).sum()),
        'both_regions_target_rows':int(target_pairs.both_regions_valid.sum()),
        'both_regions_all_target_share':float(target_pairs.both_regions_valid.mean()) if len(target_pairs) else None,
        'reused_legacy_global_keys':int((lookup.get('reused_legacy_global_key',pd.Series(index=lookup.index,dtype=object)).notna()
                                        & lookup.status.isin(TERMINAL)).sum()),
        'caveat':'Retrospective overlapping patch-16.19 benchmarks; Global includes NA. No regional substitution. Not necessarily available before target kickoff.'}
    atomic_json(run_dir/'quality_summary.json',quality)
    return lookup, summary, quality


def retry_unresolved_pages(context, *, max_pages=5, regions=('na',)):
    """Retry ONLY up to max_pages unresolved pages; never collect pending pages.

    Each selected page uses Config.max_attempts at most. Old receipt indexes are
    archived before refresh; immutable bodies, checkpoint history and successful
    pages remain intact. Calling again is a new explicit bounded retry pass.
    """
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= 100:
        raise ValueError('max_pages must be an explicit integer between 1 and 100')
    return collect(context, retry_max_pages=max_pages, retry_regions=regions)


def collect(context, *, retry_max_pages=0, retry_regions=('na',)):
    """Normal resume skips unresolved pages; bounded retry mode handles only them."""
    if (isinstance(retry_max_pages, bool) or not isinstance(retry_max_pages, int)
            or not 0 <= retry_max_pages <= 100 or not retry_regions or not set(retry_regions) <= set(REGIONS)):
        raise ValueError('Invalid bounded page retry scope')
    run_dir = context['run_dir']
    lock = run_dir/'collector.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Collector lock exists at {lock}; verify the other collector has stopped before removing ONLY that lock.') from exc
    with os.fdopen(fd,'w') as f:
        f.write(canonical({'pid':os.getpid(),'started_at_utc':now_utc()}))
    try:
        return _collect_locked(context, retry_max_pages=retry_max_pages, retry_regions=retry_regions)
    finally:
        lock.unlink(missing_ok=True)


def _collect_locked(context, *, retry_max_pages=0, retry_regions=('na',)):
    run_dir, config = context['run_dir'], context['config']
    start, started_at = time.monotonic(), now_utc()
    latest = load_results(context)
    tasks = context['plan'].to_dict('records')
    if not set(latest) <= set(context['plan'].key_id):
        raise ValueError('Checkpoint contains keys outside the pinned plan')
    cache = None
    pages = 0
    pause_reason = None
    retry_mode = retry_max_pages > 0
    try:
        cache = PageCache(run_dir, config)
        groups = defaultdict(list)
        for task in tasks:
            status = latest.get(task['key_id'], {}).get('status')
            if retry_mode:
                if status != PAGE_UNRESOLVED or task['region'] not in retry_regions:
                    continue
            elif status in SKIP_ON_RESUME:
                continue
            if not task['mapping_verified_locally']:
                row = result_record(task, {'status':'unsupported_mapping', 'error':'Unregistered champion mapping'}, context)
                append_event(run_dir/'matchup_checkpoint.jsonl', row)
                latest[task['key_id']] = row
            else:
                groups[task['batch_url']].append(task)
        selected = list(groups.items())[:retry_max_pages] if retry_mode else list(groups.items())
        print(f'{"Bounded unresolved-only retry" if retry_mode else "Resume"}: {len(selected):,} page operations; '
              'completed regional pages are reused; directional fallback disabled.', flush=True)
        failures = 0
        recent_failures = deque(maxlen=max(2, config.max_consecutive_failures*2))
        last_region = None
        for batch_url, group in selected:
            region = group[0]['region']
            if region != last_region:
                failures = 0
                recent_failures.clear()
                last_region = region
            receipt = None
            page_failed = False
            try:
                html, receipt = cache.get(batch_url, refresh_unresolved=retry_mode)
                # Inspect error bodies for blocks/historical-unavailability too.
                parse_document(html)
                if receipt['http_status'] != 200:
                    raise EvidenceError('http_unavailable',
                        f'HTTP {receipt["http_status"]}: page unavailable; matchup availability and sample unknown')
                append_event(run_dir/'representation_inventory.jsonl',
                    {'url':batch_url,'region':region,'at_utc':now_utc(),**representation_inventory(html)})
                rates, notes = parse_batch_table(html, group)
                if set(rates) != {t['key_id'] for t in group}:
                    raise EvidenceError('parsing_error', 'Verified table did not account for every requested key')
                records = [result_record(task, rates[task['key_id']], context, receipt) for task in group]
                if any(r['status'] not in TERMINAL or not r['filters_verified'] for r in records):
                    raise EvidenceError('parsing_error', 'Table adapter returned unverified/incomplete key outcomes')
                for row in records:
                    append_event(run_dir/'matchup_checkpoint.jsonl', row)
                    latest[row['key_id']] = row
                append_event(run_dir/'page_events.jsonl', {'at_utc':now_utc(), 'url':batch_url,
                    'region':region,'status':'verified_table','affected_keys':[r['key_id'] for r in records]})
                append_event(run_dir/'strategy_events.jsonl', {'url':batch_url,'region':region,'notes':notes,
                    'accepted_keys':sum(r['status']=='ok' for r in records)})
                failures = 0
                recent_failures.append(False)
            except PauseCollection as exc:
                receipt = cache.last_receipt if cache.last_receipt and cache.last_receipt['requested_url']==batch_url else receipt
                record_page_failure(context, latest, group, exc, receipt, status='blocked', failure_kind='access_or_rate_limit')
                raise
            except (EvidenceError, ValueError, TypeError, KeyError, OverflowError) as exc:
                receipt = receipt or (cache.last_receipt if cache.last_receipt and cache.last_receipt['requested_url']==batch_url else None)
                kind = exc.status if isinstance(exc, EvidenceError) else 'parsing_error'
                if kind in {'blocked','unavailable_patch'}:
                    record_page_failure(context, latest, group, exc, receipt, status=kind, failure_kind=kind)
                    raise PauseCollection(str(exc)) from exc
                record_page_failure(context, latest, group, exc, receipt, failure_kind=kind)
                page_failed = True
                failures += 1
                recent_failures.append(True)
            pages += 1
            print(f'{region} | pages this pass={pages:,}/{len(selected):,} | '
                  f'valid keys={sum(r["status"]=="ok" for r in latest.values()):,} | '
                  f'unresolved keys={sum(r["status"]==PAGE_UNRESOLVED for r in latest.values()):,} | '
                  f'HTTP attempts={cache.actual_attempts:,} | cache hits={cache.cache_hits:,}', flush=True)
            if pages % config.export_every_pages == 0:
                export_results(context, latest, time.monotonic()-start)
            if page_failed and (failures >= config.max_consecutive_failures or
                    len(recent_failures) == recent_failures.maxlen and sum(recent_failures) >= config.max_consecutive_failures):
                raise PauseCollection(f'Repeated/systemic page failures: {failures} consecutive; '
                    f'{sum(recent_failures)}/{len(recent_failures)} in recent regional pages. Inspect cached evidence.')
    except (KeyboardInterrupt, PauseCollection) as exc:
        pause_reason = str(exc) or 'KeyboardInterrupt'
        print('Collection paused:', pause_reason, '| checkpoints and source responses retained.', flush=True)
    finally:
        if cache is not None:
            cache.close()
        elapsed = time.monotonic()-start
        lookup, summary, quality = export_results(context, latest, elapsed)
        append_event(run_dir/'sessions.jsonl', {'started_at_utc':started_at,'ended_at_utc':now_utc(),
            'wall_seconds':round(elapsed,3),'pause_reason':pause_reason,
            'mode':'retry_unresolved_only' if retry_mode else 'normal_resume',
            'retry_max_pages':retry_max_pages,'retry_regions':list(retry_regions),
            'cache_hits':cache.cache_hits if cache else 0,'config':asdict(config),
            'implementation_sha256':digest(Path(__file__).read_bytes())})
        atomic_json(run_dir/'run_state.json', {'updated_at_utc':now_utc(),'pause_reason':pause_reason,
            'finished_all_keys':bool(lookup.status.isin(TERMINAL).all()),
            'all_planned_pages_attempted':quality['all_planned_pages_attempted'],
            'all_keys_valid':quality['all_keys_valid'],
            'unresolved_failed_keys':quality['unresolved_failed_keys'], 'pending_keys':quality['pending_keys']})
    print(pd.read_csv(run_dir/'page_summary_by_region.csv').to_string(index=False))
    print('All planned pages attempted:',quality['all_planned_pages_attempted'],
          '| All keys have valid rates:',quality['all_keys_valid'])
    print('Artifacts:',run_dir)
    return lookup, summary, quality
