import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Root directory of the project (src/audit/dataset_audit.py -> parents[2] is project root)
ROOT = Path(__file__).resolve().parents[2]

# Default paths
RAW_DATA_DIR = ROOT / "data" / "raw"
AUDIT_DIR = ROOT / "data" / "audit"

# Maximum number of numeric values kept for percentile estimation (streaming reservoir)
MAX_SAMPLE_SIZE = 2000


# ---------------------------------------------------------------------------
# Streaming statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: List[float], percentile: float) -> Optional[float]:
    """Return approximate percentile from a sorted list using linear interpolation."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = int(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


class StreamingNumeric:
    """Accumulate numeric statistics without storing all values."""

    def __init__(self, max_sample: int = MAX_SAMPLE_SIZE):
        self.count = 0
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self.sum = 0.0
        self.sum_sq = 0.0
        self._sample: List[float] = []
        self._max_sample = max_sample
        self._seen = 0

    def add(self, value: Any) -> None:
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        self.count += 1
        self._seen += 1
        self.sum += value
        self.sum_sq += value * value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value

        # Reservoir sampling: maintain a fixed-size random sample for percentiles
        if len(self._sample) < self._max_sample:
            self._sample.append(value)
        else:
            j = random.randint(0, self._seen - 1)
            if j < self._max_sample:
                self._sample[j] = value

    def stats(self) -> Optional[Dict[str, Any]]:
        if self.count == 0:
            return None
        mean = self.sum / self.count
        # Sample variance (unbiased)
        if self.count > 1:
            variance = (self.sum_sq - (self.sum * self.sum) / self.count) / (self.count - 1)
        else:
            variance = 0.0
        variance = max(0.0, variance)
        std = math.sqrt(variance)

        sample_sorted = sorted(self._sample)
        p25 = _percentile(sample_sorted, 25.0)
        p50 = _percentile(sample_sorted, 50.0)
        p75 = _percentile(sample_sorted, 75.0)
        p95 = _percentile(sample_sorted, 95.0)
        iqr = (p75 - p25) if (p75 is not None and p25 is not None) else None
        cv = (std / mean) if mean != 0 else None

        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": std,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "iqr": iqr,
            "cv": cv,
        }


class CategoricalCounter:
    """Accumulate counts for discrete/categorical variables."""

    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.missing = 0
        self.total = 0

    def add(self, value: Any) -> None:
        self.total += 1
        if value is None:
            self.missing += 1
        else:
            key = str(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def stats(self) -> Dict[str, Any]:
        non_missing = self.total - self.missing
        if non_missing == 0:
            return {
                "observed_categories": 0,
                "dominant": None,
                "dominant_percentage": None,
                "categories_below_1_percent": [],
                "hhi": None,
                "entropy": None,
                "normalized_entropy": None,
                "distribution": [],
            }

        dist = []
        for key, cnt in self.counts.items():
            pct = (cnt / non_missing) * 100.0
            dist.append({"value": key, "count": cnt, "percentage": pct})
        dist.sort(key=lambda x: x["count"], reverse=True)

        dominant = dist[0]["value"]
        dominant_pct = dist[0]["percentage"]
        below_1 = [d["value"] for d in dist if d["percentage"] < 1.0]

        hhi = sum((cnt / non_missing) ** 2 for cnt in self.counts.values())
        entropy = 0.0
        for cnt in self.counts.values():
            p = cnt / non_missing
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(len(self.counts)) if len(self.counts) > 1 else 0.0
        normalized_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0

        return {
            "observed_categories": len(self.counts),
            "dominant": dominant,
            "dominant_percentage": dominant_pct,
            "categories_below_1_percent": below_1,
            "hhi": hhi,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "distribution": dist,
        }


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> Optional[Any]:
    """Safely load a JSON file; return None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def _extract_clan_tag_from_member(member: Dict[str, Any], filename: str) -> Optional[str]:
    """Try to infer the clan tag from a member record.

    The Clash of Clans member endpoint sometimes does not include the clan tag
    directly. We check several likely places: nested 'clan' object, 'clanTag',
    or a filename that contains a tag after splitting on '__' (best effort).
    """
    # 1) nested clan object
    clan_obj = member.get("clan")
    if isinstance(clan_obj, dict):
        tag = clan_obj.get("tag")
        if tag:
            return tag
    # 2) direct clanTag field
    tag = member.get("clanTag")
    if tag:
        return tag
    # 3) try to parse from filename, e.g. "#CLAN__PLAYER.json"
    stem = Path(filename).stem
    if "__" in stem:
        parts = stem.split("__")
        if len(parts) >= 2:
            possible_clan = parts[0]
            if possible_clan.startswith("#"):
                return possible_clan
    return None


# ---------------------------------------------------------------------------
# Audit processing
# ---------------------------------------------------------------------------

def run_audit(raw_data_dir: Path) -> Dict[str, Any]:
    """Run a streaming audit over the raw data directories."""
    players_dir = raw_data_dir / "players"
    clans_dir = raw_data_dir / "clans"
    members_dir = raw_data_dir / "members"

    result = {
        "players": {
            "total": 0,
            "unique_tags": 0,
            "with_clan": 0,
            "without_clan": 0,
            "missing_tag": 0,
            "missing_fields": {},
            "distributions": {},
            "numeric_stats": {},
        },
        "clans": {
            "total": 0,
            "unique_tags": 0,
            "missing_tag": 0,
            "distributions": {},
            "numeric_stats": {},
        },
        "members": {
            "total_relationships": 0,
            "unique_clans_represented": 0,
            "unique_players_represented": 0,
            "missing_clan_tag": 0,
            "clan_sizes": [],
        },
        "warnings": [],
        "files": {
            "players_read": 0,
            "players_corrupt": 0,
            "clans_read": 0,
            "clans_corrupt": 0,
            "members_read": 0,
            "members_corrupt": 0,
        },
    }

    # Unique tag sets
    player_tags: set[str] = set()
    clan_tags: set[str] = set()
    member_player_tags: set[str] = set()
    member_clan_tags: set[str] = set()
    clan_sizes: List[int] = []

    # Player categorical counters
    player_categorical_fields = {
        "townHallLevel": CategoricalCounter(),
        "builderHallLevel": CategoricalCounter(),
        "expLevel": CategoricalCounter(),
    }
    player_numeric_fields = {
        "trophies": StreamingNumeric(),
        "bestTrophies": StreamingNumeric(),
        "warStars": StreamingNumeric(),
        "attackWins": StreamingNumeric(),
        "defenseWins": StreamingNumeric(),
        "donations": StreamingNumeric(),
        "donationsReceived": StreamingNumeric(),
        "clanCapitalContributions": StreamingNumeric(),
    }
    player_missing_fields = {
        "tag": 0,
        "townHallLevel": 0,
        "builderHallLevel": 0,
        "expLevel": 0,
        "trophies": 0,
        "bestTrophies": 0,
        "warStars": 0,
        "attackWins": 0,
        "defenseWins": 0,
        "donations": 0,
        "donationsReceived": 0,
        "clanCapitalContributions": 0,
    }

    # Clan categorical counters
    clan_categorical_fields = {
        "clanLevel": CategoricalCounter(),
        "location": CategoricalCounter(),  # uses "id|name" or "id" if name missing
    }
    clan_numeric_fields = {
        "clanPoints": StreamingNumeric(),
        "clanCapitalPoints": StreamingNumeric(),
        "warWins": StreamingNumeric(),
        "warLosses": StreamingNumeric(),
        "warTies": StreamingNumeric(),
        "warWinStreak": StreamingNumeric(),
        "members": StreamingNumeric(),  # memberCount from clan object
    }
    clan_missing_fields = {
        "tag": 0,
        "clanLevel": 0,
        "clanPoints": 0,
        "clanCapitalPoints": 0,
        "warWins": 0,
        "warLosses": 0,
        "warTies": 0,
        "warWinStreak": 0,
        "members": 0,
        "location": 0,
    }

    # ---------- Process players ----------
    if players_dir.exists():
        for file_path in players_dir.glob("*.json"):
            result["files"]["players_read"] += 1
            data = load_json_file(file_path)
            if data is None:
                result["files"]["players_corrupt"] += 1
                continue
            if isinstance(data, dict):
                player_objs = [data]
            elif isinstance(data, list):
                player_objs = [d for d in data if isinstance(d, dict)]
            else:
                player_objs = []

            for player in player_objs:
                result["players"]["total"] += 1
                tag = player.get("tag")
                if tag is not None:
                    player_tags.add(str(tag))
                else:
                    player_missing_fields["tag"] += 1

                has_clan = False
                if player.get("clan") is not None:
                    has_clan = True
                    # If clan tag is present, also add to clan_tags? Not needed.
                if has_clan:
                    result["players"]["with_clan"] += 1
                else:
                    result["players"]["without_clan"] += 1

                # Categorical fields
                for field, counter in player_categorical_fields.items():
                    value = player.get(field)
                    counter.add(value)
                    if value is None:
                        player_missing_fields[field] += 1
                # Numeric fields
                for field, stream in player_numeric_fields.items():
                    value = player.get(field)
                    stream.add(value)
                    if value is None:
                        player_missing_fields[field] += 1

    result["players"]["unique_tags"] = len(player_tags)
    result["players"]["missing_fields"] = player_missing_fields

    # ---------- Process clans ----------
    if clans_dir.exists():
        for file_path in clans_dir.glob("*.json"):
            result["files"]["clans_read"] += 1
            data = load_json_file(file_path)
            if data is None:
                result["files"]["clans_corrupt"] += 1
                continue
            if isinstance(data, dict):
                clan_objs = [data]
            elif isinstance(data, list):
                clan_objs = [d for d in data if isinstance(d, dict)]
            else:
                clan_objs = []

            for clan in clan_objs:
                result["clans"]["total"] += 1
                tag = clan.get("tag")
                if tag is not None:
                    clan_tags.add(str(tag))
                else:
                    clan_missing_fields["tag"] += 1

                # Categorical fields
                clan_level = clan.get("clanLevel")
                clan_categorical_fields["clanLevel"].add(clan_level)
                if clan_level is None:
                    clan_missing_fields["clanLevel"] += 1

                # Location as id|name
                location = clan.get("location")
                if location is None:
                    clan_categorical_fields["location"].add(None)
                    clan_missing_fields["location"] += 1
                else:
                    loc_id = location.get("id") if isinstance(location, dict) else None
                    loc_name = location.get("name") if isinstance(location, dict) else None
                    loc_key = f"{loc_id}|{loc_name}" if loc_id is not None else None
                    if loc_key is not None:
                        clan_categorical_fields["location"].add(loc_key)
                    else:
                        clan_categorical_fields["location"].add(None)
                        clan_missing_fields["location"] += 1

                # Numeric fields
                for field, stream in clan_numeric_fields.items():
                    value = clan.get(field)
                    stream.add(value)
                    if value is None:
                        clan_missing_fields[field] += 1

    result["clans"]["unique_tags"] = len(clan_tags)
    result["clans"]["missing_fields"] = clan_missing_fields

    # ---------- Process members ----------
    if members_dir.exists():
        for file_path in members_dir.glob("*.json"):
            result["files"]["members_read"] += 1
            data = load_json_file(file_path)
            if data is None:
                result["files"]["members_corrupt"] += 1
                continue
            if isinstance(data, dict):
                member_objs = [data]
            elif isinstance(data, list):
                member_objs = [d for d in data if isinstance(d, dict)]
            else:
                member_objs = []

            for member in member_objs:
                result["members"]["total_relationships"] += 1
                player_tag = member.get("tag")
                if player_tag is not None:
                    member_player_tags.add(str(player_tag))
                clan_tag = _extract_clan_tag_from_member(member, file_path.name)
                if clan_tag is not None:
                    member_clan_tags.add(clan_tag)
                else:
                    result["members"]["missing_clan_tag"] += 1

        # Compute clan sizes from member_clan_tags? Actually we need per-clan count.
        # For exact sizes, we would need a separate pass or store mapping.
        # Since members files don't store clan tag reliably, we use member_clan_tags as unique clans only.
        # For concentration, we need sizes. We'll compute sizes by counting in a second pass over members?
        # Better: today's implementation already processed files; we can do a second pass to collect sizes.
        # However to avoid complexity, we can use a dict here if we process again.
        # But for simplicity, we will perform a second pass to count members per clan.
        # This is acceptable in streaming because we only load one file at a time.
        if member_clan_tags:
            # Count sizes by reading again (still streaming)
            clan_size_counter: Dict[str, int] = {}
            for file_path in members_dir.glob("*.json"):
                data = load_json_file(file_path)
                if data is None:
                    continue
                if isinstance(data, dict):
                    member_objs = [data]
                elif isinstance(data, list):
                    member_objs = [d for d in data if isinstance(d, dict)]
                else:
                    member_objs = []
                for member in member_objs:
                    clan_tag = _extract_clan_tag_from_member(member, file_path.name)
                    if clan_tag is not None:
                        clan_size_counter[clan_tag] = clan_size_counter.get(clan_tag, 0) + 1
            clan_sizes = list(clan_size_counter.values())
            # Also update unique clans represented based on actual sizes
            member_clan_tags = set(clan_size_counter.keys())

    result["members"]["unique_clans_represented"] = len(member_clan_tags)
    result["members"]["unique_players_represented"] = len(member_player_tags)
    result["members"]["clan_sizes"] = clan_sizes

    # Compute relationship stats
    total_members = result["members"]["total_relationships"]
    unique_clans = len(clan_sizes)
    if unique_clans > 0:
        total_players_in_members = sum(clan_sizes)
        avg_per_clan = total_members / unique_clans
        sorted_sizes = sorted(clan_sizes)
        median_per_clan = _percentile(sorted_sizes, 50.0)
        min_per_clan = sorted_sizes[0]
        max_per_clan = sorted_sizes[-1]
        top_concentration = {}
        top_ns = [1, 10, 50, 100]
        for n in top_ns:
            if n > 0:
                top_sum = sum(sorted_sizes[-n:]) if n <= len(sorted_sizes) else sum(sorted_sizes)
                top_concentration[f"top_{n}"] = (top_sum / total_players_in_members * 100.0) if total_players_in_members else 0.0
        result["members"]["stats"] = {
            "average_players_per_clan": avg_per_clan,
            "median_players_per_clan": median_per_clan,
            "min_players_per_clan": min_per_clan,
            "max_players_per_clan": max_per_clan,
            "concentration_percentages": top_concentration,
        }
    else:
        result["members"]["stats"] = {
            "average_players_per_clan": None,
            "median_players_per_clan": None,
            "min_players_per_clan": None,
            "max_players_per_clan": None,
            "concentration_percentages": {},
        }

    # Compute distributions for players and clans
    for field, counter in player_categorical_fields.items():
        result["players"]["distributions"][field] = counter.stats()
    for field, stream in player_numeric_fields.items():
        result["players"]["numeric_stats"][field] = stream.stats()
    for field, counter in clan_categorical_fields.items():
        result["clans"]["distributions"][field] = counter.stats()
    for field, stream in clan_numeric_fields.items():
        result["clans"]["numeric_stats"][field] = stream.stats()

    # ---- Warnings (transparent) ----
    warnings = result["warnings"]

    # Concentration > 40% in player categorical distributions
    for field, dist in result["players"]["distributions"].items():
        dom_pct = dist.get("dominant_percentage")
        if dom_pct is not None and dom_pct > 40.0:
            warnings.append({
                "type": "POTENTIAL CONCENTRATION",
                "detail": f"Player field '{field}' has dominant category {dist['dominant']} with {dom_pct:.1f}% share.",
            })

    # Location concentration > 30%
    loc_dist = result["clans"]["distributions"].get("location")
    if loc_dist and loc_dist.get("dominant_percentage") is not None and loc_dist["dominant_percentage"] > 30.0:
        warnings.append({
            "type": "HIGH CONCENTRATION",
            "detail": f"Geographic distribution is highly concentrated: dominant region '{loc_dist['dominant']}' holds {loc_dist['dominant_percentage']:.1f}%.",
        })

    # Clan-level concentration > 40%
    clan_level_dist = result["clans"]["distributions"].get("clanLevel")
    if clan_level_dist and clan_level_dist.get("dominant_percentage") is not None and clan_level_dist["dominant_percentage"] > 40.0:
        warnings.append({
            "type": "POTENTIAL CONCENTRATION",
            "detail": f"Clan level distribution is heavily concentrated around level {clan_level_dist['dominant']}.",
        })

    # Players without clan > 30%
    total_players = result["players"]["total"]
    if total_players > 0:
        without_clan_pct = result["players"]["without_clan"] / total_players * 100.0
        if without_clan_pct > 30.0:
            warnings.append({
                "type": "LOW COVERAGE",
                "detail": f"{without_clan_pct:.1f}% of players have no clan association.",
            })

    # Missing important fields > 10%
    missing_fields = result["players"]["missing_fields"]
    # Use total players processed (including those possibly without field)
    if total_players > 0:
        for field, count in missing_fields.items():
            if field == "tag":
                continue
            if count / total_players > 0.10:
                warnings.append({
                    "type": "LOW COVERAGE",
                    "detail": f"Player field '{field}' is missing in {count/total_players*100:.1f}% of records.",
                })

    # Top clan concentration > 50%
    top_conc = result["members"]["stats"].get("concentration_percentages", {})
    top1 = top_conc.get("top_1")
    if top1 is not None and top1 > 50.0:
        warnings.append({
            "type": "HIGH CONCENTRATION",
            "detail": f"Top 1 clan contains {top1:.1f}% of all players represented in members.",
        })

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _safe_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}%"


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate a Markdown report from the audit result dict."""
    players = report["players"]
    clans = report["clans"]
    members = report["members"]
    warnings = report["warnings"]
    files = report["files"]

    lines = []
    lines.append("# Dataset Audit\n")
    lines.append("## Overview\n")
    lines.append(f"- Total players: {players['total']} (unique tags: {players['unique_tags']})")
    lines.append(f"- Players with clan: {players['with_clan']}")
    lines.append(f"- Players without clan: {players['without_clan']}")
    lines.append(f"- Total clans: {clans['total']} (unique tags: {clans['unique_tags']})")
    lines.append(f"- Total clan-member relationships: {members['total_relationships']}")
    lines.append(f"- Unique clans represented in members: {members['unique_clans_represented']}")
    lines.append(f"- Unique players represented in members: {members['unique_players_represented']}")
    lines.append("")

    # Player distributions
    lines.append("## Player distributions\n")
    for field, dist in players["distributions"].items():
        lines.append(f"### {field}\n")
        if dist["observed_categories"] == 0:
            lines.append("No data.\n")
            continue
        lines.append(f"- Observed categories: {dist['observed_categories']}")
        lines.append(f"- Dominant category: {dist['dominant']} ({_safe_pct(dist['dominant_percentage'])})")
        lines.append(f"- Categories with < 1%: {', '.join(dist['categories_below_1_percent']) if dist['categories_below_1_percent'] else 'None'}")
        if dist["hhi"] is not None:
            lines.append(f"- HHI: {dist['hhi']:.4f}")
        if dist["entropy"] is not None:
            lines.append(f"- Entropy: {dist['entropy']:.4f} (normalized: {dist['normalized_entropy']:.4f})")
        lines.append("")
        lines.append("| Value | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for entry in dist["distribution"]:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['percentage']:.2f}% |")
        lines.append("")

    # Numeric stats for players
    lines.append("### Numeric statistics\n")
    lines.append("| Field | Count | Min | Max | Mean | Std | p25 | p50 | p75 | p95 | IQR | CV |")
    lines.append("|-------|-------|-----|-----|------|-----|-----|-----|-----|-----|-----|----|")
    for field, stats in players["numeric_stats"].items():
        if stats is None:
            lines.append(f"| {field} | 0 | - | - | - | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {field} | {stats['count']} | {stats['min']} | {stats['max']} | {stats['mean']:.2f} | "
                f"{stats['std']:.2f} | {stats['p25']:.2f} | {stats['p50']:.2f} | {stats['p75']:.2f} | "
                f"{stats['p95']:.2f} | {stats['iqr']:.2f} | {stats['cv']:.4f} |"
            )
    lines.append("")

    # Clan distributions
    lines.append("## Clan distributions\n")
    for field, dist in clans["distributions"].items():
        lines.append(f"### {field}\n")
        if dist["observed_categories"] == 0:
            lines.append("No data.\n")
            continue
        lines.append(f"- Observed categories: {dist['observed_categories']}")
        lines.append(f"- Dominant category: {dist['dominant']} ({_safe_pct(dist['dominant_percentage'])})")
        lines.append("")
        lines.append("| Value | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for entry in dist["distribution"]:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['percentage']:.2f}% |")
        lines.append("")

    # Clan numeric stats
    lines.append("### Numeric statistics\n")
    lines.append("| Field | Count | Min | Max | Mean | Std | p25 | p50 | p75 | p95 | IQR | CV |")
    lines.append("|-------|-------|-----|-----|------|-----|-----|-----|-----|-----|-----|----|")
    for field, stats in clans["numeric_stats"].items():
        if stats is None:
            lines.append(f"| {field} | 0 | - | - | - | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {field} | {stats['count']} | {stats['min']} | {stats['max']} | {stats['mean']:.2f} | "
                f"{stats['std']:.2f} | {stats['p25']:.2f} | {stats['p50']:.2f} | {stats['p75']:.2f} | "
                f"{stats['p95']:.2f} | {stats['iqr']:.2f} | {stats['cv']:.4f} |"
            )
    lines.append("")

    # Geographic coverage
    lines.append("## Geographic coverage\n")
    loc_dist = clans["distributions"].get("location")
    if loc_dist and loc_dist["observed_categories"] > 0:
        lines.append(f"- Number of regions observed: {loc_dist['observed_categories']}")
        lines.append(f"- Dominant region: {loc_dist['dominant']} ({_safe_pct(loc_dist['dominant_percentage'])})")
        lines.append("")
        lines.append("| Region | Count | Percentage |")
        lines.append("|--------|-------|------------|")
        for entry in loc_dist["distribution"]:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['percentage']:.2f}% |")
    else:
        lines.append("No location data.")
    lines.append("")

    # Clan concentration
    lines.append("## Clan concentration\n")
    member_stats = members["stats"]
    lines.append(f"- Average players per clan: {member_stats.get('average_players_per_clan', 'N/A')}")
    lines.append(f"- Median players per clan: {member_stats.get('median_players_per_clan', 'N/A')}")
    lines.append(f"- Min players per clan: {member_stats.get('min_players_per_clan', 'N/A')}")
    lines.append(f"- Max players per clan: {member_stats.get('max_players_per_clan', 'N/A')}")
    conc = member_stats.get("concentration_percentages", {})
    for k, v in conc.items():
        label = k.replace('_', ' ')
        lines.append(f"- Percentage in {label}: {_safe_pct(v)}")
    lines.append("")

    # Missing data
    lines.append("## Missing data\n")
    lines.append("### Player missing fields\n")
    if players['total'] > 0:
        lines.append("| Field | Missing count | Missing % |")
        lines.append("|-------|---------------|-----------|")
        for field, count in players["missing_fields"].items():
            pct = count / players['total'] * 100.0
            lines.append(f"| {field} | {count} | {pct:.2f}% |")
    else:
        lines.append("No players.")
    lines.append("")
    lines.append("### Clan missing fields\n")
    if clans['total'] > 0:
        lines.append("| Field | Missing count | Missing % |")
        lines.append("|-------|---------------|-----------|")
        for field, count in clans["missing_fields"].items():
            pct = count / clans['total'] * 100.0
            lines.append(f"| {field} | {count} | {pct:.2f}% |")
    else:
        lines.append("No clans.")
    lines.append("")
    lines.append(f"- Members missing clan tag: {members['missing_clan_tag']}")

    # Diversity indicators
    lines.append("\n## Diversity indicators\n")
    # We can reuse concentration metrics already printed. Just add a short table.
    lines.append("| Variable | Categories | Dominant | Dominant % | HHI | Entropy (normalized) |")
    lines.append("|----------|------------|----------|------------|-----|----------------------|")
    for field, dist in list(players["distributions"].items()) + list(clans["distributions"].items()):
        if dist["observed_categories"] > 0:
            lines.append(f"| {field} | {dist['observed_categories']} | {dist['dominant']} | {_safe_pct(dist['dominant_percentage'])} | {dist['hhi']:.4f} | {dist['normalized_entropy']:.4f} |")
    lines.append("")

    # Potential warnings
    lines.append("## Potential warnings\n")
    if warnings:
        for w in warnings:
            lines.append(f"- **{w['type']}**: {w['detail']}")
    else:
        lines.append("No warnings detected.")
    lines.append("")

    # General assessment
    lines.append("## General assessment\n")
    # Simple transparent assessment based on thresholds.
    total_players = players['total']
    total_clans = clans['total']
    missing_pct = 0.0
    if total_players > 0:
        # Average missing percentage across important fields
        fields_count = len(players['missing_fields']) - 1  # exclude tag
        missing_sum = sum(count for field, count in players['missing_fields'].items() if field != 'tag')
        missing_pct = (missing_sum / (total_players * fields_count)) * 100.0
    dom_player = 0.0
    for dist in players['distributions'].values():
        if dist['dominant_percentage'] is not None and dist['dominant_percentage'] > dom_player:
            dom_player = dist['dominant_percentage']
    dom_location = 0.0
    loc = clans['distributions'].get('location')
    if loc and loc['dominant_percentage'] is not None:
        dom_location = loc['dominant_percentage']
    top1 = member_stats.get('concentration_percentages', {}).get('top_1')

    if total_players < 1000:
        assessment = "Limited coverage"
    elif total_players >= 10000 and missing_pct < 5.0 and dom_player < 40.0 and (top1 is None or top1 < 40.0) and players['without_clan'] / total_players < 0.10:
        assessment = "Excellent coverage"
    elif total_players >= 5000 and missing_pct < 10.0 and dom_player < 50.0 and (top1 is None or top1 < 50.0):
        assessment = "Good coverage"
    elif total_players >= 1000 and missing_pct < 15.0:
        assessment = "Moderate coverage"
    else:
        assessment = "Limited coverage"
    lines.append(f"Assessment: **{assessment}**")
    lines.append("")

    return "\n".join(lines)


def generate_json_report(report: Dict[str, Any]) -> str:
    """Return a JSON string of the audit result."""
    return json.dumps(report, indent=2, default=str)


def write_report_files(report: Dict[str, Any], audit_dir: Path) -> None:
    """Write latest_report.json and latest_report.md to the audit directory."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / "latest_report.json"
    md_path = audit_dir / "latest_report.md"
    json_path.write_text(generate_json_report(report), encoding="utf-8")
    md_path.write_text(generate_markdown_report(report), encoding="utf-8")


def main() -> None:
    """Entry point: run audit and write reports."""
    raw_data_dir = RAW_DATA_DIR
    audit_dir = AUDIT_DIR
    report = run_audit(raw_data_dir)
    write_report_files(report, audit_dir)
    print("Audit completed. Reports written to", audit_dir)


if __name__ == '__main__':
    main()
