from typing import Optional, Dict, Any, List
import re
from app.database import db


def set_brand_details(
    name: str,
    tagline: Optional[str] = None,
    mission: Optional[str] = None,
    values: Optional[List[str]] = None,
    tone: Optional[str] = None,
    website: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    props = {
        "name": name,
        "tagline": tagline,
        "mission": mission,
        "values": values or [],
        "tone": tone,
        "website": website,
        **(extras or {}),
    }
    result = db.execute_query(
        """
        MERGE (b:Brand {id:'default'})
        SET b += $props
        RETURN b AS brand
        """,
        {"props": props},
    )
    return result[0] if result else {}


def upsert_design_rule(
    rule_id: str,
    description: str,
    rule_type: str,
    params: Optional[Dict[str, Any]] = None,
    lastAmendedAt: Optional[str] = None,
    lastEvidenceScore: Optional[float] = None,
) -> Dict[str, Any]:
    result = db.execute_query(
        """
        MERGE (r:DesignRule {id: $id})
        SET r.description = $desc,
            r.type = $type,
            r.params = $params,
            r.lastAmendedAt = datetime($lastAmendedAt),
            r.lastEvidenceScore = $lastEvidenceScore
        RETURN r AS rule
        """,
        {
            "id": rule_id,
            "desc": description,
            "type": rule_type,
            "params": params or {},
            "lastAmendedAt": lastAmendedAt,
            "lastEvidenceScore": lastEvidenceScore,
        },
    )
    return result[0] if result else {}


def upsert_palette(name: str, token_keys: List[str]) -> Dict[str, Any]:
    result = db.execute_query(
        """
        MERGE (p:Palette {name: $name})
        WITH p
        UNWIND $keys AS k
        MATCH (t:BrandToken {key: k})
        MERGE (p)-[:CONTAINS]->(t)
        RETURN p AS palette
        """,
        {"name": name, "keys": token_keys},
    )
    return result[0] if result else {}


def upsert_article(
    slug: str,
    title: str,
    summary: str,
    url: Optional[str] = None,
    topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    result = db.execute_query(
        """
        MERGE (a:Article {slug: $slug})
        SET a.title = $title,
            a.summary = $summary,
            a.url = $url
        WITH a, $topics AS topics
        UNWIND topics AS topic
        MERGE (t:Topic {name: topic})
        MERGE (a)-[:ABOUT]->(t)
        RETURN a AS article
        """,
        {"slug": slug, "title": title, "summary": summary, "url": url, "topics": topics or []},
    )
    return result[0] if result else {}


def check_adherence() -> Dict[str, Any]:
    # Basic adherence checks plus rule evaluations.
    tokens = db.execute_query(
        """
        MATCH (t:BrandToken)
        RETURN t.key AS key, t.role AS role, t.hex AS hex, t.mutability_level AS mutability
        """
    )
    role_counts: Dict[str, int] = {}
    invalid_hex: List[str] = []
    invalid_mutability: List[str] = []
    allowed_mutability = {"fixed", "flexible", "adaptive"}

    for t in tokens:
        role = (t.get("role") or "").lower()
        role_counts[role] = role_counts.get(role, 0) + 1
        hexv = t.get("hex") or ""
        if not isinstance(hexv, str) or not len(hexv) in (4, 7) or not hexv.startswith("#"):
            invalid_hex.append(t.get("key"))
        mut = (t.get("mutability") or "").lower()
        if mut and mut not in allowed_mutability:
            invalid_mutability.append(t.get("key"))

    # Fetch rules
    rules = db.execute_query("MATCH (r:DesignRule) RETURN r.id AS id, r.type AS type, r.params AS params")

    issues: Dict[str, Any] = {}

    # 1) require_roles
    required_roles = set()
    for r in rules:
        if (r.get("type") or "") == "require_roles":
            required_roles.update((r.get("params") or {}).get("roles", []))
    missing_roles = [role for role in sorted(required_roles) if role_counts.get(role, 0) == 0]
    if missing_roles:
        issues["missing_roles"] = missing_roles

    # 2) mutability_allowed
    for r in rules:
        if (r.get("type") or "") == "mutability_allowed":
            allowed = set((r.get("params") or {}).get("allowed", []))
            if allowed:
                bad = [k for k in tokens if (k.get("mutability") or "").lower() not in {a.lower() for a in allowed}]
                if bad:
                    issues["invalid_mutability_by_rule"] = [b.get("key") for b in bad]

    # 3) token_name_regex
    for r in rules:
        if (r.get("type") or "") == "token_name_regex":
            pattern = (r.get("params") or {}).get("pattern")
            if pattern:
                try:
                    rx = re.compile(pattern)
                    bad = [t.get("key") for t in tokens if not rx.match(t.get("key") or "")]
                    if bad:
                        issues.setdefault("naming_mismatches", []).extend(bad)
                except re.error:
                    issues.setdefault("rule_errors", []).append(f"Invalid regex: {pattern}")

    # 4) palette_size
    for r in rules:
        if (r.get("type") or "") == "palette_size":
            params = r.get("params") or {}
            name = params.get("name")  # optional
            min_size = int(params.get("min", 0))
            max_size = int(params.get("max", 10**6))
            rows = db.execute_query(
                """
                MATCH (p:Palette)
                OPTIONAL MATCH (p)-[:CONTAINS]->(t:BrandToken)
                WITH p, count(t) as size
                RETURN p.name AS name, size AS size
                """
            )
            viol = []
            for row in rows:
                if name and row["name"] != name:
                    continue
                if row["size"] < min_size or row["size"] > max_size:
                    viol.append({"palette": row["name"], "size": row["size"], "min": min_size, "max": max_size})
            if viol:
                issues.setdefault("palette_size_violations", []).extend(viol)

    ok = not (invalid_hex or invalid_mutability or issues)

    return {
        "counts": role_counts,
        "invalid_hex": invalid_hex,
        "invalid_mutability": invalid_mutability,
        "rules_summary": rules,
        "issues": issues,
        "ok": ok,
    }


def add_rule_presets() -> List[Dict[str, Any]]:
    presets = [
        {"id": "roles-required", "type": "require_roles", "description": "Require core roles", "params": {"roles": ["primary", "secondary", "tertiary"]}},
        {"id": "mutability-default", "type": "mutability_allowed", "description": "Allowed mutability levels", "params": {"allowed": ["fixed", "flexible", "adaptive"]}},
        {"id": "naming-convention-1", "type": "token_name_regex", "description": "Token names like role-variant", "params": {"pattern": r"^(primary|secondary|tertiary)(-[a-z0-9]+)?$"}},
        {"id": "palette-size-default", "type": "palette_size", "description": "Palettes 3-8 tokens", "params": {"min": 3, "max": 8}},
    ]
    out = []
    for p in presets:
        out.append(
            upsert_design_rule(p["id"], p["description"], p["type"], p["params"])  # type: ignore
        )
    return out
