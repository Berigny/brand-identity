import json
import sys
from pathlib import Path


def load_kernel(path: Path):
    data = json.loads(path.read_text())
    palette = data.get("brand_palette", {})
    policy = data.get("mutability_policy", {})
    bands = {
        name: tuple(policy.get(name, {}).get("prime_range", [None, None]))
        for name in ("protected", "adaptive", "flexible")
    }
    issues = []
    for key, tok in palette.items():
        prime = tok.get("prime")
        level = (tok.get("mutability_level") or "").lower()
        if prime is None:
            issues.append(f"{key}: missing prime")
            continue
        def in_band(pr, band):
            lo, hi = band
            if lo is None or hi is None:
                return False
            return (lo <= pr <= hi)
        expected = None
        for name, band in bands.items():
            if in_band(prime, band):
                expected = name
                break
        if expected is None:
            issues.append(f"{key}: prime {prime} falls in no policy band")
        elif level and level != expected:
            issues.append(f"{key}: mutability_level '{level}' != expected '{expected}' (prime={prime})")
    return issues


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "brand_kernel.json")
    try:
        issues = load_kernel(path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)
    if issues:
        print("Validation issues:")
        print("- " + "\n- ".join(issues))
        sys.exit(1)
    print("Brand kernel OK: primes align with mutability policy")


if __name__ == "__main__":
    main()

