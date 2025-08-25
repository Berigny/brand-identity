import json
from pathlib import Path
from typing import List


BASE_SIM_PRIMES: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def get_high_prime_min_from_kernel(kernel_path: str = "brand_kernel.json") -> int:
    """Derive a simulator high-prime threshold from the kernel policy.

    - Reads the kernel JSON (float `prime` values per token).
    - Uses the lower bound of the 'flexible' band as the reference boundary.
    - Maps that boundary proportionally to the discrete simulator prime scale.
    - Never returns below 11 to preserve the simulator's notion of higher-level primes.
    """
    p = Path(kernel_path)
    if not p.exists():
        return 11
    data = json.loads(p.read_text())
    palette = (data.get("brand_palette") or {})
    primes = [float(v.get("prime", 0.0)) for v in palette.values() if isinstance(v, dict)]
    if not primes:
        return 11
    max_prime = max(primes) or 1.0
    policy = data.get("mutability_policy") or {}
    flexible = policy.get("flexible") or {}
    flex_low, _ = (flexible.get("prime_range") or [None, None])
    try:
        flex_low = float(flex_low)
    except Exception:
        # Default to ~current config if missing
        flex_low = 0.0125

    # Map boundary to simulator discrete primes by relative position
    ratio = max(0.0, min(1.0, flex_low / max_prime))
    idx = int(round(ratio * (len(BASE_SIM_PRIMES) - 1)))
    # Ensure we don't go below 11 for "higher-prime" boundary
    # Find position of 11 in the base list
    min_idx = BASE_SIM_PRIMES.index(11)
    idx = max(idx, min_idx)
    return BASE_SIM_PRIMES[idx]

