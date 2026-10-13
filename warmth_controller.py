"""Explicit Modal autoscaler controller; dry-run by default.

Run from Railway or a trusted scheduler twice daily, and again after deployments.
Example: python warmth_controller.py --profile active --apply
"""
import argparse

APP_NAME = "multi-model-endpoints"
PROFILES = {
    "active": {
        "gpt_oss_120b_vllm": (1, 2, 1800),
        "qwen36_35b_a3b": (1, 2, 1200),
    },
    "quiet": {
        "gpt_oss_120b_vllm": (0, 2, 1800),
        "qwen36_35b_a3b": (0, 2, 1200),
    },
}


def apply_profile(profile: str, apply: bool = False, environment: str | None = None):
    if profile not in PROFILES:
        raise ValueError("Unknown warm-capacity profile")
    changes = []
    for name, (minimum, maximum, idle) in PROFILES[profile].items():
        changes.append({
            "function": name,
            "min_containers": minimum,
            "max_containers": maximum,
            "scaledown_window": idle,
        })
    if not apply:
        return changes
    import modal
    for change in changes:
        function = modal.Function.from_name(
            APP_NAME, change["function"], environment_name=environment
        )
        function.update_autoscaler(
            min_containers=change["min_containers"],
            max_containers=change["max_containers"],
            scaledown_window=change["scaledown_window"],
        )
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--apply", action="store_true", help="Actually update Modal capacity")
    args = parser.parse_args()
    for item in apply_profile(args.profile, args.apply, args.environment):
        print(item)
    if not args.apply:
        print("DRY RUN ONLY: add --apply to change GPU capacity and incur charges.")


if __name__ == "__main__":
    main()
