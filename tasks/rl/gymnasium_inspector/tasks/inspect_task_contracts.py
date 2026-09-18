from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent


def main() -> None:
    files = sorted(ROOT.glob("GI-*.yaml"))

    print("=== GYMNASIUM / INSPECTOR TASK CONTRACT INVENTORY ===")
    print(f"Task files found: {len(files)}")
    print()

    if len(files) != 33:
        print("WARNING: expected 33 task files")

    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        print(f"--- {path.name} ---")

        if not isinstance(data, dict):
            print("INVALID: top-level YAML is not a mapping")
            print()
            continue

        print("Top-level keys:")
        for key in data:
            value = data[key]

            if isinstance(value, dict):
                print(f"  {key}: mapping")
                print(
                    "    fields: "
                    + ", ".join(str(k) for k in value.keys())
                )
            elif isinstance(value, list):
                print(
                    f"  {key}: list[{len(value)}]"
                )
            else:
                print(
                    f"  {key}: "
                    f"{type(value).__name__} = {value!r}"
                )

        print()

    print("=== CONTRACT INVENTORY COMPLETE ===")
    print("=== NO TASK FILES MODIFIED ===")
    print("=== TERMINAL REMAINS OPEN ===")


if __name__ == "__main__":
    main()
