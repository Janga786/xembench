#!/usr/bin/env python
"""Environment report: what machine is this, and is it ready for which phase?

    uv run python scripts/check_env.py        (or: make mac-check)
"""

import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MOLTENVK_ICD = "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    problems: list[str] = []
    is_mac = platform.system() == "Darwin"

    section("System")
    print(f"OS:      {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:  {sys.version.split()[0]} ({sys.executable})")

    section("PyTorch")
    try:
        import torch

        print(f"torch:   {torch.__version__}")
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        print(f"backends: cpu=yes  mps={'yes' if mps else 'no'}  "
              f"cuda={'yes' if cuda else 'no'}")
        if cuda:
            cap = torch.cuda.get_device_capability()
            print(f"cuda:    {torch.version.cuda}, device "
                  f"{torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]}")
            if cap < (12, 0):
                print("note:    below sm_120 — fine for any CUDA GPU, "
                      "the cu128 wheels target the RTX 5090")
    except ImportError as e:
        problems.append(f"torch not importable: {e}")

    section("Simulator")
    try:
        import sapien

        print(f"sapien:      {sapien.__version__}")
    except ImportError as e:
        problems.append(f"sapien not importable: {e}")
    try:
        import mani_skill

        print(f"mani_skill:  {mani_skill.__version__}")
    except ImportError as e:
        problems.append(f"mani_skill not importable: {e}")
    try:
        import mplib  # noqa: F401

        print("mplib:       available (motion-planning demo collection OK)")
    except ImportError:
        print("mplib:       NOT available"
              + (" (expected on macOS — Linux-only dependency)" if is_mac else
                 " — arm demo collection (B1) needs it; check uv sync"))

    section("Vulkan / rendering")
    if is_mac:
        icd_env = os.environ.get("VK_ICD_FILENAMES")
        print(f"VK_ICD_FILENAMES: {icd_env or '(unset — xembench sets it on import)'}")
        if Path(MOLTENVK_ICD).exists():
            print(f"MoltenVK ICD:     {MOLTENVK_ICD} (present)")
        else:
            problems.append("MoltenVK missing — `brew install molten-vk` "
                            "(rendering will fail without it)")
        drv = os.environ.get("VK_DRIVER_FILES")
        if drv == "":
            problems.append("VK_DRIVER_FILES is set but EMPTY — this breaks "
                            "the Vulkan loader; unset it or point it at the ICD")
    else:
        for p in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
            if Path(p).exists():
                print(f"ICD dir: {p} -> {sorted(x.name for x in Path(p).iterdir())}")

    section("Package")
    try:
        import xembench
        from xembench.constants import A_CANON, CANON_ACTION_DIM_NAMES

        print(f"xembench {xembench.__version__} importable; "
              f"canonical dims = {A_CANON} ({len(CANON_ACTION_DIM_NAMES)} named)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"xembench import failed: {e}")

    section("Phase readiness")
    if is_mac:
        print("macOS: Phase A dev/smoke machine. CPU sim + MPS training only.")
        print("  OK here:   make test / make test-sim / make smoke / make toy-report")
        print("  NOT here:  demo collection at scale, expert/finetune training,")
        print("             full eval matrix -> Linux RTX 5090 (docs/phase_b_commands.md)")
    else:
        print("Linux: Phase B machine. Run ./scripts/setup_linux.sh, then")
        print("       follow docs/phase_b_commands.md in order.")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
