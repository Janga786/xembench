"""Download prebuilt ManiSkill demonstration datasets from HuggingFace.

    uv run python -m xembench.scripts.download_demos            # all hf_replay tasks
    uv run python -m xembench.scripts.download_demos PickCube-v1

Files land in ~/.maniskill/demos/<env_id>/.
"""

import subprocess
import sys

from xembench.utils.config import load_yaml, repo_root


def main() -> int:
    env_ids = sys.argv[1:]
    if not env_ids:
        tasks = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"]
        env_ids = [t["env_id"] for t in tasks.values() if t["demo_source"] == "hf_replay"]
        env_ids.append("PickCube-v1")  # pipeline-check dataset (A3)
    for env_id in env_ids:
        print(f"== downloading demos: {env_id}")
        result = subprocess.run(
            [sys.executable, "-m", "mani_skill.utils.download_demo", env_id]
        )
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
