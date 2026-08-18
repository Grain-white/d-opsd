import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(destination),
    )
    print(path)


if __name__ == "__main__":
    main()
