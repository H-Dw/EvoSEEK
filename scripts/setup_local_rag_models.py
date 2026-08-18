from __future__ import annotations

import argparse
from pathlib import Path

MODELS = {
    "bge-small-en-v1.5": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "relative_path": "models/embeddings/bge-small-en-v1.5",
    },
    "medcpt-cross-encoder": {
        "repo_id": "ncbi/MedCPT-Cross-Encoder",
        "revision": "71caf65d4927987813984f54c284405a13fcca49",
        "relative_path": "models/rerankers/medcpt-cross-encoder",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision pinned local RAG models before campaign runtime"
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODELS),
        default="bge-small-en-v1.5",
        help="Pinned model profile to download",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Install the local RAG dependencies first: pip install -e .[rag]"
        ) from error

    model = MODELS[args.model]
    destination = (args.project_root / str(model["relative_path"])).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(model["repo_id"]),
        revision=str(model["revision"]),
        local_dir=destination,
    )
    print(f"Provisioned {model['repo_id']} at {destination}")
    print(f"Pinned revision: {model['revision']}")


if __name__ == "__main__":
    main()
