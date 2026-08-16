"""Prepare GB1 Kermut structure resources: C-alpha coords and ProteinMPNN probabilities.

The four Wu 2016 sites (39, 40, 41, 54) are extracted from the isolated GB1 domain
structure PDB 1PGA. Site-only arrays work for both four-letter GB1 tables and the
265-aa FLIP fusion sequences via ``resource_positions``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from fitness_agents.config import project_root

AA3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
KERMUT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
GB1_SITES = (39, 40, 41, 54)
DEFAULT_PDB_URL = "https://files.rcsb.org/download/1PGA.pdb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ca_coords(pdb_path: Path, chain: str = "A") -> tuple[np.ndarray, str, np.ndarray]:
    residues: dict[int, tuple[str, np.ndarray]] = {}
    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            resseq = int(line[22:26])
            resname = line[17:20].strip()
            amino_acid = AA3TO1.get(resname)
            if amino_acid is None:
                continue
            coords = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            residues[resseq] = (amino_acid, coords)
    if not residues:
        raise ValueError(f"No C-alpha atoms found for chain {chain} in {pdb_path}")
    numbers = np.asarray(sorted(residues), dtype=np.int64)
    sequence = "".join(residues[number][0] for number in numbers)
    coords = np.stack([residues[number][1] for number in numbers]).astype(np.float32)
    return coords, sequence, numbers


def site_indices(residue_numbers: np.ndarray, sites: tuple[int, ...]) -> np.ndarray:
    number_to_row = {int(number): index for index, number in enumerate(residue_numbers.tolist())}
    missing = [site for site in sites if site not in number_to_row]
    if missing:
        raise ValueError(f"PDB is missing GB1 sites {missing}")
    return np.asarray([number_to_row[site] for site in sites], dtype=np.int64)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def ensure_pdb(path: Path) -> Path:
    if path.is_file():
        return path
    _download(DEFAULT_PDB_URL, path)
    return path


def proteinmpnn_conditional_probs(
    pdb_path: Path,
    *,
    proteinmpnn_dir: Path,
    chain: str,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    sys.path.insert(0, str(proteinmpnn_dir))
    import torch
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

    weights = proteinmpnn_dir / "vanilla_model_weights" / "v_48_020.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"ProteinMPNN weights not found: {weights}")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    hidden_dim = 128
    num_layers = 3
    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pdb_dict_list = parse_PDB(str(pdb_path), input_chain_list=[chain])
    if not pdb_dict_list:
        raise RuntimeError(f"ProteinMPNN parse_PDB returned no chains for {pdb_path}")
    device = torch.device("cpu")
    model.to(device)
    torch.manual_seed(seed)
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    collected = []
    with torch.no_grad():
        for sample_index in range(n_samples):
            (
                X,
                S,
                mask,
                _,
                chain_M,
                chain_encoding_all,
                _,
                _,
                _,
                _,
                chain_M_pos,
                omit_AA_mask,
                residue_idx,
                dihedral_mask,
                _,
                pssm_coef,
                pssm_bias,
                pssm_log_odds_all,
                bias_by_res_all,
                tied_beta,
            ) = tied_featurize(
                pdb_dict_list,
                device,
                None,
                None,
                None,
                None,
                None,
                None,
                ca_only=False,
            )
            del omit_AA_mask, dihedral_mask, pssm_coef, pssm_bias, pssm_log_odds_all
            del bias_by_res_all, tied_beta
            randn = torch.randn(chain_M.shape, device=device)
            log_probs = model.conditional_probs(
                X, S, mask, chain_M, residue_idx, chain_encoding_all, randn
            )
            collected.append(log_probs[0].detach().cpu().numpy())
            del sample_index
    stacked = np.stack(collected, axis=0)
    mean_log = stacked.mean(axis=0)
    probs = np.exp(mean_log).astype(np.float32)
    # Drop the terminal X token and keep Kermut's 20-letter alphabet order.
    letter_index = {letter: index for index, letter in enumerate(alphabet)}
    ordered = np.stack(
        [probs[:, letter_index[letter]] for letter in KERMUT_ALPHABET], axis=1
    )
    ordered = np.clip(ordered, 1e-12, None)
    ordered = ordered / ordered.sum(axis=1, keepdims=True)
    return ordered.astype(np.float32)


def find_proteinmpnn_dir(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/tmp/fitness-agents-third-party/ProteinMPNN-main"),
            Path("/tmp/fitness-agents-third-party/ProteinMPNN"),
            project_root() / "models" / "kermut" / "proteinmpnn",
            project_root() / "third_party" / "ProteinMPNN",
        ]
    )
    for candidate in candidates:
        utils = candidate / "protein_mpnn_utils.py"
        weights = candidate / "vanilla_model_weights" / "v_48_020.pt"
        if utils.is_file() and weights.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=root / "models/kermut/1PGA.pdb")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--output-dir", type=Path, default=root / "models/kermut")
    parser.add_argument("--proteinmpnn-dir", type=Path)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdb_path = ensure_pdb(args.pdb)
    coords, sequence, residue_numbers = parse_ca_coords(pdb_path, chain=args.chain)
    rows = site_indices(residue_numbers, GB1_SITES)
    site_coords = coords[rows]
    site_sequence = "".join(sequence[int(row)] for row in rows)
    if site_sequence != "VDGV":
        raise ValueError(
            f"Expected GB1 sites 39/40/41/54 to be VDGV; found {site_sequence!r} in {pdb_path}"
        )

    proteinmpnn_dir = find_proteinmpnn_dir(args.proteinmpnn_dir)
    if proteinmpnn_dir is None:
        raise FileNotFoundError(
            "ProteinMPNN installation not found. Clone dauparas/ProteinMPNN so that "
            "protein_mpnn_utils.py and vanilla_model_weights/v_48_020.pt are available."
        )
    domain_probs = proteinmpnn_conditional_probs(
        pdb_path,
        proteinmpnn_dir=proteinmpnn_dir,
        chain=args.chain,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    if domain_probs.shape[0] != coords.shape[0]:
        raise ValueError(
            f"ProteinMPNN length {domain_probs.shape[0]} does not match coords {coords.shape[0]}"
        )
    site_probs = domain_probs[rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain_coords_path = args.output_dir / "gb1_domain_coords.npy"
    domain_probs_path = args.output_dir / "gb1_domain_conditional_probs.npy"
    site_coords_path = args.output_dir / "gb1_sites_coords.npy"
    site_probs_path = args.output_dir / "gb1_sites_conditional_probs.npy"
    np.save(domain_coords_path, coords, allow_pickle=False)
    np.save(domain_probs_path, domain_probs, allow_pickle=False)
    np.save(site_coords_path, site_coords, allow_pickle=False)
    np.save(site_probs_path, site_probs, allow_pickle=False)
    manifest = {
        "pdb": str(pdb_path),
        "pdb_sha256": sha256_file(pdb_path),
        "chain": args.chain,
        "domain_sequence": sequence,
        "site_sequence": site_sequence,
        "sites": list(GB1_SITES),
        "proteinmpnn_dir": str(proteinmpnn_dir),
        "n_samples": args.n_samples,
        "files": {
            "domain_coords": str(domain_coords_path),
            "domain_conditional_probs": str(domain_probs_path),
            "site_coords": str(site_coords_path),
            "site_conditional_probs": str(site_probs_path),
        },
        "shapes": {
            "domain_coords": list(coords.shape),
            "domain_conditional_probs": list(domain_probs.shape),
            "site_coords": list(site_coords.shape),
            "site_conditional_probs": list(site_probs.shape),
        },
    }
    manifest_path = args.output_dir / "gb1_resources_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
