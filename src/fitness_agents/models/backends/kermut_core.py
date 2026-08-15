"""Minimal Kermut computational core adapted from the official MIT implementation.

Upstream: https://github.com/petergroth/kermut
Pinned source commit: 7e9e2e62a59773f6cc8291d85e6d6006a41a6862
License: KERMUT_LICENSE.txt in this directory.

The mathematical kernels and GP layout follow the upstream implementation. The mutation-event
aggregation is expressed as explicit one-hot matrix multiplication so batches remain well-defined
when variants contain different numbers of substitutions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from gpytorch import Module
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(ALPHABET)}


class Tokenizer:
    def __call__(self, sequences: str | Sequence[str]) -> torch.LongTensor:
        single = isinstance(sequences, str)
        batch = [sequences] if single else list(sequences)
        if not batch or len({len(sequence) for sequence in batch}) != 1:
            raise ValueError("Kermut requires a non-empty batch of equal-length sequences")
        result = torch.zeros((len(batch), len(batch[0]), 20), dtype=torch.long)
        for row, sequence in enumerate(batch):
            for column, amino_acid in enumerate(sequence):
                try:
                    result[row, column, AA_TO_INDEX[amino_acid]] = 1
                except KeyError as error:
                    raise ValueError(
                        f"Kermut received non-canonical amino acid {amino_acid!r}"
                    ) from error
        flattened = result.reshape(len(batch), -1)
        return flattened[0] if single else flattened


def _hellinger_distance(probabilities: torch.Tensor) -> torch.Tensor:
    roots = torch.sqrt(probabilities)
    return torch.cdist(roots, roots, p=2.0) / torch.sqrt(
        torch.tensor(2.0, device=probabilities.device)
    )


class _MutationKernelBase(Kernel):
    def __init__(self, wild_type: torch.LongTensor) -> None:
        super().__init__()
        self.sequence_length = wild_type.numel() // 20
        wild_type = wild_type.reshape(self.sequence_length, 20)
        self.register_buffer("wild_type_tokens", torch.nonzero(wild_type)[:, 1])

    def mutation_indices(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = x1.reshape(-1, self.sequence_length, 20)
        x2 = x2.reshape(-1, self.sequence_length, 20)
        x1_tokens = torch.nonzero(x1)[:, 2].reshape(x1.size(0), -1)
        x2_tokens = torch.nonzero(x2)[:, 2].reshape(x2.size(0), -1)
        return (
            torch.argwhere(x1_tokens != self.wild_type_tokens),
            torch.argwhere(x2_tokens != self.wild_type_tokens),
            x1_tokens,
            x2_tokens,
            x1.size(0),
            x2.size(0),
        )

    @staticmethod
    def aggregate_mutation_events(
        event_kernel: torch.Tensor,
        x1_indices: torch.Tensor,
        x2_indices: torch.Tensor,
        batch1: int,
        batch2: int,
    ) -> torch.Tensor:
        if x1_indices.numel() == 0 or x2_indices.numel() == 0:
            return torch.zeros((batch1, batch2), device=event_kernel.device)
        left = torch.nn.functional.one_hot(
            x1_indices[:, 0], num_classes=batch1
        ).to(event_kernel.dtype)
        right = torch.nn.functional.one_hot(
            x2_indices[:, 0], num_classes=batch2
        ).to(event_kernel.dtype)
        return left.transpose(0, 1) @ event_kernel @ right


class SiteComparisonKernel(_MutationKernelBase):
    def __init__(
        self,
        wild_type: torch.LongTensor,
        conditional_probs: torch.Tensor,
        lengthscale: float,
    ) -> None:
        super().__init__(wild_type)
        self.register_buffer("hellinger", _hellinger_distance(conditional_probs))
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.tensor(lengthscale)))
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    def forward(self, x1_indices: torch.Tensor, x2_indices: torch.Tensor, **params):
        del params
        distances = self.hellinger[
            x1_indices[:, 1].unsqueeze(1), x2_indices[:, 1].unsqueeze(0)
        ]
        return torch.exp(-self.lengthscale * distances)


class ProbabilityKernel(_MutationKernelBase):
    def __init__(
        self,
        wild_type: torch.LongTensor,
        conditional_probs: torch.Tensor,
        lengthscale: float,
    ) -> None:
        super().__init__(wild_type)
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.tensor(lengthscale)))
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params):
        del params
        x1_indices, x2_indices, x1_tokens, x2_tokens, _, _ = self.mutation_indices(x1, x2)
        p1 = self.conditional_probs[
            x1_indices[:, 1], x1_tokens[x1_indices[:, 0], x1_indices[:, 1]]
        ]
        p2 = self.conditional_probs[
            x2_indices[:, 1], x2_tokens[x2_indices[:, 0], x2_indices[:, 1]]
        ]
        differences = torch.abs(torch.log(p1).unsqueeze(1) - torch.log(p2).unsqueeze(0))
        return torch.exp(-self.lengthscale * differences)


class DistanceKernel(_MutationKernelBase):
    def __init__(
        self,
        wild_type: torch.LongTensor,
        coords: torch.Tensor,
        lengthscale: float,
    ) -> None:
        super().__init__(wild_type)
        self.register_buffer("coords", coords)
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.tensor(lengthscale)))
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    def forward(self, x1_indices: torch.Tensor, x2_indices: torch.Tensor, **params):
        del params
        distances = torch.cdist(
            self.coords[x1_indices[:, 1]], self.coords[x2_indices[:, 1]], p=2.0
        )
        return torch.exp(-self.lengthscale * distances)


class StructureKernel(_MutationKernelBase):
    def __init__(
        self,
        wild_type: torch.LongTensor,
        conditional_probs: torch.Tensor,
        coords: torch.Tensor,
        *,
        h_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0,
        d_lengthscale: float = 1.0,
    ) -> None:
        super().__init__(wild_type)
        self.site = SiteComparisonKernel(wild_type, conditional_probs, h_lengthscale)
        self.probability = ProbabilityKernel(wild_type, conditional_probs, p_lengthscale)
        self.distance = DistanceKernel(wild_type, coords, d_lengthscale)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params):
        del params
        x1_indices, x2_indices, _, _, batch1, batch2 = self.mutation_indices(x1, x2)
        event_kernel = self.site(x1_indices, x2_indices)
        event_kernel = event_kernel * self.probability(x1, x2)
        event_kernel = event_kernel * self.distance(x1_indices, x2_indices)
        return self.aggregate_mutation_events(
            event_kernel, x1_indices, x2_indices, batch1, batch2
        )


class SequenceKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()
        self.base_kernel = RBFKernel()

    def forward(self, x1, x2, diag=False, **params):
        return self.base_kernel.forward(x1, x2, diag=diag, **params)

    @property
    def is_stationary(self) -> bool:
        return True


class CompositeKernel(Module):
    def __init__(
        self,
        wild_type: torch.LongTensor,
        conditional_probs: torch.Tensor,
        coords: torch.Tensor,
        *,
        composition: str = "weighted_sum",
        h_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0,
        d_lengthscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.structure_kernel = StructureKernel(
            wild_type,
            conditional_probs,
            coords,
            h_lengthscale=h_lengthscale,
            p_lengthscale=p_lengthscale,
            d_lengthscale=d_lengthscale,
        )
        self.sequence_kernel = SequenceKernel()
        self.composition = composition
        if composition == "weighted_sum":
            self.register_parameter("pi", torch.nn.Parameter(torch.tensor(0.5)))
            self.structure_kernel = ScaleKernel(self.structure_kernel)
        elif composition == "add":
            self.structure_kernel = ScaleKernel(self.structure_kernel)
            self.sequence_kernel = ScaleKernel(self.sequence_kernel)
        elif composition == "multiply":
            # Upstream calls ScaleKernel without its required base kernel in this optional branch.
            # Preserve the intended learned positive scale directly so config switching is usable.
            self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.tensor(0.0)))
            self.register_constraint("raw_outputscale", Positive())
        else:
            raise ValueError(f"Unknown Kermut kernel composition {composition!r}")

    @property
    def outputscale(self):
        if self.composition != "multiply":
            raise AttributeError("outputscale is only defined for multiply composition")
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def forward(self, x1, x2=None, **params):
        if x2 is None:
            x2 = x1
        x1_tokens, x1_embeddings = x1
        x2_tokens, x2_embeddings = x2
        structure = self.structure_kernel(x1_tokens, x2_tokens, **params)
        sequence = self.sequence_kernel(x1_embeddings, x2_embeddings, **params)
        if self.composition == "weighted_sum":
            weight = torch.sigmoid(self.pi)
            return structure * weight + sequence * (1 - weight)
        if self.composition == "add":
            return structure + sequence
        return self.outputscale * structure * sequence


class KermutGP(ExactGP):
    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        *,
        wild_type: torch.LongTensor,
        conditional_probs: torch.Tensor,
        coords: torch.Tensor,
        use_zero_shot_mean: bool = True,
        composition: str = "weighted_sum",
        h_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0,
        d_lengthscale: float = 1.0,
    ) -> None:
        super().__init__(train_inputs, train_targets, likelihood)
        self.covar_module = CompositeKernel(
            wild_type,
            conditional_probs,
            coords,
            composition=composition,
            h_lengthscale=h_lengthscale,
            p_lengthscale=p_lengthscale,
            d_lengthscale=d_lengthscale,
        )
        self.use_zero_shot_mean = use_zero_shot_mean
        self.mean_module = LinearMean(1, bias=True) if use_zero_shot_mean else ConstantMean()

    def forward(self, tokens, embeddings, zero_shot=None):
        mean_input = zero_shot if zero_shot is not None else tokens
        mean = self.mean_module(mean_input)
        covariance = self.covar_module((tokens, embeddings))
        return MultivariateNormal(mean, covariance)


def optimize_gp(
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    train_inputs,
    train_targets,
    *,
    learning_rate: float,
    n_steps: int,
) -> tuple[ExactGP, GaussianLikelihood]:
    gp.train()
    likelihood.train()
    marginal_likelihood = ExactMarginalLogLikelihood(likelihood, gp)
    optimizer = torch.optim.AdamW(gp.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = -marginal_likelihood(gp(*train_inputs), train_targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Kermut marginal likelihood became non-finite")
        loss.backward()
        optimizer.step()
    return gp, likelihood
