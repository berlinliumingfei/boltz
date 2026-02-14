from __future__ import annotations

from typing import Any, Literal, Optional

from pytorch_lightning import LightningModule
import torch
from torch import Tensor

from boltz.model.modules.diffusion import AtomDiffusion as AtomDiffusionV1
from boltz.model.modules.diffusionv2 import AtomDiffusion as AtomDiffusionV2


class DiffusionGenModel(LightningModule):
    """Standalone generative model built from the diffusion module only.

    This module decouples diffusion from trunk/confidence networks and consumes
    conditioning tensors provided directly in the batch.
    """

    def __init__(
        self,
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: Optional[dict[str, Any]] = None,
        optimizer_args: Optional[dict[str, Any]] = None,
        model_version: Literal["v1", "v2"] = "v1",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_version = model_version
        diffusion_cls = AtomDiffusionV1 if model_version == "v1" else AtomDiffusionV2
        self.diffusion = diffusion_cls(
            score_model_args=score_model_args,
            **diffusion_process_args,
        )

        self.diffusion_loss_args = diffusion_loss_args or {}
        self.optimizer_args = {"lr": 1e-4, **(optimizer_args or {})}

    def _extract_common_conditioning(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            "s_inputs": batch["s_inputs"],
            "s_trunk": batch["s_trunk"],
            "feats": batch["feats"],
        }

    def forward(
        self,
        batch: dict[str, Tensor],
        multiplicity: int = 1,
        num_sampling_steps: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Generate atom coordinates from diffusion noise.

        Batch inputs should provide external conditioning tensors. Required keys:
        - v1: ``s_inputs``, ``s_trunk``, ``z_trunk``, ``relative_position_encoding``, ``feats``
        - v2: ``s_inputs``, ``s_trunk``, ``diffusion_conditioning``, ``feats``
        """

        cond = self._extract_common_conditioning(batch)
        feats = cond["feats"]

        if self.model_version == "v1":
            return self.diffusion.sample(
                s_inputs=cond["s_inputs"],
                s_trunk=cond["s_trunk"],
                z_trunk=batch["z_trunk"],
                relative_position_encoding=batch["relative_position_encoding"],
                feats=feats,
                atom_mask=feats["atom_pad_mask"],
                multiplicity=multiplicity,
                num_sampling_steps=num_sampling_steps,
            )

        return self.diffusion.sample(
            s_inputs=cond["s_inputs"],
            s_trunk=cond["s_trunk"],
            diffusion_conditioning=batch["diffusion_conditioning"],
            feats=feats,
            atom_mask=feats["atom_pad_mask"],
            multiplicity=multiplicity,
            num_sampling_steps=num_sampling_steps,
        )

    def training_step(self, batch: dict[str, Tensor], _: int) -> Tensor:
        multiplicity = int(batch.get("multiplicity", 1))
        cond = self._extract_common_conditioning(batch)

        if self.model_version == "v1":
            out = self.diffusion(
                s_inputs=cond["s_inputs"],
                s_trunk=cond["s_trunk"],
                z_trunk=batch["z_trunk"],
                relative_position_encoding=batch["relative_position_encoding"],
                feats=cond["feats"],
                multiplicity=multiplicity,
            )
        else:
            out = self.diffusion(
                s_inputs=cond["s_inputs"],
                s_trunk=cond["s_trunk"],
                feats=cond["feats"],
                diffusion_conditioning=batch["diffusion_conditioning"],
                multiplicity=multiplicity,
            )

        loss_dict = self.diffusion.compute_loss(
            cond["feats"],
            out,
            multiplicity=multiplicity,
            **self.diffusion_loss_args,
        )
        self.log("train/diffusion_loss", loss_dict["loss"], prog_bar=True)
        return loss_dict["loss"]

    def predict_step(self, batch: dict[str, Tensor], _: int) -> dict[str, Tensor]:
        multiplicity = int(batch.get("multiplicity", 1))
        num_sampling_steps = batch.get("num_sampling_steps")
        return self.forward(
            batch,
            multiplicity=multiplicity,
            num_sampling_steps=(
                int(num_sampling_steps) if num_sampling_steps is not None else None
            ),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_args)
