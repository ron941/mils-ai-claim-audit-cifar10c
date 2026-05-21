from __future__ import annotations

import torch


def tta_probabilities(model: torch.nn.Module, images: torch.Tensor, horizontal_flip: bool = True) -> torch.Tensor:
    logits = model(images)
    probs = torch.softmax(logits, dim=1)
    variants = 1
    if horizontal_flip:
        flipped = torch.flip(images, dims=[3])
        flipped_logits = model(flipped)
        probs = probs + torch.softmax(flipped_logits, dim=1)
        variants += 1
    return probs / variants

