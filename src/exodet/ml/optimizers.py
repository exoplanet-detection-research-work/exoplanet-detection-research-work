"""Optimizer registry (Module 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from exodet.exceptions import PipelineError
from exodet.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["OPTIMIZERS", "build_optimizer"]

OPTIMIZERS: Registry[object] = Registry("optimizer")


def _require_torch():
    import torch

    return torch


def _params(
    parameters: Iterable["torch.nn.Parameter"],
    lr: float,
    weight_decay: float,
    extra: dict[str, Any],
) -> tuple[Iterable["torch.nn.Parameter"], dict[str, Any]]:
    kwargs = {"lr": lr, **extra}
    if weight_decay > 0:
        kwargs["weight_decay"] = weight_decay
    return parameters, kwargs


@OPTIMIZERS.register("adamw")
class _AdamWBuilder:
    def __call__(
        self,
        parameters: Iterable["torch.nn.Parameter"],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **_: object,
    ) -> "torch.optim.Optimizer":
        torch = _require_torch()
        p, kwargs = _params(parameters, lr, weight_decay, {"betas": betas, "eps": eps})
        return torch.optim.AdamW(p, **kwargs)


@OPTIMIZERS.register("adam")
class _AdamBuilder:
    def __call__(
        self,
        parameters: Iterable["torch.nn.Parameter"],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **_: object,
    ) -> "torch.optim.Optimizer":
        torch = _require_torch()
        p, kwargs = _params(parameters, lr, weight_decay, {"betas": betas, "eps": eps})
        return torch.optim.Adam(p, **kwargs)


@OPTIMIZERS.register("sgd")
class _SgdBuilder:
    def __call__(
        self,
        parameters: Iterable["torch.nn.Parameter"],
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        nesterov: bool = True,
        **_: object,
    ) -> "torch.optim.Optimizer":
        torch = _require_torch()
        p, kwargs = _params(
            parameters, lr, weight_decay, {"momentum": momentum, "nesterov": nesterov}
        )
        return torch.optim.SGD(p, **kwargs)


@OPTIMIZERS.register("rmsprop")
class _RmspropBuilder:
    def __call__(
        self,
        parameters: Iterable["torch.nn.Parameter"],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        alpha: float = 0.99,
        eps: float = 1e-8,
        **_: object,
    ) -> "torch.optim.Optimizer":
        torch = _require_torch()
        p, kwargs = _params(parameters, lr, weight_decay, {"alpha": alpha, "eps": eps})
        return torch.optim.RMSprop(p, **kwargs)


@OPTIMIZERS.register("lion")
class _LionBuilder:
    """Lion optimizer (Chen et al. 2023) — sign-momentum update."""

    def __call__(
        self,
        parameters: Iterable["torch.nn.Parameter"],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.99),
        **_: object,
    ) -> "torch.optim.Optimizer":
        torch = _require_torch()

        class Lion(torch.optim.Optimizer):
            def __init__(self, params, lr, betas, weight_decay) -> None:
                defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):  # noqa: D102
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                for group in self.param_groups:
                    lr = group["lr"]
                    beta1, beta2 = group["betas"]
                    wd = group["weight_decay"]
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                        exp_avg = state["exp_avg"]
                        if wd != 0:
                            p.mul_(1.0 - lr * wd)
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        update = exp_avg.mul(beta2).add_(grad, alpha=1.0 - beta2)
                        p.add_(torch.sign(update), alpha=-lr)
                return loss

        return Lion(list(parameters), lr=lr, betas=betas, weight_decay=weight_decay)


def build_optimizer(
    name: str,
    parameters: Iterable["torch.nn.Parameter"],
    lr: float,
    **params: object,
) -> "torch.optim.Optimizer":
    """Builds an optimizer from the registry.

    Args:
        name: Registered optimizer name.
        parameters: Model parameters.
        lr: Learning rate from training config.
        **params: Optimizer-specific hyperparameters.

    Returns:
        The constructed optimizer.
    """
    builder_cls = OPTIMIZERS.get(name)
    builder = builder_cls() if isinstance(builder_cls, type) else builder_cls
    return builder(parameters, lr=lr, **params)  # type: ignore[operator]
