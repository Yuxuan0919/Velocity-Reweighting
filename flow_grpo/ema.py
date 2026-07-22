# Copied from another repo, but I can't remember exactly which one.

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step) -> float:
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        parameters = list(parameters)

        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        # in place calculations to save memory
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    @torch.no_grad()
    def sync_with_model(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Force the EMA parameters to be a direct copy of the given model parameters.
        This is used to create a snapshot for the rollout policy.
        """
        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            ema_parameter.data.copy_(parameter.detach().data)

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True, grad=False) -> None:
        if store_temp:
            if grad:
                self.temp_stored_parameters = [parameter.data.clone() for parameter in parameters]
            else:
                self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]

        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            # Ensure the temp parameter is on the right device
            parameter.data.copy_(temp_parameter.to(parameter.device))

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict, device: torch.device = None) -> None:
        """
        从状态字典恢复 EMA 影子参数。
        Args:
            state_dict: 包含 "decay" 和 "ema_parameters" 的字典
            device: 目标设备，默认为 self.device
        """
        self.decay = state_dict.get("decay", self.decay)
        
        # 关键修复：重新赋值 self.ema_parameters
        loaded_params = state_dict.get("ema_parameters")
        if loaded_params is not None:
            target_device = device if device is not None else self.device
            self.ema_parameters = [p.clone().detach().to(target_device) for p in loaded_params]
        else:
            raise KeyError("Missing 'ema_parameters' in state_dict")

        self.device = target_device if device is not None else self.device

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }
