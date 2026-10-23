from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int = 100,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        padding: str = "center",
    ) -> None:
        super().__init__()
        if padding not in ("center", "same"):
            raise ValueError("padding must be 'center' or 'same'")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )

        pad = (self.win_length - self.hop_length) // 2
        if spec.dim() != 3:
            raise ValueError("Expected complex spectrogram with shape [B, F, T]")

        _, _, frames = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        output_size = (frames - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        window_sq = self.window.square().expand(1, frames, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]
        return y / window_envelope


class ISTFTHead(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        padding: str = "center",
    ) -> None:
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return self.istft(spec)


class LocalVocos(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = VocosBackbone()
        self.head = ISTFTHead()

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor) -> torch.Tensor:
        x = self.backbone(features_input)
        return self.head(x)
