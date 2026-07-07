"""Modality encoders shared by BC, PPO fine-tuning, and (later) diffusion."""

import torch
import torch.nn as nn

from xembench.constants import P_CANON


class NatureCNN(nn.Module):
    """DQN-style CNN sized for 128x128 RGB. Input uint8 [B,3,128,128]."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 3, 128, 128)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, out_dim), nn.ReLU())

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(rgb.float() / 255.0))


class ResNet18Encoder(nn.Module):
    """ImageNet-pretrained ResNet18, first two stages frozen. Heavier but
    stronger; select with vision='resnet18' in configs."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # -> [B,512,1,1]
        for module in list(self.backbone.children())[:5]:  # stem + layer1
            for p in module.parameters():
                p.requires_grad = False
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, out_dim), nn.ReLU())
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = (rgb.float() / 255.0 - self.mean) / self.std
        return self.fc(self.backbone(x))


VISION_ENCODERS = {"nature_cnn": NatureCNN, "resnet18": ResNet18Encoder}


class InstructionEncoder(nn.Module):
    """Embedding + single-layer LSTM; the final valid step's hidden state is
    the sentence embedding. Gather-based (no pack_padded) so it runs the same
    on cpu/mps/cuda."""

    def __init__(self, vocab_size: int, d_embed: int = 64, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.lstm = nn.LSTM(d_embed, out_dim, batch_first=True)

    def forward(self, tokens: torch.Tensor, token_len: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.embed(tokens))  # [B, L_MAX, out_dim]
        idx = (token_len.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        return out.gather(1, idx).squeeze(1)


class ProprioEncoder(nn.Module):
    """MLP over canonical proprio concatenated with its validity mask (the
    mask doubles as an embodiment cue)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * P_CANON, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )

    def forward(self, proprio: torch.Tensor, proprio_mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([proprio, proprio_mask], dim=-1))
