import copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision.models import convnext_tiny
import matplotlib

S = 192
TASKS = ["malaria", "anemia", "leukemia"]

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, channels, 1, bias=False))
    def forward(self, x):
        return x * torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True); mx = x.max(dim=1, keepdim=True)[0]
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction); self.spatial = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.spatial(self.channel(x))

class DiseaseTaskHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=2, dropout=0.30):
        super().__init__()
        self.cbam = CBAMBlock(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_channels, 256), nn.LayerNorm(256),
            nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(256, num_classes))
    def forward(self, features):
        return self.classifier(torch.flatten(self.pool(self.cbam(features)), 1))

class FedScreenMTL(nn.Module):
    def __init__(self, tasks=TASKS):
        super().__init__()
        self.tasks = list(tasks)
        self.backbone = convnext_tiny(weights=None).features
        self.heads = nn.ModuleDict({t: DiseaseTaskHead(768) for t in self.tasks})
        self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))
    def forward_all(self, x):
        f = self.backbone(x)
        return {t: h(f) for t, h in self.heads.items()}

def load_fp32(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = FedScreenMTL(); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck

def make_int8(m):
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(m).eval(), {nn.Linear}, dtype=torch.qint8)

def preprocess(pil):
    im = pil.convert("RGB").resize((S, S), Image.BILINEAR)
    arr = np.array(im, dtype=np.uint8)                      # copy -> writable (no NumPy warning)
    x = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)[None]
    x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-6)   # same per-image z-score as training
    return arr, x

@torch.no_grad()
def predict(model, x):
    model.eval()
    return {t: lo.float().softmax(1)[0, 1].item() for t, lo in model.forward_all(x).items()}

import cv2, matplotlib

# ---- new: percentile normalisation (kills the washed-out look) -------------
def _norm_pct(c, lo=2.0, hi=98.0):
    a, b = np.percentile(c, lo), np.percentile(c, hi)
    return np.clip((c - a) / max(b - a, 1e-8), 0, 1)


def _gauss(c, s=2.0):
    k = int(2 * round(2 * s) + 1)
    return cv2.GaussianBlur(c.astype(np.float32), (k, k), s)


# ---- Grad-CAM++ (kept for the "++" claim) + smoothing + percentile norm ----
def gradcam_pp(model, x, cls, tasks=None, smooth=2.0):
    tasks = list(tasks) if tasks is not None else TASKS
    model.eval()
    cams = {}
    with torch.enable_grad():
        feats = model.backbone(x)
        A = feats.detach()
        for t in tasks:
            score = model.heads[t](feats)[0, cls[t]]
            g = torch.autograd.grad(score, feats, retain_graph=True)[0]
            g2, g3 = g.pow(2), g.pow(2) * g
            alpha = g2 / (2 * g2 + A.sum((2, 3), keepdim=True) * g3 + 1e-7)
            w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
            cam = F.relu((w * A).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(S, S), mode="bilinear",
 align_corners=False)[0, 0]
            cams[t] = _gauss(_norm_pct(cam.detach().float().cpu().numpy()), smooth)
    return cams


# ---- HiResCAM: element-wise grad x activation -> FULL 192x192 detail -------
@torch.enable_grad()
def hirescam(model, x, task, cls=1):
    """Sharp, cell-level localisation. Same layer as Grad-CAM++ but keeps    per-pixel gradient detail instead of collapsing channels into6x6."""
    model.eval()
    feats = model.backbone(x)
    score = model.heads[task](feats)[0, cls]
    g = torch.autograd.grad(score, feats, retain_graph=False)[0]
    cam = F.relu((g * feats.detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(S, S), mode="bilinear",
                        align_corners=False)[0, 0]
    return _gauss(_norm_pct(cam.detach().float().cpu().numpy(), 2, 98), 1.2)


# ---- overlay: turbo colormap + alpha + bright contour edge ----------------
def overlay(arr, cam, alpha=0.55, cmap="turbo", contour=True, thr=0.55):
    heat = (matplotlib.colormaps[cmap](cam)[..., :3] * 255).astype(np.float32)
    out = arr.astype(np.float32) * (1 - alpha) + heat * alpha
    if contour:
        m = (cam > thr).astype(np.uint8) * 255
        edge = cv2.morphologyEx(cv2.Canny(m, 50, 150), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        out[edge > 0] = np.array([255, 255, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)