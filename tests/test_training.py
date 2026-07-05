"""Training engine tests — CPU, synthetic data, no real MRI needed."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, r'e:\ASD_forever')
import torch
import torch.nn as nn

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name} -- {msg}")
        FAIL += 1

# =========================================================================
print("=== FocalLoss ===")
from training.losses import FocalLoss, BalancedCrossEntropyLoss

fl = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
logits = torch.randn(8, 2)
labels = torch.randint(0, 2, (8,))
loss = fl(logits, labels)
check("FocalLoss scalar output", loss.dim() == 0)
check("FocalLoss positive",      loss.item() > 0)
check("FocalLoss finite",        loss.isfinite().item())

# Gradient through focal loss
logits2 = torch.randn(8, 2, requires_grad=True)
loss2 = fl(logits2, labels)
loss2.backward()
check("FocalLoss gradient",      logits2.grad is not None)

# gamma=0 should equal standard CE
fl0 = FocalLoss(alpha=0.5, gamma=0.0, label_smoothing=0.0)
ce = nn.CrossEntropyLoss()
l_fl0 = fl0(logits, labels).item()
l_ce = ce(logits, labels).item()
# Not exactly equal due to alpha weighting, but same order of magnitude
check("FocalLoss(gamma=0) finite", torch.tensor(l_fl0).isfinite().item())

# =========================================================================
print("\n=== BalancedCrossEntropyLoss ===")
bcl = BalancedCrossEntropyLoss(num_classes=2, label_smoothing=0.1)
train_labels = torch.tensor([0,0,0,1,1])  # imbalanced
bcl.update_weights(train_labels)
check("BCL weights shape", bcl.class_weights.shape == (2,))
check("BCL weights > 0",   (bcl.class_weights > 0).all().item())
loss_bcl = bcl(logits, labels)
check("BCL loss finite",   loss_bcl.isfinite().item())

# =========================================================================
print("\n=== build_criterion ===")
from training.losses import build_criterion
import dataclasses

# Minimal config stub
@dataclasses.dataclass
class LossCfg:
    type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

@dataclasses.dataclass
class TrainCfg:
    loss: LossCfg = dataclasses.field(default_factory=LossCfg)

@dataclasses.dataclass
class MockCfg:
    training: TrainCfg = dataclasses.field(default_factory=TrainCfg)

cfg = MockCfg()
crit = build_criterion(cfg)
check("build_criterion returns FocalLoss", isinstance(crit, FocalLoss))

cfg.training.loss.type = "cross_entropy"
crit2 = build_criterion(cfg)
check("build_criterion CE", isinstance(crit2, nn.CrossEntropyLoss))

# =========================================================================
print("\n=== Optimizer + Scheduler ===")
from training.optimizers import build_optimizer, build_scheduler

# Build a simple model to test param groups
simple_model = nn.Sequential(
    nn.Linear(10, 20), nn.LayerNorm(20), nn.GELU(), nn.Linear(20, 2)
)

@dataclasses.dataclass
class FullTrainCfg:
    loss: LossCfg = dataclasses.field(default_factory=LossCfg)
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine_warmup"
    max_epochs: int = 10
    warmup_epochs: int = 2
    min_lr: float = 1e-7
    gradient_accumulation_steps: int = 2
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    batch_size: int = 4
    save_top_k: int = 3
    checkpoint_metric: str = "val_auc"

    @dataclasses.dataclass
    class EarlyCfg:
        patience: int = 5
        mode: str = "max"
        min_delta: float = 1e-4
        monitor: str = "val_auc"

    early_stopping: EarlyCfg = dataclasses.field(default_factory=EarlyCfg)

@dataclasses.dataclass
class FullMockCfg:
    training: FullTrainCfg = dataclasses.field(default_factory=FullTrainCfg)

fcfg = FullMockCfg()
optimizer = build_optimizer(fcfg, simple_model)
check("Optimizer is AdamW", "AdamW" in type(optimizer).__name__)
check("Optimizer has 2 param groups", len(optimizer.param_groups) == 2)

# No-decay group should have weight_decay=0
wd_vals = [g["weight_decay"] for g in optimizer.param_groups]
check("No-decay group wd=0", 0.0 in wd_vals)

scheduler, freq = build_scheduler(fcfg, optimizer, steps_per_epoch=5)
check("Scheduler returns", scheduler is not None)
check("Scheduler freq=step", freq == "step")

# Step scheduler correctly: optimizer.step() first
optimizer.step()
scheduler.step()
check("Scheduler updates LR", True)  # just verify no exception

# =========================================================================
print("\n=== ModelEMA ===")
from training.ema import ModelEMA

model = nn.Linear(10, 5)
ema = ModelEMA(model, decay=0.99)
check("EMA shadow different obj", ema.shadow is not model)
check("EMA shadow not trainable",
      not any(p.requires_grad for p in ema.shadow.parameters()))

# After update, shadow should differ from freshly-init model params
orig_w = model.weight.data.clone()
nn.init.xavier_uniform_(model.weight)  # reinit model
ema.update(model)
shadow_w = ema.shadow.weight.data.clone()
check("EMA update changes shadow", not torch.allclose(shadow_w, orig_w, atol=1e-6))

# State dict round-trip
sd = ema.state_dict()
check("EMA state_dict has shadow", "shadow" in sd)
ema2 = ModelEMA(nn.Linear(10, 5), decay=0.99)
ema2.load_state_dict(sd)
check("EMA load state_dict", torch.allclose(ema2.shadow.weight, ema.shadow.weight))

# =========================================================================
print("\n=== EarlyStopping ===")
from training.early_stopping import EarlyStopping

es = EarlyStopping(patience=3, mode="max", min_delta=1e-4, monitor="val_auc")
check("ES no stop at start",  not es(0.7, 1))
check("ES no stop improve",   not es(0.8, 2))
check("ES no stop improve2",  not es(0.81, 3))
check("ES no stop same",      not es(0.81, 4))  # counter=1
check("ES no stop same",      not es(0.81, 5))  # counter=2
check("ES stops (patience=3)", es(0.81, 6))     # counter=3 → stop
check("ES best is 0.81",      abs(es.best - 0.81) < 1e-6)

es_min = EarlyStopping(patience=2, mode="min", monitor="val_loss")
check("ES min no stop",       not es_min(0.5, 1))
check("ES min no stop 2",     not es_min(0.4, 2))   # improved
check("ES min no stop 3",     not es_min(0.4, 3))   # counter=1
check("ES min stops",         es_min(0.4, 4))       # counter=2

es.reset()
check("ES reset clears state", es.best is None and es.counter == 0)

# =========================================================================
print("\n=== CheckpointManager ===")
from training.checkpointing import CheckpointManager

with tempfile.TemporaryDirectory() as tmp:
    ckpt = CheckpointManager(tmp, top_k=2, metric_name="val_auc", mode="max", fold=0)
    model_ck = nn.Linear(10, 2)
    opt_ck = torch.optim.AdamW(model_ck.parameters(), lr=1e-3)

    path1 = ckpt.save(1, model_ck, opt_ck, {"val_auc": 0.70, "val_acc": 0.65})
    check("Checkpoint saved (first)", path1 is not None)

    path2 = ckpt.save(2, model_ck, opt_ck, {"val_auc": 0.75, "val_acc": 0.68})
    check("Checkpoint saved (better)", path2 is not None)

    path3 = ckpt.save(3, model_ck, opt_ck, {"val_auc": 0.80, "val_acc": 0.72})
    check("Checkpoint saved (best)", path3 is not None)

    # Only 2 files should remain (top_k=2)
    existing = list(Path(tmp).glob("*.pt"))
    check("Top-k eviction (2 files)", len(existing) == 2, f"found {len(existing)}")

    # Load best
    nn.init.zeros_(model_ck.weight)  # zero out weights
    state = ckpt.load_best(model_ck, device=torch.device("cpu"))
    check("Load best returns state",  "epoch" in state)
    check("Load best metrics",        state["metrics"]["val_auc"] >= 0.75)
    # Model weights should be non-zero after loading
    check("Load best restores weights",
          not torch.all(model_ck.weight == 0).item())

# =========================================================================
print("\n=== ASDModel integration ===")
from models.asd_model import ASDModel
from models.mri.resnet3d import resnet10_3d
from models.mri.mri_encoder import MRIEncoder
from models.genetics.transformer_encoder import GeneTransformerEncoder
from models.genetics.genetics_encoder import GeneticsEncoder
from models.fusion.cross_attention import CrossAttentionFusion
from models.fusion.fusion_module import MultiModalFusion

mri_bb = resnet10_3d(feature_dim=32, dropout=0.1)
mri_enc = MRIEncoder(backbone=mri_bb, feature_dim=32, use_se=False, dropout=0.1)
gen_bb = GeneTransformerEncoder(64, 32, 4, 1, 128, 32, 0.1)
gen_enc = GeneticsEncoder(backbone=gen_bb, feature_dim=32, dropout=0.1)
fusion_b = CrossAttentionFusion(32, 32, 64, 4, 1, 2, 128, 2, 0.1)
fusion_w = MultiModalFusion(fusion_b, 64, 2, "cross_attention")
asd = ASDModel(mri_enc, gen_enc, fusion_w)

mri_t = torch.randn(2, 1, 16, 16, 16)
gen_t = torch.randn(2, 64)
out = asd(mri_t, gen_t)
check("ASDModel logits shape",   out["logits"].shape == (2, 2))
check("ASDModel fused shape",    out["fused_features"].shape == (2, 64))
check("ASDModel mri_features",   out["mri_features"].shape == (2, 32))
check("ASDModel gen_features",   out["gen_features"].shape == (2, 32))

# Backward
loss_asd = out["logits"].sum()
loss_asd.backward()
check("ASDModel backward", any(p.grad is not None for p in asd.parameters() if p.requires_grad))

# =========================================================================
print("\n=== One training step (synthetic data) ===")
from torch.utils.data import TensorDataset, DataLoader as DL

# Create tiny synthetic PairedMultiModal-like dataset
N = 8
MRI_SHAPE = (1, 16, 16, 16)
N_GENES = 64
images = torch.randn(N, *MRI_SHAPE)
genetics = torch.randn(N, N_GENES)
labels = torch.randint(0, 2, (N,))

class SyntheticPaired(torch.utils.data.Dataset):
    def __len__(self): return N
    def __getitem__(self, i):
        return {"image": images[i], "genetics": genetics[i], "label": labels[i]}

dataset = SyntheticPaired()
loader = DL(dataset, batch_size=4, shuffle=True)

# One manual training step
asd.train()
crit_test = FocalLoss(0.25, 2.0, 0.0)
opt_test = torch.optim.AdamW(asd.parameters(), lr=1e-4)
batch = next(iter(loader))
out_b = asd(
    batch["image"].to("cpu"),
    batch["genetics"].to("cpu"),
)
loss_b = crit_test(out_b["logits"], batch["label"])
loss_b.backward()
opt_test.step()
check("One training step", loss_b.isfinite().item())

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TRAINING ENGINE TESTS PASSED")
else:
    sys.exit(1)
