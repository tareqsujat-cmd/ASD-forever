"""Full model architecture tests — CPU only, small volumes."""
import sys
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

B = 2  # batch size
SMALL = (32, 32, 32)  # tiny volume for speed
x = torch.randn(B, 1, *SMALL)

print("=== ResNet3D ===")
from models.mri.resnet3d import resnet10_3d, resnet50_3d

r10 = resnet10_3d(feature_dim=128, dropout=0.1)
r10.eval()
with torch.no_grad():
    out = r10(x)
check("ResNet10 output shape", out.shape == (B, 128), str(out.shape))

feat = r10.forward_features(x)
check("ResNet10 feature maps 5D", feat.dim() == 5, str(feat.shape))

r50 = resnet50_3d(feature_dim=128, dropout=0.1)
r50.eval()
with torch.no_grad():
    out = r50(x)
check("ResNet50 output shape", out.shape == (B, 128), str(out.shape))

print("\n=== DenseNet3D ===")
from models.mri.densenet3d import densenet121_3d
dn = densenet121_3d(feature_dim=128, dropout=0.1)
dn.eval()
with torch.no_grad():
    out = dn(x)
check("DenseNet121 output shape", out.shape == (B, 128), str(out.shape))
feat_dn = dn.forward_features(x)
check("DenseNet feature maps 5D", feat_dn.dim() == 5, str(feat_dn.shape))

print("\n=== Swin3D ===")
from models.mri.swin3d import swin3d_tiny, SwinTransformer3D
# Use patch_size=(2,2,2) for 32³ input to get enough tokens
swin = SwinTransformer3D(
    embed_dim=48, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
    window_size=(2, 2, 2), feature_dim=128, dropout=0.1,
    patch_size=(2, 2, 2),
)
swin.eval()
with torch.no_grad():
    out = swin(x)
check("Swin3D output shape", out.shape == (B, 128), str(out.shape))
feat_swin = swin.forward_features(x)
check("Swin3D feature maps", feat_swin.dim() == 5, str(feat_swin.shape))

print("\n=== ConvNeXt3D ===")
from models.mri.convnext3d import ConvNeXt3D
cnx = ConvNeXt3D(depths=[2, 2, 2, 2], dims=[32, 64, 128, 256],
                  feature_dim=128, dropout=0.1)
cnx.eval()
with torch.no_grad():
    out = cnx(x)
check("ConvNeXt3D output shape", out.shape == (B, 128), str(out.shape))

print("\n=== SEBlock3D ===")
from models.mri.mri_encoder import SEBlock3D
se = SEBlock3D(channels=64)
feat_in = torch.randn(2, 64, 8, 8, 8)
feat_out = se(feat_in)
check("SE output shape preserved", feat_out.shape == feat_in.shape)
check("SE scales (not trivially 1)", not torch.allclose(feat_out, feat_in))

print("\n=== MRIEncoder + MRIClassifier ===")
from models.mri.mri_encoder import MRIEncoder, MRIClassifier
enc = MRIEncoder(backbone=r10, feature_dim=128, use_se=False, dropout=0.1)
enc.eval()
with torch.no_grad():
    feats = enc(x)
check("MRIEncoder output shape", feats.shape == (B, 128), str(feats.shape))

clf = MRIClassifier(enc, num_classes=2, dropout=0.1)
clf.eval()
with torch.no_grad():
    result = clf(x)
check("MRIClassifier logits shape", result["logits"].shape == (B, 2))
check("MRIClassifier features shape", result["features"].shape == (B, 128))

print("\n=== Gradient flow (backward pass) ===")
enc_train = MRIEncoder(backbone=resnet10_3d(feature_dim=64), feature_dim=64,
                        use_se=False, dropout=0.1)
clf_train = MRIClassifier(enc_train, num_classes=2)
x_train = torch.randn(2, 1, *SMALL, requires_grad=False)
labels = torch.tensor([0, 1])
out = clf_train(x_train)
loss = nn.CrossEntropyLoss()(out["logits"], labels)
loss.backward()
grads_ok = all(
    p.grad is not None and not torch.isnan(p.grad).any()
    for p in clf_train.parameters() if p.requires_grad and p.grad is not None
)
check("Backward pass no NaN gradients", grads_ok)
check("Loss is finite", loss.item() == loss.item() and loss.item() < 100)

print("\n=== Parameter counts ===")
from utilities.hardware import count_parameters
for name, model in [("ResNet10", r10), ("ResNet50", r50),
                     ("DenseNet121", dn), ("Swin3D", swin), ("ConvNeXt3D", cnx)]:
    total, trainable = count_parameters(model)
    check(f"{name} has params", trainable > 0, f"{trainable:,}")

print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL MRI MODEL TESTS PASSED")
else:
    import sys; sys.exit(1)
