import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import distance_transform_edt, binary_erosion


# ==========================================
# 1. METRICS & DICE LOSS
# ==========================================

def weighted_dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    intersection = torch.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth / 2) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth
    )


class DiceLoss(nn.Module):
    """
    Dice Loss:
    تعمل على logits مباشرة
    لذلك نستخدم sigmoid داخل الدالة.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred_logits, y_true):
        y_pred = torch.sigmoid(y_pred_logits)

        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)

        intersection = torch.sum(y_true_f * y_pred_f)

        dice = (2.0 * intersection + self.smooth) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth
        )

        return 1.0 - dice


def calculate_sensitivity(y_true, y_pred, smooth=1.):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    tp = torch.sum(y_true_f * y_pred_f)
    fn = torch.sum(y_true_f * (1 - y_pred_f))

    return (tp + smooth) / (tp + fn + smooth)


def calculate_specificity(y_true, y_pred, smooth=1.):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    tn = torch.sum((1 - y_true_f) * (1 - y_pred_f))
    fp = torch.sum((1 - y_true_f) * y_pred_f)

    return (tn + smooth) / (tn + fp + smooth)


def hd95_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    if not np.any(y_true) and not np.any(y_pred):
        return 0.0

    if not np.any(y_true) or not np.any(y_pred):
        return np.nan

    true_border = y_true ^ binary_erosion(y_true)
    pred_border = y_pred ^ binary_erosion(y_pred)

    dt_true = distance_transform_edt(~true_border)
    dt_pred = distance_transform_edt(~pred_border)

    distances_1 = dt_pred[true_border]
    distances_2 = dt_true[pred_border]

    all_distances = np.concatenate([distances_1, distances_2])

    return np.percentile(all_distances, 95)


# ==========================================
# 2. MSH-TRANS ARCHITECTURE
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        hidden = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        return x * self.sigmoid(avg_out + max_out)


class EfficientTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(EfficientTransformerBlock, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        res = x

        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = self.norm(x)

        qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: t.view(b, h * w, self.heads, -1).transpose(1, 2),
            qkv
        )

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, context)

        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.to_out(out)

        out = out.permute(0, 2, 1).view(b, c, h, w)

        return out + res


class MSHTrans(nn.Module):
    def __init__(self, in_channels=4, base_filters=32):
        super(MSHTrans, self).__init__()

        bf = base_filters

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.InstanceNorm2d(bf),
            nn.ReLU(inplace=True)
        )

        self.enc2 = self._make_layer(bf, bf * 2)
        self.enc3 = self._make_layer(bf * 2, bf * 4)
        self.enc4 = self._make_layer(bf * 4, bf * 8)

        self.bottleneck = EfficientTransformerBlock(bf * 8)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = self._make_layer(bf * 12, bf * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = self._make_layer(bf * 6, bf * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = self._make_layer(bf * 3, bf)

        self.ca3 = ChannelAttention(bf * 4)
        self.ca2 = ChannelAttention(bf * 2)
        self.ca1 = ChannelAttention(bf)

        self.head_ct = nn.Conv2d(bf, 1, kernel_size=1)
        self.head_et = nn.Conv2d(bf, 1, kernel_size=1)
        self.head_wt = nn.Conv2d(bf, 1, kernel_size=1)

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))

        b = self.bottleneck(s4)

        d3 = self.dec3(torch.cat([self.up4(b), self.ca3(s3)], dim=1))
        d2 = self.dec2(torch.cat([self.up3(d3), self.ca2(s2)], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), self.ca1(s1)], dim=1))

        return self.head_ct(d1), self.head_et(d1), self.head_wt(d1)


# ==========================================
# 3. DATASET
# ==========================================

class SingleDataDataset(Dataset):
    def __init__(
        self,
        list_IDs,
        data_dir,
        data_dir_CT_ch,
        data_dir_ET_ch,
        data_dir_WT_ch,
        dim=(160, 160),
        n_channels=4,
        shuffle=True
    ):
        self.list_IDs = list_IDs
        self.data_dir = Path(data_dir)
        self.data_dir_CT_ch = Path(data_dir_CT_ch)
        self.data_dir_ET_ch = Path(data_dir_ET_ch)
        self.data_dir_WT_ch = Path(data_dir_WT_ch)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        X = pickle.load(open(self.data_dir / f"{ID}_image.pkl", "rb")).astype(np.float32)

        CT = pickle.load(open(self.data_dir_CT_ch / f"{ID}_TC.pkl", "rb")).astype(np.float32)
        ET = pickle.load(open(self.data_dir_ET_ch / f"{ID}_ET.pkl", "rb")).astype(np.float32)
        WT = pickle.load(open(self.data_dir_WT_ch / f"{ID}_WT.pkl", "rb")).astype(np.float32)

        X = np.transpose(X, (2, 0, 1))
        CT = np.transpose(CT, (2, 0, 1))
        ET = np.transpose(ET, (2, 0, 1))
        WT = np.transpose(WT, (2, 0, 1))

        return X, CT, ET, WT


def read_data_from_file(file_name):
    with open(file_name, "r") as file:
        data = file.readlines()

    return [item.strip() for item in data]


# ==========================================
# 4. TRAINING AND TESTING
# ==========================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = Path("./CodeData_Wisam/image/")
    data_dir_CT_ch = Path("./CodeData_Wisam/seg_TC/")
    data_dir_ET_ch = Path("./CodeData_Wisam/seg_ET/")
    data_dir_WT_ch = Path("./CodeData_Wisam/seg_WT/")

    weight_dir = Path("./CodeData_Wisam/MSHTrans_DiceLoss_Adam_HD95/")
    weight_dir.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": read_data_from_file("./CodeData_Wisam/train.txt"),
        "validation": read_data_from_file("./CodeData_Wisam/validation.txt"),
        "test": read_data_from_file("./CodeData_Wisam/test.txt")
    }

    params = {
        "data_dir": data_dir,
        "data_dir_CT_ch": data_dir_CT_ch,
        "data_dir_ET_ch": data_dir_ET_ch,
        "data_dir_WT_ch": data_dir_WT_ch
    }

    train_dataset = SingleDataDataset(partition["train"], **params, shuffle=True)
    val_dataset = SingleDataDataset(partition["validation"], **params, shuffle=False)
    test_dataset = SingleDataDataset(partition["test"], **params, shuffle=False)

    batch_size = 40

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = MSHTrans().to(device)

    # ==========================================
    # DICE LOSS PER CLASS
    # ==========================================

    criterion_ct = DiceLoss().to(device)
    criterion_et = DiceLoss().to(device)
    criterion_wt = DiceLoss().to(device)

    # ==========================================
    # ADAM OPTIMIZER
    # ==========================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10
    )

    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 10
    epochs = 100

    history = {
        "epoch": [],

        "train_loss": [],
        "val_loss": [],

        "train_loss_CT": [],
        "train_loss_ET": [],
        "train_loss_WT": [],

        "val_loss_CT": [],
        "val_loss_ET": [],
        "val_loss_WT": []
    }

    print("Starting Training — MSH-Trans with Dice Loss + Adam + HD95")

    for epoch in range(epochs):

        model.train()

        tr = {
            "total": 0.0,
            "CT": 0.0,
            "ET": 0.0,
            "WT": 0.0
        }

        nb = 0

        for X, CT, ET, WT in train_loader:
            X = X.to(device)
            CT = CT.to(device)
            ET = ET.to(device)
            WT = WT.to(device)

            optimizer.zero_grad()

            p_ct, p_et, p_wt = model(X)

            loss_ct = criterion_ct(p_ct, CT)
            loss_et = criterion_et(p_et, ET)
            loss_wt = criterion_wt(p_wt, WT)

            loss = loss_ct + loss_et + loss_wt

            loss.backward()
            optimizer.step()

            tr["total"] += loss.item()
            tr["CT"] += loss_ct.item()
            tr["ET"] += loss_et.item()
            tr["WT"] += loss_wt.item()

            nb += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_loss = tr["total"] / nb
        epoch_loss_ct = tr["CT"] / nb
        epoch_loss_et = tr["ET"] / nb
        epoch_loss_wt = tr["WT"] / nb

        model.eval()

        vl = {
            "total": 0.0,
            "CT": 0.0,
            "ET": 0.0,
            "WT": 0.0
        }

        nv = 0

        with torch.no_grad():
            for X, CT, ET, WT in val_loader:
                X = X.to(device)
                CT = CT.to(device)
                ET = ET.to(device)
                WT = WT.to(device)

                p_ct, p_et, p_wt = model(X)

                loss_ct = criterion_ct(p_ct, CT)
                loss_et = criterion_et(p_et, ET)
                loss_wt = criterion_wt(p_wt, WT)

                loss = loss_ct + loss_et + loss_wt

                vl["total"] += loss.item()
                vl["CT"] += loss_ct.item()
                vl["ET"] += loss_et.item()
                vl["WT"] += loss_wt.item()

                nv += 1

        val_loss = vl["total"] / nv
        val_loss_ct = vl["CT"] / nv
        val_loss_et = vl["ET"] / nv
        val_loss_wt = vl["WT"] / nv

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"(CT {epoch_loss_ct:.4f} | ET {epoch_loss_et:.4f} | WT {epoch_loss_wt:.4f}) "
            f"Val Loss: {val_loss:.4f} "
            f"(CT {val_loss_ct:.4f} | ET {val_loss_et:.4f} | WT {val_loss_wt:.4f})"
        )

        history["epoch"].append(epoch + 1)

        history["train_loss"].append(round(epoch_loss, 6))
        history["val_loss"].append(round(val_loss, 6))

        history["train_loss_CT"].append(round(epoch_loss_ct, 6))
        history["train_loss_ET"].append(round(epoch_loss_et, 6))
        history["train_loss_WT"].append(round(epoch_loss_wt, 6))

        history["val_loss_CT"].append(round(val_loss_ct, 6))
        history["val_loss_ET"].append(round(val_loss_et, 6))
        history["val_loss_WT"].append(round(val_loss_wt, 6))

        pickle.dump(
            history,
            open(weight_dir / "train_history_DiceLoss_Adam_HD95.pkl", "wb")
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                weight_dir / "best_model_weights_DiceLoss_Adam_HD95.pth"
            )

            print("  --> Validation loss improved. Saving model weights.")

        else:
            patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    print("\nTraining complete.")
    print("History saved to:", weight_dir / "train_history_DiceLoss_Adam_HD95.pkl")
    print("Best weights saved to:", weight_dir / "best_model_weights_DiceLoss_Adam_HD95.pth")


    # ==========================================
    # 5. TESTING WITH HD95
    # ==========================================

    print("\nLoading best weights for testing...")

    model.load_state_dict(
        torch.load(
            weight_dir / "best_model_weights_DiceLoss_Adam_HD95.pth",
            map_location=device
        )
    )

    model.eval()

    accumulators = {
        "CT": {"dice": 0.0, "spec": 0.0, "sens": 0.0, "hd95": 0.0, "n_batches": 0},
        "ET": {"dice": 0.0, "spec": 0.0, "sens": 0.0, "hd95": 0.0, "n_batches": 0},
        "WT": {"dice": 0.0, "spec": 0.0, "sens": 0.0, "hd95": 0.0, "n_batches": 0},
        "loss": 0.0,
        "total_batches": 0
    }

    print("Evaluating model on test set...")

    with torch.no_grad():
        for X, CT, ET, WT in test_loader:
            X = X.to(device)
            CT = CT.to(device)
            ET = ET.to(device)
            WT = WT.to(device)

            p_ct, p_et, p_wt = model(X)

            pred_ct = torch.sigmoid(p_ct)
            pred_et = torch.sigmoid(p_et)
            pred_wt = torch.sigmoid(p_wt)

            loss_ct = criterion_ct(p_ct, CT)
            loss_et = criterion_et(p_et, ET)
            loss_wt = criterion_wt(p_wt, WT)

            accumulators["loss"] += (
                loss_ct.item() +
                loss_et.item() +
                loss_wt.item()
            )

            accumulators["total_batches"] += 1

            for name, pred, target in [
                ("CT", pred_ct, CT),
                ("ET", pred_et, ET),
                ("WT", pred_wt, WT)
            ]:
                accumulators[name]["dice"] += weighted_dice_coefficient(
                    target,
                    pred
                ).item()

                accumulators[name]["spec"] += calculate_specificity(
                    target,
                    pred
                ).item()

                accumulators[name]["sens"] += calculate_sensitivity(
                    target,
                    pred
                ).item()

                pred_np = (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
                target_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)

                hd_values = []

                for b in range(pred_np.shape[0]):
                    hd = hd95_score(
                        target_np[b, 0],
                        pred_np[b, 0]
                    )

                    if not np.isnan(hd):
                        hd_values.append(hd)

                if len(hd_values) > 0:
                    accumulators[name]["hd95"] += np.mean(hd_values)

                accumulators[name]["n_batches"] += 1

    n = accumulators["total_batches"]

    final_loss = accumulators["loss"] / n

    ct_dice = accumulators["CT"]["dice"] / accumulators["CT"]["n_batches"]
    ct_spec = accumulators["CT"]["spec"] / accumulators["CT"]["n_batches"]
    ct_sens = accumulators["CT"]["sens"] / accumulators["CT"]["n_batches"]
    ct_hd95 = accumulators["CT"]["hd95"] / accumulators["CT"]["n_batches"]

    et_dice = accumulators["ET"]["dice"] / accumulators["ET"]["n_batches"]
    et_spec = accumulators["ET"]["spec"] / accumulators["ET"]["n_batches"]
    et_sens = accumulators["ET"]["sens"] / accumulators["ET"]["n_batches"]
    et_hd95 = accumulators["ET"]["hd95"] / accumulators["ET"]["n_batches"]

    wt_dice = accumulators["WT"]["dice"] / accumulators["WT"]["n_batches"]
    wt_spec = accumulators["WT"]["spec"] / accumulators["WT"]["n_batches"]
    wt_sens = accumulators["WT"]["sens"] / accumulators["WT"]["n_batches"]
    wt_hd95 = accumulators["WT"]["hd95"] / accumulators["WT"]["n_batches"]

    mean_dice = (ct_dice + et_dice + wt_dice) / 3
    mean_hd95 = (ct_hd95 + et_hd95 + wt_hd95) / 3

    test_results = {
        "CT": {
            "dice": ct_dice,
            "spec": ct_spec,
            "sens": ct_sens,
            "hd95": ct_hd95
        },
        "ET": {
            "dice": et_dice,
            "spec": et_spec,
            "sens": et_sens,
            "hd95": et_hd95
        },
        "WT": {
            "dice": wt_dice,
            "spec": wt_spec,
            "sens": wt_sens,
            "hd95": wt_hd95
        },
        "loss": final_loss,
        "mean_dice": mean_dice,
        "mean_hd95": mean_hd95
    }

    pickle.dump(
        test_results,
        open(weight_dir / "test_results_DiceLoss_Adam_HD95.pkl", "wb")
    )

    SEP = "=" * 70

    print("\n" + SEP)
    print("MSH-Trans-Net TEST SET RESULTS - Dice Loss + Adam + HD95")
    print(SEP)

    print(f"{'Metric':<45} {'Value':>10}")
    print("-" * 70)

    metrics_dict = {
        "loss": final_loss,

        "CT_weighted_dice_coefficient": ct_dice,
        "CT_specificity": ct_spec,
        "CT_sensitivity": ct_sens,
        "CT_HD95": ct_hd95,

        "ET_weighted_dice_coefficient": et_dice,
        "ET_specificity": et_spec,
        "ET_sensitivity": et_sens,
        "ET_HD95": et_hd95,

        "WT_weighted_dice_coefficient": wt_dice,
        "WT_specificity": wt_spec,
        "WT_sensitivity": wt_sens,
        "WT_HD95": wt_hd95,

        "Mean Dice CT + ET + WT": mean_dice,
        "Mean HD95 CT + ET + WT": mean_hd95
    }

    for name, value in metrics_dict.items():
        tag = "  <-- lower is better" if "loss" in name.lower() or "hd95" in name.lower() else ""
        print(f"{name:<45}: {value:10.5f}{tag}")

    print(SEP)
    print("Weights saved to:", weight_dir / "best_model_weights_DiceLoss_Adam_HD95.pth")
    print("History saved to:", weight_dir / "train_history_DiceLoss_Adam_HD95.pkl")
    print("Results saved to:", weight_dir / "test_results_DiceLoss_Adam_HD95.pkl")
    print(SEP)

    print("MODEL FINISHED SUCCESSFULLY")