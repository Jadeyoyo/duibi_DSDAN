import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PU_CLASS_NAMES = [
    "KA01", "KA03", "KA04", "KA05", "KA07", "KA08",
    "KI01", "KI03", "KI04", "KI07", "KI16", "KI18"
]

PU_COLOR_MAP = {
    "KA01": "#1f77b4",
    "KA03": "#ff7f0e",
    "KA04": "#2ca02c",
    "KA05": "#d62728",
    "KA07": "#9467bd",
    "KA08": "#8c564b",
    "KI01": "#e377c2",
    "KI03": "#7f7f7f",
    "KI04": "#bcbd22",
    "KI07": "#17becf",
    "KI16": "#9edae5",
    "KI18": "#aec7e8",
}


def _label_name_map_pu(nclass=12):
    return {i: PU_CLASS_NAMES[i] for i in range(min(nclass, len(PU_CLASS_NAMES)))}


def _fixed_label_color_map(label_name_map):
    color_map = {}
    for label, name in sorted(label_name_map.items(), key=lambda x: x[0]):
        color_map[label] = PU_COLOR_MAP.get(name, "#333333")
    return color_map


def _balanced_tsne_indices(labels, seed=2, max_per_class=150):
    labels = np.asarray(labels)
    if labels.size == 0 or max_per_class <= 0:
        return np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == label)[0]
        if cls_idx.shape[0] > max_per_class:
            cls_idx = np.sort(rng.choice(cls_idx, size=max_per_class, replace=False))
        selected.append(cls_idx)
    return np.concatenate(selected, axis=0) if selected else np.arange(labels.shape[0], dtype=np.int64)


def _scatter_discrete(ax, coords, labels, title, label_name_map):
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    color_map = _fixed_label_color_map(label_name_map)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.7,
            color=color_map.get(label, "#333333"),
            label=label_name_map.get(label, str(label)),
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=7, frameon=False)


@torch.no_grad()
def save_dsan_tsne_same_style(args, model, dataloader, save_path):
    device = next(model.parameters()).device
    model.eval()

    feat_all = []
    gt_all = []
    pred_all = []

    for data, target in dataloader:
        data = data.float().to(device)
        if isinstance(target, torch.Tensor):
            y = target.long().cpu().numpy()
        else:
            y = np.array([int(x) for x in target], dtype=np.int64)

        if hasattr(model, 'extract_feat'):
            feat = model.extract_feat(data)
        else:
            feat = model.feature_layers(data)
        logits = model.predict(data)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        feat_all.append(feat.detach().cpu().numpy())
        gt_all.append(y)
        pred_all.append(pred)

    feat_all = np.concatenate(feat_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0).astype(np.int64)
    pred_all = np.concatenate(pred_all, axis=0).astype(np.int64)

    select_idx = _balanced_tsne_indices(gt_all, seed=int(args.seed), max_per_class=int(getattr(args, 'tsne_max_per_class', 150)))
    feat_all = feat_all[select_idx]
    gt_all = gt_all[select_idx]
    pred_all = pred_all[select_idx]

    # robust guards against collapse / zero variance
    feat_all = np.asarray(feat_all, dtype=np.float64)
    feat_all = np.nan_to_num(feat_all, nan=0.0, posinf=0.0, neginf=0.0)
    std_all = np.std(feat_all)
    if not np.isfinite(std_all) or std_all < 1e-12:
        feat_all = feat_all + np.random.default_rng(int(args.seed)).normal(0, 1e-6, size=feat_all.shape)

    feat_all = StandardScaler().fit_transform(feat_all)
    feat_all = np.nan_to_num(feat_all, nan=0.0, posinf=0.0, neginf=0.0)

    pca_dim = int(min(50, feat_all.shape[0] - 1, feat_all.shape[1]))
    if pca_dim >= 2:
        feat_all = PCA(n_components=pca_dim, random_state=int(args.seed)).fit_transform(feat_all)
        feat_all = np.nan_to_num(feat_all, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(feat_all) < 1e-12:
        feat_all = feat_all + np.random.default_rng(int(args.seed)).normal(0, 1e-6, size=feat_all.shape)

    perplexity = min(30, max(5, feat_all.shape[0] // 20))
    tsne_feat = TSNE(
        n_components=2,
        init="pca",
        random_state=int(args.seed),
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(feat_all)

    label_name_map = _label_name_map_pu(int(args.nclass))
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    _scatter_discrete(ax, tsne_feat, gt_all, "Ground Truth", label_name_map)
    ax = plt.subplot(1, 2, 2)
    _scatter_discrete(ax, tsne_feat, pred_all, "Prediction", label_name_map)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
