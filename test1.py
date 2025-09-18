import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
# from forward_utils1 import (
#     get_adapted_text_embedding,
#     calculate_similarity_map,
#     metrics_eval,
#     visualize,
# )
try:
    from forward_utils import (
        get_adapted_text_embedding,
        get_adapted_single_class_text_embedding,
        calculate_similarity_map,
        metrics_eval,
        visualize,
    )
    import forward_utils as fu_mod
except ImportError:
    from forward_utils1 import (
        get_adapted_text_embedding,
        get_adapted_single_class_text_embedding,
        calculate_similarity_map,
        metrics_eval,
        visualize,
    )
    import forward_utils1 as fu_mod

import warnings
warnings.filterwarnings("ignore")
import inspect
import sys
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features
def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        # class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        epoch_text_feature = class_text_embeddings
        raw_prompts = class_text_embeddings
        # forward image
        patch_features, det_feature = model(image)

        # PREP TEXT EMBEDDINGS IN TWO FORMS ==================
        # 1) epoch_text_feature: for image-level logits
        # 2) state_emb: for patch-level similarity

        if isinstance(class_text_embeddings, list):
            cols = []
            for t in class_text_embeddings:
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                # average over prompts to get a single vector per class (normal/anomaly)
                if t.dim() == 2:
                    v = t.mean(dim=0)   
                else:
                    v = t.view(-1)     
                # L2 normalize
                v = v / (v.norm() + 1e-12)
                cols.append(v)
            epoch_text_feature = torch.stack(cols, dim=1).to(device=det_feature.device, dtype=det_feature.dtype)

            # For patches, all prompts stacked
            state_emb = torch.cat(class_text_embeddings, dim=0).to(device=det_feature.device, dtype=det_feature.dtype)

        elif isinstance(class_text_embeddings, torch.Tensor):
            epoch_text_feature = class_text_embeddings.to(device=det_feature.device, dtype=det_feature.dtype)

            state_emb = epoch_text_feature.T  
        else:
            raise TypeError(f"Unexpected type for class_text_embeddings: {type(class_text_embeddings)}")
        

        # DEBUG: shapes for matmul sanity ====================
        if isinstance(class_text_embeddings, torch.Tensor):
            print("[DEBUG] det_feature.shape:", tuple(det_feature.shape),
                "| epoch_text_feature(type/shape): Tensor", tuple(class_text_embeddings.shape))
        else:
            print("[DEBUG] det_feature.shape:", tuple(det_feature.shape),
                "| epoch_text_feature(type):", type(class_text_embeddings))
        # ================================

        # Ensure epoch_text_feature is a tensor
        #epoch_text_feature = torch.stack(epoch_text_feature).to( device=det_feature.device, dtype=det_feature.dtype )       
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
      
       
        B = image.size(0)
        _fixed = []
        for pf in patch_features:
            if pf.dim() == 2:
                if pf.size(0) % B != 0:
                    raise RuntimeError(f"Cannot reshape patch features: got {tuple(pf.shape)} with batch {B}")
                pf = pf.view(B, -1, pf.size(-1)).contiguous()
            _fixed.append(pf)
        patch_features = _fixed
        print("[DEBUG] patch_features AFTER FIX:", [tuple(p.shape) for p in patch_features],
            "| state_emb:", tuple(state_emb.shape))

        patch_preds = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, raw_prompts, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names
def extract_text_embeddings(adapted_model, dataset_name, data_loader, device):
    adapted_model.eval()
    normal_embeddings = []
    anomaly_embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting Text Embeddings"):
            class_names = batch["class_name"]
            for class_name in class_names:
                text_embed = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )  
                normal_embeddings.append(text_embed[:, 0].cpu().numpy())
                anomaly_embeddings.append(text_embed[:, 1].cpu().numpy())
                labels.append((class_name, "normal"))
                labels.append((class_name, "anomaly"))
    return np.array(normal_embeddings), np.array(anomaly_embeddings), labels
def plot_tsne(normal_embeds, anomaly_embeds, labels, save_path="tsne_plot.png"):
    from matplotlib.cm import get_cmap
    all_embeddings = np.vstack([normal_embeds, anomaly_embeds])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    half = len(embeddings_2d) // 2
    embeds_norm = embeddings_2d[:half]
    embeds_anom = embeddings_2d[half:]
    class_names = sorted(list(set([lbl[0] for lbl in labels[::2]])))
    color_map = {cls: i for i, cls in enumerate(class_names)}
    cmap = get_cmap("tab20")
    plt.figure(figsize=(12, 8))
    for i, ((cls, _), norm_pt, anom_pt) in enumerate(zip(labels[::2], embeds_norm, embeds_anom)):
        color = cmap(color_map[cls] % 20)
        plt.scatter(norm_pt[0], norm_pt[1], color=color, label=cls if i < len(class_names) else "", marker='o', alpha=0.6)
        plt.scatter(anom_pt[0], anom_pt[1], color=color, marker='x', alpha=0.6)
    plt.title("t-SNE of Normal (o) vs Anomaly (x) Text Embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    args = parser.parse_args()
    ###t-SNE####
    tsne_embeddings = []
    tsne_labels = []
    tsne_classnames = []
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    assert len(text_file) >= 0, "text adapter checkpoint not found"
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False
    files = sorted(glob(args.save_path + "/image_adapter_20.pth"))
    assert len(files) > 0, "image adapter checkpoint not found"
    for file in files:
        checkpoint = torch.load(file)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")
        # ========================================================
        # load dataset
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
        )
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )

        print("\n[DEBUG] Using functions from:")
        print("  get_adapted_text_embedding  ->", inspect.getsourcefile(get_adapted_text_embedding))
        print("  get_adapted_single_class_text_embedding ->", inspect.getsourcefile(get_adapted_single_class_text_embedding))

        # Peek one class entry from the dict to see what the values look like
        _any_class = next(iter(text_embeddings))
        _val = text_embeddings[_any_class]
        print(f"[DEBUG] text_embeddings keys sample: '{_any_class}'")
        print("[DEBUG] type(text_embeddings[_any_class]) =", type(_val))
        if isinstance(_val, torch.Tensor):
            print("[DEBUG] shape(text_embeddings[_any_class]) =", tuple(_val.shape), "dtype:", _val.dtype, "device:", _val.device)
        elif isinstance(_val, list) and len(_val) > 0:
            print("[DEBUG] list length =", len(_val), "elem0 type:", type(_val[0]))
            if isinstance(_val[0], torch.Tensor):
                print("[DEBUG] elem0 shape =", tuple(_val[0].shape), "dtype:", _val[0].dtype, "device:", _val[0].device)
        print()

        
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )
           

            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]  # pass the LIST

                # (optional) quick debug
                print(f"[DEBUG] class '{class_name}':", type(class_text_embeddings),
                    "-> elem0", (tuple(class_text_embeddings[0].shape) if isinstance(class_text_embeddings, list) else "n/a"))

                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,  # LIST goes in
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                )


            # ========================================================
            if args.visualize:
                visualize(
                    masks,
                    preds,
                    file_names,
                    args.save_path,
                    args.dataset,
                    class_name=class_name,
                )
            class_result_dict = metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)
        #df.loc[len(df)] = df.mean()
        #df.loc[len(df) - 1]["class name"] = "Average"
        numeric_mean = df.drop(columns=['class name']).mean()
        df.loc[len(df)] = ['Average'] + numeric_mean.tolist()
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
       
if __name__ == "__main__":
    main()
