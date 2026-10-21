#!/usr/bin/env python
"""Aggregate seg + cls agreement across the model x dataset matrix.

Reads seg_agreement_r224/summary.csv and cls_agreement_r224/model_summary.csv
for each model/dataset, prints comparison tables, and dumps a JSON blob with all
numbers (per-organ + per-classifier) for building the Notion page.
"""
import csv
import json
import os

MODELS = {  # display name -> root
    "spatial": "output/main_result/rct_token_224_p14_chexworld",
    "pooled": "output/rct_pooled_224_chexworld",
    "RCDM": "output/main_result/rcdm_chexworld_cxr",
    "FeatInv": "baseline/FeatInv/output",
}
DATASETS = {"MIMIC": "fid_eval", "NIH": "fid_eval_nih", "CheXpert": "fid_eval_chexpert"}
MODEL_ORDER = ["spatial", "pooled", "RCDM", "FeatInv"]


def read_seg(root, ds):
    p = os.path.join(root, ds, "seg_agreement_r224", "summary.csv")
    if not os.path.exists(p):
        return None
    organs, macro = {}, {}
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["organ"] == "MACRO_MEAN":
                macro = {"dice": float(row["mean_dice"]), "iou": float(row["mean_iou"]),
                         "n": int(row["n_valid"])}
            else:
                organs[row["organ"]] = {"dice": float(row["mean_dice"]),
                                        "iou": float(row["mean_iou"]),
                                        "n": int(row["n_valid"])}
    return {"macro": macro, "organs": organs}


def read_cls(root, ds):
    p = os.path.join(root, ds, "cls_agreement_r224", "model_summary.csv")
    if not os.path.exists(p):
        return None
    per_model, prs, maes, agrs = {}, [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            pr = float(row["mean_pearson"]); mae = float(row["mean_mae"])
            ag = float(row["mean_bin_agree"])
            per_model[row["model"]] = {"pearson": pr, "mae": mae, "bin_agree": ag,
                                       "n_path": int(row["n_informative_path"])}
            prs.append(pr); maes.append(mae); agrs.append(ag)
    n = len(prs)
    return {"per_model": per_model,
            "mean_pearson": sum(prs) / n, "mean_mae": sum(maes) / n,
            "mean_bin_agree": sum(agrs) / n}


def main():
    os.chdir("/home/yspark/RCT")
    data = {}
    for ds, dsdir in DATASETS.items():
        data[ds] = {}
        for m in MODEL_ORDER:
            root = MODELS[m]
            data[ds][m] = {"seg": read_seg(root, dsdir), "cls": read_cls(root, dsdir)}

    with open("/tmp/claude-1001/-home-yspark-RCT/905fc9ef-98c7-4124-8b8a-b8d26ff0a52c/scratchpad/agreement_all.json", "w") as f:
        json.dump(data, f, indent=2)

    def fmt_table(title, getter, cols, higher_better):
        print(f"\n### {title}")
        print("dataset  model    " + "  ".join(f"{c:>10}" for c in cols))
        for ds in DATASETS:
            # collect col values for ranking within dataset
            vals = {c: [] for c in cols}
            present = []
            for m in MODEL_ORDER:
                v = getter(data[ds][m])
                if v is None:
                    continue
                present.append(m)
                for c in cols:
                    vals[c].append((m, v[c]))
            rank = {}
            for c in cols:
                order = sorted(vals[c], key=lambda x: x[1], reverse=higher_better[c])
                rank[c] = {order[0][0]: "*", order[1][0]: "_"} if len(order) > 1 else {}
            for m in MODEL_ORDER:
                if m not in present:
                    continue
                v = getter(data[ds][m])
                cells = []
                for c in cols:
                    mark = rank[c].get(m, " ")
                    cells.append(f"{mark}{v[c]:.4f}{mark}")
                print(f"{ds:<8} {m:<8} " + "  ".join(f"{c:>10}" for c in cells))

    fmt_table("Segmentation agreement (Dice/IoU, higher better)",
              lambda d: None if d["seg"] is None else d["seg"]["macro"],
              ["dice", "iou"], {"dice": True, "iou": True})

    fmt_table("Classification agreement (7-model mean)",
              lambda d: None if d["cls"] is None else d["cls"],
              ["mean_pearson", "mean_mae", "mean_bin_agree"],
              {"mean_pearson": True, "mean_mae": False, "mean_bin_agree": True})

    print("\nJSON -> scratchpad/agreement_all.json")


if __name__ == "__main__":
    main()
