import shutil
from pathlib import Path

import pandas as pd
import tyro
from tqdm.auto import tqdm

from ._imagenet import ImageNet, parse_loc_synset_mapping


def export_subset(
    in1k_root: str | Path,
    synset_mapping_txt: str | Path,
    output_dir: str | Path,
):
    in1k_root = Path(in1k_root)
    output_dir = Path(output_dir)

    synset_df = parse_loc_synset_mapping(synset_mapping_txt)
    wnid_set = {*synset_df["wnid"]}

    for split in ("train", "val"):
        ds = ImageNet(in1k_root, split=split)
        indices = [idx for idx, wnid in enumerate(ds.wnids) if wnid in wnid_set]

        in1k_ann_dir = in1k_root / f"ILSVRC/Annotations/CLS-LOC/{split}"
        ann_dir = output_dir / f"ILSVRC/Annotations/CLS-LOC/{split}"
        for idx in tqdm(indices, desc="Copying annotations"):
            path = ds.paths[idx]
            src = in1k_ann_dir / (path + ".xml")
            if not src.exists():
                continue
            dest = ann_dir / (path + ".xml")
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)

        in1k_data_dir = in1k_root / f"ILSVRC/Data/CLS-LOC/{split}"
        data_dir = output_dir / f"ILSVRC/Data/CLS-LOC/{split}"
        for idx in tqdm(indices, desc="Copying images"):
            path = ds.paths[idx]
            src = in1k_data_dir / (path + ".JPEG")
            dest = data_dir / (path + ".JPEG")
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)

        lists = {
            "train": ["train_cls.txt", "train_loc.txt"],
            "val": ["val.txt"],
        }[split]

        for list_name in lists:
            in1k_list = in1k_root / "ILSVRC/ImageSets/CLS-LOC" / list_name
            dest_list = output_dir / "ILSVRC/ImageSets/CLS-LOC" / list_name

            with open(in1k_list, "r") as f:
                paths = []
                for line in f:
                    path, index = line.strip().split(" ")
                    if int(index) - 1 != len(paths):
                        raise RuntimeError("Invalid class order")
                    paths.append(path)

            dest_list.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_list, "w") as f:
                for idx, path in enumerate(paths, start=1):
                    f.write(f"{path} {idx}\n")

        sol_df = pd.read_csv(in1k_root / f"LOC_{split}_solution.csv")
        image_ids = {ds.paths[idx].split("/")[-1] for idx in indices}
        sol_df = sol_df[sol_df["ImageId"].isin(image_ids)]
        sol_df.to_csv(output_dir / f"LOC_{split}_solution.csv")

    shutil.copy(synset_mapping_txt, output_dir / "LOC_synset_mapping.txt")


def main():
    tyro.cli(export_subset)


if __name__ == "__main__":
    main()
