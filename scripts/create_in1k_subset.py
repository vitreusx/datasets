"""Create a subset of ImageNet."""

import shutil
from pathlib import Path

import pandas as pd
import tyro
from tqdm.auto import tqdm

from rsrch_data.imagenet import ImageNet, parse_loc_synset_mapping


def create_in1k_subset(  # noqa: C901
    in1k_root: str | Path,
    output_dir: str | Path,
    synset_mapping_txt: str | Path | None = None,
    num_classes: int | None = None,
    seed: int = 0,
) -> None:
    """Create a subset of ImageNet.

    :param in1k_root: Data root to ImageNet-1k dataset.
    :param output_dir: Output data root to the subset of IN-1k.
    :param synset_mapping_txt: Custom LOC_synset_mapping.txt file.
    :param num_classes: If provided, take a sample of the class set obtained
        from the LOC_synset_mapping.txt file.
    """
    in1k_root = Path(in1k_root)
    output_dir = Path(output_dir)

    if synset_mapping_txt is None:
        synset_mapping_txt = in1k_root / "LOC_synset_mapping.txt"
    synset_df = parse_loc_synset_mapping(synset_mapping_txt)
    if num_classes is not None:
        num_classes = min(num_classes, len(synset_df))
        synset_df = synset_df.sample(num_classes, replace=False, random_state=seed)

    wnid_set = {*synset_df["wnid"]}

    for split in ("train", "val"):
        if (output_dir / f"LOC_{split}_solution.csv").exists():
            continue
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
            dest_list = output_dir / "ILSVRC/ImageSets/CLS-LOC" / list_name
            dest_list.parent.mkdir(parents=True, exist_ok=True)
            with dest_list.open("w") as f:
                f.writelines(
                    f"{ds.paths[old_idx]} {new_idx}\n"
                    for new_idx, old_idx in enumerate(indices, start=1)
                )

        sol_df = pd.read_csv(in1k_root / f"LOC_{split}_solution.csv")
        image_ids = {ds.paths[idx].split("/")[-1] for idx in indices}
        sol_df = sol_df[sol_df["ImageId"].isin(image_ids)]
        sol_df.to_csv(output_dir / f"LOC_{split}_solution.csv", index=False)

    with (output_dir / "LOC_synset_mapping.txt").open("w") as f:
        for _, row in synset_df.iterrows():
            f.write(f"{row['wnid']} {row['defs']}\n")


def main() -> None:
    """Entry point."""
    tyro.cli(create_in1k_subset)


if __name__ == "__main__":
    main()
