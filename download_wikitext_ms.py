#!/usr/bin/env python3
"""
download_wikitext_ms.py
功能：用 ModelScope SDK 把 WikiText-103 下载到本地
用法：
    python download_wikitext_ms.py --out_dir /data/wikitext-103-v1
"""

import os
import argparse
from modelscope.msdatasets import MsDataset
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./wikitext-103-v1")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 拉取数据集（train / validation / test）
    dataset = MsDataset.load(
        "modelscope/wikitext",
        subset_name="wikitext-103-v1",
        split=None        # 拉取所有 split
    )

    # 2. 依次保存为 parquet
    for split_name, ds in dataset.items():
        parquet_path = os.path.join(args.out_dir, f"wikitext-103-{split_name}.parquet")
        ds.to_pandas().to_parquet(parquet_path, index=False)
        print(f"Saved {split_name} -> {parquet_path}")

    print("Download & save completed.")


if __name__ == "__main__":
    main()