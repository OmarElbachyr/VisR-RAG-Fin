from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class QueryQrelsBuilder:
    """Return queries and qrels from chunked_pages.csv"""

    def __init__(self, csv_path: str | Path, deduplicate: bool = True) -> None:
        df = pd.read_csv(csv_path)
        if deduplicate:
            df = df.drop_duplicates(subset=["query", "image_filename"])
        self.df = df

    def build(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        queries: Dict[str, str] = {}
        qrels: Dict[str, Dict[str, int]] = {}
        for i, (qtext, grp) in enumerate(self.df.groupby("query", sort=False), 1):
            qid = f"q{i}"
            queries[qid] = qtext
            page_ids = (
                grp["image_filename"].astype(str)
                    .str.rsplit(".", n=1)
                    .str[0]
                    .unique()
            )
            qrels[qid] = {pid: 1 for pid in page_ids}
        return queries, qrels
