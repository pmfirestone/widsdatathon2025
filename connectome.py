"""connectome.py: load connectome data as a graph in networkx."""

import re
from typing import Union, List, Tuple
import zipfile

from joblib import Parallel, delayed
import pandas as pd
import networkx as nx
import kaggle
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset


class WiDSDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [
            "TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv",
            "TRAIN/TRAINING_SOLUTIONS.xlsx",
        ]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ["connectomes.pt"]

    def download(self) -> None:
        kaggle.api.authenticate()
        kaggle.api.competition_download_files("widsdatathon2025", '.')
        zipfile.ZipFile('widsdatathon2025.zip').extractall()

    def process(self):
        connectomes_df = pd.read_csv(
            "TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv", index_col=0
        )
        values_df = pd.read_excel("TRAIN/TRAINING_SOLUTIONS.xlsx", index_col=0)
        data_list = []

        def convert_data_to_graph(participant_id):
            # FIXME: Is this two-step conversion necessary, or could we go straight to torch_geometric data?
            # Put dataframe contents into networkx graph.
            connectome = nx.Graph()
            for index, value in zip(
                connectomes_df.loc[participant_id].index,
                connectomes_df.loc[participant_id],
            ):
                u, v = tuple(map(int, re.findall(r"\d+", index)))
                connectome.add_edge(u, v, weight=value)

            # Turn networkx graph into torch_geometric data.
            connectome = from_networkx(connectome)
            connectome.edge_attr = connectome.weight.reshape((39800, 1))
            connectome.y = tuple(values_df.loc[participant_id])
            return connectome

        data_list = Parallel(n_jobs=8, verbose=10)(
            delayed(convert_data_to_graph)(participant_id)
            for participant_id in connectomes_df.index
        )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
