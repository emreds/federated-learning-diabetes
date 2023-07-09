import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]


class DirichletDist:
    def __init__(
        self,
        data_path,
        class_col,
        num_clients,
        num_classes,
        random_state,
        test_split=0.2,
    ) -> None:
        self.data_path = data_path
        self.class_col = class_col
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.test_split = test_split
        self.random_state = random_state

    def _read_data(self):
        data = pd.read_csv(self.data_path)[:70692]

        return data

    def _dirichlet_sample(self, density):
        # Creates an output of shape num_classes x num_clients
        dirichlet_sample = stats.dirichlet.rvs(
            alpha=[density for _ in range(self.num_clients)],
            size=self.num_classes,
            random_state=self.random_state,
        )
        dirichlet_sample.tolist()

        return dirichlet_sample

    def get_dirichlet_noniid_splits(self, density=0.5):
        """
        We create a dirichlet distribution based on the number of clients and the number of target classes.
        :var density: Determins the level of heterogeneity. The lower the density the more heterogeneous the data will be
        distributed.
        :return: A dictionary of dataframes for each client and a separate test dataframe.
        """
        # Open list of file indices
        data = self._read_data()
        dirichlet_sample = self._dirichlet_sample(density)
        train_df, test_df = train_test_split(
            data, test_size=self.test_split, random_state=self.random_state
        )
        # print(len(dirichlet_sample)) 2x10
        with open("dirichlet_sample.txt", "w") as f:
            f.write(str(dirichlet_sample))

        client_dfs = {}
        for client in range(self.num_clients):
            client_df = pd.DataFrame()

            for class_idx in range(self.num_classes):
                subset = train_df.loc[(train_df[self.class_col] == class_idx)]
                # print(len(subset))
                lower_bound = math.floor(
                    len(subset) * sum(dirichlet_sample[class_idx][0:client])
                )
                # print(len(subset))
                # Sometimes the difference between the number at index client and client+1 is so small
                # that the lower bound is equal to the upper bound
                upper_bound = math.floor(
                    len(subset) * sum(dirichlet_sample[class_idx][0:client + 1])
                )
                subset = subset[lower_bound:upper_bound]
                # print(len(subset))
                client_df = pd.concat([client_df, subset])
                # print("Client df: ", len(client_df))
                    

            client_dfs[client] = {
                "target": client_df[self.class_col],
                "data": client_df.drop([self.class_col], axis=1),
            }
        # print(f"This is client_dfs: {client_dfs}")
        test_df = {
            "target": test_df[self.class_col],
            "data": test_df.drop([self.class_col], axis=1),
        }

        return client_dfs, test_df
