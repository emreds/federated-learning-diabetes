#Below's a snippet how to create non-IID subsets for federated training. IID subsets can be created in a similar way.
#Best 
#Herbert

import json
import math

import pandas as pd


def create_dirichlet_noniid_splits(self, density=1):
    """
    We create a dirichlet distribution based on the number of clients and the number of target classes.
    :var density: Determins the level of heterogeneity. The lower the density the more heterogeneous the data will be
    distributed.
    :return: None (stored indices on disk).
    """

    # Open most recent class map
    with open(f"{self.file_path}/class_map.json", "r") as f:
        self.class_map = json.load(f)
        f.close()

    # Open list of file indices
    indices = pd.read_csv(f"{self.file_path}/raw_data/events_medal.csv", index_col=0)

    # Creates an output of shape num_classes x num_clients
    dirichlet_sample = dirichlet(alpha=[density for _ in range(self.num_clients)], size=self.num_classes)
    dirichlet_sample.tolist()

    # Transpose class map to have numerals as keys
    class_map_transposed = {v: k for k, v in self.class_map.items()}
    clients = {}

    for client in range(self.num_clients):

        client_df = pd.DataFrame()

        for class_idx, class_dist in enumerate(dirichlet_sample):
            for fold in ["train", "val", "test"]:
                class_name = class_map_transposed[class_idx]
                subset = indices.loc[(indices["Type"] == class_name) & (indices["fold"] == fold)]
                lower_bound = math.floor(len(subset) * sum(dirichlet_sample[class_idx][0:client]))
                upper_bound = math.floor(len(subset) * sum(dirichlet_sample[class_idx][0:client + 1]))
                subset = subset[lower_bound:upper_bound]
                client_df = pd.concat([client_df, subset])

        client_df.to_csv(path_or_buf=f"{self.file_path}/processed/dirichlet/client_{client}.csv")
        print(f"> Dirichlet split for client {client} written to disk.")