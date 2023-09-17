import csv
from functools import reduce
import json
import numpy as np
import pandas as pd

class arxivHypergraph:
    def __init__(self, name):
        self.dataname = name
        self.df_hypergraph = pd.read_csv(self.dataname + '2_hyper_df')
        self.hyperevents_t = self.relabel_nodeid()

    def relabel_nodeid(self, startid=0):
        series_hypergraph = self.df_hypergraph.groupby('timestamp')['nodes'].agg(
            lambda s: list(map(lambda x: json.loads('[' + x[1:-1] + ']'), s)))
        nodeid_map = dict()
        hyper_events = []
        id = startid
        for _, val in series_hypergraph.items():
            hyper_events.append([])
            for e in val:
                for i in e:
                    if not i in nodeid_map:
                        nodeid_map[i] = id
                        id += 1
                hyper_events[-1].append([nodeid_map[i] for i in e])
        return hyper_events

    def savehyperTN2txt(self):
        with open(f'{self.dataname}_hypergraph.dat', 'w') as f:
            for events in self.hyperevents_t:
                f.write(json.dumps(events)+'\n')



if __name__ == '__main__':
    dataname = 'hep-lat'
    data = arxivHypergraph(dataname)
    data.save2txt()