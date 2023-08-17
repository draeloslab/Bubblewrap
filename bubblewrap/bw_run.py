# todo: maybe put JAX here to prevent gpu usage?
from bubblewrap import Bubblewrap
import datetime
import pickle
import os

class BWRun:
    def __init__(self, bw: Bubblewrap, data_source):
        # todo: account for total runtime
        self.data_source = data_source
        self.save_location = None
        self.bw = bw

    def save(self, directory="generated/bubblewrap_runs"):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.outfile = os.path.join(directory, f"bubblewrap_run_{time_string}.pickle")
        with open(self.outfile, "wb") as fhan:
            pickle.dump(self, fhan)