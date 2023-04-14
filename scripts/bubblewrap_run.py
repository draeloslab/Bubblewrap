

class BubblewrapRun:
    def __init__(self, bw: Bubblewrap, file, bw_parameters=None, time_to_run=None):
        self.file = file
        self.bw_parameters = bw_parameters
        self.time_to_run = time_to_run

        self.A = np.array(bw.A)
        self.mu = np.array(bw.mu)
        self.L = np.array(bw.L)
        self.n_obs = np.array(bw.n_obs)
        self.pred_list = np.array(bw.pred_list)
        self.entropy_list = np.array(bw.entropy_list)
        self.dead_nodes = np.array(bw.dead_nodes)

    def save(self, dir="generated/bubblewrap_runs"):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(dir, f"bubblewrap_run_{time_string}.pickle"), "wb") as fhan:
            pickle.dump(self, fhan)
