import pandas as pd
import pickle as pkl

class PickleToCSV:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def convert(self):
        with open(self.file_path, "rb") as f:
            obj = pkl.load(f)
        df = pd.DataFrame(obj)
        df.to_csv(self.file_path[:-4] + ".csv", index=False)