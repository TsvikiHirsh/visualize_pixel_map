import pandas as pd
import numpy as np

class Data:
    """Class to hold and process pixel data from empirphot CSV files."""
    
    def __init__(self):
        self.df = None
        self.h = {}
        self.keys = []

    @classmethod
    def from_csv(cls, filepath):
        """Load data from a CSV file with x, y, toa, tof columns."""
        instance = cls()
        instance.df = pd.read_csv(filepath)
        instance.df.columns = ["x", "y", "toa", "tof"]
        return instance

    def prepare_histograms(self, bins):
        """Prepare 2D histograms from the data."""
        self.df["tbin"] = pd.cut(self.df["toa"], bins, labels=bins[:-1])
        df1 = self.df.query("toa < 0.01").set_index("tbin")
        
        self.h = {}
        for tbin in df1.index.unique():
            try:
                subset = df1.loc[tbin]
                if not subset.empty:
                    self.h[tbin], x_edges, y_edges = np.histogram2d(
                        subset["x"], subset["y"], bins=[np.arange(256), np.arange(256)]
                    )
            except Exception:
                continue
        
        self.keys = list(self.h.keys())