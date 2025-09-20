import pandas as pd
import numpy as np

class Data:
    """Class to hold and process pixel data from empirphot CSV files."""
    
    def __init__(self, csv_filepath="data.empirphot.csv", start_time=0, end_time=0.01, time_step=10):
        """
        Initialize Data with CSV file and histogram bin parameters.
        
        Parameters:
        csv_filepath (str): Path to the empirphot CSV file.
        start_time (float): Start time in seconds for histogram bins.
        end_time (float): End time in seconds for histogram bins.
        time_step (float): Time step in nanoseconds for bin edges.
        """
        self.df = pd.read_csv(csv_filepath)
        self.df.columns = ["x", "y", "toa", "tof"]
        self.h = {}
        self.keys = []
        
        # Convert time_step from nanoseconds to seconds
        time_step_sec = time_step * 1e-9
        bins = np.arange(start_time, end_time + time_step_sec, time_step_sec)
        self.prepare_histograms(bins)

    def prepare_histograms(self, bins):
        """Prepare 2D histograms from the data."""
        self.df["tbin"] = pd.cut(self.df["toa"], bins, labels=bins[:-1])
        df1 = self.df.query("toa < @bins[-1]").set_index("tbin")
        
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

    def plot(self, key=800, zoom_region=None, zoom_size=20,
                         custom_color=None, show_background=False, despine=True,
                         time_bins=None, show_scale=False, cmap=None,
                          show_labels=True, show_legend=False):
        """
        Plot the pixel hit time development using the stored histograms.
        
        Parameters:
        key (int): Starting index for the time bins.
        zoom_region (tuple): (x, y) coordinates for zooming in.
        zoom_size (int): Size of the zoom region.
        custom_color (str): Custom color map for plotting.
        show_background (bool): Whether to show background pixels.
        despine (bool): Whether to remove plot spines.
        time_bins: maximum time in nanoseconds (e.g., 40 for [10, 20, 30, 40], 60 for [10, 20, 30, 40, 50, 60]),
               or None for default [10, 20, 30, 40]
        show_scale (bool): Whether to show scale on the plot.
        cmap (str): Colormap to use for the plot.
        show_labels: whether to display timestamp text labels on pixels (default True)
        show_legend: whether to display a legend with timestamp colors (default False)
        """

        from visualize_pixel_map.visualize import plot_time_development
        plot_time_development(self.h, self.keys, key, zoom_region, zoom_size,
                              custom_color, show_background, despine,
                              time_bins, show_scale, cmap,
                              show_labels, show_legend)