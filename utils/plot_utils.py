import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple
from utils.data_utils import *


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, event: List[List]):
        self.event = event
        self.stream = self.data_stream(event)

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=25,
                                           init_func=self.setup_plot, blit=False)
        self.scat = None

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, c, s = next(self.stream)
        self.scat = self.ax.scatter(x, y, c=c, s=5 * s + 15)
        self.ax.axis([0, 50, 0, 100])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self, event: List[List]) -> List[Tuple]:
        """

        :param event: containing possession info
        :return:
        """

        teams = set(np.array(event[0][-1])[:, 0])
        color_map = {val: key for key, val in enumerate(teams)}
        for moment in event:
            moment_location = moment[-1]
            frame = np.array(moment_location)
            yield frame[:, 3], frame[:, 2], [color_map[key] for key in frame[:, 0]], frame[:, 4]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        # Set x and y data...
        self.scat.set_offsets(np.concatenate([data[0], data[1]]).reshape((11, 2)))
        return self.scat,


if __name__ == '__main__':
    for data_file in [os.path.join("data", f) for f in os.listdir("../data")]:
        raw_data = read_data("../" + data_file)
        for i, event in enumerate(raw_data['events']):
            a = AnimatedScatter(event['moments'])
            plt.show()
