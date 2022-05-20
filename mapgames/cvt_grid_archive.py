import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from sklearn.neighbors import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mapgames import AbstractContainer
from mapgames import Individual

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

######### CVT-related functions


def make_hashable(array):
    return tuple(map(float, array))


def centroids_filename(k, dim):
    return "CVT/centroids_" + str(k) + "_" + str(dim) + ".dat"


def write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = centroids_filename(k, dim)
    with open(filename, "w") as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + " ")
            f.write("\n")


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            if dim == 1:
                if k == 1:
                    return np.expand_dims(
                        np.expand_dims(np.loadtxt(fname), axis=0), axis=1
                    )
                return np.expand_dims(np.loadtxt(fname), axis=1)
            else:
                if k == 1:
                    return np.expand_dims(np.loadtxt(fname), axis=0)
                return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)
    x = np.random.rand(samples, dim)
    k_means = KMeans(
        init="k-means++",
        n_clusters=k,
        n_init=1,
        max_iter=1000000,
        verbose=1,
        tol=1e-8,
    )  # Full is the proper Expectation Maximization algorithm
    k_means.fit(x)
    write_centroids(k_means.cluster_centers_)
    return k_means.cluster_centers_


class CvtGridArchive(AbstractContainer):
    def __init__(self, n_niches: int, dim: int, cvt_samples, cvt_use_cache=True):
        super().__init__()
        self.archive = {}

        self.n_niches = n_niches
        self.dim = dim
        self.cvt_samples = cvt_samples
        self.cvt_use_cache = cvt_use_cache

        self.cvt_filename = centroids_filename(n_niches, dim)
        self.cvt = cvt(n_niches, dim, cvt_samples, cvt_use_cache)
        self.kdt = KDTree(self.cvt, leaf_size=30, metric="euclidean")

    def _attempt_add_individual(self, individual: Individual) -> bool:
        niche_index = self.kdt.query([individual.desc], k=1)[1][0][0]
        niche = self.kdt.data[niche_index]

        # usable id of the individual's niche
        n = make_hashable(niche)
        # add the centroid id info to the individual instance
        individual.centroid = n

        # if there is already an individual in this niche
        if n in self.archive:
            # Try to add to cell
            if individual.fitness > self.archive[n].fitness:
                # store information in the controller
                individual.novel = False
                individual.delta_f = individual.fitness - self.archive[n].fitness
                self.archive[n] = individual
                return 1
            else:
                return 0
        # if it is the first time we encounter an individual in this niche
        else:
            # Create the cell
            self.archive[n] = individual
            # add the information to the individual instance
            individual.novel = True
            individual.delta_f = (
                individual.fitness
            )  # maybe we should beware the cases where the fitness can be negative
            self.archive[n] = individual
            return 1

    def _direct_add_individual(self, individual: Individual):
        print("WARNING: not implemented")
        pass

    def get_individual_at_index(self, flattened_index):
        return list(self.archive.values())[flattened_index]

    def get_all_individuals(self):
        return [i for _, i in self.archive.items()]

    def __len__(self):
        return len(self.archive)

    def clear(self):
        self.archive.clear()

    def empty_copy(self):
        return CvtGridArchive(self.n_niches, self.dim, self.cvt_samples, True)

    def refresh(self):
        pass

    def plot(self, x_name="", y_name="", save_path=None, ax=None):
        centroids = load_centroids(self.cvt_filename)
        colormap = mpl.cm.viridis

        fit, beh = [], []
        for i in self.get_all_individuals():
            fit.append(i.fitness)
            beh.append(i.centroid)

        if len(fit) == 0:
            print("\n!!!WARNING!!! Empty archive\n")
            return None
        if len(beh[0]) > 2:
            print("\n!!!WARNING!!! Archive has too many dimensions.")
            return None

        min_fit = min(fit)
        max_fit = max(fit)

        params = {
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "text.usetex": False,
            "figure.figsize": [6, 6],
            "figure.dpi": 800,
        }
        mpl.rcParams.update(params)

        # Plot
        if ax is None:
            fig, ax = plt.subplots(facecolor="white", edgecolor="white")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set(adjustable="box", aspect="equal")
        norm = mpl.colors.Normalize(vmin=min_fit, vmax=max_fit)

        try:
            plot_cvt(ax, centroids, fit, beh, norm, colormap)
        except Exception as e:
            print("\n!!!WARNING!!! Error when plotting archive")
            print(e)

        # Add axis name and colorbar (except if paper_mode)
        ax.set_xticks([])
        ax.set_yticks([])
        if len(x_name) == 0 or len(y_name) == 0:
            ax.set(xlabel=None)
            ax.set(ylabel=None)
        else:
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax
            )
            cbar.set_label("Fitness", size=24, labelpad=5)
            cbar.ax.tick_params(labelsize=18)

        if save_path != None:
            fig.savefig(save_path + ".png", bbox_inches="tight")

        plt.close()


############################ Sub-plot functions


def plot_cvt(ax, centroids, fit, desc, norm, colormap):
    """
    Plot each cell using polygon shapes.
    """
    # compute Voronoi tesselation
    vor = Voronoi(centroids)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    kdt = KDTree(centroids, leaf_size=30, metric="euclidean")
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.5, edgecolor="black", facecolor="white", lw=1)

    k = 0
    for i in range(0, len(desc)):
        q = kdt.query([desc[i]], k=1)
        index = q[1][0][0]
        region = regions[index]
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.9, color=colormap(norm(fit[i])))
        k += 1


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


############################ Files load functions


def load_centroids(filename):
    """Load a file."""
    points = np.loadtxt(filename)
    return points


def load_data(filename, dim, verbose=False):
    """Load a cvt archive file and transform it into a usable dataframe."""

    data = np.loadtxt(filename)

    # print("Raw data : ", data)
    verbose and print("Raw data dimension: ", data.ndim)
    if data.ndim == 1:
        if len(data) == 0:
            fit = np.array([])
            desc = np.array([])
            x = np.array([])
        else:
            fit = np.array([data[0:1]])
            desc = np.array([data[dim + 1 : 2 * dim + 1]])
            x = np.array([data[dim + 1 : 2 * dim + 1]])
    else:
        fit = data[:, 0:1]
        desc = data[:, dim + 1 : 2 * dim + 1]
        x = data[:, dim + 1 : 2 * dim + 1 :]

    return fit, desc, x
