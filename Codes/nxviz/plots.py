import logging

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.path import Path

from .geometry import (circos_radius, get_cartesian, node_theta, group_theta,
                       text_alignment)
from .polcart import to_degrees
from .utils import (cmaps, infer_data_type, is_data_diverging,
                    num_discrete_groups, n_group_colorpallet, items_in_groups)


def despine(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


class BasePlot(object):
    """
    BasePlot: An extensible class for designing new network visualizations.

    The BasePlot constructor takes in a NetworkX graph object, and a series of
    keyword arguments specifying how nodes and edges should be styled and
    ordered.

    An optional `data_types` dictionary can be passed in to bypass data type
    inference.

    :param graph: A NetworkX graph object.
    :type graph: `nx.Graph`, `nx.DiGraph`, `nx.MultiGraph`, `nx.MultiDiGraph`

    :param node_order: The node attribute on which to specify the coloring of nodes.
    :type node_order: `dict_key` (often `str`)

    :param node_size: The node attribute on which to specify the size of nodes.
    :type node_size: `dict_key` (often `str`)

    :param node_grouping: The node attribute on which to specify the grouping position of nodes.
    :type node_grouping: `dict_key` (often `str`)

    :param node_color: The node attribute on which to specify the colour of nodes.
    :type node_color: `dict_key` (often `str`)

    :param node_labels: Boolean, whether to use node objects as labels or not.
    :type node_labels: `bool`

    :param edge_width: The edge attribute on which to specify the width of edges.
    :type edge_with: `dict_key` (often `str`)

    :param edge_color: The edge attribute on which to specify the colour of edges.
    :type edge_color: `dict_key` (often `str`)

    :param data_types: A mapping of node and edge data types that are stored.
    :type data_types: `dict`

    :param nodeprops: A `matplotlib`-compatible `props` dictionary.
    :type nodeprops: `dict`

    :param edgeprops: A `matplotlib-compatible `props` dictioanry.
    :type edgeprops: `dict`
    """  # noqa
    def __init__(self, graph, node_order=None, node_size=None,
                 node_grouping=None, node_color=None, node_labels=None,
                 edge_width=None, edge_color=None, data_types=None,
                 nodeprops=None, edgeprops=None, node_label_color=False,
                 group_label_position=None, group_label_color=False, **kwargs):
        super(BasePlot, self).__init__()
        # Set graph object
        self.graph = graph
        self.nodes = list(graph.nodes())  # keep track of nodes separately.
        self.edges = list(graph.edges())
        # Set node arrangement
        self.node_order = node_order
        self.node_grouping = node_grouping
        self.group_and_sort_nodes()

        # Set node radius
        self.node_size = node_size

        # Set node colors
        self.node_color = node_color
        self.sm = None  # sm -> for scalarmappable. See https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots  # noqa
        logging.debug('INIT: {0}'.format(self.sm))
        if self.node_color:
            self.node_colors = []
            self.compute_node_colors()
        else:
            self.node_colors = ['blue'] * len(self.nodes)
        self.node_labels = node_labels

        # Set edge properties
        self.edge_width = edge_width
        self.edge_color = edge_color
        if self.edge_color:
            self.edge_colors = []
            self.compute_edge_colors()
        else:
            self.edge_colors = ['black'] * len(self.edges)

        # Set data_types dictionary
        if not data_types:
            self.data_types = dict()
        else:
            self.check_data_types(data_types)
            self.data_types = data_types

        figsize = (6, 6)
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        self.figure = plt.figure(figsize=figsize)
        self.ax = self.figure.add_subplot(1, 1, 1)
        despine(self.ax)

        # We provide the following attributes that can be set by the end-user.
        # nodeprops are matplotlib patches properties.
        if nodeprops:
            self.nodeprops = nodeprops
        else:
            self.nodeprops = {'radius': 1}
        # edgeprops are matplotlib line properties. These can be set after
        # instantiation but before calling the draw() function.
        if edgeprops:
            self.edgeprops = edgeprops
        else:
            self.edgeprops = {'facecolor': 'none',
                              'alpha': 0.2}
        if node_label_color:
            self.node_label_color = self.node_colors
        else:
            self.node_label_color = ['black'] * len(self.nodes)

        # Compute each node's positions.
        self.compute_node_positions()

        # Conditionally compute node label positions.
        self.compute_node_label_positions()

        # set group properties
        self.group_label_position = group_label_position
        if group_label_position:
            self.groups = []
            self.compute_group_label_positions()
            if group_label_color:
                self.compute_group_colors()
            else:
                self.group_label_color = ['black'] * len(self.nodes)

    def check_data_types(self, data_types):
        """
        Checks the data_types passed into the Plot constructor and makes sure
        that the values passed in belong to 'ordinal', 'categorical', or
        'continuous'.

        :param data_types: A dictionary mapping of data types.
        :type data_types: `dict`
        """
        for k, v in data_types.items():
            assert v in ['ordinal', 'categorical', 'continuous']

    def draw(self):
        """
        Draws the Plot to screen.

        If there is a continuous datatype for the nodes, it will be reflected
        in self.sm being constructed (in `compute_node_colors`). It will then
        automatically add in a colorbar to the plot and scale the plot axes
        accordingly.
        """
        self.draw_nodes()
        self.draw_edges()
        # if self.groups:
        #     self.draw_group_labels()
        logging.debug('DRAW: {0}'.format(self.sm))
        if self.sm:
            self.figure.subplots_adjust(right=0.8)
            cax = self.figure.add_axes([0.85, 0.2, 0.05, 0.6])
            self.figure.colorbar(self.sm, cax=cax)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_aspect('equal')

    def compute_node_colors(self):
        """Compute the node colors. Also computes the colorbar."""
        data = [self.graph.nodes[n][self.node_color] for n in self.nodes]
        data_reduced = sorted(list(set(data)))
        dtype = infer_data_type(data)
        n_grps = num_discrete_groups(data)

        if dtype == 'categorical' or dtype == 'ordinal':
            if n_grps <= 8:
                cmap = \
                    get_cmap(cmaps['Accent_{0}'.format(n_grps)].mpl_colormap)
            else:
                cmap = n_group_colorpallet(n_grps)
        elif dtype == 'continuous' and not is_data_diverging(data):
            cmap = get_cmap(cmaps['continuous'].mpl_colormap)
        elif dtype == 'continuous' and is_data_diverging(data):
            cmap = get_cmap(cmaps['diverging'].mpl_colormap)

        for d in data:
            idx = data_reduced.index(d)*2 / n_grps
            self.node_colors.append(cmap(idx))

        # Add colorbar if required.ListedColormap
        logging.debug('length of data_reduced: {0}'.format(len(data_reduced)))
        logging.debug('dtype: {0}'.format(dtype))
        if len(data_reduced) > 1 and dtype == 'continuous':
            self.sm = plt.cm.ScalarMappable(cmap=cmap,
                                            norm=plt.Normalize(vmin=min(data_reduced),  # noqa
                                                               vmax=max(data_reduced)   # noqa
                                                               )
                                            )
            self.sm._A = []

    def compute_group_colors(self):
        """Computes the group colors according to node colors"""
        seen = set()
        self.group_label_color = \
            [x for x in self.node_colors if not (x in seen or seen.add(x))]

    def compute_edge_colors(self):
        """Compute the edge colors."""
        data = [self.graph.edges[n][self.edge_color] for n in self.edges]
        data_reduced = sorted(list(set(data)))
        dtype = infer_data_type(data)
        n_grps = num_discrete_groups(data)
        if dtype == 'categorical' or dtype == 'ordinal':
            if n_grps <= 8:
                cmap = \
                    get_cmap(cmaps['Accent_{0}'.format(n_grps)].mpl_colormap)
            else:
                cmap = n_group_colorpallet(n_grps)
        elif dtype == 'continuous' and not is_data_diverging(data):
            cmap = get_cmap(cmaps['weights'])

        for d in data:
            idx = data_reduced.index(d) / n_grps
            self.edge_colors.append(cmap(idx))
        # Add colorbar if required.
        logging.debug('length of data_reduced: {0}'.format(len(data_reduced)))
        logging.debug('dtype: {0}'.format(dtype))
        if len(data_reduced) > 1 and dtype == 'continuous':
            self.sm = plt.cm.ScalarMappable(cmap=cmap,
                                            norm=plt.Normalize(vmin=min(data_reduced),  # noqa
                                                               vmax=max(data_reduced)   # noqa
                                                               )
                                            )
            self.sm._A = []

    def compute_group_label_positions(self):
        """Computes the position of each group label according to the wanted
        position ("beginning", "middle", "end")

        Needs to be implemented for each plot type
        """
        pass

    def compute_node_positions(self):
        """
        Computes the positions of each node on the plot.

        Needs to be implemented for each plot type.
        """
        pass

    def compute_node_label_positions(self):
        """
        Computes the positions of each node's labels on the plot. The
        horizontal and vertical alignment of the text varies according to the
        location.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_nodes(self):
        """
        Renders the nodes to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_edges(self):
        """
        Renders the nodes to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_group_labels(self):
        """
        Renders the group labels to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass

    def group_and_sort_nodes(self):
        """
        Groups and then sorts the nodes according to the criteria passed into
        the Plot constructor.
        """
        if self.node_grouping and not self.node_order:
            self.nodes = [n for n, d in
                          sorted(self.graph.nodes(data=True),
                                 key=lambda x: x[1][self.node_grouping])]

        elif self.node_order and not self.node_grouping:
            self.nodes = [n for n, _ in
                          sorted(self.graph.nodes(data=True),
                                 key=lambda x: x[1][self.node_order])]

        elif self.node_grouping and self.node_order:
            self.nodes = [n for n, d in
                          sorted(self.graph.nodes(data=True),
                                 key=lambda x: (x[1][self.node_grouping],
                                                x[1][self.node_order]))]


class CircosPlot(BasePlot):
    """
    Plotting object for CircosPlot.
    """

    def __init__(self, graph, **kwargs):
        """Create the CircosPlot.

        Accepts the following additional arguments apart from the ones in
        `BasePlot`:

        :param rotate_labels: Whether to rotate node labels.
        :type node_color: `bool`
        """
        self.rotate = kwargs.pop("rotate_labels", False)
        super(CircosPlot, self).__init__(graph, **kwargs)

    def compute_group_label_positions(self):
        """
        Computes the x,y positions of the group labels.
        """
        assert self.group_label_position in ["beginning", "middle", "end"]
        data = [self.graph.nodes[n][self.node_grouping] for n in self.nodes]
        node_length = len(data)
        groups = items_in_groups(data)
        radius = 1.02 * (self.plot_radius + self.nodeprops['radius'])
        xs = []
        ys = []
        has = []
        vas = []
        node_idcs = np.cumsum(list(groups.values()))
        node_idcs = np.insert(node_idcs, 0, 0)
        if self.group_label_position == "beginning":
            for idx in node_idcs[:-1]:
                x, y = get_cartesian(r=radius, theta=group_theta(node_length,
                                                                 idx))
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        elif self.group_label_position == "middle":
            node_idcs = node_idcs.reshape(len(node_idcs), 1)
            node_idcs = np.concatenate((node_idcs[:-1], node_idcs[1:]), axis=1)
            for idx in node_idcs:
                theta1 = group_theta(node_length, idx[0])
                theta2 = group_theta(node_length, idx[1]-1)
                x, y = get_cartesian(r=radius, theta=(theta1+theta2)/2)
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        elif self.group_label_position == "end":
            for idx in node_idcs[1::]:
                x, y = get_cartesian(r=radius, theta=group_theta(node_length,
                                                                 idx-1))
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        self.group_label_coords = {'x': xs, 'y': ys}
        self.group_label_aligns = {'has': has, 'vas': vas}
        self.groups = groups.keys()

    def compute_node_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        in the Circos plot.
        """
        xs = []
        ys = []
        node_r = self.nodeprops['radius']
        radius = circos_radius(n_nodes=len(self.graph.nodes()), node_r=node_r)
        self.plot_radius = radius
        self.nodeprops['linewidth'] = radius * 0.01
        for node in self.nodes:
            x, y = get_cartesian(r=radius, theta=node_theta(self.nodes, node))
            xs.append(x)
            ys.append(y)
        self.node_coords = {'x': xs, 'y': ys}

    def compute_node_label_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        label in the Circos plot.

        This method is always called after the compute_node_positions
        method, so that the plot_radius is pre-computed.
        This will also add a new attribute, `node_label_rotation` to the object
        which contains the rotation angles for each of the nodes. Together with
        the node coordinates this can be used to add additional annotations
        with rotated text.
        """
        xs = []
        ys = []
        has = []
        vas = []
        rotations = []
        for node in self.nodes:
            theta = node_theta(self.nodes, node)
            radius = 1.02 * (self.plot_radius + self.nodeprops['radius'])
            x, y = get_cartesian(r=radius, theta=theta)

            # Computes the text alignment
            if x == 0:
                ha = 'center'
            elif x > 0:
                ha = 'left'
            else:
                ha = 'right'
            if y == 0:
                va = 'center'
            elif y > 0:
                va = 'bottom'
            else:
                va = 'top'

            if self.rotate:
                va = "center"

            # Computes the text rotation
            theta_deg = to_degrees(theta)
            if theta_deg >= -90 and theta_deg < 90:   # right side
                rot = theta_deg
            else:  # left side
                rot = theta_deg - 180

            xs.append(x)
            ys.append(y)
            has.append(ha)
            vas.append(va)
            rotations.append(rot)

        self.node_label_coords = {'x': xs, 'y': ys}  # node label coordinates
        self.node_label_aligns = {'has': has, 'vas': vas}  # node label alignments  # noqa
        self.node_label_rotation = rotations

    def draw_nodes(self):
        """
        Renders nodes to the figure.
        """
        node_r = self.nodeprops['radius']
        lw = self.nodeprops['linewidth']
        for i, node in enumerate(self.nodes):
            x = self.node_coords['x'][i]
            y = self.node_coords['y'][i]
            color = self.node_colors[i]
            node_patch = patches.Circle((x, y), node_r,
                                        lw=lw, color=color,
                                        zorder=2)
            self.ax.add_patch(node_patch)
            if self.node_labels:
                label_x = self.node_label_coords['x'][i]
                label_y = self.node_label_coords['y'][i]
                label_ha = self.node_label_aligns['has'][i]
                label_va = self.node_label_aligns['vas'][i]
                rot = 0
                if self.rotate:
                    rot = self.node_label_rotation[i]

                self.ax.text(s=node,
                             x=label_x, y=label_y,
                             ha=label_ha, va=label_va, rotation=rot,
                             rotation_mode="anchor",
                             color=self.node_label_color[i], fontsize=10)

    def draw_edges(self):
        """
        Renders edges to the figure.
        """
        for i, (start, end) in enumerate(self.graph.edges()):
            start_theta = node_theta(self.nodes, start)
            end_theta = node_theta(self.nodes, end)
            verts = [get_cartesian(self.plot_radius, start_theta),
                     (0, 0),
                     get_cartesian(self.plot_radius, end_theta)]
            color = self.edge_colors[i]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, lw=1, edgecolor=color,
                                      zorder=1, **self.edgeprops)
            self.ax.add_patch(patch)

    def draw_group_labels(self):
        """
        Renders group labels to the figure.
        """
        for i, label in enumerate(self.groups):
            label_x = self.group_label_coords['x'][i]
            label_y = self.group_label_coords['y'][i]
            label_ha = self.group_label_aligns['has'][i]
            label_va = self.group_label_aligns['vas'][i]
            color = self.group_label_color[i]
            self.ax.text(s=label,
                         x=label_x, y=label_y,
                         ha=label_ha, va=label_va,
                         color=color)

# class HivePlot(BasePlot):
#     """
#     Plotting object for HivePlot.
#     """
#     def __init__(self, graph, node_order=None, node_size=None,
#                  node_grouping=None, node_color=None, edge_width=None,
#                  edge_color=None, data_types=None):
#
#         # Initialize using BasePlot
#         BasePlot.__init__(self, graph, node_order=node_order,
#                           node_size=node_size, node_grouping=node_grouping,
#                           node_color=node_color, edge_width=edge_width,
#                           edge_color=edge_color, data_types=data_types)
#
#     def set_xylims(self):
#         pass
#
#     def set_nodes(self, nodes):
#         """
#         Sets the nodes attribute.
#
#         Checks that `nodes` is a dictionary where the key is the group ID and
#         the value is a list of nodes that belong in that group.
#         """
#         assert isinstance(nodes, dict), 'nodes must be a dictionary'
#         for k, v in nodes.items():
#             assert isinstance(v, list) or isinstance(v, tuple),\
#                 'each value in the dictionary must be a list or tuple'
#         assert len(nodes.keys()) in [2, 3], 'there can only be 2 or 3 groups'
#
#         self.nodes = nodes
#
#     def set_major_angle(self):
#         """
#         Sets the major angle between the plot groups, in radians.
#         """
#         self.major_angle = 2 * np.pi / len(self.nodes.keys())
#
#     def set_minor_angle(self):
#         """
#         Sets the minor angle between the plot groups, in radians.
#         """
#         self.minor_angle = 2 * np.pi / 12
#
#     def set_nodecolors(self, nodecolors):
#         """
#         Sets the nodecolors.
#
#         `nodecolors` can be specified as either a single string, or as a
#         dictionary of group_identifier:strings.
#
#         Performs type checking to ensure that nodecolors is specified as a
#         dictionary where the keys must match up with the .nodes attribute.
#
#         Otherwise
#         """
#         assert isinstance(nodecolors, dict) or isinstance(nodecolors, str)
#         if isinstance(nodecolors, dict):
#             assert set(nodecolors.keys()) == set(self.nodes.keys())
#             self.nodecolors = nodecolors
#         elif isinstance(nodecolors, str):
#             self.nodecolors = dict()
#             for k in self.nodes.keys():
#                 self.nodecolors[k] = nodecolors
#
#     def has_edge_within_group(self, group):
#         """
#         Checks whether there are within-group edges or not.
#         """
#         assert group in self.nodes.keys(),\
#             "{0} not one of the group of nodes".format(group)
#         nodelist = self.nodes[group]
#         for n1, n2 in self.edges:
#             if n1 in nodelist and n2 in nodelist:
#                 return True
#
#     def compute_node_positions(self):
#         """
#         Computes the positions of each node on the plot.
#
#         Sets the node_coords attribute inherited from BasePlot.
#         """
#         xs = defaultdict(lambda: defaultdict(list))
#         ys = defaultdict(lambda: defaultdict(list))
#         for g, (grp, nodes) in enumerate(self.nodes.items()):
#             if self.has_edge_within_group(grp):
#                 for r, node in enumerate(nodes):
#                     theta = g * self.major_angle - self.minor_angle
#                     x, y = get_cartesian(r+2, theta)
#                     xs[grp]['minus'].append(x)
#                     ys[grp]['minus'].append(y)
#
#                     theta = g * self.major_angle + self.minor_angle
#                     x, y = get_cartesian(r+2, theta)
#                     xs[grp]['plus'].append(x)
#                     ys[grp]['plus'].append(y)
#             else:
#                 for r, node in enumerate(nodes):
#                     theta = g * self.major_angle
#                     x, y = get_cartesian(r+2, theta)
#                     xs[grp]['axis'].append(x)
#                     ys[grp]['axis'].append(y)
#         self.node_coords = dict(x=xs, y=ys)
#
#     def draw_nodes(self):
#         """
#         Renders nodes to the figure.
#         """
#         for grp, nodes in self.nodes.items():
#             for r, n in enumerate(nodes):
#                 if self.has_edge_within_group(grp):
#                     x = self.node_coords['x'][grp]['minus'][r]
#                     y = self.node_coords['y'][grp]['minus'][r]
#                     self.draw_node(grp, x, y)
#
#                     x = self.node_coords['x'][grp]['plus'][r]
#                     y = self.node_coords['y'][grp]['plus'][r]
#                     self.draw_node(grp, x, y)
#
#                 else:
#                     x = self.node_coords['x'][grp]['axis'][r]
#                     y = self.node_coords['y'][grp]['axis'][r]
#                     self.draw_node(grp, x, y)
#
#     def draw_node(self, group, x, y):
#         """
#         Convenience function for simplifying the code in draw_nodes().
#         """
#         circle = plt.Circle(xy=(x, y), radius=self.nodeprops['r'],
#                             color=self.nodecolors[group], linewidth=0)
#         self.ax.add_patch(circle)
#
#     def draw_edges(self):
#         """
#         Renders the edges to the figure.
#         """
#         pass
#
#     def draw_edge(self, n1, n2):
#         """
#
#         """
#         pass


class MatrixPlot(BasePlot):
    """
    Plotting object for the MatrixPlot.
    """
    def __init__(self, graph, node_order=None, node_size=None,
                 node_grouping=None, node_color=None, edge_width=None,
                 edge_color=None, data_types=None, nodeprops=None,
                 edgeprops=None):

        # Initialize using BasePlot
        BasePlot.__init__(self, graph, node_order=node_order,
                          node_size=node_size, node_grouping=node_grouping,
                          node_color=node_color, edge_width=edge_width,
                          edge_color=edge_color, data_types=data_types,
                          nodeprops=nodeprops, edgeprops=edgeprops)

        # The following atribute is specific to MatrixPlots
        self.cmap = cmaps['continuous'].mpl_colormap

    def draw(self):
        """
        Draws the plot to screen.

        Note to self: Do NOT call super(MatrixPlot, self).draw(); the
        underlying logic for drawing here is completely different from other
        plots, and as such necessitates a different implementation.
        """
        matrix = nx.to_numpy_matrix(self.graph, nodelist=self.nodes)
        self.ax.matshow(matrix, cmap=self.cmap)


class ArcPlot(BasePlot):
    """
    Plotting object for ArcPlot.
    """

    def compute_node_positions(self):
        """
        Computes nodes positions.

        Arranges nodes in a line starting at (x,y) = (0,0). Node radius is
        assumed to be equal to 0.5 units. Nodes are placed at integer
        locations.
        """
        xs = []
        ys = []

        for node in self.nodes:
            xs.append(self.nodes.index(node))
            ys.append(0)

        self.node_coords = {'x': xs, 'y': ys}

    def draw_nodes(self):
        """
        Draw nodes to screen.
        """
        node_r = 1
        for i, node in enumerate(self.nodes):
            x = self.node_coords['x'][i]
            y = self.node_coords['y'][i]
            color = self.node_colors[i]
            node_patch = patches.Ellipse((x, y), node_r, node_r,
                                         lw=0, color=color, zorder=2)
            self.ax.add_patch(node_patch)

    def draw_edges(self):
        """
        Renders edges to the figure.
        """
        for i, (start, end) in enumerate(self.graph.edges()):
            start_idx = self.nodes.index(start)
            start_x = self.node_coords['x'][start_idx]
            start_y = self.node_coords['y'][start_idx]

            end_idx = self.nodes.index(end)
            end_x = self.node_coords['x'][end_idx]
            end_y = self.node_coords['y'][end_idx]

            arc_radius = abs(end_x - start_x) / 2
            # we do min(start_x, end_x) just in case start_x is greater than
            # end_x.
            middle_x = min(start_x, end_x) + arc_radius
            middle_y = arc_radius * 2

            verts = [(start_x, start_y),
                     (middle_x, middle_y),
                     (end_x, end_y)]

            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            path = Path(verts, codes)
            patch = patches.PathPatch(path, lw=1, zorder=1, **self.edgeprops)
            self.ax.add_patch(patch)

    def draw(self):
        super(ArcPlot, self).draw()
        xlimits = (-1, len(self.nodes) + 1)
        self.ax.set_xlim(*xlimits)
        self.ax.set_ylim(*xlimits)
