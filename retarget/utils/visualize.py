import matplotlib.pyplot as plt

def plot_pose(pose, edges, ax=None, **kwargs):

    if not ax:
        ax = plt.axes(projection='3d')

    ax.view_init(90, -90)

    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], **kwargs)

    for i in range(edges.shape[0]):
        start = edges[i, 1]
        end = edges[i, 0]
        ax.plot([pose[start, 0], pose[end, 0]], [pose[start, 1], pose[end, 1]], [pose[start, 2], pose[end, 2]], **kwargs)

    ax.axis('equal')