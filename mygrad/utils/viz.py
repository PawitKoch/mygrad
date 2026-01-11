from mygrad.core.value import Value
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def trace(root: Value):
    nodes, edges = set(), set()

    def build_graph(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build_graph(child)

    build_graph(root)
    return nodes, edges


def draw_dot(root):
    nodes, edges = trace(root)
    dot = nx.DiGraph()

    for n in nodes:
        label = f"{n._op or 'leaf'}\n{n.data:.4f}\ngrad={n.grad:.4f}"
        color = (
            "lightgreen"
            if len(n._prev) == 0
            # Leaf/input nodes
            else "orange"
            if n._op
            else "lightblue"
        )
        dot.add_node(n, label=label, color=color)

    for u, v in edges:
        dot.add_edge(u, v)

    pos = nx.spring_layout(dot)
    labels = nx.get_node_attributes(dot, "label")
    colors = [dot.nodes[n]["color"] for n in dot.nodes]
    nx.draw(
        dot,
        pos,
        with_labels=True,
        labels=labels,
        node_size=1800,
        node_color=colors,
        font_size=8,
        edgecolors="black",
        linewidths=0.5,
    )
    plt.show()


def plot_decision_boundary(model, X, Y, resolution=100):
    # X: (N, 2), Y: (N, 1)
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = []

    for pt in grid:
        pred = model(Value(pt.reshape(1, -1))).data
        preds.append(pred.item())  # flatten Value → float

    zz = np.array(preds).reshape(xx.shape)

    # Plot filled contour
    plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red"])
    plt.colorbar()

    # Overlay training data
    for i, (x, y) in enumerate(zip(X, Y)):
        color = "red" if y == 1 else "blue"
        plt.scatter(x[0], x[1], color=color, edgecolor="k", s=100)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.grid(True)
    plt.show()
