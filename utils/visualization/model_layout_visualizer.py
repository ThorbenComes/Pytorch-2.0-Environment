import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def stepLayout(blocks):
    scale = 1
    center = np.array([.5, .5])
    output_dict = {}

    step_number = 0
    for b in blocks:
        if b.earliest > step_number:
            step_number = b.earliest
    # now we got the total number of steps
    # 1 / amount of steps
    y_divisor = 1 / (step_number + 1)

    nodes_per_step = [0 for _ in range(step_number + 1)]

    for b in blocks:
        nodes_per_step[b.earliest] += 1

    x_divisors = []
    # 1 / amount of nodes for each step
    x_divisors.extend([1/(nodes_per_step[i]) for i in range(len(nodes_per_step))])
    x_done = [0 for _ in range(len(nodes_per_step))]

    for b in blocks:
        x_pos = x_done[b.earliest] * x_divisors[b.earliest] + x_divisors[b.earliest]/2
        x_done[b.earliest] += 1
        pos = [x_pos, b.earliest * y_divisor]
        output_dict.update({b.id: pos})

    # fix and scale positions
    for key in output_dict.keys():

        output_dict.update({key: np.array(output_dict[key]) + np.array([0, 1]) * (1/2) * (1/step_number)})
        output_dict.update({key: np.array([1, 1]) - output_dict[key]})
        output_dict.update({key: center + scale * (np.array(output_dict[key]) - center)})

    return output_dict


def create_view(graph, blocks):
    G = nx.DiGraph()
    G.add_edges_from(graph)
    plt.figure(figsize=(9, 9))
    labels = {}
    for b in blocks:
        labels.update({b.id: b.type})
    nx.draw_networkx(G, pos=stepLayout(blocks), labels=labels)

    # ax = plt.gca()
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    plt.show()
