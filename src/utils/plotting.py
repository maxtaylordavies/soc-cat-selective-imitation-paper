import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")


def make_condition_barplots(data, flavour="binary", filename=None, format="svg"):
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 8))
    for i, vis in enumerate([False, True]):
        df = data[data["own_group_visible"] == vis]

        tmp = df.loc[df["agents_known"] == True]
        barplot(tmp, flavour=flavour, ax=axs[0, i], show_x_label=False, show_y_label=i == 0)

        tmp = df.loc[(df["agents_known"] == False) & (df["groups_relevant"] == True)]
        barplot(tmp, flavour=flavour, ax=axs[1, i], show_x_label=False, show_y_label=i == 0)

        tmp = df.loc[(df["agents_known"] == False) & (df["groups_relevant"] == False)]
        barplot(tmp, flavour=flavour, ax=axs[2, i], show_x_label=True, show_y_label=i == 0)

    fig.tight_layout()

    filename = filename or f"barplots_{flavour}"
    fig.savefig(f"results/{filename}.{format}")


def barplot(data, ax, flavour="binary", show_x_label=True, show_y_label=True):
    if len(data) == 0:
        return

    # estimator = lambda x: sum(x) / len(x) if flavour == "binary" else "mean"
    estimator = "mean"

    x_label = "Agent" if show_x_label else ""
    y_label = "Imitation" if show_y_label else ""

    plot = sns.barplot(
        data=data,
        x="agent",
        y=f"similarity",
        palette=["#01C58A", "#1100D1"],
        estimator=estimator,
        ax=ax,
    )
    plot.axhline(0.5, ls="--", color="black")
    plot.set(ylim=(0, 1), xlabel=x_label, ylabel=y_label)


def surfaceplot(
    data, keys, labels, ticks=None, ticklabels=None, filename="surf", format="svg"
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        data[keys[0]], data[keys[1]], data[keys[2]], cmap=plt.cm.viridis, linewidth=0.2
    )
    ax.set_xlabel(labels[0], fontsize=14)
    ax.set_ylabel(labels[1], fontsize=14)
    ax.set_zlabel(labels[2], fontsize=14)
    ax.view_init(25, 160)

    if ticks:
        if "x" in ticks:
            ax.set_xticks(ticks["x"])
        if "y" in ticks:
            ax.set_yticks(ticks["y"])
        if "z" in ticks:
            ax.set_zticks(ticks["z"])

    if ticklabels:
        if "x" in ticklabels:
            ax.set_xticklabels(ticklabels["x"])
        if "y" in ticklabels:
            ax.set_yticklabels(ticklabels["y"])
        if "z" in ticklabels:
            ax.set_zticklabels(ticklabels["z"])

    fig.savefig(f"{filename}.{format}")
