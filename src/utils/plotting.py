import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd

sns.set_theme(context="paper", style="darkgrid")
pylab.rcParams.update(
    {
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        # "xtick.labelsize": "x-large",
        # "ytick.labelsize": "x-large",
    }
)


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


def plot_strategy_performance(data, weights, phis, vselfs, model_name, filename):
    fig, axs = plt.subplots(3, len(vselfs), figsize=(12, 8), sharex=False, sharey="row")
    palette = sns.color_palette("viridis", n_colors=len(phis))

    for i, vself in enumerate(vselfs):
        df = data[data["vself"] == i]

        sns.kdeplot(
            data=df,
            stat="probability",
            x="vx",
            y="vy",
            hue="phi",
            alpha=0.7,
            zorder=2,
            levels=5,
            fill=True,
            legend=False,
            ax=axs[0, i],
            palette=palette,
        )
        axs[0, i].set(
            xlabel="$\mathbf{v}_x$",
            ylabel="$\mathbf{v}_y$",
            title=f"$\mathbf{{v}}^{{(self)}}=[{vself[0]},{vself[1]}]$",
        )

        # add red x at vself
        axs[0, i].scatter(
            vself[0],
            vself[1],
            marker="x",
            color="red",
            s=100,
            linewidth=2.5,
            zorder=3,
        )

        # plot imitation rewards
        sns.stripplot(
            data=df,
            x="phi",
            y="reward",
            hue="phi",
            dodge=True,
            alpha=0.2,
            zorder=1,
            legend=False,
            ax=axs[1, i],
            palette=palette,
        )
        sns.pointplot(
            data=df,
            x="phi",
            y="reward",
            hue="phi",
            join=False,
            dodge=0.8 - 0.8 / 3,
            scale=1.25,
            errorbar=None,
            ax=axs[1, i],
            palette=palette,
        )
        ylabel = "Imitation reward" if i == 0 else None
        axs[1, i].set(xlabel=None, xticks=[], xticklabels=[], ylabel=ylabel)
        axs[1, i].legend([], [], frameon=False)

        # plot computed weights
        sns.barplot(
            x=np.array(phis),
            y=np.array(weights[model_name][i]),
            ax=axs[2, i],
            palette=palette,
        )
        ylabel = "Weight" if i == 0 else None
        axs[2, i].set(ylim=(0, 1), xlabel="$\phi$", ylabel=ylabel)

    fig.suptitle(model_name, fontsize=16)
    fig.tight_layout()
    fig.savefig(filename)


def make_condition_barplots(
    data, flavour="binary", plot_dir="results/tmp", filename=None, formats=["svg"]
):
    forbidden_combinations = {
        (True, "arbitrary"),  # (groups relevant, own group label)
        (False, "matched"),
        (False, "mismatched"),
    }

    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(16, 8))
    for col, own_group in enumerate(["hidden", "arbitrary", "matched", "mismatched"]):
        df = data[data["own group label"] == own_group]

        # tmp = df.loc[df["agents known"] == True]
        # barplot(tmp, flavour=flavour, ax=axs[0, i], legend=i == 0)

        df = df.loc[df["agents known"] == False]
        for row, groups_relevant in enumerate([False, True]):
            tmp = df.loc[df["groups relevant"] == groups_relevant]
            if (groups_relevant, own_group) in forbidden_combinations:
                continue
            barplot(tmp, flavour=flavour, ax=axs[row, col])

    fig.tight_layout()
    for fmt in formats:
        filename = filename or f"barplots"
        fig.savefig(f"{plot_dir}/{filename}.{fmt}")


def barplot_stacked(data, ax, flavour="binary", show_x_label=False, show_y_label=False):
    if len(data) == 0:
        return

    so.Plot(
        data=data,
        x="strategy",
        y="imitation",
        color="same group",
        # palette=["#01C58A", "#1100D1"],
        # ax=ax,
    ).add(so.Bar(), so.Agg("sum"), so.Norm(func="sum", by=["x"]), so.Stack()).scale(
        color=["#01C58A", "#1100D1"]
    ).on(
        ax
    ).plot()

    ax.axhline(0.5, ls="--", color="black")

    x_label = "Strategy" if show_x_label else ""
    y_label = "% imitation agent 1" if show_y_label else ""
    ax.set(ylim=(0, 1), xlabel=x_label, ylabel=y_label)


def barplot(
    data, ax, flavour="binary", show_x_label=False, show_y_label=False, legend=False
):
    if len(data) == 0:
        return

    # estimator = lambda x: sum(x) / len(x) if flavour == "binary" else "mean"
    estimator = lambda x: 100 * sum(x) / len(x) if len(x) > 0 else 0

    x_label = "Strategy" if show_x_label else ""
    y_label = "% imitation agent 1" if show_y_label else ""

    plot = sns.barplot(
        data,
        x="strategy",
        hue="group",
        y=f"imitation",
        palette=["#01C58A", "#1100D1"],
        estimator=estimator,
        ax=ax,
        legend=legend,
    )
    plot.axhline(50, ls="--", color="black")
    plot.set(ylim=(0, 100), xlabel=x_label, ylabel=y_label)
