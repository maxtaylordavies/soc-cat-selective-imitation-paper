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

    # fig.savefig(f"{filename}.{format}")
    plt.show()


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
            dodge=True,
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


def make_barplots(data, plot_dir="results/tmp", filename="barplots"):
    # first make figure for known agents phase
    df = data.loc[data["agents known"] == True]
    if len(df):
        fig, ax = plt.subplots()
        barplot(
            df[~df["strategy"].str.contains("group")],
            ax=ax,
            legend=True,
        )
        # plt.show()
        save_figure(fig, f"{plot_dir}/{filename}_known")

    # then make figure for unknown agents phase
    df = data.loc[data["agents known"] == False]
    if len(df):
        conds = [  # (groups relevant, own group label)
            (False, "hidden"),
            (False, "arbitrary"),
            (True, "hidden"),
            (True, "matched"),
            (True, "mismatched"),
        ]
        fig, axs = plt.subplots(
            1, len(conds), sharex=True, sharey=True, figsize=(20, 4)
        )
        for i, (gr, ogl) in enumerate(conds):
            barplot(
                df.loc[(df["groups relevant"] == gr) & (df["own group label"] == ogl)],
                ax=axs[i],
                legend=i == 0,
            )
        save_figure(fig, f"{plot_dir}/{filename}_unknown")
        # plt.show()


def barplot(data, ax, y_label=False, legend=False):
    if len(data) == 0:
        return

    plot = sns.barplot(
        data,
        x="strategy",
        hue="group",
        y=f"imitation",
        palette=["#FF0000", "#0000FF"],
        errorbar="se",
        ax=ax,
        legend=legend,
    )
    if legend:
        handles, _ = plot.get_legend_handles_labels()
        plot.legend(handles=handles, labels=["red", "blue"], title="Excplicit group")
    plot.axhline(0.5, ls="--", color="black", alpha=0.5)
    plot.set(ylim=(0, 1), xlabel="", ylabel="% imitation" if y_label else "")


def save_figure(fig, path, formats=["pdf", "svg"]):
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}")
