import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set_theme()


def make_condition_barplots(data, flavour="binary"):
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
    fig.savefig(f"../results/barplots_{flavour}.svg")


def barplot(data, ax, flavour="binary", show_x_label=True, show_y_label=True):
    print(len(data))

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
