import matplotlib.pyplot as plt


def gen_power_law_dis(alpha: float, num_models: int) -> list[float]:
    rates = [(x + 1)**(-alpha) for x in range(num_models)]
    rates_sum = sum(rates)
    rates_ratio = [x / rates_sum for x in rates]

    return rates_ratio


def plot_single_graph(ax,
                      x_values,
                      y_values,
                      xlabel,
                      ylabel,
                      label,
                      marker="o"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_xticks(x_values)
    # ax.set_xticklabels(x_values)
    ax.plot(x_values, y_values, label=label, marker=marker, markersize=2)
    ax.grid()


def gen_config_with_power_law():
    # num_models = 19  # 12 x 7B; 4 x 13B; 2 x 30B; 1 x 65B
    num_models = 100
    alpha_lis = [0.7, 0.9, 1.3, 1.7, 2.1]
    max_rate_lis = [40]
    rate_scale_lis = [0.75]  # 25, 40

    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    x_label = 'Top-k Models (%)'
    y_label = 'Cumulative Rate (%)'

    x = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for alpha in alpha_lis:
        label = f"α: {alpha}"
        print(f"* α: {alpha}")
        for rate_scale in rate_scale_lis:
            rates_ratio = gen_power_law_dis(alpha, num_models)
            rates_ratio = sorted(rates_ratio, reverse=True)
            x_axis = x
            y_axis = [
                sum(rates_ratio[:r]) * 100 / sum(rates_ratio) for r in x_axis
            ]
            print(x_axis, y_axis)
            # y_axis = [x * 100 for x in rates_ratio]
            print(f"=== rate scale: {rate_scale}")
            plot_single_graph(ax, x_axis, y_axis, x_label, y_label, label)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,
               labels,
               loc='upper center',
               ncol=3,
               bbox_to_anchor=(0.5, 1.23))

    fig.tight_layout()

    # fig.savefig("benchmark/end_to_end/plot_alpha_rate_dist.jpg",
    fig.savefig("benchmark/end_to_end/plot_alpha_rate_dist.pdf",
                bbox_inches='tight',
                pad_inches=0.05)


if __name__ == "__main__":
    gen_config_with_power_law()
