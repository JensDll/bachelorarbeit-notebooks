import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pathlib

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 16,
    }
)


def save_fig(name, tight_layout=True):
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    print(f"Saving figure {name} ...")
    if tight_layout:
        plt.tight_layout(pad=0.5)
    plt.savefig(f"plots/{name}.pdf", backend="pgf")


def plot_sine():
    styles = (
        ("results/sine_cpu.npy", "b-"),
        ("results/sine_tpu.npy", "g-"),
        ("results/sine_cpu_quant.npy", "y-"),
    )

    for npy, style in styles:
        data = np.load(npy)
        plt.plot(range(1, len(data) + 1), data, style, linewidth=0.8)

    save_fig("sine_model_inference_time")
    plt.show()


def plot_mobilenet():
    styles = (
        ("results/mobilenet_cpu_quant.npy", "b-"),
        ("results/mobilenet_tpu.npy", "g-"),
    )

    for npy, style in styles:
        data = np.load(npy)
        plt.plot(range(1, len(data) + 1), data, style, linewidth=0.8)

    save_fig("mobilenet_inference_time")
    plt.show()


plot_sine()
plot_mobilenet()
