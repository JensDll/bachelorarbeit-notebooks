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
        ("data/sine_cpu.npy", "y-", "Sinus CPU"),
        ("data/sine_cpu_quant.npy", "b-", "Sinus quantisiert CPU"),
        ("data/sine_tpu.npy", "g-", "Sinus Edge TPU"),
    )

    for npy, style, label in styles:
        data = np.load(npy)
        # plt.xlim(1, len(data))
        plt.plot(range(1, len(data) + 1), data, style, linewidth=0.7, label=label)

    plt.legend()
    plt.grid()
    plt.ylim([-0.02, 0.6])
    plt.xlabel("Messung")
    plt.ylabel("Inferenzgeschwindigkeit (ms)")
    save_fig("sine_model_inference_time")
    plt.show()


def plot_mobilenet():
    styles = (
        ("data/mobilenet_cpu_quant.npy", "b-", "MobileNet V2 CPU"),
        ("data/mobilenet_tpu.npy", "g-", "MobileNet V2 Edge TPU"),
    )

    for npy, style, label in styles:
        data = np.load(npy)
        # plt.xlim([1, len(data)])
        plt.plot(range(1, len(data) + 1), data, style, linewidth=0.7, label=label)

    plt.legend()
    plt.grid()
    plt.xlabel("Messung")
    plt.ylabel("Inferenzgeschwindigkeit (ms)")
    save_fig("mobilenet_inference_time")
    plt.show()


plot_sine()
plot_mobilenet()
