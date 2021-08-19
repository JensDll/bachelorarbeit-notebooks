import time
import tflite_runtime.interpreter as tflite
import platform
import numpy as np
import os

EDGETUP_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

delegates = [tflite.load_delegate(EDGETUP_LIB)]

insect_model_cpu = tflite.Interpreter("mobilenet_v2_1.0_224_inat_insect_quant.tflite")
insect_model_tpu = tflite.Interpreter(
    "mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite",
    experimental_delegates=delegates,
)

insect_model_cpu.allocate_tensors()
insect_model_tpu.allocate_tensors()

rng = np.random.default_rng(42)


def measure_performance(model: tflite.Interpreter):
    input_details = model.get_input_details()[0]
    input_index = input_details["index"]
    input_shape = input_details["shape"][1:]

    test_data = rng.integers(low=0, high=256, size=[1, *input_shape], dtype=np.uint8)

    inference_times = []

    for i in range(11):
        model.set_tensor(input_index, test_data)
        start = time.perf_counter()
        model.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        if not i == 0:
            inference_times.append(inference_time)

    return np.array(inference_times)


def write_to_file(result: np.ndarray, name):
    if not os.path.isdir("results"):
        os.mkdir("results")
    with open(f"results/{name}.npy", "wb") as f:
        np.save(f, result)


result = measure_performance(insect_model_cpu)
write_to_file(result, "mobilenet_cpu_quant")
print(f"CPU Inferenzgeschwindigkeit: {result.sum():.4f} ms")

result = measure_performance(insect_model_tpu)
write_to_file(result, "mobilenet_tpu")
print(f"TPU Inferenzgeschwindigkeit: {result.sum():.4f} ms")
