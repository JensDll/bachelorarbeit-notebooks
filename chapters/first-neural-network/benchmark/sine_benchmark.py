import os
import time
import tflite_runtime.interpreter as tflite
import platform
import numpy as np

EDGETUP_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

delegates = [tflite.load_delegate(EDGETUP_LIB)]

sine_model_cpu = tflite.Interpreter("sine_model.tflite")
sine_model_quant_cpu = tflite.Interpreter("sine_model_quant.tflite")
sine_model_tpu = tflite.Interpreter(
    "sine_model_quant_edgetpu.tflite",
    experimental_delegates=delegates,
)

sine_model_cpu.allocate_tensors()
sine_model_quant_cpu.allocate_tensors()
sine_model_tpu.allocate_tensors()

rng = np.random.default_rng(42)


def measure_performance(model: tflite.Interpreter):
    input_details = model.get_input_details()[0]
    input_index = input_details["index"]
    input_shape = input_details["shape"]
    dtype = input_details["dtype"]

    test_data = rng.random(size=[1010, *input_shape], dtype=np.float32) * 6

    if dtype == np.int8:
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(dtype)

    inference_times = []

    for i, value in enumerate(test_data):
        model.set_tensor(input_index, value)
        start = time.perf_counter()
        model.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        if i >= 10:
            inference_times.append(inference_time)

    return np.array(inference_times)


def write_to_file(result: np.ndarray, name):
    if not os.path.isdir("results"):
        os.mkdir("results")
    with open(f"results/{name}.npy", "wb") as f:
        np.save(f, result)


result = measure_performance(sine_model_cpu)
write_to_file(result, "sine_cpu")
print(
    f"CPU Inferenzgeschwindigkeit: {result.sum():.4f} ms - ({np.average(result):.4f})"
)

result = measure_performance(sine_model_quant_cpu)
write_to_file(result, "sine_cpu_quant")
print(
    f"CPU quantisierte Inferenzgeschwindigkeit: {result.sum():.4f} ms - ({np.average(result):.4f})"
)

result = measure_performance(sine_model_tpu)
write_to_file(result, "sine_tpu")
print(
    f"TPU Inferenzgeschwindigkeit: {result.sum():.4f} ms - ({np.average(result):.4f})"
)
