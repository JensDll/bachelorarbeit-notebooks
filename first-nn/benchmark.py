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

sine_model_cpu = tflite.Interpreter("sine_model_quant.tflite")
sine_model_tpu = tflite.Interpreter(
    "sine_model_quant_edgetpu.tflite", experimental_delegates=delegates
)

sine_model_cpu.allocate_tensors()
sine_model_tpu.allocate_tensors()

input_details_cpu = sine_model_cpu.get_input_details()[0]
input_details_tpu = sine_model_tpu.get_input_details()[0]

input_index_cpu = input_details_cpu["index"]
input_index_tpu = input_details_tpu["index"]

input_shape = input_details_cpu["shape"]

rng = np.random.default_rng(42)
test_data = rng.integers(low=0, high=6, size=[200000, *input_shape], dtype=np.int8)

result = 0
for i, value in enumerate(test_data):
    sine_model_cpu.set_tensor(input_index_cpu, value)
    start = time.perf_counter()
    sine_model_cpu.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    if i > 10:
        result += inference_time
print(f"CPU Inferenz Zeit: {result} ms")

result = 0
for i, value in enumerate(test_data):
    sine_model_cpu.set_tensor(input_index_tpu, value)
    start = time.perf_counter()
    sine_model_cpu.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    if i > 10:
        result += inference_time
print(f"TPU Inferenz Zeit: {result} ms")
