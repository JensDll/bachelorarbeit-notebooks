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
    "sine_model_quant_edgetpu.tflite", experimental_delegates=delegates
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

    test_data = rng.random(size=[20000, *input_shape], dtype=np.float32) * 6

    if dtype == np.int8:
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(dtype)

    result = 0
    for i, value in enumerate(test_data):
        model.set_tensor(input_index, value)
        start = time.perf_counter()
        model.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        if i > 10:
            result += inference_time
    return result


result = measure_performance(sine_model_cpu)
print(f"CPU Inferenz Zeit: {result} ms")
result = measure_performance(sine_model_quant_cpu)
print(f"CPU quantisierte Inferenz Zeit: {result} ms")
result = measure_performance(sine_model_tpu)
print(f"TPU Inferenz Zeit: {result} ms")
