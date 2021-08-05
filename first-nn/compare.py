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

insect_model_cpu = tflite.Interpreter("mobilenet_v2_inat_insect_quant.tflite")
insect_model_tpu = tflite.Interpreter(
    "mobilenet_v2_inat_insect_quant_edgetpu.tflite", experimental_delegates=delegates
)

insect_model_cpu.allocate_tensors()
insect_model_tpu.allocate_tensors()


rng = np.random.default_rng(42)


def measure_performance(model: tflite.Interpreter):
    input_details = model.get_input_details()[0]
    input_index = input_details["index"]
    input_shape = input_details["shape"][1:]

    test_data = rng.integers(low=0, high=256, size=[1, *input_shape], dtype=np.uint8)

    result = 0
    for i in range(11):
        model.set_tensor(input_index, test_data)
        start = time.perf_counter()
        model.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        if i > 0:
            result += inference_time
    return result


result = measure_performance(insect_model_cpu)
print(f"CPU Inferenz Zeit: {result} ms")
result = measure_performance(insect_model_tpu)
print(f"TPU Inferenz Zeit: {result} ms")
