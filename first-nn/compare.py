import time
import tensorflow as tf
import platform

EDGETUP_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

delegates = [tf.lite.experimental.load_delegate(EDGETUP_LIB)]

insect_model_cpu = tf.lite.Interpreter("mobilenet_v2_inat_insect_quant.tflite")

insect_model_tpu = tf.lite.Interpreter(
    "mobilenet_v2_inat_insect_quant_edgetpu.tflite", experimental_delegates=delegates
)

insect_model_cpu.allocate_tensors()
insect_model_tpu.allocate_tensors()

input_details_cpu = insect_model_cpu.get_input_details()[0]
input_details_tpu = insect_model_tpu.get_input_details()[0]

input_index_cpu = input_details_cpu["index"]
input_index_tpu = input_details_tpu["index"]

input_shape = input_details_cpu["shape"][1:]

test_data = tf.random.uniform(input_shape, minval=0, maxval=256)
test_data = tf.cast(test_data, dtype=tf.uint8)
test_data = tf.expand_dims(test_data, axis=0)

result = 0

for i in range(10):
    insect_model_cpu.set_tensor(input_index_cpu, test_data)
    start = time.perf_counter()
    insect_model_cpu.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    result += inference_time

print(f"CPU Inferenz Zeit: {result} ms")

result = 0

for i in range(11):
    insect_model_tpu.set_tensor(input_index_tpu, test_data)
    start = time.perf_counter()
    insect_model_tpu.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    if i > 0:
        result += inference_time

print(f"TPU Inferenz Zeit: {result} ms")
