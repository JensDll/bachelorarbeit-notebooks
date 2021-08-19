#!/bin/bash

[ -e "models" ] && rm -ri models

mkdir models && \
wget -q -O - "https://github.com/JensDll/bachelorarbeit-notebooks/releases/download/$1/sine_models.tar.gz" | \
    tar --extract --gzip --directory=models -f -
wget -O models/mobilenet_benchmark.py "https://raw.githubusercontent.com/JensDll/bachelorarbeit-notebooks/$1/first-neural-network/benchmark/mobilenet_benchmark.py" && \
wget -O models/sine_benchmark.py "https://raw.githubusercontent.com/JensDll/bachelorarbeit-notebooks/$1/first-neural-network/benchmark/sine_benchmark.py" && \
wget -O models/mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite" && \
wget -O models/mobilenet_v2_1.0_224_inat_insect_quant.tflite \
    "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_insect_quant.tflite"