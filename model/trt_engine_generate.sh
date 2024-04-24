#!/bin/bash

H=150
W=300
C=1
current_dir=$(pwd)

/usr/src/tensorrt/bin/trtexec --onnx=superpoint_v1_sim_int32.onnx --saveEngine=superpoint_v1_sim_int32_${H}_${W}.trt --shapes=input:1x1x${H}x${W} --best