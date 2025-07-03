#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/Detquark:/code/Detquark \
    -it --rm detquark_train bash -c "
echo '
from fire import Fire
from Detquark.model import main_torch2onnx

if __name__ == \"__main__\":
    Fire(main_classifier_torch2onnx)
' > /code/torch2onnx.py && python /code/torch2onnx.py --cfg_name $1
"