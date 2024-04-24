DOCKERIMAGE="uno:x86"
xhost +
echo "[INFO] Start docker container with mapping current dir to docker container"
CURRENT_DIR=$(pwd)

#check container exist or not.
if [ "$(docker ps -a -q -f name=UNO_x86_container)" ]; then
  # attach to the container
  echo "[INFO] Container already exist, attach to the container"
  docker start UNO_x86_container
  docker exec -it UNO_x86_container /bin/bash
else 
  # create new container
  echo "[INFO] Container not exist, create new container"
  docker run -it --runtime=nvidia --gpus all  --net=host -v ${CURRENT_DIR}:/root/workspace/hfe \
  -v /dev/:/dev/  --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name="UNO_x86_container" ${DOCKERIMAGE} /bin/bash 
fi

# docker run -it --runtime=nvidia --gpus all  --net=host -v ${CURRENT_DIR}:/root/workspace/hfe \
#   -v /dev/:/dev/  --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name="UNO_x86_container" ${DOCKERIMAGE} /bin/bash 