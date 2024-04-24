FROM nvcr.io/nvidia/tensorrt:23.04-py3
ARG OPENCV_VERSION=4.8.0

#Install some basic tools
RUN apt-get -y update && \
    apt-get install -y wget curl vim gdb

#Install VPI
RUN apt-get install  -y gnupg &&\
    apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc &&\
    apt install -y software-properties-common && \
    add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/focal r36.2 main' && \
    apt update && \
    apt install -y libnvvpi3 vpi3-dev vpi3-samples

#Install 3rd party
#Install spdlog
RUN   wget https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.tar.gz && \
      tar -zxvf v1.12.0.tar.gz && \
      cd spdlog-1.12.0 && \
      mkdir build && cd build && \
      cmake .. && make -j$(nproc) && \
      make install

#Install OpenCV4 with CUDA
RUN   apt update && \
      apt install libgtk2.0-dev -y && \
      wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
      unzip opencv.zip && \
      rm opencv.zip && \
      git clone https://github.com/opencv/opencv_contrib.git -b ${OPENCV_VERSION}
RUN   cd opencv-${OPENCV_VERSION} && \
      mkdir build && cd build && \
      cmake .. \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr/local \
            -D WITH_CUDA=ON \
            -D WITH_CUDNN=ON \
            -D WITH_CUBLAS=ON \
            -D WITH_TBB=ON \
            -D OPENCV_DNN_CUDA=ON \
            -D OPENCV_ENABLE_NONFREE=ON \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
            -D BUILD_EXAMPLES=OFF \
            -D BUILD_opencv_java=OFF \
            -D BUILD_opencv_python=OFF \
            -D BUILD_TESTS=OFF \
            -D BUILD_PERF_TESTS=OFF \
            -D BUILD_opencv_apps=OFF \
            -D BUILD_LIST=calib3d,features2d,highgui,dnn,imgproc,imgcodecs,\
cudev,cudaoptflow,cudaimgproc,cudalegacy,cudaarithm,cudacodec,cudastereo,\
cudafeatures2d,xfeatures2d,tracking,stereo,\
aruco,videoio,ccalib && \
      make -j$(nproc) && \
      make install

#Install yaml-cpp
RUN wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.zip -O yaml-cpp.zip && \
    unzip yaml-cpp.zip && \
    rm yaml-cpp.zip && \
    cd yaml-cpp-0.8.0 && \
    mkdir build && cd build && \
    cmake -DBUILD_SHARED_LISB=on .. && make -j$(nproc) && \
    make install

#Install Eigen
RUN wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip Eigen.zip && \
    cd eigen-3.4.0 && \
    mkdir build && cd build && \
    cmake .. && \
    make install


