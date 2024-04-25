// Based on SuperPoint in AirVO
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_POINT_H_
#define SUPER_POINT_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include "buffers.h"

using tensorrt_common::TensorRTUniquePtr;
using tensorrt_buffer::BufferManager;

const int32_t kSuperPointDescDim = 256;
class SuperPoint {
public:
    struct SuperPointConfig {
        int max_keypoints = 100;
        int remove_borders = 1;
        int dla_core;
        int fp_16;
        int input_width;
        int input_height;
        float keypoint_threshold = 0.015;
        std::vector<std::string> input_tensor_names;
        std::vector<std::string> output_tensor_names;
        std::string onnx_file;
        std::string engine_file;
        std::string engine_path;
    };

    explicit SuperPoint(const SuperPointConfig &super_point_config);
    bool build();
    // bool infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features);
    bool infer(const cv::Mat &image, std::vector<Eigen::Vector2f> &keypoints, std::vector<Eigen::VectorXf> &descriptors);
    void visualization(const std::string &image_name, const cv::Mat &image,std::vector<Eigen::Vector2f> &keypoints);
    void saveEngine();
    bool deserializeEngine();
private:
    SuperPointConfig super_point_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<float>> descriptors_;
    bool constructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                           TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                           TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

    bool processInput(const tensorrt_buffer::BufferManager &buffers, const cv::Mat &image);

    // bool processOutput(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<float, kSuperPointDescDim, Eigen::Dynamic> &features);
    bool processOutput(const tensorrt_buffer::BufferManager &buffers, std::vector<Eigen::Vector2f> &keypoints,
                        std::vector<Eigen::VectorXf> &descriptors);

    void removeBorders( std::vector<Eigen::Vector2f> &keypoints, std::vector<float> &scores, int border, int height,
                        int width);

    std::vector<size_t> sortIndexes(std::vector<float> &data);

    void topKeypoints( std::vector<Eigen::Vector2f> &keypoints, std::vector<float> &scores, int k);

    void findHighScoreIndex(std::vector<float> &scores,  std::vector<Eigen::Vector2f> &keypoints, int h, int w,
                               float threshold);

    void sampleDescriptors( std::vector<Eigen::Vector2f> &keypoints, float *des_map,
                            std::vector<Eigen::VectorXf> &descriptors, int dim, int h, int w, int s = 8);
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif //SUPER_POINT_H_
