#include "SuperPoint.hpp"
#include <iostream>
#include <spdlog/spdlog.h>
#include "logger.h"

SuperPoint::SuperPoint(const SuperPointConfig &super_point_config):
    super_point_config_(super_point_config),engine_(nullptr),context_(nullptr){
    tensorrt_log::setReportableSeverity(tensorrt_log::Logger::Severity::kINFO);
}


bool SuperPoint::build() {
    if(deserializeEngine()){
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(tensorrt_log::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, tensorrt_log::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }
    
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMIN, Dims4(1, 1, 100, 100));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kOPT, Dims4(1, 1, 500, 500));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMAX, Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);
    
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }
    auto profile_stream = tensorrt_common::makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);
    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(tensorrt_log::gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }
    saveEngine();
    ASSERT(network->getNbInputs() == 1);
    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);
    ASSERT(network->getNbOutputs() == 2);
    semi_dims_ = network->getOutput(0)->getDimensions();
    ASSERT(semi_dims_.nbDims == 3);
    desc_dims_ = network->getOutput(1)->getDimensions();
    ASSERT(desc_dims_.nbDims == 4);
    return true;
}

bool SuperPoint::constructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                   TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                   TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                   TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(super_point_config_.onnx_file.c_str(),
                                        static_cast<int>(tensorrt_log::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMaxWorkspaceSize(512_MiB);
    if (super_point_config_.fp_16) {
        config->setFlag(BuilderFlag::kFP16);
    }
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
    // enableDLA(builder.get(), config.get(), super_point_config_.dla_core);
    return true;
}

bool SuperPoint::infer(const cv::Mat &image, std::vector<Eigen::Vector2f> &keypoints, std::vector<Eigen::VectorXf> &descriptors) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }
    
    assert(engine_->getNbBindings() == 3);

    const int input_index = engine_->getBindingIndex(super_point_config_.input_tensor_names[0].c_str());

    context_->setBindingDimensions(input_index, Dims4(1, 1, image.rows, image.cols));

    BufferManager buffers(engine_, 0, context_.get());
    
    ASSERT(super_point_config_.input_tensor_names.size() == 1);
    if (!processInput(buffers, image)) {
        return false;
    }
    buffers.copyInputToDeviceAsync();
    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream);
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }
    buffers.copyOutputToHostAsync();
    if (!processOutput(buffers, keypoints ,descriptors)) {
        return false;
    }
    return true;
}

bool SuperPoint::processInput(const BufferManager &buffers, const cv::Mat &image) {
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;
    auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(super_point_config_.input_tensor_names[0]));
    cv::Mat mono_image;
    image.convertTo(mono_image, CV_32FC1, 1.0 / 255.0);
    memcpy(host_data_buffer, mono_image.data, mono_image.rows * mono_image.cols * sizeof(float));
    return true;
}

//replace to NMS
void SuperPoint::findHighScoreIndex(std::vector<float> &scores, std::vector<Eigen::Vector2f> &keypoints,
                                    int h, int w, float threshold) {
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            Eigen::Vector2f location = {i % w, int(i / w)}; //u,v
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}


void SuperPoint::removeBorders( std::vector<Eigen::Vector2f> &keypoints, std::vector<float> &scores, int border,
                                int height,
                                int width) {
    std::vector<Eigen::Vector2f> keypoints_selected;
    std::vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][1] >= border) && (keypoints[i][1] < (height - border)); //v 
        bool flag_w = (keypoints[i][0] >= border) && (keypoints[i][0] < (width - border)); //u
        if (flag_h && flag_w) {
            keypoints_selected.push_back(keypoints[i]);
            scores_selected.push_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

std::vector<size_t> SuperPoint::sortIndexes(std::vector<float> &data) {
    std::vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::topKeypoints( std::vector<Eigen::Vector2f> &keypoints, std::vector<float> &scores, int k) {
    if (k < keypoints.size() && k != -1) {
        std::vector<Eigen::Vector2f> keypoints_top_k;
        std::vector<float> scores_top_k;
        std::vector<size_t> indexes = sortIndexes(scores);
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.push_back(keypoints[indexes[i]]);
            scores_top_k.push_back(scores[indexes[i]]);
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}

void normalize_keypoints(const std::vector<Eigen::Vector2f> &keypoints, std::vector<Eigen::Vector2f> &keypoints_norm,
                    int h, int w, int s) {
    for (auto &keypoint : keypoints) {
        Eigen::Vector2f kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}

int clip(int val, int max) {
    if (val < 0) return 0;
    return std::min(val, max - 1);
}

void grid_sample(const float *input, std::vector<Eigen::Vector2f> &grid,
                 std::vector<Eigen::VectorXf> &output, int dim, int h, int w) {
    // descriptors 1x256x(W/8)X(H/8)
    // keypoints 1x1xDynmx2
    // out 1x256x1xnumber
    for (auto &g : grid) {
        float ix = ((g[0] + 1) / 2) * (w - 1);
        float iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        Eigen::VectorXf descriptor(kSuperPointDescDim,1);
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 whd
            // x * Height * Depth + y * Depth + z
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor[i] = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }
}

void normalize_descriptors(std::vector<Eigen::VectorXf> &dest_descriptors) {
    for (auto &descriptor : dest_descriptors) {
        double norm_inv = 1.0 / descriptor.norm();
        descriptor *= norm_inv;
    }
}

void SuperPoint::sampleDescriptors(std::vector<Eigen::Vector2f> &keypoints, float *des_map,
                                    std::vector<Eigen::VectorXf> &descriptors, int dim, int h, int w, int s) {
    std::vector<Eigen::Vector2f> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    grid_sample(des_map, keypoints_norm, descriptors, dim, h, w);
    normalize_descriptors(descriptors);
}

bool SuperPoint::processOutput(const BufferManager &buffers, std::vector<Eigen::Vector2f> &keypoints,
                        std::vector<Eigen::VectorXf>  &descriptors) {
    keypoints.clear();
    descriptors.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[0]));
    auto *output_desc = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[1]));
    int semi_feature_map_h = semi_dims_.d[1];
    int semi_feature_map_w = semi_dims_.d[2];
    std::vector<float> scores_vec(output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    
    findHighScoreIndex(scores_vec, keypoints, semi_feature_map_h, semi_feature_map_w,
                          super_point_config_.keypoint_threshold);
    removeBorders(keypoints, scores_vec, super_point_config_.remove_borders, semi_feature_map_h, semi_feature_map_w);
    topKeypoints(keypoints, scores_vec, super_point_config_.max_keypoints);
    spdlog::debug("super point number is {}", keypoints.size());
    descriptors.resize(scores_vec.size());
    int desc_feature_dim = desc_dims_.d[1];
    int desc_feature_map_h = desc_dims_.d[2];
    int desc_feature_map_w = desc_dims_.d[3];
    sampleDescriptors(keypoints, output_desc, descriptors, desc_feature_dim, desc_feature_map_h, desc_feature_map_w);
    return true;
}


void SuperPoint::visualization(const std::string &image_name, const cv::Mat &image,std::vector<Eigen::Vector2f> &keypoints ) {
    cv::Mat image_display;
    if(image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();
    for (auto &keypoint : keypoints) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])), 1, cv::Scalar(255, 0, 0), -1, 16);
    }
    cv::imshow("superpoint", image_display);
    cv::waitKey(0);
}

void SuperPoint::saveEngine(){
    if (super_point_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(super_point_config_.engine_file, std::ios::binary);
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool SuperPoint::deserializeEngine(){
    std::ifstream file(super_point_config_.engine_file.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(tensorrt_log::gLogger);
        if (runtime == nullptr) {
            delete[] model_stream;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        delete[] model_stream;
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}

