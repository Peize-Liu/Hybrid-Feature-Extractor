#include <iostream>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include "SuperPoint.hpp"

int main(int argc, char** argv){
    if (argc < 2){
        std::cout << "Usage: ./superpoint_test config.yaml" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node sp_config = config["superpoint_extractor"];
    SuperPoint::SuperPointConfig params;
    spdlog::set_level(spdlog::level::debug);
    params.onnx_file = sp_config["onnx_file"].as<std::string>();
    // params.engine_path = sp_config["engine_dir"].as<std::string>();
    YAML::Node input_tensor_names = sp_config["input_tensor_names"];
    for (YAML::const_iterator it = input_tensor_names.begin(); it != input_tensor_names.end(); ++it){
        params.input_tensor_names.push_back(it->as<std::string>());
    }
    YAML::Node output_tensor_names = sp_config["output_tensor_names"];
    for (YAML::const_iterator it = output_tensor_names.begin(); it != output_tensor_names.end(); ++it){
        params.output_tensor_names.push_back(it->as<std::string>());
    }

    // params.input_tensor_names = sp_config["input_tensor_names"].as<std::vector<std::string>>();
    // params.output_tensor_names = sp_config["output_tensor_names"].as<std::vector<std::string>>();
    params.input_width = sp_config["input_width"].as<int32_t>();
    params.input_height = sp_config["input_height"].as<int32_t>();
    params.dla_core = sp_config["enable_dla"].as<int32_t>();
    std::string test_image_path = sp_config["test_image"].as<std::string>();
    params.engine_file = sp_config["engine_file"].as<std::string>();
    spdlog::info("Engine file: {}", params.engine_file);
    // params.streams = sp_config["streams"].as<int32_t>();
    params.keypoint_threshold = sp_config["threshold"].as<float>();
    // params.nms_radius = sp_config["nms_radius"].as<int32_t>();
    // params.max_point = sp_config["max_point"].as<int32_t>();


    //print output and input tensor names if engine is already built, use built engine
    spdlog::info("Engine:{} input_tensor_names  {} output_tensor_names {} {} input width {} and height {}",params.engine_file, 
        params.input_tensor_names[0], params.output_tensor_names[0], params.output_tensor_names[1], params.input_width, params.input_height);
    SuperPoint sp(params);
    sp.build();
    cv::Mat image = cv::imread(test_image_path, cv::IMREAD_GRAYSCALE);
    std::vector<Eigen::Vector2f> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    if(sp.infer(image, keypoints, descriptors)){
        spdlog::info("Inference successful");
    } else {
        spdlog::error("Inference failed");
    }
    sp.visualization(test_image_path, image,keypoints);
    return 0;
}