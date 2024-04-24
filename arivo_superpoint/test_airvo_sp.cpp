#include "super_point.hpp"
#include <iostream>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv){
    if (argc < 2){
        std::cout << "Usage: ./superpoint_test config.yaml" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node sp_config = config["superpoint_extractor"];
    SuperPointConfig params;
    spdlog::set_level(spdlog::level::debug);
    params.onnx_file = sp_config["onnx_file"].as<std::string>();
    params.engine_path = sp_config["engine_dir"].as<std::string>();
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
    std::string engine_file = std::string("super_point")+ "_" +std::to_string(params.input_width)+ "_" +
        std::to_string(params.input_height)+".engine";
    params.engine_file = params.engine_path + "/" + engine_file;
    spdlog::info("Engine file: {}", params.engine_file);
    // params.streams = sp_config["streams"].as<int32_t>();
    // params.threshold = sp_config["threshold"].as<float>();
    // params.nms_radius = sp_config["nms_radius"].as<int32_t>();
    // params.max_point = sp_config["max_point"].as<int32_t>();


    //print output and input tensor names if engine is already built, use built engine
    spdlog::info("Engine:{} input_tensor_names  {} output_tensor_names {} {} input width {} and height {}",params.engine_file, 
        params.input_tensor_names[0], params.output_tensor_names[0], params.output_tensor_names[1], params.input_width, params.input_height);
    SuperPoint sp(params);
    sp.build();

    cv::Mat image = cv::imread(test_image_path, cv::IMREAD_GRAYSCALE);
    Eigen::Matrix<double, 259, Eigen::Dynamic> features;
    if(sp.infer(image, features)){
        spdlog::info("Inference successful");
    } else {
        spdlog::error("Inference failed");
    }
    sp.visualization(test_image_path, image);
    return 0;
}