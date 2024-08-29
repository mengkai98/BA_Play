#include "ceres/ceres.h"
#include "types.h"
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
class BASolver
{
  private:
    std::vector<FeaturePoint> feature_points;
    std::vector<KeyFrame> key_frames;

    std::vector<KeyFrame> gt_frames;

    Camera camera;

    struct OptimizationParam
    {
        int max_num_iterations = 10;
        std::string save_dir = "./";
        int fixed_key_frames = 2;
        bool fixed_feature_point = false;
        std::string to_string()
        {
            std::stringstream ss;
            ss << "\n max_num_iterations: " << max_num_iterations
               << "\n fixed_feature_point: " << fixed_feature_point << "\n save_dir: " << save_dir
               << "\n fixed_key_frames: " << fixed_key_frames;
            return ss.str();
        }

    } optimization_param;
    void readDataset(const std::string &noise_file_path, const std::string &gt_file_path);
    void saveData(const std::string &file_path) const;
    void optimization();
    void addFeaturePointConstraint(ceres::Problem &problem);
    double evaluate();

  public:
    BASolver(const std::string &noise_file_path, const std::string &gt_file_path);
};
