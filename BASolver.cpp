#include "BASolver.h"
#include "BaErrorTerm.h"
#include <fstream>

void BASolver::readDataset(const std::string &noise_file_path, const std::string &gt_file_path)
{
    std::ifstream f(noise_file_path);
    nlohmann::json json_data = nlohmann::json::parse(f);
    int fp_nums = json_data["feature_points"].size();
    feature_points.resize(fp_nums);
    for (int i = 0; i < fp_nums; i++)
    {
        int point_id = json_data["feature_points"][i]["point_idx"].get<int>();
        feature_points[point_id] = json_data["feature_points"][i].get<FeaturePoint>();
        assert(point_id == feature_points[point_id].point_idx);
    }

    int kf_nums = json_data["key_frames"].size();
    key_frames.resize(kf_nums);
    for (int i = 0; i < kf_nums; i++)
    {
        int kf_id = json_data["key_frames"][i]["kf_idx"].get<int>();
        key_frames[kf_id] = json_data["key_frames"][i].get<KeyFrame>();
        assert(kf_id == key_frames[kf_id].kf_idx);
    }
    camera = json_data["camera"].get<Camera>();
    f.close();

    std::ifstream gt_f(gt_file_path);
    json_data = nlohmann::json::parse(gt_f);
    kf_nums = json_data["key_frames"].size();
    gt_frames.resize(kf_nums);
    for (int i = 0; i < kf_nums; i++)
    {
        int kf_id = json_data["key_frames"][i]["kf_idx"].get<int>();
        gt_frames[kf_id] = json_data["key_frames"][i].get<KeyFrame>();
        assert(kf_id == gt_frames[kf_id].kf_idx);
    }
    gt_f.close();
}

void BASolver::saveData(const std::string &file_path) const
{
    nlohmann::json j;
    j["camera"] = camera;
    j["feature_points"] = feature_points;
    j["key_frames"] = key_frames;
    std::fstream json_record(file_path, std::ios::out);
    json_record << j;
    json_record.close();
}

void BASolver::optimization()
{
    ceres::Problem problem;

    addFeaturePointConstraint(problem);

    for (int i = 0; i < optimization_param.fixed_key_frames; i++)
    {
        problem.SetParameterBlockConstant(key_frames[i].rotation.coeffs().data());
        problem.SetParameterBlockConstant(key_frames[i].position.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = optimization_param.max_num_iterations;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = 15;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << '\n';
    saveData(optimization_param.save_dir + "optimization.json");
}

void BASolver::addFeaturePointConstraint(ceres::Problem &problem)
{
    ceres::LossFunction *loss_function = nullptr;
    ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
    for (auto &&point : feature_points)
    {
        for (auto &&obs : point.obs_infos)
        {
            ceres::CostFunction *cost_function = ReprojectPointErrorTerm::Create(
                obs.pixel_location.x(), obs.pixel_location.y(), camera);
            problem.AddResidualBlock(
                cost_function, loss_function, key_frames[obs.frame_idx].rotation.coeffs().data(),
                key_frames[obs.frame_idx].position.data(), point.point_world.data());
            problem.SetManifold(key_frames[obs.frame_idx].rotation.coeffs().data(),
                                quaternion_manifold);
        }
        if (optimization_param.fixed_feature_point)
        {
            problem.SetParameterBlockConstant(point.point_world.data());
        }
    }
}

double BASolver::evaluate()
{
    double apt = 0;
    int vertex_size = 0;
    for (size_t i = optimization_param.fixed_key_frames; i < key_frames.size(); i++)
    {
        Eigen::Vector3d p_gt = gt_frames[i].position;
        Eigen::Quaterniond q_gt = gt_frames[i].rotation;
        Eigen::Vector3d p_esti = key_frames[i].position;
        Eigen::Quaterniond q_esti = key_frames[i].rotation;

        double norm = (q_gt.conjugate() * (p_esti - p_gt)).norm();
        apt += norm * norm;
        vertex_size++;
    }
    apt = apt / vertex_size;
    apt = sqrt(apt);
    return apt;
}

BASolver::BASolver(const std::string &noise_file_path, const std::string &gt_file_path)
{
    readDataset(noise_file_path, gt_file_path);
    double before = evaluate();
    optimization();
    double after = evaluate();
    std::cout << "______________------________-----------_________" << std::endl;
    std::cout << "apt: before: " << before << " after: " << after << std::endl;
}