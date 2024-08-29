#pragma once
#include "json.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

class PointObsInfo
{
  public:
    int frame_idx;                  //被哪个关键帧观测到
    Eigen::Vector3i pixel_location; // 像素坐标系坐标 [u,v,1]
    friend void to_json(nlohmann::json &j, const PointObsInfo &m)
    {
        j = nlohmann::json{{"kf_idx", m.frame_idx},
                           {"pixel_pos", {m.pixel_location.x(), m.pixel_location.y()}}};
    }

    friend void from_json(const nlohmann::json &j, PointObsInfo &m)
    {
        m.frame_idx = j["kf_idx"].get<int>();

        m.pixel_location[0] = j["pixel_pos"][0].get<int>();
        m.pixel_location[1] = j["pixel_pos"][1].get<int>();
        m.pixel_location[2] = 1;
    }
};

struct FeaturePoint
{
    int point_idx;
    Eigen::Vector3d point_world;
    std::vector<PointObsInfo> obs_infos;

    friend void to_json(nlohmann::json &j, const FeaturePoint &m)
    {
        j = nlohmann::json{
            {"point_idx", m.point_idx},
            {"point_world", {m.point_world.x(), m.point_world.y(), m.point_world.z()}},
            {"observation", m.obs_infos}};
    }

    friend void from_json(const nlohmann::json &j, FeaturePoint &m)
    {
        m.point_idx = j["point_idx"].get<int>();

        m.point_world[0] = j["point_world"][0].get<double>();
        m.point_world[1] = j["point_world"][1].get<double>();
        m.point_world[2] = j["point_world"][2].get<double>();
        m.obs_infos.resize(j["observation"].size());
        for (int i = 0; i < j["observation"].size(); i++)
        {
            m.obs_infos[i] = j["observation"][i].get<PointObsInfo>();
        }
    }
};

struct KeyFrame
{
    int kf_idx;
    int camera_idx;
    // Rp_c+P = p_w
    Eigen::Quaterniond rotation;
    Eigen::Vector3d position;
    friend void to_json(nlohmann::json &j, const KeyFrame &m)
    {
        j = nlohmann::json{
            {"kf_idx", m.kf_idx},
            {"camera_idx", m.camera_idx},
            {"rotation", {m.rotation.x(), m.rotation.y(), m.rotation.z(), m.rotation.w()}},
            {"position", {m.position.x(), m.position.y(), m.position.z()}}};
    }

    friend void from_json(const nlohmann::json &j, KeyFrame &m)
    {
        m.kf_idx = j["kf_idx"].get<int>();
        m.camera_idx = j["camera_idx"].get<int>();
        m.position[0] = j["position"][0].get<double>();
        m.position[1] = j["position"][1].get<double>();
        m.position[2] = j["position"][2].get<double>();

        m.rotation.x() = j["rotation"][0].get<double>();
        m.rotation.y() = j["rotation"][1].get<double>();
        m.rotation.z() = j["rotation"][2].get<double>();
        m.rotation.w() = j["rotation"][3].get<double>();
    }
};

class Camera
{

  public:
    // 300 dpi a = 11810 pixel/m;
    Eigen::Matrix3d intrinsics;
    int width, height; // pixel

    Camera()
    {
        intrinsics.setIdentity();
        // 300ppi 50mm  1M摄像头
        intrinsics(0, 0) = 590;
        intrinsics(1, 1) = 590;
        intrinsics(0, 2) = 512;
        intrinsics(1, 2) = 512;
        width = 1024;
        height = 1024;
    }

    inline Eigen::Vector3d reproject(const Eigen::Vector3d &point) const
    {

        return intrinsics * (point / point.z());
    }
    friend void to_json(nlohmann::json &j, const Camera &m)
    {
        j = nlohmann::json{
            {"intrinsics",
             {m.intrinsics(0, 0), m.intrinsics(0, 2), m.intrinsics(1, 1), m.intrinsics(1, 2)}},
            {"resolution", {m.width, m.height}}};
    }

    friend void from_json(const nlohmann::json &j, Camera &m)
    {
        m.intrinsics.setIdentity();
        m.intrinsics(0, 0) = j["intrinsics"][0].get<double>();
        m.intrinsics(0, 2) = j["intrinsics"][1].get<double>();
        m.intrinsics(1, 1) = j["intrinsics"][2].get<double>();
        m.intrinsics(1, 2) = j["intrinsics"][3].get<double>();
        m.width = j["resolution"][0].get<double>();
        m.height = j["resolution"][1].get<double>();
    }
};
