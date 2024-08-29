#pragma once
#include "types.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct ReprojectPointErrorTerm
{
    double observed_u_;
    double observed_v_;
    Camera camera_;
    ReprojectPointErrorTerm(double observed_u, double observed_v, Camera camera)
        : observed_u_(observed_u), observed_v_(observed_v), camera_(camera)
    {
    }

    template <typename T>
    bool operator()(const T *const c_rotation, const T *const c_position, const T *const point_ptr,
                    T *residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> c_qua(c_rotation);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> c_pos(c_position);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> eigen_point(point_ptr);
        Eigen::Matrix<T, 3, 1> p_c = c_qua.conjugate() * (eigen_point - c_pos);
        Eigen::Matrix<T, 3, 1> predicted = camera_.intrinsics.cast<T>() * (p_c / p_c.z());

        residuals[0] = predicted[0] - observed_u_;
        residuals[1] = predicted[1] - observed_v_;

        return true;
    }
    static ceres::CostFunction *Create(const double observed_x, const double observed_y,
                                       const Camera camera)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectPointErrorTerm, 2, 4, 3, 3>(
            new ReprojectPointErrorTerm(observed_x, observed_y, camera)));
    }
};
