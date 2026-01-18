# NDT topics for recording
# Sourced by run_demo.sh

# NDT output topics
NDT_OUTPUT_TOPICS=(
    /localization/pose_estimator/pose
    /localization/pose_estimator/pose_with_covariance
    /localization/pose_estimator/ndt_marker
    /localization/pose_estimator/points_aligned
    /localization/pose_estimator/monte_carlo_initial_pose_marker
    /localization/pose_estimator/transform_probability
    /localization/pose_estimator/nearest_voxel_transformation_likelihood
    /localization/pose_estimator/iteration_num
    /localization/pose_estimator/exe_time_ms
    /localization/pose_estimator/initial_pose_with_covariance
    /localization/pose_estimator/initial_to_result_distance
    /localization/pose_estimator/initial_to_result_relative_pose
)

# NDT input topics (for debugging)
NDT_INPUT_TOPICS=(
    /localization/pose_twist_fusion_filter/biased_pose_with_covariance
    /localization/util/downsample/pointcloud
)

# All NDT topics combined
NDT_TOPICS=("${NDT_OUTPUT_TOPICS[@]}" "${NDT_INPUT_TOPICS[@]}")
