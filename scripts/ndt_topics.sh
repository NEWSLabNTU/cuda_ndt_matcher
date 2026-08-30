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

# Degeneracy: how well each frame's geometry constrained the pose. Recorded
# because the narrow-FOV question is about which axes go unobserved, and that is
# invisible in the pose output until it has already gone wrong.
NDT_DEGENERACY_TOPICS=(
    /localization/pose_estimator/degeneracy/translation_anisotropy
    /localization/pose_estimator/degeneracy/rotation_anisotropy
    /localization/pose_estimator/degeneracy/translation_min_eigenvalue
    /localization/pose_estimator/degeneracy/rotation_min_eigenvalue
)

# NDT input topics (for debugging)
NDT_INPUT_TOPICS=(
    /localization/pose_twist_fusion_filter/biased_pose_with_covariance
    /localization/util/downsample/pointcloud
)

# All NDT topics combined
NDT_TOPICS=("${NDT_OUTPUT_TOPICS[@]}" "${NDT_INPUT_TOPICS[@]}" "${NDT_DEGENERACY_TOPICS[@]}")
