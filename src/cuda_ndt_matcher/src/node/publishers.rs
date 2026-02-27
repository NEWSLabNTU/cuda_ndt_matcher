use geometry_msgs::msg::Pose;
#[cfg(feature = "debug-markers")]
use ndt_cuda::AlignmentDebug;
use rclrs::{log_error, Publisher};
use std_msgs::msg::Header;
use tf2_msgs::msg::TFMessage;
use visualization_msgs::msg::Marker;
#[cfg(feature = "debug-markers")]
use visualization_msgs::msg::MarkerArray;

use super::state::NODE_NAME;

/// Publish TF transform from map frame to ndt_base_frame.
///
/// This matches Autoware's `publish_tf()` behavior in ndt_scan_matcher_core.cpp:
/// - Parent frame: map_frame (typically "map")
/// - Child frame: ndt_base_frame (typically "ndt_base_link")
/// - Transform: The NDT result pose
///
/// The TF is published to the `/tf` topic as a TFMessage containing a single
/// TransformStamped message.
pub(crate) fn publish_tf(
    tf_pub: &Publisher<TFMessage>,
    stamp: &builtin_interfaces::msg::Time,
    pose: &Pose,
    map_frame: &str,
    ndt_base_frame: &str,
) {
    // Convert Pose to Transform
    // Pose uses position/orientation, Transform uses translation/rotation
    let transform = geometry_msgs::msg::Transform {
        translation: geometry_msgs::msg::Vector3 {
            x: pose.position.x,
            y: pose.position.y,
            z: pose.position.z,
        },
        rotation: pose.orientation.clone(),
    };

    // Create TransformStamped message
    let transform_stamped = geometry_msgs::msg::TransformStamped {
        header: Header {
            stamp: stamp.clone(),
            frame_id: map_frame.to_string(),
        },
        child_frame_id: ndt_base_frame.to_string(),
        transform,
    };

    // Create TFMessage with the single transform
    let tf_msg = TFMessage {
        transforms: vec![transform_stamped],
    };

    // Publish to /tf
    if let Err(e) = tf_pub.publish(&tf_msg) {
        log_error!(NODE_NAME, "Failed to publish TF: {e}");
    }
}

/// Create an arrow marker representing a pose
pub(crate) fn create_pose_marker(header: &Header, pose: &Pose, id: i32) -> Marker {
    Marker {
        header: header.clone(),
        ns: "result_pose_matrix_array".to_string(),
        id,
        type_: 0,  // ARROW
        action: 0, // ADD
        pose: pose.clone(),
        scale: geometry_msgs::msg::Vector3 {
            x: 0.3,
            y: 0.1,
            z: 0.1,
        },
        color: std_msgs::msg::ColorRGBA {
            r: 0.0,
            g: 0.7,
            b: 1.0,
            a: 0.999,
        },
        lifetime: builtin_interfaces::msg::Duration { sec: 0, nanosec: 0 },
        frame_locked: false,
        points: vec![],
        colors: vec![],
        texture_resource: String::new(),
        texture: sensor_msgs::msg::CompressedImage::default(),
        uv_coordinates: vec![],
        text: String::new(),
        mesh_resource: String::new(),
        mesh_file: visualization_msgs::msg::MeshFile::default(),
        mesh_use_embedded_materials: false,
    }
}

/// Create pose history markers from AlignmentDebug data.
///
/// Publishes arrows showing the pose at each iteration of NDT optimization.
/// Matches Autoware's transformation_array visualization.
#[cfg(feature = "debug-markers")]
pub(crate) fn create_pose_history_markers(header: &Header, debug: &AlignmentDebug) -> MarkerArray {
    let mut markers = Vec::new();

    // Convert each 4x4 transformation matrix to a Pose and create a marker
    for (i, matrix_flat) in debug.transformation_array.iter().enumerate() {
        if matrix_flat.len() < 16 {
            continue; // Skip malformed matrices
        }

        // Extract translation from matrix (last column: indices 3, 7, 11)
        let position = geometry_msgs::msg::Point {
            x: matrix_flat[3],
            y: matrix_flat[7],
            z: matrix_flat[11],
        };

        // Extract rotation matrix (3x3 upper-left block)
        let r00 = matrix_flat[0];
        let r01 = matrix_flat[1];
        let r02 = matrix_flat[2];
        let r10 = matrix_flat[4];
        let r11 = matrix_flat[5];
        let r12 = matrix_flat[6];
        let r20 = matrix_flat[8];
        let r21 = matrix_flat[9];
        let r22 = matrix_flat[10];

        // Convert rotation matrix to quaternion
        let rot_matrix = nalgebra::Matrix3::new(r00, r01, r02, r10, r11, r12, r20, r21, r22);
        let rotation = nalgebra::Rotation3::from_matrix_unchecked(rot_matrix);
        let quat = nalgebra::UnitQuaternion::from_rotation_matrix(&rotation);

        let orientation = geometry_msgs::msg::Quaternion {
            x: quat.i,
            y: quat.j,
            z: quat.k,
            w: quat.w,
        };

        let pose = Pose {
            position,
            orientation,
        };

        // Create marker with gradient color (blue -> cyan -> green)
        // to show progression through iterations
        let progress = i as f32 / debug.transformation_array.len().max(1) as f32;
        let (r, g, b) = if progress < 0.5 {
            // Blue to cyan
            let t = progress * 2.0;
            (0.0, t, 1.0)
        } else {
            // Cyan to green
            let t = (progress - 0.5) * 2.0;
            (0.0, 1.0, 1.0 - t)
        };

        markers.push(Marker {
            header: header.clone(),
            ns: "result_pose_matrix_array".to_string(),
            id: i as i32,
            type_: 0,  // ARROW
            action: 0, // ADD
            pose,
            scale: geometry_msgs::msg::Vector3 {
                x: 0.3,
                y: 0.1,
                z: 0.1,
            },
            color: std_msgs::msg::ColorRGBA { r, g, b, a: 0.8 },
            lifetime: builtin_interfaces::msg::Duration { sec: 0, nanosec: 0 },
            frame_locked: false,
            points: vec![],
            colors: vec![],
            texture_resource: String::new(),
            texture: sensor_msgs::msg::CompressedImage::default(),
            uv_coordinates: vec![],
            text: String::new(),
            mesh_resource: String::new(),
            mesh_file: visualization_msgs::msg::MeshFile::default(),
            mesh_use_embedded_materials: false,
        });
    }

    MarkerArray { markers }
}
