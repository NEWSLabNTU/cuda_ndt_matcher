//! CPU-only point cloud conversion and serialization.
//!
//! This module handles PointCloud2 message parsing and construction.
//! All code here is pure CPU with no GPU dependencies.

use anyhow::{bail, Result};
use sensor_msgs::msg::{PointCloud2, PointField};
use std_msgs::msg::Header;

/// Field offsets for XYZ point cloud
struct XyzOffsets {
    x: usize,
    y: usize,
    z: usize,
    point_step: usize,
}

impl XyzOffsets {
    fn from_pointcloud2(msg: &PointCloud2) -> Result<Self> {
        let mut x_offset = None;
        let mut y_offset = None;
        let mut z_offset = None;

        for field in &msg.fields {
            match field.name.as_str() {
                "x" => x_offset = Some(field.offset as usize),
                "y" => y_offset = Some(field.offset as usize),
                "z" => z_offset = Some(field.offset as usize),
                _ => {}
            }
        }

        let x = x_offset.ok_or_else(|| anyhow::anyhow!("Missing 'x' field"))?;
        let y = y_offset.ok_or_else(|| anyhow::anyhow!("Missing 'y' field"))?;
        let z = z_offset.ok_or_else(|| anyhow::anyhow!("Missing 'z' field"))?;

        Ok(Self {
            x,
            y,
            z,
            point_step: msg.point_step as usize,
        })
    }
}

/// Convert PointCloud2 message to Vec of [x, y, z] points
pub(crate) fn from_pointcloud2(msg: &PointCloud2) -> Result<Vec<[f32; 3]>> {
    if msg.data.is_empty() {
        return Ok(Vec::new());
    }

    let offsets = XyzOffsets::from_pointcloud2(msg)?;
    let num_points = (msg.width as usize) * (msg.height as usize);

    if msg.data.len() < num_points * offsets.point_step {
        bail!(
            "PointCloud2 data too short: {} < {}",
            msg.data.len(),
            num_points * offsets.point_step
        );
    }

    let mut points = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let base = i * offsets.point_step;

        let x = read_f32(&msg.data, base + offsets.x);
        let y = read_f32(&msg.data, base + offsets.y);
        let z = read_f32(&msg.data, base + offsets.z);

        // Skip NaN points
        if x.is_finite() && y.is_finite() && z.is_finite() {
            points.push([x, y, z]);
        }
    }

    Ok(points)
}

/// Read f32 from byte slice (little endian)
fn read_f32(data: &[u8], offset: usize) -> f32 {
    let bytes = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    f32::from_le_bytes(bytes)
}

/// Standard XYZ PointField definitions (FLOAT32, offsets 0/4/8).
fn xyz_fields() -> Vec<PointField> {
    vec![
        PointField {
            name: "x".into(),
            offset: 0,
            datatype: 7, // FLOAT32
            count: 1,
        },
        PointField {
            name: "y".into(),
            offset: 4,
            datatype: 7,
            count: 1,
        },
        PointField {
            name: "z".into(),
            offset: 8,
            datatype: 7,
            count: 1,
        },
    ]
}

/// Encode XYZ data as little-endian bytes.
fn encode_xyz_data(points: &[[f32; 3]]) -> Vec<u8> {
    let mut data = Vec::with_capacity(points.len() * 12);
    for p in points {
        data.extend_from_slice(&p[0].to_le_bytes());
        data.extend_from_slice(&p[1].to_le_bytes());
        data.extend_from_slice(&p[2].to_le_bytes());
    }
    data
}

/// Convert Vec of [x, y, z] points with RGB colors to PointCloud2 message.
///
/// Each point has xyz coordinates and a packed RGB value (0x00RRGGBB format).
/// This is used for per-point score visualization where colors indicate quality.
#[cfg(feature = "debug-markers")]
pub(crate) fn to_pointcloud2_with_rgb(
    points: &[[f32; 3]],
    rgb_values: &[u32],
    header: &Header,
) -> PointCloud2 {
    // Point format: x, y, z (float32), rgb (packed as float32 by reinterpreting bits)
    // Total: 16 bytes per point (same as Autoware's XYZI format)
    let point_step = 16u32;
    let mut data = Vec::with_capacity(points.len() * point_step as usize);

    for (i, p) in points.iter().enumerate() {
        data.extend_from_slice(&p[0].to_le_bytes());
        data.extend_from_slice(&p[1].to_le_bytes());
        data.extend_from_slice(&p[2].to_le_bytes());
        // RGB is packed as a float32 by reinterpreting the bits (ROS convention)
        let rgb = rgb_values.get(i).copied().unwrap_or(0);
        data.extend_from_slice(&f32::from_bits(rgb).to_le_bytes());
    }

    let mut fields = xyz_fields();
    fields.push(PointField {
        name: "rgb".into(),
        offset: 12,
        datatype: 7, // FLOAT32 (bits reinterpreted as RGB)
        count: 1,
    });

    // Safe: point clouds are always << u32::MAX points
    PointCloud2 {
        header: header.clone(),
        height: 1,
        width: points.len() as u32,
        fields,
        is_bigendian: false,
        point_step,
        row_step: point_step * points.len() as u32,
        data,
        is_dense: true,
    }
}

/// Convert Vec of [x, y, z] points to PointCloud2 message
pub(crate) fn to_pointcloud2(points: &[[f32; 3]], header: &Header) -> PointCloud2 {
    let point_step = 12u32; // 3 * sizeof(f32)
    let data = encode_xyz_data(points);

    // Safe: point clouds are always << u32::MAX points
    PointCloud2 {
        header: header.clone(),
        height: 1,
        width: points.len() as u32,
        fields: xyz_fields(),
        is_bigendian: false,
        point_step,
        row_step: point_step * points.len() as u32,
        data,
        is_dense: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_pointcloud(points: &[[f32; 3]]) -> PointCloud2 {
        to_pointcloud2(points, &Default::default())
    }

    #[test]
    fn test_from_pointcloud2() {
        let input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let msg = make_test_pointcloud(&input);

        let result = from_pointcloud2(&msg).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], [1.0, 2.0, 3.0]);
        assert_eq!(result[1], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_empty_pointcloud() {
        let msg = make_test_pointcloud(&[]);
        let result = from_pointcloud2(&msg).unwrap();
        assert!(result.is_empty());
    }
}
