from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def write_waymo_parquet_dataset(dataset_root: Path) -> Path:
    image_root = dataset_root / "camera_image"
    box_root = dataset_root / "camera_box"
    image_root.mkdir(parents=True, exist_ok=True)
    box_root.mkdir(parents=True, exist_ok=True)

    segment_name = "segment_001"
    timestamps = [111, 222]
    cameras = [1, 2]

    image_table = pa.table(
        {
            "key.frame_timestamp_micros": pa.array(timestamps, type=pa.int64()),
            "key.camera_name": pa.array(cameras, type=pa.int32()),
            "[CameraImageComponent].image": pa.array(
                [
                    _jpeg_bytes((64, 32), color="black"),
                    _jpeg_bytes((64, 32), color="gray"),
                ],
                type=pa.binary(),
            ),
        }
    )
    pq.write_table(image_table, image_root / f"{segment_name}.parquet")

    box_table = pa.table(
        {
            "key.frame_timestamp_micros": pa.array([111, 111], type=pa.int64()),
            "key.camera_name": pa.array([1, 1], type=pa.int32()),
            "[CameraBoxComponent].type": pa.array([1, 4], type=pa.int32()),
            "[CameraBoxComponent].box.center.x": pa.array([20.0, 40.0], type=pa.float32()),
            "[CameraBoxComponent].box.center.y": pa.array([10.0, 14.0], type=pa.float32()),
            "[CameraBoxComponent].box.size.x": pa.array([8.0, 10.0], type=pa.float32()),
            "[CameraBoxComponent].box.size.y": pa.array([6.0, 8.0], type=pa.float32()),
        }
    )
    pq.write_table(box_table, box_root / f"{segment_name}.parquet")
    return dataset_root


def _jpeg_bytes(size: tuple[int, int], *, color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()