"""Create presentation images and videos from one validated CARLA episode."""

import argparse
import json
from pathlib import Path
import shutil

import cv2


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def choose(records, count):
    if len(records) <= count:
        return records
    indexes = sorted(set([0, len(records) // 2, len(records) - 1] +
                         [int(i * (len(records) - 1) / float(count - 1)) for i in range(count)]))
    return [records[i] for i in indexes[:count]]


def label(image, record, title):
    out = image.copy()
    text = "{} | frame {} | {:.1f} ms | {}".format(
        title, record["frame_id"],
        record["results"]["yolov5"]["result"].get("inference_latency_ms", 0.0),
        record["results"]["yolov5"]["result"].get("device_actual", "unknown"),
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/midterm_media"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sample-count", type=int, default=6)
    args = parser.parse_args()
    episode = args.episode.resolve()
    output = args.output.resolve()
    raw_dir = episode / "yolo_debug" / "raw"
    annotated_dir = episode / "yolo_debug" / "annotated"
    records = read_records(episode / "perception.jsonl")
    pairs = []
    for record in records:
        frame_id = int(record["frame_id"])
        raw = raw_dir / "frame_{:010d}.png".format(frame_id)
        annotated = annotated_dir / "frame_{:010d}.png".format(frame_id)
        if raw.is_file() and annotated.is_file():
            pairs.append((record, raw, annotated))
    if not pairs:
        raise RuntimeError("episode has no matching raw/annotated image pairs")
    output.mkdir(parents=True, exist_ok=True)
    samples_dir = output / "samples"
    samples_dir.mkdir(exist_ok=True)
    selected = choose([item[0] for item in pairs], args.sample_count)
    by_frame = {item[0]["frame_id"]: item for item in pairs}
    rows = []
    for index, record in enumerate(selected, 1):
        _, raw_path, annotated_path = by_frame[record["frame_id"]]
        raw = cv2.imread(str(raw_path))
        annotated = cv2.imread(str(annotated_path))
        raw_labeled = label(raw, record, "RAW front_rgb")
        ann_labeled = label(annotated, record, "YOLOv5s result")
        cv2.imwrite(str(samples_dir / "{:02d}_raw.png".format(index)), raw_labeled)
        cv2.imwrite(str(samples_dir / "{:02d}_detected.png".format(index)), ann_labeled)
        rows.extend([raw_labeled, ann_labeled])
    sheet_width = rows[0].shape[1]
    sheet_height = rows[0].shape[0]
    sheet = cv2.vconcat(rows)
    cv2.imwrite(str(output / "sample_comparison_sheet.png"), sheet)
    # Compact contact sheet for slides and reports.
    tiles = []
    for index in range(1, len(selected) + 1):
        raw = cv2.imread(str(samples_dir / "{:02d}_raw.png".format(index)))
        detected = cv2.imread(str(samples_dir / "{:02d}_detected.png".format(index)))
        tiles.extend([cv2.resize(raw, (480, 270)), cv2.resize(detected, (480, 270))])
    compact_rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start:start + 4]
        while len(row) < 4:
            row.append(cv2.copyMakeBorder(row[-1], 0, 0, 0, 0, cv2.BORDER_REPLICATE))
        compact_rows.append(cv2.hconcat(row))
    cv2.imwrite(str(output / "sample_grid.png"), cv2.vconcat(compact_rows))

    first = cv2.imread(str(pairs[0][1]))
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(str(output / "front_rgb_original.mp4"), fourcc, args.fps, (width, height))
    ann_writer = cv2.VideoWriter(str(output / "front_rgb_yolov5_detected.mp4"), fourcc, args.fps, (width, height))
    compare_writer = cv2.VideoWriter(str(output / "front_rgb_comparison.mp4"), fourcc, args.fps, (width * 2, height))
    for record, raw_path, annotated_path in pairs:
        raw = label(cv2.imread(str(raw_path)), record, "RAW front_rgb")
        annotated = label(cv2.imread(str(annotated_path)), record, "YOLOv5s result")
        raw_writer.write(raw)
        ann_writer.write(annotated)
        compare_writer.write(cv2.hconcat([raw, annotated]))
    raw_writer.release()
    ann_writer.release()
    compare_writer.release()
    summary = {
        "source_episode": str(episode),
        "frames": len(pairs),
        "sample_frames": [int(r["frame_id"]) for r in selected],
        "videos_fps": args.fps,
        "files": {
            "comparison_sheet": str(output / "sample_comparison_sheet.png"),
            "raw_video": str(output / "front_rgb_original.mp4"),
            "detected_video": str(output / "front_rgb_yolov5_detected.mp4"),
            "comparison_video": str(output / "front_rgb_comparison.mp4"),
        },
    }
    (output / "README.md").write_text(
        "# 中期汇报媒体\n\n"
        "源 episode：`{}`\n\n"
        "- `front_rgb_original.mp4`：front_rgb 原始图像视频。\n"
        "- `front_rgb_yolov5_detected.mp4`：YOLOv5s 检测结果视频。\n"
        "- `front_rgb_comparison.mp4`：左原图、右检测结果。\n"
        "- `sample_comparison_sheet.png`：原图和检测结果纵向对照。\n"
        "- `samples/`：逐帧原图和检测结果图片。\n\n"
        "检测结果来自独立感知日志；真值未输入 YOLO 推理，仅用于离线评估。\n".format(episode),
        encoding="utf-8",
    )
    (output / "media_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
