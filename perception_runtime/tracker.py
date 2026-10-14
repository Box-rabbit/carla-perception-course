"""Small truth-free IoU tracker for online 2D detections."""


def _iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection) / union if union else 0.0


class IoUTracker:
    """Greedy class-aware tracker with bounded missed-frame retention."""

    def __init__(self, iou_threshold=0.3, max_missed_frames=3):
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if isinstance(max_missed_frames, bool) or int(max_missed_frames) < 0:
            raise ValueError("max_missed_frames must be nonnegative")
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self._next_id = 1
        self._tracks = {}

    def update(self, detections):
        if not isinstance(detections, list):
            raise TypeError("detections must be a list")
        candidates = []
        for index, detection in enumerate(detections):
            candidates.append((index, detection))
        pairs = []
        for track_id, track in self._tracks.items():
            for index, detection in candidates:
                if detection.get("class") != track["class"]:
                    continue
                score = _iou(track["bbox2d_xyxy_px"], detection["bbox2d_xyxy_px"])
                if score >= self.iou_threshold:
                    pairs.append((score, track_id, index))
        used_tracks = set()
        used_detections = set()
        for _score, track_id, index in sorted(pairs, reverse=True):
            if track_id in used_tracks or index in used_detections:
                continue
            detection = detections[index]
            track = self._tracks[track_id]
            track["bbox2d_xyxy_px"] = list(detection["bbox2d_xyxy_px"])
            track["confidence"] = float(detection["confidence"])
            track["missed_frames"] = 0
            track["hits"] += 1
            used_tracks.add(track_id)
            used_detections.add(index)

        for index, detection in candidates:
            if index in used_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = {
                "track_id": track_id,
                "class": detection["class"],
                "subclass": detection["subclass"],
                "bbox2d_xyxy_px": list(detection["bbox2d_xyxy_px"]),
                "confidence": float(detection["confidence"]),
                "hits": 1,
                "missed_frames": 0,
            }
            used_tracks.add(track_id)

        expired = []
        for track_id, track in self._tracks.items():
            if track_id not in used_tracks:
                track["missed_frames"] += 1
                if track["missed_frames"] > self.max_missed_frames:
                    expired.append(track_id)
        for track_id in expired:
            del self._tracks[track_id]

        return [dict(track) for track in sorted(self._tracks.values(), key=lambda item: item["track_id"])]

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
