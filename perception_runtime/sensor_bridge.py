"""Synchronous CARLA sensor queues.

Callbacks only convert and enqueue images.  Inference stays outside the
callback so a slow model cannot block CARLA sensor delivery.
"""

import queue
import threading

from .frame_schema import SensorFrameBundle, frame_from_carla_image


class FrameSyncError(RuntimeError):
    """A requested simulation frame was missing or could not be paired."""


class CarlaSensorBridge:
    def __init__(self, episode_id, camera_ids, expected_sizes=None, queue_size=4):
        if not camera_ids:
            raise ValueError("at least one camera ID is required")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.episode_id = episode_id
        self.camera_ids = tuple(camera_ids)
        self.expected_sizes = dict(expected_sizes or {})
        self._queues = {
            camera_id: queue.Queue(maxsize=int(queue_size))
            for camera_id in self.camera_ids
        }
        self._pending = {camera_id: None for camera_id in self.camera_ids}
        self._sensors = {}
        self._lock = threading.Lock()
        self._stats = {
            "frames_received": {camera_id: 0 for camera_id in self.camera_ids},
            "frames_dropped": {camera_id: 0 for camera_id in self.camera_ids},
            "frames_discarded_stale": {camera_id: 0 for camera_id in self.camera_ids},
        }

    def attach_sensor(self, camera_id, sensor):
        if camera_id not in self._queues:
            raise KeyError(f"unknown camera_id: {camera_id}")
        if not callable(getattr(sensor, "listen", None)):
            raise TypeError("sensor must provide listen(callback)")
        if camera_id in self._sensors:
            raise RuntimeError(f"sensor already attached: {camera_id}")

        def callback(image):
            self._on_image(camera_id, image)

        sensor.listen(callback)
        self._sensors[camera_id] = sensor

    def _on_image(self, camera_id, image):
        try:
            frame = frame_from_carla_image(
                image,
                episode_id=self.episode_id,
                camera_id=camera_id,
                expected_size=self.expected_sizes.get(camera_id),
            )
        except (TypeError, ValueError):
            with self._lock:
                self._stats["frames_dropped"][camera_id] += 1
            return
        with self._lock:
            self._stats["frames_received"][camera_id] += 1
        target_queue = self._queues[camera_id]
        try:
            target_queue.put_nowait(frame)
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._stats["frames_dropped"][camera_id] += 1
            target_queue.put_nowait(frame)

    def _take_for_frame(self, camera_id, frame_id, timeout_s):
        pending = self._pending[camera_id]
        if pending is not None:
            self._pending[camera_id] = None
            if pending.frame_id == frame_id:
                return pending
            if pending.frame_id > frame_id:
                self._pending[camera_id] = pending
                raise FrameSyncError(
                    f"{camera_id} advanced to frame {pending.frame_id}; "
                    f"requested {frame_id}"
                )

        while True:
            try:
                frame = self._queues[camera_id].get(timeout=timeout_s)
            except queue.Empty as error:
                raise FrameSyncError(
                    f"timeout waiting for {camera_id} frame {frame_id}"
                ) from error
            if frame.frame_id < frame_id:
                with self._lock:
                    self._stats["frames_discarded_stale"][camera_id] += 1
                continue
            if frame.frame_id > frame_id:
                self._pending[camera_id] = frame
                raise FrameSyncError(
                    f"{camera_id} advanced to frame {frame.frame_id}; "
                    f"requested {frame_id}"
                )
            return frame

    def get_bundle(self, frame_id, timeout_s=1.0):
        """Return all camera images for one ``world.tick()`` frame."""
        frames = {
            camera_id: self._take_for_frame(camera_id, int(frame_id), timeout_s)
            for camera_id in self.camera_ids
        }
        timestamps = {frame.timestamp_ns for frame in frames.values()}
        if len(timestamps) != 1:
            raise FrameSyncError(
                f"timestamp mismatch at frame {frame_id}: {sorted(timestamps)}"
            )
        return SensorFrameBundle(
            episode_id=self.episode_id,
            frame_id=int(frame_id),
            timestamp_ns=next(iter(timestamps)),
            frames=frames,
        )

    def stats(self):
        with self._lock:
            return {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in self._stats.items()
            }

    def close(self):
        for sensor in self._sensors.values():
            stop = getattr(sensor, "stop", None)
            if callable(stop):
                stop()
        self._sensors.clear()
        for pending in self._pending:
            self._pending[pending] = None
        for target_queue in self._queues.values():
            while True:
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    break
