"""Cube poses from the external L515 color stream with aprilcube (one detector per cube), in a separate process.

Each camera frame: aprilcube CubePoseEstimator.process_frame (its default Kalman filter on, as in the aprilcube API) gives
the cube pose in the camera optical frame (translation in millimeters, converted to meters here); it is mapped to base_link
with the camera transform (frames.load_camera_transform).
The cube frame is the aprilcube frame (centered; faces +X..-Z carry tags first..last ID), identical to the sim cube frame.
Results go to shared memory with a sequence counter (seqlock), so the control loop reads them without locks or the
interpreter lock of this process. Capture time: the frame's global (host-synchronized) timestamp converted to
time.monotonic(); if the camera does not provide global time, the arrival time is used and flagged.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

SLOT = 14   # valid, capture_t, pos(3), quat(4), reproj_px, n_tags, predicted (always 0: only measurements kept), arrival_t, global_time
CUBES = ("receptive", "insertive")


class CubeTracker:
    def __init__(self, T_base_camera, detector_configs, *, width=1280, height=720, fps=30, serial=None,
                 enable_filter=True, preview=False):
        self.args = (np.asarray(T_base_camera, dtype=float).tolist(), {k: str(v) for k, v in detector_configs.items()},
                     int(width), int(height), int(fps), serial, bool(enable_filter))
        self.preview = bool(preview)
        self.data = mp.Array("d", SLOT * len(CUBES), lock=False)
        self.seq = mp.Value("q", 0, lock=False)
        self.frames = mp.Value("q", 0, lock=False)
        self.intrinsics = mp.Array("d", 4 + 8, lock=False)
        self.error = mp.Array("c", 512, lock=False)
        self.stop_flag = mp.Value("i", 0, lock=False)
        self.process = None

    def start(self, timeout_s=10.0):
        ctx = mp.get_context("spawn")
        self.process = ctx.Process(target=_tracker_main, daemon=True,
                                   args=(*self.args, self.data, self.seq, self.frames, self.intrinsics, self.error,
                                         self.stop_flag), kwargs={"preview": self.preview})
        self.process.start()
        deadline = time.monotonic() + timeout_s
        while self.frames.value == 0:
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError("No camera frame within 10 s")
            time.sleep(0.01)

    def check(self):
        if self.error.value:
            raise RuntimeError("Tracker process: " + self.error.value.decode(errors="replace"))
        if self.process is not None and not self.process.is_alive():
            raise RuntimeError("Tracker process exited")

    def latest(self):
        """{cube: dict(valid, capture_t, pos, quat, reproj_px, n_tags, predicted, arrival_t, global_time)}."""
        for _ in range(100):
            s1 = self.seq.value
            if s1 % 2:
                continue
            raw = np.frombuffer(self.data, dtype=np.float64).copy()
            if self.seq.value == s1:
                break
        else:
            raise RuntimeError("Tracker shared memory kept changing during read")
        out = {}
        for i, name in enumerate(CUBES):
            r = raw[i * SLOT:(i + 1) * SLOT]
            out[name] = dict(valid=bool(r[0]), capture_t=r[1], pos=r[2:5].copy(), quat=r[5:9].copy(), reproj_px=r[9],
                             n_tags=int(r[10]), predicted=bool(r[11]), arrival_t=r[12], global_time=bool(r[13]))
        return out

    def stop(self):
        self.stop_flag.value = 1
        if self.process is not None:
            self.process.join(3.0)
            if self.process.is_alive():
                self.process.terminate()


def _draw_preview(image, K, coeffs, results, last, capture):
    """Camera image with each detected cube's axes (40 mm) and a status line per cube; shown after the poses are published."""
    import cv2
    vis = image.copy()
    for i, (name, res) in enumerate(zip(CUBES, results)):
        seen = res["success"] and res["T"] is not None and not res.get("predicted")
        if seen:
            T_mm = np.asarray(res["T"], dtype=float)
            cv2.drawFrameAxes(vis, K, coeffs, cv2.Rodrigues(T_mm[:3, :3])[0], T_mm[:3, 3], 40.0)
            status = f"seen ({int(res['n_tags'])} tags)"
        else:
            status = f"last seen {capture - last[i, 1]:.1f} s ago" if last[i, 0] else "never seen"
        label = {"receptive": "bottom cube 10-15", "insertive": "carried cube 20-25"}[name]
        cv2.putText(vis, f"{label}: {status}", (12, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (60, 220, 60) if seen else (40, 40, 230), 2, cv2.LINE_AA)
    cv2.imshow("L515 cubes (preview)", vis)
    cv2.waitKey(1)


def _tracker_main(T_base_camera, detector_configs, width, height, fps, serial, enable_filter, data, seq, frames,
                  intrinsics, error, stop_flag, preview=False):
    pipeline = None
    try:
        import aprilcube
        import pyrealsense2 as rs
        from state_policy.frames import matrix_to_pos_quat
        T = np.asarray(T_base_camera)
        pipeline, config = rs.pipeline(), rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = pipeline.start(config)
        sensor = profile.get_device().first_color_sensor()
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        coeffs = np.asarray(intr.coeffs[:5], dtype=float)
        intrinsics[:4] = [intr.fx, intr.fy, intr.ppx, intr.ppy]
        intrinsics[4:9] = coeffs
        cam = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
        detectors = [aprilcube.detector(detector_configs[name], cam, dist_coeffs=coeffs, enable_filter=enable_filter)
                     for name in CUBES]
        K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        buf = np.frombuffer(data, dtype=np.float64)
        last = np.zeros((len(CUBES), SLOT))   # last MEASURED pose per cube; its capture time ages until re-detected
        while not stop_flag.value:
            fs = pipeline.wait_for_frames(1000)
            arrival = time.monotonic()
            color = fs.get_color_frame()
            if not color:
                continue
            global_time = color.get_frame_timestamp_domain() == rs.timestamp_domain.global_time
            capture = color.get_timestamp() / 1000.0 - (time.time() - time.monotonic()) if global_time else arrival
            image = np.asanyarray(color.get_data())
            results = []
            for i, det in enumerate(detectors):
                res = det.process_frame(image, timestamp=capture)
                results.append(res)
                # Only tag measurements update the pose; the filter's prediction-only output (no tag seen) does not.
                if res["success"] and res["T"] is not None and not res.get("predicted"):
                    T_cam_cube = np.array(res["T"], dtype=float)
                    T_cam_cube[:3, 3] /= 1000.0   # process_frame's T is in mm (aprilcube world_pose() divides by 1000)
                    pos, quat = matrix_to_pos_quat(T @ T_cam_cube)
                    last[i] = 0.0
                    last[i, 0], last[i, 1], last[i, 2:5], last[i, 5:9] = 1.0, capture, pos, quat
                    last[i, 9], last[i, 10], last[i, 12], last[i, 13] = (float(res["reproj_error"]), float(res["n_tags"]),
                                                                        arrival, float(global_time))
            seq.value += 1          # odd: writing
            buf[:] = last.ravel()
            seq.value += 1          # even: complete
            frames.value += 1
            if preview:
                _draw_preview(image, K, coeffs, results, last, capture)
    except BaseException as exc:
        error.value = f"{type(exc).__name__}: {exc}".encode()[:511]
    finally:
        if preview:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
