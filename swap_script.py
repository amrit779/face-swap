import argparse
import cv2
import insightface
import importlib
import logging
import math
import numpy as np
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import onnxruntime as ort
from insightface.app import FaceAnalysis


def _patch_torchvision_imports():
    try:
        importlib.import_module('torchvision.transforms.functional_tensor')
    except ModuleNotFoundError:
        compat_module = importlib.import_module('torchvision.transforms._functional_tensor')
        sys.modules['torchvision.transforms.functional_tensor'] = compat_module


_patch_torchvision_imports()

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
import torch


LOG_LEVEL = os.getenv('FACE_SWAP_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
LOGGER = logging.getLogger('face_swap')
DEFAULT_ALLOWED_MODULES = ('detection', 'recognition')
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_DETECTION_INTERVAL = 1
DEFAULT_ENHANCEMENT_INTERVAL = 5
DEFAULT_IDENTITY_THRESHOLD = 0.35
DEFAULT_COLOR_MATCH_STRENGTH = 0.35
DEFAULT_EXPORT_SCALE = 1.0
DEFAULT_EXPORT_CRF = 18
DEFAULT_EXPORT_PRESET = 'slow'
DEFAULT_SUPERRES_MODEL_NAME = 'RealESRGAN_x4plus'
DEFAULT_SUPERRES_TILE = 256
DEFAULT_SUPERRES_TILE_PAD = 16
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}
SUPERRES_MODEL_CONFIGS = {
    'RealESRGAN_x4plus': {
        'scale': 4,
        'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'rrdbnet': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
        },
    },
}


def _validate_file(path_str, label):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')
    LOGGER.debug('%s resolved to %s', label, path.resolve())
    return path


def _validate_directory(path_str, label, create=False):
    path = Path(path_str)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f'{label} not found: {path}')
    LOGGER.debug('%s resolved to %s', label, path.resolve())
    return path


def _log_run_header(source_path, video_path, output_path, preflight=False):
    if preflight:
        LOGGER.info('Starting preflight')
    else:
        LOGGER.info('Starting face swap')
    LOGGER.info('Source image: %s', source_path)
    LOGGER.info('Target video: %s', video_path)
    LOGGER.info('Output video: %s', output_path)


def _format_seconds(seconds):
    if seconds is None:
        return 'unknown'
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def _get_available_onnx_providers():
    return ort.get_available_providers()


def _get_default_onnx_providers():
    available = _get_available_onnx_providers()
    if 'CoreMLExecutionProvider' in available:
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def _parse_provider_list(provider_names):
    available = _get_available_onnx_providers()
    normalized = [provider.strip() for provider in provider_names if provider.strip()]
    unsupported = [provider for provider in normalized if provider not in available]
    if unsupported:
        raise ValueError(
            f'Unsupported ONNX Runtime providers: {unsupported}. Available providers: {available}'
        )
    return normalized


def _resolve_onnx_providers(provider_names):
    if not provider_names:
        providers = _get_default_onnx_providers()
    else:
        providers = _parse_provider_list(provider_names)
    LOGGER.info('Using ONNX Runtime providers: %s', providers)
    return providers


def _resolve_base_paths(source_path):
    model_dir = Path('models')
    return {
        'source_path': _validate_file(source_path, 'Source image'),
        'swapper_model_path': _validate_file(model_dir / 'inswapper_128.onnx', 'Inswapper model'),
        'enhancer_model_path': _validate_file(model_dir / 'GFPGANv1.3.pth', 'GFPGAN model'),
    }


def _collect_target_videos(target_path):
    target_path = Path(target_path)
    if target_path.is_file():
        return [_validate_file(target_path, 'Target video')]

    target_dir = _validate_directory(target_path, 'Target directory')
    videos = sorted(
        path for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f'No video files found in target directory: {target_dir}')
    LOGGER.info('Found %s target video(s) in %s', len(videos), target_dir)
    return videos


def _resolve_output_path(target_video_path, output_path, batch_mode=False):
    output_path = Path(output_path)
    if batch_mode or output_path.is_dir() or output_path.suffix == '':
        output_dir = _validate_directory(output_path, 'Output directory', create=True)
        return output_dir / target_video_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _get_temp_output_path(output_path, marker='incomplete'):
    output = Path(output_path)
    return output.with_name(f'{output.stem}.{marker}{output.suffix}')


def _should_log_progress(frame_index, progress_every):
    return frame_index == 1 or frame_index % progress_every == 0


def _select_largest_face(faces):
    return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def _normalize_embedding(embedding):
    if embedding is None:
        return None

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def _embedding_similarity(embedding_a, embedding_b):
    if embedding_a is None or embedding_b is None:
        return None
    return float(np.dot(embedding_a, embedding_b))


def _update_locked_embedding(locked_embedding, candidate_embedding, momentum=0.8):
    candidate_embedding = _normalize_embedding(candidate_embedding)
    if candidate_embedding is None:
        return locked_embedding
    if locked_embedding is None:
        return candidate_embedding

    mixed_embedding = momentum * locked_embedding + (1.0 - momentum) * candidate_embedding
    return _normalize_embedding(mixed_embedding)


def _to_tracking_face(face):
    return SimpleNamespace(
        bbox=np.array(face.bbox, dtype=np.float32),
        kps=np.array(face.kps, dtype=np.float32),
        det_score=getattr(face, 'det_score', 1.0),
        normed_embedding=_normalize_embedding(getattr(face, 'normed_embedding', None)),
    )


def _track_face(prev_gray, frame_gray, tracked_face):
    prev_points = tracked_face.kps.astype(np.float32).reshape(-1, 1, 2)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        frame_gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if next_points is None or status is None:
        return None

    valid_mask = status.reshape(-1).astype(bool)
    valid_count = int(valid_mask.sum())
    if valid_count < max(3, math.ceil(len(prev_points) / 2)):
        return None

    prev_valid = prev_points.reshape(-1, 2)[valid_mask]
    next_valid = next_points.reshape(-1, 2)[valid_mask]
    delta = next_valid.mean(axis=0) - prev_valid.mean(axis=0)
    bbox = tracked_face.bbox.astype(np.float32).copy()
    bbox[[0, 2]] += delta[0]
    bbox[[1, 3]] += delta[1]

    return SimpleNamespace(
        bbox=bbox,
        kps=next_points.reshape(-1, 2),
        det_score=getattr(tracked_face, 'det_score', 1.0),
        normed_embedding=getattr(tracked_face, 'normed_embedding', None),
    )


def _select_identity_face(faces, locked_embedding, identity_threshold, frame_index):
    if not faces:
        return None, locked_embedding

    if locked_embedding is None:
        selected_face = _select_largest_face(faces)
        updated_embedding = _update_locked_embedding(
            None,
            getattr(selected_face, 'normed_embedding', None),
        )
        LOGGER.debug('Frame %s initialized target identity lock', frame_index)
        return _to_tracking_face(selected_face), updated_embedding

    scored_faces = []
    for face in faces:
        face_embedding = _normalize_embedding(getattr(face, 'normed_embedding', None))
        similarity = _embedding_similarity(locked_embedding, face_embedding)
        if similarity is not None:
            scored_faces.append((similarity, face, face_embedding))

    if not scored_faces:
        fallback_face = _select_largest_face(faces)
        LOGGER.debug('Frame %s fell back to largest face because no embeddings were available', frame_index)
        return _to_tracking_face(fallback_face), locked_embedding

    best_similarity, best_face, best_embedding = max(scored_faces, key=lambda item: item[0])
    LOGGER.debug('Frame %s best identity similarity=%.3f', frame_index, best_similarity)
    if best_similarity < identity_threshold:
        LOGGER.debug(
            'Frame %s did not meet identity threshold %.2f; skipping swap for this detection',
            frame_index,
            identity_threshold,
        )
        return None, locked_embedding

    return _to_tracking_face(best_face), _update_locked_embedding(locked_embedding, best_embedding)


def _create_face_analysis(det_size, providers):
    LOGGER.info('Loading InsightFace detector with modules=%s', DEFAULT_ALLOWED_MODULES)
    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=list(DEFAULT_ALLOWED_MODULES),
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def _create_swapper(model_path, providers):
    LOGGER.info('Loading swapper model')
    return insightface.model_zoo.get_model(str(model_path), download=False, providers=providers)


def _create_face_enhancer(model_path, device, skip_enhancement):
    if skip_enhancement:
        LOGGER.info('GFPGAN enhancement disabled')
        return None

    LOGGER.info('Loading GFPGAN enhancer on device=%s', device)
    return GFPGANer(model_path=str(model_path), upscale=1, device=device)


def _load_source_face(app, source_path):
    LOGGER.info('Reading source image')
    img = cv2.imread(source_path)
    if img is None:
        raise ValueError(f'Failed to read source image: {source_path}')

    source_faces = app.get(img)
    if not source_faces:
        raise ValueError('No face detected in source image')

    LOGGER.info('Detected %s face(s) in source image', len(source_faces))
    return _select_largest_face(source_faces)


def _detect_target_face(app, frame, frame_index, locked_embedding, identity_threshold):
    faces = app.get(frame)
    if not faces:
        LOGGER.debug('No face detected in frame %s', frame_index)
        return None, locked_embedding

    return _select_identity_face(faces, locked_embedding, identity_threshold, frame_index)


def _resolve_target_face(
    app,
    frame,
    frame_index,
    prev_gray,
    tracked_face,
    detection_interval,
    locked_embedding,
    identity_threshold,
):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    should_detect = (
        tracked_face is None or
        detection_interval <= 1 or
        ((frame_index - 1) % detection_interval == 0)
    )

    if should_detect:
        target_face, locked_embedding = _detect_target_face(
            app,
            frame,
            frame_index,
            locked_embedding,
            identity_threshold,
        )
        if target_face is not None:
            LOGGER.debug('Frame %s used fresh detection', frame_index)
        return target_face, frame_gray, locked_embedding

    if prev_gray is None or tracked_face is None:
        target_face, locked_embedding = _detect_target_face(
            app,
            frame,
            frame_index,
            locked_embedding,
            identity_threshold,
        )
        return target_face, frame_gray, locked_embedding

    tracked = _track_face(prev_gray, frame_gray, tracked_face)
    if tracked is not None:
        LOGGER.debug('Frame %s reused tracked face', frame_index)
        return tracked, frame_gray, locked_embedding

    LOGGER.debug('Tracking failed in frame %s; falling back to detection', frame_index)
    target_face, locked_embedding = _detect_target_face(
        app,
        frame,
        frame_index,
        locked_embedding,
        identity_threshold,
    )
    return target_face, frame_gray, locked_embedding


def _build_bbox_face_mask(frame_shape, bbox):
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    axes = (
        max(int((x2 - x1) * 0.42), 1),
        max(int((y2 - y1) * 0.56), 1),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur_width = min(max((axes[0] // 2) * 2 + 1, 3), width | 1)
    blur_height = min(max((axes[1] // 2) * 2 + 1, 3), height | 1)
    return cv2.GaussianBlur(mask, (blur_width, blur_height), 0)


def _build_landmark_face_mask(frame_shape, target_face):
    keypoints = np.asarray(getattr(target_face, 'kps', None), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] < 5:
        return _build_bbox_face_mask(frame_shape, target_face.bbox)

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = target_face.bbox.astype(np.float32)
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)

    left_eye, right_eye, nose, left_mouth, right_mouth = keypoints[:5]
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    vertical = mouth_center - eye_center
    if np.linalg.norm(vertical) < 1e-3:
        vertical = np.array([0.0, face_height * 0.35], dtype=np.float32)

    forehead_shift = vertical * 0.95
    polygon = np.array([
        left_eye - forehead_shift * 0.85,
        eye_center - forehead_shift * 1.10,
        right_eye - forehead_shift * 0.85,
        np.array([x2 - face_width * 0.08, y1 + face_height * 0.55], dtype=np.float32),
        np.array([x2 - face_width * 0.18, y2 - face_height * 0.06], dtype=np.float32),
        np.array([(x1 + x2) / 2.0, y2 - face_height * 0.01], dtype=np.float32),
        np.array([x1 + face_width * 0.18, y2 - face_height * 0.06], dtype=np.float32),
        np.array([x1 + face_width * 0.08, y1 + face_height * 0.55], dtype=np.float32),
        left_mouth,
        nose,
        right_mouth,
    ], dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.float32)
    hull = cv2.convexHull(polygon.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)

    blur_width = max(int(face_width * 0.22), 3)
    blur_height = max(int(face_height * 0.22), 3)
    blur_width = min((blur_width | 1), width | 1)
    blur_height = min((blur_height | 1), height | 1)
    return cv2.GaussianBlur(mask, (blur_width, blur_height), 0)


def _get_mask_bounds(mask, threshold=1e-3):
    ys, xs = np.nonzero(mask > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _match_face_colors(original_roi, swapped_roi, strength):
    original_mean = original_roi.mean(axis=(0, 1), keepdims=True)
    original_std = original_roi.std(axis=(0, 1), keepdims=True)
    swapped_mean = swapped_roi.mean(axis=(0, 1), keepdims=True)
    swapped_std = swapped_roi.std(axis=(0, 1), keepdims=True)
    corrected_roi = (swapped_roi - swapped_mean) * (original_std / np.maximum(swapped_std, 1.0)) + original_mean
    corrected_roi = np.clip(corrected_roi, 0, 255)
    return (1.0 - strength) * swapped_roi + strength * corrected_roi


def _refine_swapped_face(original_frame, swapped_frame, target_face, color_match_strength):
    face_mask = _build_landmark_face_mask(swapped_frame.shape, target_face)
    bounds = _get_mask_bounds(face_mask)
    if bounds is None:
        return swapped_frame

    x1, y1, x2, y2 = bounds

    swapped_roi = swapped_frame[y1:y2, x1:x2].astype(np.float32)
    original_roi = original_frame[y1:y2, x1:x2].astype(np.float32)
    if swapped_roi.size == 0 or original_roi.size == 0:
        return swapped_frame

    matched_roi = swapped_roi
    if color_match_strength > 0:
        matched_roi = _match_face_colors(original_roi, swapped_roi, color_match_strength)

    blend_mask = face_mask[y1:y2, x1:x2][..., None]

    refined_frame = swapped_frame.astype(np.float32)
    refined_frame[y1:y2, x1:x2] = blend_mask * matched_roi + (1.0 - blend_mask) * swapped_roi
    return np.clip(refined_frame, 0, 255).astype(np.uint8)


def _swap_frame(frame, swapper, source_face, target_face, color_match_strength):
    original_frame = frame.copy()
    swapped_frame = swapper.get(frame, target_face, source_face, paste_back=True)
    return _refine_swapped_face(original_frame, swapped_frame, target_face, color_match_strength)


def _probe_video(video_path):
    LOGGER.info('Opening target video')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Failed to open target video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f'Invalid video metadata: fps={fps}, width={width}, height={height}')
    if not ok or frame is None:
        raise ValueError(f'Failed to read the first frame from target video: {video_path}')

    LOGGER.info(
        'Video metadata: fps=%.2f, width=%s, height=%s, frames=%s',
        fps,
        width,
        height,
        frame_count,
    )
    return fps, width, height, frame_count, frame


def _create_video_writer(output_path, fps, width, height):
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        raise ValueError(f'Failed to create output video: {output_path}')
    return out


def _validate_video_writer(output_path, fps, width, height, sample_frame):
    LOGGER.info('Validating video write support at %s', output_path)
    writer = _create_video_writer(output_path, fps, width, height)
    try:
        writer.write(sample_frame)
    finally:
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f'Video writer produced an empty file: {output_path}')
    output_path.unlink()


def _log_progress_with_eta(frame_index, swapped_frames, total_frames, start_time, progress_every, phase='swap'):
    if not _should_log_progress(frame_index, progress_every):
        return

    elapsed = time.perf_counter() - start_time
    processing_fps = frame_index / elapsed if elapsed > 0 else 0.0
    remaining_frames = max(total_frames - frame_index, 0) if total_frames > 0 else None
    eta_seconds = (remaining_frames / processing_fps) if processing_fps > 0 and remaining_frames is not None else None
    percent = (frame_index / total_frames * 100) if total_frames > 0 else None

    if percent is None:
        LOGGER.info(
            '%s pass: processed %s frames, swapped %s frames, elapsed=%s, speed=%.2f fps, eta=%s',
            phase.capitalize(),
            frame_index,
            swapped_frames,
            _format_seconds(elapsed),
            processing_fps,
            _format_seconds(eta_seconds),
        )
        return

    LOGGER.info(
        '%s pass: processed %s/%s frames (%.1f%%), swapped %s frames, elapsed=%s, speed=%.2f fps, eta=%s',
        phase.capitalize(),
        frame_index,
        total_frames,
        percent,
        swapped_frames,
        _format_seconds(elapsed),
        processing_fps,
        _format_seconds(eta_seconds),
    )


def _log_enhancement_progress(frame_index, enhanced_frames, total_frames, start_time, progress_every):
    if not _should_log_progress(frame_index, progress_every):
        return

    elapsed = time.perf_counter() - start_time
    processing_fps = frame_index / elapsed if elapsed > 0 else 0.0
    remaining_frames = max(total_frames - frame_index, 0) if total_frames > 0 else None
    eta_seconds = (remaining_frames / processing_fps) if processing_fps > 0 and remaining_frames is not None else None
    percent = (frame_index / total_frames * 100) if total_frames > 0 else None

    LOGGER.info(
        'Enhancement pass: processed %s/%s frames (%.1f%%), enhanced %s frames, elapsed=%s, speed=%.2f fps, eta=%s',
        frame_index,
        total_frames,
        percent or 0.0,
        enhanced_frames,
        _format_seconds(elapsed),
        processing_fps,
        _format_seconds(eta_seconds),
    )


def _find_binary(name):
    return shutil.which(name)


def _should_run_superres(export_scale):
    return export_scale > 1.0 + 1e-6


def _resolve_superres_model_path(model_name, model_path=None):
    if model_path:
        if model_path.startswith('https://'):
            return model_path
        return str(_validate_file(model_path, 'Super-resolution model'))

    if model_name not in SUPERRES_MODEL_CONFIGS:
        raise ValueError(f'Unsupported super-resolution model: {model_name}')
    return SUPERRES_MODEL_CONFIGS[model_name]['model_path']


def _create_superres_engine(model_name, model_path=None):
    config = SUPERRES_MODEL_CONFIGS[model_name]
    resolved_model_path = _resolve_superres_model_path(model_name, model_path)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = RRDBNet(scale=config['scale'], **config['rrdbnet'])
    LOGGER.info('Loading RealESRGAN model=%s on device=%s', model_name, device)
    return RealESRGANer(
        scale=config['scale'],
        model_path=resolved_model_path,
        model=model,
        tile=DEFAULT_SUPERRES_TILE,
        tile_pad=DEFAULT_SUPERRES_TILE_PAD,
        pre_pad=0,
        half=False,
        device=device,
    )


def _source_has_audio(video_path):
    ffprobe_path = _find_binary('ffprobe')
    if ffprobe_path is None:
        LOGGER.warning('ffprobe is not available; audio stream detection skipped')
        return False

    command = [
        ffprobe_path,
        '-v',
        'error',
        '-select_streams',
        'a',
        '-show_entries',
        'stream=index',
        '-of',
        'csv=p=0',
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe failed while checking audio streams: {result.stderr.strip()}')
    return bool(result.stdout.strip())


def _run_superres_pass(input_video_path, output_path, outscale, model_name, model_path, progress_every):
    if not _should_run_superres(outscale):
        return input_video_path

    upsampler = _create_superres_engine(model_name, model_path)
    fps, _, _, total_frames, _ = _probe_video(input_video_path)
    LOGGER.info('Opening processed video for super-resolution pass')
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f'Failed to open processed video for super-resolution: {input_video_path}')

    out = None
    frame_index = 0
    start_time = time.perf_counter()
    completed = False
    LOGGER.info('Starting super-resolution pass with outscale=%.2f', outscale)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                LOGGER.info('Reached end of super-resolution stream after %s frames', frame_index)
                break

            frame_index += 1
            enhanced_frame, _ = upsampler.enhance(frame, outscale=outscale)
            if out is None:
                out_height, out_width = enhanced_frame.shape[:2]
                out = _create_video_writer(output_path, fps, out_width, out_height)

            out.write(enhanced_frame)
            _log_enhancement_progress(frame_index, frame_index, total_frames, start_time, progress_every)

        completed = True
    finally:
        LOGGER.info('Releasing super-resolution pass resources')
        cap.release()
        if out is not None:
            out.release()
        if not completed and output_path.exists():
            output_path.unlink()

    return output_path


def _export_final_video(processed_video_path, source_video_path, output_path, export_crf, export_preset):
    ffmpeg_path = _find_binary('ffmpeg')
    if ffmpeg_path is None:
        LOGGER.warning('ffmpeg is not available; leaving output video without export pass')
        os.replace(processed_video_path, output_path)
        return

    export_output_path = _get_temp_output_path(output_path, marker='export')
    _cleanup_temp_file(export_output_path)
    has_audio = _source_has_audio(source_video_path)
    LOGGER.info(
        'Exporting final video with crf=%s, preset=%s, audio=%s',
        export_crf,
        export_preset,
        has_audio,
    )

    command = [ffmpeg_path, '-y', '-i', str(processed_video_path)]
    if has_audio:
        command.extend(['-i', str(source_video_path)])

    command.extend(['-map', '0:v:0'])
    if has_audio:
        command.extend(['-map', '1:a:0'])

    command.extend([
        '-c:v',
        'libx264',
        '-preset',
        export_preset,
        '-crf',
        str(export_crf),
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
    ])
    if has_audio:
        command.extend(['-c:a', 'aac', '-b:a', '192k', '-shortest'])

    command.append(str(export_output_path))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed while exporting final video: {result.stderr.strip()}')

    processed_video_path.unlink()
    os.replace(export_output_path, output_path)


def _finalize_output(
    temp_output_path,
    output_path,
    frame_index,
    swapped_frames,
    source_video_path,
    quality_settings,
    progress_every,
):
    temp_size = temp_output_path.stat().st_size if temp_output_path.exists() else 0
    if temp_size == 0:
        raise RuntimeError(f'Temporary output file is empty: {temp_output_path}')

    final_video_path = temp_output_path
    if _should_run_superres(quality_settings.export_scale):
        superres_output_path = _get_temp_output_path(output_path, marker='superres')
        _cleanup_temp_file(superres_output_path)
        final_video_path = _run_superres_pass(
            temp_output_path,
            superres_output_path,
            quality_settings.export_scale,
            quality_settings.superres_model_name,
            quality_settings.superres_model_path,
            progress_every,
        )
        if temp_output_path.exists() and temp_output_path != final_video_path:
            temp_output_path.unlink()

    _export_final_video(
        final_video_path,
        source_video_path,
        output_path,
        quality_settings.export_crf,
        quality_settings.export_preset,
    )
    LOGGER.info('Finished processing. Total frames=%s, swapped frames=%s', frame_index, swapped_frames)
    LOGGER.info('Output written to %s', output_path.resolve())


def _cleanup_temp_file(path):
    if path.exists():
        LOGGER.warning('Removing stale temporary output: %s', path)
        path.unlink()


def _remove_completed_target_video(target_video_path, batch_mode):
    if not batch_mode:
        return

    LOGGER.info('Removing processed source video %s', target_video_path)
    target_video_path.unlink()


def _run_swap_pass(
    source_path,
    video_path,
    output_path,
    det_size,
    providers,
    progress_every,
    detection_interval,
    identity_threshold,
    color_match_strength,
):
    app = _create_face_analysis(det_size, providers)
    swapper = _create_swapper(Path('models') / 'inswapper_128.onnx', providers)
    source_face = _load_source_face(app, str(source_path))

    fps, width, height, total_frames, _ = _probe_video(video_path)
    LOGGER.info('Opening target video for processing')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Failed to open target video: {video_path}')

    out = _create_video_writer(output_path, fps, width, height)
    frame_index = 0
    swapped_frames = 0
    swapped_frame_indices = set()
    prev_gray = None
    tracked_face = None
    locked_embedding = None
    start_time = time.perf_counter()
    completed = False
    LOGGER.info('Starting swap loop with detection_interval=%s', detection_interval)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                LOGGER.info('Reached end of video stream after %s frames', frame_index)
                break

            frame_index += 1
            target_face, frame_gray, locked_embedding = _resolve_target_face(
                app,
                frame,
                frame_index,
                prev_gray,
                tracked_face,
                detection_interval,
                locked_embedding,
                identity_threshold,
            )
            if target_face is not None:
                frame = _swap_frame(frame, swapper, source_face, target_face, color_match_strength)
                tracked_face = target_face
                swapped_frames += 1
                swapped_frame_indices.add(frame_index)
            else:
                tracked_face = None

            prev_gray = frame_gray
            out.write(frame)
            _log_progress_with_eta(
                frame_index,
                swapped_frames,
                total_frames,
                start_time,
                progress_every,
                phase='swap',
            )

        completed = True
    finally:
        LOGGER.info('Releasing swap pass resources')
        cap.release()
        out.release()
        if not completed and output_path.exists():
            output_path.unlink()

    return frame_index, swapped_frames, swapped_frame_indices


def _run_enhancement_pass(
    input_video_path,
    output_path,
    enhancer_model_path,
    progress_every,
    enhancement_interval,
    swapped_frame_indices,
):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    face_enhancer = _create_face_enhancer(enhancer_model_path, device, skip_enhancement=False)
    fps, width, height, total_frames, _ = _probe_video(input_video_path)
    LOGGER.info('Opening swapped video for enhancement')
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f'Failed to open swapped video: {input_video_path}')

    out = _create_video_writer(output_path, fps, width, height)
    frame_index = 0
    enhanced_frames = 0
    swapped_seen = 0
    start_time = time.perf_counter()
    completed = False
    LOGGER.info('Starting enhancement pass with enhancement_interval=%s', enhancement_interval)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                LOGGER.info('Reached end of enhanced video stream after %s frames', frame_index)
                break

            frame_index += 1
            should_enhance = False
            if frame_index in swapped_frame_indices:
                swapped_seen += 1
                should_enhance = ((swapped_seen - 1) % enhancement_interval == 0)

            if should_enhance:
                _, _, frame = face_enhancer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                enhanced_frames += 1

            out.write(frame)
            _log_enhancement_progress(frame_index, enhanced_frames, total_frames, start_time, progress_every)

        completed = True
    finally:
        LOGGER.info('Releasing enhancement pass resources')
        cap.release()
        out.release()
        if not completed and output_path.exists():
            output_path.unlink()

    return frame_index, enhanced_frames

def run_preflight(source_path, video_path, output_path, det_size, skip_enhancement, providers, quality_settings):
    _log_run_header(source_path, video_path, output_path, preflight=True)
    resolved_paths = _resolve_base_paths(source_path)
    target_videos = _collect_target_videos(video_path)

    app = _create_face_analysis(det_size, providers)
    _create_swapper(resolved_paths['swapper_model_path'], providers)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    _create_face_enhancer(resolved_paths['enhancer_model_path'], device, skip_enhancement)

    _load_source_face(app, str(resolved_paths['source_path']))

    batch_mode = Path(video_path).is_dir()
    for target_video in target_videos:
        fps, width, height, _, first_frame = _probe_video(target_video)
        resolved_output_path = _resolve_output_path(target_video, output_path, batch_mode=batch_mode)
        temp_output_path = _get_temp_output_path(resolved_output_path)
        _cleanup_temp_file(temp_output_path)
        _validate_video_writer(temp_output_path, fps, width, height, first_frame)
        if temp_output_path.exists():
            temp_output_path.unlink()

        LOGGER.info('Preflight validated %s -> %s', target_video.name, resolved_output_path.name)

    if _should_run_superres(quality_settings.export_scale):
        LOGGER.info('Validating RealESRGAN super-resolution model')
        upsampler = _create_superres_engine(
            quality_settings.superres_model_name,
            quality_settings.superres_model_path,
        )
        sample_video = target_videos[0]
        _, _, _, _, first_frame = _probe_video(sample_video)
        superres_frame, _ = upsampler.enhance(first_frame, outscale=quality_settings.export_scale)
        if superres_frame is None or superres_frame.size == 0:
            raise RuntimeError('RealESRGAN super-resolution validation failed')
    LOGGER.info('Preflight checks passed. videos=%s, det_size=%s, enhancement=%s', len(target_videos), det_size, not skip_enhancement)


def run_swap(
    source_path,
    video_path,
    output_path,
    det_size,
    skip_enhancement,
    providers,
    progress_every,
    quality_settings,
):
    _log_run_header(source_path, video_path, output_path)
    resolved_paths = _resolve_base_paths(source_path)
    source_path = resolved_paths['source_path']
    enhancer_model_path = resolved_paths['enhancer_model_path']
    target_videos = _collect_target_videos(video_path)
    batch_mode = Path(video_path).is_dir()

    for index, target_video_path in enumerate(target_videos, start=1):
        resolved_output_path = _resolve_output_path(target_video_path, output_path, batch_mode=batch_mode)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_temp_output_path = _get_temp_output_path(resolved_output_path, marker='incomplete')
        swap_temp_output_path = _get_temp_output_path(resolved_output_path, marker='swap')
        _cleanup_temp_file(final_temp_output_path)
        _cleanup_temp_file(swap_temp_output_path)

        swap_output_path = swap_temp_output_path if not skip_enhancement else final_temp_output_path
        LOGGER.info('Processing video %s/%s: %s', index, len(target_videos), target_video_path.name)
        LOGGER.info('Writing swap pass output to %s', swap_output_path)
        frame_index, swapped_frames, swapped_frame_indices = _run_swap_pass(
            source_path,
            target_video_path,
            swap_output_path,
            det_size,
            providers,
            progress_every,
            quality_settings.detection_interval,
            quality_settings.identity_threshold,
            quality_settings.color_match_strength,
        )

        if skip_enhancement:
            _finalize_output(
                final_temp_output_path,
                resolved_output_path,
                frame_index,
                swapped_frames,
                target_video_path,
                quality_settings,
                progress_every,
            )
            _remove_completed_target_video(target_video_path, batch_mode)
            continue

        LOGGER.info('Running post-process enhancement on selected swapped frames')
        enhanced_frame_index, enhanced_frames = _run_enhancement_pass(
            swap_temp_output_path,
            final_temp_output_path,
            enhancer_model_path,
            progress_every,
            quality_settings.enhancement_interval,
            swapped_frame_indices,
        )
        if swap_temp_output_path.exists():
            swap_temp_output_path.unlink()

        LOGGER.info('Enhancement pass completed. frames=%s, enhanced_frames=%s', enhanced_frame_index, enhanced_frames)
        _finalize_output(
            final_temp_output_path,
            resolved_output_path,
            frame_index,
            swapped_frames,
            target_video_path,
            quality_settings,
            progress_every,
        )
        _remove_completed_target_video(target_video_path, batch_mode)


def _build_parser():
    parser = argparse.ArgumentParser(description='Face swap a source image into a target video.')
    parser.add_argument('--source', default='source_face.jpg', help='Path to the source face image.')
    parser.add_argument('--target', default='target', help='Path to a target video file or a directory containing target videos.')
    parser.add_argument('--output', default='output', help='Path to an output MP4 file or a directory for batch outputs.')
    parser.add_argument(
        '--det-size',
        nargs=2,
        type=int,
        metavar=('WIDTH', 'HEIGHT'),
        default=(640, 640),
        help='Detection resolution for InsightFace. Lower values are faster but less accurate.',
    )
    parser.add_argument(
        '--provider',
        action='append',
        default=None,
        help='Preferred ONNX Runtime provider. Repeat to set fallback order, e.g. --provider CoreMLExecutionProvider --provider CPUExecutionProvider.',
    )
    parser.add_argument(
        '--skip-enhancement',
        action='store_true',
        help='Disable GFPGAN restoration to improve speed.',
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help='Log progress every N frames.',
    )
    parser.add_argument(
        '--detection-interval',
        type=int,
        default=DEFAULT_DETECTION_INTERVAL,
        help='Run full face detection every N frames and track faces in between.',
    )
    parser.add_argument(
        '--enhancement-interval',
        type=int,
        default=DEFAULT_ENHANCEMENT_INTERVAL,
        help='In post-process mode, enhance every Nth swapped frame. Use 1 to enhance every swapped frame.',
    )
    parser.add_argument(
        '--identity-threshold',
        type=float,
        default=DEFAULT_IDENTITY_THRESHOLD,
        help='Minimum embedding similarity required to keep swapping the same target identity.',
    )
    parser.add_argument(
        '--color-match-strength',
        type=float,
        default=DEFAULT_COLOR_MATCH_STRENGTH,
        help='Strength of local face color correction after swapping. Use 0 to disable.',
    )
    parser.add_argument(
        '--superres-model',
        default=DEFAULT_SUPERRES_MODEL_NAME,
        choices=sorted(SUPERRES_MODEL_CONFIGS),
        help='RealESRGAN model to use for the final super-resolution pass.',
    )
    parser.add_argument(
        '--superres-model-path',
        default=None,
        help='Optional local path or URL for the RealESRGAN weights file.',
    )
    parser.add_argument(
        '--export-scale',
        type=float,
        default=DEFAULT_EXPORT_SCALE,
        help='Final RealESRGAN outscale factor. Use 1.0 to skip the super-resolution pass.',
    )
    parser.add_argument(
        '--export-crf',
        type=int,
        default=DEFAULT_EXPORT_CRF,
        help='Final ffmpeg x264 CRF value. Lower is higher quality.',
    )
    parser.add_argument(
        '--export-preset',
        default=DEFAULT_EXPORT_PRESET,
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
        help='Final ffmpeg x264 preset for the export pass.',
    )
    parser.add_argument(
        '--preflight',
        action='store_true',
        help='Validate models, inputs, face detection, and video write support without processing the full video.',
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.progress_every <= 0:
        parser.error('--progress-every must be greater than 0')
    if args.detection_interval <= 0:
        parser.error('--detection-interval must be greater than 0')
    if args.enhancement_interval <= 0:
        parser.error('--enhancement-interval must be greater than 0')
    if not 0.0 <= args.identity_threshold <= 1.0:
        parser.error('--identity-threshold must be between 0 and 1')
    if not 0.0 <= args.color_match_strength <= 1.0:
        parser.error('--color-match-strength must be between 0 and 1')
    if args.export_scale < 1.0:
        parser.error('--export-scale must be greater than or equal to 1.0')
    if args.export_crf < 0:
        parser.error('--export-crf must be greater than or equal to 0')

    det_size = tuple(args.det_size)
    providers = _resolve_onnx_providers(args.provider or [])
    quality_settings = SimpleNamespace(
        detection_interval=args.detection_interval,
        enhancement_interval=args.enhancement_interval,
        identity_threshold=args.identity_threshold,
        color_match_strength=args.color_match_strength,
        superres_model_name=args.superres_model,
        superres_model_path=args.superres_model_path,
        export_scale=args.export_scale,
        export_crf=args.export_crf,
        export_preset=args.export_preset,
    )

    if args.preflight:
        run_preflight(args.source, args.target, args.output, det_size, args.skip_enhancement, providers, quality_settings)
        return None

    run_swap(
        args.source,
        args.target,
        args.output,
        det_size,
        args.skip_enhancement,
        providers,
        args.progress_every,
        quality_settings,
    )
    return None

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception('Face swap failed')
        sys.exit(1)