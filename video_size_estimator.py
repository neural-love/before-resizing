import cv2
import numpy as np
import sys
import os
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def analyze_frequency_spectrum(frame, new_shape=(2048, 2048)):
    """Apply FFT to analyze the frequency spectrum of a frame using zero-padding for higher resolution."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Zero-padding to increase the resolution of the FFT
    f_transform = np.fft.fft2(gray, new_shape)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    return magnitude_spectrum


def find_peak_intervals(signal, min_distance, height_multiplier):
    """Find intervals between peaks in a signal with smoothing and adaptive thresholding."""
    signal_smoothed = np.convolve(signal, np.ones(5) / 5, mode='same')

    mean_signal = np.mean(signal_smoothed)
    std_signal = np.std(signal_smoothed)
    peaks, _ = find_peaks(signal_smoothed, height=mean_signal + height_multiplier * std_signal, distance=min_distance)

    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks)
    return np.median(intervals) if intervals.size > 0 else None


def detect_aliasing_patterns(freq_spectrum, min_distance, height_multiplier):
    """Estimate intervals based on aliasing patterns in the frequency spectrum using improved peak detection."""
    horizontal_sum = np.sum(freq_spectrum, axis=0)
    vertical_sum = np.sum(freq_spectrum, axis=1)

    horizontal_interval = find_peak_intervals(horizontal_sum, min_distance, height_multiplier)
    vertical_interval = find_peak_intervals(vertical_sum, min_distance, height_multiplier)

    return horizontal_interval, vertical_interval


def generate_reference_patterns(min_distance, height_multiplier):
    """Generate reference patterns for bicubic interpolation."""
    reference_patterns = {}
    base_square = np.ones((5, 5), dtype=np.uint8) * 0  # 5x5 black square
    base_image = np.ones((100, 100), dtype=np.uint8) * 255  # White background
    base_image[47:52, 47:52] = base_square

    for scale in np.linspace(0.5, 3, 30):  # Example scales from 0.5x to 3x
        scaled_image = cv2.resize(base_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Convert to 3-channel color image
        scaled_image_color = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
        freq_spectrum = analyze_frequency_spectrum(scaled_image_color)
        horizontal_sum = np.sum(freq_spectrum, axis=0)
        vertical_sum = np.sum(freq_spectrum, axis=1)
        horizontal_interval = find_peak_intervals(horizontal_sum, min_distance, height_multiplier)
        vertical_interval = find_peak_intervals(vertical_sum, min_distance, height_multiplier)
        reference_patterns[scale] = (horizontal_interval, vertical_interval)

    return reference_patterns


def match_reference_patterns(freq_spectrum, reference_patterns, min_distance, height_multiplier):
    """Match the frequency spectrum of a frame with reference patterns."""
    horizontal_sum = np.sum(freq_spectrum, axis=0)
    vertical_sum = np.sum(freq_spectrum, axis=1)
    horizontal_interval = find_peak_intervals(horizontal_sum, min_distance, height_multiplier)
    vertical_interval = find_peak_intervals(vertical_sum, min_distance, height_multiplier)

    best_match = None
    best_score = float('inf')

    for scale, (ref_horiz, ref_vert) in reference_patterns.items():
        horiz_diff = abs(ref_horiz - horizontal_interval) if ref_horiz and horizontal_interval else float('inf')
        vert_diff = abs(ref_vert - vertical_interval) if ref_vert and vertical_interval else float('inf')

        score = horiz_diff + vert_diff

        if score < best_score:
            best_score = score
            best_match = scale

    return best_match


def estimate_size_from_scale(resized_size, scale):
    """Estimate the original size based on the detected scale."""
    if scale is None:
        return resized_size
    return int(resized_size[0] / scale), int(resized_size[1] / scale)


def process_frames(video_path, start_frame, end_frame, reference_patterns, resized_size, min_distance, height_multiplier):
    """Process a range of frames and return the estimated scales."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    estimated_scales = []
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Unable to read frame number {frame_num}")
            continue

        try:
            freq_spectrum = analyze_frequency_spectrum(frame)
            scale = match_reference_patterns(freq_spectrum, reference_patterns, min_distance, height_multiplier)
            if scale:
                estimated_scales.append(scale)
            print(f"Processed frame {frame_num + 1}/{end_frame}, Current Scale Estimate: {scale}")
        except Exception as e:
            print(f"Error processing frame {frame_num + 1}/{end_frame}: {e}")
            continue

    cap.release()
    return estimated_scales


def estimate_original_size(video_path, min_distance, height_multiplier):
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None

    resized_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resized_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resized_size = (resized_width, resized_height)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")

    num_threads = os.cpu_count() or 1
    frames_per_thread = (frame_count + num_threads - 1) // num_threads

    print(f"Using {num_threads} threads for processing")

    reference_patterns = generate_reference_patterns(min_distance, height_multiplier)

    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start_frame = i * frames_per_thread
            end_frame = min((i + 1) * frames_per_thread, frame_count)
            futures.append(executor.submit(process_frames, video_path, start_frame, end_frame, reference_patterns, resized_size, min_distance, height_multiplier))

    all_estimated_scales = []
    for future in as_completed(futures):
        result = future.result()
        if result:
            all_estimated_scales.extend(result)

    if all_estimated_scales:
        estimated_scale = np.median(all_estimated_scales)
    else:
        estimated_scale = 1

    estimated_width, estimated_height = estimate_size_from_scale(resized_size, estimated_scale)

    print(f"Most probable original size: {estimated_width}x{estimated_height}")
    return estimated_width, estimated_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the original size of a video before resizing")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--min_distance", type=int, default=10, help="Minimum distance between peaks")
    parser.add_argument("--height_multiplier", type=float, default=1.0, help="Multiplier for peak detection threshold")
    args = parser.parse_args()

    estimate_original_size(args.video_path, args.min_distance, args.height_multiplier)
