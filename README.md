# Video Original Size Estimator

This Python script estimates the original size of a video before resizing using frequency spectrum analysis and pattern matching techniques. It can be useful for recovering the original dimensions of a video that has been resized using bicubic interpolation.

## Features

- Analyzes the frequency spectrum of video frames using Fast Fourier Transform (FFT) with zero-padding for higher resolution
- Detects aliasing patterns in the frequency spectrum to estimate the scaling factor
- Generates reference patterns for various scaling factors using bicubic interpolation
- Matches the frequency spectrum of each frame with the reference patterns to determine the best matching scale
- Utilizes multi-threading to process video frames in parallel for faster execution
- Provides a command-line interface for easy usage

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- SciPy
- argparse

## Usage

1. Install the required dependencies:
   ```
   pip install opencv-python numpy scipy argparse
   ```

2. Run the script from the command line:
   ```
   python video_size_estimator.py video_path [--min_distance MIN_DISTANCE] [--height_multiplier HEIGHT_MULTIPLIER]
   ```
   - `video_path`: Path to the video file
   - `--min_distance` (optional): Minimum distance between peaks in the frequency spectrum (default: 10)
   - `--height_multiplier` (optional): Multiplier for peak detection threshold (default: 1.0)

## How It Works

1. The script reads the video file and extracts the resized frame dimensions.
2. It generates reference patterns for various scaling factors using bicubic interpolation.
3. The video frames are processed in parallel using multi-threading to analyze their frequency spectra.
4. For each frame, the script detects aliasing patterns in the frequency spectrum and matches them with the reference patterns to estimate the scaling factor.
5. The estimated scaling factors from all frames are aggregated, and the median value is used as the final estimate.
6. The script calculates the estimated original size by dividing the resized dimensions by the estimated scaling factor.
7. Finally, it prints the most probable original size of the video.

## Example

```
python video_size_estimator.py sample_video.mp4 --min_distance 15 --height_multiplier 1.2
```

Output:
```
Total frames in video: 1000
Using 8 threads for processing
Processed frame 1/125, Current Scale Estimate: 1.5
...
Processed frame 1000/1000, Current Scale Estimate: 1.5
Most probable original size: 1920x1080
```

## Notes

- The script assumes that the video was resized using bicubic interpolation. It may not work accurately for other resizing methods.
- The accuracy of the size estimation depends on the quality and characteristics of the video, as well as the chosen parameters (`min_distance` and `height_multiplier`).
- Adjusting the `min_distance` and `height_multiplier` parameters may help improve the estimation accuracy for specific videos.
- The script may take some time to process long videos or high-resolution frames. Multi-threading is used to speed up the processing.

Feel free to explore and modify the code to suit your specific needs. If you have any questions or encounter any issues, please don't hesitate to reach out.
