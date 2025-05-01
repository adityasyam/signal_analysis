import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_profiles(profiles, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    # print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros((heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map

def plot_profiles_split_channels(profiles, n_channels, maxval, minval=None, sampling_rate=96000, 
                                frame_length=600, start_time=0.0, interval_duration=None):
    """
    Plot STFT profiles with proper time axis labels.
    
    Parameters:
    -----------
    profiles : numpy.ndarray
        The STFT profiles data
    n_channels : int
        Number of channels to display
    maxval : float
        Maximum value for color scaling
    minval : float, optional
        Minimum value for color scaling
    sampling_rate : int, optional
        Sampling rate of the audio file in Hz
    frame_length : int, optional
        Length of each frame in samples
    start_time : float, optional
        Start time of the interval in seconds, used for correct x-axis labeling
    interval_duration : float, optional
        Duration of the interval in seconds, used for correct x-axis scaling
    
    Returns:
    --------
    numpy.ndarray
        Image representation of STFT profiles
    """
    channel_width = profiles.shape[0] // n_channels
    
    # Calculate actual frequency and time values
    # For frequency axis (y-axis)
    freq_resolution = sampling_rate / frame_length
    max_freq = sampling_rate / 2  # Nyquist frequency
    freq_ticks = np.linspace(0, max_freq, channel_width)
    
    # For time axis (x-axis)
    # Time resolution depends on hop length, which is frame_length/2 by default
    hop_length = frame_length // 2
    time_resolution = hop_length / sampling_rate
    
    # Calculate the actual time span of the analyzed interval
    if interval_duration is None:
        # Calculate from the data if not provided
        interval_duration = profiles.shape[1] * time_resolution
    else:
        # Make sure our time axis matches the actual interval
        # This is important for correct time labeling
        time_resolution = interval_duration / profiles.shape[1]
    
    # Create margins for labels
    margin_left = 120   # Space for y-axis labels
    margin_bottom = 60  # Space for x-axis labels
    margin_top = 50     # Space for title
    margin_right = 40   # Space on the right
    
    # Create a balanced image size
    # Use a reasonable width that's not too stretched
    target_width = 800
    
    # MAJOR CHANGE: Make channel display height MUCH LARGER to see patterns clearly
    # This will increase the vertical resolution significantly
    # Default channel_width might be around 200-300, we'll make it much larger
    channel_display_height = 400  # Significantly larger height for each channel
    
    # Adjust the profile width to maintain a good aspect ratio
    if profiles.shape[1] > target_width:
        indices = np.linspace(0, profiles.shape[1]-1, target_width).astype(int)
        downsampled_profiles = profiles[:, indices]
    else:
        # For short intervals, keep original time resolution
        downsampled_profiles = profiles
    
    # Create image with space for labels - now with much taller channels
    img_height = (channel_display_height + 40) * n_channels + margin_bottom + margin_top
    img_width = target_width + margin_left + margin_right
    profiles_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Add channel spectrograms
    for n in range(n_channels):
        # Extract channel data
        channel_data = downsampled_profiles[n * channel_width: (n + 1) * channel_width]
        
        # Resize the channel data to have more vertical resolution
        channel_img_full = plot_profiles(channel_data, maxval, minval)
        
        # Resize to taller height while preserving width
        channel_img = cv2.resize(channel_img_full, (channel_img_full.shape[1], channel_display_height))
        
        # Position in the main image
        y_start = n * (channel_display_height + 40) + margin_top
        y_end = y_start + channel_display_height
        
        # Place the channel image
        x_end = margin_left + channel_img.shape[1]
        profiles_img[y_start:y_end, margin_left:x_end, :] = channel_img
        
        # Add channel label
        cv2.putText(profiles_img, f"Channel {n}", 
                   (10, y_start + channel_display_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Add frequency ticks and labels for each channel - more frequency ticks for better detail
    for n in range(n_channels):
        y_start = n * (channel_display_height + 40) + margin_top
        
        # Add more frequency labels (y-axis) for better detail
        # Add ticks at 0%, 10%, 20%, ..., 100% of the frequency range
        for i, percent in enumerate(np.linspace(0, 1.0, 11)):
            y_pos = int(y_start + channel_display_height * (1 - percent))
            freq_val = int(max_freq * percent / 1000)  # Convert to kHz
            
            # Draw tick
            cv2.line(profiles_img, (margin_left - 5, y_pos), (margin_left, y_pos), (0, 0, 0), 1)
            
            # Add label
            cv2.putText(profiles_img, f"{freq_val} kHz", 
                       (margin_left - 100, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add faint horizontal grid lines for better frequency tracking
            cv2.line(profiles_img, (margin_left, y_pos), (margin_left + target_width, y_pos), 
                    (200, 200, 200), 1, cv2.LINE_AA)
    
    # Add time labels (x-axis) - CORRECTLY ADJUSTED FOR THE ACTUAL INTERVAL
    num_ticks = 8  # Time ticks for balanced detail
    for i in range(num_ticks):
        x_percent = i / (num_ticks - 1)
        x_pos = int(margin_left + target_width * x_percent)
        
        # Calculate the actual time value with interval start time included
        # This now correctly reflects the current interval's time range
        time_val = start_time + interval_duration * x_percent
        
        # Draw tick
        cv2.line(profiles_img, (x_pos, img_height - margin_bottom), 
                (x_pos, img_height - margin_bottom + 5), (0, 0, 0), 1)
        
        # Add label - use more decimal places for short intervals
        if interval_duration < 1.0:
            time_label = f"{time_val*1000:.1f} ms"  # Show in milliseconds for short windows
        else:
            time_label = f"{time_val:.3f} s"
            
        cv2.putText(profiles_img, time_label, 
                   (x_pos - 30, img_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add faint vertical grid lines for better time tracking
        cv2.line(profiles_img, (x_pos, margin_top), (x_pos, img_height - margin_bottom), 
                (200, 200, 200), 1, cv2.LINE_AA)
    
    # Add axis titles - larger font for better visibility
    cv2.putText(profiles_img, "Frequency", 
               (20, margin_top // 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    cv2.putText(profiles_img, "Time", 
               (img_width // 2, img_height - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Add main title with sampling info and interval time - larger font
    interval_end_time = start_time + interval_duration
    title = f"STFT Analysis (SR: {sampling_rate/1000:.1f}kHz, Window: {frame_length/sampling_rate*1000:.1f}ms)"
    subtitle = f"Interval: {start_time:.2f}s - {interval_end_time:.2f}s"
    
    cv2.putText(profiles_img, title, 
               (margin_left, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    cv2.putText(profiles_img, subtitle, 
               (margin_left, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return profiles_img

def plot_profiles_file(file_path, draw_diff, n_channels, maxval, maxdiff, mindiff, 
                      sampling_rate=96000, frame_length=600):
    profiles = np.load(file_path)
    
    # Extract start time from the filename if it exists
    # Format: filename_interval_1_1.23s-1.45s_profiles.npy
    start_time = 0.0
    end_time = None
    import re
    start_match = re.search(r'_(\d+\.\d+)s-', file_path)
    end_match = re.search(r'-(\d+\.\d+)s_', file_path)
    if start_match:
        start_time = float(start_match.group(1))
    if end_match:
        end_time = float(end_match.group(1))
        interval_duration = end_time - start_time
    else:
        interval_duration = None
    
    profiles_img = plot_profiles_split_channels(profiles, n_channels, maxval, None, 
                                              sampling_rate, frame_length, 
                                              start_time, interval_duration)
    cv2.imwrite(file_path[:-4] + '.png', profiles_img)
    
    if draw_diff:
        diff_profiles = profiles[:, 1:] - profiles[:, :-1]
        diff_profiles_img = plot_profiles_split_channels(diff_profiles, n_channels, maxdiff, mindiff, 
                                                      sampling_rate, frame_length, 
                                                      start_time, interval_duration)
        cv2.imwrite(file_path[:-13] + '_diff_profiles.png', diff_profiles_img)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--profiles-file', help='path to the .npy profiles file')
    parser.add_argument('--diff', help='whether to output differential profiless', action='store_true')
    parser.add_argument('-n', '--n-channels', help='number of channels', type=int, default=2)
    parser.add_argument('-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=int, default=8)
    parser.add_argument('-md', '--maxdiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=int, default=0.3)
    parser.add_argument('-nd', '--mindiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=int, default=-0.3)
    parser.add_argument('--sampling_rate', help='sampling rate of audio file', type=float, default=96000)
    parser.add_argument('--frame_length', help='length of each audio frame', type=int, default=600)

    args = parser.parse_args()
    plot_profiles_file(args.profiles_file, args.diff, args.n_channels, args.maxval, args.maxdiffval, args.mindiffval, args.sampling_rate, args.frame_length)