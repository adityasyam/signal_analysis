import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from adaptive_static import adaptive_static


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

def plot_profiles_split_channels(profiles, n_channels, maxval, minval=None, sampling_rate=96000, frame_length=600):
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
    max_time = profiles.shape[1] * time_resolution
    
    # Create margins for labels
    margin_left = 100   # Space for y-axis labels
    margin_bottom = 50  # Space for x-axis labels
    margin_top = 40     # Space for title
    margin_right = 30   # Space on the right
    
    # Increase the width-to-height ratio significantly - make it much wider
    # Use a fixed channel height that's smaller than default to make it wider
    channel_display_height = min(channel_width, 150)  # Limit channel height for better proportions
    
    # Calculate a good image width - make it significantly wider
    # For a 0.2 second window, aim for a width that's about 4-5x the height of a single channel
    target_width = 800  # Fixed width for better visualization
    
    # If the profiles are too long, downsample them to target width
    # If they're too short, keep them as is
    if profiles.shape[1] > target_width:
        indices = np.linspace(0, profiles.shape[1]-1, target_width).astype(int)
        downsampled_profiles = profiles[:, indices]
    else:
        # If original data is shorter than target, pad with zeros on the right
        padding_width = target_width - profiles.shape[1]
        if padding_width > 0:
            downsampled_profiles = np.pad(profiles, ((0, 0), (0, padding_width)), mode='constant')
        else:
            downsampled_profiles = profiles
    
    # Create image with space for labels - fixed width, appropriate height
    img_height = (channel_display_height + 20) * n_channels + margin_bottom + margin_top
    img_width = target_width + margin_left + margin_right
    profiles_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Add channel spectrograms
    for n in range(n_channels):
        # Extract channel data
        channel_data = downsampled_profiles[n * channel_width: (n + 1) * channel_width]
        
        # If the channel data is taller than our display height, resize it
        if channel_width > channel_display_height:
            # Resize to our target height while preserving width
            channel_img_full = plot_profiles(channel_data, maxval, minval)
            channel_img = cv2.resize(channel_img_full, (channel_img_full.shape[1], channel_display_height))
        else:
            channel_img = plot_profiles(channel_data, maxval, minval)
        
        # Position in the main image
        y_start = n * (channel_display_height + 20) + margin_top
        y_end = y_start + channel_display_height
        
        # Place the channel image
        x_end = margin_left + channel_img.shape[1]
        profiles_img[y_start:y_end, margin_left:x_end, :] = channel_img
        
        # Add channel label
        cv2.putText(profiles_img, f"Channel {n}", 
                   (10, y_start + channel_display_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Add frequency ticks and labels for each channel
    for n in range(n_channels):
        y_start = n * (channel_display_height + 20) + margin_top
        
        # Add frequency labels (y-axis)
        # Add ticks at 0%, 25%, 50%, 75%, and 100% of the frequency range
        for i, percent in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
            y_pos = int(y_start + channel_display_height * (1 - percent))
            freq_val = int(max_freq * percent / 1000)  # Convert to kHz
            
            # Draw tick
            cv2.line(profiles_img, (margin_left - 5, y_pos), (margin_left, y_pos), (0, 0, 0), 1)
            
            # Add label
            cv2.putText(profiles_img, f"{freq_val} kHz", 
                       (margin_left - 80, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add time labels (x-axis)
    num_ticks = 8  # More time ticks for better detail
    for i in range(num_ticks):
        x_percent = i / (num_ticks - 1)
        x_pos = int(margin_left + target_width * x_percent)
        
        # Calculate the actual time value
        time_val = max_time * x_percent
        
        # Draw tick
        cv2.line(profiles_img, (x_pos, img_height - margin_bottom), 
                (x_pos, img_height - margin_bottom + 5), (0, 0, 0), 1)
        
        # Add label - use more decimal places for short intervals
        if max_time < 1.0:
            time_label = f"{time_val*1000:.1f} ms"  # Show in milliseconds for short windows
        else:
            time_label = f"{time_val:.3f} s"
            
        cv2.putText(profiles_img, time_label, 
                   (x_pos - 30, img_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add axis titles
    cv2.putText(profiles_img, "Frequency", 
               (20, margin_top // 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    cv2.putText(profiles_img, "Time", 
               (img_width // 2, img_height - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Add main title with sampling info
    cv2.putText(profiles_img, f"STFT Analysis (SR: {sampling_rate/1000:.1f}kHz, Window: {frame_length/sampling_rate*1000:.1f}ms)", 
               (margin_left, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return profiles_img

def plot_profiles_file(file_path, draw_diff, n_channels, maxval, maxdiff, mindiff, sampling_rate=96000, frame_length=600):
    profiles = np.load(file_path)
    profiles_img = plot_profiles_split_channels(profiles, n_channels, maxval, None, sampling_rate, frame_length)
    cv2.imwrite(file_path[:-4] + '.png', profiles_img)
    if draw_diff:
        diff_profiles = profiles[:, 1:] - profiles[:, :-1]
        diff_profiles_img = plot_profiles_split_channels(diff_profiles, n_channels, maxdiff, mindiff, sampling_rate, frame_length)
        cv2.imwrite(file_path[:-13] + '_diff_profiles.png', diff_profiles_img)


# def draw_profiles_static(profiles, large_window, avg_window, n_avg_windows, n_channels=2):

#     print('Calculating static profiles')
#     static_profiles = adaptive_static(profiles, large_window, avg_window, n_avg_windows)
#     nostatic_profiles = profiles - static_profiles

#     print('Ploting static and nostatic profiless')
#     static_profiles_img = plot_profiles_split_channels(static_profiles, n_channels)
#     nostatic_profiles_img = plot_profiles_split_channels(nostatic_profiles, n_channels)

#     return static_profiles, static_profiles_img, nostatic_profiles_img

# def draw_profiles_static_with_path(profiles_path, large_window, avg_window, n_avg_windows, n_channels=2):
#     profiles = np.load(profiles_path)

#     static_profiles, static_profiles_img, nostatic_profiles_img = draw_profiles_static(profiles, large_window, avg_window, n_avg_windows, n_channels)

#     print('Saving files')
#     np.save(profiles_path[:-7] + 'static_profiles.npy', static_profiles)
#     # np.save(profiles_path[:-7] + '_simple_nostatic_profiles.png', nostatic_profiles_img)

#     cv2.imwrite(profiles_path[:-7] + 'static_profiles.png', static_profiles_img)
#     cv2.imwrite(profiles_path[:-7] + 'simple_nostatic_profiles.png', nostatic_profiles_img)

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