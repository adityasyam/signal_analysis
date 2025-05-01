import os
import cv2
import argparse
import numpy as np
import scipy.signal as signal
import random
import matplotlib.pyplot as plt

from load_audio import load_audio
from plot_profiles import plot_profiles_split_channels

def echo_profiles(audio_file, maxval, maxdiff, mindiff, sampling_rate=96000, n_channels=2, 
                  frame_length=600, no_overlapp=False, no_diff=False, interval_start=None, interval_length=2.0):
    
    _, all_audio = load_audio(audio_file)
    all_audio = np.reshape(all_audio, (-1, n_channels))
    
    # Calculate total duration in seconds
    total_duration = len(all_audio) / sampling_rate
    print(f'Total audio duration: {total_duration:.2f} seconds')
    
    # Set the interval start position
    if interval_start is None:
        # Choose a random start if not specified
        max_start_point = int((total_duration - interval_length) * sampling_rate)
        if max_start_point <= 0:
            print(f"Audio file is shorter than {interval_length} seconds. Processing entire file.")
            start_sample = 0
        else:
            start_sample = random.randint(0, max_start_point)
    else:
        # Use the specified start position
        if interval_start < 0 or interval_start >= total_duration:
            print(f"Invalid interval_start: {interval_start}. Using 0.")
            interval_start = 0
        start_sample = int(interval_start * sampling_rate)
    
    # Calculate end sample
    end_sample = start_sample + int(interval_length * sampling_rate)
    
    # Ensure we don't exceed the audio length
    end_sample = min(end_sample, len(all_audio))
    
    # Extract the exact interval we want to analyze
    interval_audio = all_audio[start_sample:end_sample]
    
    # Calculate actual time values
    interval_start_time = start_sample / sampling_rate
    interval_end_time = end_sample / sampling_rate
    actual_interval_length = (end_sample - start_sample) / sampling_rate
    
    print(f'Processing interval: {interval_start_time:.2f}s - {interval_end_time:.2f}s (Duration: {actual_interval_length:.2f}s)')
    
    coi = list(range(n_channels))
    n_coi = len(coi)
    
    profiles_list = []
    
    for c in range(n_coi):
        # Only process the selected interval
        f, t, Zxx = signal.stft(interval_audio[:, c], fs=sampling_rate, nperseg=frame_length, 
                               noverlap=None if no_overlapp else frame_length // 2)
        profiles_list.append(np.abs(Zxx).ravel())
    
    # Dynamically set the profile array size
    min_size = min(map(len, profiles_list))
    profiles = np.array([p[:min_size] for p in profiles_list])
    
    # Generate a descriptive filename with the time interval
    interval_filename = f"{audio_file[:-4]}_interval_{interval_start_time:.2f}s-{interval_end_time:.2f}s"
    
    # Pass the correct time values
    profiles_img = plot_profiles_split_channels(profiles, n_coi, maxval, minval=0, 
                                              sampling_rate=sampling_rate, frame_length=frame_length,
                                              start_time=interval_start_time,
                                              interval_duration=actual_interval_length)
    
    if profiles_img is not None:
        cv2.imwrite(f'{interval_filename}_profiles.png', profiles_img)
        np.save(f'{interval_filename}_profiles.npy', profiles)
    
    if not no_diff:
        diff_profiles = np.abs(profiles[:, 1:]) - np.abs(profiles[:, :-1])
        
        # Also pass correct time values to the diff profiles plot
        diff_profiles_img = plot_profiles_split_channels(diff_profiles, n_coi, maxdiff, mindiff,
                                                      sampling_rate=sampling_rate, frame_length=frame_length,
                                                      start_time=interval_start_time,
                                                      interval_duration=actual_interval_length)
        
        if diff_profiles_img is not None:
            cv2.imwrite(f'{interval_filename}_diff_profiles.png', diff_profiles_img)
            np.save(f'{interval_filename}_diff_profiles.npy', diff_profiles)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Echo profile calculation')
    parser.add_argument('-a', '--audio', help='path to the audio file, .wav or .raw')
    parser.add_argument('-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-md', '--maxdiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-nd', '--mindiffval', help='minval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('--sampling_rate', help='sampling rate of audio file', type=float, default=96000)
    parser.add_argument('--n_channels', help='number of channels in audio file', type=int, default=2)
    parser.add_argument('--frame_length', help='length of each audio frame', type=int, default=600)
    parser.add_argument('--no_overlapp', help='disable overlapping while processing frames', action='store_true')
    parser.add_argument('--no_diff', help='do not generate differential echo profiles', action='store_true')
    parser.add_argument('--interval_start', help='start time of the interval in seconds', type=float, default=None)
    parser.add_argument('--interval_length', help='length of the interval in seconds', type=float, default=2.0)
    
    args = parser.parse_args()
    
    echo_profiles(args.audio, args.maxval, args.maxdiffval, args.mindiffval, 
                 args.sampling_rate, args.n_channels, args.frame_length, 
                 args.no_overlapp, args.no_diff, args.interval_start, args.interval_length)

#USAGE: python echo_profiles.py -a ../facial1.raw --interval_start 25.53 --interval_length 2.0