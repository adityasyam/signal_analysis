import os
import cv2
import argparse
import numpy as np
import scipy.signal as signal

from load_audio import load_audio
from plot_profiles import plot_profiles_split_channels

def echo_profiles(audio_file, maxval, maxdiff, mindiff, sampling_rate=96000, n_channels=2, frame_length=600, no_overlapp=False, no_diff=False, time_window=20, max_image_width=1000):
    
    _, all_audio = load_audio(audio_file)
    all_audio = np.reshape(all_audio, (-1, n_channels))
    
    coi = list(range(n_channels))
    n_coi = len(coi)
    
    start_pos = 0
    print('Detected start pos: %d' % start_pos)
    
    profiles_list = []
    
    for c in range(n_coi):
        f, t, Zxx = signal.stft(all_audio[:, c], fs=sampling_rate, nperseg=frame_length, 
                                noverlap=None if no_overlapp else frame_length // 2)
        
        # Extract only the first `time_window` seconds
        time_mask = t <= time_window
        Zxx = Zxx[:, time_mask]

        profiles_list.append(np.abs(Zxx))  # Store the magnitude of STFT
    
    # Convert list to array and ensure all have the same width
    min_size = min(map(lambda x: x.shape[1], profiles_list))  # Get min width (time bins)
    profiles = np.array([p[:, :min_size] for p in profiles_list])  # Crop all to min width

    # Reshape: Flatten frequency bins into a single vector per time step
    profiles = profiles.reshape(n_coi, -1)

    # Limit image width to prevent libpng errors
    profiles = profiles[:, :max_image_width]  # Crop to max width

    profiles_img = plot_profiles_split_channels(profiles, n_coi, maxval, minval=0)

    if profiles_img is not None:
        cv2.imwrite('%s_profiles_short.png' % (audio_file[:-4]), profiles_img)
        np.save('%s_profiles_short.npy' % (audio_file[:-4]), profiles)
    
    if not no_diff:
        diff_profiles = np.abs(profiles[:, 1:]) - np.abs(profiles[:, :-1])

        # Crop differential profiles to max image width
        diff_profiles = diff_profiles[:, :max_image_width]

        diff_profiles_img = plot_profiles_split_channels(diff_profiles, n_coi, maxdiff, mindiff)

        if diff_profiles_img is not None:
            cv2.imwrite('%s_diff_profiles_short.png' % (audio_file[:-4]), diff_profiles_img)
            np.save('%s_diff_profiles_short.npy' % (audio_file[:-4]), diff_profiles)

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
    parser.add_argument('--time_window', help='length of time (seconds) to extract for visualization', type=int, default=20)
    
    args = parser.parse_args()
    
    echo_profiles(args.audio, args.maxval, args.maxdiffval, args.mindiffval, args.sampling_rate, args.n_channels, args.frame_length, args.no_overlapp, args.no_diff, args.time_window)
