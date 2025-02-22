import os
import cv2
import argparse
import numpy as np
import scipy.signal as signal

from load_audio import load_audio
from plot_profiles import plot_profiles_split_channels

def crop_and_save_images(profiles, prefix, n_coi, maxval, minval, segment_width=1000):
    """Crop profiles into segments and save as separate images"""
    num_segments = int(np.ceil(profiles.shape[1] / segment_width))
    
    for i in range(num_segments):
        start_idx = i * segment_width
        end_idx = min((i + 1) * segment_width, profiles.shape[1])
        cropped_profile = profiles[:, start_idx:end_idx]
        
        profile_img = plot_profiles_split_channels(cropped_profile, n_coi, maxval, minval)
        if profile_img is not None:
            img_filename = f"{prefix}_segment_{i}.png"
            npy_filename = f"{prefix}_segment_{i}.npy"
            cv2.imwrite(img_filename, profile_img)
            np.save(npy_filename, cropped_profile)

def echo_profiles(audio_file, maxval, maxdiff, mindiff, sampling_rate=96000, n_channels=2, frame_length=600, no_overlapp=False, no_diff=False, segment_width=1000):
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
        profiles_list.append(np.abs(Zxx).ravel())
    
    min_size = min(map(len, profiles_list))
    profiles = np.array([p[:min_size] for p in profiles_list])
    
    crop_and_save_images(profiles, f"{audio_file[:-4]}_profiles", n_coi, maxval, minval=0, segment_width=segment_width)
    
    if not no_diff:
        diff_profiles = np.abs(profiles[:, 1:]) - np.abs(profiles[:, :-1])
        crop_and_save_images(diff_profiles, f"{audio_file[:-4]}_diff_profiles", n_coi, maxdiff, mindiff, segment_width=segment_width)
    
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
    parser.add_argument('--segment_width', help='width of each cropped segment', type=int, default=1000)
    
    args = parser.parse_args()
    
    echo_profiles(args.audio, args.maxval, args.maxdiffval, args.mindiffval, args.sampling_rate, args.n_channels, args.frame_length, args.no_overlapp, args.no_diff, args.segment_width)
