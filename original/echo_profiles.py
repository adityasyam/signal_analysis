'''
Calculate Echo profiles
2/17/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import json
import argparse
import numpy as np

from load_audio import load_audio
from filters import butter_bandpass_filter
from plot_profiles import plot_profiles_split_channels

def echo_profiles(audio_file, maxval, maxdiff, mindiff, sampling_rate=96000, n_channels=2, frame_length=600, no_overlapp=False, no_diff=False):

    _, all_audio = load_audio(audio_file)
    all_audio = np.reshape(all_audio, (-1, n_channels))

    tx_files = ['fmcw20000_b9000_l600.wav', 'fmcw31000_b9000_l600.wav']  #['fmcw35000_b12000_l600.wav', 'fmcw20000_b12000_l600.wav']
    tx_signals = []
    for f in tx_files:
        _, this_tx = load_audio(os.path.join('tx_signals', f))
        this_frame_length = this_tx.shape[0]
        assert(this_frame_length == frame_length)    # frame_length must match
        tx_signals += [this_tx]
    n_tx = len(tx_signals)
    bp_ranges = [[20000, 29000], [31000, 40000]]  #[[35000, 47000], [20000, 32000]]

    coi = list(range(n_channels))
    n_coi = len(coi)
    filtered_audio = np.zeros((n_tx, n_coi, all_audio.shape[0]))
    start_profiles = np.zeros((n_tx, n_coi, min(frame_length * 100000, all_audio.shape[0] // (frame_length * n_tx * n_coi) * (frame_length * n_tx * n_coi))))
  
    for n, tx in enumerate(tx_signals):
        for i, c in enumerate(coi):
            filtered_audio[n, i] = butter_bandpass_filter(all_audio[:, c], bp_ranges[n][0], bp_ranges[n][1], sampling_rate)
            # find the start pos
            channel_start_profiles = np.correlate(filtered_audio[n, i][:start_profiles.shape[2]], tx, mode='full')
            start_profiles[n, i, :] = channel_start_profiles[frame_length - 1:]
    
    start_profiles.shape = (n_tx, n_coi, -1, frame_length)
    start_profiles.shape = (-1, start_profiles.shape[3])
    start_profiles = np.mean(np.abs(start_profiles), axis=0)

    # start_pos = 520 + 300
    start_pos = np.argmax(start_profiles)
    start_pos = (start_pos + frame_length - frame_length // 2) % frame_length
    print('Detected start pos: %d' % start_pos)

    # return
    filtered_audio = filtered_audio[:, :, start_pos:]
    filtered_audio = filtered_audio[:, :, :filtered_audio.shape[2] - filtered_audio.shape[2] % frame_length]   # c x seq_len = c x (n_frames x frame_length)
    filtered_audio = np.reshape(filtered_audio, (filtered_audio.shape[0], filtered_audio.shape[1], -1, frame_length))      # c x n_frames x frame_length
    profiles = np.zeros((n_tx, n_coi, filtered_audio.shape[2] * filtered_audio.shape[3]))

    if no_overlapp:
        no_overlapp_text = '_no_overlapp'
        profiles = np.reshape(profiles, (n_tx, profiles.shape[0], -1, frame_length))
        for n, tx in enumerate(tx_signals):
            for c in range(n_coi):
                for f in range(profiles.shape[1]):
                    profiles[n, c, f, :] = np.correlate(filtered_audio[n, c, f, :], tx, mode='full')[frame_length - 1:]
    else:
        no_overlapp_text = ''
        for n, tx in enumerate(tx_signals):
            for c in range(n_coi):
                profiles[n, c, :] = np.correlate(filtered_audio[n, c].ravel(), tx, mode='full')[frame_length - 1:]
        profiles = np.reshape(profiles, (n_tx, profiles.shape[1], -1, frame_length))
    # profiles = np.reshape(profiles, (-1, profiles.shape[2], profiles.shape[3]))
    profiles = profiles.swapaxes(2, 3)                  # c x n_frames x frame_length -> c x frame_length x n_frames -> (c x frame_length) x n_frames
    profiles = np.reshape(profiles, (-1, profiles.shape[3]))
    # profiles.shape = -1, profiles.shape[2]
    profiles_img = plot_profiles_split_channels(profiles, n_tx * n_coi, maxval, minval=0)

    if not no_diff:
        diff_profiles = np.abs(profiles[:, 1:]) - np.abs(profiles[:, :-1])
        diff_profiles_img = plot_profiles_split_channels(diff_profiles, n_tx * n_coi, maxdiff, mindiff)

    cv2.imwrite('%s_fmcw%s_profiles.png' % (audio_file[:-4], no_overlapp_text), profiles_img)
    np.save('%s_fmcw%s_profiles.npy' % (audio_file[:-4], no_overlapp_text), profiles)
    if not no_diff:
        cv2.imwrite('%s_fmcw%s_diff_profiles.png' % (audio_file[:-4], no_overlapp_text), diff_profiles_img)
        np.save('%s_fmcw%s_diff_profiles.npy' % (audio_file[:-4], no_overlapp_text), diff_profiles)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Echo profile calculation')
    parser.add_argument('-a', '--audio', help='path to the audio file, .wav or .raw')
    parser.add_argument('-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-md', '--maxdiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-nd', '--mindiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('--sampling_rate', help='sampling rate of audio file', type=float, default=96000)
    parser.add_argument('--n_channels', help='number of channels in audio file', type=int, default=2)
    parser.add_argument('--frame_length', help='length of each audio frame', type=int, default=600)
    parser.add_argument('--no_overlapp', help='no overlapping while processing frames', action='store_true')
    parser.add_argument('--no_diff', help='do not generate differential echo profiles', action='store_true')

    parser.add_argument('--channels_of_interest', help='channels of interest (starting from 0)', default='')
    parser.add_argument('-tb', '--target_bitwidth', help='target bitwidth, 2-16', type=int, default=16)
    parser.add_argument('--sample_depth', help='sampling depth (bit) of audio file, comma-separated', type=int, default=16)
    parser.add_argument('--bandpass_range', help='bandpass range, LOWCUT,HIGHCUT', default='')

    args = parser.parse_args()
    
    echo_profiles(args.audio, args.maxval, args.maxdiffval, args.mindiffval, args.sampling_rate, args.n_channels, args.frame_length, args.no_overlapp, args.no_diff)