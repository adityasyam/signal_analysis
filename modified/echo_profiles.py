import os
import cv2
import json
import argparse
import numpy as np
import scipy.signal as signal

from load_audio import load_audio
from filters import butter_bandpass_filter
from plot_profiles import plot_profiles_split_channels

def echo_profiles(audio_file, maxval, maxdiff, mindiff, sampling_rate=96000, n_channels=2, frame_length=600, no_overlapp=False, no_diff=False):

    _, all_audio = load_audio(audio_file)
    all_audio = np.reshape(all_audio, (-1, n_channels))

    tx_files = ['fmcw20000_b9000_l600.wav', 'fmcw31000_b9000_l600.wav']
    tx_signals = []
    for f in tx_files:
        _, this_tx = load_audio(os.path.join('tx_signals', f))
        this_frame_length = this_tx.shape[0]
        assert(this_frame_length == frame_length)
        tx_signals += [this_tx]
    n_tx = len(tx_signals)
    bp_ranges = [[20000, 29000], [31000, 40000]]

    coi = list(range(n_channels))
    n_coi = len(coi)
    filtered_audio = np.zeros((n_tx, n_coi, all_audio.shape[0]))
    
    for n, tx in enumerate(tx_signals):
        for i, c in enumerate(coi):
            filtered_audio[n, i] = butter_bandpass_filter(all_audio[:, c], bp_ranges[n][0], bp_ranges[n][1], sampling_rate)
    
    start_pos = 0
    print('Detected start pos: %d' % start_pos)

    filtered_audio = filtered_audio[:, :, start_pos:]
    filtered_audio = filtered_audio[:, :, :filtered_audio.shape[2] - filtered_audio.shape[2] % frame_length]
    filtered_audio = np.reshape(filtered_audio, (filtered_audio.shape[0], filtered_audio.shape[1], -1, frame_length))
    
    profiles = np.zeros((n_tx, n_coi, filtered_audio.shape[2] * filtered_audio.shape[3]))

    if no_overlapp:
        no_overlapp_text = '_no_overlapp'
        for n in range(n_tx):
            for c in range(n_coi):
                f, t, Zxx = signal.stft(filtered_audio[n, c].ravel(), fs=sampling_rate, nperseg=frame_length)
                profiles[n, c, :] = np.abs(Zxx).ravel()
    else:
        no_overlapp_text = ''
        for n in range(n_tx):
            for c in range(n_coi):
                f, t, Zxx = signal.stft(filtered_audio[n, c].ravel(), fs=sampling_rate, nperseg=frame_length, noverlap=frame_length // 2)
                profiles[n, c, :] = np.abs(Zxx).ravel()
    
    profiles = np.reshape(profiles, (-1, profiles.shape[2]))
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
    
    args = parser.parse_args()
    
    echo_profiles(args.audio, args.maxval, args.maxdiffval, args.mindiffval, args.sampling_rate, args.n_channels, args.frame_length, args.no_overlapp, args.no_diff)
