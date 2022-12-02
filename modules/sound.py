
import matplotlib.pyplot as plt
from scipy.io import wavfile
#import time
import os

import scipy.signal as sps
from scipy import stats
import numpy as np
import torch
import torchaudio
#import sounddevice as sd
import random


class Stimulus:
    """ Base class for sound stimulus
    """
    def __init__(self, sample_rate_in=10000000, sample_rate_out=48000, sensitivity = 117.53,
        Vat0dBFS = 0.9163, length_in_seconds = 0.5, interstimulus = 0.05,
        dB_SPL = 70, lowcut = 20, highcut = 1400):
        """
        Noise synthesis:
        Arguments:
            stimulus parameters:
                type -- white noise
                total duration -- 0.5s
                onset -- 50*10^-3s squared-cosine
                offset -- 50*10^-3s squared-cosine
                interstimulus -- several possibilities, 0.3s 0.2s 50*10^-3s
                dB_SPL -- several possibilities,  60 70 80
                rand -- randomly first left or first right
                s --- TODO headphones sensitivity
                v -- TODO device output voltage for 0 DBFS @ 1 kHz
        """

        # AUDIO SETTINGS
        self.sample_rate_in = sample_rate_in
        self.sample_rate_out = sample_rate_out
        self.sensitivity = sensitivity
        self.Vat0dBFS = Vat0dBFS
        self.resampler = torchaudio.transforms.Resample(self.sample_rate_in, self.sample_rate_out)

        # SOUND SETTINGS
        self.length_in_seconds = length_in_seconds
        self.length_in_samples = int(self.sample_rate_in * self.length_in_seconds)
        self.length_ramp = int(0.005 * self.sample_rate_in)
        self.ramp = np.linspace(0, self.length_ramp - 1, self.length_ramp)
        self.interstimulus = interstimulus
        self.interstimuli_samples = int(self.sample_rate_in * self.interstimulus)
        self.dB_SPL = dB_SPL
        self.lowcut = lowcut
        self.highcut = highcut

        # NOISE SOURCE TODO CHECK IF -1, 1 OR 0, 1
        self.noise = stats.truncnorm(-1, 1, scale=min(2**16, 1)).rvs(self.length_in_samples)
        # wavfile.write('noise.wav', sample_rate, noise.astype(np.int16))

        # BANDPASS FILTER
        nyq = 0.5 * self.sample_rate_in
        low = self.lowcut / nyq
        high = self.highcut / nyq
        order = 6
        sos = sps.butter(order, [low, high], analog=False, btype='band', output='sos')
        # w, h = sps.sosfreqz(sos, worN=2000) FREQUENCY RESPONSE
        filtered = sps.sosfilt(sos, self.noise)

        # ENVELOPE
        onset = np.square(np.cos(np.pi*(self.ramp/(2 * self.length_ramp) + 0.5)))
        onset = np.pad(onset, (0, self.length_in_samples - 2 * self.length_ramp), 'constant', constant_values=1)
        offset = np.square(np.cos(np.pi*(self.ramp/(2 * self.length_ramp))))
        adsr = np.concatenate((onset, offset))
        self.enveloped = filtered * adsr

        # AMPLITUDE TODO !!!!!!
        dB_FS = dB_SPL - sensitivity + Vat0dBFS
        rms = self.rms(self.enveloped)
        gain =  (10 ** (dB_FS / 20)) / rms
        # print('gain', gain)
        self.stimulus = self.enveloped # * gain

    
    def rms(self, arr):
        return np.sqrt(np.mean(np.square(arr), axis=-1))

    
    def convert_to_decibel(self, arr):
        """Just for checking"""
        rms = self.rms(arr)
        if rms !=0:
            return 20 * np.log10(rms)
        else:
            return -60


    def play(self, itd, i=[0]):
        """ Play the reference and test stimuli with the given ITD difference
        Arguments:
            model -- the model
        """
        i[0] += 1

        # ITD FROM MICROSECONDS TO SAMPLES
        samplesITD = int(itd * 0.5 * self.sample_rate_in * 10 ** (-6))
        # print('ITD in microseconds', 2 * samplesITD * (10 ** 6) / self.sample_rate_in)

        # RANDOM START LEFT OR RIGHT LEADING
        first_reference = random.SystemRandom().randint(0, 1)

        # PANNING (target always leading in the right ear)
        delayed = np.pad(self.stimulus, (samplesITD, 0), 'constant', constant_values=0)
        not_delayed = np.pad(self.stimulus, (0, samplesITD), 'constant', constant_values=0)
        not_delayed_first = np.pad(not_delayed, (0, self.interstimuli_samples), 'constant', constant_values=0)
        not_delayed_first = np.concatenate((not_delayed_first, delayed))
        delayed_first = np.pad(delayed, (0, self.interstimuli_samples), 'constant', constant_values=0)
        delayed_first = np.concatenate((delayed_first, not_delayed))
        audio = np.zeros([delayed_first.shape[0],2], dtype=np.float32)   
        if first_reference:
            # audio = np.vstack((delayed_first, not_delayed_first))
            audio[:, 0] = delayed_first[:] #1 for right speaker, 0 for  left
            audio[:, 1] = not_delayed_first[:]
            print('First RIGHT, second LEFT')
            rightmost = 1

        else:
            # audio = np.vstack((not_delayed_first, delayed_first))
            audio[:, 0] = not_delayed_first[:] #1 for right speaker, 0 for  left
            audio[:, 1] = delayed_first[:] 
            print('First LEFT, second RIGHT')
            rightmost = 2

        # RESAMPLE TO PLAY OUT
        audio_out = self.resampler(torch.Tensor(np.transpose(audio)))
        self.audio_out = np.transpose(audio_out.numpy())
        # print('left dB FS', self.convert_to_decibel(self.audio_out[:, 0]) + self.sensitivity)
        # print('right dB FS', self.convert_to_decibel(self.audio_out[:, 1]) + self.sensitivity)
        # PLAY
        #sd.play(self.audio_out, self.sample_rate_out)
        #sd.wait()
        filename = 'audio/ITD.wav'
        wav = 'static/audio/ITD' + str(i[0]%2) + '.wav'
        #os.remove(wav)
        wavfile.write(wav, self.sample_rate_out, self.audio_out)
        #sd.play(self.audio_out, self.sample_rate_out)
        #sd.wait()
        #playsound(wav)

        # RETURN THE RIGHTMOST STIMULUS
        return rightmost, wav

    
    def plot_specgram(self, title="Spectrogram"):
        waveform = np.transpose(self.audio_out)
        num_channels, _ = waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=self.sample_rate_out)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)
        #plt.show(block=False)