import numpy as np
from numpy.fft  import fft, ifft
from scipy.io import wavfile

class shift:
    """
    A class to implement the TD-PSOLA algorithm, as desribed in this video:

    http://speech.zone/courses/speech-processing/module-5-speech-synthesis-waveform-generation/
    videos/td-psola/

    """


    def __init__(self, input_wav = "input.wav", output_wav = "output.wav", max_hz = 300, min_hz = 80,\
                 f_ratio = 0.8, Es = 100000):
        """
        The init function reads in a WAV file and outputs a pitch-shifted WAV file using the shift
        class function.

        :param input_wav: input WAV filename
        :param output_wav: output WAV filename
        :param max_hz: maximum frequency detectable
        :param min_hz: minimum frequency detectable
        :param f_ratio: ratio of spacing in original signal to spacing in synthesised signal
        :param Es: threshold value for determining whether a section in the signal is voiced
        :classattribute sample_rate: integer value for the sample rate of original signal
        :classattribute signal: numpy array for original signal
        :classattribute min_period: minimum period which determines the lower bound on the peak delay
        :classattribute max_period: maximum period which determines the upper bound on the peak delay
        """
        self.Es = Es
        self.f_ratio = f_ratio
        self.sample_rate, self.signal = wavfile.read(input_wav) #signal is read in as numpy array
        self.analysis_frame = int(0.04 * self.sample_rate)
        self.min_period = self.sample_rate // max_hz
        self.max_period = self.sample_rate // min_hz
        self.main(self.signal, self.analysis_frame)
        wavfile.write(output_wav, self.sample_rate, self.new_signal.astype(np.int16))

    def main(self, signal, analysis_frame):
        """
        Function to create a shifted version of the original signal

        :param signal: numpy array of original signal
        :param analysis_frame: integer value for the length of the analysis windo
        :classattribute new_signal: numpy array for new synthesised signal
        """
        num_frames = len(signal) // analysis_frame
        peak_delays = []
        voiced = []

        for i in range(num_frames):

            if (i * analysis_frame) > (len(signal) - analysis_frame):
                break

            peak_delay = self.find_delays(i, signal, analysis_frame)
            peak_delays.append(peak_delay)

            voice_value = self.detect_voicing(i, signal, analysis_frame)
            voiced.append(voice_value)


        self.new_signal = np.zeros(len(signal))


        self.find_peaks_psola(analysis_frame, signal, peak_delays, voiced, num_frames)


        return None

    def find_delays(self, i, signal, analysis_frame):
        """
        Function to find peak delays using an autocorrelation function, followed by finding
        the points where the second derivative is negative.

        :param i: integer value representing the current frame
        :param signal: numpy array of original signal
        :param analysis_frame: integer vlaue for the length of the analysis window
        :classattribute min_period: minimum period which determines the lower bound on the peak delay
        :classattribute max_period: maximum period which determines the upper bound on the peak delay
        :return peak_delay: highest autocorrelation value between max and min period
        """

        signal_frame = signal[i * analysis_frame:(i+1) * analysis_frame ]
        frame_fft = fft(signal_frame)
        autoc = ifft(frame_fft * np.conj(frame_fft)).real
        inflection = enumerate(np.diff(np.sign(np.diff(autoc))))
        autoc_peaks = [idx for idx,dif in inflection if dif < 0 and\
                       self.min_period < idx < self.max_period]

        try:
            peak_delay = autoc_peaks[np.argmax(autoc[autoc_peaks])]
        except:
            peak_delay = 0

        return peak_delay

    def detect_voicing(self, i, signal, analysis_frame):
        """
        Function to detect voicing in a segement of the signal within the analysis frame

        :param i: integer value representing the current frame
        :param signal: numpy array of original signal
        :param analysis_frame: integer value for the length of the analysis window
        :classattribute Es: threshold value for determining whether or not a section is voiced
        :return voice_value: either a 1 for voiced or 0 for unvoiced
        """
        signal_frame = signal[i * analysis_frame:(i+1) * analysis_frame ]
        sum_abs_val = np.sum(np.abs(signal_frame))

        if sum_abs_val > self.Es:
            voice_value = 1
        else:
            voice_value = 0

        return voice_value

    def find_peaks_psola(self, analysis_frame, signal, peak_delays, voiced, num_frames):
        """
        Function which takes the peaks from the autocorrelation function and estimates the peaks
        in the original signal. These peaks are then fed into another function which carries out
        the PSOLA algorithm and adds the result to the new, synthesised signal.

        :param analysis_frame: integer value for the length of the analysis window
        :param signal: numpy array of original signal
        :param peak_delays: list of peak autocorrelation values for their respective analysis frame
        :param voiced: list determining whether a frame is voiced or not. 1 = voiced, 0 = unvoiced
        :param num_frames: number of analysis frames for the original signal
        :classattribute new_signal: numpy array for new synthesised signal
        """

        prev_peaks = [0]

        for i in range(num_frames):
            frame_floor = i * analysis_frame
            frame_ceiling = (i+1) * analysis_frame
            signal_frame = signal[frame_floor:frame_ceiling]

            if voiced[i] == 0:
                self.new_signal[frame_floor:frame_ceiling] += signal_frame
            else:
                peaks = []
                peaks_frame = []
                first_peak = np.argmax(signal_frame[:peak_delays[i]])
                peaks_frame.append(first_peak)
                peaks.append(first_peak + i * analysis_frame)

                while peaks[-1] + 1.1 * peak_delays[i] < len(signal_frame) + frame_floor:
                    peak_dist = int(int(peak_delays[i]) * 0.9 +\
                        np.argmax(signal_frame[int(peaks_frame[-1] + 0.9 * peak_delays[i]):\
                                                        int(peaks_frame[-1] + 1.1 * peak_delays[i])]))
                    peak_frame = peaks_frame[-1] + peak_dist
                    peaks_frame.append(peak_frame)
                    peak = peaks[-1] + peak_dist
                    peaks.append(peak)

                self.voiced_psola(peaks, peaks_frame, prev_peaks[-1], signal_frame, \
                                  self.f_ratio, frame_floor)

                prev_peaks.append(peaks[-1])

        return None

    def voiced_psola(self, peaks, peaks_frame, prev_peak, signal_frame, f_ratio, frame_floor):
        """
        Function which carries out the PSOLA algorithm on a voiced section of the original signal.
        An impulse window is created around the peaks in the original signal which is then added
        to the new signal, spaced out at points determined by the f_ratio.

        :param peaks: list of peak positions in the original signal
        :param peaks_fram: list of peak positions relative to the start of the analysis frame
        :param prev_peak: position of the last peak before the start of the new analysis frame
        :param signal_frame: window of original signal determined by the length of the analysis frame
        :param f_ratio: ratio of spacing in original signal to spacing in synthesised signal
        :param frame_floor: integer value representing the starting position of the analysis frame
        :classattribute signal: numpy array of original signal
        :classattribute new_signal: numpy array of synthesised signal
        """
        temp_diff = 0

        for i in range(0, len(peaks) - 1):
            temp_diff += abs(peaks[i+1] - peaks[i])

        diff = int(temp_diff/(len(peaks)-1))
        syn_diff = int(f_ratio * diff)
        new_peaks_ref = np.arange(max(0,(prev_peak + syn_diff)-frame_floor), len(signal_frame), syn_diff)
        #creates a numpy array of peak reference positions for the new signal, spaced at points
        #determined by the previous peak and f_ratio

        for i in range(len(new_peaks_ref)):

            peak_id = self.find_nearest_peak_id(peaks_frame, new_peaks_ref[i])
            impulse = self.signal[peaks[peak_id] - diff: peaks[peak_id] + diff]
            hanning = np.hanning(len(impulse))
            window = np.multiply(hanning, impulse)
            new_peak = frame_floor + new_peaks_ref[i]
            self.new_signal[new_peak - diff: new_peak + diff] += window


        return None

    def find_nearest_peak_id(self, peaks_frame, new_peak):
        """
        Function to determine which peak in the original signal is closest to the new peak reference
        position

        :param peaks_frame: list of peak positions in original signal relative to the analysis frame
        :param new_peak: numpy array of reference positions for peaks in the new signal
        :classattribute analysis_frame: integer value for length of analysis window
        :return peak_id: index position for closest peak
        """

        smallest_diff = self.analysis_frame
        for i in range(len(peaks_frame)):
            diff = abs(peaks_frame[i] - new_peak)
            if diff < smallest_diff:
                smallest_diff = diff
                peak_id = i
            else:
                continue
        return peak_id

