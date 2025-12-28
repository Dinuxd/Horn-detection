import time
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite


TFLITE_PATH = "./Models/horn_model.tflite"
NORM_PATH   = "./Models/norm_stats.npz"


TARGET_SR  = 44100
WINDOW_SEC = 1.0
STEP_SEC   = 0.25     
BLOCK_SEC  = 0.05     
THRESH     = 0.70

N_SAMPLES  = int(TARGET_SR * WINDOW_SEC)

N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 512


stats = np.load(NORM_PATH)
MEAN = float(stats["mean"])
STD  = float(stats["std"])
print("? Loaded norm_stats:", NORM_PATH)


interpreter = tflite.Interpreter(model_path=TFLITE_PATH, num_threads=2)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print("? Loaded tflite:", TFLITE_PATH, "input:", in_det["shape"])


def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i+1], bins[i+2]
        if center == left: center += 1
        if right == center: right += 1
        if right <= left: continue

        fb[i, left:center]  = (fft_freqs[left:center] - hz[i]) / (hz[i+1] - hz[i] + 1e-9)
        fb[i, center:right] = (hz[i+2] - fft_freqs[center:right]) / (hz[i+2] - hz[i+1] + 1e-9)
    return fb

MEL_FB = mel_filterbank(TARGET_SR, N_FFT, N_MELS)
WINDOW = np.hanning(N_FFT).astype(np.float32)

def logmel_1s(y):
    
    pad = N_FFT // 2
    y = np.pad(y, (pad, pad), mode="reflect").astype(np.float32)
    n_frames = 1 + (len(y) - N_FFT) // HOP_LENGTH

    power = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = y[start:start+N_FFT] * WINDOW
        spec = np.fft.rfft(frame, n=N_FFT)
        power[i] = (spec.real**2 + spec.imag**2).astype(np.float32)

    mel = (MEL_FB @ power.T).astype(np.float32)
    mel = np.maximum(mel, 1e-10)

    
    log_mel = 10.0 * np.log10(mel)
    log_mel -= 10.0 * np.log10(np.max(mel) + 1e-10)
    return log_mel  

def preprocess(y):
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[-N_SAMPLES:]

    lm = logmel_1s(y)
    lm = (lm - MEAN) / (STD + 1e-9)
    X = lm[np.newaxis, ..., np.newaxis].astype(np.float32)
    return X


ring = np.zeros(N_SAMPLES, dtype=np.float32)
write_pos = 0
filled = 0

def add_samples(x):
    global write_pos, filled
    x = x.astype(np.float32).flatten()
    n = len(x)
    if n >= N_SAMPLES:
        ring[:] = x[-N_SAMPLES:]
        write_pos = 0
        filled = N_SAMPLES
        return

    end = write_pos + n
    if end <= N_SAMPLES:
        ring[write_pos:end] = x
    else:
        part = N_SAMPLES - write_pos
        ring[write_pos:] = x[:part]
        ring[:end - N_SAMPLES] = x[part:]

    write_pos = (write_pos + n) % N_SAMPLES
    filled = min(N_SAMPLES, filled + n)

def get_last():
    if filled < N_SAMPLES:
        return None
    if write_pos == 0:
        return ring.copy()
    return np.concatenate([ring[write_pos:], ring[:write_pos]]).copy()

def callback(indata, frames, time_info, status):
    if status:
        pass
    add_samples(indata[:, 0])

def predict_prob(y_1s):
    X = preprocess(y_1s)
    interpreter.set_tensor(in_det["index"], X)
    interpreter.invoke()
    return float(interpreter.get_tensor(out_det["index"])[0, 0])

def main():
    print("?? Running continuous horn detection (Ctrl+C to stop)")
    next_infer = time.time() + WINDOW_SEC
    blocksize = int(TARGET_SR * BLOCK_SEC)

    with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                        blocksize=blocksize, callback=callback):
        while True:
            now = time.time()
            if now >= next_infer:
                next_infer = now + STEP_SEC
                y = get_last()
                if y is None:
                    continue

                rms = float(np.sqrt(np.mean(y*y)))
                if rms < 0.005:
                    print("?? silence rms={:.4f}".format(rms))
                    continue

                p = predict_prob(y)
                if p >= THRESH:
                    print(f"???? HORN DETECTED! prob={p:.2f} rms={rms:.4f}")
                else:
                    print(f"no horn prob={p:.2f} rms={rms:.4f}")

            time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n?? Stopped.")
