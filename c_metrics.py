from multiprocessing import Pool
from argparse import ArgumentParser

from pesq import pesq

import os
import tablib
from pystoi import stoi

from tqdm import tqdm
import numpy as np
import soundfile as sf

# Ported 
def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)

    ref_energy = np.sum(ref_sig ** 2) + eps    

    S_target = np.sum(ref_sig * out_sig) * ref_sig / ref_energy

    e_noise = out_sig - S_target  
    ratio = np.sum(S_target ** 2) / (np.sum(e_noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def evaluate_file(args):
    clean_path, enhance_path, fs = args
    clean_speech, _ = sf.read(clean_path)
    enhanced_speech, _ = sf.read(enhance_path)
    lengths = min(len(clean_speech), len(enhanced_speech))
    clean_speech = clean_speech[:lengths]
    enhanced_speech = enhanced_speech[:lengths]
    try:
        pesq_score = pesq(fs, clean_speech, enhanced_speech)
    except Exception as e:
        print(f'Error calculating PESQ for {clean_path}: {str(e)}')
        pesq_score = 0
    stoi_score = stoi(clean_speech, enhanced_speech, fs, extended=True)
    si_SNR = cal_SISNR(clean_speech, enhanced_speech)

    return (clean_path, pesq_score, stoi_score, si_SNR)

def evaluation(clean_dir, enhance_dir, excel_path, sample_rate):

    headers = ("audio_names", "PESQ", "ESTOI", "SISNR")
    metrics_seq = []

    names = os.listdir(enhance_dir)

    paths = [(os.path.join(clean_dir, na), os.path.join(enhance_dir, na), sample_rate) for na in names]

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_file, paths), total=len(paths), desc="Processing files"))

    for result in results:
        if result:
            metrics_seq.append((os.path.basename(result[0]), *result[1:]))
    
    if metrics_seq:
        metrics_seq = sorted(metrics_seq, key=lambda x: x[0])
        sum_pesq, sum_stoi, sum_sisnr, sum_csig, sum_cbak, sum_covl = 0, 0, 0, 0, 0, 0,
        count = len(metrics_seq)


        for _, pesq, stoi, sisnr in metrics_seq:
            sum_pesq += pesq
            sum_stoi += stoi
            sum_sisnr += sisnr

        mean_pesq = sum_pesq / count
        mean_stoi = sum_stoi / count
        mean_sisnr = sum_sisnr / count

        metrics_seq.append(('MEAN', mean_pesq, mean_stoi, mean_sisnr))
    
    data = tablib.Dataset(*metrics_seq, headers=headers)
    with open(csvpath, "w", newline='', encoding='utf-8') as f:
        f.write(data.export("csv"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--csvname", type=str, required=True, help='saved csv name')
    args = parser.parse_args()
    SAMPLE_RATE = 16000

    NOISY_DATA_PATH = args.enhanced_dir
    TARG_DATA_PATH  = args.clean_dir

    csvpath = args.csvname
    evaluation(TARG_DATA_PATH, NOISY_DATA_PATH, csvpath, SAMPLE_RATE)