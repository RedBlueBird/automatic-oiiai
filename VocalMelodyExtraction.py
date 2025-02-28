import argparse
import os

import numpy as np
import librosa
import soundfile

import warnings
warnings.filterwarnings("ignore")

from project.MelodyExt import feature_extraction
from project.utils import load_model, save_model, matrix_parser
from project.test import inference
from project.model import seg, seg_pnn, sparse_loss
from project.train import train_audio

# Postprocessing parameters
tolerance = 0.05 # Separate pitch contours into different notes based on pitch percentage delta
sr = 44100 # Sampling Rate
k_round = 1.5 # Pitch shift strength, 1 is strongest, infinity is weakest

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase',
                        help='phase: training or testing (default: %(default)s',
                        type=str, default='testing')

    #arguments for training
    parser.add_argument('-t', '--model_type',
                        help='model type: seg or pnn (default: %(default)s',
                        type=str, default='seg')
    parser.add_argument('-d', '--data_type',
                        help='data type: audio or symbolic (default: %(default)s',
                        type=str, default='audio')
    parser.add_argument('-da', '--dataset_path', nargs='+',
                        help='path to data set (default: %(default)s',
                        type=str, default='dataset')
    parser.add_argument('-la', '--label_path', nargs='+',
                        help='path to data set label (default: %(default)s',
                        type=str, default='dataset_label')
    parser.add_argument('-ms', '--model_path_symbolic',
                        help='path to symbolic model (default: %(default)s',
                        type=str, default='model_symbolic')

    parser.add_argument('-w', '--window_width',
                        help='width of the input feature (default: %(default)s',
                        type=int, default=128)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size during training (default: %(default)s',
                        type=int, default=12)
    parser.add_argument('-e', '--epoch',
                        help='number of epoch (default: %(default)s',
                        type=int, default=5)
    parser.add_argument('-n', '--steps',
                        help='number of step per epoch (default: %(default)s',
                        type=int, default=6000)

    parser.add_argument('-o', '--output_model_name',
                        help='name of the output model (default: %(default)s',
                        type=str, default="out")

    #arguments for testing
    parser.add_argument('-m', '--model_path',
                        help = 'path to existing model (default: %(default)s',
                        type = str, default = 'pretrained_models/Seg')
    parser.add_argument('-i', '--input_file',
                        help='path to input file (default: %(default)s',
                        type=str, default='train01.wav')
    parser.add_argument('-bb', '--batch_size_test',
                        help='batch size during testing (default: %(default)s',
                        type=int, default=10)

    args = parser.parse_args()
    # print(args)

    if(args.phase == "training"):
        #arguments setting
        TIMESTEPS = args.window_width

        #dataset_path = ["medleydb_48bin_all_4features", "mir1k_48bin_all_4features"]
        #label_path = ["medleydb_48bin_all_4features_label", "mir1k_48bin_all_4features_label"]
        dataset_path = args.dataset_path
        label_path = args.label_path


        # load or create model
        if("seg" in args.model_type):
            model = seg(multi_grid_layer_n=1, feature_num=384, input_channel=1, timesteps=TIMESTEPS)
        elif("pnn" in args.model_type):
            model = seg_pnn(multi_grid_layer_n=1, feature_num=384, timesteps=TIMESTEPS,
                            prev_model=args.model_path_symbolic)

        model.compile(optimizer="adam", loss={'prediction': sparse_loss}, metrics=['accuracy'])

        #train
        train_audio(model,
                    args.epoch,
                    args.steps,
                    args.batch_size_train,
                    args.window_width,
                    dataset_path,
                    label_path)

        #save model
        save_model(model, args.output_model_name)
    else:
        # load wav
        og_song = args.input_file

        # Possible Security Risk Implementation
        print("========================")
        print("Separating audio into vocal and instrumental...")
        os.system(f"spleeter separate -o output {og_song}")
        print("========================")
        og_song_name = ''.join(og_song.split('.')[:-1])
        song = f"output/{og_song_name}/vocals.wav"

        # Feature extraction
        feature = feature_extraction(song)
        feature = np.transpose(feature[0:4], axes=(2, 1, 0))

        # load model
        model = load_model(args.model_path)

        # Inference
        print(feature[:, :, 0].shape)
        extract_result = inference(feature= feature[:, :, 0],
                                   model = model,
                                   batch_size=args.batch_size_test)

        # Get VocalMelody AI Output
        # This AI algorithm uses mir_eval, 2nd column output is in Cent
        # Reverse engineered the formula Hertz to Cent from
        # https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/melody.py#L138
        print("========================")
        print("Extracting Melody Frequency Values...")
        r = matrix_parser(extract_result)
        np.savetxt("out_seg.txt", r)

        fin = open("out_seg.txt", 'r')
        text = [str(i) for i in fin.read().splitlines()]
        n = len(text)
        freq = []
        for i in range(n):
            cent = float(text[i].split(" ")[1])
            if cent != 0:
                freq.append(2 ** (cent / 1200) * 10)
            else:
                freq.append(0)
        pitch_values = freq

        # Postprocessing
        curr_len = 0
        filtered_freq = []
        pitch_values.append(0)
        for i in pitch_values:
            filtered_freq.append(i)
            if i == 0 and curr_len != 0:
                if curr_len < 9:
                    # seg.append("x")
                    for j in range(curr_len):
                        filtered_freq[-(j + 2)] = 0
                curr_len = 0
            elif i != 0:
                curr_len += 1
        pitch_values = filtered_freq
        filtered_freq = []
        curr_len = 0
        for i in pitch_values:
            filtered_freq.append(i)
            if i != 0 and curr_len != 0:
                if curr_len < 9:
                    for j in range(curr_len):
                        filtered_freq[-(j + 2)] = filtered_freq[-(curr_len + 2)]
                curr_len = 0
            elif i == 0:
                curr_len += 1
        pitch_values = filtered_freq
        pitch_values.pop(-1)

        seg = []
        curr_len = 0
        mode = -1
        pitch_values.insert(0, 0)
        for index, i in enumerate(pitch_values):
            if index == 0:
                continue
            pitch_ratio = pitch_values[index] / pitch_values[index - 1] if pitch_values[index - 1] != 0 else 100
            pitch_ratio = 1 if pitch_values[index - 1] == 0 and pitch_values[index] == 0 else pitch_ratio

            curr_mode = 1 if i != 0 else 0
            if curr_mode == mode and 1.00 - tolerance < pitch_ratio < 1.00 + tolerance:
                seg[-1].append(i)
                curr_len += 1
            else:
                mode = curr_mode
                seg.append([i])
                curr_len = 1
        pitch_values.pop(0)

        segs = []
        new_seg = False
        for i in seg:
            if i[0] == 0:
                new_seg = False
                segs.append([i])
                continue
            if not new_seg:
                segs.append([])
                new_seg = True
            segs[-1].append(i)
        seg = segs

        # Output brainrot music
        print("========================")
        print("Synthesizing brainrot...")
        skeleton = np.array([])
        u_note, _ = librosa.load("board/u.wav", sr=sr)
        i_note, _ = librosa.load("board/i.wav", sr=sr)
        a_note, _ = librosa.load("board/a.wav", sr=sr)
        notes = [u_note, i_note, a_note, i_note]

        for i in range(len(seg)):
            if seg[i][0][0] == 0:
                skeleton = np.append(skeleton, [0] * 882 * len(seg[i][0]))
            else:
                curr = 0
                for note in range(len(seg[i])):
                    curr_note = notes[curr % len(notes)]
                    divider = 10
                    if curr % 4 == 0:
                        divider = 30
                    if curr % 4 == 2:
                        divider = 1000
                    pieces = int(np.floor(len(seg[i][note]) / divider))
                    pieces = 1 if pieces == 0 else pieces
                    curr_seg = []
                    for j in range(pieces):
                        curr_seg.append(seg[i][note][round(len(seg[i][note]) / pieces * j):round(
                            len(seg[i][note]) / pieces * (j + 1))])
                    for note_seg in curr_seg:
                        note_seg_len = len(note_seg)
                        if curr % 4 == 1 or curr % 4 == 3:
                            note_seg_len = max(min(6, note_seg_len), round(note_seg_len * 2 / 3))
                        if curr % 4 == 0 or curr % 4 == 2:
                            note_seg_len = max(min(9, note_seg_len), round(note_seg_len * 2 / 3))
                        stretched_note = librosa.effects.time_stretch(curr_note,
                                                                      rate=(len(curr_note) / 882) / note_seg_len)
                        bin_len = round(len(stretched_note) / note_seg_len)
                        for j in range(note_seg_len):
                            stretched_note[bin_len * j:bin_len * (j + 1)] = librosa.effects.pitch_shift(
                                stretched_note[bin_len * j:bin_len * (j + 1)], sr=sr,
                                n_steps=12 * np.log2(np.power(note_seg[j] / 256, 1 / k_round)))
                        if note_seg_len < len(note_seg):
                            stretched_note = np.append(stretched_note, [0] * 882 * (len(note_seg) - note_seg_len))
                        skeleton = np.append(skeleton, stretched_note)
                    curr += 1

        song_instrumental, _ = librosa.load(f"output/{og_song_name}/accompaniment.wav", sr=sr)
        song_instrumental *= 0.25
        skeleton *= 3
        for i in range(len(song_instrumental)):
            if i < len(skeleton):
                skeleton[i] += song_instrumental[i]
            else:
                skeleton = np.append(skeleton, song_instrumental[i])

        print("========================")
        print("Done!")
        soundfile.write("synthesized-output.wav", skeleton, sr)

if __name__ == '__main__':
    main()
