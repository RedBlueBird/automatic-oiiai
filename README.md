# Automatic OIIAI Synthsizer 

Generate OIIAI Cat remix from any input music containing vocal using AI.

## Requirements

```
python=3.7
tensorflow=2.9
spleeter=2.4
librosa=0.10
soundfile=0.13
```

## How to use
Download the repository locally.

Under project directory, run command in terminal `python CatSynthsize.py -i {input_audio_file_path}`. If input path empty, it is defaulted to `train01.wav`.

Depending on the tempo and richness of the input music, 1 minute takes about 1-6 minutes to process.

The output will show up in the same directory with the name `synthesized-output.wav`.

## Credits

The automatic OIIAI algorithm utilizes two separate AI models: [Spleeter](https://github.com/deezer/spleeter) for vocal isolation, [Vocal-Melody-Extraction](https://github.com/s603122001/Vocal-Melody-Extraction) for vocal melody extraction.
