# Automatic OIIAI Synthsizer 

## Requirements

```
python=3.7
tensorflow=2.9
librosa=0.10
soundfile=0.13
```

## How to use
Download the repository locally.

Under project directory, run command in terminal `python CatSynthsize.py -i {input_audio_file_path}`

Depending on the tempo and richness of the input lyrics, 1 minute of music takes about 1-6 minutes to process.

## Credits

The automatic OIIAI algorithm utilizes two separate AI models: [Spleeter](https://github.com/deezer/spleeter) for vocal isolation, [VocalExtractor](https://github.com/s603122001/Vocal-Melody-Extraction) for vocal melody extraction.
