# prework
1, speaker embedding extract

https://github.com/PlayVoice/VI-Speaker

2, ppg extract

https://github.com/PlayVoice/whisper_ppg

# visvc
1, python prepare/preprocess_wave.py

2, python prepare/preprocess.py

3, python train.py -c configs/singing_base.json -m singing_base

4, python visvc_infer.py -s [waves] -p [ppg&hubert] -e [speaker_embedding]

# about
 VI-SVC model is just VITS without MAS and DurationPredictor. Big data [more and more wave] make things to be interesing!
