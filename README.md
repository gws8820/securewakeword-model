# securewakeword-model
SecureWakeWord is a voice authenticable wakeword system based on [OpenWakeWord](https://github.com/dscripka/openWakeWord).

This repository helps to train/evaluate wakeword and voiceauth models for [wyoming-securewakeword](https://github.com/gws8820/wyoming-securewakeword).

Read our [research paper](https://arxiv.org/abs/2501.12194) for detailed instruction.

## Prepare Data
You should prepare at least 500 actual wakeword-speaking data. Use included voice-recorder to collect these data.

You also need to prepare conversation dataset. For korean wakeword, we recommend [this dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=109).

## Train Wakeword Model
Run `wakeword.ipynb` for train wakeword model. Google Colab Environment recommended.

**Make sure to put collected wakeword-speaking data to ./my_custom_model/YOUR_MODEL_NAME/positive_train after train step 1.**

Custom-trained wakeword model can be found under `./securewakeword-model/model/wakeword/tflite`.

## Train Voiceauth Model
Put positive audio files (user to authenticate) under `./dataset/Raw/`.

You can choose among three different models: Resemblyzer, x-vector, and ECAPA.

<img width="637" alt="image" src="https://github.com/user-attachments/assets/c838c32a-a78c-47ab-a446-1e34ed1797f0" />

Trained voiceauth embedding can be found under `./securewakeword-model/model/voiceauth`.

## Evaluate Model
SecureWakeWord follows three metrics below.

1. FRR (False Rejection Rate) : The proportion of actual positive cases that are incorrectly classified as negative.
2. FAR (False Acceptance Rate) : The proportion of actual negative cases that the model incorrectly classifies as positive.
3. EER (Equal Error Rate) : Error rate at which the FAR and the FRR are equal.

You can run `./evaluate/wakeword.py` and `./evaluate/voiceauth.py` to get each metrics by sweeping threshold from 0 to 1 in increments of 0.05 resolution.


## License
This project uses [openWakeWord](https://github.com/dscripka/openWakeWord) developed by [dscripka](https://github.com/dscripka).

See [LICENSE](LICENSE) for details.
