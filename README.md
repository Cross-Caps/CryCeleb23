<h1 align="center">
<p>CryCeleb23 :baby:</p>
<p align="center">
<img alt="GitHub" src="https://img.shields.io/github/license/cross-caps/AFLI?color=green&logo=GNU&logoColor=green">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>
<h2 align="center">
<p>Cross-Caps Lab's Wining System Submission for CryCeleb23 Challenge</p>
</h2>


Code to reproduce the results on the private test set and our learning takeaways after participating in the challenge. Place the dataset in a folder named 'data' inside the repository.
# Approach

Here's a list of everything we tried, none of which improved the final EER:

1. Adding `TimeDomainSpecAugment` to the raw signal during the training phase
2. Using `SincConv` on the raw signal as an approximation of learnable filters before passing the output to the ECAPA
3. Using STFTs instead of MelSpectrograms
4. Inverted mel spectrograms, i.e flipping the STFT along the frequency axis before applying the Fbanks to it.
5. Feeding the audio to Large scale transformer models(Like Unispeech SAT and WavLM) followed by a TDNN head.[Reference](https://huggingface.co/docs/transformers/v4.20.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector)
6. Large-margin finetuning

In the end, our best performing model was an ECAPA-TDNN with the same configuration and pipeline as given in the baseline. We modified the baseline in 2 different ways:

1. Performing Test Time Augmentation during evaluation as suggested by [dienhoa](https://github.com/dienhoa).
2. We expanded the training data of the model to include the entire train set, as well as the birth recordings in the dev set. Since our metric for evaluation is EER on those pairs with one birth and one discharge recording, dev EER still retains its usefulness as a metric to judge a model's performance. In fact, we have a lower gap in dev vs test EER (20% vs 25.7%) than the baseline (22% vs 30%). We also tried including the entire dev set in the training data and not performing validation, but that led to a lower test EER.
