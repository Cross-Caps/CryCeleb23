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

# Some Learnings: What didn't work?

Here's a non-exhaustive list of selected methods we tried, none of which improved the final EER:

- Adding `WavAugument` or `SpecAugment` to the signal during the training phase
- Using trainable `SincConv` on the raw signal as an approximation of learnable filters before passing the output to the ECAPA
- Using `LogSpectrograms` instead of `MelSpectrograms`
- `Inverse-Mel filterbanks` that have been shown to be effective in literature for modelling high-pitched speakers like children.
- Feeding the audio to Large scale Wave2Vec/Transformer models (Like Unispeech-SAT and WavLM) followed by a TDNN head. [Reference](https://huggingface.co/docs/transformers/v4.20.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector)
  - Pretrained Unispeech-SAT models performed better than pretrained ECAPA models. Finetunned models didn't perform well due to a lack of training data - A direction worth exploring.
- Large-margin finetuning


# Best Model
Our best-performing model was an ECAPA-TDNN with the same configuration and pipeline as given in the baseline. We modified the baseline in 2 different ways:

- Performing Test Time Augmentation during evaluation as suggested by [dienhoa](https://github.com/dienhoa).
- We expanded the training data of the model to include the entire train set, as well as the birth recordings in the dev set.
  - The metric for evaluation is EER on trial pairs with one birth and one discharge recording; hence dev EER still retains its usefulness as a metric to judge a model's performance.
  - In fact, we have a lower gap in dev vs test EER (20% vs 25.7%) than the baseline (22% vs 30%).
  - To our surprise, including the entire dev set as training data led to a lower test EER.

# Contact 
Siddhant Rai Viksit <siddhant21565@iiitd.ac.in>

Vinayak Abrol <abrol@iiitd.ac.in>
