# Traditional ASR vs. Large Audio-Language Models on Four Indian Languages

Code, training logs, and model outputs for the paper:

> Manmohan Mahapatra and Abhik Jana. 2026. *Towards Robust ASR for Indian Languages: Evaluating Traditional Speech Recognition Architectures and Large Audio-Language Models.* AACL-IJCNLP 2026 Student Research Workshop.

We compare eight open ASR systems on a speaker-disjoint subset of IndicVoices (Assamese, Bengali, Hindi, Odia; about 8 hours of training audio per language; 4,041 test utterances):

| Family | Models |
|---|---|
| Traditional ASR | Whisper-Large-v3, SeamlessM4T-v2-Large, MMS-1B-All, IndicConformer-600M (in-domain reference, not fine-tuned) |
| Audio-language models | Qwen2-Audio-7B-Instruct, Qwen3-ASR-1.7B, Phi-4-Multimodal-Instruct, Voxtral-Mini-3B-2507 |

## Repository layout

```
training/            fine-tuning scripts (LoRA / QLoRA), one per adapted model
inference/           inference + scoring scripts for the fine-tuned models and IndicConformer
logs/<model>/        training and validation logs of the reported runs
results/finetuned/   per-utterance predictions and per-language summaries (fine-tuned + IndicConformer)
results/zero_shot/   per-utterance predictions and summaries of the zero-shot runs
analysis/            scripts that recompute the numbers reported in the paper
data/                test-set file IDs, split statistics, and a helper to export split IDs
```

## Reproducing the reported numbers

The analysis scripts need only `numpy` and `pandas` and run on the files in this repository:

```bash
pip install numpy pandas
cd analysis
python corpus_metrics.py     # per-language corpus WER/CER of the fine-tuned systems (Table 2, FT)
python paired_bootstrap.py   # paired bootstrap tests, 2,000 resamples, seed 42 (appendix)
python training_logs.py      # best validation epoch, training time, peak memory (appendix)
```

Corpus WER is `sum(WER_i * N_i) / sum(N_i)`, where `N_i` is the number of reference words of utterance `i` (CER uses reference characters). The per-utterance scores come from each model's own inference script.

## Data

We use a four-language subset of [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (CC BY 4.0; access requires accepting the dataset conditions on Hugging Face). We do not redistribute audio.

- `data/train_ids.csv`, `data/validation_ids.csv`, and `data/test_ids.csv` list the 15,713 training, 846 validation, and 4,041 test utterances (file name, language, IndicVoices speaker ID, duration in seconds). No speaker appears in more than one split.
- `data/split_summary.csv` gives hours and utterance counts per split and language.
- `data/export_split_ids.py` exports the file IDs of all splits from a local copy of the subset.

To rebuild the subset, download IndicVoices from Hugging Face and select the listed files.

The prediction files in `results/` contain the IndicVoices reference transcripts of the test utterances, shared under CC BY 4.0 with attribution to Javed et al. (2024).

## Notes on the scripts

- Scripts are provided as used for the reported runs. Absolute data paths have been replaced with `/path/to/data`; set them to your local copy before running.
- The training and validation logs are the record of what was run. A few constants in the scripts are stale (for example, `EPOCHS = 3` in the MMS script; its log shows that all five epochs were run).
- Exact package versions were not recorded. See `requirements.txt` for the main dependencies.
- The zero-shot inference scripts are not included; their outputs and summaries are in `results/zero_shot/`.
- Model checkpoints and LoRA adapters are not included.

## Citation

```bibtex
@inproceedings{mahapatra2026robustasr,
  title     = {Towards Robust {ASR} for {I}ndian Languages: Evaluating Traditional Speech Recognition Architectures and Large Audio-Language Models},
  author    = {Mahapatra, Manmohan and Jana, Abhik},
  booktitle = {Proceedings of the AACL-IJCNLP 2026 Student Research Workshop},
  year      = {2026}
}
```

Please also cite IndicVoices:

```bibtex
@inproceedings{javed2024indicvoices,
  title     = {{IndicVoices}: Towards Building an Inclusive Multilingual Speech Dataset for {I}ndian Languages},
  author    = {Javed, Tahir and Nawale, Janki and George, Eldho and Joshi, Sakshi and Bhogale, Kaushal and Mehendale, Deovrat and Sethi, Ishvinder and Ananthanarayanan, Aparna and Faquih, Hafsah and Palit, Pratiti and Ravishankar, Sneha and Sukumaran, Saranya and Panchagnula, Tripura and Murali, Sunjay and Gandhi, Kunal and R, Ambujavalli and M, Manickam and Vaijayanthi, C and Karunganni, Krishnan and Kumar, Pratyush and Khapra, Mitesh},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  year      = {2024}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Transcripts in `results/` come from IndicVoices and remain under CC BY 4.0.
