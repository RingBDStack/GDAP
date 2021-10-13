# GDAP
Code for [Generating Disentangled Arguments with Prompts: A Simple Event Extraction Framework that Works](https://arxiv.org/abs/2110.04525)

## Environment

- Python (verified: v3.8)
- CUDA (verified: v11.1)
- Packages (see [requirements.txt](./requirements.txt))

## Usage

### Preprocessing
We follow [dygiepp](https://github.com/dwadden/dygiepp) for data preprocessing.

- `text2et`: Event Type Detection
- `ettext2tri`: Trigger Extraction
- `etrttext2role`: Argument Extraction

```bash
# data processed by dyieapp
data/text2target/dyiepp_ace1005_ettext2tri_subtype
├── event.schema 
├── test.json
├── train.json
└── val.json

# data processed by  data_convert.convert_text_to_target
data/text2target/dyiepp_ace1005_ettext2tri_subtype
├── event.schema
├── test.json
├── train.json
└── val.json
```
Useful commands:

```bash
python -m data_convert.convert_text_to_target # data/raw_data -> data/text2target
python convert_dyiepp_to_sentence.py data/raw_data/dyiepp_ace2005 # doc -> sentence, used in evaluation
```

### Training
Relevant scripts:

- `run_seq2seq.py`: Python code entry, modified from the transformers/examples/seq2seq/run_seq2seq.py
- `run_seq2seq_span.bash`: Model training script logging to the log file.

Example (see the above two files for more details):

```bash
# ace05 event type detection t5-base, the metric_format use eval_trigger-F1 
bash run_seq2seq_span.bash --data=dyiepp_ace2005_text2et_subtype --model=t5-base --format=et --metric_format=eval_trigger-F1

# ace05 tri extraction t5-base
bash run_seq2seq_span.bash --data=dyiepp_ace2005_ettext2tri_subtype --model=t5-base --format=tri --metric_format=eval_trigger-F1

# ace05 argument extraction t5-base
bash run_seq2seq_span.bash --data=dyiepp_ace2005_etrttext2role_subtype --model=t5-base --format=role --metric_format=eval_role-F1

```

Trained models are saved in the `models/` folder.

### Evaluation
- `run_tri_predict.bash`: trigger extraction evaluation and inference script.
- `run_arg_predict.bash`: argument extraction evaluation and inference script.

## Todo
We aim to expand the codebase for a wider range of tasks, including
- [ ] Name Entity Recognition
- [ ] Keyword Generation
- [ ] Event Relation Identification 

## If you find this repo helpful...
Please give us a :star: and cite our paper as
```bibtex
@misc{si2021-GDAP,
      title={Generating Disentangled Arguments with Prompts: A Simple Event Extraction Framework that Works}, 
      author={Jinghui Si and Xutan Peng and Chen Li and Haotian Xu and Jianxin Li},
      year={2021},
      eprint={2110.04525},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

> This project borrows code from [Text2Event](https://github.com/luyaojie/text2event)
