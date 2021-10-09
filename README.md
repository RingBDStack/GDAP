# GDAP
The code of paper "Generating Disentangled Arguments with Prompts: a Simple Event Extraction Framework that Works"

## Quick links
* [Event Datasets Preprocessing](#Event-Datasets-Preprocessing)
* [Model Training](#Model-Training)
* [Model Evaluation and Inference](#Model-Evaluation-and-Inference)
* [Expand to Other Tasks](#Expand-to-Other-Tasks)

## Event Datasets Preprocessing
We use code and environments [[dygiepp](https://github.com/dwadden/dygiepp)] for data preprocessing.

### Environments

- Python (verified on 3.8)
- CUDA (verified on 11.1)
- Python Packages (seen in requirements.txt)

### Data Format

- `text2et`: event type detection
- `ettext2tri`: trigger extraction
- `etrttext2role`: argument extraction

```bash
# the data after dyieapp deal
data/text2target/dyiepp_ace1005_ettext2tri_subtype
├── event.schema 
├── test.json
├── train.json
└── val.json

# the data after data_convert.convert_text_to_target deal
data/text2target/dyiepp_ace1005_ettext2tri_subtype
├── event.schema
├── test.json
├── train.json
└── val.json
```
The commands may be used:

```bash
python -m data_convert.convert_text_to_target # data/raw_data -> data/text2target
python convert_dyiepp_to_sentence.py data/raw_data/dyiepp_ace2005 # doc -> sentence, used in evaluation
```

## Model Training
Training scripts as follows:

- `run_seq2seq.py`: Python code entry, modified from the transformers/examples/seq2seq/run_seq2seq.py
- `run_seq2seq_span.bash`: Model training script logging to the log file.

The command for the training is as follows (see bash scripts and Python files for the corresponding command-line
arguments):

```bash
# ace05 event type detection t5-base, the metric_format use eval_trigger-F1 
bash run_seq2seq_span.bash --data=dyiepp_ace2005_text2et_subtype --model=t5-base --format=et --metric_format=eval_trigger-F1

# ace05 tri extraction t5-base
bash run_seq2seq_span.bash --data=dyiepp_ace2005_ettext2tri_subtype --model=t5-base --format=tri --metric_format=eval_trigger-F1

# ace05 argument extraction t5-base
bash run_seq2seq_span.bash --data=dyiepp_ace2005_etrttext2role_subtype --model=t5-base --format=role --metric_format=eval_role-F1

```
Format:
- `et`: event type detection
- `tri`: trigger extraction
- `role`: argument extraction

Trained models are saved in the `models/` folder.

## Model Evaluation and Inference
- `run_tri_predict.bash`: trigger extraction evaluation and inference script.
- `run_arg_predict.bash`: argument extraction evaluation and inference script.

## Expand to Other Tasks
please wait a minute

# code main reference
- Thanks to Dygiepp: https://github.com/dwadden/dygiepp
- Thanks to Text2Event: https://github.com/luyaojie/text2event