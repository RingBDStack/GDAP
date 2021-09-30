# GDAP
The code of paper "Code for "Generating Disentangled Arguments with Prompts: a Simple Event Extraction Framework that Works""

## Event Datasets Preprocessing
We use code and environments [[dygiepp](https://github.com/dwadden/dygiepp)] for data preprocessing.
Thanks to them！

## Quick Start

### Data Format

- `text2et`: event type detection
- `ettext2tri`: trigger extraction
- `etrttext2role`: argument extraction

```text
data/text2target/dyiepp_ace1005_ettext2tri_subtype
├── event.schema
├── test.json
├── train.json
└── val.json
```

```bash
python -m data_convert.convert_text_to_target # data/raw_data -> data/text2target
python convert_dyiepp_to_sentence.py data/raw_data/dyiepp_ace2005 # doc -> sentence 
```



# please wait a minute

# code main reference
- Text2Event: https://github.com/luyaojie/text2event