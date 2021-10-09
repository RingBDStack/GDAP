export device="0"
export et_model_path="XXX"
export role_model_path="XXX"
export et_data_name=dyiepp_ace2005_et_subtype_span
export task_name="event"
export batch=16
export constraint_decoding="--constraint_decoding"

export decoding_format='role'

et_data_folder=data/text2target/${et_data_name}

# et result 转化
python convert_et_result.py \
  --et_pred_file=${et_model_path}/test_preds_seq2seq.txt \
  --et_text_file=${et_data_folder}/test.json \
  --et_output_file=${et_data_folder}/test_pre.json \
  --schema_file=${et_data_folder}/event.schema

# # 依赖上述的转化文件进行 role predict 预测, decode_format = noetrtspan
CUDA_VISIBLE_DEVICES=${device} python run_seq2seq.py \
  --do_predict --task=${task_name} --predict_with_generate \
  --validation_file=${et_data_folder}/val.json \
  --test_file=${et_data_folder}/test_pre.json \
  --event_schema=${et_data_folder}/event.schema \
  --model_name_or_path=${role_model_path} \
  --output_dir="${role_model_path}"_test \
  --source_prefix="span: " \
  ${constraint_decoding} \
  --per_device_eval_batch_size=${batch} \
  --decoding_format ${decoding_format}

# evaluate 评估, 结果打印在控制台
python evaluation.py \
  --text_file=${et_data_folder}/test_pre.json \
  --pred_file="${role_model_path}"_test/test_preds_seq2seq.txt \
  --gold_file="data/raw_data/dyiepp_ace2005/test_convert.json" \
  --schema_file=${et_data_folder}/event.schema \
  --decoding_format ${decoding_format} \
  --format="dyiepp" > "${role_model_path}"_test/total_dyiepp_result.txt

# evaluate 评估, 结果打印在控制台(non-events 评估)
  python evaluation.py \
  --text_file=${et_data_folder}/test_pre.json \
  --pred_file="${role_model_path}"_test/test_preds_seq2seq.txt \
  --gold_file="data/text2target/dyiepp_ace2005_etrttext2role_subtype/test.json" \
  --schema_file=${et_data_folder}/event.schema \
  --decoding_format ${decoding_format} \
  --format="text2target" > "${role_model_path}"_test/total_text2target_result.txt


