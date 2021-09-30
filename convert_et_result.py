# -*- coding:utf-8 -*-
import codecs
import argparse
import json

from extraction.event_schema import EventSchema
from extraction.predict_parser.target_predict_parser import ETPredictParser
from data_convert.format.text2target import type_start, type_end


parser = argparse.ArgumentParser(description='Convert et result')

parser.add_argument('--et_pred_file', type=str)
parser.add_argument('--et_text_file', type=str) # 与preds 文件对应的text文件名称
parser.add_argument('--et_output_file', type=str)
parser.add_argument('--schema_file', type=str)
parser.add_argument('--mode', type=str, default="role")
args = parser.parse_args()


def read_file(file_name):
    return [line.strip() for line in open(file_name).readlines()]


def et_text2role(schema, et_list, text):
    et_rt_dict = schema.type_role_dict
    source_list = []
    target_list = []
    # 遍历event
    for event_type in et_list:
        for role_type in et_rt_dict[event_type]:
            source_text = event_type + " </s> " + role_type + " </s> " + text
            target_text = ""

            source_list.append(source_text)
            target_list.append(target_text)

    # 在所有的target 上统一加上起始位置
    for i in range(len(target_list)):
        target_list[i] = f'{type_start} ' + target_list[i] + f' {type_end}'
    
    return source_list, target_list

def et_text2tri(et_list, text):
    source_list = []
    target_list = []
    # 遍历event
    for event_type in et_list:
        source_text = event_type + " </s> " + text
        target_text = ""

        source_list.append(source_text)
        target_list.append(target_text)

    # 在所有的target 上统一加上起始位置
    for i in range(len(target_list)):
        target_list[i] = f'{type_start} ' + target_list[i] + f' {type_end}'
    
    return source_list, target_list


if __name__ == "__main__":
    
    label_schema = EventSchema.read_from_file(
        filename=args.schema_file
    )

    # 采用解析评估函数对结果文件进行解析
    pred_reader = ETPredictParser(schema=label_schema)     
    event_list, _ = pred_reader.decode(
        gold_list=[],
        pred_list=read_file(args.et_pred_file),
        text_list=[json.loads(line)['text']
                       for line in read_file(args.et_text_file)]
    )
    
    # 输出文件
    event_output = codecs.open(args.et_output_file, 'w', 'UTF-8')

    for item in event_list:
        text = item["text"]
        event_list = item["pred_event"]

        if args.mode == "role":
            source_list, target_list = et_text2role(schema=label_schema, et_list=event_list, text=text)
        else: # trigger
            source_list, target_list = et_text2tri(et_list=event_list, text=text)

        # 将处理后的信息写入文件
        assert len(source_list) == len(target_list)
        for i in range(len(source_list)):
            event_output.write(json.dumps(
                {'text': source_list[i], 'event': target_list[i]}, ensure_ascii=False) + '\n')  
    







