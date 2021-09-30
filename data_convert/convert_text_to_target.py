#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
from collections import Counter, defaultdict
from data_convert.format.text2target import ETRTText2Role, Text2ET, ETText2Tri
from data_convert.task_format.event_extraction import Event, DyIEPP
from data_convert.utils import read_file, check_output, data_counter_to_table, get_schema, output_schema
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english') + ["'s", "'re", "%"])

def convert_file_tuple(file_tuple, data_class=Event, target_class=ETRTText2Role,
                       output_folder='data/text2target/framenet',
                       ignore_nonevent=False, zh=False,
                       mark_tree=False, type_format='subtype'):
    counter = defaultdict(Counter)
    data_counter = defaultdict(Counter)

    event_schema_set = set()

    span_output_folder = output_folder

    if not os.path.exists(span_output_folder):
        os.makedirs(span_output_folder)

    for in_filename, output_filename in file_tuple(output_folder):
        span_event_output = open(output_filename + '.json', 'w')

        for line in read_file(in_filename):
            document = data_class(json.loads(line.strip())) # 每行都是一个json文件格式
            for sentence in document.generate_sentence(type_format=type_format):

                if ignore_nonevent and len(sentence['events']) == 0:
                    continue

                # 处理schema数据信息, 并进行统计
                for event in sentence['events']:
                    event_schema_set = event_schema_set | get_schema(event) # set((ET, RT)) 合并后再遍历根据et 整理 dict
                    sep = '' if zh else ' '
                    predicate = sep.join([sentence['tokens'][index]
                                          for index in event['tokens']]) # 触发词的文本信息
                    counter['pred'].update([predicate]) 
                    counter['type'].update([event['type']]) # 事件类型
                    data_counter[in_filename].update(['event'])
                    for argument in event['arguments']:
                        data_counter[in_filename].update(['argument'])
                        counter['role'].update([argument[0]])

                data_counter[in_filename].update(['sentence'])

                # 训练集与验证集、测试集区分处理的类别
                if (target_class == ETText2Tri or target_class == ETRTText2Role) and "train" not in in_filename:
                    # 处理生成 span target
                    span_source_list, span_target_list = target_class.annotate_span(
                        tokens=sentence['tokens'],
                        predicate_arguments=sentence['events'],
                        zh=zh,
                        mark_tree=mark_tree,
                        isTest=True
                    )
                    # print("================================")
                else:
                    # 处理生成 span target
                    span_source_list, span_target_list = target_class.annotate_span(
                        tokens=sentence['tokens'],
                        predicate_arguments=sentence['events'],
                        zh=zh,
                        mark_tree=mark_tree
                    )

                # 将处理后的span结果信息写入文件
                assert len(span_source_list) == len(span_target_list)
                for i in range(len(span_source_list)):
                    span_event_output.write(
                        json.dumps({'text': span_source_list[i], 'event': span_target_list[i]}, ensure_ascii=False) + '\n')

        span_event_output.close()

        check_output(output_filename)
        print('\n')

    # train、dev、test 转换完成后将整体的schema信息写入文件
    output_schema(event_schema_set, output_file=os.path.join(
        span_output_folder, 'event.schema'))
    print('Pred:', len(counter['pred']), counter['pred'].most_common(10))
    print('Type:', len(counter['type']), counter['type'].most_common(10))
    print('Role:', len(counter['role']), counter['role'].most_common(10))
    print(data_counter_to_table(data_counter))
    print('\n\n\n')


def convert_dyiepp_event(output_folder='data/text2target/ace2005_event', type_format='subtype',
                         ignore_nonevent=False, mark_tree=False, target_class=ETRTText2Role):
    from data_convert.task_format.event_extraction import DyIEPP_ace2005_file_tuple
    convert_file_tuple(file_tuple=DyIEPP_ace2005_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       data_class=DyIEPP,
                       target_class = target_class
                       )

if __name__ == "__main__":
    type_format_name = 'subtype'
    
    
    # ET + RT + src -> ( (Role) (Role) )
    convert_dyiepp_event("data/text2target/dyiepp_ace2005_etrttext2role_%s" % type_format_name,
                        type_format=type_format_name,
                        ignore_nonevent=False, mark_tree=False, target_class=ETRTText2Role
                        )

    # src -> ((ET)(ET))
    convert_dyiepp_event("data/text2target/dyiepp_ace2005_text2et_%s" % type_format_name,
                        type_format=type_format_name,
                        ignore_nonevent=False, mark_tree=False, target_class=Text2ET
                        )
    
    
    # ET + src -> ((Tri)(Tri))
    convert_dyiepp_event("data/text2target/dyiepp_ace2005_ettext2tri_%s" % type_format_name,
                        type_format=type_format_name,
                        ignore_nonevent=False, mark_tree=False, target_class=ETText2Tri
                        )
    

    
