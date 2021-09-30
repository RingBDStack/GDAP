#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
from typing import Dict, List

from extraction.event_schema import EventSchema


# 通过span获取到具体的文本内容(start, end位置)
def get_str_from_tokens(tokens, sentence, separator=' '):
    start, end_exclude = tokens[0], tokens[-1] + 1
    return separator.join(sentence[start:end_exclude])

# T5 tokenizer使用的特殊符号
type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
role_start = '<extra_id_2>'
role_end = '<extra_id_3>'

class TargetFormat:
    @staticmethod
    def annotate_spans(tokens: List[str], predicate_arguments: List[Dict], zh=False): pass


# et + rt + src -> ((Role)(Role)) 论元抽取任务, 在给定schema文件的情况下才可以使用该方案
class ETRTText2Role(TargetFormat):
    
    @staticmethod
    def annotate_span(tokens, predicate_arguments, mark_tree=False, zh=False, isTest=False):
        """
        src: et + </s> + RT + </s> + source text    </s> 为 t5-base 分隔符
        traget: ((Role))   只对role进行生成, 如果同一个et同样的rt存在多个role, 则生成 ((Role)(Role)...)
        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :return:
        """

        token_separator = '' if zh else ' '

        event_str_rep_list = list()
        
        source_list = []
        target_list = []

        # 若出现单句多个相同 type 事件, 则将其合并
        et_rt_index_dict = {}
        et_rt_span_dict = {} # 若出现span重复的情况则进行合并
        

        # 加载 schema文件中的信息, 目前每次都会读取文件, 很低效, 后续将这一步改为传参优化处理
        if zh:
            schema = EventSchema.read_from_file("data/raw_data/duee/event.schema")
            et_rt_dict = schema.type_role_dict
            schema_et_list = schema.type_list
        else:
            schema = EventSchema.read_from_file("data/raw_data/dyiepp_ace2005/event.schema")
            et_rt_dict = schema.type_role_dict
            schema_et_list = schema.type_list

        et_set = set() # 文本已经包含的事件类型
        
        # 遍历event
        for predicate_argument in predicate_arguments:
            event_type = predicate_argument['type']
            et_set.add(event_type)

            # 遍历role
            for role_type, role_tokens in predicate_argument['arguments']:
                if role_type == event_type:
                    continue
                
                if event_type + "-" + role_type not in et_rt_span_dict:
                    et_rt_span_dict[event_type + "-" + role_type] = set()
                                
                role_text = get_str_from_tokens(role_tokens, tokens, separator=token_separator)

                # 判断role是否出现过
                if role_text in et_rt_span_dict[event_type + "-" + role_type]: continue
                et_rt_span_dict[event_type + "-" + role_type].add(role_text) # 将当前role加入set中

                # role_str = ' '.join([type_start, role_type, role_text, type_end])
                role_str = ' '.join([type_start, role_text, type_end])

                # 在此处修改 source 与 target 格式
                source_text = event_type + " </s> " + role_type + " </s> " + token_separator.join(tokens)
                target_text = role_str

                # 判断当前ET-RT类型是否重复
                if event_type + "-" + role_type in et_rt_index_dict:
                    # print("duplicate event:", event_type, " ", tokens, " ", predicate_arguments)
                    target_list[et_rt_index_dict[event_type + "-" + role_type]] += " " + target_text
                else:
                    source_list.append(source_text)
                    target_list.append(target_text)
                    et_rt_index_dict[event_type + "-" + role_type] = len(source_list) - 1 # 将位置进行记录
            
            # print(et_rt_dict)
            # 补全在当前事件类型下, 未出现的rt样例
            for role_type in et_rt_dict[event_type]:
                if event_type + "-" + role_type not in et_rt_span_dict:
                    et_rt_span_dict[event_type + "-" + role_type] = set()
                    
                    source_text = event_type + " </s> " + role_type + " </s> " + token_separator.join(tokens)
                    target_text = ""
                    source_list.append(source_text)
                    target_list.append(target_text)
                    
                    et_rt_index_dict[event_type + "-" + role_type] = len(source_list) - 1 # 将补全的位置进行记录
        
        # negative sample on role train data
        if len(predicate_arguments) > 0 and not isTest:
            for tmp_et in random.sample(set(schema_et_list) - et_set, 4):
                for role_type in et_rt_dict[tmp_et]:
                    source_text = tmp_et + " </s> " + role_type + " </s> " + token_separator.join(tokens)
                    target_text = ""
                    source_list.append(source_text)
                    target_list.append(target_text)

        # 在所有的target 上统一加上起始位置
        for i in range(len(target_list)):
            target_list[i] = f'{type_start} ' + target_list[i] + f' {type_end}'

        return source_list, target_list       

# src -> ((ET)) 事件检测任务, 
class Text2ET(TargetFormat):
    
    @staticmethod
    def annotate_span(tokens, predicate_arguments, mark_tree=False, zh=False):
        """
        src: source text
        traget: ((ET))   只对ET进行生成, ((ET)(ET)...)
        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :return:
        """

        token_separator = '' if zh else ' '
        
        et_list = []

        # 去除重复的事件类型
        et_set = set()

        # 遍历event
        for predicate_argument in predicate_arguments:
            event_type = predicate_argument['type']
            if event_type in et_set: continue
            else: et_set.add(event_type)
            
            # 在此处修改 source 与 target 格式
            
            et_text = f'{type_start} ' + event_type + f' {type_end}'
            # et_text = event_type
            et_list.append(et_text)

        source_text = token_separator.join(tokens) 
        target_text = f'{type_start} ' + " ".join(et_list) + f' {type_end}'
        # target_text = " ".join(et_list)


        return [source_text], [target_text]       

# et + src -> ((tri)) 遍历事件类型, 针对每个给定的事件类型生成触发词(如果包含)
class ETText2Tri(TargetFormat):
    
    @staticmethod
    def annotate_span(tokens, predicate_arguments, mark_tree=False, zh=False, isTest = False):
        """
        src: et + </s> + source text    </s> 为 t5-base 分隔符
        traget: ((Tri))   只对tri进行生成, 如果同一个et存在多个Tri, 则生成 ((Tri)(Tri)...)
        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :return:
        """

        token_separator = '' if zh else ' '

        event_str_rep_list = list()
        
        source_list = []
        target_list = []

        # 若出现单句多个相同 type 事件, 则将其合并
        et_tri_dict = {} # {et + src: tri_list}
        et_set = set() # 文本已经包含的事件类型

        # 加载 schema文件中的信息, 目前每次都会读取文件, 很低效, 后续将这一步改为传参优化处理
        et_list = EventSchema.read_from_file("data/raw_data/dyiepp_ace2005/event.schema").type_list

        # 针对事件类型遍历制作训练样本
        for et in et_list:
            et_tri_dict[et + " </s> " + token_separator.join(tokens)] = set()


        # 遍历event
        for predicate_argument in predicate_arguments:
            et = predicate_argument['type']
            et_set.add(et)

            tri_text = get_str_from_tokens(predicate_argument['tokens'], tokens, separator=token_separator) # 此处的 predicate_argument['tokens'] 为 [start, end](多个单词), 或者[start] (一个单词)

            et_tri_dict[et + " </s> " + token_separator.join(tokens)].add(tri_text) 

        
        for src, tri_set in et_tri_dict.items():
            if not tri_set: continue # 过滤掉不包含触发词的样本
            source_list.append(src)
            tmp_list = []
            for tri_text in tri_set:
                tmp_list.append(' '.join([type_start, tri_text, type_end]))
            target_text = f'{type_start} ' + " ".join(tmp_list) + f' {type_end}'
            target_list.append(target_text)
        
        # negative sample on tri train data
        if not isTest:
            for tmp_et in random.sample(set(et_list) - et_set, 6):
                source_list.append(tmp_et + " </s> " + token_separator.join(tokens))
                target_list.append(f'{type_start} ' + " ".join([]) + f' {type_end}')


        return source_list, target_list       


if __name__ == "__main__":
    pass
