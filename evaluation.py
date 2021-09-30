import argparse
import json
import os
import sys
import numpy as np
from copy import deepcopy
from pprint import pprint
from extraction.event_schema import EventSchema
from extraction.predict_parser.target_predict_parser import TreePredictParser, SpanPredictParser, RTRolePredictParser, RolePredictParser, TriPredictParser


def read_file(file_name):
    return [line.strip() for line in open(file_name).readlines()]


def generate_sentence_dyiepp(filename, type_format='subtype'):
    for line in open(filename):
        instance = json.loads(line)
        sentence = instance['sentence']
        sentence_start = instance.get(
            's_start', instance.get('_sentence_start'))
        events = instance['event']

        # 不进行去重
        trigger_list = list()
        role_list = list()
        
        # 进行去重
        trigger_set = set()
        role_set = set()

        for event in events:
            trigger, event_type = event[0]
            trigger -= sentence_start

            suptype, subtype = event_type.split('.')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            # trigger_list += [(event_type, (trigger, trigger))]
            trigger_list += [(event_type, sentence[trigger])]
            trigger_set.add((event_type, sentence[trigger]))
            for start, end, role in event[1:]:
                start -= sentence_start
                end -= sentence_start
                role_list += [(event_type, role, " ".join(sentence[start: end+1]))]
                role_set.add((event_type, role, " ".join(sentence[start: end+1])))

        # yield ' '.join(sentence), trigger_list, role_list # 不进行去重
        yield ' '.join(sentence), list(trigger_set), list(role_set) # 进行去重

def generate_sentence_text2target(filename, pred_reader):
    text_gold_dict = {}
    event_list, _ = pred_reader.decode(
        gold_list=read_file(filename),
        pred_list=read_file(filename),
        text_list=[json.loads(line)['text']
                    for line in read_file(filename)],
    )
    # print(event_list)
    for item in event_list:
        if item["text"] in text_gold_dict:
            # print("Warning: text duplicate , text: ", item["text"])
            text_gold_dict[item["text"]][0] += item['gold_event']
            text_gold_dict[item["text"]][1] += item['gold_role']
        else:
            text_gold_dict[item["text"]] = [item['gold_event'], item['gold_role']]
    # print(text_gold_dict)

    gold_list = []
    for text, events in text_gold_dict.items():
        gold_list.append([text, events[0], events[1]])
    return gold_list

def match_sublist(the_list, to_match):
    """
    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list

def record_to_offset(instance):
    """
    Find Role's offset using closest matched with trigger work.
    :param instance:
    :return:
    """
    trigger_list = list()
    role_list = list()

    token_list = instance['text'].split()

    trigger_matched_set = set()
    for record in instance['pred_record']:
        event_type = record['type']
        trigger = record['trigger']
        matched_list = match_sublist(token_list, trigger.split())

        trigger_offset = None
        for matched in matched_list:
            if matched not in trigger_matched_set:
                trigger_list += [(event_type, matched)]
                trigger_offset = matched
                trigger_matched_set.add(matched)
                break

        # No trigger word, skip the record
        if trigger_offset is None:
            break

        for _, role_type, text_str in record['roles']:
            matched_list = match_sublist(token_list, text_str.split())
            if len(matched_list) == 1:
                role_list += [(event_type, role_type, matched_list[0])]
            elif len(matched_list) == 0:
                sys.stderr.write("[Cannot reconstruct]: %s %s\n" %
                                 (text_str, token_list))
            else:
                abs_distances = [abs(match[0] - trigger_offset[0])
                                 for match in matched_list]
                closest_index = np.argmin(abs_distances)
                role_list += [(event_type, role_type,
                               matched_list[closest_index])]

    return instance['text'], trigger_list, role_list

class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list, verbose=False, text=None):
        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)
            else:
                print("text: ", text)
                print("gold_list: ", gold_list)
                print("no tp pred:", pred)
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str)
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--gold_file', type=str)
    parser.add_argument('--schema_file', type=str)

    parser.add_argument('--format', type=str, default="dyiepp")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--decoding_format', type=str, default='noetrtspan')
    options = parser.parse_args()


    label_schema = EventSchema.read_from_file(
        filename=options.schema_file
    )

    decoding_format_dict = {
        'tree': TreePredictParser,
        'span': SpanPredictParser,
        'rtrole': RTRolePredictParser,
        'role': RolePredictParser,
        'tri': TriPredictParser
    }

    # 替换为自己的predict parser
    pred_reader = decoding_format_dict[options.decoding_format](schema=label_schema) 


    trigger_metric = Metric()
    argument_metric = Metric()

    # Reconstruct the offset of predicted event records.
    text_filename = options.text_file
    pred_filename = options.pred_file
    gold_filename = options.gold_file
    print("pred_filename: ", pred_filename)
    print("gold_filename: ", gold_filename)
    
    # 离线评估
    # 在此处处理的时候, 需要将 et、rt特殊处理的部分进行添加, 以及src相同的部分进行合并
    event_list, _ = pred_reader.decode(
        gold_list=[],
        pred_list=read_file(pred_filename),
        text_list=[json.loads(line)['text']
                    for line in read_file(text_filename)],
    )
    # print(event_list[0])

    # text 中空格一类的做key会有影响, 后续可考虑用id来指代
    text_pred_dict = {} # 构建 text: ([tri_list][role_list]) 类型的字典 
    text_gold_dict = {}
    
    for item in event_list:
        if item["text"] in text_pred_dict:
            # print("Warning: text duplicate , text: ", item["text"])
            text_pred_dict[item["text"]][0] += item['pred_event']
            text_pred_dict[item["text"]][1] += item['pred_role']
        else:
            text_pred_dict[item["text"]] = [item['pred_event'], item['pred_role']]
    
    # print(text_pred_dict)

    # Read gold event annotation with offsets.
    if options.format == 'dyiepp':
        gold_list = [event for event in generate_sentence_dyiepp(gold_filename)] # 根据dyiepp预处理后的文件获取gold
    else:
        # 使用 text2target文件处理, pred_num原因低在于test文件在制作时候自动过滤了未出现事件类型的句子, 因此需要引入pred中有结果而gold中无结果的句子进行计数
        gold_list = generate_sentence_text2target(gold_filename, pred_reader) # 根据text2target预处理后的文件获取gold
    
    # print("gold_list: ", gold_list)
    
    # 遍历计算tp
    gold_text_set = set()        
    for gold in gold_list:
        if gold[0] in text_pred_dict:
            trigger_metric.count_instance(
                gold_list=gold[1],
                pred_list=text_pred_dict[gold[0]][0],
                verbose=options.verbose,
                text=gold[0]
            )
            argument_metric.count_instance(
                gold_list=gold[2],
                pred_list=text_pred_dict[gold[0]][1],
                verbose=options.verbose,
                text=gold[0]
            )
        else:
            # print(gold)
            trigger_metric.count_instance(
                gold_list=gold[1],
                pred_list=[],
                verbose=options.verbose,
                text=gold[0]
            )
            argument_metric.count_instance(
                gold_list=gold[2],
                pred_list=[],
                verbose=options.verbose,
                text=gold[0]
            )
    
    # 计算未在gold却在pred中的样本数量

    trigger_result = trigger_metric.compute_f1(prefix='result-trig-')
    role_result = argument_metric.compute_f1(prefix='result-role-')

    pprint(trigger_result)
    pprint(role_result)


if __name__ == "__main__":
    main()
