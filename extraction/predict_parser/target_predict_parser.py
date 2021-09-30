from collections import Counter
from typing import Tuple, List, Dict

from nltk.tree import ParentedTree
import re

from extraction.predict_parser.predict_parser import PredictParser


type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
role_start = '<extra_id_2>'
role_end = '<extra_id_3>'


left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>") # t5-base/mt5-base
specical_str = "</s>"


def add_space(text):
    """
    add space between special token
    :param text:
    :return:
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]): # 此处将第一个左括号 左边的token全部去掉 (如果以[CLS]开头则会被丢弃)
        new_text_list += item
    return ' '.join(new_text_list) # 空格组合对于中文效果？

def find_bracket_num(tree_str):
    """
    Count Bracket Number, 0 indicate num_left = num_right
    :param tree_str:
    :return:
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()
    # bracket_num = find_bracket_num(tree_str_list)
    # bracket_num = find_bracket_num(tree_str_list)

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def add_bracket(tree_str): # 补全不够的右括号
    """
    add right bracket to fill ill-formed
    :param tree_str:
    :return:
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """
    get str from event tree
    :param tree:
    :return:
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


# ET + RT + Src -> ((Role)(Role)), ETRTText2Role 使用
class RolePredictParser(PredictParser):
    
    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List[Dict], Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            Counter:
        """
        counter = Counter()
        well_formed_list = []

        def convert_bracket(_text):
            _text = add_space(_text)
            for start in [role_start, type_start]:
                _text = _text.replace(start, left_bracket)
            for end in [role_end, type_end]:
                _text = _text.replace(end, right_bracket)
            return _text

        if gold_list is None or len(gold_list) == 0: # 不存在标注信息的情况下根据pred_list 全部置为空
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list, raw_list):
            # print(gold)
            # print("*************************************")
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            gold = clean_text(gold)
            pred = clean_text(pred)
            # print(gold)

            et = None # 事件类型
            rt = None # 角色类型
            if text and specical_str in text:
                et = text.split(specical_str)[0].strip()
                rt = text.split(specical_str)[1].strip()
                text = text.split(specical_str)[-1].strip()

            instance = {'gold': gold,
                        'pred': pred,
                        'gold_tree': None,
                        'text': text,
                        'raw_data': raw_data
                        }

            # 重点修改部分
            instance['pred_event'], instance['pred_role'], instance['pred_record'] = self.get_event_list(
                span_str=instance["pred"],
                text=instance['text'],
                et = et,
                rt = rt
            )
            instance['gold_event'], instance['gold_role'], instance['gold_record'] = self.get_event_list(
                span_str=instance["gold"],
                text=instance['text'],
                et = et,
                rt = rt
            )

            
            # span中该部分无意义
            counter.update(['gold_tree'])
            counter.update(['pred_tree'])
            counter.update(['well-formed'])

            self.count_multi_event_role_in_instance(instance=instance, counter=counter)

            well_formed_list += [instance]

        return well_formed_list, counter
    
    
    def get_event_list(self, span_str, text=None, et=None, rt=None):

        event_list = list()
        role_list = list()
        record_list = list()

        # 将target结果格式化处理
        spans = []
        for item in span_str.replace(left_bracket, "").split(right_bracket):
            t = item.strip()
            if len(t) > 0: spans.append(t)

        cur_et_type = et # 在 span 生成时候使用
        for span_item in spans:

            if len(span_item) == 0:
                continue
            
            span_text = span_item
            
            # role text
            if text is not None and span_text not in text: continue 
            role_list += [(cur_et_type, rt, span_text)]

        record = {'roles': role_list, 'type': event_list, 'trigger': None}

            
        record_list += [record]

        return event_list, role_list, record_list
        
# Src -> ((ET)(ET)), Text2ET 使用
class ETPredictParser(PredictParser):
    
    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List[Dict], Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            Counter:
        """
        counter = Counter()
        well_formed_list = []

        def convert_bracket(_text):
            _text = add_space(_text)
            for start in [role_start, type_start]:
                _text = _text.replace(start, left_bracket)
            for end in [role_end, type_end]:
                _text = _text.replace(end, right_bracket)
            return _text

        if gold_list is None or len(gold_list) == 0: # 不存在标注信息的情况下根据pred_list 全部置为空
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list, raw_list):
            # print(gold)
            # print("*************************************")
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            gold = clean_text(gold)
            pred = clean_text(pred)
            # print(gold)

            et = None # 事件类型
            rt = None # 角色类型
            if text and specical_str in text:
                et = text.split(specical_str)[0].strip()
                rt = text.split(specical_str)[1].strip()
                text = text.split(specical_str)[-1].strip()

            instance = {'gold': gold,
                        'pred': pred,
                        'gold_tree': None,
                        'text': text,
                        'raw_data': raw_data
                        }

            # 重点修改部分
            instance['pred_event'], instance['pred_role'], instance['pred_record'] = self.get_event_list(
                span_str=instance["pred"],
                text=instance['text'],
                et = et,
                rt = rt
            )
            instance['gold_event'], instance['gold_role'], instance['gold_record'] = self.get_event_list(
                span_str=instance["gold"],
                text=instance['text'],
                et = et,
                rt = rt
            )

            
            # span中该部分无意义
            counter.update(['gold_tree'])
            counter.update(['pred_tree'])
            counter.update(['well-formed'])

            self.count_multi_event_role_in_instance(instance=instance, counter=counter)

            well_formed_list += [instance]

        return well_formed_list, counter
    
    
    def get_event_list(self, span_str, text=None, et=None, rt=None):

        event_list = list()
        role_list = list()
        record_list = list()

        # 将target结果格式化处理
        spans = []
        for item in span_str.replace(left_bracket, "").split(right_bracket):
            t = item.strip()
            if len(t) > 0: spans.append(t)

        cur_et_type = et # 在 span 生成时候使用
        for span_item in spans:

            if len(span_item) == 0:
                continue
            
            span_text = span_item

            event_list += [(span_text)]

        record = {'roles': role_list, 'type': event_list, 'trigger': None}

        record_list += [record]

        return event_list, role_list, record_list

# ET + Src -> ((Tri)(Tri)), ETText2Tri 使用
class TriPredictParser(PredictParser):
    
    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List[Dict], Counter]:
        counter = Counter()
        well_formed_list = []

        def convert_bracket(_text):
            _text = add_space(_text)
            for start in [role_start, type_start]:
                _text = _text.replace(start, left_bracket)
            for end in [role_end, type_end]:
                _text = _text.replace(end, right_bracket)
            return _text

        if gold_list is None or len(gold_list) == 0: # 不存在标注信息的情况下根据pred_list 全部置为空
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list, raw_list):
            # print(gold)
            # print("*************************************")
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            gold = clean_text(gold)
            pred = clean_text(pred)
            # print(gold)

            et = None # 事件类型
            rt = None # 角色类型
            if text and specical_str in text:
                et = text.split(specical_str)[0].strip()
                rt = text.split(specical_str)[1].strip()
                text = text.split(specical_str)[-1].strip()

            instance = {'gold': gold,
                        'pred': pred,
                        'gold_tree': None,
                        'text': text,
                        'raw_data': raw_data
                        }

            # 重点修改部分
            instance['pred_event'], instance['pred_role'], instance['pred_record'] = self.get_event_list(
                span_str=instance["pred"],
                text=instance['text'],
                et = et,
                rt = rt
            )
            instance['gold_event'], instance['gold_role'], instance['gold_record'] = self.get_event_list(
                span_str=instance["gold"],
                text=instance['text'],
                et = et,
                rt = rt
            )

            
            # span中该部分无意义
            counter.update(['gold_tree'])
            counter.update(['pred_tree'])
            counter.update(['well-formed'])

            self.count_multi_event_role_in_instance(instance=instance, counter=counter)

            well_formed_list += [instance]

        return well_formed_list, counter
    
    def get_event_list(self, span_str, text=None, et=None, rt=None):

        event_list = list()
        role_list = list()
        record_list = list()

        # 将target结果格式化处理
        spans = []
        for item in span_str.replace(left_bracket, "").split(right_bracket):
            t = item.strip()
            if len(t) > 0: spans.append(t)

        cur_et_type = et # 在 span 生成时候使用
        for span_item in spans:
            if len(span_item) == 0:
                continue
            span_text = span_item
            
            # role text
            if text is not None and span_text not in text: continue 
            event_list += [(cur_et_type, span_text)]

        record = {'roles': role_list, 'type': event_list, 'trigger': None}

            
        record_list += [record]

        return event_list, role_list, record_list

