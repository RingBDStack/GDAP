#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict

from data_convert.format.text2target import type_start, type_end
from extraction.label_tree import get_label_name_tree

import os

debug = True if 'DEBUG' in os.environ else False
debug_step = True if 'DEBUG_STEP' in os.environ else False


def match_sublist(the_list, to_match):
    """

    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def find_bracket_position(generated_text, _type_start, _type_end):
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position

# 根据generated 找到 src_sequence中与其匹配的tokens并进行返回
def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):
    print(generated, src_sequence) if debug else None

    if len(generated) == 0:
        # It has not been generated yet. All SRC are valid.
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    if end_sequence_search_tokens:
        valid_token += end_sequence_search_tokens

    return valid_token


def get_constraint_decoder(tokenizer, type_schema, decoding_schema, source_prefix=None):

    decoding_format_dict = {
        'role': RoleConstraintDecoder,
        'et': ETConstraintDecoder,
        'tri': TriConstraintDecoder,
    }

    if decoding_schema in decoding_format_dict:
        return decoding_format_dict[decoding_schema](tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
    else:
        raise NotImplementedError(
            'Type Schema %s, Decoding Schema %s do not map to constraint decoder.' % (
                decoding_schema, decoding_schema)
        )


class ConstraintDecoder:
    def __init__(self, tokenizer, source_prefix):
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []

    def get_state_valid_tokens(self, src_sentence: List[str], tgt_generated: List[str]) -> List[str]:
        pass

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        if debug:
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))

        valid_token_ids = self.get_state_valid_tokens(
            src_sentence.tolist(),
            tgt_generated.tolist()
        )

        if debug:
            print('========================================')
            print('valid tokens:', self.tokenizer.convert_ids_to_tokens(
                valid_token_ids), valid_token_ids)
            if debug_step:
                input()

        # return self.tokenizer.convert_tokens_to_ids(valid_tokens)
        return valid_token_ids

# ET + RT + Src -> ((Role)(Role)), ETRTText2Role 使用 
class RoleConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_schema = type_schema
        self.type_tree = get_label_name_tree(type_schema.role_list,
                                             tokenizer=self.tokenizer,
                                             end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id: # t5-base
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))
        # print(special_index_token)
        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end]) # 计算左右括号的数量

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence: ET </s> RT </s> src </s>
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        old_src = src_sentence
        if self.tokenizer.eos_token_id in src_sentence:
            if src_sentence.count(self.tokenizer.eos_token_id) > 1: # 有新增的</s>
                first_index = src_sentence.index(self.tokenizer.eos_token_id) # index函数会定位第一个出现的位置
                second_index = first_index + 1 + src_sentence[first_index + 1:].index(self.tokenizer.eos_token_id) # 注意要加上 first_index + 1的偏移
                third_index = second_index + 1 + src_sentence[second_index + 1: ].index(self.tokenizer.eos_token_id)
                src_sentence = src_sentence[second_index + 1: third_index] # 输入端 原句 src
                
            else:
                src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]
                # print("eos < 1 in src_sentence:", src_sentence)

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Error:")
            print("Old src:", old_src)
            # print("first_index:", first_index)
            # print("second_index:", second_index)
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id] # t5-base 使用 </s>
            # valid_tokens = [self.tokenizer.sep_token_id] # uer/t5-small-chinese-cluecorpussmall 使用 [SEP]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' %
                                   (self.type_end, tgt_generated))
            else:
                valid_tokens = generated_search_src_sequence(
                    generated=tgt_generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=[self.type_end],
                )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]
            # valid_tokens = [self.tokenizer.sep_token_id] # uer/t5-small-chinese-cluecorpussmall 使用

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens

# Src -> ((ET)(ET)), Text2ET 使用
class ETConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_tree = get_label_name_tree(type_schema.type_list,
                                             tokenizer=self.tokenizer,
                                             end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id: # t5-base
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))
        # print(special_index_token)
        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end]) # 计算左右括号的数量

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree(self, generated: List[str], prefix_tree: Dict,
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                return end_sequence_search_tokens

            if self.tree_end in tree:
                return end_sequence_search_tokens

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence: ET </s> RT </s> src </s>
        :param tgt_generated:
        :return:
            List[str], valid token list
        """

        state, index = self.check_state(tgt_generated)

        # print("State: %s" % state) if debug else None
        # print("State: %s" % state)

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id] # t5-base 使用 </s>
            # valid_tokens = [self.tokenizer.sep_token_id] # uer/t5-small-chinese-cluecorpussmall 使用 [SEP]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' %
                                   (self.type_end, tgt_generated))

            else:
                valid_tokens = self.search_prefix_tree(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    end_sequence_search_tokens=[self.type_end]
                )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]
            # valid_tokens = [self.tokenizer.sep_token_id] # uer/t5-small-chinese-cluecorpussmall 使用

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens

# ET + Src -> ((Tri)(Tri)), ETText2Tri 使用
class TriConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_schema = type_schema
        self.type_tree = get_label_name_tree(type_schema.role_list,
                                             tokenizer=self.tokenizer,
                                             end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id: # t5-base
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))
        # print(special_index_token)
        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end]) # 计算左右括号的数量

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
        else:
            state = 'error'
        return state, last_special_index


    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence: ET </s> src </s>
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        old_src = src_sentence
        if self.tokenizer.eos_token_id in src_sentence:
            if src_sentence.count(self.tokenizer.eos_token_id) > 1: # 有新增的</s>
                first_index = src_sentence.index(self.tokenizer.eos_token_id) # index函数会定位第一个出现的位置
                second_index = first_index + 1 + src_sentence[first_index + 1:].index(self.tokenizer.eos_token_id) # 注意要加上 first_index + 1的偏移
                src_sentence = src_sentence[first_index + 1: second_index] # 输入端 原句 src
                
            else:
                src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]
                # print("eos < 1 in src_sentence:", src_sentence)

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Error:")
            print("Old src:", old_src)
            # print("first_index:", first_index)
            # print("second_index:", second_index)
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id] # t5-base 使用 </s>
            # valid_tokens = [self.tokenizer.sep_token_id] # uer/t5-small-chinese-cluecorpussmall 使用 [SEP]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' %
                                   (self.type_end, tgt_generated))
            else:
                valid_tokens = generated_search_src_sequence(
                    generated=tgt_generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=[self.type_end],
                )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]
            # valid_tokens = [self.tokenizer.sep_token_id] # uer/t5-small-chinese-cluecorpussmall 使用

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens

