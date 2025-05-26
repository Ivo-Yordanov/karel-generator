#!/usr/bin/env python
import os
import argparse
import sys
from argparse import Namespace
import numpy as np

from karel import KarelWithCurlyParser, KarelForSynthesisParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError, InvalidOperation
from karel.parser_base import Parser

try:
    from tqdm import trange, tqdm
except:
    trange = range


def generate_random_code(config: Namespace, parser: Parser, name: str):
    data_num = getattr(config, "num_{}".format(name))
    codes = np.empty(data_num, dtype=str)
    for i in trange(data_num):
        code = parser.random_code(stmt_max_depth=config.max_depth)
        codes[i] = code

    return codes


def save_codes(codes: list | np.ndarray, config: Namespace, name: str):
    codes_str = ""
    codes_path = os.path.join(config.data_dir, "{}_single_line_codes.txt".format(name))
    pretty_codes_str = ""
    pretty_codes_path = os.path.join(config.data_dir, "{}_multi_line_codes.txt".format(name))


    for code in codes:
        codes_str += code + "\n"
        pretty_codes_str += beautify(code) + "\n"

    with open(codes_path, 'w') as f:
        f.write(codes_str)
    with open(pretty_codes_path, 'w') as f:
        f.write(pretty_codes_str)



def generate_world_from_code(config: Namespace, parser: Parser, code: str, cutoff=np.inf, allow_equal_start_and_end=True):
    i = 0
    while i < cutoff:
        i += 1
        parser.new_game(world_size=(config.world_width, config.world_height))
        init_world_str = parser.draw(no_print=True)
        input_world = parser.get_state()

        try:
            parser.run(code)
            output_world = parser.get_state()
            if not allow_equal_start_and_end and np.array_equal(input_world, output_world):
                continue
        except TimeoutError:
            continue
        except IndexError:
            continue
        except InvalidOperation:
            continue


        return input_world, output_world, init_world_str

    raise TimeoutError("Generated too many worlds for code example")


def save_code_and_examples(config: Namespace, parser: Parser, name: str, percentage_examples_no_change=0.5):
    data_num = getattr(config, "num_{}".format(name))
    if data_num <= 0:
        return
    inputs, outputs, codes, code_lengths = [], [], [], []
    for _ in trange(0, data_num, config.num_examples, file=sys.stdout):
        while True:
            code = parser.random_code(stmt_max_depth=config.max_depth)
            num_no_change = percentage_examples_no_change * config.num_examples
            curr_inputs, curr_outputs, curr_code_lengths = [], [], [] # We are not sure if these will be added yet

            if config.debug:
                tqdm.write("")
                tqdm.write("------------------")
                tqdm.write(code)
            try:
                for _ in range(config.num_examples):
                    input_world, output_world, init_world_str = generate_world_from_code(config, parser, code, cutoff=10000,
                                                                allow_equal_start_and_end=(num_no_change > 0))
                    if np.array_equal(input_world, output_world):
                        num_no_change -= 1

                    curr_inputs.append(input_world)
                    curr_outputs.append(output_world)

                    token_idxes = parser.lex_to_idx(code, details=True)
                    # codes.append(token_idxes)
                    curr_code_lengths.append(len(token_idxes))
            except TimeoutError:
                if config.debug:
                    tqdm.write("Fail. Could not find enough worlds for code snippet.")
                continue
            if config.debug:
                tqdm.write("Pass. Generated enough worlds for code snippet.")
            inputs.extend(curr_inputs)
            outputs.extend(curr_outputs)
            codes.append(code)
            code_lengths.extend(curr_code_lengths)
            break

    assert len(inputs) % config.num_examples == 0

    if name == 'train':
        name = 'data'
    npz_path = os.path.join(config.data_dir, name)
    np.savez(npz_path,
             num_examples_per_code=np.array(config.num_examples),
             inputs=np.array(inputs),
             outputs=np.array(outputs),
             code_lengths=np.array(code_lengths))

    if config.mode == "both":
        save_codes(codes, config, name)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_train', type=int, default=100)
    arg_parser.add_argument('--num_test', type=int, default=0)
    arg_parser.add_argument('--num_val', type=int, default=0)
    arg_parser.add_argument('--num_examples', type=int, default=10, help='Number of examples per generated code')
    arg_parser.add_argument('--parser_type', type=str, default='curly', choices=['curly', 'synthesis'])
    arg_parser.add_argument('--data_dir', type=str, default='data')
    arg_parser.add_argument('--max_depth', type=int, default=5)
    arg_parser.add_argument('--mode', type=str, default='both', choices=['code_only', 'examples_only', 'both'],
                          help='What to save in the output file - only the generated code or also example worlds with that code')
    arg_parser.add_argument('--world_height', type=int, default=8, help='Height of square grid world')
    arg_parser.add_argument('--world_width', type=int, default=8, help='Width of square grid world')
    arg_parser.add_argument('--debug', type=str2bool, default=False, help='Print generated worlds and code')
    config = arg_parser.parse_args()

    if config.num_train % config.num_examples != 0 or config.num_train % config.num_examples != 0 or \
        config.num_train % config.num_examples != 0:
        arg_parser.error("Number of examples must be divisible by the number of examples per generated code")

    # Make directories
    makedirs(config.data_dir)
    datasets = ['train', 'test', 'val']

    # Generate datasets
    if config.parser_type == "curly":
        parser = KarelWithCurlyParser()
    else:
        parser = KarelForSynthesisParser()

    for name in datasets:
        if config.mode == 'code_only':
            codes = generate_random_code(config, parser, name)
            save_codes(codes, config, name)
        else:
            save_code_and_examples(config, parser, name)

if __name__ == '__main__':
    main()