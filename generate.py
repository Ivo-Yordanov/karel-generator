#!/usr/bin/env python
import os
import argparse
from argparse import Namespace
import numpy as np

from karel import KarelWithCurlyParser, KarelForSynthesisParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError
from karel.parser_base import Parser

try:
    from tqdm import trange
except:
    trange = range


def save_code(config: Namespace, parser: Parser, name: str):
    data_num = getattr(config, "num_{}".format(name))
    text = ""
    text_path = os.path.join(config.data_dir, "{}.txt".format(name))
    for _ in trange(data_num):
        code = parser.random_code(stmt_max_depth=config.max_depth)
        if config.beautify:
            code = beautify(code)
        text += code + "\n"
    with open(text_path, 'w') as f:
        f.write(text)


def generate_world_from_code(config: Namespace, parser: Parser, code: str):
    while True:
        parser.new_game(world_size=(config.world_width, config.world_height))
        init_world_str = parser.draw(no_print=True)
        input_world = parser.get_state()

        try:
            parser.run(code)
            output_world = parser.get_state()
        except TimeoutError:
            continue
        except IndexError:
            continue

        return input_world, output_world, init_world_str


def save_code_and_examples(config: Namespace, parser: Parser, name: str):
    data_num = getattr(config, "num_{}".format(name))
    inputs, outputs, codes, code_lengths = [], [], [], []
    for _ in trange(0, data_num, config.num_examples):
        code = parser.random_code(stmt_max_depth=config.max_depth)
        for _ in range(config.num_examples):
            input_world, output_world, init_world_str = generate_world_from_code(config, parser, code)

            if config.debug:
                print("\n".join(init_world_str), "\n")
                parser.draw()
                print()
                print(beautify(code))
                print("------------------", "\n")

            inputs.append(input_world)
            outputs.append(output_world)

            token_idxes = parser.lex_to_idx(code, details=True)
            codes.append(token_idxes)
            code_lengths.append(len(token_idxes))

    npz_path = os.path.join(config.data_dir, name)
    np.savez(npz_path,
             inputs=np.array(inputs, dtype=object),
             outputs=np.array(outputs, dtype=object),
             codes=np.array(codes, dtype=object),
             code_lengths=np.array(code_lengths, dtype=object))

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_train', type=int, default=1000000)
    arg_parser.add_argument('--num_test', type=int, default=5000)
    arg_parser.add_argument('--num_val', type=int, default=5000)
    arg_parser.add_argument('--num_examples', type=int, default=10, help='Number of examples per generated code')
    arg_parser.add_argument('--parser_type', type=str, default='curly', choices=['curly', 'synthesis'])
    arg_parser.add_argument('--data_dir', type=str, default='data')
    arg_parser.add_argument('--max_depth', type=int, default=5)
    arg_parser.add_argument('--mode', type=str, default='examples_and_code', choices=['code_only', 'examples_and_code'],
                          help='What to save in the output file - only the generated code or also example worlds with that code')
    arg_parser.add_argument('--beautify', type=str2bool, default=False)
    arg_parser.add_argument('--world_height', type=int, default=8, help='Height of square grid world')
    arg_parser.add_argument('--world_width', type=int, default=8, help='Width of square grid world')
    arg_parser.add_argument('--debug', type=str2bool, default=False, help='Print generated worlds and code')
    config = arg_parser.parse_args()

    if config.num_train % config.num_examples != 0 or config.num_train % config.num_examples != 0 or \
        config.num_train % config.num_examples != 0:
        arg_parser.error("Number of examples per generated code and world must be divisible")

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
            save_code(config, parser, name)
        else:
            save_code_and_examples(config, parser, name)

if __name__ == '__main__':
    main()