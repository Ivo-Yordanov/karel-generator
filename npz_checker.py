import numpy as np
import argparse
import karel


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_file', type=str, default='data/train.npz')
    config = arg_parser.parse_args()

    data = np.load(config.data_file, allow_pickle=True)
    print("num_examples_per_code:", data['num_examples_per_code'])
    for input_world, output_world, code in zip(data['inputs'], data['outputs'], data['codes']):
        print(code)
        parser = karel.KarelForSynthesisParser()
        print("\nExpected:")
        expected_karel = karel.Karel(state=input_world)
        parser.karel = expected_karel
        parser.draw()
        parser.run(code)
        print("\nActual:")
        parser.draw()

if __name__ == '__main__':
    main()