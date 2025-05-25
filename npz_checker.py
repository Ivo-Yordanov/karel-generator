import unittest
import numpy as np
import karel

class TestKarelExamples(unittest.TestCase):

    def test_karel_outputs_per_code_block(self):
        data_file = 'data/data.npz'
        codes_file = 'data/data.txt'
        debug = True

        data = np.load(data_file)
        num_per_code = int(data['num_examples_per_code'])  # should be scalar or 1D array

        total_examples = len(data['inputs'])
        self.assertEqual(total_examples % num_per_code, 0,
                         "Data length must be divisible by num_examples_per_code")

        with open(codes_file, 'r') as f:
            codes = [line.strip() for line in f]

        for i in range(0, total_examples, num_per_code):
            code = codes[i // num_per_code]
            if debug:
                print("\n========================")
                print("Code:")
                print(code)
                print("\nInput:")

            for j in range(num_per_code):
                input_world = data['inputs'][i + j]
                output_world = data['outputs'][i + j]

                parser = karel.KarelForSynthesisParser()
                input_karel = karel.Karel(state=input_world)
                if debug:
                    print("---------------")
                    input_karel.draw()

                parser.karel = input_karel

                parser.run(code)

                # Compare final state to output_world
                actual_output = parser.karel.state

                if debug:
                    print("\nExpected:")
                    output_karel = karel.Karel(state=output_world)
                    output_karel.draw()
                    print("\nActual:")
                    parser.draw()

                self.assertTrue(np.array_equal(actual_output, output_world),
                                msg="Karel output world does not match expected output.")

if __name__ == '__main__':
    unittest.main()

