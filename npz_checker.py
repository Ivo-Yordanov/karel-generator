import unittest
import numpy as np
import karel

class TestKarelExamples(unittest.TestCase):

    def test_karel_outputs_per_code_block(self):
        data_file = 'data/data.npz'  # Adjust path as needed
        debug = False

        data = np.load(data_file)
        num_per_code = int(data['num_examples_per_code'])  # should be scalar or 1D array

        total_examples = len(data['inputs'])
        self.assertEqual(total_examples % num_per_code, 0,
                         "Data length must be divisible by num_examples_per_code")

        codes = data['codes']
        for i in range(0, total_examples, num_per_code):
            for j in range(num_per_code):
                self.assertEqual(codes[i+j], codes[i], "The code in the block should be the same")

        for input_world, output_world, code in zip(data['inputs'], data['outputs'], data['codes']):
            parser = karel.KarelForSynthesisParser()
            input_karel = karel.Karel(state=input_world)
            if debug:
                print("\nCode:")
                print(code)
                print("\nInput:")
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

