import pandas as pd

#Set weight1, weight2, and bias for AND Gate
weight1_and = 1.0
weight2_and = 1.0
bias_and = -1.5

#Set weight1, weight2 and bias for OR Gate
weight1_or = 1.0
weight2_or = 1.0
bias_or = -0.5

#Equation
#weight1 * input_1 + weight2 * input_2 + b
#Output will be a 0 or 1

# Inputs and outputs for XORs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, True, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    # OR Perceptron 1
    #Linear Operation
    linear_combination_p1 = weight1_or * \
        test_input[0] + weight2_or * test_input[1] + bias_or
    # Activation function - Step Function in our case
    output_p1 = int(linear_combination_p1 >= 0)

    # NAND Perceptron 2
    #Linear Operation
    linear_combination_p2 = weight1_and * \
        test_input[0] + weight2_and * test_input[1] + bias_and
    # Activation function - Step Function in our case
    output_p2 = int(linear_combination_p2 < 0)

   # AND Perceptron 3
    linear_combination_p3 = weight1_and * \
        output_p1 + weight2_and * output_p2 + bias_and
    # Activation function - Step Function in our case
    output_p3 = int(linear_combination_p3 >=0)

    is_correct_string = 'Yes' if output_p3 == correct_output else 'No'
    outputs.append([test_input[0], test_input[1],
                    linear_combination_p3, output_p3, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=[
                            'Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
