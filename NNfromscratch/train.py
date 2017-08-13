import numpy

# tell numpy the dimensions of our arrays
samples = numpy.empty([0, 256])
results = numpy.empty([0, 10])

with open(os.path.dirname(os.path.realpath(__file__)) + '/semeion.data') as file:
    for line in file:
        # split line to array using space as separator
        numbers = line.split(' ')
        # as line read from the file is always is string, we need to convert first 256 parts to decimals,
        # and following 10 to integers
        sample = map(lambda x: float(x), numbers[0:256])
        result = map(lambda x: int(x), numbers[256:266])

        # after that, append freshly read sample and result to arrays
        samples = numpy.concatenate((samples, numpy.array([sample])), axis=0)
        results = numpy.concatenate((results, numpy.array([result])), axis=0)

# logistic function
def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))

# numpy.random returns 0..1, by multiplying by 2 we get 0..2,
# by subtracting 1 we get -1..1, and by division by 100 we get -0.01..0.01
first_layer = (2 * numpy.random.random((256, 256)) - 1) / 100  # the array has 256x256 dimensions
second_layer = (2 * numpy.random.random((256, 10)) - 1) / 100  # the array has 256x10 dimensions

# Feed forward through both layers
first_output = sigmoid(numpy.dot(sample, first_layer))
second_output = sigmoid(numpy.dot(first_output, second_layer))
