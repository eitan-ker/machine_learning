import sys
import random


def main():
    # parameters should be: train_x train_y
    training_examples, training_labels = sys.argv[1], sys.argv[2]
    taken = random.sample(range(1, 55001), 5000)
    taken.sort()
    print(taken)
    with open('train_x_short', 'w') as out1:
        i = 1
        f1 = open(training_examples)
        for line1 in f1:
            if i in taken:
                out1.write(line1)
            i += 1
        f1.close()
    with open('train_y_short', 'w') as out2:
        i = 1
        f2 = open(training_labels)
        for line2 in f2:
            if i in taken:
                out2.write(line2)
            i += 1
        f2.close()


if __name__ == "__main__":
    main()