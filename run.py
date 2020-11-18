from ml import train_neuralnetwork, validate_txt
import sys

if __name__ == "__main__":
    if 'train' in sys.argv:
        train_neuralnetwork(splice=True)
    else:
        msg_dict = {
            0: "valid",
            1: "neutral",
            2: "disinformation"
        }
        while True:
            res = validate_txt(input("enter text to validate: "))
            print("----------------------------\nThe text is " + msg_dict[res] + "\n-----------------------------")