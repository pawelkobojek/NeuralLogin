from __future__ import print_function
from emails import get_emails
import os

def from_file(filename, output_dir="results_data2layer"):
    with open(filename) as f:
        results = f.read().splitlines()
    i = 0
    while i < len(results):
        line = results[i]
        if len(line) != 0:
            elements = line.split()
            if elements[0] == "TotalFP:" or elements[0] == "Acc":
                return
            mail = elements[1]
            fp = int(elements[3])
            fn = int(elements[5])
            tp = int(elements[7])
            tn = int(elements[9])
            acc = float(tp + tn) / float(tp + tn + fn + fp)
            predictions_line = results[i+1]
            should_be_line   = results[i+2]

            out_file = os.path.join(output_dir, mail + ".txt")
            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))

            with open(out_file, "w+") as out:
                out.write(predictions_line.split(": [")[1][:-1].replace(" ", ""))
                out.write("\n")
                out.write(should_be_line.split(": [")[1][:-1].replace(" ", ""))
                out.write("\n")
                out.write(str(acc))
            i+=3
        else:
            i+=1

if __name__ == '__main__':
    import sys
    input = 'output2LSTM.txt'
    output_dir = 'result_data2layer'
    if len(sys.argv) > 1:
        input = sys.argv[1]
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]

    emails = get_emails()
    from_file(input, output_dir)
