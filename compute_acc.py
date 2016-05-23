import sys, os
import numpy as np

from emails import get_emails

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: compute_acc.py <subjects_file> <results_folder>")
        exit()

    subjects = get_emails(sys.argv[1])

    accs = []
    fp_free_accs = []
    for subject in subjects:
        with open(os.path.join(sys.argv[2], subject + ".txt")) as f:
            lines = f.readlines()
            accs.append(float(lines[0].split(",")[0]))
            fp_free_accs.append(float(lines[1].split(",")[0]))

    accs = np.array(accs)
    fp_free_accs = np.array(fp_free_accs)
    print("average ACC:", np.mean(accs))
    print("max:", np.max(accs))
    print("min:", np.min(accs))
    print("std:", np.std(accs))

    print()

    print("average ACC:", np.mean(fp_free_accs))
    print("max:", np.max(fp_free_accs))
    print("min:", np.min(fp_free_accs))
    print("std:", np.std(fp_free_accs))
