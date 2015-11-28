def load_data(mail, base_dir="../../training_data"):
    import os
    filename = os.path.join(base_dir, mail + ".txt")
    # filename = "../../training_data/" + mail + ".txt"
    with open(filename) as f:
        dataset = [d.split(',') for d in f.read().splitlines()]
        Y_data = [int(c[0]) for c in dataset]
        X_data = [c[1:] for c in dataset]
        X_data = map(lambda x: map(lambda y: int(y), x), X_data)

    return X_data, Y_data
