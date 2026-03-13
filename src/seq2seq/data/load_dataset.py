from datasets import load_dataset


def load_colloquial_tamil_dataset():
    dataset = load_dataset("janisrebekahv/colloquial_tamil", split="train")
    print(dataset)
    ## features: ["instruction", "input", "output"]
    ## num_rows: 16269

    print(type(dataset))

    print(dataset[0])

    dataset = dataset.remove_columns("instruction")
    print(dataset)

    dataset = dataset.select(range(5000))

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(split_dataset)
    train = split_dataset["train"]
    test = split_dataset["test"]

    print(type(train))
    print(train[:2])
    print("Train and test returned from load_dataset")
    return (train, test)
