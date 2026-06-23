import pandas as pd

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

data.to_csv("spam.csv", index=False)

print("Dataset converted successfully!")
print("Total messages:", len(data))