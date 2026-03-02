import pandas

def main():
    df = pandas.read_csv("train.csv")
    print(df.head())
    print(df.describe())
    for i in range(2, 10):
        print(df.iloc[:, i].unique().tolist())

if __name__ == "__main__":
    main()
