from data import GPTTokenizedData
from model import get_best_model_definition
from train import train_model
from evaluation import perplexity





def main():
    # get dataloaders (data.py)
    tokenized = GPTTokenizedData()
    dataloaders = tokenized.dataloaders # all 3 dataloaders in a dictionary with keys 'train', 'test', 'val
    vocab_size = tokenized.vocab_size


    # instantiate model (model.py)
    model = get_best_model_definition(vocab_size)

    # train model (train.py)
    train_model(model, dataloaders)


    # evaluate perplexity for all three splits (evaluate.py)
    evaluation = perplexity(model, dataloaders['test'])
    print(f"Test Perplexity: {evaluation}")


if __name__ == "__main__":
    main()
