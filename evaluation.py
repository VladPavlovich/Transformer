import torch
from util import get_model_device
from data import GPTTokenizedData
from model import get_best_model_definition
from torch import nn
import math
import argparse

def perplexity(model, dataloader):
    model.eval()
    device = get_model_device(model)
    
    loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)

    total_loss = 0.
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]

            padding_mask = batch['attention_mask'].to(device)
            target_padding_mask = padding_mask[:, 1:]
            input_padding_mask = padding_mask[:, :-1]
            
            # replace targets with -100 where it's padding
            targets = targets.masked_fill(target_padding_mask==0, -100).view(-1)
            
            logits = model(inputs, input_padding_mask)
            B, S, V = logits.shape
            logits = logits.view(-1, V)

            total_loss += loss_fn(logits, targets).item() 

            total_tokens += target_padding_mask.sum().item() 
    
    perplexity = math.exp(total_loss / total_tokens)

    return perplexity, total_loss


def parse_arguments():
    """
    Parse command line arguments for model evaluation.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer model")
    parser.add_argument('--model_path', default='./best_model.pt', help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    tokenized = GPTTokenizedData(args.batch_size)
    dataloaders = tokenized.dataloaders

    vocab_size = tokenized.vocab_size
    model = get_best_model_definition(vocab_size)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ppl, loss = perplexity(model, dataloaders['test'])

    print(f"Perplexity: {ppl}, Total loss: {loss}")

if __name__ == "__main__":
    main()



