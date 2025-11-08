"""
TODO write your training loop here.
Things to take care with:
    - make sure to use the correct loss function for the task
    - make sure that the targets are correct (each token should predict the next token in the sequence)
    - there should be no loss for padding tokens.
"""
import torch
from torch import nn
from util import count_trainable_parameters

def train_model(model, dataloader, save_path='best_model.pt'):
    # check for gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    #ignore index -100
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # can add ignore_index=pad_token_id if using padding

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    epochs = 25

    # training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        for batch in dataloader['train']:
            # [B, S]
            # move batch to device (may be tensor or dict depending on dataset)
            tokens = batch["input_ids"].to(device)
            #adding the padding mask called attention mask in data.py
            attn_mask = batch["attention_mask"].to(device)

            # create inputs and targets (next-token prediction)
            #add on device
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:].contiguous()  # shift by 1
            input_mask = attn_mask[:,:-1]
            target_mask = attn_mask[:, 1:]

            targets = targets.masked_fill(target_mask == 0, -100)

            optimizer.zero_grad()
            logits = model(inputs,input_mask)  # [B, S-1, vocab_size]

            # reshape for cross entropy (B*S, V)


            #pass in attention to the targets as well
            # loss isn't affected by padding tokens
            # targets -> replace padding otkens with -100 then CEloss will ignore
            #msked.fill targerts tensore apply mask.fill, put tensore with same shape as targets, (boolean or maybe 0 or 1, -100)
            # ex ) (2,3) 2> -100 , (T,F) 3 ->  final (-100, 3, ...)
            #padding masking in data.py is called attention mask


            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader['train'])

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader['val']:
                tokens = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)

                inputs = tokens[:, :-1]
                targets = tokens[:, 1:].contiguous()
                input_mask = attn_mask[:, :-1]
                target_mask = attn_mask[:, 1:]

                targets = targets.masked_fill(target_mask == 0, -100)
                logits = model(inputs, input_mask)

                val_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dataloader['val'])
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Model saved (new best).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # save final model state at end of training (even if not best)
    torch.save(model.state_dict(), save_path)
    print("Final model saved to", save_path)