from sklearn.metrics import f1_score, accuracy_score
from tqdm.notebook import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def train(model, dataloader, optimizer, num_epochs, device):
    """Train loop"""

    for epoch in tqdm(range(num_epochs)):
        model.train()

        train_loss = 0.0
        y_pred = []
        y_test = []

        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            batch_predicted_labels = outputs.logits.argmax(dim=1)
            y_pred.extend(batch_predicted_labels.tolist())
            y_test.extend(labels.tolist())

            loss.backward()
            optimizer.step()

        average_train_loss = train_loss / len(dataloader)
        train_accuracy = accuracy_score(y_test, y_pred)
        train_f1 = f1_score(y_test, y_pred)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1 Score: {train_f1:.4f}')


def test(model, dataloader, device):
    """Test loop"""

    model.eval()
    eval_loss = 0.0
    predictions = []
    ground_truth = []

    train_loss = 0.0
    y_pred = []
    y_test = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()

            batch_predicted_labels = outputs.logits.argmax(dim=1)
            y_pred.extend(batch_predicted_labels.tolist())
            y_test.extend(labels.tolist())


    precision_ = [round(precision_score(y_test, y_pred), 3)]
    recall_score_ = [round(recall_score(y_test, y_pred), 3)]
    f1_score_ = [round(f1_score(y_test, y_pred), 3)]
    accuracy_score_ = [round(accuracy_score(y_test, y_pred), 3)]

    return {
            'precision': precision_,
            'recall_score': recall_score_,
            'accuracy_score': accuracy_score_,
            'f1_score': f1_score_
            } 
