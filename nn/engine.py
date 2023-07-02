from sklearn.metrics import f1_score, accuracy_score
from tqdm.notebook import tqdm
import torch


def train(model, dataloader, optimizer, num_epochs, device):
    """Train loop"""

    for epoch in tqdm(range(num_epochs)):
        model.train()

        train_loss = 0.0
        predicted_labels = []
        true_labels = []

        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            batch_predicted_labels = outputs.logits.argmax(dim=1)
            predicted_labels.extend(batch_predicted_labels.tolist())
            true_labels.extend(labels.tolist())

            loss.backward()
            optimizer.step()

        average_train_loss = train_loss / len(dataloader)
        train_accuracy = accuracy_score(true_labels, predicted_labels)
        train_f1 = f1_score(true_labels, predicted_labels)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1 Score: {train_f1:.4f}')


def test(model, dataloader, device):
    """Test loop"""

    model.eval()
    eval_loss = 0.0
    predictions = []
    ground_truth = []

    train_loss = 0.0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()

            batch_predicted_labels = outputs.logits.argmax(dim=1)
            predicted_labels = predicted_labels.extend(batch_predicted_labels.tolist())
            true_labels = true_labels.extend(labels.tolist())


    average_test_loss = eval_loss / len(dataloader)
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    train_f1 = f1_score(true_labels, predicted_labels)

    print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')
