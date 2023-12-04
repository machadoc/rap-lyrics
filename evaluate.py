from sklearn.metrics import accuracy_score, f1_score

def evaluate(predictions, labels):
    return accuracy_score(labels, predictions), f1_score(labels, predictions, average='macro'), f1_score(labels, predictions, average='micro')