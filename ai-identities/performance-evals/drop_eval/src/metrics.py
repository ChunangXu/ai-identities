from utils import normalize

def drop_metric(predicted, gold_answers):
    predicted_normalized = normalize(predicted)
    gold_normalized = [normalize(answer) for answer in gold_answers]
    em_score = 1.0 if predicted_normalized in gold_normalized else 0.0
    f1_score = calculate_f1(predicted_normalized, gold_normalized)
    return em_score, f1_score

def calculate_f1(predicted, gold_answers):
    # Calculate F1 score based on token overlap
    predicted_tokens = set(predicted.split())
    gold_tokens = set(gold_answers[0].split())
    intersection = predicted_tokens.intersection(gold_tokens)
    precision = len(intersection) / len(predicted_tokens) if predicted_tokens else 0.0
    recall = len(intersection) / len(gold_tokens) if gold_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1