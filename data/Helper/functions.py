
from sklearn.metrics import f1_score
from typing import Set, Tuple, Union, Dict

def calculate_precision_recall_fscore(candidate_set: Set[str], reference_set: Set[str]) -> Dict[str, Union[float, None]]:
    # Calculate precision
    if len(candidate_set) == 0:
        precision = 0.0
    else:
        common_items = len(candidate_set.intersection(reference_set))
        precision = common_items / len(candidate_set)

    # Calculate recall
    if len(reference_set) == 0:
        recall = 0.0
    else:
        common_items = len(candidate_set.intersection(reference_set))
        recall = common_items / len(reference_set)

    # Calculate F-score
    if precision == 0.0 or recall == 0.0 or (precision + recall) == 0.0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    result = {
        'precision': precision,
        'recall': recall,
        'f_score': f_score
    }

    return result

def avg_precision_recall_fscore(results, cheat=False):
    precision = 0.0
    recall = 0.0
    f_score = 0.0

    
    count = 0
    # myresults = copy.deepcopy(results)
    for result in results:
        #Compute precision, recall and f_score
        reference_set = set(result[0])
        candidate_set = set(result[1][:len(reference_set)] if cheat else result[1])
        entry_metrics = calculate_precision_recall_fscore(candidate_set, reference_set)

        precision += entry_metrics['precision']
        recall += entry_metrics['recall']
        f_score += entry_metrics['f_score']

        count += 1

    #Compute avg metrics
    precision /= count
    recall /= count
    f_score /= count

    precision *= 100
    recall *= 100
    f_score *= 100

    return precision, recall, f_score

def normalized_confidence_thresh(predictions, factor, conf = False):
    quantity_idx = 1 if conf else 0
    
    total = 0
    
    for i in predictions:
        total += i[1][quantity_idx]
    
    predictions = [i for i in predictions if i[1][quantity_idx]/total > factor]
    
    return predictions