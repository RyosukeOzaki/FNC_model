#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
import numpy as np

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated', 'related']
RELATED = LABELS[0:3]

'''
def score_submission_(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm
'''


def score_submission(gold_labels, test_labels):
    score = 0.0
    s1 = 0.0
    s2 = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance in RELATED and t_stance in RELATED:
            s1 += 0.25
        if not g_stance in RELATED and not t_stance in RELATED:
            s1 += 0.25
        if g_stance != 'unrelated' and g_stance == t_stance:
            s2 += 0.75
        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1
    score = s1 + s2
    return score, s1, s2, cm


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(predicted, actual):
    predicted = predicted.cpu().detach().numpy()
    actual = actual.cpu().detach().numpy()
    predicted = np.argmax(predicted, axis=1)
    predicted = predicted.tolist()
    actual = actual.tolist()
    actual = [LABELS[e] for e in actual]
    predicted = [LABELS[e] for e in predicted]
    score, s1, s2, cm = score_submission(actual, predicted)
    best_score, best_s1, best_s2, _ = score_submission(actual, actual)

    print_confusion_matrix(cm)
    print("Score: " + str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    print("S1:{0}/{1}".format(s1, best_s1) + "\tS2:{0}/{1}".format(s2, best_s2))
    print("accS1:{0}".format(s1*100/best_s1) + "\taccS2:{0}\n".format(s2*100/best_s2))
    return score*100/best_score


if __name__ == "__main__":
    actual = [0, 0, 0, 0, 1, 1, 0, 3, 3]
    predicted = [0, 0, 0, 0, 1, 1, 2, 3, 3]

    report_score([LABELS[e] for e in actual], [LABELS[e] for e in predicted])
