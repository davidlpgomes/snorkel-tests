import gensim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.utils import probs_to_preds

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve


VECTOR_SIZE = 300
WINDOW_SIZE = 10
MIN_COUNT = 2
WORKERS = 12


LABELS = {
    'None': 0,
    'Gain privileges': 1,
    'Sql Injection': 2,
    'Obtain Information': 3,
    'Memory corruption': 4,
    'CSRF': 5,
    'Execute Code': 6,
    'Denial Of Service': 7,
    'Cross Site Scripting': 8,
    'Http response splitting': 9,
    'Directory traversal': 10,
    'Bypass a restriction or similar': 11,
    'Overflow': 12
}

ABSTAIN = -1


def print_most_used_words_in_cves_by_label(label, words_count=10):
    cves_memory = cves[cves['labels'] == label].reset_index(drop=True)
    print(f'{label} - cves count: {cves_memory.shape[0]}')
    memory_words_count = {}
    for summary in cves_memory['summary'].to_numpy():
        words = summary.lower().split(' ')
        for word in words:
            if len(word) < 3:
                continue

            if word in memory_words_count.keys():
                memory_words_count[word] += 1
            else:
                memory_words_count[word] = 1

    sorted_count = dict(sorted(memory_words_count.items(), key=lambda item: item[1], reverse=True))
    count = 0
    for key, value in sorted_count.items():
        if count == words_count:
            break

        print(f'\t{key}: {value}')

        count += 1
    print()


# Memory corruption labeling functions
@labeling_function()
def lf_has_memory(x):
    return LABELS['Memory corruption'] if 'memory' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_corruption(x):
    return LABELS['Memory corruption'] if 'corruption' in x.summary.lower() else ABSTAIN


# Gain privileges labeling functions
@labeling_function()
def lf_has_gain(x):
    return LABELS['Gain privileges'] if 'gain' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_privilege(x):
    return LABELS['Gain privileges'] if 'privilege' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_gain_privilege(x):
    return LABELS['Gain privileges'] if 'gain privilege' in x.summary.lower() else ABSTAIN


# Obtain Information labeling functions
@labeling_function()
def lf_has_obtain(x):
    return LABELS['Obtain Information'] if 'obtain' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_information(x):
    return LABELS['Obtain Information'] if 'information' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_obtain_information(x):
    return LABELS['Obtain Information'] if 'obtain information' in x.summary.lower() else ABSTAIN


# None labeling functions
@labeling_function()
def lf_has_user(x):
    return LABELS['None'] if 'user' in x.summary.lower() else ABSTAIN


# Sql Injection labeling functions
@labeling_function()
def lf_has_sql(x):
    return LABELS['Sql Injection'] if 'sql' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_injection(x):
    return LABELS['Sql Injection'] if 'injection' in x.summary.lower() else ABSTAIN


# CSRF labeling functions
@labeling_function()
def lf_has_csrf(x):
    return LABELS['CSRF'] if 'csrf' in x.summary.lower() else ABSTAIN


# Execute Code labeling functions
@labeling_function()
def lf_has_code(x):
    return LABELS['Execute Code'] if 'code' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_execution(x):
    return LABELS['Execute Code'] if 'execution' in x.summary.lower() else ABSTAIN


# Denial Of Service labeling functions
@labeling_function()
def lf_has_denial(x):
    return LABELS['Denial Of Service'] if 'denial' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_service(x):
    return LABELS['Denial Of Service'] if 'service' in x.summary.lower() else ABSTAIN


# Cross Site Scripting labeling functions
@labeling_function()
def lf_has_crosssite(x):
    return LABELS['Cross Site Scripting'] if 'cross-site' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_scripting(x):
    return LABELS['Cross Site Scripting'] if 'scripting' in x.summary.lower() else ABSTAIN


# Http response splitting labeling functions
@labeling_function()
def lf_has_html(x):
    return LABELS['Http response splitting'] if 'html' in x.summary.lower() else ABSTAIN


# Directory traversal labeling functions
@labeling_function()
def lf_has_traversal(x):
    return LABELS['Directory traversal'] if 'traversal' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_directory(x):
    return LABELS['Directory traversal'] if 'directory' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_files(x):
    return LABELS['Directory traversal'] if 'files' in x.summary.lower() else ABSTAIN


# Bypass a restriction or similar labeling functions
@labeling_function()
def lf_has_bypass(x):
    return LABELS['Bypass a restriction or similar'] if 'bypass' in x.summary.lower() else ABSTAIN


# Overflow labeling functions
@labeling_function()
def lf_has_overflow(x):
    return LABELS['Overflow'] if 'overflow' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_buffer(x):
    return LABELS['Overflow'] if 'buffer' in x.summary.lower() else ABSTAIN

@labeling_function()
def lf_has_heap(x):
    return LABELS['Overflow'] if 'heap' in x.summary.lower() else ABSTAIN


cves = pd.read_csv('cves_labeled.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)

multiple_labels_idxs = [i for i in range(cves.shape[0]) if ',' in cves['labels'][i]]
cves = cves.drop(multiple_labels_idxs).reset_index(drop=True)


lfs = [
    lf_has_memory,
    lf_has_corruption,
    lf_has_gain,
    lf_has_privilege,
    lf_has_gain_privilege,
    lf_has_obtain,
    lf_has_information,
    lf_has_user,
    lf_has_sql,
    lf_has_injection,
    lf_has_code,
    lf_has_execution,
    lf_has_denial,
    lf_has_service,
    lf_has_crosssite,
    lf_has_scripting,
    lf_has_html,
    lf_has_traversal,
    lf_has_directory,
    lf_has_files,
    lf_has_bypass,
    lf_has_overflow,
    lf_has_buffer,
    lf_has_heap,
]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=cves)
y_train = pd.DataFrame([LABELS[l.strip()] for l in cves['labels'].to_list()])

LFAnalysis(L=L_train, lfs=lfs).lf_summary()

label_model = LabelModel(cardinality=len(LABELS), verbose=False)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
probas = label_model.predict_proba(L=L_train)
label_acc = label_model.score(L=L_train, Y=y_train, tie_break_policy='random')['accuracy']

print('--- Accuracies')
print(f'Label model: {label_acc * 100:06.03f}')

y = probs_to_preds(probs=probas)

summaries = cves['summary'].to_numpy()

def preprocess(data):
    for i in range(len(data)):
        yield gensim.utils.simple_preprocess(data[i])

documents = list(preprocess(summaries))

model = gensim.models.Word2Vec(
    documents,
    vector_size=VECTOR_SIZE,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS
)

model.train(
    documents,
    total_examples=len(documents),
    epochs=10
)

sentences = [gensim.utils.simple_preprocess(sentence) for sentence in cves['summary'].to_numpy()]

X = []

for sentence in sentences:
    wvs = []
    
    for word in sentence:
        try:
            wvs.append(model.wv[word])
        except KeyError:
            wvs.append(np.zeros(VECTOR_SIZE))
            
    X.append(np.mean(wvs, axis=0))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def evaluate(labels, scores):
    print(f'\tAcc: {accuracy_score(labels, scores)}')
    print(f'\tPrecision: {precision_score(labels, scores, average="micro", zero_division=0)}')
    print(f'\tRecall: {recall_score(labels, scores, average="micro", zero_division=0)}')
    print(f'\tF1: {f1_score(labels, scores, average="micro", zero_division=0)}')


def train_with_random_forest():
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'\n----Random Forest')
    evaluate(y_test, y_pred)


train_with_random_forest()
