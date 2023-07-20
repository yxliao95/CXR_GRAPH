task_ner_labels = {
    "ace04": ["FAC", "WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "ace05": ["FAC", "WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "scierc": ["Method", "OtherScientificTerm", "Task", "Generic", "Material", "Metric"],
}

task_rel_labels = {
    "ace04": ["PER-SOC", "OTHER-AFF", "ART", "GPE-AFF", "EMP-ORG", "PHYS"],
    "ace05": ["ART", "ORG-AFF", "GEN-AFF", "PHYS", "PER-SOC", "PART-WHOLE"],
    "scierc": ["PART-OF", "USED-FOR", "FEATURE-OF", "CONJUNCTION", "EVALUATE-FOR", "HYPONYM-OF", "COMPARE"],
}


def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label


binary_labels = ["yes", "no"]


def get_binary_labelmap(label_list):
    """All labels have id=1; predicted id 1 will be considered as label 'IS-ENT'."""
    label2id = {}
    id2label = {1: "IS-ENT"}
    for _, label in enumerate(label_list):
        label2id[label] = 1
    return label2id, id2label
