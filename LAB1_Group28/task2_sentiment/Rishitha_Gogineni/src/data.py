from datasets import load_dataset


def load_yelp(config):
    return load_dataset(config["dataset_name"])


def get_text_column(dataset):
    if "text" in dataset.column_names:
        return "text"
    raise ValueError("Text column was not found.")


def get_label_column(dataset):
    if "label" in dataset.column_names:
        return "label"
    raise ValueError("Label column was not found.")


def check_split(dataset, text_col, label_col):
    missing_text = 0
    empty_text = 0
    missing_label = 0
    malformed_label = 0

    for row in dataset:
        text = row.get(text_col)
        label = row.get(label_col)

        if text is None:
            missing_text += 1
        elif not isinstance(text, str) or not text.strip():
            empty_text += 1

        if label is None:
            missing_label += 1
        elif label not in (0, 1):
            malformed_label += 1

    return {
        "missing_text": missing_text,
        "empty_or_malformed_text": empty_text,
        "missing_label": missing_label,
        "malformed_label": malformed_label,
    }
