# -*- coding: utf-8 -*-
import pandas as pd


def remove_columns_except_headlines_and_text(data_frame):
    cols_to_remove = data_frame.columns.tolist()
    cols_to_remove.remove('headlines')
    cols_to_remove.remove('text')

    data_frame.drop(cols_to_remove, axis='columns', inplace=True)


def merge_headlines_and_text_from_csv_files(path_1, path_2, encoding):
    data_frame_1 = pd.read_csv(path_1,
                               encoding=encoding).reset_index(drop=True)
    data_frame_2 = pd.read_csv(path_2,
                               encoding=encoding).reset_index(drop=True)

    # Remove columns
    remove_columns_except_headlines_and_text(data_frame_1)
    remove_columns_except_headlines_and_text(data_frame_2)

    # Concatenate two data frames that only have 'headlines' and 'text'
    # The headlines column will be treated as summary for the text.
    data_frame = pd.concat([data_frame_1, data_frame_2], axis='rows')
    del data_frame_1, data_frame_2

    return data_frame


def percentage_of_words_that_count_under_limit(column, limit):
    count = 0
    for sentence in column:
        if len(sentence.split()) <= limit:
            count += 1

    return round(count / len(column), 2)
