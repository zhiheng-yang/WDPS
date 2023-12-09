import pandas as pd
import json

def load_and_process_Boolq(jsonl_file,output_file):
    records = []
    with open(jsonl_file, 'r', encoding= 'utf-8') as file:
        for line in file:
            records.append(json.loads(line))
    boolq_df = pd.DataFrame.from_records(records)
    boolq_df['label'] = 1  # yes or no
    boolq_df = boolq_df[['question', 'label']]
    boolq_df.to_csv(output_file, index=False)
    return boolq_df
def load_and_process_squad(json_file, output_file):
    # read json
    with open(json_file, 'r', encoding='utf-8') as file:
        squad_data = json.load(file)
    # extract question
    questions = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])
    squad_df = pd.DataFrame({
        "question": questions,
        "label": [0] * len(questions)
    })
    squad_df = squad_df.tail(7000)
    squad_df.to_csv(output_file, index=False, encoding='utf-8')

    return squad_df
def load_and_process_Wikiqa(txt_file, output_file, separator = '\t'):
    wikiqa_df = pd.read_csv(txt_file, sep='\t', header=None, names=['question', 'passage', 'label'])
    wikiqa_df['label'] = 0
    wikiqa_df = wikiqa_df[['question', 'label']].drop_duplicates()
    wikiqa_df.to_csv(output_file, index=False)
    return wikiqa_df

def combine_boolq_wikiqa(boolq_df, wikiqa_df):
    combined_df = pd.concat([boolq_df, wikiqa_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_df
def main():
    # processing boolq and wikiqa dataset
    # boolq_file = 'dataset/boolq/train.jsonl'
    # boolq_output = 'dataset/boolq/processed_boolq_data.csv'
    # wikiqa_file = 'dataset/WikiQACorpus/WikiQA-train.txt'
    # wikiqa_output = 'dataset/WikiQACorpus/processed_wikiqa_data.csv'
    # boolq_df = load_and_process_Boolq(boolq_file,boolq_output)
    # wikiqa_df = load_and_process_Wikiqa(wikiqa_file,wikiqa_output)
    # boolq_df = pd.read_csv('dataset/boolq/processed_boolq_data.csv')
    # wikiqa_df = pd.read_csv('dataset/WikiQACorpus/processed_wikiqa_data.csv')
    # combined_df = combine_boolq_wikiqa(boolq_df, wikiqa_df)
    # combined_df.to_csv('dataset/train_question.csv', index=False)  # 输出到CSV文件

    #
    # process the dataset of squad
    # squad_file = 'dataset/squad/train-v2.0.json'
    # squad_output = 'dataset/squad/processed_squad_data.csv'
    # squad_df = load_and_process_squad(squad_file,squad_output)

    # #test
    # boolq_test = 'dataset/boolq/dev.jsonl'
    # boolq_test_output = 'dataset/boolq/processed_test_boolq_data.csv'
    # wikiqa_test = 'dataset/WikiQACorpus/WikiQA-test.txt'
    # wikiqa_test_output = 'dataset/WikiQACorpus/processed_test_wikiqa_data.csv'
    # boolq_test_output_df = load_and_process_Boolq(boolq_test,boolq_test_output)
    # wikiqa_test_output_df = load_and_process_Wikiqa(wikiqa_test,wikiqa_test_output)
    # combined_test_df = combine_boolq_wikiqa(boolq_test_output_df,wikiqa_test_output_df)
    # combined_test_df.to_csv('dataset/test_question.csv', index= False)

    # processing squad and wiki_boolq dataset
    # squad_df = pd.read_csv('dataset/squad/processed_squad_data.csv')
    # wiki_boolq_df = pd.read_csv('dataset/train_question.csv')
    # combine_all_df = combine_boolq_wikiqa(squad_df,wiki_boolq_df)
    # combine_all_df.to_csv('dataset/train.csv', index= False)

    # add fill_in_form to the train dataset
    fill_df = pd.read_csv('dataset/fill_in_form/fill_in_form.csv')
    train_df = pd.read_csv('dataset/train.csv')
    combine_fill_df = combine_boolq_wikiqa(fill_df, train_df)
    combine_fill_df.to_csv('dataset/train.csv', index=False)


if __name__ == '__main__':
    main()


