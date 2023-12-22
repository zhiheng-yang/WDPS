### Question Detection

This is a question detection classifier to detect a question is whether a yes/no question or other questions.

This classifier use the combination of two datasets: <a href="https://huggingface.co/datasets/trec">Trec</a> and <a href="https://huggingface.co/datasets/boolq">Boolq</a>.

We append extra question marks to the end of each question in the BoolQ dataset to maintain data consistency with the TREC dataset. In our labeling scheme, questions from the BoolQ dataset are labeled with '1', while those from the TREC dataset are labeled with '0'. We then fine-tune the 'bert-base-uncased' model on this combined dataset.