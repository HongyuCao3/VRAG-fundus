def collate_fn(batches, tokenizer):

    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    input_ids = tokenizer(questions, return_tensors='pt', padding='longest')

    return question_ids, input_ids.input_ids, input_ids.attention_mask, annotations