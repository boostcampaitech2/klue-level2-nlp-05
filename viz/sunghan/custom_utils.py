import pandas as pd
from IPython.core.display import HTML

def label_highlight(label, s_obj, o_obj):
    for obj, word in zip([s_obj, o_obj], label.split(':')):
        label = label.replace(
            word,
            f"<span style='color:{obj['color']}; background-color:{obj['bgcolor']}'>{word}</span>"
        )
    return label

def sentence_highlight(text, s_obj, o_obj, max_len):
    objs = [s_obj, o_obj]
    temp_list = []
    
    for obj in objs:
        word = text[obj['s_idx']:obj['e_idx']+1]
        temp = obj['prefix'] + '#'*(len(word)-1)
        text = text[:obj['s_idx']] + temp + text[obj['e_idx']+1:]
        temp_list.append(temp)
    
    for obj, temp in zip(objs, temp_list):
        replace_str = f"<span style='color:{obj['color']}; background-color:{obj['bgcolor']}'>{obj['word']}</span>"
        text = text.replace(temp, replace_str)
        if max_len is not None:
            max_len = max_len + len(replace_str) - len(temp)
    
    if max_len is not None:
        pre, post = text[:max_len], text[max_len:]
        post = f"<span style='color:white; background-color:black'>{post}</span>"
        text = pre + post
    
    return text

def highlight(new_df, id=None, max_len=None):
    if id is None:
        sample = new_df.sample(1)
    else:
        sample = new_df.loc[new_df['id'] == id, :]
        
    s_obj={
        'prefix': 's',
        'word': sample['subject_entity_word'].values[0],
        's_idx': sample['subject_entity_start_idx'].values[0],
        'e_idx': sample['subject_entity_end_idx'].values[0],
        'color': 'white',
        'bgcolor': '#96C4ED'
    }
    o_obj={
        'prefix': 'o',
        'word': sample['object_entity_word'].values[0],
        's_idx': sample['object_entity_start_idx'].values[0],
        'e_idx': sample['object_entity_end_idx'].values[0],    
        'color': 'white',
        'bgcolor': '#B19CD9'
    }

    text = label_highlight(
        'subject:object',
        s_obj=s_obj,
        o_obj=o_obj
    ) + '<br/><br/>'

    text += '<b>label</b>:<br/>'
    text += label_highlight(
        sample['label'].values[0],
        s_obj=s_obj,
        o_obj=o_obj
    ) + '<br/><br/>'

    text += '<b>sentence</b>:<br/>'
    text += sentence_highlight(
        sample['sentence'].values[0],
        s_obj=s_obj,
        o_obj=o_obj,
        max_len=max_len
    ) + '<br/><br/>'

    return HTML(text)