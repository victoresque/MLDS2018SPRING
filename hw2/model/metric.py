import numpy as np
from datasets.MLDS_hw2_1_data.bleu_eval import BLEU as BLEU_


def bleu(y_input, y_target):
    # Modified from bleu_eval.py
    # TODO: modified to our format
    """ Target Format:

        test format:
            [{'caption': ['...', '...'], 'id': '...'}, ...]
        result format:
            {video_id: caption, ...}
    """
    test = y_target
    result = y_input
    bleus = []
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU_(result[item['id']], captions, True))
        bleus.append(score_per_video[0])
    average = sum(bleus) / len(bleus)
    return average
