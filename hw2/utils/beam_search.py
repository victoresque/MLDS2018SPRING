import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from preprocess.embedding import OneHotEmbedder


def beam_search(model, embedder, in_seq, seq_len, beam_size):
    encoder = model.encoder
    decoder = model.decoder
    bos = embedder.encode_word('<BOS>')
    emb_size = bos.shape[0]
    dec_in = Variable(torch.FloatTensor(np.array([bos])))
    dec_in = dec_in.view(1,*dec_in.size())
    
    enc_out, hidden = encoder(in_seq)
    if model.attn:
        hiddens = (hidden, hidden)
        z_0 = model.z_0
        z_batch = z_0.repeat(enc_out.size(1), 1)
        result = np.zeros((1,0,emb_size))
        targ_idx = np.array([enc_out.size(0)/2])
    
        stack0, stack1 = [], [(1, dec_in, z_batch, hiddens, result)]
        for i in range(seq_len):
            stack0 = stack1
            stack1 = []
            while stack0:
                prob, dec_in, z_batch, hiddens, result = stack0.pop()
                (out_seq, z_batch), hiddens = decoder(enc_out, dec_in, z_batch, hiddens, 1, targ_idx)
                out_prob_flatten = F.softmax(out_seq, dim=-1).data.cpu().numpy().flatten()
                out_prob_flatten_argsorted = np.flip(out_prob_flatten.argsort(axis=0), axis=0)
                for j in range(beam_size):
                    word_idx = out_prob_flatten_argsorted[j]
                    dec_in = np.zeros((1,1,emb_size))
                    dec_in[0][0][word_idx] = 1
                    node_result = np.concatenate((result, dec_in), axis=1)
                    
                    dec_in = Variable(torch.FloatTensor(dec_in))
                    if next(model.parameters()).is_cuda: dec_in = dec_in.cuda()
                    stack1.append((prob*out_prob_flatten[word_idx], dec_in, z_batch, hiddens, node_result))
            sorted(stack1, key=lambda x: x[0], reverse=True)
            stack1 = stack1[:beam_size]
    """
    for i in stack1:
        print(embedder.decode_lines(i[4]))
    """
    return stack1[0][4]
