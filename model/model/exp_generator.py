import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_hidden, h, dropout=.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_hidden)
        self.fc_k = nn.Linear(d_model, h * d_hidden)
        self.fc_v = nn.Linear(d_model, h * d_hidden)
        self.fc_o = nn.Linear(h * d_hidden, d_model)
        self.dropout = dropout
        self.res_dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.h = h

        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_uniform_(self.fc_q.weight)
        # nn.init.xavier_uniform_(self.fc_k.weight)
        # nn.init.xavier_uniform_(self.fc_v.weight)
        # nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.normal_(self.fc_q.weight, std=0.02)
        nn.init.normal_(self.fc_k.weight, std=0.02)
        nn.init.normal_(self.fc_v.weight, std=0.02)
        nn.init.normal_(self.fc_o.weight, std=0.02)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys=None, values=None, attention_mask=None, past=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). -inf indicates masking.
        :param past: Past features (past_k, past_v) = (2, b_s, h, nk, d_hidden)
        :return:
        """
        assert (keys is not None and values is not None) or past is not None, "No input keys or values!"

        b_s, nq = queries.shape[:2]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_hidden).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nq, d_hidden)
        if keys is not None:
            nk = keys.shape[1]
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_hidden).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nk, d_hidden)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_hidden).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nk, d_hidden)

        if past is not None and keys is not None:
            k = torch.cat((past[0], k), dim=-2).contiguous()
            v = torch.cat((past[1], v), dim=-2).contiguous()
        elif past is not None:
            k = past[0]
            v = past[1]

        if attention_mask is not None:
            attn_weight = softmax_one((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attention_mask.unsqueeze(1), dim=-1)
            attn_weight = torch.dropout(attn_weight, self.dropout, train=self.training)
            out = attn_weight @ v
            #out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1), dropout_p=self.dropout)
        else:
            attn_weight = softmax_one((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            attn_weight = torch.dropout(attn_weight, self.dropout, train=self.training)
            out = attn_weight @ v
            #out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_hidden)  # (b_s, nq, h*d_hidden)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = self.res_dropout(out)

        present = torch.stack((k, v))
        return out, present


def softmax_one(x, dim=-1, _stacklevel=3):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_hidden, h, dropout=.1):
        super(TransformerLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_hidden, h, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_model),
                                          nn.GELU(),
                                          nn.Linear(d_model, d_model),
                                          nn.Dropout(dropout))
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, input, mask=None, past=None):
        normed_input = self.ln_1(input)
        feature, present = self.self_att(normed_input, normed_input, normed_input, mask, past=past)
        feature = input + feature
        ml_feature = self.feed_forward(self.ln_2(feature))
        feature = ml_feature + feature

        return feature, present


class BiTransformer(nn.Module):
    def __init__(self, nb_vocab, id2visual, num_head, d_model, dropout=.1, use_position_emb=False, max_seq_len=18):
        super(BiTransformer, self).__init__()
        self.id2visual = id2visual
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab + 2, embedding_dim=d_model, padding_idx=0)
        self.transformer_layer1 = TransformerLayer(d_model, d_model // 2, num_head, dropout=dropout)
        self.transformer_layer2 = TransformerLayer(d_model, d_model // 2, num_head, dropout=dropout)

        self.use_position_emb = use_position_emb
        if self.use_position_emb:
            tmp = torch.Tensor(max_seq_len, d_model)
            nn.init.xavier_normal_(tmp)
            self.position_emb = nn.Parameter(tmp.unsqueeze(0))

    def token_emb(self, ids, visual_feat):
        if ids.ndim == 1:
            ids = ids.unsqueeze(1)

        textual_emb = self.exp_embed(ids)
        visual_ids = self.id2visual(ids)
        word_mask = visual_ids.sum(-1, keepdims=True).bool()
        visual_emb = torch.bmm(visual_ids, visual_feat)

        if self.use_position_emb:
            return textual_emb.masked_fill(word_mask, 0) + visual_emb + self.position_emb[:, :ids.shape[1]]
        else:
            return textual_emb.masked_fill(word_mask, 0) + visual_emb

    def forward(self, pre_words, img_features):
        word_features = self.token_emb(pre_words, img_features)
        word_features, _ = self.transformer_layer1(word_features)
        word_features, _ = self.transformer_layer2(word_features)

        return word_features


class ProgramEncoder(nn.Module):
    def __init__(self, nb_pro, num_head, d_model, dropout=.1):
        super(ProgramEncoder, self).__init__()
        self.pro_embed = nn.Embedding(num_embeddings=nb_pro, embedding_dim=300, padding_idx=0)
        self.input = nn.Linear(8 * 300, d_model)
        self.n_layer = 6
        self.ssa_layers = nn.ModuleList(
            [TransformerLayer(d_model, d_model // 2, num_head, dropout=dropout) for _ in range(self.n_layer)])
        self.cma_layers = nn.ModuleList(
            [MultiHeadAttention(d_model, d_model // 2, num_head, dropout=dropout) for _ in range(self.n_layer)])

    def forward(self, program, pro_adj, img_features):
        if not isinstance(img_features, list):
            img_features = [img_features] * self.n_layer
        bs = program.size(0)
        length = program.size(1)

        pro_adj = (1 - pro_adj) * (-1e6)
        pro_feature = self.pro_embed(program).view(bs, length, -1)
        pro_feature = self.input(pro_feature)

        for i, (ssa_layer, cma_layer) in enumerate(zip(self.ssa_layers, self.cma_layers)):
            pro_feature, _ = ssa_layer(pro_feature, pro_adj)
            vis_level = len(img_features) - self.n_layer + i
            cma_feature, _ = cma_layer(pro_feature, img_features[vis_level], img_features[vis_level])
            pro_feature = pro_feature + cma_feature

        return pro_feature


class ExpGenerator(nn.Module):
    def __init__(self, num_roi, nb_vocab, lang_dir, max_len=18):
        super(ExpGenerator, self).__init__()
        self.num_roi = num_roi
        self.max_len = max_len
        self.nb_vocab = nb_vocab
        self.hidden_size = 768
        self.num_head = 4
        self.drop_prob_lm = 0.1

        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab + 2, embedding_dim=self.hidden_size, padding_idx=0)

        self.structure_gate = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.GELU(),
                                            nn.LayerNorm(self.hidden_size),
                                            nn.Linear(self.hidden_size, 1))
        self.structure_mapping = nn.Parameter(
            torch.load(os.path.join(lang_dir, 'structure_mapping.pth')).T.contiguous(), requires_grad=False)

        tmp = torch.Tensor(max_len + 1, self.hidden_size)
        nn.init.xavier_normal_(tmp)
        self.position_emb = nn.Parameter(tmp.unsqueeze(0))

        self.n_layers = 4
        self.trans_layers = nn.ModuleList([TransformerLayer(self.hidden_size, self.hidden_size // 2, self.num_head,
                                                            dropout=self.drop_prob_lm) for _ in range(self.n_layers)])
        self.cross_layers = nn.ModuleList([MultiHeadAttention(self.hidden_size, self.hidden_size // 2, self.num_head,
                                                              dropout=self.drop_prob_lm) for _ in range(self.n_layers)])
        self.output_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 2),
                                           nn.GELU(),
                                           nn.LayerNorm(self.hidden_size * 2),
                                           nn.Linear(self.hidden_size * 2, nb_vocab + 1))

        self.id2visual = nn.Embedding(num_embeddings=nb_vocab + 2, embedding_dim=num_roi, padding_idx=0,
                                      _weight=torch.cat((torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                                         torch.zeros(1, num_roi)), dim=0))
        self.id2visual.weight.requires_grad = False

        self.exp_feature_trans = BiTransformer(nb_vocab, self.id2visual, self.num_head, self.hidden_size, dropout=.1,
                                               use_position_emb=True, max_seq_len=max_len + 1)

    def token_emb(self, ids, visual_feat, pos_shift=0):
        if ids.ndim == 1:
            ids = ids.unsqueeze(1)

        textual_emb = self.exp_embed(ids)
        visual_ids = self.id2visual(ids)
        word_mask = visual_ids.sum(-1, keepdims=True).bool()
        visual_emb = torch.bmm(visual_ids, visual_feat)

        return textual_emb.masked_fill(word_mask, 0) + visual_emb + self.position_emb[:, pos_shift:ids.shape[1]+pos_shift]

    def forward(self, pre_words, question_features, img_features, concat_mask, pro_features, pro_mask):
        mm_features = pro_features
        #mm_features = torch.cat((question_features, img_features, pro_features), dim=1)
        concat_mask = pro_mask

        if self.training:
            #pre_words = torch.max(pre_words, dim=-1)[1]
            cls = torch.ones((len(img_features), 1), dtype=torch.long, device=img_features.device) * (self.nb_vocab + 1)
            pre_words = torch.cat((cls, pre_words), dim=1)[:, :self.max_len]
            word_features = self.token_emb(pre_words, img_features)  # self.exp_embed(pre_words)
            seq_len = pre_words.shape[1]
            exp_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0) * (-1e6)
            exp_mask = exp_mask.to(question_features.device)
            for i in range(self.n_layers):
                word_features, _ = self.trans_layers[i](word_features, exp_mask)
                att_features, _ = self.cross_layers[i](word_features, mm_features, mm_features,
                                                       attention_mask=concat_mask)
                word_features = word_features + att_features

            similarity = torch.bmm(word_features, img_features.transpose(1, 2))
            structure_gates = torch.sigmoid(self.structure_gate(word_features))
            similarity = F.softmax(similarity, dim=-1)
            structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.num_roi,
                                                                           self.nb_vocab + 1)
            r_mask = torch.bmm(torch.ones_like(similarity), structure_mapping).bool()
            sim_pred = torch.bmm(similarity, structure_mapping)
            pred_pro = torch.softmax(self.output_linear(word_features).masked_fill(r_mask, -np.inf), -1)
            pred_pro = structure_gates * pred_pro + (1 - structure_gates) * sim_pred
            pred_output = torch.max(pred_pro, dim=-1)[1]

            exp_feature = self.exp_feature_trans(pre_words, img_features)[:, 0]
            pred_exp_feature = self.exp_feature_trans(torch.cat((cls, pred_output), dim=1), img_features)[:, 0]
        else:
            pred_pro = []
            structure_gates = []
            pasts = [(None, None)] * self.n_layers
            pre_words = torch.ones((len(img_features), 1), dtype=torch.long, device=img_features.device) * (self.nb_vocab + 1)
            pred_word = pre_words
            for step in range(self.max_len):
                word_features = self.token_emb(pred_word, img_features, pos_shift=step)  # self.exp_embed(pre_words)

                for i in range(self.n_layers):
                    word_features, trans_present = self.trans_layers[i](word_features, past=pasts[i][0])
                    if pasts[i][1] is None:
                        att_features, cross_present = self.cross_layers[i](word_features, mm_features, mm_features,
                                                                           attention_mask=concat_mask)
                    else:
                        att_features, cross_present = self.cross_layers[i](word_features, attention_mask=concat_mask,
                                                                           past=pasts[i][1])
                    word_features = word_features + att_features
                    pasts[i] = (trans_present, cross_present)

                similarity = torch.bmm(word_features, img_features.transpose(1, 2))
                structure_gate = torch.sigmoid(self.structure_gate(word_features))
                similarity = F.softmax(similarity, dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.num_roi,
                                                                               self.nb_vocab + 1)
                r_mask = torch.bmm(torch.ones_like(similarity), structure_mapping).bool()
                sim_pred = torch.bmm(similarity, structure_mapping)
                cur_pred_pro = torch.softmax(self.output_linear(word_features).masked_fill(r_mask, -np.inf), -1)
                cur_pred_pro = structure_gate * cur_pred_pro + (1 - structure_gate) * sim_pred
                structure_gates.append(structure_gate)
                pred_pro.append(cur_pred_pro)
                pred_word = torch.max(cur_pred_pro.squeeze(1), dim=-1)[1]
                pred_word = pred_word.unsqueeze(1)
                pre_words = torch.cat((pre_words, pred_word), dim=1)

            pred_pro = torch.cat([_ for _ in pred_pro], dim=1)
            structure_gates = torch.cat([_ for _ in structure_gates], dim=1)
            pred_output = pre_words[:, 1:]

            exp_feature = None
            pred_exp_feature = self.exp_feature_trans(pre_words, img_features)[:, 0]

        return pred_pro, pred_output, structure_gates.squeeze(-1), exp_feature, pred_exp_feature


class GRU(nn.Module):
    """
    Gated Recurrent Unit without long-term memory
    """

    def __init__(self, input_size, embed_size=512):
        super(GRU, self).__init__()
        self.update_x = nn.Linear(input_size, embed_size, bias=True)
        self.update_h = nn.Linear(embed_size, embed_size, bias=True)
        self.reset_x = nn.Linear(input_size, embed_size, bias=True)
        self.reset_h = nn.Linear(embed_size, embed_size, bias=True)
        self.memory_x = nn.Linear(input_size, embed_size, bias=True)
        self.memory_h = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, x, state):
        z = torch.sigmoid(self.update_x(x) + self.update_h(state))
        r = torch.sigmoid(self.reset_x(x) + self.reset_h(state))
        mem = torch.tanh(self.memory_x(x) + self.memory_h(torch.mul(r, state)))
        state = torch.mul(1 - z, state) + torch.mul(z, mem)
        return state


class LSTMGenerator(nn.Module):
    def __init__(self, num_roi, nb_vocab, lang_dir, max_len=12):
        super(LSTMGenerator, self).__init__()
        self.num_roi = num_roi
        self.nb_vocab = nb_vocab
        self.img_size = 2048
        self.hidden_size = 768
        self.num_step = max_len

        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab + 1, embedding_dim=self.hidden_size, padding_idx=0)

        # attentive RNN
        self.att_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.att_v = nn.Linear(self.img_size, self.hidden_size)
        self.att_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.att = nn.Linear(self.hidden_size, 1)
        self.att_rnn = GRU(3 * self.hidden_size, self.hidden_size)

        # language RNN
        self.q_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_fc = nn.Linear(self.img_size, self.hidden_size)

        self.language_rnn = GRU(2 * self.hidden_size, self.hidden_size)
        self.language_fc = nn.Linear(self.hidden_size, nb_vocab + 1)
        self.structure_gate = nn.Linear(self.hidden_size, 1)
        self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                              requires_grad=False)

    def forward(self, img, cls_feat, visual_feat):
        # pre-computed features for attention computation
        v_att = torch.tanh(self.att_v(img))
        q_att = torch.tanh(self.att_q(cls_feat))
        q_att = q_att.view(q_att.size(0), 1, -1)
        fuse_feat = torch.mul(v_att, q_att.expand_as(v_att))

        # pre-compute features for language prediction
        q_enc = torch.tanh(self.q_fc(cls_feat))

        # initialize hidden state
        prev_word = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        # h_1 = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        h_2 = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        # prev_word, h_1, h_2 = self.init_hidden_state(len(fuse_feat), fuse_feat.device)

        # loop for explanation generation
        pred_exp = []
        pred_gate = []
        x_1 = torch.cat((fuse_feat.mean(1), h_2, prev_word), dim=-1)
        for i in range(self.num_step):
            # attentive RNN
            h_1 = self.att_rnn(x_1, h_2)
            att_h = torch.tanh(self.att_h(h_1).unsqueeze(1).expand_as(fuse_feat) + fuse_feat)
            att = F.softmax(self.att(att_h), dim=1)  # with dropout
            # pred_att.append(att.squeeze(1))

            # use separate layers to encode the attended features
            att_x = torch.bmm(att.transpose(1, 2).contiguous(), img).squeeze()
            v_enc = torch.tanh(self.v_fc(att_x))
            fuse_enc = torch.mul(v_enc, q_enc)

            x_2 = torch.cat((fuse_enc, h_1), dim=-1)

            # language RNN
            h_2 = self.language_rnn(x_2, h_2)
            pred_word = F.softmax(self.language_fc(h_2), dim=-1)  # without dropout

            structure_gate = torch.sigmoid(self.structure_gate(h_2))
            similarity = torch.bmm(h_2.unsqueeze(1), visual_feat.transpose(1, 2)).squeeze(1)
            similarity = F.softmax(similarity, dim=-1)
            structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.nb_vocab + 1,
                                                                           self.num_roi)
            sim_pred = torch.bmm(structure_mapping, similarity.unsqueeze(-1)).squeeze(-1)
            pred_word = structure_gate * pred_word + (1 - structure_gate) * sim_pred
            pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # sampling
            prev_word = torch.max(pred_word, dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1), h_2, prev_word), dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp], dim=1)
        output_gate = torch.cat([_ for _ in pred_gate], dim=1)

        return output_sent, None, output_gate
