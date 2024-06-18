import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisualBertModel

from modules import GaussianEncoder
from exp_generator import ExpGenerator, LSTMGenerator, ProgramEncoder
from lxrt.entry import LXRTEncoder
from loss import exp_generative_loss, structure_bce, kl_div_gaussian


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


class VisualBert_REX(nn.Module):
    """
    Baseline method
    """
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=12, lang_dir=None, args=None, explainable=True):
        super(VisualBert_REX, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.explainable = explainable
        self.args = args
        base_model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        self.embedding = base_model.embeddings
        self.bert_encoder = base_model.encoder
        self.sent_cls = nn.Linear(768, self.nb_vocab + 1)
        self.ans_cls = nn.Linear(768, self.nb_answer)

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
        # self.att_drop = nn.Dropout(0.2)
        # self.fc_drop = nn.Dropout(0.2)

        if self.explainable:
            self.structure_gate = nn.Linear(self.hidden_size, 1)
            self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                                  requires_grad=False)

        for module in [self.embedding, self.bert_encoder]:
            for para in module.parameters():
                para.requires_grad = True  # fixed pretrained or not

    def create_att_mask(self, batch, ori_mask, device):
        visual_mask = torch.ones(batch, self.num_roi).to(device)
        mask = torch.cat((ori_mask, visual_mask,), dim=1)
        return mask

    def init_hidden_state(self, batch, device):
        init_word = torch.zeros(batch, self.hidden_size).to(device)
        init_att_h = torch.zeros(batch, self.hidden_size).to(device)
        init_language_h = torch.zeros(batch, self.hidden_size).to(device)
        return init_word, init_att_h, init_language_h

    def forward(self, img, box, text_input, token_type, attention_mask, exp=None, valid_mask=None, ans=None, structure_gate=None):
        embedding = self.embedding(input_ids=text_input, token_type_ids=token_type, visual_embeds=img)
        visual_mask = torch.ones(len(embedding), self.num_roi).cuda()
        concat_mask = torch.cat((attention_mask, visual_mask,), dim=1)
        # concat_mask = self.create_att_mask(len(embedding),attention_mask, embedding.device)

        # manually create attention mask for bert encoder (copy from PreTrainedModel's function)
        extended_mask = concat_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        bert_feat = self.bert_encoder(embedding, extended_mask)[0]
        visual_feat = bert_feat[:, -int(self.num_roi):, :].contiguous()
        cls_feat = bert_feat[:, 0]

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
        pred_att = []
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

            if self.explainable:
                structure_gate = torch.sigmoid(self.structure_gate(h_2))
                similarity = torch.bmm(h_2.unsqueeze(1), visual_feat.transpose(1, 2)).squeeze(1)
                similarity = F.softmax(similarity, dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.nb_vocab + 1,
                                                                               self.num_roi)
                sim_pred = torch.bmm(structure_mapping, similarity.unsqueeze(-1)).squeeze(-1)
                pred_word = structure_gate * pred_word + (1 - structure_gate) * sim_pred
                pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # schedule sampling
            prev_word = torch.max(pred_word, dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1), h_2, prev_word), dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp], dim=1)
        output_ans = self.ans_cls(cls_feat)
        # output_att = torch.cat([_.unsqueeze(1) for _ in pred_att],dim=1)
        output_gate = torch.cat([_ for _ in pred_gate], dim=1)

        if self.training:
            ans_loss = F.cross_entropy(output_ans, ans)
            exp_loss = self.args.alpha * exp_generative_loss(output_sent, exp, valid_mask)
            structure_loss = self.args.beta * structure_bce(output_gate, structure_gate)
            loss = ans_loss + exp_loss + structure_loss
            return loss, output_ans, output_sent
        else:
            return output_ans, output_sent


class LXMERT_REX(nn.Module):
    """
    Baseline method based on LXMERT
    """
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=12, lang_dir=None, args=None, explainable=True):
        super(LXMERT_REX, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.sent_cls = nn.Linear(768, self.nb_vocab + 1)
        self.ans_cls = nn.Linear(768, self.nb_answer)

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
        # self.att_drop = nn.Dropout(0.2)
        # self.fc_drop = nn.Dropout(0.2)

        if self.explainable:
            self.structure_gate = nn.Linear(self.hidden_size, 1)
            self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                                  requires_grad=False)

        #for module in [self.embedding, self.bert_encoder]:
        #    for para in module.parameters():
        #        para.requires_grad = True  # fixed pretrained or not

    def create_att_mask(self, batch, ori_mask, device):
        visual_mask = torch.ones(batch, self.num_roi).to(device)
        mask = torch.cat((ori_mask, visual_mask,), dim=1)
        return mask

    def init_hidden_state(self, batch, device):
        init_word = torch.zeros(batch, self.hidden_size).to(device)
        init_att_h = torch.zeros(batch, self.hidden_size).to(device)
        init_language_h = torch.zeros(batch, self.hidden_size).to(device)
        return init_word, init_att_h, init_language_h

    def forward(self, img, box, text_input, token_type, attention_mask, exp=None, valid_mask=None, ans=None, structure_gate=None):
        visual_mask = torch.ones(len(img), self.num_roi).cuda()

        feat_seq, cls_feat, _ = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                               attention_mask=attention_mask, visual_feats=(img, box),
                                               visual_attention_mask=visual_mask)
        que_feat = feat_seq[0]
        visual_feat = feat_seq[1]

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

            if self.explainable:
                structure_gate = torch.sigmoid(self.structure_gate(h_2))
                similarity = torch.bmm(h_2.unsqueeze(1), visual_feat.transpose(1, 2)).squeeze(1)
                similarity = F.softmax(similarity, dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.nb_vocab + 1,
                                                                               self.num_roi)
                sim_pred = torch.bmm(structure_mapping, similarity.unsqueeze(-1)).squeeze(-1)
                pred_word = structure_gate * pred_word + (1 - structure_gate) * sim_pred
                pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # schedule sampling
            prev_word = torch.max(pred_word, dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1), h_2, prev_word), dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp], dim=1)
        output_ans = self.ans_cls(cls_feat)
        output_gate = torch.cat([_ for _ in pred_gate], dim=1)

        if self.training:
            ans_loss = F.cross_entropy(output_ans, ans)
            exp_loss = self.args.alpha * exp_generative_loss(output_sent, exp, valid_mask)
            structure_loss = self.args.beta * structure_bce(output_gate, structure_gate)
            loss = ans_loss + exp_loss + structure_loss
            return loss, output_ans, output_sent
        else:
            return output_ans, output_sent


class VCIN(nn.Module):
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, nb_pro=2000, num_step=18, lang_dir=None, args=None, explainable=True):
        super(VCIN, self).__init__()
        self.nb_vocab = nb_vocab
        self.nb_pro = nb_pro
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.num_head = 4
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)     # "max_seq_length" not used
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                     nn.GELU(),
                                     nn.LayerNorm(768 * 2),
                                     nn.Linear(768 * 2, self.nb_answer))

        # language generator
        if explainable:
            self.exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)
            self.exp_var_feature = GaussianEncoder(768, 768)
            self.ans_cls = nn.Sequential(nn.Linear(768 * 2, 768 * 2),
                                         nn.GELU(),
                                         nn.LayerNorm(768 * 2),
                                         nn.Linear(768 * 2, self.nb_answer))

    def forward(self, img, box, text_input, token_type, attention_mask, pro=None, pro_adj=None, exp=None, valid_mask=None, ans=None, structure_gate=None):
        visual_mask = torch.ones(len(img), self.num_roi).to(img.device)
        #concat_mask = torch.cat((attention_mask, visual_mask,), dim=1)
        concat_mask = torch.cat(((1 - attention_mask).float().unsqueeze(1) * (-1e6), 
                                 (1 - visual_mask).float().unsqueeze(1) * (-1e6)), dim=-1)

        feat_seq, pooled_output, _ = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_mask)
        que_feat = feat_seq[0]
        visual_feat = feat_seq[1]
        cls_feat = pooled_output

        if self.explainable:
            mm_features = torch.cat((que_feat, visual_feat), dim=1)
            pred_pro, pred_output, structure_gates, exp_feature, pred_exp_feature = self.exp_generator(exp, visual_feat, mm_features, concat_mask)
            p_mean, p_var = self.exp_var_feature(exp_feature)
            q_mean, q_var, q_val = self.exp_var_feature(pred_exp_feature, sampling=4)
            ans_feat = torch.concat((cls_feat.unsqueeze(1).expand(-1, q_val.shape[1], -1), q_val), -1)
            output_ans_final = self.ans_cls(ans_feat)
        else:
            pred_pro, structure_gates, exp_feature, pred_exp_feature = None, None, None, None
            q_mean, q_var, p_mean, p_var = None, None, None, None
            output_ans_final = self.ans_cls(cls_feat)

        if self.training:
            if output_ans_final.ndim > 2:
                ans = ans.unsqueeze(1)
            if self.explainable:
                ans_loss = F.cross_entropy(output_ans_final.view(-1, output_ans_final.size(-1)), ans.repeat(1, output_ans_final.size(1)).view(-1))
                exp_loss = self.args.alpha * exp_generative_loss(pred_pro, exp, valid_mask)
                structure_loss = self.args.beta * structure_bce(structure_gates, structure_gate)
                kl_div = kl_div_gaussian(q_mean, q_var, p_mean, p_var)
                loss = ans_loss + exp_loss + structure_loss + kl_div
            else:
                ans_loss = F.cross_entropy(output_ans_final, ans)
                loss = ans_loss
            return loss.unsqueeze(0), output_ans_final.squeeze(1), pred_pro
        else:
            return output_ans_final.squeeze(1), pred_pro


class Pro_VCIN(nn.Module):
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, nb_pro=2000, num_step=18, lang_dir=None, args=None, explainable=True):
        super(Pro_VCIN, self).__init__()
        self.nb_vocab = nb_vocab
        self.nb_pro = nb_pro
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.num_head = 4
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)     # "max_seq_length" not used
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                     nn.GELU(),
                                     nn.LayerNorm(768 * 2),
                                     nn.Linear(768 * 2, self.nb_answer))

        # language generator
        if explainable:
            self.pro_encoder = ProgramEncoder(nb_pro, self.num_head, self.hidden_size)
            self.exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)
            self.exp_var_feature = GaussianEncoder(768, 768)
            self.ans_cls = nn.Sequential(nn.Linear(768 * 2, 768 * 2),
                                         nn.GELU(),
                                         nn.LayerNorm(768 * 2),
                                         nn.Linear(768 * 2, self.nb_answer))

    def forward(self, img, box, text_input, token_type, attention_mask, pro=None, pro_adj=None, exp=None, valid_mask=None, ans=None, structure_gate=None):
        visual_mask = torch.ones(len(img), self.num_roi).to(img.device)
        #concat_mask = torch.cat((attention_mask, visual_mask,), dim=1)

        feat_seq, pooled_output, hie_visn_feats = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_mask)
        que_feat = feat_seq[0]
        visual_feat = feat_seq[1]
        cls_feat = pooled_output

        if self.explainable:
            pro_features = self.pro_encoder(pro, pro_adj, hie_visn_feats)
            pro_mask = (pro[:, :, 0] == 0).float().unsqueeze(1) * (-1e6)
            concat_mask = torch.cat(((1 - attention_mask).float().unsqueeze(1) * (-1e6),
                                     (1 - visual_mask).float().unsqueeze(1) * (-1e6), pro_mask), dim=-1)
            mm_features = torch.cat((que_feat, visual_feat, pro_features), dim=1)
            pred_pro, pred_output, structure_gates, exp_feature, pred_exp_feature = self.exp_generator(exp, visual_feat, mm_features, pro_mask)
            p_mean, p_var = self.exp_var_feature(exp_feature)
            q_mean, q_var, q_val = self.exp_var_feature(pred_exp_feature, sampling=4)
            ans_feat = torch.concat((cls_feat.unsqueeze(1).expand(-1, q_val.shape[1], -1), q_val), -1)
            output_ans_final = self.ans_cls(ans_feat)
        else:
            pred_pro, structure_gates, exp_feature, pred_exp_feature = None, None, None, None
            q_mean, q_var, p_mean, p_var = None, None, None, None
            output_ans_final = self.ans_cls(cls_feat)

        if self.training:
            if output_ans_final.ndim > 2:
                ans = ans.unsqueeze(1)
            if self.explainable:
                ans_loss = F.cross_entropy(output_ans_final.view(-1, output_ans_final.size(-1)), ans.repeat(1, output_ans_final.size(1)).view(-1))
                exp_loss = self.args.alpha * exp_generative_loss(pred_pro, exp, valid_mask)
                structure_loss = self.args.beta * structure_bce(structure_gates, structure_gate)
                kl_div = kl_div_gaussian(q_mean, q_var, p_mean, p_var)
                loss = ans_loss + exp_loss + structure_loss + kl_div
            else:
                ans_loss = F.cross_entropy(output_ans_final, ans)
                loss = ans_loss
            return loss.unsqueeze(0), output_ans_final.squeeze(1), pred_pro
        else:
            return output_ans_final.squeeze(1), pred_pro
