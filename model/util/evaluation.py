import numpy as np
import os
import json

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()


def construct_sentence(pred, idx2exp):
    processed_exp = []
    for i in range(len(pred)):
        cur_exp = pred[i]
        cur_exp = np.argmax(cur_exp, axis=-1)

        tmp_exp = []
        for j in range(len(cur_exp)):
            if cur_exp[j] == 0:  # <EOS>
                break
            cur_word = idx2exp[cur_exp[j]]
            tmp_exp.append(cur_word)

        tmp_exp = ' '.join(tmp_exp)
        tmp_exp = tmp_exp if tmp_exp != '' else 'EMPTY'
        processed_exp.append(tmp_exp)
    # print(tmp_exp)

    return processed_exp


def organize_eval_data(pred, gt, qid_list):
    processed_data_pred = dict()
    processed_data_gt = dict()
    error_count = 0
    for i, qid in enumerate(qid_list):
        if gt[i] == '':  # no ground truth explanation
            continue
        elif pred[i] == 'EMPTY':  # empty prediction
            error_count += 1
            continue

        # format the prediction
        processed_data_pred[qid] = []
        tmp_pred = dict()
        tmp_pred['explanation'] = pred[i]
        tmp_pred['id'] = i
        processed_data_pred[qid].append(tmp_pred)

        # format the ground truth
        processed_data_gt[qid] = []
        tmp_gt = dict()
        tmp_gt['explanation'] = gt[i]
        tmp_gt['id'] = i
        processed_data_gt[qid].append(tmp_gt)

    return processed_data_pred, processed_data_gt, error_count


class Grounding_Evaluator:
    # evaluator for visual grounding
    def __init__(self, anno_dir, bbox_dir, scene_graph_dir):
        self.annotation = json.load(open(os.path.join(anno_dir, 'grounding_annotation.json')))
        self.upper_bound = json.load(open(os.path.join(anno_dir, 'grounding_upper_bound.json')))
        self.bbox_dir = bbox_dir
        self.scene_graph = json.load(open(os.path.join(scene_graph_dir, 'val_sceneGraphs.json')))

    def eval_grounding(self, prediction, qid_list):
        result = []
        record_grounding = dict()
        for idx, qid in enumerate(qid_list):
            if self.upper_bound[qid] == 0:
                continue
            img_id = self.annotation[qid]['imgid']
            width, height = self.annotation[qid]['width'], self.annotation[qid]['height']
            cur_bbox = np.load(os.path.join(self.bbox_dir, str(img_id) + '.npy'))
            cur_scene_graph = self.scene_graph[img_id]['objects']
            gt = np.zeros([height, width])
            for cur_obj in self.annotation[qid]['roi']:
                x1 = cur_scene_graph[cur_obj]['x']
                y1 = cur_scene_graph[cur_obj]['y']
                x2 = x1 + cur_scene_graph[cur_obj]['w']
                y2 = y1 + cur_scene_graph[cur_obj]['h']
            gt[y1:y2, x1:x2] = 1

            cur_pred = np.zeros([height, width])
            pred_pool = []
            for cur in prediction[idx].split(' '):
                if '#' in cur and cur[1:].isnumeric():
                    pred_pool.append(int(cur[1:]))
            for cur_idx in pred_pool:
                x1, y1, x2, y2 = cur_bbox[cur_idx]
                cur_pred[int(y1):int(y2), int(x1):int(x2)] = 1
            cur_iou = np.logical_and(cur_pred, gt).sum() / np.logical_or(cur_pred, gt).sum()
            if np.isnan(cur_iou):
                cur_iou = 0
            result.append(cur_iou / self.upper_bound[qid])
            record_grounding[qid] = (cur_iou / self.upper_bound[qid])

        return np.mean(result), record_grounding


class Attribute_Evaluator:
    # evaluator for visual grounding
    def __init__(self, anno_dir):
        self.attribute_eval_list = json.load(open(os.path.join(anno_dir, 'attribute_eval_list.json')))
        self.attr2cat = json.load(open(os.path.join(anno_dir, 'attr2cat.json')))
        self.attr_pool = json.load(open(os.path.join(anno_dir, 'attribute_bank.json')))

    def eval_attribute(self, prediction, qid_list):
        attr_eval = dict()
        record_performance = dict()  # record the recall rate on a sample basis for further analysis
        for cat in self.attr_pool:
            attr_eval[cat] = []

        for idx, qid in enumerate(qid_list):
            if qid not in self.attribute_eval_list:
                continue

            record_performance[qid] = dict()

            for attribute in self.attribute_eval_list[qid]:
                if self.attr2cat[attribute] not in record_performance[qid]:
                    record_performance[qid][self.attr2cat[attribute]] = []

                #if (attribute + ' ') in prediction[idx] or (' ' + attribute) in prediction[idx]:
                if attribute in prediction[idx]:
                    attr_eval[self.attr2cat[attribute]].append(1)
                    record_performance[qid][self.attr2cat[attribute]].append(1)
                else:
                    attr_eval[self.attr2cat[attribute]].append(0)
                    record_performance[qid][self.attr2cat[attribute]].append(0)

            for attr in record_performance[qid]:
                record_performance[qid][attr] = np.mean(record_performance[qid][attr])

        for cat in self.attr_pool:
            attr_eval[cat] = np.mean(attr_eval[cat])

        return attr_eval, record_performance


def eval_consistency(exp, ans, gt):
    valid_q = 0
    hit = 0
    for key in ans.keys():
        if gt[key]['answer'] in ['yes', 'no', 'color', 'material', 'shape']:
            continue
        else:
            valid_q += 1

        cur_ans = ans[key]
        cur_exp = ' ' + exp[key] + ' '
        if cur_ans + ' ' in cur_exp:
            hit += 1
        # to singular
        elif wnl.lemmatize(cur_ans, pos='n') + ' ' in cur_exp:
            hit += 1
        elif cur_ans in ['wooden', 'metallic'] and {'wooden': 'wood', 'metallic': 'metal'}[cur_ans] + ' ' in cur_exp:
            hit += 1
        #The image is split as:
        #┌───┬───┬───┬───┐
        #│@1 │@2 │@3 │@4 │
        #├───┼───┼───┼───┤
        #│@5 │@6 │@7 │@8 │
        #├───┼───┼───┼───┤
        #│@9 │@10│@11│@12│
        #├───┼───┼───┼───┤
        #│@13│@14│@15│@16│
        #└───┴───┴───┴───┘
        elif cur_ans == 'left' and ('@1' in cur_exp or '@2' in cur_exp or '@5' in cur_exp or '@6' in cur_exp or
                                    '@9' in cur_exp or '@10' in cur_exp or '@13' in cur_exp or '@14' in cur_exp):
            hit += 1
        elif cur_ans == 'right' and ('@3' in cur_exp or '@4' in cur_exp or '@7' in cur_exp or '@8' in cur_exp or
                                     '@11' in cur_exp or '@12' in cur_exp or '@15' in cur_exp or '@16' in cur_exp):
            hit += 1
        #else:
        #    print("Wrong {} | {}".format(exp[key], cur_ans))

    consistency = hit / valid_q
    return consistency
