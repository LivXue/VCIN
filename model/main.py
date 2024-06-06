import sys
import argparse
import os
import time
import json

sys.path.append('./util')
sys.path.append('./model')
sys.path.append('pycocoevalcap')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import logging

from dataloader import Batch_generator, Batch_generator_submission
from evaluation import organize_eval_data, construct_sentence, Grounding_Evaluator, Attribute_Evaluator, eval_consistency
from model import VisualBert_REX, LXMERT_REX, Pro_VCIN
from eval_exp import COCOEvalCap
from lxrt.optimization import BertAdam


parser = argparse.ArgumentParser(description='Multi-task learning experiment')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--anno_dir', type=str, default='./processed_data', help='Directory to annotations')
parser.add_argument('--sg_dir', type=str, default='../preprocessing/data/sceneGraphs', help='Directory to scene graph')
parser.add_argument('--ood_dir', type=str, default='./processed_data', help='Directory to annotations')
parser.add_argument('--lang_dir', type=str, default='./processed_data', help='Directory to preprocessed language files')
parser.add_argument('--img_dir', type=str, default='../preprocessing/data/extracted_features/features', help='Directory to image features')
parser.add_argument('--bbox_dir', type=str, default='../preprocessing/data/extracted_features/box', help='Directory to bounding box information')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for saving checkpoint')
parser.add_argument('--weights', type=str, default='./checkpoints/model_best_2023-11-26 13:25:09.pth', help='Trained model to be loaded (default: None)')
parser.add_argument('--epoch', type=int, default=12, help='Defining maximal number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='Defining initial learning rate (default: 4e-4)')
parser.add_argument('--batch_size', type=int, default=256, help='Defining batch size for training (default: 150)')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping to prevent gradient explode (default: 0.1)')
parser.add_argument('--max_qlen', type=int, default=30, help='Maximum length of question')
parser.add_argument('--max_exp_len', type=int, default=18, help='Maximum length of explanation')
parser.add_argument('--seq_len', type=int, default=32, help='Sequence length after padding')
parser.add_argument('--alpha', type=float, default=1, help='Balance factor for sentence loss')
parser.add_argument('--beta', type=float, default=1, help='Balance factor for structure gate loss')
parser.add_argument('--percentage', type=int, default=100, help='percentage of training data')
parser.add_argument('--explainable', type=bool, default=True, help='if generating explanations')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

NetSeed = 123
# random.seed(NetSeed)
np.random.seed(NetSeed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(NetSeed)  # set random seed
torch.cuda.manual_seed(NetSeed)  # set random seed
device = 'cuda'


def adjust_learning_rate(init_lr, optimizer, epoch):
    """adaptively adjust lr based on epoch"""
    lr = init_lr * (0.25 ** int((epoch + 1) / 8))  # previously 0.25/8

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train_data = Batch_generator(args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, args.max_qlen,
                                 args.max_exp_len, args.seq_len, 'train_balanced', args.percentage, args.explainable)
    val_data = Batch_generator(args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, 30,
                               args.max_exp_len, 35, 'val_balanced', explainable=args.explainable)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=6)
    valloader = DataLoader(val_data, batch_size=256, drop_last=False, shuffle=False, num_workers=6)
    ans2idx = train_data.ans2idx
    exp2idx = train_data.exp2idx
    pro2idx = train_data.pro2idx

    print("Loaded {} train samples, {} val samples!".format(len(train_data), len(val_data)))

    ood_data = dict()
    for keyword in ['all', 'head', 'tail']:
        ood_data[keyword] = json.load(open(os.path.join(args.ood_dir, 'ood_val_' + keyword + '.json')))

    # create mapping from index to word
    idx2ans = dict()
    for k in ans2idx:
        idx2ans[ans2idx[k]] = k
    idx2exp = dict()
    for k in exp2idx:
        idx2exp[exp2idx[k]] = k

    # initialize evaluator for visual grounding
    grounding_evaluator = Grounding_Evaluator(args.lang_dir, args.bbox_dir, args.sg_dir)

    # initialize evaluator for attributes
    attribute_evaluator = Attribute_Evaluator(args.lang_dir)

    model = Pro_VCIN(nb_answer=len(ans2idx), nb_vocab=len(exp2idx), nb_pro=len(pro2idx), num_step=args.max_exp_len,
                     lang_dir=args.lang_dir, explainable=args.explainable, args=args).to(device)
    #model = LXMERT_REX(nb_answer=len(ans2idx), nb_vocab=len(exp2idx), num_step=args.max_exp_len, lang_dir=args.lang_dir)

    model = nn.DataParallel(model)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
    #                       eps=1e-08, weight_decay=0)  # 1e-8
    optimizer = BertAdam(list(model.parameters()), lr=args.lr, warmup=0.1, t_total=len(trainloader) * args.epoch)

    def train():
        loss_history = []
        model.train()
        for batch_idx, (img, box, text_input, token_type, attention_mask, ans, exp, valid_mask, structure_gate, pro, pro_adj) in tqdm(enumerate(trainloader), total=len(trainloader)):
            img, box, text_input, token_type, attention_mask, ans, exp, valid_mask, structure_gate, pro, pro_adj = \
                (img.to(device), box.to(device), text_input.to(device), token_type.to(device), attention_mask.to(device), ans.to(device),
                 exp.to(device),  valid_mask.to(device), structure_gate.to(device), pro.to(device), pro_adj.to(device))
            optimizer.zero_grad()

            loss, pred_ans, pred_exp = model(img, box, text_input, token_type, attention_mask, pro, pro_adj, exp=exp,
                                             valid_mask=valid_mask, ans=ans, structure_gate=structure_gate)
            loss = loss.mean()
            loss.backward()
            loss_history.append(loss.item())

            if not args.clip == 0:
                clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

        epoch_loss = sum(loss_history) / len(loss_history)
        logging.info("loss = {}".format(epoch_loss))
        return model

    def test(epoch):
        model.eval()
        res = []
        gt = []
        qid_list = []
        total_acc = []
        answers = dict()
        exps = dict()

        for batch_idx, (img, box, text_input, token_type, attention_mask, qid, pro, pro_adj) in tqdm(enumerate(valloader), total=len(valloader)):
            img, box, text_input, token_type, attention_mask, pro, pro_adj = \
                (img.to(device), box.to(device), text_input.to(device), token_type.to(device), attention_mask.to(device),
                 pro.to(device), pro_adj.to(device))

            with torch.no_grad():
                raw_pred, pred = model(img, box, text_input, token_type, attention_mask, pro, pro_adj)

            # computing accuracy
            raw_pred = raw_pred.data.cpu().numpy()
            pred_ans = raw_pred.argmax(-1)
            if args.explainable:
                pred = pred.data.cpu().numpy()
                pred_exp = construct_sentence(pred, idx2exp)
                res.extend(pred_exp)

            converted_ans = []
            for idx, cur_id in enumerate(qid):
                qid_list.append(cur_id)
                converted_ans.append(idx2ans[pred_ans[idx]])
                answers[cur_id] = idx2ans[pred_ans[idx]]
                if args.explainable:
                    gt.append(val_data.explanation[cur_id])
                    exps[cur_id] = pred_exp[idx]
            ans_score, corr_ans = val_data.eval_qa_score(converted_ans, qid)
            total_acc.extend(ans_score)

        if args.explainable:
            grounding_score, record_grounding = grounding_evaluator.eval_grounding(res, qid_list)
            attr_score, record_attr = attribute_evaluator.eval_attribute(res, qid_list)

            res, gt, error_count = organize_eval_data(res, gt, qid_list)
            exp_evaluator = COCOEvalCap(gt, res)
            exp_score = exp_evaluator.evaluate()

        # compute ood VQA accuracy
        ood_acc = dict()
        for keyword in ood_data:
            ood_acc[keyword] = 0
            for idx, qid in enumerate(qid_list):
                if qid in ood_data[keyword]:
                    ood_acc[keyword] += total_acc[idx]
            ood_acc[keyword] /= len(ood_data[keyword])

        # compute normal VQA accuracy
        total_acc = np.mean(total_acc) * 100

        logging.info("Epoch {}: Val Acc = {}".format(epoch, total_acc))

        for keyword in ood_data:
            logging.info('OOD-' + keyword + ' accuracy {}'.format(ood_acc[keyword] * 100))

        if args.explainable:
            consistency = eval_consistency(exps, answers, val_data.question)
            logging.info('Consistency: ' + str(consistency * 100))
            for metric in exp_score:
                logging.info(metric + ': ' + str(exp_score[metric] * len(res) / (len(res) + error_count)))
            logging.info('Grounding Score ' + str(grounding_score * 100))
            for attribute in attr_score:
                logging.info('Recall for attribute ' + attribute + ': {}'.format(attr_score[attribute] * 100))

        return total_acc, answers, exps

    # main loop for training:
    val_score = 0
    for epoch in range(args.epoch):
        logging.info('Start training model: Epoch = {}'.format(epoch))
        #adjust_learning_rate(args.lr, optimizer, epoch)
        train()
        torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, 'model_{}.pth'.format(time_identifier)))

        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.epoch:
            cur_score, answers, exps = test(epoch)
            # save the best checkpoint
            if cur_score > val_score:
                torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, 'model_best_{}.pth'.format(time_identifier)))
                val_score = cur_score
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open('answers/answer_val_{}.json'.format(cur_time), 'w') as f:
                    json.dump(answers, f)
                logging.info('\033[32;0mSaved answer_val_{}.json!\033[0m'.format(cur_time))
                if args.explainable:
                    with open('explanations/explanation_val_{}.json'.format(cur_time), 'w') as f:
                        json.dump(exps, f)
                    logging.info('\033[32;0mSaved explanation_val_{}.json!\033[0m'.format(cur_time))
    evaluation()


def evaluation():
    test_data = Batch_generator(args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, 30,
                                args.max_exp_len, 32, 'testdev_balanced', explainable=args.explainable)
    testloader = DataLoader(test_data, batch_size=128, drop_last=False, shuffle=False, num_workers=6)

    ans2idx = test_data.ans2idx
    exp2idx = test_data.exp2idx
    pro2idx = test_data.pro2idx

    ood_data = dict()
    for keyword in ['all', 'head', 'tail']:
        ood_data[keyword] = json.load(open(os.path.join(args.ood_dir, 'ood_testdev_' + keyword + '.json')))

    # create mapping from index to word
    idx2ans = dict()
    for k in ans2idx:
        idx2ans[ans2idx[k]] = k

    idx2exp = dict()
    for k in exp2idx:
        idx2exp[exp2idx[k]] = k

    model = Pro_VCIN(nb_answer=len(ans2idx), nb_vocab=len(exp2idx), nb_pro=len(pro2idx), num_step=args.max_exp_len,
                         lang_dir=args.lang_dir, explainable=args.explainable, args=args).to(device)
    #model = LXMERT_REX(nb_answer=len(ans2idx), nb_vocab=len(exp2idx), num_step=args.max_exp_len, lang_dir=args.lang_dir)

    if args.mode == 'train':
        logging.info('Loading {}'.format(os.path.join(args.checkpoint_dir, 'model_best_{}.pth'.format(time_identifier))))
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'model_best_{}.pth'.format(time_identifier))))
    else:
        logging.info('Loading {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights))

    model.eval()
    qid_list = []
    total_acc = []
    answers = dict()

    for batch_idx, (img, box, text_input, token_type, attention_mask, qid, pro, pro_adj) in enumerate(testloader):
        img, box, text_input, token_type, attention_mask, pro, pro_adj = \
            img.cuda(), box.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda(), pro.cuda(), pro_adj.cuda()

        with torch.no_grad():
            pred_ans, pred = model(img, box, text_input, token_type, attention_mask, pro, pro_adj)

        # computing accuracy
        pred_ans = pred_ans.data.cpu().numpy()
        pred_ans = pred_ans.argmax(-1)

        converted_ans = []
        for idx, cur_id in enumerate(qid):
            qid_list.append(cur_id)
            converted_ans.append(idx2ans[pred_ans[idx]])
            answers[cur_id] = converted_ans[-1]
        ans_score, corr_ans = test_data.eval_qa_score(converted_ans, qid)
        total_acc.extend(ans_score)

    # compute ood VQA accuracy
    ood_acc = dict()
    for keyword in ood_data:
        ood_acc[keyword] = 0
        for idx, qid in enumerate(qid_list):
            if qid in ood_data[keyword]:
                ood_acc[keyword] += total_acc[idx]
        ood_acc[keyword] /= len(ood_data[keyword])

    # compute normal VQA accuracy
    total_acc = np.mean(total_acc) * 100

    logging.info('VQA Accuracy: %.2f' % total_acc)
    for keyword in ood_data:
        logging.info('OOD-%s accuracy: %.5f' % (keyword, ood_acc[keyword] * 100))
    with open('answers/answer_testdev_{}.json'.format(args.weights.split('/')[-1].split('.')[0]), 'w') as f:
        json.dump(answers, f)
    logging.info('\033[32;0mSaved answer_testdev_{}.json!\033[0m'.format(args.weights.split('/')[-1].split('.')[0]))


def submission():
    test_data = Batch_generator_submission(args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, 35, mode='submission_all')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=6)

    ans2idx = test_data.ans2idx
    exp2idx = test_data.exp2idx

    # create mapping from index to word
    idx2ans = dict()
    for k in ans2idx:
        idx2ans[ans2idx[k]] = k

    idx2exp = dict()
    for k in exp2idx:
        idx2exp[exp2idx[k]] = k

    model = Pro_VCIN(nb_answer=len(ans2idx), nb_vocab=len(exp2idx), num_step=args.max_exp_len,
                         lang_dir=args.lang_dir, explainable=args.explainable, args=args)
    model.load_state_dict(torch.load(args.weights))
    model = nn.DataParallel(model)
    model = model.cuda()

    model.eval()
    submission = []
    for batch_idx, (img, box, text_input, token_type, attention_mask, qid, pro, pro_adj) in tqdm(enumerate(testloader), total=len(testloader)):
        img, box, text_input, token_type, attention_mask, pro, pro_adj = \
            img.cuda(), box.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda(), pro.cuda(), pro_adj.cuda()

        with torch.no_grad():
            pred_ans, pred = model(img, box, text_input, token_type, attention_mask, pro, pro_adj)

        # computing accuracy
        pred_ans = pred_ans.data.cpu().numpy()
        pred_ans = pred_ans.argmax(-1)

        for idx, cur_id in enumerate(qid):
            cur_ans = idx2ans[pred_ans[idx]]
            tmp_res = {"questionId": str(cur_id), "prediction": cur_ans}
            submission.append(tmp_res)

    with open('./submission/submission.json', 'w') as f:
        json.dump(submission, f)


time0 = time.time()
if not os.path.exists("./log/"):
    os.mkdir("./log/")
time_identifier = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
path = os.path.join("./log/" + 'time=' + time_identifier)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(path + '.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('Saved ' + path + '.txt!')

logging.info(args)

if args.mode == 'train':
    main()
elif args.mode == 'eval':
    evaluation()
elif args.mode == 'submission':
    submission()
else:
    raise RuntimeError('Invalid mode selected')

cost_time = time.time() - time0
minutes = int(cost_time // 60)
logging.info('Costing time: {}m {:.2f}s'.format(minutes, cost_time - 60 * minutes))
