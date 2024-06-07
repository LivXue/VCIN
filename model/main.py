import sys
import argparse
import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from collections.abc import Sequence

sys.path.extend(['./util', './model', 'pycocoevalcap'])

from dataloader import Batch_generator, Batch_generator_submission
from evaluation import (
    organize_eval_data, construct_sentence, Grounding_Evaluator,
    Attribute_Evaluator, eval_consistency
)
from model import VisualBert_REX, LXMERT_REX, VCIN, Pro_VCIN
from eval_exp import COCOEvalCap
from lxrt.optimization import BertAdam

# Constants
SEED = 123
BATCH_SIZE = 256
NUM_WORKERS = 6


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-task learning experiment')
    parser.add_argument('--mode', type=str, default='train', help='Running mode (default: train)')
    parser.add_argument('--anno_dir', type=str, default='./processed_data', help='Directory to annotations')
    parser.add_argument('--sg_dir', type=str, default='../preprocessing/data/sceneGraphs',
                        help='Directory to scene graph')
    parser.add_argument('--ood_dir', type=str, default='./processed_data', help='Directory to annotations')
    parser.add_argument('--lang_dir', type=str, default='./processed_data',
                        help='Directory to preprocessed language files')
    parser.add_argument('--img_dir', type=str, default='../preprocessing/data/extracted_features/features',
                        help='Directory to image features')
    parser.add_argument('--bbox_dir', type=str, default='../preprocessing/data/extracted_features/box',
                        help='Directory to bounding box information')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for saving checkpoint')
    parser.add_argument('--weights', type=str, default='', help='Trained model to be loaded (default: None)')
    parser.add_argument('--epoch', type=int, default=12, help='Maximal number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping to prevent gradient explode (default: 1.0)')
    parser.add_argument('--max_qlen', type=int, default=30, help='Maximum length of question')
    parser.add_argument('--max_exp_len', type=int, default=18, help='Maximum length of explanation')
    parser.add_argument('--seq_len', type=int, default=32, help='Sequence length after padding')
    parser.add_argument('--alpha', type=float, default=1, help='Balance factor for sentence loss')
    parser.add_argument('--beta', type=float, default=1, help='Balance factor for structure gate loss')
    parser.add_argument('--percentage', type=int, default=100, help='Percentage of training data')
    parser.add_argument('--explainable', type=bool, default=True, help='If generating explanations')
    return parser.parse_args()


def configure_environment(seed=SEED):
    """Set random seed for reproducibility and configure CUDA environment."""
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'


def adjust_learning_rate(init_lr, optimizer, epoch):
    """Adaptively adjust the learning rate based on the current epoch."""
    lr = init_lr * (0.25 ** int((epoch + 1) / 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_ood_data(ood_dir):
    """Load out-of-distribution (OOD) data."""
    ood_data = {}
    for keyword in ['all', 'head', 'tail']:
        try:
            with open(os.path.join(ood_dir, f'ood_val_{keyword}.json')) as f:
                ood_data[keyword] = json.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: ood_val_{keyword}.json")
    return ood_data


def create_index_mapping(data):
    """Create a mapping from index to word."""
    return {v: k for k, v in data.items()}


def initialize_logging():
    """Initialize logging configuration."""
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    time_identifier = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_dir = "./log/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'time={time_identifier}.txt')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(f'Saved {log_file_path}!')
    return time_identifier


def main():
    # args = parse_arguments()
    # configure_environment()
    # time_identifier = initialize_logging()
    logging.info(args)

    try:
        train_data = Batch_generator(
            args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, args.max_qlen,
            args.max_exp_len, args.seq_len, 'train_balanced', args.percentage, args.explainable
        )
        val_data = Batch_generator(
            args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, 30,
            args.max_exp_len, 35, 'val_balanced', explainable=args.explainable
        )
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    trainloader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True,
                             num_workers=NUM_WORKERS)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS)
    ood_data = load_ood_data(args.ood_dir)

    idx2ans = create_index_mapping(train_data.ans2idx)
    idx2exp = create_index_mapping(train_data.exp2idx)
    grounding_evaluator = Grounding_Evaluator(args.lang_dir, args.bbox_dir, args.sg_dir)
    attribute_evaluator = Attribute_Evaluator(args.lang_dir)

    model = Pro_VCIN(
        nb_answer=len(train_data.ans2idx), nb_vocab=len(train_data.exp2idx), nb_pro=len(train_data.pro2idx),
        num_step=args.max_exp_len, lang_dir=args.lang_dir, explainable=args.explainable, args=args
    ).to('cuda')
    model = nn.DataParallel(model)
    optimizer = BertAdam(list(model.parameters()), lr=args.lr, warmup=0.1, t_total=len(trainloader) * args.epoch)

    def train():
        """Train the model."""
        model.train()
        loss_history = []
        for batch in tqdm(trainloader, total=len(trainloader)):
            batch = [item.to('cuda') for item in batch]
            img, box, text_input, token_type, attention_mask, ans, exp, valid_mask, structure_gate, pro, pro_adj = batch
            optimizer.zero_grad()
            loss, _, _ = model(img, box, text_input, token_type, attention_mask, pro, pro_adj, exp=exp,
                               valid_mask=valid_mask, ans=ans, structure_gate=structure_gate)
            loss = loss.mean()
            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss_history.append(loss.item())
        avg_loss = sum(loss_history) / len(loss_history)
        logging.info(f"Training Loss: {avg_loss}")
        return model

    def evaluate(epoch):
        """Evaluate the model on the validation set."""
        model.eval()
        total_acc = []
        answers = {}
        exps = {}
        res = []
        gt = []
        qid_list = []
        with torch.no_grad():
            for batch in tqdm(valloader, total=len(valloader)):
                batch = [item.to('cuda') if type(item) == torch.Tensor else item for item in batch]
                img, box, text_input, token_type, attention_mask, qid, pro, pro_adj = batch
                raw_pred, pred = model(img, box, text_input, token_type, attention_mask, pro, pro_adj)
                raw_pred = raw_pred.cpu().numpy()
                pred_ans = raw_pred.argmax(-1)
                if args.explainable:
                    pred = pred.cpu().numpy()
                    pred_exp = construct_sentence(pred, idx2exp)
                    res.extend(pred_exp)
                converted_ans = []
                for idx, cur_id in enumerate(qid):
                    qid_list.append(cur_id)
                    converted_ans.append(idx2ans[pred_ans[idx]])
                    answers[cur_id] = converted_ans[-1]
                    if args.explainable:
                        gt.append(val_data.explanation[cur_id])
                        exps[cur_id] = pred_exp[idx]
                ans_score, _ = val_data.eval_qa_score(converted_ans, qid)
                total_acc.extend(ans_score)
        total_acc = np.mean(total_acc) * 100
        logging.info(f"Epoch {epoch} Val Accuracy: {total_acc}")

        if args.explainable:
            grounding_score, _ = grounding_evaluator.eval_grounding(res, qid_list)
            attr_score, _ = attribute_evaluator.eval_attribute(res, qid_list)
            res, gt, _ = organize_eval_data(res, gt, qid_list)
            exp_evaluator = COCOEvalCap(gt, res)
            exp_score = exp_evaluator.evaluate()
            consistency = eval_consistency(exps, answers, val_data.question)
            logging.info(f"Consistency: {consistency * 100}")
            for metric, score in exp_score.items():
                logging.info(f"{metric}: {score}")
            logging.info(f"Grounding Score: {grounding_score * 100}")
            for attribute, score in attr_score.items():
                logging.info(f"Recall for attribute {attribute}: {score * 100}")

        return total_acc, answers, exps

    val_score = 0
    for epoch in range(args.epoch):
        logging.info(f'Starting training: Epoch {epoch}')
        train()
        torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, f'model_{time_identifier}.pth'))
        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.epoch:
            cur_score, answers, exps = evaluate(epoch)
            if cur_score > val_score:
                torch.save(model.module.state_dict(),
                           os.path.join(args.checkpoint_dir, f'model_best_{time_identifier}.pth'))
                val_score = cur_score
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(f'answers/answer_val_{cur_time}.json', 'w') as f:
                    json.dump(answers, f)
                logging.info(f'Saved answer_val_{cur_time}.json!')
                if args.explainable:
                    with open(f'explanations/explanation_val_{cur_time}.json', 'w') as f:
                        json.dump(exps, f)
                    logging.info(f'Saved explanation_val_{cur_time}.json!')
    evaluation()


def evaluation():
    """Evaluate the model on the test set."""
    # args = parse_arguments()
    # configure_environment()
    # time_identifier = initialize_logging()

    test_data = Batch_generator(
        args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, 30,
        args.max_exp_len, 32, 'testdev_balanced', explainable=args.explainable
    )
    testloader = DataLoader(test_data, batch_size=128, drop_last=False, shuffle=False, num_workers=NUM_WORKERS)

    ans2idx = test_data.ans2idx
    exp2idx = test_data.exp2idx
    ood_data = load_ood_data(args.ood_dir)

    idx2ans = create_index_mapping(ans2idx)
    idx2exp = create_index_mapping(exp2idx)

    model = Pro_VCIN(
        nb_answer=len(ans2idx), nb_vocab=len(exp2idx), nb_pro=len(test_data.pro2idx),
        num_step=args.max_exp_len, lang_dir=args.lang_dir, explainable=args.explainable, args=args
    ).to('cuda')

    if args.mode == 'train':
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f'model_best_{time_identifier}.pth')))
    else:
        model.load_state_dict(torch.load(args.weights))

    model.eval()
    answers = {}
    qid_list = []
    total_acc = []

    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader)):
            batch = [item.to('cuda') if type(item) == torch.Tensor else item for item in batch]
            img, box, text_input, token_type, attention_mask, qid, pro, pro_adj = batch
            pred_ans, _ = model(img, box, text_input, token_type, attention_mask, pro, pro_adj)
            pred_ans = pred_ans.cpu().numpy().argmax(-1)
            converted_ans = [idx2ans[pred] for pred in pred_ans]
            for idx, cur_id in enumerate(qid):
                qid_list.append(cur_id)
                answers[cur_id] = converted_ans[idx]
            ans_score, _ = test_data.eval_qa_score(converted_ans, qid)
            total_acc.extend(ans_score)

    total_acc = np.mean(total_acc) * 100
    logging.info(f'VQA Accuracy: {total_acc:.2f}')
    for keyword, ood_scores in ood_data.items():
        ood_acc = sum(total_acc[idx] for idx, qid in enumerate(qid_list) if qid in ood_scores) / len(ood_scores)
        logging.info(f'OOD-{keyword} accuracy: {ood_acc:.2f}')
    with open(f'answers/answer_testdev_{args.weights.split("/")[-1].split(".")[0]}.json', 'w') as f:
        json.dump(answers, f)
    logging.info(f'Saved answer_testdev_{args.weights.split("/")[-1].split(".")[0]}.json!')


def submission():
    """Prepare and save the submission file."""
    # args = parse_arguments()
    # configure_environment()
    # time_identifier = initialize_logging()

    test_data = Batch_generator_submission(
        args.img_dir, args.anno_dir, args.lang_dir, args.bbox_dir, 35, mode='submission_all'
    )
    testloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

    idx2ans = create_index_mapping(test_data.ans2idx)
    idx2exp = create_index_mapping(test_data.exp2idx)

    model = Pro_VCIN(
        nb_answer=len(test_data.ans2idx), nb_vocab=len(test_data.exp2idx), num_step=args.max_exp_len,
        lang_dir=args.lang_dir, explainable=args.explainable, args=args
    )
    model.load_state_dict(torch.load(args.weights))
    model = nn.DataParallel(model).cuda()

    model.eval()
    submission_data = []
    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader)):
            batch = [item.to('cuda') for item in batch]
            img, box, text_input, token_type, attention_mask, qid, pro, pro_adj = batch
            pred_ans, _ = model(img, box, text_input, token_type, attention_mask, pro, pro_adj)
            pred_ans = pred_ans.cpu().numpy().argmax(-1)
            for idx, cur_id in enumerate(qid):
                submission_data.append({"questionId": str(cur_id), "prediction": idx2ans[pred_ans[idx]]})

    with open('./submission/submission.json', 'w') as f:
        json.dump(submission_data, f)


if __name__ == '__main__':
    start_time = time.time()
    args = parse_arguments()
    configure_environment()
    time_identifier = initialize_logging()

    if args.mode == 'train':
        main()
    elif args.mode == 'eval':
        evaluation()
    elif args.mode == 'submission':
        submission()
    else:
        raise RuntimeError('Invalid mode selected')

    cost_time = time.time() - start_time
    minutes = int(cost_time // 60)
    logging.info(f'Costing time: {minutes}m {cost_time - 60 * minutes:.2f}s')
