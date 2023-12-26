import os.path
from tkinter import *
import json
import numpy as np
import cv2
import random


obj_info = json.load(open('../preprocessing/obj_info.json'))
box_p = '../preprocessing/data/extracted_features/box/'
que = json.load(open('processed_data/val_balanced_questions_clean.json'))
model_names = ['REX_LXMERT', 'REX', 'VCIN', 'Pro_VCIN']
exps = ['explanations/explanation_val_{}.json'.format(mname) for mname in model_names]
exps = [json.load(open(path)) for path in exps]
#exps[0] = json.load(open("processed_data/converted_explanation_val_balanced.json"))     # for debug
anss = ['answers/answer_val_{}.json'.format(mname) for mname in model_names]
anss = [json.load(open(path)) for path in anss]

# colors for 36 object boxes
color = [(0, 0, 128),       (0, 191, 255),      (0, 100, 0),        (0, 255, 0),        (189, 183, 107),    (255, 255, 0),
         (255, 215, 0),     (184, 134, 11),     (188, 143, 143),    (205, 92, 92),      (205, 133, 63),     (165, 42, 42),
         (255, 160, 122),   (255, 165, 0),      (255, 99, 71),      (255, 0, 0),        (255, 20, 147),     (176, 48, 96),
         (255, 0, 255),     (221, 160, 221),    (153, 50, 204),     (147, 112, 219),    (216, 191, 216),    (139, 131, 120),
         (255, 222, 173),   (255, 250, 205),    (240, 255, 240),    (205, 193, 197),    (0, 255, 255),      (0, 139, 139),
         (82, 139, 139),    (193, 255, 193),    (69, 139, 0),       (162, 205, 90),     (255, 246, 143),    (255, 181, 197)]


def process_exp(qid, mid):
    q = que[qid]
    img = cv2.imread('../../images/' + q['imageId'] + '.jpg')
    box = np.load(box_p + q['imageId'] + '.npy')
    img_w, img_h = obj_info[q['imageId']]['img_w'], obj_info[q['imageId']]['img_h']
    exp = exps[mid][qid]
    ans = anss[mid][qid]
    qus = q['question']

    bid_list = []
    rid_list = []
    for word in exp.split():
        if word[0] == '#':
            bid = int(word[1:])
            if bid in bid_list:
                continue
            cv2.rectangle(img, (int(box[bid, 0]), int(box[bid, 3])), (int(box[bid, 2]), int(box[bid, 1])), color[bid], 2)
            cv2.putText(img, word, (int(box[bid, 0]), int(box[bid, 3])-6), cv2.FONT_HERSHEY_SIMPLEX, 1, color[bid], 2)
            bid_list.append(bid)

        if word[0] == '@':
            rid = int(word[1:])
            if rid in rid_list:
                continue
            x = (rid - 1) % 4
            y = (rid - 1 - x) / 4
            x = x * img_w / 4
            y = y * img_h / 4
            cv2.rectangle(img, (int(x), int(y)), (int(x + img_w / 4), int(y + img_h / 4)), color[rid], 2)
            cv2.putText(img, word, (int(x), int(y + img_h / 4) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, color[rid], 2)
            rid_list.append(rid)

    return qus, img, exp, ans


def sampling(qid_list, sampling_num):
    shuffled_qid = random.sample(qid_list, len(qid_list))
    num_model = len(model_names)
    samping_list = []
    sampled_num = 0
    for qid in shuffled_qid:
        q = que[qid]
        img_path = '../../images/' + q['imageId'] + '.jpg'      # path of images

        # Confirm test image exists
        if not os.path.exists(img_path):
            continue

        exp_equal = sum([exps[mid][qid] == exps[mmid][qid] for mmid in range(num_model) for mid in
                         range(mmid + 1, num_model)]) == num_model * (num_model - 1) / 2
        ans_equal = sum([anss[mid][qid] == anss[mmid][qid] for mmid in range(num_model) for mid in
                         range(mmid + 1, num_model)]) == num_model * (num_model - 1) / 2

        # Confirm predictions are not identical
        if not exp_equal or not ans_equal:
            samping_list.append(qid)
            sampled_num += 1

        if sampled_num >= sampling_num:
            break

    assert len(samping_list) == sampling_num, "{} samples to be sampled but {} are sampled!".format(sampling_num, len(samping_list))
    return samping_list


if __name__ == '__main__':
    random.seed(113)
    test_num = 250  # number of test samples
    ind = -1    # index of current question
    mind = 0    # index of evaluated model
    qid_list = list(exps[0].keys())
    test_list = sampling(qid_list, test_num)
    num_model = len(exps)
    mid_shuffle = random.sample(list(range(num_model)), num_model)
    #mid_shuffle = list(range(num_model))    # for debug


    # score records
    vis = [dict() for _ in range(num_model)]
    tex = [dict() for _ in range(num_model)]

    top = Tk()
    top.title('EVQA Evaluation')
    top.geometry('400x400')
    top.resizable(width=True, height=True)

    frame1 = Frame(top)
    frame1.grid(row=0, column=0, sticky='w')
    que_title_box = Message(frame1, text="Question: ", width=400, justify=LEFT, font=20)
    que_title_box.pack(side='left')

    frame2 = Frame(top)
    frame2.grid(row=1, column=0, sticky='w')
    que_var = StringVar()
    quevar_title_box = Message(frame2, textvariable=que_var, relief=GROOVE, width=400, font=18)
    quevar_title_box.pack()

    frame3 = Frame(top)
    frame3.grid(row=2, column=0, sticky='w')
    exp_title_box = Message(frame3, text="Explanation: ", width=400, justify=LEFT, font=20)
    exp_title_box.pack(side='left')

    frame4 = Frame(top)
    frame4.grid(row=3, column=0, sticky='w')
    exp_var = StringVar()
    exp_var_box = Message(frame4, textvariable=exp_var, relief=GROOVE, width=400, font=18)
    exp_var_box.pack()

    frame5 = Frame(top)
    frame5.grid(row=4, column=0, sticky='w')
    ans_title_box = Message(frame5, text="Answer: ", width=400, justify=LEFT, font=20)
    ans_title_box.pack(side='left')

    frame6 = Frame(top)
    frame6.grid(row=5, column=0, sticky='w')
    ans_var = StringVar()
    ans_var_box = Message(frame6, textvariable=ans_var, relief=GROOVE, width=400, font=18)
    ans_var_box.pack()

    frame7 = Frame(top)
    frame7.grid(row=6, column=0, sticky='w')
    vis_title_box = Message(frame7, text="Visual Consistency: ", width=400, justify=LEFT, font=20)
    vis_title_box.pack(side='left')

    frame8 = Frame(top)
    frame8.grid(row=7, column=0, sticky='w')
    vis_var = IntVar()
    vis_but1 = Radiobutton(frame8, text="1", variable=vis_var, value=1, font=18)
    vis_but1.pack(side='left')
    vis_but2 = Radiobutton(frame8, text="2", variable=vis_var, value=2, font=18)
    vis_but2.pack(side='left')
    vis_but3 = Radiobutton(frame8, text="3", variable=vis_var, value=3, font=18)
    vis_but3.pack(side='left')
    vis_but4 = Radiobutton(frame8, text="4", variable=vis_var, value=4, font=18)
    vis_but4.pack(side='left')
    vis_but5 = Radiobutton(frame8, text="5", variable=vis_var, value=5, font=18)
    vis_but5.pack(side='left')

    frame9 = Frame(top)
    frame9.grid(row=8, column=0, sticky='w')
    tex_title_box = Message(frame9, text="Textual Consistency: ", width=800, justify=LEFT, font=20)
    tex_title_box.pack(side='left')

    frame10 = Frame(top)
    frame10.grid(row=9, column=0, sticky='w')
    tex_var = IntVar()
    tex_but1 = Radiobutton(frame10, text="1", variable=tex_var, value=1, font=18)
    tex_but1.pack(side='left')
    tex_but2 = Radiobutton(frame10, text="2", variable=tex_var, value=2, font=18)
    tex_but2.pack(side='left')
    tex_but3 = Radiobutton(frame10, text="3", variable=tex_var, value=3, font=18)
    tex_but3.pack(side='left')
    tex_but4 = Radiobutton(frame10, text="4", variable=tex_var, value=4, font=18)
    tex_but4.pack(side='left')
    tex_but5 = Radiobutton(frame10, text="5", variable=tex_var, value=5, font=18)
    tex_but5.pack(side='left')

    def set_message(qus, exp, ans):
        que_var.set(qus)
        exp_var.set(exp)
        ans_var.set(ans)
        completed_num = ind * num_model + mind if ind * num_model + mind >=0 else 0
        comp_var.set("Completed {}/{}".format(completed_num, test_num * num_model))

    def submit():
        global ind, mind, mid_shuffle
        # Update Counter
        vis_value = vis_var.get()
        tex_value = tex_var.get()
        qid = test_list[ind]
        mid = mid_shuffle[mind]
        if 0 <= ind < test_num:
            vis[mid][qid] = vis_value
            tex[mid][qid] = tex_value

        # Move to next model
        mind += 1
        # Move to next question
        if mind >= num_model or ind == -1:
            ind += 1
            mind = 0
            mid_shuffle = random.sample(list(range(num_model)), num_model)

        mid = mid_shuffle[mind]

        if ind >= test_num:
            s = 'Thanks for completing the evaluation!'
            set_message(s, s, s)
            if ind == test_num:
                ind += 1
                mean_vis = [sum([vis[model][question] for question in test_list]) / test_num for model in range(num_model)]
                mean_tex = [sum([tex[model][question] for question in test_list]) / test_num for model in range(num_model)]
                for model in range(num_model):
                    vis[model] = {'mean': mean_vis[model], **vis[model]}
                    tex[model] = {'mean': mean_tex[model], **tex[model]}

                json_output = {'models': model_names, 'visual_consistency': vis, 'textual_consistency': tex}
                json_output = json.dumps(json_output)
                with open('./log/human_results.json', 'w') as f:
                    f.write(json_output)
        else:
            qid = test_list[ind]
            qus, img, exp, ans = process_exp(qid, mid)
            set_message(qus, exp, ans)
            qid_var.set('qid: ' + qid)
            cv2.imshow('Image', img)
            #cv2.waitKey(0)

    frame13 = Frame(top)
    frame13.grid(row=12, column=0, sticky='w', padx=100)
    submit_but = Button(frame13, text='Submit', font=20, command=submit)
    submit_but.pack()

    frame14 = Frame(top)
    frame14.grid(row=13, column=0, sticky='w', padx=50)
    comp_var = StringVar()
    comp_box = Message(frame14, textvariable=comp_var, width=400, font=18)
    comp_box.pack()

    # for debug
    frame15 = Frame(top)
    frame15.grid(row=14, column=0, sticky='w', padx=50)
    qid_var = StringVar()
    qid_box = Message(frame15, textvariable=qid_var, width=400, font=18)
    qid_box.pack()

    start_text = 'Submit to start!'
    set_message(start_text, start_text, start_text)

    top.mainloop()
