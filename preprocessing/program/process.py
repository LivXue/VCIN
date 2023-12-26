import json

import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    splits = ['train_balanced', 'val_balanced', 'testdev_balanced', 'submission']
    for split in splits:
        pro = json.load(open("{}_program.json".format(split)))
        results = {}
        # reverse functions to put the program export at the 1st place
        for item in tqdm(pro):
            qid = item[5]
            p_len = len(item[3])
            item[3].reverse()
            adj = np.identity(9, dtype=int)[:p_len]
            for edge in item[4]:
                adj[p_len-1-edge[0]][p_len-1-edge[1]] = 1

            results[qid] = {'question': item[1], 'program': item[3], 'adj': adj.tolist()}

        with open("processed_{}_program.json".format(split), 'w') as f:
            json.dump(results, f)

