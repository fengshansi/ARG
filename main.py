import os
import argparse
import json
import pickle
from utils.utils import get_tensorboard_writer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="ARG")
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--max_len", type=int, default=170)
parser.add_argument("--early_stop", type=int, default=5)
parser.add_argument("--language", type=str, default="en")
parser.add_argument("--root_path", type=str)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=3759)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--co_attention_dim", type=int, default=300)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--save_log_dir", type=str, default="./logs")
parser.add_argument("--save_param_dir", type=str, default="./param_model")
parser.add_argument("--param_log_dir", type=str, default="./logs/param")

# extra parameter
parser.add_argument("--tensorboard_dir", type=str, default="./logs/tensorlog")
parser.add_argument("--bert_path", type=str, default="/path/to/bert-base-uncased")
parser.add_argument("--data_type", type=str, default="rationale")
parser.add_argument("--data_name", type=str)
parser.add_argument("--eval_mode", type=bool, default=False)
parser.add_argument("--eval_model_path", type=str, default="")

# model structure control
parser.add_argument("--expert_interaction_method", type=str, default="cross_attention")
parser.add_argument("--llm_judgment_predictor_weight", type=float, default=-1)
parser.add_argument("--rationale_usefulness_evaluator_weight", type=float, default=-1)

# distill config
parser.add_argument("--kd_loss_weight", type=float, default=1)
parser.add_argument("--teacher_path", type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(
    "lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {};".format(
        args.lr, args.model_name, args.batchsize, args.epoch, args.gpu
    )
)
print(
    "data_type: {}; data_path: {}; data_name: {};".format(
        args.data_type, args.root_path, args.data_name
    )
)

config = {
    "use_cuda": True,
    "seed": args.seed,
    "batchsize": args.batchsize,
    "max_len": args.max_len,
    "early_stop": args.early_stop,
    "language": args.language,
    "root_path": args.root_path,
    "weight_decay": 5e-5,
    "model": {
        "mlp": {"dims": [384], "dropout": 0.2},
        "llm_judgment_predictor_weight": args.llm_judgment_predictor_weight,
        "rationale_usefulness_evaluator_weight": args.rationale_usefulness_evaluator_weight,
        "kd_loss_weight": args.kd_loss_weight,
    },
    "emb_dim": args.emb_dim,
    "co_attention_dim": args.co_attention_dim,
    "lr": args.lr,
    "epoch": args.epoch,
    "model_name": args.model_name,
    "seed": args.seed,
    "save_log_dir": args.save_log_dir,
    "save_param_dir": args.save_param_dir,
    "param_log_dir": args.param_log_dir,
    "tensorboard_dir": args.tensorboard_dir,
    "bert_path": args.bert_path,
    "data_type": args.data_type,
    "data_name": args.data_name,
    "eval_mode": args.eval_mode,
    "teacher_path": args.teacher_path,
    "month": 1,
    "eval_model_path": args.eval_model_path,
}

if __name__ == "__main__":
    writer = get_tensorboard_writer(config)
    print("before in config")
    print(config)

    # slm进行eval
    if config["eval_mode"]:
        from models.arg import Trainer as ARGTrainer
        from models.argd import Trainer as ARGDTrainer
        from models.slm import Trainer as SLMTrainer

        a = SLMTrainer(config=config, writer=writer)
        train_loader, val_loader, test_future_loader = a.getloader()
        (
            the_metrics,
            label,
            test_pred,
            id,
            ae,
            accuracy,
        ) = a.predict(test_future_loader)
        # print(the_metrics)
        (
            the_metrics,
            label,
            train_pred,
            id,
            ae,
            accuracy,
        ) = a.predict(train_loader)
        (
            the_metrics,
            label,
            val_pred,
            id,
            ae,
            accuracy,
        ) = a.predict(val_loader)

        the_dict = {
            "train_slm_pred": train_pred,
            "val_slm_pred": val_pred,
            "test_slm_pred": test_pred,
        }

        with open(f'slm_{config["language"]}.pkl', "wb") as file:
            pickle.dump(the_dict, file)

    else:
        # =================================================================================
        # 网格搜索
        best_metric = Run(config=config, writer=writer).main()

        save_dir = "./logs/log"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, config["data_name"] + ".json")
        with open(save_path, "w") as file:
            json.dump(best_metric, file, indent=4, ensure_ascii=False)
    # =================================================================================
