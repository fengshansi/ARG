# ARG
# python main.py \
#     --seed 3759 \
#     --gpu 2 \
#     --lr 2e-5 \
#     --model_name ARG \
#     --language ch \
#     --root_path /home/tzh/ARG/data/zh \
#     --bert_path /home/tzh/model/chinese-bert-wwm-ext \
#     --data_name zh-arg \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 2.2 \
#     --llm_judgment_predictor_weight 1.8 


python main.py \
    --seed 0 \
    --gpu 2 \
    --lr 2e-5 \
    --model_name ARG \
    --language ch \
    --root_path /home/tzh/ARG/data/zh \
    --bert_path /home/tzh/model/chinese-bert-wwm-ext \
    --data_name 测试种子 \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 2.2 \
    --llm_judgment_predictor_weight 1.8 


# # ARG-D
# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-4 \
#     --model_name ARG-D \
#     --language ch \
#     --root_path /home/tzh/ARG/data/zh \
#     --bert_path /home/tzh/model/chinese-bert-wwm-ext \
#     --data_name zh-argd \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 2.2 \
#     --llm_judgment_predictor_weight 1.8 \
#     --kd_loss_weight 15 \
#     --teacher_path ./param_model/ARG_zh-arg/1/parameter_bert.pkl


# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-4 \
#     --model_name SLM \
#     --language ch \
#     --root_path /home/tzh/ARG/data/zh \
#     --bert_path /home/tzh/model/chinese-bert-wwm-ext \
#     --data_name zh-SLM \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 2.2 \
#     --llm_judgment_predictor_weight 1.8 \


# 直接test 不train 为了尝试特征融合corrector写的slm纯test代码
# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-4 \
#     --model_name SLM \
#     --language ch \
#     --root_path /home/tzh/ARG/data/zh \
#     --bert_path /home/tzh/model/chinese-bert-wwm-ext \
#     --data_name zh-SLM \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 2.2 \
#     --llm_judgment_predictor_weight 1.8 \
#     --eval_mode 1 \
#     --eval_model_path /home/tzh/ARG/param_model/SLM_zh-SLM/1/parameter_bert.pkl
