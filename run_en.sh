# ARG
# python main.py \
#     --seed 3759 \
#     --gpu 2 \
#     --lr 5e-5 \
#     --model_name ARG \
#     --language en \
#     --root_path /home/tzh/ARG/data/en \
#     --bert_path /home/tzh/model/bert-base-uncased \
#     --data_name en-arg \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 1.5 \
#     --llm_judgment_predictor_weight 1.0


# ARG
python main.py \
    --seed 4 \
    --gpu 1 \
    --lr 5e-5 \
    --model_name ARG \
    --language en \
    --root_path /home/tzh/ARG/data/en \
    --bert_path /home/tzh/model/bert-base-uncased \
    --data_name 测试种子 \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 1.5 \
    --llm_judgment_predictor_weight 1.0

# ARG-D
# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-5 \
#     --model_name ARG-D \
#     --language en \
#     --root_path /home/tzh/ARG/data/en \
#     --bert_path /home/tzh/model/bert-base-uncased \
#     --data_name en-argd \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 1.5 \
#     --llm_judgment_predictor_weight 1.0 \
#     --kd_loss_weight 1.0 \
#     --teacher_path ./param_model/ARG_en-arg/1/parameter_bert.pkl



# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-5 \
#     --model_name SLM \
#     --language en \
#     --root_path /home/tzh/ARG/data/en \
#     --bert_path /home/tzh/model/bert-base-uncased \
#     --data_name en-SLM \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 1.5 \
#     --llm_judgment_predictor_weight 1.0 \


# 直接test 不train 为了尝试特征融合corrector写的slm纯test代码
# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-5 \
#     --model_name SLM \
#     --language en \
#     --root_path /home/tzh/ARG/data/en \
#     --bert_path /home/tzh/model/bert-base-uncased \
#     --data_name en-SLM \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 1.5 \
#     --llm_judgment_predictor_weight 1.0 \
#     --eval_mode 1 \
#     --eval_model_path /home/tzh/ARG/param_model/SLM_en-SLM/1/parameter_bert.pkl


 