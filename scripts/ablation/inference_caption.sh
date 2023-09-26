
which python
export CUDA_VISIBLE_DEVICES=0
Baseline:
echo "=====================BEGIN TO BASELINE====================="
python inference_flamingo_fast.py -m train=clip_ice_text \
                            ex_name="ab_baseline"\
                            test_iclm=true\
                            inference_bs=4\
                            iclm_path="result/model_cpk/ab_baseline/epoch\=0-step\=160-min_loss:6.0931.pth","result/model_cpk/ab_baseline/last-val_loss:6.4229.pth"


echo "=====================BEGIN TO 10000SAMPLENUM====================="
python inference_flamingo_fast.py -m train=clip_ice_text\
                            ex_name="ab_10000sample_num"\
                            test_iclm=true\
                            inference_bs=4\
                            iclm_path="result/model_cpk/ab_10000sample_num/epoch\=0-step\=320-min_loss:5.9055.pth","result/model_cpk/ab_10000sample_num/last-val_loss:6.3133.pth"

# candidate_method
echo "=====================BEGIN TO CANDIDATEMETHOD: RANDOM====================="
python inference_flamingo_fast.py -m train=clip_ice_text\
                            ex_name="ab_random_sample"\
                            test_iclm=true\
                            inference_bs=4\
                            iclm_path="result/model_cpk/ab_random_sample/epoch\=0-step\=160-min_loss:6.1619.pth","result/model_cpk/ab_random_sample/last-val_loss:6.8718.pth"


echo "=====================BEGIN TO CANDIDATEMETHOD: RANDOM 128 CANDIDATE====================="
python inference_flamingo_fast.py -m train=clip_ice_text\
                            ex_name="ab_random_sample_128candidate_set_num"\
                            test_iclm=true\
                            inference_bs=4\
                            iclm_path="result/model_cpk/ab_random_sample_128candidate_set_num/epoch\=0-step\=160-min_loss:6.1486.pth","result/model_cpk/ab_random_sample_128candidate_set_num/last-val_loss:6.8451.pth"



echo "=====================BEGIN TO CANDIDATEMETHOD: image_sim====================="
python inference_flamingo_fast.py -m train=clip_ice_text\
                            ex_name="ab_image_simmethod"\
                            test_iclm=true\
                            inference_bs=4\
                            iclm_path="result/model_cpk/ab_image_simmethod/epoch\=0-step\=160-min_loss:6.0541.pth","result/model_cpk/ab_image_simmethod/last-val_loss:6.4658.pth"

