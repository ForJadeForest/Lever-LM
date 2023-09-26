
which python
# Baseline:
echo "=====================BEGIN TO BASELINE====================="
python train.py train=clip_ice_text \
                data_files="caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" \
                epochs=20 \
                val_step=80\
                ex_name="ab_baseline"\
                device_num=2\

echo "=====================BEGIN TO 10000SAMPLENUM====================="
python train.py train=clip_ice_text\
                data_files="caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:10000.json" \
                epochs=20 \
                val_step=160\
                ex_name="ab_10000sample_num"\
                device_num=2\

# candidate_method
echo "=====================BEGIN TO CANDIDATEMETHOD: RANDOM====================="
python train.py train=clip_ice_text\
                        data_files="caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" \
                        epochs=20 \
                        val_step=80\
                        ex_name="ab_random_sample"\
                        device_num=2\


echo "=====================BEGIN TO CANDIDATEMETHOD: RANDOM 128 CANDIDATE====================="
python train.py train=clip_ice_text\
                        data_files="caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" \
                        epochs=20 \
                        val_step=80\
                        ex_name="ab_random_sample_128candidate_set_num"\
                        device_num=2\


echo "=====================BEGIN TO CANDIDATEMETHOD: image_sim====================="
python train.py train=clip_ice_text\
                        data_files="caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" \
                        epochs=20 \
                        val_step=80\
                        ex_name="ab_image_simmethod"\
                        device_num=2\
