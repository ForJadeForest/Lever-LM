
which python
# Baseline:
echo "=====================BEGIN TO BASELINE====================="
python generate_data.py beam_size=5 few_shot_num=2 candidate_set_num=64 sample_num=10000 only_y_loss=true random_sample_candidate_set=false gpu_ids="[0,1]"
# sample num for 1w. test the sample num effect. Only get the first 5k data to compare with other

# beam_size:
echo "=====================BEGIN TO BEAMSIZE====================="
python generate_data.py beam_size=10 few_shot_num=2 candidate_set_num=64 sample_num=5000 only_y_loss=true random_sample_candidate_set=false gpu_ids="[0,1]"

# candidate_method
echo "=====================BEGIN TO CANDIDATEMETHOD: RANDOM====================="
python generate_data.py beam_size=5 few_shot_num=2 candidate_set_num=64 sample_num=5000 only_y_loss=true random_sample_candidate_set=true gpu_ids="[0,1]"
echo "=====================BEGIN TO CANDIDATEMETHOD: SIM_IMAGE====================="
python generate_data.py beam_size=5 few_shot_num=2 candidate_set_num=64 sample_num=5000 only_y_loss=true random_sample_candidate_set=false sim_method="image" gpu_ids="[0,1]" 


echo "=====================BEGIN TO CANDIDATEMETHOD: RANDOM_BIG_CANDIDATE_NUM====================="
python generate_data.py beam_size=5 few_shot_num=2 candidate_set_num=128 sample_num=5000 only_y_loss=true random_sample_candidate_set=true gpu_ids="[0,1]" 

# only_y_loss
echo "=====================BEGIN TO Not Only Y Loss====================="
python generate_data.py beam_size=5 few_shot_num=2 candidate_set_num=128 sample_num=5000 only_y_loss=false random_sample_candidate_set=true gpu_ids="[0,1]" 
