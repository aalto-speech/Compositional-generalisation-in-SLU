# Example usage

# config:
compound_type=scenario_action
prepped_data_dir=prepped_data/${compound_type}
comdiv=0.5
splits_dir=splits/${compound_type}_comdiv${comdiv}

# Prepare data:
python3 split_data.py \
    --prepare-data \
    --prepped-data-dir $prepped_data_dir \
    --compound-type $compound_type

# Split data:
sbatch split_data.slrm \
    --prepped-data-dir $prepped_data_dir \
    --compound-type $compound_type \
    --split-data \
    --comdiv $comdiv \
    --splitted-data-dir $splits_dir

# split data with a pre-defined test set:
iter=9000
mkdir $splits_dir/iter${iter}
cp  $splits_dir/test_set_iter${iter}_* $splits_dir/iter${iter}
mv $splits_dir/iter${iter}/test_set* $splits_dir/iter${iter}/test.txt
touch $splits_dir/iter${iter}/train.txt

comdiv=0
new_splits_dir=$splits_dir/iter${iter}/comdiv$comdiv
presplit=$splits_dir/iter${iter}

sbatch split_data.slrm \
    --prepped-data-dir $prepped_data_dir \
    --compound-type $compound_type \
    --split-data \
    --comdiv $comdiv \
    --splitted-data-dir $new_splits_dir \
    --from-presplit $presplit


# Prepare speechbrain data:
new_iter=8000
python split_data.py \
    --prepare-sb-data \
    --splitted-data-ids $new_splits_dir/{train,test}_set_iter${new_iter}_*.txt \
    --train-dev-split 0.9 \
    --prepped-data-dir $prepped_data_dir \
    --sb-data-dir $new_splits_dir/iter${new_iter}

# get split stats:
python split_data.py --analyse-splits --sb-data-dir $new_splits_dir/iter${new_iter} \
    >> $new_splits_dir/iter${new_iter}/stats_weight.txt
