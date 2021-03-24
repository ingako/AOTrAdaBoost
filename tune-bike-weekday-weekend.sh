#!/usr/bin/env bash

count=0

nohup src/main.py --max_samples 10000000 --data_format arff --generator_name agrawal --generator_traits abrupt --generator_seed 0 \
    -t 60 -c 120 -s --cd_kappa_threshold 0.0 --edit_distance_threshold 100 --reuse_rate_upper_bound 0.9 --reuse_rate_lower_bound 0.9 --reuse_window_size 0 --lossy_window_size 100000000 \
    --kappa_window 60 --poisson_lambda 1 --random_state 0 \
    --transfer_tree \
    --transfer_streams_paths 'data/bike/dc-weekday-source.arff;data/bike/weekend.arff' \
    --exp_code 'bike-weekday-weekend' \
    --boost_mode 'disable_transfer' &





for transfer_kappa_threshold in 0.0 0.05 0.1 0.15 ; do
for least_transfer_warning_period_instances_length in 50 100 150 200 250 300 350 400 500 ; do
for num_diff_distr_instances in 20 40 60 80 100 150 200 250 300 350 400 450 500 1000; do


nohup src/main.py --max_samples 10000000 --data_format arff --generator_name agrawal --generator_traits abrupt --generator_seed 0 \
    -t 60 -c 120 -s --cd_kappa_threshold 0.0 --edit_distance_threshold 100 --reuse_rate_upper_bound 0.9 --reuse_rate_lower_bound 0.9 --reuse_window_size 0 --lossy_window_size 100000000 \
    --kappa_window 60 --poisson_lambda 1 --random_state 0 \
    --transfer_tree \
    --transfer_streams_paths 'data/bike/dc-weekday-source.arff;data/bike/weekend.arff' \
    --exp_code 'bike-weekday-weekend' \
    --least_transfer_warning_period_instances_length $least_transfer_warning_period_instances_length   \
    --instance_store_size 5000 \
    --eviction_interval 1000000  \
    --num_diff_distr_instances $num_diff_distr_instances \
    --transfer_kappa_threshold $transfer_kappa_threshold \
    --boost_mode 'no_boost' &



for bbt_pool_size in 10 20 30 40 50 60 70 80 90 100 ; do # 10
for transfer_gamma in 1 2 3 4 5 6 7 8 9 10 20 ; do
for boost_type in ozaboost tradaboost ; do


nohup src/main.py --max_samples 10000000 --data_format arff --generator_name agrawal --generator_traits abrupt --generator_seed 0 \
    -t 60 -c 120 -s --cd_kappa_threshold 0.0 --edit_distance_threshold 100 --reuse_rate_upper_bound 0.9 --reuse_rate_lower_bound 0.9 --reuse_window_size 0 --lossy_window_size 100000000 \
    --kappa_window 60 --poisson_lambda 1 --random_state 0 \
    --transfer_tree \
    --transfer_streams_paths 'data/bike/dc-weekday-source.arff;data/bike/weekend.arff' \
    --exp_code 'bike-weekday-weekend' \
    --exp_code 'bike-weekday-weekend' \
    --least_transfer_warning_period_instances_length $least_transfer_warning_period_instances_length   \
    --instance_store_size 5000 \
    --eviction_interval 1000000  \
    --num_diff_distr_instances $num_diff_distr_instances \
    --bbt_pool_size $bbt_pool_size \
    --transfer_kappa_threshold $transfer_kappa_threshold \
    --boost_mode $boost_type  \
    --transfer_gamma $transfer_gamma &



count=$(($count+1))
if [ $(($count%37)) == 0 -a $count -ge 37 ] ; then
    wait $PIDS
    PIDS=""
fi

done
done
done
done
done
done
