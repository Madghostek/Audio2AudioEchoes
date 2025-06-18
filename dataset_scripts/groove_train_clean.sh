python3 prepare_echo_dataset_groove.py --dataset_path ~/datasets/raw/groove \
                                --output_path ~/datasets/echo_75/groove_rave_echo_test_split \
                                --temp_dir `mktemp -d` \
                                --use_rave 1 \
				--split test \
				--lags -1