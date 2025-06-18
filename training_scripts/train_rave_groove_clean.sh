rave train --name groove_clean --db_path \
	/home/tsroka/datasets/clean/groove_rave_train --out_path \
	./groove_clean_pitch0.9 \
	--config v2 \
	--config noise \
	--config snake \
	--augment compress \
	--save_every 100000 \
	--channels 1 \
	--max_steps 2000000 \
	--rand_pitch 0.75,1.25 \
	--rand_pitch_prob 0.9 \
	--ckpt ./groove_clean_pitch0.9/groove_clean_8854ce22c0/version_11/checkpoints/epoch-epoch=3382.ckpt 