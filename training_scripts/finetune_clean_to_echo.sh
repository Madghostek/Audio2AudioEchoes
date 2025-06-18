rave train --name finetune_groove_to_echo \
	--db_path /home/tsroka/datasets/echo_75/groove_rave_echo_test_split/ \
	--out_path ./groove_finetune_echo \
	--config v2 \
	--config noise \
	--config snake \
	--augment compress \
	--save_every 100000 \
	--channels 1 \
	--max_steps 2500000 \
	--rand_pitch 0.75,1.25 \
	--rand_pitch_prob 0.9 \
	--ckpt groove_clean_pitch0.9/groove_clean_8854ce22c0/version_12/checkpoints/best.ckpt
 
