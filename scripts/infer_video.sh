python scripts/infer_video.py \
  --checkpoint ./results_20220604/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=106_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.060.ckpt \
  --input /home/sigma/slam/matterport/test/matterport005_000_0610/cam_left \
  --output /home/sigma/slam/matterport/test/matterport005_000_0610/infer_video_use_0.060.ckpt \
  --sample_rate 1 \
  --data_type matterport \
  --ply_mode \
  --use_depth_gt \
  --use_pose_gt \
  --max_frames 500
