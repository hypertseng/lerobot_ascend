Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# ln -s /usr/lib64/libffi.so.8 $CONDA_PREFIX/lib/libffi.so.8 # libffi不兼容需要在系统中装好兼容的版本，在链接到虚拟环境中

# msprof --output=./profiling_output python eval.py
# python  -m lerobot.scripts.lerobot_eval  --policy.path=./ckpt/smolvla_aloha/checkpoints/last/pretrained_model \
#                 --env.type=aloha \
#                 --env.task=AlohaInsertion-v0 \
#                 --policy.device=npu \
#                 --output_dir=./eval_output/smolvla_aloha

# python eval.py  --policy.path=./ckpt/smolvla_pusht/checkpoints/last/pretrained_model \
#                 --env.type=pusht \
#                 --policy.device=npu \
#                 --output_dir=./eval_output/smolvla_pusht

# python eval.py  --policy.path=./ckpt/pi0_pusht/checkpoints/last/pretrained_model \
#                 --env.type=pusht \
#                 --policy.device=npu \
#                 --output_dir=./eval_output/pi0_pusht

python -m lerobot.scripts.lerobot_eval\
  --policy.path=$PROJECT_ROOT/ckpt/smolvla_libero/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=npu \
  --output_dir=./eval_output/smolvla_libero


# python -m lerobot.scripts.lerobot_eval\
#   --policy.path=HuggingFaceVLA/smolvla_libero \
#   --env.type=libero \
#   --env.task=libero_object \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --policy.device=npu \
#   --output_dir=./eval_output/smolvla_libero_official