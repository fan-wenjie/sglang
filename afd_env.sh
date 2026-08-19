# The environment this tree needs on a Blackwell (sm_120) card, written down because every one of
# these was a separate failure with a misleading symptom.
#
#   PATH        sglang JIT-compiles kernels with ninja; without the env's bin the child process
#               reports FileNotFoundError: 'ninja' from inside a model load.
#   CUDA_HOME   /usr/bin/nvcc is 12.8 here and sm_120a needs >= 12.9. The cu13 wheel ships a
#               complete 13.3 toolchain; point at it rather than the system one.
#   LIBRARY_PATH  the cu13 wheels ship libcudart.so.13 and no .so development symlink, so the
#               link step of the JIT fails with "cannot find -lcudart". .cuda-devlinks holds the
#               symlinks; it is generated, not checked in.
#
# One more thing has to be true and is not set here: nvidia-cuda-runtime must match
# nvidia-cuda-nvcc. They arrived as 13.0 and 13.3 and CCCL refuses the pair with "CUDA compiler
# and CUDA toolkit headers are incompatible", which reads like a broken install and is a version
# skew between two wheels.
export SGLANG_SRC=/home/user/experiment/sglang
export CU13=/home/user/.conda/envs/sglang/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$CU13"
export PATH="$CU13/bin:/home/user/.conda/envs/sglang/bin:$PATH"
export LIBRARY_PATH="$SGLANG_SRC/.cuda-devlinks:$CU13/lib"
export LD_LIBRARY_PATH="$CU13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SGLANG_SRC/python"
export AFD_MODEL=/home/user/experiment/models/Qwen3.8-27B-FP8
