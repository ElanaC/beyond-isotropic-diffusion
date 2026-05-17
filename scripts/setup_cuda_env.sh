# Source before training on GCP DL VM:  source scripts/setup_cuda_env.sh
# Prepends all pip-shipped NVIDIA libs (cublas, cudnn, etc.) to LD_LIBRARY_PATH.

_nvlibs=$(python3 -c "
import glob, site
paths = []
for sp in site.getsitepackages():
    paths.extend(glob.glob(sp + '/nvidia/*/lib'))
print(':'.join(sorted(set(paths))))
" 2>/dev/null)

if [ -n "$_nvlibs" ]; then
  export LD_LIBRARY_PATH="${_nvlibs}:${LD_LIBRARY_PATH:-}"
  echo "LD_LIBRARY_PATH updated (nvidia pip libs)"
else
  echo "No nvidia/*/lib under site-packages; skip"
fi
