#!/usr/bin/env bash
# One object end to end (experiment). The object is described by a config file (configs/*.sh) that sets
#   RUN NAME SCENE PLY YAW PITCH FOV PROMPT PARTS(array, may be empty) REST(floor|table)
#   EXTRA_RUNS(array, optional): other runs of the same object whose crop + mask join the views as given views
# steps, in order:
#   crop     panorama -> square perspective crop facing the object (+ cam.json)
#   align    render the 3DGS from the same camera, side by side with the crop (check_align.jpg)
#   mask     SAM3 text prompt -> mask_base.png (best instance)
#   parts    add what lies on it (PARTS words) -> mask_final.png; no PARTS = the base mask as is
#   gen      SAM 3D Objects -> gen/<NAME>.glb, gen/<NAME>_posed.{glb,ply}, gen/<NAME>_pose.json
#   preview  textured mesh from 4 directions next to the crop -> gen/<NAME>_preview.jpg
#   place    fit the mesh onto the object's scene points -> place.json, gen/<NAME>_world.glb
#   carve    delete the original object's gaussians -> scene_carved.ply (+ removed.ply)
#   carve    (old) delete what falls in a box around the placed mesh -> scene_carved.ply (+ removed.ply)
#   -- multi-view flow (10-14): --
#   views    6 views around the object rendered from the 3DGS + the given views (own crop, EXTRA_RUNS) -> views/
#   vmask    SAM3 on every rendered view (PROMPT + PARTS) -> views/<k>/mask.png
#   vote     which gaussians are the old object -> labels.npy
#   carvel   delete the voted object (+ unseen points inside its extent) -> scene_carved.ply (+ removed.ply)
#   fitmv    measure the old object, refine the mesh against all views (+-10% stretch) -> gen/<NAME>_world.glb, fit_mv.json
#   overlay  mask vs mesh outline in every view -> views_overlay.jpg
#   check    render before / after / after + mesh / removed from 3 cameras -> check.jpg, check_silhouette.jpg
#   compare  per view: original / after deletion / deleted, with the mask outline -> views/<k>/cmp_*.jpg, compare.json
#   debug    data for the 3D check page viewer/debug.html?run=<RUN> -> debug/
# usage: scripts/run_object.sh configs/<object>.sh [step ...]     no step = all
#   SEED=<n> GEN=<dir> before the command: generate with another seed into runs/<RUN>/<dir> (gen and preview only)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$1"; shift
S=$ROOT/scripts
D=$ROOT/runs/$RUN
SC=$ROOT/物品摆放测试/$SCENE
STEPS=${*:-crop align mask parts gen preview place views vmask vote carvel fitmv overlay check compare debug}
EXTRA=(); for e in "${EXTRA_RUNS[@]:-}"; do [ -n "$e" ] && EXTRA+=("$ROOT/runs/$e"); done
PY3D=/home/ubuntu/miniforge3/envs/sam3d/bin/python
PYS3=/home/ubuntu/miniforge3/envs/sam3/bin/python
mkdir -p "$D"

# Claim a card the way the sam3d pipeline does: hold /tmp/outdoor_pipeline_gpu<N>.lock for the whole run.
# 22 GB free because SAM 3D generation (with texture baking) needs about a whole 3090.
NEED_MB=22000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CARD=""
until [ -n "$CARD" ]; do
  for row in $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n); do
    IFS=, read -r idx used total <<< "$row"
    exec 9>>"/tmp/outdoor_pipeline_gpu${idx}.lock"
    if flock -n 9 && [ $(( total - used )) -ge "$NEED_MB" ]; then CARD=$idx; break; fi
    exec 9>&-
  done
  [ -n "$CARD" ] || sleep 20
done
export CUDA_VISIBLE_DEVICES=$CARD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[run] $RUN on gpu $CARD, steps: $STEPS"

for st in $STEPS; do
  echo "[run] == $st"
  case $st in
    crop)    $PY3D -I $S/crop.py $SC/pano.jpg $D $YAW $PITCH $FOV 1024 ;;
    align)   $PY3D -I $S/render_gs.py $SC/$PLY $D ;;
    mask)    $PYS3 -I $S/mask.py $D "$PROMPT" ;;
    parts)   if [ ${#PARTS[@]} -gt 0 ]; then $PYS3 -I $S/mask_parts.py $D mask_base.png "${PARTS[@]}"
             else cp $D/mask_base.png $D/mask_final.png; echo "[parts] none, mask_final = mask_base"; fi ;;
    gen)     $PY3D -I $S/gen.py $D $D/crop.jpg $D/mask_final.png $NAME ${SEED:-42} ${GEN:-gen} ;;
    preview) $PY3D -I $S/preview_mesh.py $D $D/${GEN:-gen}/${NAME}_posed.glb $D/${GEN:-gen}/${NAME}_preview.jpg ;;
    place)   $PY3D -I $S/place.py $D $SC/$PLY $NAME $REST ;;
    carve)   $PY3D -I $S/carve.py $D $SC/$PLY $D/scene_carved.ply ;;
    views)   $PY3D -I $S/render_views.py $D $SC/$PLY $D "${EXTRA[@]}" ;;
    vmask)   $PYS3 -I $S/mask_views.py $D "$PROMPT" "${PARTS[@]}" ;;
    vote)    $PY3D -I $S/vote.py $D $SC/$PLY ;;
    carvel)  $PY3D -I $S/carve_labels.py $D $SC/$PLY $D/scene_carved.ply ;;
    fitmv)   $PY3D -I $S/fit_mv.py $D $SC/$PLY $NAME ;;
    overlay) $PY3D -I $S/views_overlay.py $D $NAME ;;
    check)   $PY3D -I $S/check.py $D $SC/$PLY $D/scene_carved.ply $NAME ;;
    compare) $PY3D -I $S/view_compare.py $D $SC/$PLY $D/scene_carved.ply ;;
    debug)   $PY3D -I $S/debug_export.py $D $SC/$PLY $NAME ;;
    *)       echo "[run] unknown step: $st"; exit 2 ;;
  esac
done
