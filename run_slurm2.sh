module use /sapps/etc/modules/start/
module load generic

for j in {0..8}; do
    sbatch run.sh decode_model.py 0 $j lang
    sbatch run.sh decode_model.py 0 $j langPOS
    sbatch run.sh decode_model.py 0 $j langPOSsem
    sbatch run.sh decode_model.py 0 $j langPOSsemetym
    sbatch run.sh decode_model_null.py 0 $j
done

