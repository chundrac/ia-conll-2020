module use /sapps/etc/modules/start/
module load generic

for j in {0..7}; do
    sbatch run.sh run_model.py 0 $j lang
    sbatch run.sh run_model.py 0 $j langPOS
    sbatch run.sh run_model.py 0 $j langPOSsem
    sbatch run.sh run_model.py 0 $j langPOSsemetym
    sbatch run.sh run_model_null.py 0 $j
done

