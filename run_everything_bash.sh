for i in {0..7}
do
	for j in {1..20}
	do
		python3 run_model.py 0 $i lang
		python3 run_model.py 0 $i langPOS
		python3 run_model.py 0 $i langPOSsem
		python3 run_model.py 0 $i langPOSsemetym
		python3 run_model_null.py 0 $1
	done
done

for i in {0..7}
do
	for j in {1..20}
	do
		python3 decode_model.py 0 $i lang
		python3 decode_model.py 0 $i langPOS
		python3 decode_model.py 0 $i langPOSsem
		python3 decode_model.py 0 $i langPOSsemetym
		python3 decode_model_null.py 0 $1
	done
done