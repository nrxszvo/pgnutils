name=default
nproc=8
nthread=8

for var in "$@"
do
	param=(${var//=/ })
	if [[ ${param[0]} = "name" ]]
	then
		name=${param[1]}
	elif [[ ${param[0]} = "nproc" ]]
	then
		echo 'eq nproc'
		nproc=${param[1]}
	elif [[ ${param[0]} = "nthread" ]]
	then
		nthread=${param[1]}
	else
		echo 'unequal'
	fi
done
cmd="OMP_NUM_THREADS=${nthread} torchrun --standalone --nnodes 1 --nproc-per-node ${nproc} train.py --name ${name}"
echo $cmd
eval $cmd
