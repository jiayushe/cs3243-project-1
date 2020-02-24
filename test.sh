#!/bin/bash
exec 2> /dev/null

ctrl_c() {
    kill -9 ${pid}
    kill -9 ${waiter}
    exit 2
}
trap ctrl_c INT

echo_cyan() {
    echo -ne "\033[0;36m"
    echo "${1}"
    echo -ne "\033[0m"
}

if [ $# -ne 4 ]
then
    echo "Format:  ./test.sh solution_index time_limit dimension sample_size"
    echo "Example: ./test.sh 1 10 4 100"
    exit 0
fi

solution="CS3243_P1_02_${1}.py"
timelimit=${2}
dimension=${3}
sample_size=${4}

if [ -d "./experiment" ];
then
    rm -rf experiment
fi

mkdir experiment

for (( i=1; i <= ${sample_size}; i++ ))
do
    python gen.py ${dimension} ./experiment/${dimension}_${i}.in
done

total_pass=0
total_runtime=0
runtime_arr=()

for i in ./experiment/*.in
do
    start=`date +%s.%N`
    # Spawn a child process.
    python ${solution} $i ${i/.in/.out} 2>&1 1> /dev/null & pid=$!
    (sleep ${timelimit} && kill -9 ${pid}) & waiter=$!
    # wait on our worker process and return the exitcode
    wait ${pid}
    exitcode=$?
    # kill the waiter subshell, if it still runs
    kill -9 ${waiter}
    end=`date +%s.%N`
    runtime=$(python -c "print(${end} - ${start})")

    echo_cyan "> ${i/.txt}"
    if [ ${exitcode} -eq 137 ]
    then
        echo "Time limit exceeded (${timelimit} seconds)"
    elif [ ${exitcode} -ne 0 ]
    then
        echo "Program crashed (${runtime} seconds)"
    else
        echo "Program finished (${runtime} seconds)"
        runtime_arr[$total_pass]=${runtime}
        total_pass=$(python -c "print(${total_pass} + 1)")
        total_runtime=$(python -c "print(${total_runtime} + ${runtime})")
    fi
done

pass_rate=$(python -c "print(${total_pass} * 100.0 / ${sample_size})")
echo_cyan "Pass Rate"
echo "$pass_rate%"

ave_runtime=$(python -c "print(${total_runtime} / ${sample_size})")
echo_cyan "Average Runtime"
echo "${ave_runtime}s"

std_dev=0
for i in ${runtime_arr[@]}
do
    std_dev=$(python -c "print((${i} - ${ave_runtime}) ** 2 + ${std_dev})")
done
std_dev=$(python -c "print((${std_dev} / ${total_pass}) ** 0.5)")
echo_cyan "Standard Deviation"
echo "${std_dev}s"

max=0
for i in ${runtime_arr[@]}
do
    max=$(python -c "print(max(${max}, ${i}))")
done
echo_cyan "Max Runtime"
echo "${max}s"
