#!/bin/bash
exec 2> /dev/null

ctrl_c() {
    kill -9 $pid
    kill -9 $waiter
    exit 2
}
trap ctrl_c INT

echo_cyan() {
    echo -ne "\033[0;36m"
    echo "$1"
    echo -ne "\033[0m"
}

if [ $# -ne 2 ]
then
    echo "Format:  ./runall.sh solution_index time_limit"
    echo "Example: ./runall.sh 1 10"
    exit 0
fi

solution="CS3243_P1_02_$1.py"
timelimit=$2

for i in ./public_tests_p1/*/*.out
do
    rm $i
done

for i in ./public_tests_p1/*/*.txt
do
    start=`date +%s.%N`
    # Spawn a child process. Time limit: 60s
    python $solution $i ${i/.txt/.out} 2>&1 1> /dev/null & pid=$!
    (sleep $timelimit && kill -9 $pid) & waiter=$!
    # wait on our worker process and return the exitcode
    wait $pid 
    exitcode=$?
    # kill the waiter subshell, if it still runs
    kill -9 $waiter
    end=`date +%s.%N`
    runtime=$(python -c "print(${end} - ${start})")

    echo_cyan "> ${i/.txt}"
    if [ $exitcode -eq 137 ]
    then
        echo "Time Limit Exceeded ($timelimit seconds)"
    elif [ $exitcode -ne 0 ]
    then
        echo "Program crashed"
    else
        echo "Program finished ($runtime seconds)"
    fi
done


