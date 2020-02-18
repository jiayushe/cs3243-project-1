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

if [ -z "$1" ]
then
    echo "Enter the index of the solution file as the first argument."
    echo "eg: ./runall.sh 1"
    exit 0
fi

timelimit="10"
solution="CS3243_P1_02_$1.py"

for i in ./public_tests_p1/*/*.txt
do
    # Spawn a child process. Time limit: 60s
    python $solution $i ${i/.txt/.out} 2>&1 1> /dev/null & pid=$!
    (sleep $timelimit && kill -9 $pid) & waiter=$!
    # wait on our worker process and return the exitcode
    wait $pid 
    exitcode=$?
    # kill the waiter subshell, if it still runs
    kill -9 $waiter
    
    echo_cyan "> ${i/.txt}"
    if [ $exitcode -eq 137 ]
    then
        echo "Time Limit Exceeded ($timelimit seconds)"
    elif [ $exitcode -ne 0 ]
    then
        echo "Program crashed"
    else
        echo "Program finished"
    fi
done


