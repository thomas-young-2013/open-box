#!/usr/bin/env bash
echo "litebo 20 START RUNNING......"
for i in $( seq 1 5 )
do
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem lgb_kc1 --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem svc_kc1 --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem lgb_pollen --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem svc_pollen --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem lgb_spambase --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem svc_spambase --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem lgb_abalone --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem svc_abalone --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem lgb_car\(2\) --n 200 --rep 1 --start_id $i
    wait
    python3 test/optimizer/benchmark_so_litebo_ml.py --problem svc_car\(2\) --n 200 --rep 1 --start_id $i
    wait
done
echo "litebo 20 ALL JOBS DONE!!!"
