echo baseline-early-exit
python cdq_test_clotho_baseline_early_exit.py > ./logs/baseline-early-exit.log 2>&1

# sleep 60

echo baseline-full-bs-32
python cdq_test_clotho_baseline_full.py --bs 32 --ee 11 > ./logs/baseline-full-bs-32.log 2>&1

# sleep 60

echo baseline-full-bs-1
python cdq_test_clotho_baseline_full.py --bs 1 --ee 11 > ./logs/baseline-full-bs-1.log 2>&1

# sleep 60

for ee in 1 2 3 4 5 6 7 8 9 10 11
do
    echo "Running cdq_test_clotho_progressive.py with ee=${ee} and batch_size=32"
    python cdq_test_clotho_progressive.py --ee $ee --bs 32 > ./logs/cdq_ee${ee}_bs32.log 2>&1
    # sleep 60
done
