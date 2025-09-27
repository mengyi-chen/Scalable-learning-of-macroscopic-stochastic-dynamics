python batch_solver.py  --nx 100 --bs 50
python batch_solver.py --nx 200 --bs 20

k_values=(0.05 0.1 0.15)
m_values=(0.45 0.5 0.55)

for k in "${k_values[@]}"; do
    for m in "${m_values[@]}"; do
        python batch_solver.py --nx 200 --bs 50 --flag_test --m $m --k $k
    done
done

