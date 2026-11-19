"""Run Q4 parts a -> d in order (uv run python run_all.py)."""
import q4a_ml, q4b_qubo, q4c_annealing, q4d_parallel

if __name__ == "__main__":
    for m in (q4a_ml, q4b_qubo, q4c_annealing, q4d_parallel):
        print("\n" + "=" * 20, m.__name__, "=" * 20)
        m.main()
