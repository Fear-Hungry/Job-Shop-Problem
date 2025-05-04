import numpy as np
from pathlib import Path
import json
from typing import Dict, Any


def parse_instance(filepath: str) -> Dict[str, Any]:
    """Converte uma instância do formato original para JSON."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove comentários e linhas vazias
    lines = [line.strip() for line in lines if line.strip()
             and not line.startswith('#')]

    # Lê número de jobs e máquinas
    num_jobs, num_machines = map(int, lines[0].split())

    # Lê os tempos de processamento
    processing_times = np.zeros((num_jobs, num_machines), dtype=int)
    machine_sequence = np.zeros((num_jobs, num_machines), dtype=int)

    for i in range(num_jobs):
        values = list(map(int, lines[i+1].split()))
        for j in range(num_machines):
            machine = values[2*j]
            time = values[2*j + 1]
            processing_times[i, j] = time
            machine_sequence[i, j] = machine

    return {
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'processing_times': processing_times.tolist(),
        'machine_sequence': machine_sequence.tolist(),
        'name': Path(filepath).stem
    }


def convert_all_instances(input_dir: str, output_dir: str):
    """Converte todas as instâncias do diretório de entrada para o diretório de saída."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filepath in input_path.glob('*'):
        if filepath.is_file() and not filepath.name.startswith('.'):
            try:
                instance = parse_instance(str(filepath))
                output_file = output_path / f"{instance['name']}.json"

                with open(output_file, 'w') as f:
                    json.dump(instance, f, indent=2)

                print(f"Convertido: {filepath.name} -> {output_file.name}")

            except Exception as e:
                print(f"Erro ao converter {filepath.name}: {e}")


def main():
    """Função principal."""
    input_dir = "data"
    output_dir = "data/benchmarks"

    print("Iniciando conversão das instâncias...")
    convert_all_instances(input_dir, output_dir)
    print("Conversão concluída!")


if __name__ == "__main__":
    main()
