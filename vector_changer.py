import torch
import os


def process_tensor(tensor):

    assert tensor.dim() == 2 and tensor.size(0) == 1 and tensor.size(1) == 768, \
        "Тензор должен иметь размерность [1, seq_len, 768]"


    last_step_squeezed = tensor.squeeze(0)

    linear = torch.nn.Linear(768, 32)
    compressed = linear(last_step_squeezed)

    return compressed


def process_all_tensors(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            filepath = os.path.join(directory, filename)
            tensor = torch.load(filepath)
            processed = process_tensor(tensor)

            output_path = os.path.join(output_directory, filename)
            torch.save(processed, output_path)
            print(f"Обработан и сохранён: {output_path}")


def inspect_tensors(directory, num_elements=5):
    if not os.path.exists(directory):
        print(f"Директория {directory} не существует.")
        return

    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    if not pt_files:
        print(f"В директории {directory} нет .pt файлов.")
        return

    for filename in pt_files:
        filepath = os.path.join(directory, filename)
        try:
            tensor = torch.load(filepath)
            print(f"\nФайл: {filename}")

            # Проверка типа объекта
            if isinstance(tensor, torch.Tensor):
                print(f"Тип объекта: torch.Tensor")
                print(f"Размерности тензора: {tensor.shape}")

                # Вывод первых элементов для примера
                if tensor.dim() == 0:
                    print(f"Содержимое тензора (скаляр): {tensor.item()}")
                else:
                    print(f"Пример содержимого тензора:")
                    print(tensor.view(-1)[:min(num_elements, tensor.numel())])
            elif isinstance(tensor, dict):
                print(f"Тип объекта: dict")
                print("Ключи в словаре:")
                for key in tensor.keys():
                    print(f"  - {key}")
                    value = tensor[key]
                    if isinstance(value, torch.Tensor):
                        print(f"    Размерности: {value.shape}")
                        print(f"    Пример содержимого: {value.view(-1)[:min(num_elements, value.numel())]}")
                    else:
                        print(f"    Тип значения: {type(value)}")
            else:
                print(f"Тип объекта: {type(tensor)}")
                print(f"Содержимое: {tensor}")

        except Exception as e:
            print(f"Не удалось загрузить {filename}: {e}")


if __name__ == "__main__":
    input_dir = "vectorscp"
    output_dir = "vectors_processed16cp"
    process_all_tensors(input_dir, output_dir)
    inspect_tensors(output_dir)

