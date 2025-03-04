from megatron.training.training import build_train_valid_test_data_iterators


def get_train_valid_test_data_iterators():
    # [confused] train_valid_test_datasets_provider 貌似没有定义，但是该函数在model.llama_hf等等的train_dist_random.py中有使用
    train_valid_test_datasets_provider.is_distributed = True
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )
    return train_data_iterator, valid_data_iterator, test_data_iterator


def compile_helpers():
    # 在运行时编译C++辅助函数
    # 该函数用于编译 Megatron 核心数据集模块中的 C++ 辅助函数，确保这些函数在运行时可用。
    # 注意：此函数应在单进程中调用，以避免多进程编译冲突。
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import os
    import subprocess

    current_dir = os.path.dirname(__file__)
    target_dir = os.path.join(current_dir, "../../site_package/megatron/core/datasets")

    print("Compiling the C++ dataset helper functions...")
    command = ["make", "-C", os.path.abspath(target_dir)]
    if subprocess.run(command).returncode != 0:
        import sys

        print("Failed to compile the C++ dataset helper functions")
        sys.exit(1)
