#!/usr/bin/env python3
"""
Gemma3 集成测试脚本
验证 Gemma3 模型配置、tokenizer 和 weight loader 是否正确集成
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_gemma3_config():
    """测试 Gemma3 模型配置"""
    print("=" * 60)
    print("测试 1: Gemma3 模型配置")
    print("=" * 60)
    
    from openpi.models import gemma
    
    # 测试所有 Gemma3 变体
    variants = ["gemma3_4b", "gemma3_4b_lora", "gemma3_300m", "gemma3_300m_lora"]
    
    for variant in variants:
        try:
            config = gemma.get_config(variant)
            print(f"✓ {variant:25s} - vocab_size: {config.vocab_size:,}, depth: {config.depth}, width: {config.width}")
            
            # 验证 vocab_size
            if variant.startswith("gemma3"):
                assert config.vocab_size == gemma.GEMMA3_VOCAB_SIZE, f"Expected {gemma.GEMMA3_VOCAB_SIZE}, got {config.vocab_size}"
            
            # 验证 LoRA
            if "lora" in variant:
                assert len(config.lora_configs) > 0, "LoRA configs should be present"
            else:
                assert len(config.lora_configs) == 0, "LoRA configs should not be present"
                
        except Exception as e:
            print(f"✗ {variant:25s} - FAILED: {e}")
            return False
    
    print("✓ 所有 Gemma3 配置测试通过\n")
    return True


def test_gemma3_tokenizer():
    """测试 Gemma3 Tokenizer"""
    print("=" * 60)
    print("测试 2: Gemma3 Tokenizer")
    print("=" * 60)
    
    from openpi.models import tokenizer
    import numpy as np
    
    # 检查 tokenizer 路径
    gemma3_path = Path("/root/.cache/kagglehub/models/google/gemma-3/flax/gemma3-4b-it/1")
    tokenizer_path = gemma3_path / "tokenizer.model"
    
    if not tokenizer_path.exists():
        print(f"✗ Gemma3 tokenizer not found at {tokenizer_path}")
        print("  Please download: kagglehub.model_download('google/gemma-3/flax/gemma3-4b-it')")
        return False
    
    print(f"✓ Tokenizer path exists: {tokenizer_path}")
    
    try:
        # 测试 Pi0 格式 (without state)
        tok = tokenizer.Gemma3Tokenizer(max_len=48)
        tokens, mask = tok.tokenize("pick up the cube")
        print(f"✓ Pi0 format: tokens shape={tokens.shape}, mask shape={mask.shape}")
        
        # 测试 Pi0.5 格式 (with state)
        state = np.random.randn(7)
        tokens, mask = tok.tokenize("pick up the cube", state=state)
        print(f"✓ Pi0.5 format: tokens shape={tokens.shape}, mask shape={mask.shape}")
        
        print("✓ Gemma3 Tokenizer 测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemma3_weight_loader():
    """测试 Gemma3 Weight Loader"""
    print("=" * 60)
    print("测试 3: Gemma3 Weight Loader")
    print("=" * 60)
    
    from openpi.training import weight_loaders
    
    # 检查 checkpoint 路径
    ckpt_path = Path("/root/.cache/kagglehub/models/google/gemma-3/flax/gemma3-4b-it/1/gemma3-4b-it")
    
    if not ckpt_path.exists():
        print(f"✗ Gemma3 checkpoint not found at {ckpt_path}")
        print("  Please download: kagglehub.model_download('google/gemma-3/flax/gemma3-4b-it')")
        return False
    
    print(f"✓ Checkpoint path exists: {ckpt_path}")
    
    try:
        loader = weight_loaders.Gemma3WeightLoader(target_img_size=224)
        print(f"✓ Gemma3WeightLoader created successfully")
        print(f"  Target image size: {loader.target_img_size}")
        
        print("✓ Gemma3 Weight Loader 测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ Weight loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_config():
    """测试训练配置"""
    print("=" * 60)
    print("测试 4: 训练配置")
    print("=" * 60)
    
    from openpi.training import config
    
    # 查找 Gemma3 LIBERO 配置
    gemma3_configs = [c for c in config._CONFIGS if "gemma3_libero" in c.name]
    
    if not gemma3_configs:
        print("✗ No Gemma3 LIBERO configs found")
        return False
    
    for cfg in gemma3_configs:
        print(f"✓ Found config: {cfg.name}")
        print(f"  Model: {cfg.model.paligemma_variant} + {cfg.model.action_expert_variant}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Train steps: {cfg.num_train_steps}")
    
    print("✓ 训练配置测试通过\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Gemma3 集成测试套件")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("模型配置", test_gemma3_config()))
    results.append(("Tokenizer", test_gemma3_tokenizer()))
    results.append(("Weight Loader", test_gemma3_weight_loader()))
    results.append(("训练配置", test_training_config()))
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 所有测试通过！Gemma3 集成成功！")
        print("\n下一步:")
        print("  1. 计算归一化统计: uv run scripts/compute_norm_stats.py --config-name pi05_gemma3_libero")
        print("  2. 开始训练: ./train_gemma3_libero.sh")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
