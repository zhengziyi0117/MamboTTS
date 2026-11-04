"""使用随机初始化的小 BERT 验证特征等价，不下载模型、不访问 GPU。"""
from types import SimpleNamespace as Namespace
import unittest

import torch
from transformers import BertConfig, BertForMaskedLM

from engine_runtime import optimize_bert


class BertFeatureTests(unittest.TestCase):
    def test_encoder_and_device_expansion_match_original_hidden_features(self):
        torch.manual_seed(123)
        model = BertForMaskedLM(BertConfig(vocab_size=31, hidden_size=24,
            num_hidden_layers=4, num_attention_heads=4, intermediate_size=32)).eval()
        inputs = {"input_ids": torch.tensor([[2, 5, 8, 7, 3]])}
        word2ph = [2, 1, 3]
        with torch.no_grad():
            original = model(**inputs, output_hidden_states=True)["hidden_states"][-3][0, 1:-1].cpu()
            expected = torch.cat([row.repeat(count, 1) for row, count in zip(original, word2ph)]).T
        called_head = []
        hook = model.cls.register_forward_hook(lambda *args: called_head.append(True))
        api = Namespace(bert_model=model, torch=torch, device="cpu",
                        tokenizer=lambda *a, **k: inputs, get_bert_feature=lambda *a: None)
        optimize_bert(api)
        actual = api.get_bert_feature("测试句", word2ph)
        hook.remove()
        self.assertEqual(called_head, [])
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(tuple(actual.shape), (24, 6))
        self.assertFalse(actual.requires_grad)


if __name__ == "__main__":
    unittest.main()
