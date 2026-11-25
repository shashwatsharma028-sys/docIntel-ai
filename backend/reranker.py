"""Optional cross-encoder reranker (DOCINTEL_RERANKER=minilm). Off by default.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2, ONNX export, pinned revision, run with onnxruntime on
CPU. Requires optional packages not in the core runtime: onnxruntime, tokenizers, huggingface_hub
(model files are fetched once from Hugging Face or read from DOCINTEL_RERANKER_DIR). Evaluated only
with local BGE retrieval on QASPER (evaluation/qasper/ablation_*.json); not validated with Gemini.
"""
import os
import threading

import numpy as np

MODEL = ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")


class MiniLMReranker:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        local = os.environ.get("DOCINTEL_RERANKER_DIR")
        if local:
            model_path, tok_path = os.path.join(local, "onnx", "model.onnx"), os.path.join(local, "tokenizer.json")
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(MODEL[0], "onnx/model.onnx", revision=MODEL[1])
            tok_path = hf_hub_download(MODEL[0], "tokenizer.json", revision=MODEL[1])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("DOCINTEL_RERANKER_THREADS", "4"))
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(tok_path)
        self.tok.enable_truncation(512)
        self.tok.enable_padding()
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.lock = threading.Lock()  # one inference at a time per process (bounded CPU use)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch([(query, t) for t in texts])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        with self.lock:
            return self.session.run(None, feed)[0][:, 0]
