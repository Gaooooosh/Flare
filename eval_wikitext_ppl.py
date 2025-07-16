"""
eval_wikitext_ppl.py
功能：给定模型 + tokenizer + max_seq_len，返回 WikiText-103 上的滑动窗口 PPL
使用evaluate库优化内存占用
"""
import os

from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import torch
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer,AutoModelForCausalLM
from patch_qwen_rope import patch_qwen_rope

def create_token_segments(test_ds, tokenizer, max_len, stride):
    """
    A generator that yields token segments without loading the whole dataset into memory.
    """
    buffer = []
    for sample in test_ds["text"]:
        # Skip empty lines which are common in wikitext
        if not sample.strip():
            continue
            
        # Add tokenized sample to the buffer
        buffer.extend(tokenizer(sample)["input_ids"])
        
        # Yield all complete segments from the current buffer
        while len(buffer) >= max_len:
            yield buffer[:max_len]
            # Slide the buffer forward
            buffer = buffer[stride:]
            
    # Yield the final, potentially shorter, segment
    if buffer:
        yield buffer

@torch.no_grad()
def compute_ppl(model, tokenizer, max_len, batch_size=1):
    model.eval()
    device = next(model.parameters()).device

    # 1. Load the dataset object (this part is fine and memory-efficient)
    test_ds = load_dataset("/raid_sdh/home/xyg/wikitext", 'wikitext-103-raw-v1', split="test")

    # 2. Use our memory-efficient generator instead of creating giant lists
    stride = max_len // 2
    token_segments_iterator = create_token_segments(test_ds, tokenizer, max_len, stride)

    # 3. The DataCollator and DataLoader work directly with the iterator
    #    Note: For the final batch, we need a collator that can handle padding.
    #    DataCollatorForLanguageModeling does this automatically.
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(list(token_segments_iterator), batch_size=batch_size, collate_fn=collator, shuffle=False)
    
    # --- The rest of the function is identical ---
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0

    print("Calculating PPL...")
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Get model output
        logits = model(**batch).logits
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        
        # Calculate loss
        nll = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Accumulate total loss and token count
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    # Calculate perplexity
    ppl = math.exp(total_nll / total_tokens)
    
    model.train() # Set model back to training mode
    return ppl
from transformers import TrainerCallback

class WikiPPLCallback(TrainerCallback):
    def __init__(self, tokenizer, max_len, eval_every=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every == 0 and state.global_step > 0:
            ppl = compute_ppl(model, self.tokenizer, self.max_len)
            print(f"Step {state.global_step}: WikiText PPL = {ppl:.2f}")
            if state.tb_writer is not None:
                state.tb_writer.add_scalar("eval/wiki_ppl", ppl, state.global_step)


if __name__ == '__main__':
    model_path = '/raid_sdh/home/xyg/output_qwen3b_redpajama_nope'
    # model_path = '/raid_sdh/home/xyg/PRETRAINED_MODEL/qwen-3B'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    patch_qwen_rope(model, no_rope_layers=list(range(20,33)))
    tokenizer.pad_token = tokenizer.eos_token
    print(compute_ppl(model,tokenizer,max_len=1024*16,batch_size=1))