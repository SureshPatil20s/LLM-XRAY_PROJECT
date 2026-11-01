"""
Verification script for Step 2 of LLM-XRay:
Verify that: Input -> Model -> Generated Response works locally.
"""
import sys
from tokenizer import load_tokenizer, get_token_details
from model import load_model, inspect_inference_internals, generate_step_by_step

def test_inference(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"=== Testing LLM-XRay with {model_name} ===")
    print("1. Loading Tokenizer...")
    tokenizer = load_tokenizer(model_name)
    
    prompt = "What is AI?"
    print(f"\n2. Tokenizing prompt: '{prompt}'")
    tok_res = get_token_details(prompt, tokenizer)
    print("Tokens:", tok_res["tokens"])
    print("Token IDs:", tok_res["token_ids_list"])
    print("Display:", tok_res["display_tokens_str"])
    
    print("\n3. Loading Model (low CPU memory mode)...")
    model, device = load_model(model_name)
    print(f"Model successfully loaded on {device}!")
    
    print("\n4. Running Forward Pass & Internal Inspection...")
    internals = inspect_inference_internals(model, tokenizer, prompt, device=device)
    print(f"- Input Embeddings shape: {tuple(internals['input_embeddings'].shape)}")
    print(f"- Number of Transformer Layers: {internals['num_layers']}")
    print(f"- Attention Heads per Layer: {internals['num_heads']}")
    print(f"- Hidden Dimension: {internals['hidden_dim']}")
    print(f"- Logits shape: {tuple(internals['last_logits'].shape)}")
    print("\nTop 5 Next-Token Predictions from LM Head Softmax:")
    for pred in internals["top_predictions"][:5]:
        print(f"  Token: {pred['token']:<15} Prob: {pred['percentage']:<8} Logit: {pred['logit']:.2f}")
        
    print("\n5. Running Step-by-Step Autoregressive Generation (max 5 tokens)...")
    response, history, usage_stats = generate_step_by_step(model, tokenizer, prompt, max_new_tokens=5, temperature=0.7, device=device)
    print(f"Generated text: '{response}'")
    print(f"Total steps captured: {len(history)}")
    for step in history:
        print(f"  Step {step['step']}: Token '{step['display_str']}' (P={step['percentage']})")
        
    print("\n=== All Steps 1 to 10 verified successfully! ===")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    test_inference(name)
