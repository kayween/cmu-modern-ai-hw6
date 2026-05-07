# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.5",
#     "mugrade @ git+https://github.com/locuslab/mugrade.git",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Homework 6 - Supervised finetuning for chat and DPO

    In this homework, you will finetune a pretrained model (of the _exact_ same model cdoe you developed in the previous assignment, just trained for longer) for chat training, using both supervised finetuning (SFT) and direct preference optimization (DPO).
    """)
    return


@app.cell
def _(subprocess):
    ### Run this cell to download and install the necessary modules for the homework
    # packages added via marimo's package management: git+https://github.com/locuslab/mugrade.git !pip install --upgrade --no-deps git+https://github.com/locuslab/mugrade.git
    #! wget -nc https://raw.githubusercontent.com/modernaicourse/hw6/refs/heads/main/hw6_tests.py
    subprocess.call(['wget', '-nc', 'https://raw.githubusercontent.com/modernaicourse/hw6/refs/heads/main/hw6_tests.py'])

    import os
    import math
    import mugrade
    import torch
    import tiktoken
    from torch.nn import Module, ModuleList, Parameter, Buffer
    import json
    import copy


    from hw6_tests import (test_Linear,
                           test_Embedding,
                           test_silu,
                           test_rms_norm,
                           test_self_attention,
                           test_MultiHeadAttentionKVCache,
                           test_MLP,
                           test_TransformerBlock,
                           test_cross_entropy_loss,
                           test_Adam,
                           test_LLM,
                           test_generate,
                           test_convert_to_chat_format, submit_convert_to_chat_format,
                           test_pretokenize_chat, submit_pretokenize_chat,
                           test_get_loss_mask, submit_get_loss_mask,
                           test_DataLoaderChat, submit_DataLoaderChat,
                           test_train_llm_chat, submit_train_llm_chat,
                           test_log_probs, submit_log_probs,
                           test_softplus, submit_softplus,
                           test_dpo_loss, submit_dpo_loss,
                           test_train_dpo, submit_train_dpo,
                           test_eval_llm_chat, submit_eval_llm_chat,
                           test_eval_llm_dpo, submit_eval_llm_dpo)

    os.environ["MUGRADE_HW"] = "Homework 6"
    os.environ["MUGRADE_KEY"] = "" ### Your key here
    return (
        Buffer,
        Module,
        ModuleList,
        Parameter,
        copy,
        json,
        math,
        mugrade,
        os,
        tiktoken,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 0 - Setting up the model

    In this part, you'll want to copy the relevant model and training code from the previous lecture.  The following portions should all be _identical_ to the solutions in HW5, so we are including them with the same exact (local only) tests that were present in this previous homework, rather than new tests (and also leaving out the docstrings on the specific functions and classes ... the aim here is just to _exactly_ copy your implementation from the previous portion).
    """)
    return


@app.cell
def _(Buffer, Module, Parameter, math, mugrade, torch):
    @mugrade.local_tests
    class Linear(Module):
        """ Linear layer with no bias term.  The parameters of the layer are stored in a .weight Parameter"""

        def __init__(self, in_dim, out_dim):
            """
            Initialize a linear layer without a bias term.

            Inputs:
                in_dim : int - input feature dimension
                out_dim : int - output feature dimension
            """
            super().__init__()  ### BEGIN YOUR CODE
            self.weight = Parameter(torch.randn(out_dim, in_dim) * math.sqrt(2 / in_dim))
      ### END YOUR CODE
        def forward(self, X):
            """
            Apply the linear layer to one or more input vectors.

            Input:
                X : torch.Tensor[float] (... x in_dim) - input tensor
            Output:
                torch.Tensor[float] (... x out_dim) - transformed tensor
            """
            return X @ self.weight.T
      ### BEGIN YOUR CODE
    @mugrade.local_tests
    class Embedding(Module):  ### END YOUR CODE

        def __init__(self, num_tokens, dim):
            """
            Initialize an embedding table over a fixed vocabulary.

            Inputs:
                num_tokens : int - vocabulary size
                dim : int - embedding dimension
            """
            super().__init__()
            self.weight = Parameter(torch.randn(num_tokens, dim))

        def forward(self, Y):
            """  ### BEGIN YOUR CODE
            Look up embeddings for an integer tensor of token ids.
      ### END YOUR CODE
            Input:
                Y : torch.Tensor[int] (...) - tensor of token indices in [0, num_tokens)
            Output:
                torch.Tensor[float] (... x dim) - embedding vectors for each token id
            """
            return self.weight[Y, :]

    @mugrade.local_tests
    def silu(x):
        """
        Apply the SiLU nonlinearity elementwise.  ### BEGIN YOUR CODE

        Input:  ### END YOUR CODE
            x : torch.Tensor[float] (...) - input tensor
        Output:
            torch.Tensor[float] (...) - tensor after applying SiLU
        """
        return x * torch.sigmoid(x)

    @mugrade.local_tests
    def rms_norm(X, eps=1e-05):
        """
        Apply RMS normalization along the final dimension of the input.

        Inputs:
            X : torch.Tensor[float] (... x dim) - input tensor  ### BEGIN YOUR CODE
            eps : float - numerical stability constant
        Output:  ### END YOUR CODE
            torch.Tensor[float] (... x dim) - RMS-normalized tensor
        """
        return X / X.square().mean(dim=-1, keepdim=True).add(eps).sqrt()

    @mugrade.local_tests
    def self_attention(Q, K, V, mask=None):
        d = Q.shape[-1]
        dot_product = Q @ K.transpose(-1, -2) / math.sqrt(d)
        if mask is not None:
            dot_product = dot_product + mask
        return torch.softmax(dot_product, dim=-1) @ V

    @mugrade.local_tests
    class MultiHeadAttentionKVCache(Module):  ### BEGIN YOUR CODE

        def __init__(self, dim, n_heads, max_cache_size):  ### END YOUR CODE
            """
            Initialize a multi-head self-attention layer with KV cache buffers.

            Inputs:
                dim : int - total embedding dimension  ### BEGIN YOUR CODE
                n_heads : int - number of attention heads
                max_cache_size : int - maximum sequence length stored in the cache
            """
            super().__init__()
            self.n_heads = n_heads
            self.wq = Linear(in_dim=dim, out_dim=dim)
            self.wk = Linear(in_dim=dim, out_dim=dim)
            self.wv = Linear(in_dim=dim, out_dim=dim)
            self.wp = Linear(in_dim=dim, out_dim=dim)  ### END YOUR CODE
            self.k_cache = Buffer(torch.empty(1, max_cache_size, dim))
            self.v_cache = Buffer(torch.empty(1, max_cache_size, dim))

        def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
            """
            Apply multi-head self-attention, optionally updating and using the KV cache.

            Inputs:
                X : torch.Tensor[float] (batch_size x seq_len x dim) - input sequence embeddings
                mask : torch.Tensor[float] (seq_len x total_len) or None - additive attention mask
                seq_pos : int - starting sequence position for cached tokens
                use_kv_cache : bool - whether to update and use the KV cache
            Output:
                torch.Tensor[float] (batch_size x seq_len x dim) - attention output
            """  ### BEGIN YOUR CODE
            batch_shape = X.shape[:-2]
            seq_len = X.shape[-2]
            Q = self.wq(X)
            K = self.wk(X)
            V = self.wv(X)

            def split_heads(tensor):
                batch_shape = tensor.shape[:-2]
                seq_len = tensor.shape[-2]
                dim = tensor.shape[-1]
                return tensor.reshape(*batch_shape, seq_len, self.n_heads, dim // self.n_heads).transpose(-2, -3)  ### END YOUR CODE
            if use_kv_cache:
                self.k_cache[:, seq_pos:seq_pos + seq_len, :] = K
                self.v_cache[:, seq_pos:seq_pos + seq_len, :] = V
                output = self_attention(Q=split_heads(Q), K=split_heads(self.k_cache[:, :seq_pos + seq_len, :]), V=split_heads(self.v_cache[:, :seq_pos + seq_len, :]), mask=mask)
            else:
                output = self_attention(Q=split_heads(Q), K=split_heads(K), V=split_heads(V), mask=mask)
            output = output.transpose(-2, -3).reshape(*batch_shape, seq_len, -1)
            return self.wp(output)

    @mugrade.local_tests
    class MLP(Module):

        def __init__(self, dim, ffn_dim):
            """
            Initialize a simple two layer feed-forward network used in the transformer block.

            Inputs:
                dim : int - model dimension
                ffn_dim : int - hidden feed-forward dimension
            """
            super().__init__()
            self.w1 = Linear(out_dim=ffn_dim, in_dim=dim)
            self.w2 = Linear(out_dim=dim, in_dim=ffn_dim)

        def forward(self, X):
            """
            Apply a simple two layer feed-forward network to the input tensor.

            Input:  # The cache is only used in inference: batch size is one
                X : torch.Tensor[float] (... x dim) - input tensor
            Output:
                torch.Tensor[float] (... x dim) - transformed tensor
            """
            X = self.w1(X)
            X = silu(X)
            X = self.w2(X)
            return X

    @mugrade.local_tests
    class TransformerBlock(Module):

        def __init__(self, dim, n_heads, ffn_dim, max_cache_size):
            """
            Initialize a transformer block with attention, normalization, and gated MLP layers.

            Inputs:
                dim : int - model dimension
                n_heads : int - number of attention heads
                ffn_dim : int - hidden feed-forward dimension
                max_cache_size : int - maximum sequence length stored in the attention cache
            """
            super().__init__()
            self.attn = MultiHeadAttentionKVCache(dim=dim, n_heads=n_heads, max_cache_size=max_cache_size)
            self.mlp = MLP(dim=dim, ffn_dim=ffn_dim)

        def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
            """
            Apply one transformer block with residual connections.

            Inputs:
                X : torch.Tensor[float] (batch_size x seq_len x dim) - input sequence embeddings
                mask : torch.Tensor[float] (seq_len x total_len) or None - additive attention mask
                seq_pos : int - starting sequence position for cached tokens  ### BEGIN YOUR CODE
                use_kv_cache : bool - whether to update and use the attention cache
            Output:
                torch.Tensor[float] (batch_size x seq_len x dim) - transformed sequence embeddings  ### END YOUR CODE
            """
            X = X + self.attn(rms_norm(X), mask=mask, seq_pos=seq_pos, use_kv_cache=use_kv_cache)
            X = X + self.mlp(rms_norm(X))
            return X

    @mugrade.local_tests
    def cross_entropy_loss(logits, y):
        """
        Compute average cross entropy loss over a minibatch.

        Inputs:  ### BEGIN YOUR CODE
            logits : ND torch.Tensor[float] (... x k) - predicted logits for each example
            y : (N-1)D torch.Tensor[int] (...) - desired class for each example
        Output:
            scalar torch.Tensor[float] - average cross entropy loss
        """  ### END YOUR CODE
        correct_logits = logits.gather(dim=-1, index=y.unsqueeze(-1))
        correct_logits = correct_logits.squeeze()
        log_sum_exp = torch.logsumexp(logits, dim=-1)
        return torch.mean(-correct_logits + log_sum_exp)

    @mugrade.local_tests
    class Adam:

        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
            """
            Initialize Adam optimizer state for a set of parameters.

            Inputs:
                params : iterable[torch.nn.Parameter] - parameters to optimize
                lr : float - learning rate
                betas : tuple(float, float) - decay rates for first and second moments  ### BEGIN YOUR CODE
                eps : float - numerical stability constant
            """
            self.params = list(params)  ### END YOUR CODE
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.u = [torch.zeros_like(p) for p in self.params]
            self.v = [torch.zeros_like(p) for p in self.params]
            self.t = 0

        @torch.no_grad()
        def step(self):
            """
            Apply one Adam update to all stored parameters.
            """
            self.t = self.t + 1
            for (p, u, v) in zip(self.params, self.u, self.v):  ### BEGIN YOUR CODE
                if p.grad is None:
                    continue
                u.mul_(self.betas[0]).add_(p.grad, alpha=1 - self.betas[0])
                v.mul_(self.betas[1]).add_(p.grad.square(), alpha=1 - self.betas[1])  ### END YOUR CODE
                u_hat = u / (1 - self.betas[0] ** self.t)
                v_hat = v / (1 - self.betas[1] ** self.t)
                p = p - self.lr * u_hat / v_hat.sqrt().add(self.eps)

        def zero_grad(self):
            """
            Zero out gradients for all stored parameters when gradients exist.
            """
            for p in self.params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()  ### BEGIN YOUR CODE  ### END YOUR CODE  ### BEGIN YOUR CODE  ### END YOUR CODE  ### BEGIN YOUR CODE  ### END YOUR CODE  ### BEGIN YOUR CODE  ### END YOUR CODE

    return (
        Adam,
        Embedding,
        Linear,
        TransformerBlock,
        cross_entropy_loss,
        rms_norm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In addition to the solutions above, there are some _very_ minor changes required to two functions or classes.  First, we're augmenting the LLM implementation from the previous homework to allow you to load and save models in a Llama-like format.  We're giving you the code for this, so you can just copy your solution to the respective functions, but you should not copy the _entire_ class from the previous assignments, but you should copy the `__init__()` and `forward()` functions and leave the other two functions as they are. The next two cells also have only local tests.
    """)
    return


@app.cell
def _(
    Buffer,
    Embedding,
    Linear,
    Module,
    ModuleList,
    Parameter,
    TransformerBlock,
    mugrade,
    rms_norm,
    torch,
):
    @mugrade.local_tests
    class LLM(Module):
        def __init__(self, num_tokens, dim, n_heads, max_seq_len, ffn_dim, num_layers):
            """
            Initialize the simplified Llama 3 model used in this homework.

            Inputs:
                num_tokens : int - vocabulary size
                dim : int - model dimension
                n_heads : int - number of attention heads per layer
                max_seq_len : int - maximum supported sequence length
                ffn_dim : int - hidden feed-forward dimension in each block
                num_layers : int - number of transformer blocks
            """
            super().__init__()
            ### BEGIN YOUR CODE
            self.embedding = Embedding(num_tokens=num_tokens, dim=dim)
            self.pos_embeddings = Parameter(torch.randn(max_seq_len, dim))

            self.layers = ModuleList(
                [TransformerBlock(dim=dim, n_heads=n_heads, ffn_dim=ffn_dim, max_cache_size=max_seq_len) for _ in range(num_layers)]
            )

            self.output = Linear(out_dim=num_tokens, in_dim=dim)

            mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.mask = Buffer(mask)
            ### END YOUR CODE

        def forward(self, tokens, seq_pos=0, use_kv_cache=False):
            """
            Apply the full language model to a batch of token sequences.

            Inputs:
                tokens : torch.Tensor[int] (batch_size x seq_len) - input token ids
                seq_pos : int - starting sequence position for positional embeddings and cached attention
                use_kv_cache : bool - whether to update and use cached keys and values
            Output:
                torch.Tensor[float] (batch_size x seq_len x num_tokens) - output logits
            """
            ### BEGIN YOUR CODE
            batch_size, seq_len = tokens.shape

            # Embed tokens and add positional embeddings
            h = self.embedding(tokens) + self.pos_embeddings[seq_pos : seq_pos + seq_len]

            # Slice the causal mask: rows are queries (new tokens), columns are all keys (cached + new)
            mask = self.mask[seq_pos : seq_pos + seq_len, : seq_pos + seq_len]

            for layer in self.layers:
                h = layer(h, mask=mask, seq_pos=seq_pos, use_kv_cache=use_kv_cache)

            return self.output(rms_norm(h))
            ### END YOUR CODE

        def load_weights(self, filename):
            """ Load a model from a Llama-like checkpoint """
            checkpoint = torch.load(filename, map_location = self.embedding.weight.device)
            self.embedding.weight.data = checkpoint["tok_embeddings.weight"]
            self.pos_embeddings.data = checkpoint["pos_embeddings.weight"]
            self.output.weight.data = checkpoint["output.weight"]

            for i,layer in enumerate(self.layers):
                layer.attn.wq.weight.data = checkpoint[f"layers.{i}.attention.wq.weight"]
                layer.attn.wk.weight.data = checkpoint[f"layers.{i}.attention.wk.weight"]
                layer.attn.wv.weight.data = checkpoint[f"layers.{i}.attention.wv.weight"]
                layer.attn.wp.weight.data = checkpoint[f"layers.{i}.attention.wo.weight"]

                layer.mlp.w1.weight.data = checkpoint[f"layers.{i}.feed_forward.w1.weight"]
                layer.mlp.w2.weight.data = checkpoint[f"layers.{i}.feed_forward.w2.weight"]

        def save_weights(self, filename):
            """ Save a model to a Llama-like checkpoint."""
            checkpoint = {}
            checkpoint["tok_embeddings.weight"] = self.embedding.weight.detach().to(torch.bfloat16).cpu()
            checkpoint["pos_embeddings.weight"] = self.pos_embeddings.detach().to(torch.bfloat16).cpu()
            checkpoint["output.weight"] = self.output.weight.detach().to(torch.bfloat16).cpu()

            for i, layer in enumerate(self.layers):
                checkpoint[f"layers.{i}.attention.wq.weight"] = layer.attn.wq.weight.detach().to(torch.bfloat16).cpu()
                checkpoint[f"layers.{i}.attention.wk.weight"] = layer.attn.wk.weight.detach().to(torch.bfloat16).cpu()
                checkpoint[f"layers.{i}.attention.wv.weight"] = layer.attn.wv.weight.detach().to(torch.bfloat16).cpu()
                checkpoint[f"layers.{i}.attention.wo.weight"] = layer.attn.wp.weight.detach().to(torch.bfloat16).cpu()
                checkpoint[f"layers.{i}.feed_forward.w1.weight"] = layer.mlp.w1.weight.detach().to(torch.bfloat16).cpu()
                checkpoint[f"layers.{i}.feed_forward.w2.weight"] = layer.mlp.w2.weight.detach().to(torch.bfloat16).cpu()
            torch.save(checkpoint, filename)

    return (LLM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You also will need to slightly revise the `generate()` function from the previous assignment to use an arbitrary token as the end token, since in chat contexts you will use the end assistant tag to stop generating further.
    """)
    return


@app.cell
def _(mugrade, torch):
    @mugrade.local_tests
    def generate(model, prompt_tokens, tokenizer, eot_token=None, temp=0.7, max_tokens=500, verbose=True):
        """
        Autoregressively sample tokens from a language model using its KV cache.

        Inputs:
            model : Module - language model mapping token sequences to logits
            prompt_tokens : list[int] - initial prompt tokens
            tokenizer : object - tokenizer with decode()
            eot_token: int - index of token at which to top generation
            temp : float - sampling temperature
            max_tokens : int - maximum number of new tokens to generate
            verbose : bool - whether to print each generated token as it is sampled
        Output:
            list[int] - generated tokens, excluding the prompt tokens
        """
        seq_pos = 0  ### BEGIN YOUR CODE
        generated_tokens = []
        device = next(model.parameters()).device
        inputs = torch.tensor(prompt_tokens, device=device).unsqueeze(-2)
        logits = model(inputs, seq_pos=0, use_kv_cache=True)
        logits = logits[:, -1, :]
        token = torch.multinomial(torch.softmax(logits / temp, dim=-1), 1).item()  # add a batch size
        generated_tokens.append(token)  # shape (1, num_prompt_tokens, vocab_size)
        seq_pos = seq_pos + len(prompt_tokens)
        while len(generated_tokens) < max_tokens and token != tokenizer.eot_token:
            inputs = torch.tensor(token).view(1, 1)
            logits = model(inputs, seq_pos=seq_pos, use_kv_cache=True)
            logits = logits[:, -1, :]
            token = torch.multinomial(torch.softmax(logits / temp, dim=-1), 1).item()
            generated_tokens.append(token)
            seq_pos = seq_pos + 1  # (batch_size, seq_len)
        if verbose:
            print(tokenizer.decode(generated_tokens), end='', flush=True)
        return generated_tokens  ### END YOUR CODE

    return (generate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, use the following code to download and load a pretrained model of this type.  This was trained with the same process as in the previous homework (in slightly more detail, it was trained using an 8-GPU parallel version of that homework's trainind code), but instead of the TinyStories dataset, it was trained on about 1.1 trillion tokens from the [FineWebEDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset (this took about 3 days on 8 GPUs).  You won't need to do any of the re-training, but you should confirm that the model is able to generate text using the code below.
    """)
    return


@app.cell
def _(os):
    ### Downlaod the models an datasets for the assignment
    from huggingface_hub import hf_hub_download

    repo = "zkolter/Chat-Tuning-Homework"
    filenames = ["model_base.pth",
                 "model_chat.pth",
                 "params.json",
                 "ultrachat_short.json",
                 "ultrachat_dpo_neg.json",
                 "ultrachat_dpo_pos.json"]

    for filename in filenames:
        if not os.path.exists(filename):
            hf_hub_download(repo_id=repo, filename=filename, repo_type="model", local_dir=".")
    return


@app.cell
def _(LLM, json):
    ### load the model
    with open("params.json", "rt") as f:
        params = json.load(f)
    model = LLM(
        params["num_tokens"],
        params["dim"],
        params["n_heads"],
        params["max_seq_len"],
        params["ffn_dim"],
        params["n_layers"]
    )
    model.load_weights("model_base.pth")
    model.float();
    return (model,)


@app.cell
def _(generate, model, tiktoken):
    ### generate some text
    base_tokenizer = tiktoken.get_encoding("gpt2")
    # prompt = "In a surprising discovery,"
    prompt = "Who is the president of the US?"
    generate(model, base_tokenizer.encode(prompt,allowed_special="all"), base_tokenizer, temp=0.4, max_tokens=100);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part I - Chat training via supervised finetuninng
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the pretrained model above, in this part, you'll implement code to tune it for chat.  For this purpose, we're using a subject of the [Ultrachat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset, a common dataset used for this kind of finetuning.  The subset of these chats included in the homework just filter the original data to only include those conversations that tokenize to 1024 tokens or fewer (the sequence length for our model.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 1 - Conversion to chat format

    First, implement a funciton that converts a set of ultrachat-style messages to a text format with user and assistant tags.  Specifically, Ultrachat uses a message format like the following
    ```json
    [
        {
            'role': 'user',
            'content': 'This is a user question.'
        },
        {
            'role': 'assistant',
            'content': 'This is an assistant response.'
        },
        {
            'role': 'user',
            'content': 'This is a followup user question.'
        },
        {
            'role': 'assistant',
            'content': 'This is a boring conversation.'
        }
    ]
    ```

    You will want to convert this into a single text string that uses the tags `<USER>`, `</USER>`, `<ASSISTANT>`, and `</ASSISTANT>` to demarcate different portions of the conversation, with no additional newlines or whitespace.  For instance, the above content would become
    ```
    <USER>This is a user question</USER><ASSISTANT>This is an assistant response.</ASSISTANT><USER>This is a followup user question.</USER><ASSISTANT>This is a boring conversation.</ASSISTANT>
    ```
    """)
    return


@app.cell
def _(mugrade):
    @mugrade.local_tests
    def convert_to_chat_format(messages):
        """
        Convert a list of chat messages into a single tagged text string.

        Input:
            messages : list[dict[str, str]] - conversation with "role" and "content" keys
        Output:
            str - concatenated conversation using USER and ASSISTANT tags
        """
        ### BEGIN YOUR CODE
        output = []

        for message in messages:
            if message['role'] == "user":
                output.append(f"<USER>{message['content']}</USER>")

            elif message['role'] == "assistant":
                output.append(f"<ASSISTANT>{message['content']}</ASSISTANT>")

        return "".join(output)
        ### END YOUR CODE
    return (convert_to_chat_format,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 2 - Pretokenizing chat data

    As in the previous homework, it's often convenient to create a pre-tokenized version of the dataset for more rapid iteration at training time.  This function should process data from the `ultrachat_short.json` file (which contains a list of conversations in the above format), and write another json file that contains a list of the same conversations, but where each conversation is represented itself as a list of integer tokens, as produced by applying the tokenizer to the chat-formatted version of the conversation.

    As in the previous assignment, you'll want to tokenize the data using the function
    ```python
    tokenizer.encode(text, allowed_special="all")
    ```

    You should use the `json.load()` and `json.dump()` (with the optional argument `indent=4` if you want prettier outputs) to load and save the files.
    """)
    return


@app.cell
def _(convert_to_chat_format, json, mugrade):
    @mugrade.local_tests
    def pretokenize_chat(tokenizer, in_filename, out_filename):
        """
        Convert a chat dataset from json conversations to json token lists.

        Inputs:
            tokenizer : object - tokenizer with encode(text, allowed_special="all")
            in_filename : str - input json filename containing chat conversations
            out_filename : str - output json filename for tokenized conversations
        Output:
            None - writes tokenized conversations to out_filename
        """
        ### BEGIN YOUR CODE
        with open(in_filename, 'r') as f:
            conversations = json.load(f)

        tokenized_conversations = []
        for messages in conversations:
            chat_string = convert_to_chat_format(messages)
            tokenized_chat = tokenizer.encode(chat_string, allowed_special="all")
            tokenized_conversations.append(tokenized_chat)

        with open(out_filename, 'w') as f:
            json.dump(tokenized_conversations, f, indent=4)
        ### END YOUR CODE
    return (pretokenize_chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you have implemented the function, you should be able to convert the full `ultrachat_short.json` file to this pretokenized format using the following code.  Note that this build the tokenizer to explicitly add our special tokens, so that these elements are always tokenized to their special ids.  This function takes some time to run, as it has to parse all ~90K of the included conversations.
    """)
    return


@app.cell
def _(pretokenize_chat, tiktoken):
    base_tokenizer_1 = tiktoken.get_encoding('gpt2')
    tokenizer = tiktoken.Encoding(name='gpt2_chat', pat_str=base_tokenizer_1._pat_str, mergeable_ranks=base_tokenizer_1._mergeable_ranks, special_tokens={**base_tokenizer_1._special_tokens, '<USER>': 50257, '</USER>': 50258, '<ASSISTANT>': 50259, '</ASSISTANT>': 50300})
    pretokenize_chat(tokenizer, 'ultrachat_short.json', 'ultrachat_tokenized.json')
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 3 - Chat masking

    Write a function that will output a mask that characterizes which tokens should be trained on within a chat conversation.  For our purposes, we _only_ want to train on tokens that occur _after_ an `<ASSISTANT>` up through (and including) the `</ASSISTANT>` token.  So if the text was the following:
    ```
    <USER>This is a user question</USER><ASSISTANT>This is an assistant response.</ASSISTANT><USER>This is a followup user question.</USER><ASSISTANT>This is a boring conversation.</ASSISTANT>
    ```
    Then the mask should be `True` only for those tokens corresponding to the text `This is an assistant response.</ASSISTANT>` and `This is a boring conversation.</ASSISTANT>`
    """)
    return


@app.cell
def _(tokenizer):
    tokenizer.n_vocab
    return


@app.cell
def _(mugrade):
    @mugrade.local_tests
    def get_loss_mask(tokens, tokenizer):
        """
        Build a boolean mask selecting assistant-response tokens for training.

        Inputs:
            tokens : list[int] - tokenized chat conversation
            tokenizer : object - tokenizer with special token ids for assistant tags
        Output:
            list[bool] - True for tokens after <ASSISTANT> through </ASSISTANT>
        """
        ### BEGIN YOUR CODE
        mask = []

        switch = False
        for token in tokens:
            mask.append(switch)

            if token == tokenizer._special_tokens["<ASSISTANT>"]:
                switch = True

            elif token == tokenizer._special_tokens["</ASSISTANT>"]:
                switch = False

        return mask
        ### END YOUR CODE
    return (get_loss_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 4 - A data loader for chat

    Next, you'll build a data loader that loads (pretokenized) chat conversations and returns them as data suitable for training.  Unlike the data loader we built previously (which just operated on a binary storage file, so we could easily seek to different blocks of data), this one will just load the entire pretokenized json file and generate each batch on the fly.

    As opposed to the data loader in the previous homework, which returned two tensors `x` and `y` where `y` was a shifted version of `x`, this loader will return _three_ element as follows
    ```python
    for x,y,mask in loader:
        ...
    ```
    where `x` and `y` are the same as before (`batch_size` by `seq_len` integer tensors), but where `mask` is a `batch_size` by `seq_len` _boolean_ tensor formed by the above `get_loss_mask()` function.

    Although it isn't strictly necessary for supervised finetuning, in order for this same data loader to work for DPO, we're going to operate under the convention that each batch element consists of only a _single_ conversation, and you should zero-pad the end of the `x` and `y` outputs to ensure that they are all of length `seq_len`.  You will also zero-pad the end of `mask` with `False` entries, so that the zero-tokens at the end of the sequence are ignored by the loss function. You should also drop incomplete batches, i.e., batches which have size < `batch_size`.
    """)
    return


@app.cell
def _():
    # with open("ultrachat_tokenized.json", 'r') as f:
    #     data = json.load(f)
    return


@app.cell
def _(get_loss_mask, json, mugrade, torch):
    @mugrade.local_tests
    class DataLoaderChat:

        def __init__(self, filename, seq_len, batch_size, tokenizer, device='cpu'):
            """
            Initialize a chat data loader backed by a tokenized json file.

            Inputs:
                filename : str - json filename containing tokenized conversations
                seq_len : int - number of tokens per input sequence
                batch_size : int - number of conversations per minibatch
                tokenizer : object - tokenizer used to build assistant-response masks
                device : str - device on which to place each minibatch tensor
            """  ### BEGIN YOUR CODE
            with open(filename, 'r') as f:
                self.data = json.load(f)
            self.seq_len = seq_len
            self.batch_size = batch_size
            self.tokenizer = tokenizer
            self.device = device
            self.pos = 0

        def __iter__(self):  ### END YOUR CODE
            """
            Reset iteration state and return the iterator object.

            Output:
                DataLoaderChat - iterator over chat minibatches
            """
            self.pos = 0
            return self
      ### BEGIN YOUR CODE
        def __next__(self):
            """
            Return the next chat minibatch with masked targets.  ### END YOUR CODE

            Output:
                tuple(torch.Tensor, torch.Tensor, torch.Tensor) - input tokens, next-token targets, and boolean loss mask
            """
            if self.pos + self.batch_size > len(self.data):
                raise StopIteration
            batch_x = []
            batch_y = []
            batch_mask = []  ### BEGIN YOUR CODE
            for i in range(self.batch_size):
                tokens = self.data[self.pos + i]
                mask = get_loss_mask(tokens, self.tokenizer)
                x = tokens[:self.seq_len]
                y = tokens[1:self.seq_len + 1]
                m = mask[1:self.seq_len + 1]
                pad_len = self.seq_len - len(x)
                if pad_len > 0:
                    x = x + [0] * pad_len
                    y = y + [0] * pad_len
                    m = m + [False] * pad_len
                if len(y) < self.seq_len:  # Extract up to seq_len + 1 tokens for shifting
                    y = y + [0] * (self.seq_len - len(y))  # then slice to get x (0 to N-1) and y (1 to N)
                    m = m + [False] * (self.seq_len - len(m))
                batch_x.append(x)
                batch_y.append(y)
                batch_mask.append(m)
            self.pos = self.pos + self.batch_size
            return (torch.tensor(batch_x, device=self.device), torch.tensor(batch_y, device=self.device), torch.tensor(batch_mask, device=self.device))  # If y was shorter because tokens ended  ### END YOUR CODE

    return (DataLoaderChat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 5 - Chat training loop

    Write a function to finetune a model based upon chat training.  This should use essentially the same logic as your LLM training code in homework 5, with the only difference being the application of the mask to the relevant data.  Note that it is specifically helpful to use PyTorch's boolean indexing for this purpose.  That is, if e.g., `y` is a tensor of some size and `mask` is a boolean tensor of the same size then
    ```python
    y[mask]
    ```
    will return a 1D tensor _only_ of the entires in `y` corresponding to `True` entries in `mask`.  The same can be used to index based upon only the leading dimensions of a higher-order tensor.  You can see this e.g., in the following code:
    ```python
    A = torch.randn(3,4,5)
    mask = torch.randn(3,4) > 0
    print(A[mask])
    ```
    which will be a `n` by 5 dimensionsal tensor, where `n` is the total number of `True` entires in mask.

    We also pass an additional `max_iter` parameter to keep things simple, where the training loop should return after `max_iter` iterations.
    """)
    return


@app.cell
def _(cross_entropy_loss, mugrade):
    @mugrade.local_tests
    def train_llm_chat(model, chat_loader, opt, max_iter = None):
        """
        Run one pass of supervised chat finetuning with a masked next-token loss.

        Inputs:
            model : Module - language model mapping token sequences to logits
            chat_loader : iterable - yields minibatches of chat inputs, targets, and masks
            opt : optimizer - object with zero_grad() and step() methods
            max_iter : int or None - maximum number of minibatches to process
        Output:
            None - updates model parameters in place
        """
        ### BEGIN YOUR CODE
        model.train()

        for i, (x, y, mask) in enumerate(chat_loader):
            opt.zero_grad()

            logits = model(x)
            loss = cross_entropy_loss(logits[mask], y[mask])
            loss.backward()

            opt.step()

            print(f"Iteration {i+1}: Loss {loss.item()}")

            if max_iter is not None and i + 1 >= max_iter:
                break

        ### END YOUR CODE
    return (train_llm_chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you implemented the function above correctly, then the following code should perform a few iterations of chat tuning on the pretrained model.  Note that we recommend you finetune the model with 32 bit floating point precision (instead of the 16 bit precision you used for pretraining in Homework 5).
    """)
    return


@app.cell
def _(DataLoaderChat, model, tokenizer):
    loader = DataLoaderChat("ultrachat_tokenized.json", 1024, 2, tokenizer, device="cpu")
    model.load_weights("model_base.pth")
    model.float()
    return


@app.cell
def _():
    # print(next(model.parameters()).device)

    # opt = Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.95))
    # train_llm_chat(model, loader, opt, max_iter=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that we're training a model that is already fine-tuned, so you won't see a huge loss decrease, but the following code will test you system (the test cases mainly look to see if the chat tokens, especially the end assistant tag have increased in probability).
    """)
    return


@app.cell
def _():
    # @mugrade.local_tests
    # def eval_llm_chat():
    #     return model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can train a larger chat model (again using a GPU, which would be necessary for this), using the following code.
    """)
    return


@app.cell
def _(DataLoaderChat, model, tokenizer):
    loader_1 = DataLoaderChat('ultrachat_tokenized.json', 1024, 32, tokenizer, device='cuda')
    model.load_weights('model_base.pth')
    model.float().cuda()  # keep in float32 format for finetuning
    return (loader_1,)


@app.cell
def _(Adam, loader_1, model, train_llm_chat):
    opt = Adam(model.parameters(), lr=2e-05, betas=(0.9, 0.95))
    train_llm_chat(model, loader_1, opt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can try generating text from the model using the following code.
    """)
    return


@app.cell
def _(generate, model, tokenizer):
    prompt_1 = '<USER>What is the capital of the United States?</USER><ASSISTANT>'
    generate(model, tokenizer.encode(prompt_1, allowed_special='all'), tokenizer, eot_token=50300, temp=0.4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part II - Direct preference optimization

    In this section of the homework you'll implement the direct preference optimization (DPO) algorithm to further finetune the chat model with human preference data.  If you were able to finetune the complete chat model on the entire dataset, you can use that model as a starting point, but we also include a `model_chat.json` (which we already trained on the full dataset) that you can use as the starting poing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 6 - Log probability calculation

    For DPO, we will need the ability to compute the sum of log probabilities (i.e., negative cross entropy losses), not just averaged over the entire batch, but for each element in the batch separately.  Notationally, we will use the convention of computing the log proability instead of negating the cross-entropy loss, because this is more common in presentation of DPO, and it's helpful in better understanding the connection between the two perspectives.

    Implement the following function, which takes similar inputs to the cross entropy loss function, but which instead computes the _sum_ of the (masked) log probabilities (negative cross entropy loss, computed in the same manner), over each element of the batch separately.
    """)
    return


@app.cell
def _(mugrade):
    @mugrade.local_tests
    def log_probs(logits, y, mask):
        """
        Compute masked sequence log probabilities for each batch element.

        Inputs:
            logits : torch.Tensor[float] (batch_size x seq_len x num_tokens) - predicted logits
            y : torch.Tensor[int] (batch_size x seq_len) - desired next-token ids
            mask : torch.Tensor[bool] (batch_size x seq_len) - mask selecting tokens to include
        Output:
            torch.Tensor[float] (batch_size,) - summed masked log probabilities per example
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 7 - DPO Loss

    Next, implement a function that computes the DPO loss, which is defined as the following.  Given a triple of examples $x,y^+, y^-$ (prompt, positive completion, and negative completion), the loss is given by
    $$
    L_{\text{DPO}} = \text{softplus}\left (-\log\frac{p(y^+ | x)}{p_{\text{ref}}(y^+|x)} + \log\frac{p(y^- | x)}{p_{\text{ref}}(y^-|x)}, \beta \right )
    $$
    where $p(\cdot)$ denotes probability under the model being trained, and $p_\mathrm{ref}$ is a reference model that remains fixed (in this case, we'll use the initial model before DPO as a fixed reference), and where the softplus function is defined as
    $$
    \text{softplus}(x,\beta) = \log\left (1+e^{\beta x} \right ).
    $$
    [The softplus function acts as a softer version of a ReLU-like function, and capture the intuition that we want to minimize the term inside the DPO loss only when the inner term is positive, i.e., the probability of the negative completion is larger than the probability of the positive one).

    You shouldn't implement the softplus function using `log` or `exp` functions directly, but instead use the `torch.logaddexp()` function to compute it in a more numerically stable fashion.

    For efficiency and to ensure that you aren't actually modifying the reference model, you should ensure that all use of the reference model is done within a `torch.no_grad()` both, similar to how you implemented this in the optimizer.
    """)
    return


@app.cell
def _(mugrade):
    @mugrade.local_tests
    def softplus(x, beta):
        """
        Compute the beta-scaled softplus function elementwise.

        Inputs:
            x : torch.Tensor[float] (...) - input tensor
            beta : float - scaling factor inside the softplus
        Output:
            torch.Tensor[float] (...) - softplus(beta * x)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE
    return


@app.cell
def _(mugrade):
    @mugrade.local_tests
    def dpo_loss(model, model_ref, xp,yp,maskp, xn, yn, maskn, beta):
        """
        Compute the DPO loss for paired preferred and dispreferred completions.

        Inputs:
            model : Module - model being optimized
            model_ref : Module - fixed reference model
            xp : torch.Tensor[int] (batch_size x seq_len) - preferred input tokens
            yp : torch.Tensor[int] (batch_size x seq_len) - preferred next-token targets
            maskp : torch.Tensor[bool] (batch_size x seq_len) - preferred loss mask
            xn : torch.Tensor[int] (batch_size x seq_len) - dispreferred input tokens
            yn : torch.Tensor[int] (batch_size x seq_len) - dispreferred next-token targets
            maskn : torch.Tensor[bool] (batch_size x seq_len) - dispreferred loss mask
            beta : float - inverse temperature for the DPO objective
        Output:
            torch.Tensor[float] (batch_size,) - DPO loss for each preference pair
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 8 - DPO training loop

    Finally, we'll implement the training function for DPO.  To start, we'll note that this training loop will use _two_ data loaders, capturing the positive and negative examples (in the exact same order, so that you can iterate over them simultaneously).  The positive examples here were included in Ultrachat 200k, while the negative examples came using the same prompts to generate responses from our original chat-tuned model.
    """)
    return


@app.cell
def _(pretokenize_chat, tokenizer):
    ### pre-tokenize the two
    pretokenize_chat(tokenizer, "ultrachat_dpo_neg.json", "ultrachat_neg_tokenized.json")
    pretokenize_chat(tokenizer, "ultrachat_dpo_pos.json", "ultrachat_pos_tokenized.json")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, implement the following `train_dpo` function.  This should follow the same rough approach as the other training loops, but use the DPO loss instead of a cross entropy loss.
    """)
    return


@app.cell
def _(mugrade):
    @mugrade.local_tests
    def train_dpo(model, model_ref, loader_pos, loader_neg, opt, beta=0.1, max_iter = None):
        """
        Run one pass of DPO finetuning over paired positive and negative minibatches.

        Inputs:
            model : Module - model being optimized
            model_ref : Module - fixed reference model
            loader_pos : iterable - yields preferred chat minibatches
            loader_neg : iterable - yields dispreferred chat minibatches
            opt : optimizer - object with zero_grad() and step() methods
            beta : float - inverse temperature for the DPO objective
            max_iter : int or None - maximum number of minibatches to process
        Output:
            None - updates model parameters in place
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE
    return (train_dpo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 9 - Training a DPO model

    Finally, we can put all these functions together to train a DPO model.  We will start by initializing the data loaders and model.  You can dirctly use the model you trained above as a starting point, but for ease of use, we recommend that you use a version of the model that we fine-tuned using the entire data set above, stored in `model_chat.json`.  To start, you can train a small version on CPU.
    """)
    return


@app.cell
def _(DataLoaderChat, copy, model, tokenizer):
    loader_neg = DataLoaderChat("ultrachat_neg_tokenized.json", 1024, 2, tokenizer, device="cpu")
    loader_pos = DataLoaderChat("ultrachat_pos_tokenized.json", 1024, 2, tokenizer, device="cpu")
    model.load_weights("model_chat.pth") # comment out if you want to use your own model
    model.float().cpu();
    # shallow copy shares structure
    model_ref = copy.copy(model)
    model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
    model_ref = model_ref.cpu().float();
    return loader_neg, loader_pos, model_ref


@app.cell
def _(Adam, loader_neg, loader_pos, model, model_ref, train_dpo):
    opt_1 = Adam(model.parameters(), lr=1e-06, betas=(0.9, 0.95))
    train_dpo(model, model_ref, loader_pos, loader_neg, opt_1, max_iter=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can test the trained model using the following code.
    """)
    return


@app.cell
def _(model, mugrade):
    @mugrade.local_tests
    def eval_llm_dpo():
        return model

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally, if you want you can train a GPU version with the entire dataset.
    """)
    return


@app.cell
def _(DataLoaderChat, copy, model, tokenizer):
    ### Set up data loaders for large DPO run
    loader_neg_1 = DataLoaderChat('ultrachat_neg_tokenized.json', 1024, 16, tokenizer, device='cuda')
    loader_pos_1 = DataLoaderChat('ultrachat_pos_tokenized.json', 1024, 16, tokenizer, device='cuda')
    model.load_weights('model_chat.pth')  # comment out if you want to use your own model
    model.float().cuda()
    model_ref_1 = copy.deepcopy(model)
    return loader_neg_1, loader_pos_1, model_ref_1


@app.cell
def _(Adam, loader_neg_1, loader_pos_1, model, model_ref_1, train_dpo):
    ### Train DPO
    opt_2 = Adam(model.parameters(), lr=1e-06, betas=(0.9, 0.95))
    train_dpo(model, model_ref_1, loader_pos_1, loader_neg_1, opt_2)
    return


@app.cell
def _(generate, model, tokenizer):
    ### Test some prompt generation
    prompt_2 = '<USER>Write a poem about flowers.</USER><ASSISTANT>'
    generate(model, tokenizer.encode(prompt_2, allowed_special='all'), tokenizer, eot_token=50300, temp=0.4)
    return


if __name__ == "__main__":
    app.run()
