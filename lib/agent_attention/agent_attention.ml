open Torch

module AgentSelfAttention = struct
  let scale dim_head heads = 2.0 ** (-0.5) /. float_of_int dim_head *. float_of_int heads

  let create
      ~dim
      ~num_agent_tokens
      ?(dim_head = 64)
      ?(heads = 8)
      ?(dropout = 0.)
      ?(talking_heads = true)
      ?(gate = true)
      ?(combine_agent_tokens = false)
      () =
    let scale_factor = scale dim_head heads in

    let to_qkv =
      nn.Sequential.of_list
        [
          nn.Linear.create dim (dim_head * heads * 3) ~bias:false;
          Rearrange.create "b n (qkv h d) -> qkv b h n d" ~h:heads ~qkv:3;
        ]
    in

    let to_gates =
      if gate then
        Some
          (nn.Sequential.of_list
             [
               nn.Linear.create dim heads;
               Rearrange.create "b n h -> b h n 1";
               nn.Sigmoid.create ();
             ])
      else
        None
    in

    let agent_tokens =
      let tokens = Tensor.zeros [| heads; num_agent_tokens; dim_head |] in
      Tensor.(nn.init.normal_ (Parameter.create tokens) ~std:0.02)
    in

    let qa_talking_heads =
      if talking_heads then nn.Conv2d.create heads heads 1 ~bias:false
      else nn.Identity.create ()
    in

    let ak_talking_heads =
      if talking_heads then nn.Conv2d.create heads heads 1 ~bias:false
      else nn.Identity.create ()
    in

    let qa_dropout = nn.Dropout.create dropout in
    let ak_dropout = nn.Dropout.create dropout in

    let to_out =
      nn.Sequential.of_list
        [
          Rearrange.create "b h n d -> b n (h d)";
          nn.Linear.create (dim_head * heads) dim ~bias:false;
        ]
    in

    object
      method forward
          (x : Tensor.t)
          ?(mask : Tensor.t option)
          ?(agent_tokens : Tensor.t option)
          ?(return_agent_tokens : bool = false)
          : Tensor.t =
        let batch = Tensor.shape x |> Array.get 0 in

        let q, k, v = to_qkv#forward x in

        let a =
          match agent_tokens with
          | Some a -> a
          | None -> Tensor.(repeat agent_tokens ~repeats:[| batch; 1; 1; 1 |])
        in

        let a = Tensor.(a * scale_factor) in

        let qa_sim = Tensor.einsum "b h i d, b h j d -> b h i j" [q; a] in
        let ak_sim = Tensor.einsum "b h i d, b h j d -> b h i j" [a; k] in

        let max_neg_value = Float.neg_infinity in

        let ak_sim =
          match mask with
          | Some mask -> Tensor.(ak_sim.masked_fill ~value:max_neg_value (Tensor.rearrange mask "b j -> b 1 1 j"))
          | None -> ak_sim
        in

        let qa_attn = Tensor.softmax qa_sim ~dim:-1 in
        let ak_attn = Tensor.softmax ak_sim ~dim:-1 in

        let qa_attn = qa_attn |> qa_dropout#forward |> qa_talking_heads#forward in
        let ak_attn = ak_attn |> ak_dropout#forward |> ak_talking_heads#forward in

        let agent_gathered_tokens = Tensor.einsum "b h i j, b h j d -> b h i d" [ak_attn; v] in

        let out = Tensor.einsum "b h i j, b h j d -> b h i d" [qa_attn; agent_gathered_tokens] in

        let out =
          match mask with
          | Some mask -> Tensor.(out.masked_fill ~value:0. (Tensor.rearrange mask "b n -> b 1 n 1"))
          | None -> out
        in

        let out =
          match to_gates with
          | Some gates -> Tensor.(out * gates#forward x)
          | None -> out
        in

        if not return_agent_tokens then out
        else (out, agent_gathered_tokens)
    end
end