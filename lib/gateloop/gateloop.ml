open Torch

module CausalFullAttention = struct
  type t = {
    scale: float;
    rotary_emb: RotaryEmbedding.t option;
    to_qkv: Layer.t;
    data_dependent_rel_pos: bool;
    frac_gradient_data_dependent_rel_pos: float;
    to_a: Layer.t option;
    to_gates: Layer.t option;
    to_out: Layer.t;
    softmax_normalize: bool;
  }

  let create 
      ~dim 
      ?(dim_head=64) 
      ?(heads=8) 
      ?(rotary_emb=false) 
      ?(add_swish_gating=false) 
      ?(data_dependent_rel_pos=false) 
      ?(frac_gradient_data_dependent_rel_pos=0.5) 
      ?(softmax_normalize=None) () =

    let dim_inner = dim_head * heads in
    let softmax_normalize = Option.value softmax_normalize ~default:(not data_dependent_rel_pos) in
    let scale = 1. /. sqrt (float dim_head) in
    let rotary_emb = if rotary_emb then Some (RotaryEmbedding.create dim_head) else None in
    let to_qkv = 
      Layer.sequential [
        Layer.linear ~input_dim:dim (dim_inner * 3);
        Layer.fn (fun x -> Tensor.view x ~size:[|Tensor.size x 0; heads; -1; 3 * dim_head|])
      ]
    in

    let to_a = if data_dependent_rel_pos then
        Some (
          Layer.sequential [
            Layer.linear ~input_dim:dim dim_inner;
            Layer.fn (fun x -> Tensor.view x ~size:[|Tensor.size x 0; heads; -1; dim_head; 2|])
          ]
        )
      else None
    in

    let to_gates = if add_swish_gating then
        Some (
          Layer.sequential [
            Layer.linear ~input_dim:dim dim_inner;
            Layer.silu ();
            Layer.fn (fun x -> Tensor.view x ~size:[|Tensor.size x 0; heads; -1; dim_head|])
          ]
        )
      else None
    in

    let to_out = 
      Layer.sequential [
        Layer.fn (fun x -> Tensor.view x ~size:[|Tensor.size x 0; -1|]);
        Layer.linear ~input_dim:dim_inner dim
      ]
    in

    { scale; rotary_emb; to_qkv; data_dependent_rel_pos; frac_gradient_data_dependent_rel_pos; to_a; to_gates; to_out; softmax_normalize }

  let forward t x ?(ablate_complex=false) ?(ablate_state_transition=false) =
    let qkv = Layer.forward t.to_qkv x in
    let q, k, v = Tensor.chunk qkv ~chunks:3 ~dim:(-1) in
    let q, k = 
      match t.rotary_emb with
      | Some emb -> RotaryEmbedding.rotate_queries_or_keys emb q, RotaryEmbedding.rotate_queries_or_keys emb k
      | None -> q, k
    in
    let q = Tensor.(q * f t.scale) in

    let q, k = if t.data_dependent_rel_pos && not ablate_state_transition then
        let frac_gradient = t.frac_gradient_data_dependent_rel_pos in
        match t.to_a with
        | Some to_a ->
            let a = Layer.forward to_a x in
            let a = Tensor.(a * f frac_gradient + detach a * (f (1. -. frac_gradient))) in
            let a = Tensor.view_as_complex a in
            let a = if ablate_complex then Tensor.real a +. Tensor.zeros_like a else a in
            let magnitude, phase = Tensor.(abs a, angle a) in
            let a = Tensor.polar (Tensor.sigmoid magnitude) phase in
            let a_cumprod = Tensor.cumprod a ~dim:(-2) in
            let a_cumprod_real = Tensor.clamp (Tensor.real a_cumprod) ~min:1e-10 in
            let a_cumprod_real_inverse = Tensor.(f 1. /. a_cumprod_real) in
            let q, k = Tensor.(view q ~size:[|dim q 0; dim q 1; dim q 2; -1; 2|]), Tensor.(view k ~size:[|dim k 0; dim k 1; dim k 2; -1; 2|]) in
            let q = Tensor.(q * a_cumprod_real) in
            let k = Tensor.(k * a_cumprod_real_inverse) in
            Tensor.(view q ~size:[|dim q 0; dim q 1; dim q 2; -1|]), Tensor.(view k ~size:[|dim k 0; dim k 1; dim k 2; -1|])
        | None -> q, k
      else q, k
    in

    let sim = Tensor.einsum ~equation:"b h i d, b h j d -> b h i j" [q; k] in
    let i, j = Tensor.shape sim |> List.rev |> List.hd, Tensor.shape sim |> List.rev |> List.hd in
    let causal_mask = Tensor.ones [|i; j|] ~dtype:Torch.Bool ~device:(Tensor.device x) |> Tensor.triu ~diagonal:(j - i + 1) in
    let attn = if t.softmax_normalize then
        let sim = Tensor.masked_fill sim ~mask:causal_mask ~value:(-1. *. Tensor.finfo sim).Tensor.max in
        Tensor.softmax sim ~dim:(-1)
      else
        Tensor.masked_fill sim ~mask:causal_mask ~value:0.
    in

    let out = Tensor.einsum ~equation:"b h i j, b h j d -> b h i d" [attn; v] in
    let out = match t.to_gates with
      | Some to_gates -> Tensor.(out * Layer.forward to_gates x)
      | None -> out
    in
    Layer.forward t.to_out out
end