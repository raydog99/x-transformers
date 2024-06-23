open Torch

let attention dim ~heads ~dim_head ~and_self_attend =
  let inner_dim = heads * dim_head in
  let scale = Float.pow (float_of_int dim_head) (-0.5) in

  let to_q = nn_linear ~input_dim:dim ~output_dim:inner_dim ~bias:false () in
  let to_kv = nn_linear ~input_dim:dim ~output_dim:(inner_dim * 2) ~bias:false () in
  let to_out = nn_linear ~input_dim:inner_dim ~output_dim:dim ~bias:false () in

  fun x context ?mask () ->
    let h = heads in
    let scale = scale in

    let context =
      if and_self_attend then
        Tensor.cat [x; context] ~dim:(-2)
      else context
    in

    let mask =
      match mask with
      | Some mask -> F.pad mask ~padding:(x.shape.(0), 0) ~value:(Tensor.ones_like mask)
      | None -> None
    in

    let q = x |> to_q in
    let k, v = context |> to_kv |> Tensor.chunk ~chunks:2 ~dim:(-1) in

    let q, k, v =
      (q, k, v)
      |> List.map (fun t -> Torch.rearrange t ~dims:"b n (h d)" ~args:[Dim h; Dim dim_head])
    in

    let dots = Torch.einsum "b h i d, b h j d -> b h i j" [q; k] |> Tensor.mul_scalar scale in

    begin match mask with
      | Some mask ->
          let mask_value = -1e9 in
          let mask = Tensor.rearrange mask ~dims:"b n" ~args:[Dim 1; Dim 1; Dim 1; Dim 1] in
          Tensor.masked_fill_ dots ~mask ~value:mask_value
      | None -> ()
    end;

    let attn = dots |> Tensor.softmax ~dim:(-1) in
    let out = Torch.einsum "b h i j, b h j d -> b h i d" [attn; v] in
    let out = Torch.rearrange out ~dims:"b h n d" ~args:[Dim h; Dim dim_head] in
    out |> to_out