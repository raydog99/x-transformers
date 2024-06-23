open Torch

let exists v = Torch.(v <> none)

let default v d = if exists v then v else d

let inner_dot_product x y ~dim ~keepdim =
  Torch.(sum (x * y) ~dim ~keepdim)

module LayerNorm = struct
  let create dim =
    let gamma = Var (torch_ones [dim]) in
    let beta = Var (torch_zeros [dim]) in
    fun x ->
      Layer.norm x ~normalized_shape:[|x.shape.(Array.length x.shape - 1)|] ~weight:gamma ~bias:beta
end

module VNLinear = struct
  let create dim_in dim_out ~bias_epsilon =
    let weight = Var (torch_randn [dim_out; dim_in]) in
    let bias =
      if bias_epsilon > 0.
      then Some (Var (torch_randn [dim_out]))
      else None
    in
    fun x ->
      let out = Torch.einsum "i...c,oi->o...c" [x; weight] in
      match bias with
      | Some b ->
        let bias = Tensor.(F.normalize b ~dim:(-1) * bias_epsilon) in
        Tensor.(out + (rearrange bias "...->... 1"))
      | None -> out
end