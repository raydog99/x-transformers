open Base
open Torch

let residual fn x =
  let fn_out = fn x in
  Tensor.add fn_out x

let upsample ?(dim_out=None) dim =
  Sequential.of_list [
    Upsample.create ~scale_factor:2 ~mode:`Nearest ();
    Conv2d.conv2d_ ?padding:Some 1 ~stride:1 ~kernel_size:(3,3) ~input_dim:dim ~output_dim:(default dim_out dim) ()
  ]

let downsample ?(dim_out=None) dim =
  Sequential.of_list [
    Rearrange.create "b c (h p1) (w p2) -> b (c p1 p2) h w" ~p1:2 ~p2:2;
    Conv2d.conv2d_ ~stride:1 ~kernel_size:(1,1) ~input_dim:(dim * 4) ~output_dim:(default dim_out dim) ()
  ]

let layer_norm dim =
  object
    val g = Var_store.new_var ~name:"gamma" (Tensor.ones ~shape:[|1; dim; 1; 1|])
    val b = Var_store.new_var ~name:"beta" (Tensor.zeros ~shape:[|1; dim; 1; 1|])

    method forward x =
      let eps = if Tensor.float_type () = `Float32 then 1e-5 else 1e-3 in
      let variance = Tensor.var ~dim:(List.init (Tensor.numel x) ~f:Fn.id) x ~unbiased:false ~keepdim:true in
      let mean = Tensor.mean ~dim:(List.init (Tensor.numel x) ~f:Fn.id) x ~keepdim:true in
      let norm = ((Tensor.sub x mean) * (Tensor.rsqrt (Tensor.add variance eps))) * g#value + if Option.is_none b then 0 else 0 in
      norm
  end

let sinusoidal_pos_emb dim =
  object
    method forward x =
      let device = Tensor.device x in
      let half_dim = dim / 2 in
      let emb = Float.log 10000. / Float.to_int (half_dim - 1) in
      let emb = (Tensor.exp (Tensor.arange ~device half_dim) * (-emb)) * x#to_float32 in
      let emb = Tensor.cat (List [Tensor.sin emb; Tensor.cos emb]) ~dim:1 in
      emb
  end