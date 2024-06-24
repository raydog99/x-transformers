open Torch

module VectorQuantizer = struct
  type t = { embedding : Tensor.t; k : int; d : int; beta : float }

  let create num_embeddings embedding_dim beta =
    let k = num_embeddings in
    let d = embedding_dim in
    let beta = beta in
    let embedding = Tensor.empty [ k; d ] |> Tensor.uniform_ ~a:(-1. /. float k) ~b:(1. /. float k) in
    { embedding; k; d; beta }

  let forward { embedding; k; d; beta } latents =
    let latents = Tensor.permute latents ~dims:[ 0; 2; 3; 1 ] |> Tensor.contiguous in
    let latents_shape = Tensor.shape latents in
    let flat_latents = Tensor.view latents ~size:[ -1; d ] in

    let dist =
      Tensor.(sum (flat_latents ** 2.) ~dtype:Dtype.Float |> unsqueeze ~dim:1)
      + Tensor.sum (embedding ** 2.) ~dtype:Dtype.Float ~dim:(List.tl_exn (Tensor.shape embedding))
      - (Tensor.matmul flat_latents (Tensor.transpose embedding ~dim0:0 ~dim1:1))
    in

    let encoding_inds = Tensor.argmin dist ~dim:1 |> Tensor.unsqueeze ~dim:1 in

    let encoding_one_hot = Tensor.zeros [ List.hd_exn (Tensor.shape encoding_inds); k ] ~options:(Torch_device.Cpu (Torch_device.Cpu.mk ())) in
    Tensor.scatter_ encoding_one_hot ~dim:1 ~index:encoding_inds ~src:(Tensor.ones [ List.hd_exn (Tensor.shape encoding_inds) ] ~options:(Torch_device.Cpu (Torch_device.Cpu.mk ()))) ;

    let quantized_latents = Tensor.matmul encoding_one_hot embedding |> Tensor.view ~size:latents_shape in

    let commitment_loss = Tensor.mse_loss quantized_latents latents ~reduction:Reduction.Mean in
    let embedding_loss = Tensor.mse_loss quantized_latents latents ~reduction:Reduction.Mean in

    let vq_loss = Tensor.mul commitment_loss beta + embedding_loss in

    let quantized_latents = latents + (quantized_latents - latents) in

    Tensor.permute quantized_latents ~dims:[ 0; 3; 1; 2 ] |> Tensor.contiguous, vq_loss
end

module ResidualLayer = struct
  let create in_channels out_channels =
    let resblock =
      Seq.init 1 ~f:(fun x -> (nn_conv2d ~padding:1 ~stride:1 ~kernel_size:3 in_channels out_channels ~bias:false x), nn_relu x, nn_conv2d ~bias:false ~kernel_size:1 out_channels out_channels x)
    in
    fun input -> Tensor.add input (Seq.fold ~init:input ~f:fun a x -> x a) 
end

module VQVAE = struct
  type t = { encoder : Tensor.t; vq_layer : VectorQuantizer.t; decoder : Tensor.t; embedding_dim : int; num_embeddings : int; img_size : int; beta : float }

  let create in_channels embedding_dim num_embeddings ?(hidden_dims = Some [ 128; 256 ]) ?(beta = Some 0.25) ?(img_size = Some 64) ?(kwargs = Some) () =
    let t = { embedding_dim; num_embeddings; img_size; beta } in
    let vq_layer = VectorQuantizer.create num_embeddings embedding_dim beta in

    let encoder = Seq.(create [| for h_dim in hidden_dims -> Sequential.create [| nn_conv2d out_channels:h_dim ~stride:2 ~padding:1 ~kernel_size:4 in_channels out_channels:h_dim ~relu:Leaky_ReLU ||| nn_conv2d in_channels out_channels ~padding:1 ~kernel_size:3 ~relu:Leaky_ReLU ||| for _ in 6 -> ResidualLayer.create in_channels in_channels ||| nn_conv2d in_channels out_channels:embedding_dim ~stride:1 ~kernel_size:1 ~relu:Leaky_ReLU |] |] ) 
 in

 VQVAE.forward input kwargs encoding vq_layer = let result [ B quantized results let loss , : embeddings ] return result
end