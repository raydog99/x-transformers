open Torch

module TiTokTokenizer = struct
  type t = {
    image_size: int;
    latents: Tensor.t;
    pos_emb: Tensor.t;
    mask_tokens: Tensor.t;
    image_to_tokens: Layer.t;
    encoder: Layer.t;
    vq: VQ.t;
    decoder: Layer.t;
    tokens_to_image: Layer.t;
  }

  let create ~dim 
      ?(image_size=256) 
      ?(patch_size=32) 
      ?(channels=3) 
      ?(num_latent_tokens=32) 
      ?(enc_depth=6) 
      ?(enc_heads=8) 
      ?(enc_dim_head=64) 
      ?(dec_depth=6) 
      ?(dec_heads=8) 
      ?(dec_dim_head=64) 
      ?(codebook_size=8192) 
      ?(enc_kwargs=[]) 
      ?(dec_kwargs=[]) 
      ?(vq_kwargs=[]) () =

    assert (image_size mod patch_size = 0);

    let dim_patch = channels * (patch_size * patch_size) in
    let num_tokens = (image_size / patch_size) * (image_size / patch_size) in

    let latents = Tensor.new_zeros [|num_latent_tokens; dim|] |> Tensor.normal_ ~std:0.02 in
    let pos_emb = Tensor.new_zeros [|num_tokens; dim|] |> Tensor.normal_ ~std:0.02 in
    let mask_tokens = Tensor.new_zeros [|num_tokens; dim|] |> Tensor.normal_ ~std:0.02 in

    let image_to_tokens = 
      Layer.sequential [
        Layer.fn (fun x -> Tensor.view x ~size:[|Tensor.size x 0; channels; patch_size; patch_size|]);
        Layer.linear ~input_dim:dim_patch dim
      ]
    in

    let encoder = Encoder.create ~dim ~depth:enc_depth ~heads:enc_heads ~attn_dim_head:enc_dim_head enc_kwargs in
    let vq = VQ.create ~dim ~codebook_dim:dim ~codebook_size vq_kwargs in
    let decoder = Encoder.create ~dim ~depth:dec_depth ~heads:dec_heads ~attn_dim_head:dec_dim_head dec_kwargs in

    let tokens_to_image = 
      Layer.sequential [
        Layer.linear ~input_dim:dim dim_patch;
        Layer.fn (fun x -> Tensor.view x ~size:[|Tensor.size x 0; channels; patch_size; patch_size|])
      ]
    in

    { image_size; latents; pos_emb; mask_tokens; image_to_tokens; encoder; vq; decoder; tokens_to_image }

  let tokenize t images =
    let open Torch in
    Tensor.no_grad (fun () -> forward t images ~return_codebook_ids:true)

  let codebook_ids_to_images t token_ids =
    let codes = VQ.get_output_from_indices t.vq token_ids in
    decode t codes

  let decode t latents =
    let open Torch in
    let batch = Tensor.size latents 0 in

    let mask_tokens = Tensor.repeat_interleave t.mask_tokens ~repeats:batch ~dim:0 in
    let tokens = Tensor.cat2 mask_tokens latents ~dim:1 in

    let tokens = Layer.forward t.decoder tokens in
    let tokens = Tensor.view tokens ~size:[|batch; t.image_size; t.image_size; -1|] in

    Layer.forward t.tokens_to_image tokens

  let forward t images ?(return_codebook_ids=false) ?(return_recon_images=false) =
    let open Torch in
    assert (Tensor.dim images = 4 && Tensor.size images 1 = t.image_size && Tensor.size images 2 = t.image_size);

    let batch = Tensor.size images 0 in

    let tokens = Layer.forward t.image_to_tokens images in
    let tokens = Tensor.view tokens ~size:[|batch; t.image_size; t.image_size; -1|] in

    let pos_emb = Tensor.repeat_interleave t.pos_emb ~repeats:batch ~dim:0 in
    let tokens = Tensor.(tokens + pos_emb) in

    let latents = Tensor.repeat_interleave t.latents ~repeats:batch ~dim:0 in
    let tokens = Tensor.cat2 tokens latents ~dim:1 in

    let tokens = Layer.forward t.encoder tokens in
    let _, latents = Tensor.split_with_size tokens ~sizes:[|Tensor.size t.latents 0; -1|] ~dim:1 in

    let quantized, indices, _ = VQ.forward t.vq latents in

    if return_codebook_ids then
      indices
    else
      let recon_images = decode t quantized in
      let recon_loss = Tensor.mse_loss recon_images images in

      if return_recon_images then
        (recon_loss, recon_images)
      else
        recon_loss
end