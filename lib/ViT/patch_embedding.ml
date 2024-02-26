open Torch

module PatchEmbedding = struct
  type t = {
    to_patch_embedding: Layer.Sequential.t;
    pos_embedding: Tensor.t;
  }

  let posemb_sincos_2d ?(temperature=10000.) patches =
    let h, w, dim, device, dtype = 
      Tensor.(to_tuple (shape patches), device patches, kind patches) in

    let y, x = Tensor.meshgrid (Tensor.arange ~device ~dtype h) (Tensor.arange ~device ~dtype w) ~indexing:`ij in

    assert ((dim mod 4) = 0) "feature dimension must be multiple of 4 for sincos emb";
    let omega = Tensor.arange ~device ~dtype:(Kind.float32 ()) (float_of_int (dim / 4)) / float_of_int (dim / 4 - 1) in
    let omega = Tensor.pow (Scalar.Float temperature) omega in

    let y = Tensor.(flatten y |> unsqueeze ~dim:1) *^ omega ^|> sin in
    let x = Tensor.(flatten x |> unsqueeze ~dim:1) *^ omega ^|> cos in
    let pe = Tensor.cat ~dim:1 [x; y] in

    Tensor.to_type ~type_:(Kind.float32 ()) pe

  let create vs image_size patch_size channels dim =
    let pair x = (x, x) in
    let image_height, image_width = pair image_size in
    let patch_height, patch_width = pair patch_size in

    assert (image_height % patch_height = 0 && image_width % patch_width = 0)
      "Image dimensions must be divisible by the patch size.";

    let patch_dim = channels * patch_height * patch_width in

    let to_patch_embedding =
      Layer.Sequential.create
        [ Layer.Rearrange.create "b c (h p1) (w p2) -> b (h w) (p1 p2 c)"
            ~p1:patch_height ~p2:patch_width;
          Layer.LayerNorm.create patch_dim;
          Layer.Linear.create vs "linear" ~input_dim:patch_dim ~output_dim:dim;
          Layer.LayerNorm.create dim
        ]
    in

    let pos_embedding =
      let h = image_height / patch_height in
      let w = image_width / patch_width in
      posemb_sincos_2d ~h ~w ~dim
    in

    { to_patch_embedding; pos_embedding }
end