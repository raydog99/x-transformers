open Base
open Torch

module TwinsSVT : sig
  type t

  val create :
    num_classes:int ->
    s1_emb_dim:int ->
    s1_patch_size:int ->
    s1_local_patch_size:int ->
    s1_global_k:int ->
    s1_depth:int ->
    s2_emb_dim:int ->
    s2_patch_size:int ->
    s2_local_patch_size:int ->
    s2_global_k:int ->
    s2_depth:int ->
    s3_emb_dim:int ->
    s3_patch_size:int ->
    s3_local_patch_size:int ->
    s3_global_k:int ->
    s3_depth:int ->
    s4_emb_dim:int ->
    s4_patch_size:int ->
    s4_local_patch_size:int ->
    s4_global_k:int ->
    s4_depth:int ->
    peg_kernel_size:int ->
    dropout:float ->
    t
end = struct
  type t = {
    s1_emb_dim : int;
    s1_patch_size : int;
    s1_local_patch_size : int;
    s1_global_k : int;
    s1_depth : int;
    s2_emb_dim : int;
    s2_patch_size : int;
    s2_local_patch_size : int;
    s2_global_k : int;
    s2_depth : int;
    s3_emb_dim : int;
    s3_patch_size : int;
    s3_local_patch_size : int;
    s3_global_k : int;
    s3_depth : int;
    s4_emb_dim : int;
    s4_patch_size : int;
    s4_local_patch_size : int;
    s4_global_k : int;
    s4_depth : int;
    peg_kernel_size : int;
    dropout : float;
    layers : nn.ModuleList.t;
  }

  let create ~num_classes ~s1_emb_dim ~s1_patch_size ~s1_local_patch_size ~s1_global_k ~s1_depth
             ~s2_emb_dim ~s2_patch_size ~s2_local_patch_size ~s2_global_k ~s2_depth
             ~s3_emb_dim ~s3_patch_size ~s3_local_patch_size ~s3_global_k ~s3_depth
             ~s4_emb_dim ~s4_patch_size ~s4_local_patch_size ~s4_global_k ~s4_depth
             ~peg_kernel_size ~dropout =
    let dim = ref 3 in
    let layers = ref [] in
  ;;
end