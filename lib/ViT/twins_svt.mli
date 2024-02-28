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
end