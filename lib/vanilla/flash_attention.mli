type flash_attention

val attention :
  query:Tensor.t ->
  key:Tensor.t ->
  value:Tensor.t ->
  mask:Tensor.t option ->
  causal:bool ->
  q_bucket_size:int ->
  k_bucket_size:int ->
  unit

val create :
  unit ->
  flash_attention

val forward :
  flash_attention ->
  query:Tensor.t ->
  key:Tensor.t ->
  value:Tensor.t ->
  mask:Tensor.t option ->
  causal:bool ->
  q_bucket_size:int ->
  k_bucket_size:int ->
  Tensor.t