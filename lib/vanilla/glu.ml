open Torch

let glu x w v =
  Tensor.sigmoid (Tensor.mm x w) * Tensor.mm x v

let bilinear x w v =
  Tensor.mm x w * Tensor.mm x v  

let reglu x w v =
  Tensor.relu (Tensor.mm x w) * Tensor.mm x v

let geglu x w v =
  Tensor.gelu (Tensor.mm x w) * Tensor.mm x v

let swiglu x w v =
  (Tensor.sigmoid (Tensor.mm x w) * Tensor.mm x w) * Tensor.mm x v