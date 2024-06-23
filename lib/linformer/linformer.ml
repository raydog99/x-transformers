open Torch

class residual (fn : Tensor.t -> Tensor.t) =
  object (self)
    method forward (x : Tensor.t) : Tensor.t = Tensor.(x + fn x)
  end

class preNorm (dim : int) (fn : Tensor.t -> Tensor.t) =
  object (self)
    val norm = LayerNorm.create [| dim |]
    val fn = fn

    method forward (x : Tensor.t) : Tensor.t =
      Tensor.(x |> norm |> fn)
  end

class gelu_ =
  object
    method forward (x : Tensor.t) : Tensor.t =
      let open Tensor.D in
      x * (f 0.5 * (f 1 + tanh (sqrt (f 2. / f Float.pi) * (x + f 0.044715 * pow x 3.))))
  end