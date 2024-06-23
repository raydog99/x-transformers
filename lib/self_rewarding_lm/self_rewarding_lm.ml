open Torch

type rewardConfig = {
  prompt_template : string;
  reward_regex_template : string option;
  parse_reward : (string -> float option) option;
  template_fn : (string -> string);
  auto_dedent : bool;
}

let init (self : rewardConfig) =
  if self.auto_dedent then
    self.prompt_template <- String.dedent self.prompt_template;
    match self.reward_regex_template with
    | Some regex_template -> self.reward_regex_template <- Some (String.dedent regex_template)
    | None -> ();

  let prompt_template = self.prompt_template in
  assert (find_variables_from_jinja_template prompt_template == ["prompt"; "response"]) "template must include prompt and response templating variables";
  self.template_fn <- Jinja2_env.from_string prompt_template |> Jinja2_env.render;

  if Option.is_none self.parse_reward then
    assert (Option.is_some self.reward_regex_template) "reward_regex_template must be given if parse_reward is not passed in";
    self.parse_reward <- Some (create_parse_reward_fn (Option.get self.reward_regex_template));

  self