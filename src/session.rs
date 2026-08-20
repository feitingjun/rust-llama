use crate::llama::Llama;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::LlamaChatMessage;
use regex::Regex;
use std::num::NonZeroU32;

pub struct LlamaSessionParams {
  n_ctx: u32,
  think: bool,
}

pub struct LlamaSession<'a> {
  // 上下文
  ctx: LlamaContext<'a>,
  llama: &'a Llama,
  messages: Vec<LlamaChatMessage>,
  // context中的KV缓存token数量，用于创建LlamaBatch时指定新的token位置，便于使用KV缓存
  pos: u32,
}

impl LlamaSessionParams {
  pub fn default() -> LlamaSessionParams {
    LlamaSessionParams {
      n_ctx: 2048,
      think: true,
    }
  }

  pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
    self.n_ctx = n_ctx;
    self
  }

  pub fn with_think(mut self, think: bool) -> Self {
    self.think = think;
    self
  }
}

impl<'a> LlamaSession<'a> {
  // 创建会话
  pub fn new(llama: &'a Llama, params: LlamaSessionParams) -> LlamaSession<'a> {
    // 创建上下文
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(params.n_ctx));
    let ctx = llama.new_context(ctx_params).expect("创建上下文失败");

    // 创建初始消息
    let mut msg = Vec::new();
    // 如果 think = false 关闭深度思考
    if !params.think {
      msg.push(LlamaChatMessage::new(String::from("system"), String::from("/nothink")).unwrap());
    }
    LlamaSession {
      ctx,
      llama,
      messages: msg,
      pos: 0,
    }
  }

  // 添加消息
  pub fn add_message(&mut self, message: LlamaChatMessage) -> () {
    self.messages.push(message);
  }

  // 添加提示词
  pub fn add_system_message(&mut self, message: &str) -> () {
    self.add_message(LlamaChatMessage::new(String::from("system"), message.to_string()).unwrap());
  }

  // 添加用户消息
  pub fn add_user_message(&mut self, message: &str) -> () {
    self.add_message(LlamaChatMessage::new(String::from("user"), message.to_string()).unwrap());
  }

  // 添加回复
  pub fn add_assistant_message(&mut self, message: &str) -> () {
    self
      .add_message(LlamaChatMessage::new(String::from("assistant"), message.to_string()).unwrap());
  }

  // 清空消息
  pub fn clear_messages(&mut self) -> () {
    self.messages.clear();
  }

  // 对话
  pub fn prompt(
    &mut self,
    message: &str,
    token_gen_cb: Option<fn(String)>,
  ) -> Result<String, String> {
    // 添加用户消息
    self.add_user_message(message);

    let template = self
      .ctx
      .model
      .chat_template(None)
      .map_err(|e| format!("获取聊天模板失败: {}", e))?;

    // 应用聊天模板
    let prompt = self
      .ctx
      .model
      .apply_chat_template(&template, &self.messages, true)
      .map_err(|e| format!("应用聊天模板失败: {}", e))?;

    let (response, pos) = self
      .llama
      .generate(&mut self.ctx, &prompt, self.pos, token_gen_cb)?;

    self.pos = pos;
    self.add_assistant_message(&response);

    let re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    Ok(re.replace_all(&response.to_string(), "").to_string())
  }
}
