use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

use std::ptr::fn_addr_eq;

pub struct LlamaParams {
  // 单次推理生成的最大token数
  max_tokens: u32,
  // 模型加载进度回调
  model_load_progress: Option<fn(f32) -> bool>,
}

pub struct Llama {
  // 模型路径
  model_path: String,
  // 参数
  params: LlamaParams,
  // 后端
  backend: LlamaBackend,
  // 模型实例
  model: Option<LlamaModel>,
  // token生成回调
  token_gen_callbacks: Vec<fn(String)>,
}

impl LlamaParams {
  // 创建模型参数
  pub fn default() -> LlamaParams {
    LlamaParams {
      max_tokens: 512,
      model_load_progress: None,
    }
  }

  // 设置最大token数
  pub fn with_max_tokens(&mut self, max_tokens: u32) -> &Self {
    self.max_tokens = max_tokens;
    self
  }
}

impl Llama {
  pub fn new(model_path: &str, params: LlamaParams) -> Result<Llama, String> {
    // 初始化后端
    let mut backend = LlamaBackend::init().map_err(|e| format!("初始化backend失败: {}", e))?;
    backend.void_logs();

    let mut llama = Llama {
      params,
      model_path: String::from(model_path),
      backend,
      model: None,
      token_gen_callbacks: Vec::new(),
    };

    // 加载模型
    llama
      .load_model()
      .map_err(|e| format!("加载模型失败: {}", e))?;

    Result::Ok(llama)
  }

  // 加载模型
  pub fn load_model(&mut self) -> Result<(), String> {
    if self.model.is_some() {
      return Ok(());
    }

    // 设置模型参数
    let mut m_params = LlamaModelParams::default();

    // 设置模型加载进度回调
    if let Some(callback) = self.params.model_load_progress {
      m_params = m_params.with_progress_callback(callback);
    }

    let model = LlamaModel::load_from_file(&self.backend, self.model_path.clone(), &m_params)
      .map_err(|e| format!("加载模型失败: {}", e))?;
    self.model = Some(model);
    Ok(())
  }

  // 卸载模型
  pub fn unload_model(&mut self) {
    self.model = None;
  }

  // 创建上下文
  pub fn new_context(&self, params: LlamaContextParams) -> Result<LlamaContext<'_>, String> {
    let ctx = self
      .model
      .as_ref()
      .ok_or("模型未加载")?
      .new_context(&self.backend, params)
      .map_err(|e| format!("创建上下文失败: {}", e))?;

    Ok(ctx)
  }

  // 删除token生成回调
  fn del_token_gen_cb(&mut self, token_gen_cb: fn(String)) {
    self
      .token_gen_callbacks
      .retain(|cb| !fn_addr_eq(*cb, token_gen_cb));
  }

  // 设置token生成回调
  pub fn set_token_gen_cb(&mut self, token_gen_cb: fn(String)) -> impl Fn(&mut Self) {
    self.token_gen_callbacks.push(token_gen_cb);
    move |me: &mut Self| me.del_token_gen_cb(token_gen_cb)
  }

  // 补全文本
  pub fn generate(
    &self,
    ctx: &mut LlamaContext,
    prompt: &str,
    // context中的token位置
    pos: u32,
    token_gen_cb: Option<fn(String)>,
  ) -> Result<(String, u32), String> {
    let model = self.model.as_ref().ok_or("模型未加载")?;
    // 将输入文本转换为token
    let input_tokens = model
      .str_to_token(prompt, AddBos::Never)
      .map_err(|e| format!("字符串转token失败: {}", e))?;

    // 输入token长度
    let n_input_tokens = input_tokens.len();
    // 创建输入token batch
    let mut input_batch = LlamaBatch::new(n_input_tokens, 1);

    // 将输入token添加到batch
    for (i, &token) in input_tokens.iter().enumerate() {
      input_batch
        .add(token, pos as i32 + i as i32, &[0], i == n_input_tokens - 1)
        .map_err(|e| format!("添加token失败: {}", e))?;
    }

    // 执行推理
    ctx
      .decode(&mut input_batch)
      .map_err(|e| format!("执行推理失败: {}", e))?;

    // 设置采样器
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    // 设置解码器
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    // 记录生成token总数
    let mut n_tokens: u32 = u32::try_from(n_input_tokens).unwrap() + pos;

    // 输出结果
    let mut result = String::new();

    // 循环直到生成指定数量的token
    while n_tokens < self.params.max_tokens {
      // 采集下一个token
      let next_token = sampler.sample(&ctx, -1);

      // 告诉采样器刚刚采集的token
      sampler.accept(next_token);

      // 记录token总数(需要将结束符算进去，方便对context缓存的token进行计数，所以写在is_eog_token之前)
      n_tokens += 1;

      // 生成token是结束符则退出
      if model.is_eog_token(next_token) {
        break;
      }

      // 将采集的token装换为字符串
      let piece = model
        .token_to_piece(next_token, &mut decoder, true, None)
        .map_err(|e| format!("token转字符串失败: {}", e))?;

      // 依次触发token生成回调
      if self.token_gen_callbacks.len() > 0 {
        self
          .token_gen_callbacks
          .iter()
          .for_each(|cb| cb(piece.clone()));
      }

      // 触发complete传入的token生成回调
      if let Some(cb) = token_gen_cb {
        cb(piece.clone());
      }

      // 追加到结果
      result.push_str(&piece);

      // 构建下一轮batch
      let mut next_batch = LlamaBatch::new(1, 1);
      next_batch
        .add(next_token, (n_tokens - 1).try_into().unwrap(), &[0], true)
        .map_err(|e| format!("添加token失败: {}", e))?;

      // 执行推理
      ctx
        .decode(&mut next_batch)
        .map_err(|e| format!("执行推理失败2: {}", e))?;
    }

    Ok((result, n_tokens))
  }

  // 单次问答
  pub fn complete(
    &self,
    prompt: &str,
    n_ctx: u32,
    token_gen_cb: Option<fn(String)>,
  ) -> Result<String, String> {
    let params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    let mut ctx = self.new_context(params)?;

    let messages = vec![LlamaChatMessage::new(String::from("user"), prompt.to_string()).unwrap()];

    let template = ctx
      .model
      .chat_template(None)
      .map_err(|e| format!("获取聊天模板失败: {}", e))?;

    // 应用聊天模板
    let msg = ctx
      .model
      .apply_chat_template(&template, &messages, true)
      .map_err(|e| format!("应用聊天模板失败: {}", e))?;

    let (response, _) = self.generate(&mut ctx, &msg, 0, token_gen_cb)?;
    Ok(response)
  }
}
